#!/bin/bash
# Unified LLM server manager for Claude Code.
# Usage:
#   ./run.sh                        list available models
#   ./run.sh start <name> [slot]    start server + proxy in background (slot 1 or 2, default 1)
#   ./run.sh stop [slot]            stop slot (or all if omitted)
#   ./run.sh status                 show running state
#   source ./run.sh env <name> [slot]  export Claude Code env vars in this shell
#
# Ports:  slot 1 → server :8001  proxy :8081
#         slot 2 → server :8002  proxy :8082

# Resolve script directory
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ "$0" == -* ]]; then
    # Sourced from zsh ($0 is shell name like -zsh) — use PWD
    SCRIPT_DIR="$(pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

PROXY_SCRIPT="$SCRIPT_DIR/proxy.py"
PORT_BASE_SERVER=8000  # slot N → port 8000+N
PORT_BASE_PROXY=8080   # slot N → port 8080+N
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"

# Detect sourcing:
# - bash: BASH_SOURCE[0] != $0 when sourced
# - zsh: BASH_VERSION is unset when zsh sources a bash script
if [[ "${BASH_SOURCE[0]:-}" != "$0" ]] || [[ -z "${BASH_VERSION:-}" ]]; then
    _IS_SOURCED=true
else
    _IS_SOURCED=false
fi

# ── Load model definitions ──────────────────────────────────
_MODELS=()
source "$SCRIPT_DIR/models.conf"

# Parse a model entry by name; sets _r_* variables.
_resolve_model() {
    local name="$1"
    for entry in "${_MODELS[@]}"; do
        IFS='|' read -r m_name m_binary m_model m_mmproj m_alias m_label m_args m_client m_rocm_env m_hf_repo m_hf_includes <<< "$entry"
        if [[ "$m_name" == "$name" ]]; then
            _r_name="$m_name"
            _r_binary="$m_binary"
            _r_model="$m_model"
            _r_mmproj="$m_mmproj"
            _r_alias="$m_alias"
            _r_label="$m_label"
            _r_args=($m_args)
            _r_client="${m_client:-$m_alias}"
            _r_rocm_env="${m_rocm_env:-}"
            _r_hf_repo="${m_hf_repo:-}"
            _r_hf_includes="${m_hf_includes:-}"
            return 0
        fi
    done
    return 1
}

# Check that required commands exist; prints error and returns 1 if any are missing.
_check_deps() {
    local missing=()
    for cmd in "$@"; do
        command -v "$cmd" > /dev/null 2>&1 || missing+=("$cmd")
    done
    if (( ${#missing[@]} > 0 )); then
        echo "Error: missing dependencies: ${missing[*]}"
        return 1
    fi
}

# ── Commands ────────────────────────────────────────────────

cmd_list() {
    printf "\n\033[1mAvailable models:\033[0m\n\n"
    printf "  %-14s %-32s %-40s %s\n" "NAME" "LABEL" "BINARY" "ROCm ENV"
    printf "  %-14s %-32s %-40s %s\n" "----" "-----" "------" "---------"
    for entry in "${_MODELS[@]}"; do
        IFS='|' read -r m_name m_binary m_model m_mmproj m_alias m_label m_args m_client m_rocm_env _ _ <<< "$entry"
        printf "  \033[36m%-14s\033[0m %-32s %-40s %s\n" "$m_name" "$m_label" "$m_binary" "${m_rocm_env:-—}"
    done
    echo ""
    echo "  ./run.sh start <name> [slot]     start server + proxy (slot 1 or 2)"
    echo "  source ./run.sh env <name> [slot]  set Claude Code env vars in this shell"
    echo ""
}

cmd_start() {
    local name="${1:-}"
    local slot="${2:-1}"
    [[ -z "$name" ]] && { echo "Usage: $0 start <model-name> [slot]"; exit 1; }
    [[ "$slot" != "1" && "$slot" != "2" ]] && { echo "Error: slot must be 1 or 2"; exit 1; }

    local port_server=$(( PORT_BASE_SERVER + slot ))
    local port_proxy=$(( PORT_BASE_PROXY + slot ))

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; exit 1; }
    _check_deps python3 || exit 1

    # Guard: abort if ports are already in use
    if ss -tlnp 2>/dev/null | grep -q ":${port_server} "; then
        echo "Error: Port $port_server already in use. Is slot $slot already running?"
        exit 1
    fi

    mkdir -p "$PID_DIR" "$LOG_DIR"

    local proxy_log="$LOG_DIR/proxy-${slot}.log"
    local server_log="$LOG_DIR/server-${slot}.log"

    # Expand tilde in model paths
    local model_path="${_r_model//\~/$HOME}"
    local mmproj_path=""
    if [[ -n "$_r_mmproj" ]]; then
        mmproj_path="${_r_mmproj//\~/$HOME}"
    fi

    # Build llama-server command
    local -a cmd=("${_r_binary}" \
        --model "$model_path" \
        --port "$port_server")

    [[ -n "$mmproj_path" ]] && cmd+=(--mmproj "$mmproj_path")
    cmd+=(--alias "$_r_alias" "${_r_args[@]}")

    # Start proxy and poll until it listens on the port (max 10s)
    echo "Starting proxy [slot $slot] on port $port_proxy..."
    LLM_BACKEND_URL="http://localhost:${port_server}" \
    LLM_PROXY_PORT="${port_proxy}" \
    python3 "$PROXY_SCRIPT" > "$proxy_log" 2>&1 &
    local proxy_pid=$!
    echo "$proxy_pid" > "$PID_DIR/proxy-${slot}.pid"

    local max_wait=10
    for (( i=0; i<max_wait; i++ )); do
        sleep 1
        if ! ps -p "$proxy_pid" > /dev/null 2>&1; then
            echo "Error: Proxy failed to start. See $proxy_log"
            exit 1
        fi
        if ss -tlnp 2>/dev/null | grep -q ":${port_proxy} "; then
            echo "Proxy running (PID: $proxy_pid)"
            break
        fi
        if (( i == max_wait - 1 )); then
            echo "Error: Proxy did not listen on port $port_proxy within ${max_wait}s. See $proxy_log"
            exit 1
        fi
    done

    # Start llama-server in background
    echo "Starting llama.cpp server [slot $slot] on port $port_server..."
    echo "Model: $_r_label ($_r_alias)"
    echo "ROCm env: ${_r_rocm_env:-—}"

    # _r_rocm_env is intentionally unquoted — word-splits space-separated KEY=VAL pairs for env
    if [[ -n "$_r_rocm_env" ]]; then
        env $_r_rocm_env "${cmd[@]}" > "$server_log" 2>&1 &
    else
        "${cmd[@]}" > "$server_log" 2>&1 &
    fi
    local server_pid=$!
    echo "$server_pid" > "$PID_DIR/server-${slot}.pid"

    echo "Server running (PID: $server_pid)"
    echo ""
    echo "  Logs:  tail -f $server_log"
    echo "         tail -f $proxy_log"
    echo "  Stop:  $0 stop $slot"
}

# Find a PID listening on a given port (uses ss or lsof).
_pid_on_port() {
    local port="$1"
    if command -v ss > /dev/null; then
        ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1
    elif command -v lsof > /dev/null; then
        lsof -ti ":$port" 2>/dev/null | head -1
    fi
}

_stop_slot() {
    local slot="$1"
    local port_server=$(( PORT_BASE_SERVER + slot ))
    local port_proxy=$(( PORT_BASE_PROXY + slot ))
    local stopped=0

    # Stop server — prefer stored PID, fall back to port scan
    local server_pid=""
    [[ -f "$PID_DIR/server-${slot}.pid" ]] && server_pid=$(cat "$PID_DIR/server-${slot}.pid")
    if [[ -z "$server_pid" ]] || ! ps -p "$server_pid" > /dev/null 2>&1; then
        server_pid=$(_pid_on_port "$port_server")
    fi
    if [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null 2>&1; then
        kill "$server_pid" 2>/dev/null
        echo "Slot $slot: llama-server (PID: $server_pid) stopped."
        stopped=$(( stopped + 1 ))
    fi

    # Stop proxy — prefer stored PID, fall back to port scan
    local proxy_pid=""
    [[ -f "$PID_DIR/proxy-${slot}.pid" ]] && proxy_pid=$(cat "$PID_DIR/proxy-${slot}.pid")
    if [[ -z "$proxy_pid" ]] || ! ps -p "$proxy_pid" > /dev/null 2>&1; then
        proxy_pid=$(_pid_on_port "$port_proxy")
    fi
    if [[ -n "$proxy_pid" ]] && ps -p "$proxy_pid" > /dev/null 2>&1; then
        kill "$proxy_pid" 2>/dev/null
        echo "Slot $slot: proxy (PID: $proxy_pid) stopped."
        stopped=$(( stopped + 1 ))
    fi

    rm -f "$PID_DIR/server-${slot}.pid" "$PID_DIR/proxy-${slot}.pid"
    return $(( stopped == 0 ))
}

cmd_stop() {
    local slot="${1:-}"

    if [[ -n "$slot" ]]; then
        [[ "$slot" != "1" && "$slot" != "2" ]] && { echo "Error: slot must be 1 or 2"; return 1; }
        _stop_slot "$slot" || echo "Slot $slot: nothing was running."
    else
        local total=0
        for s in 1 2; do
            _stop_slot "$s" && total=$(( total + 1 ))
        done
        [[ "$total" -eq 0 ]] && echo "Nothing was running."
    fi

    rmdir "$PID_DIR" 2>/dev/null || true
}

cmd_status() {
    local found=0
    for slot in 1 2; do
        local port_server=$(( PORT_BASE_SERVER + slot ))
        local port_proxy=$(( PORT_BASE_PROXY + slot ))

        local server_pid=""
        [[ -f "$PID_DIR/server-${slot}.pid" ]] && server_pid=$(cat "$PID_DIR/server-${slot}.pid")
        if [[ -z "$server_pid" ]] || ! ps -p "$server_pid" > /dev/null 2>&1; then
            server_pid=$(_pid_on_port "$port_server")
        fi

        local proxy_pid=""
        [[ -f "$PID_DIR/proxy-${slot}.pid" ]] && proxy_pid=$(cat "$PID_DIR/proxy-${slot}.pid")
        if [[ -z "$proxy_pid" ]] || ! ps -p "$proxy_pid" > /dev/null 2>&1; then
            proxy_pid=$(_pid_on_port "$port_proxy")
        fi

        local slot_active=0
        if [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null 2>&1; then
            echo "Slot $slot: llama-server (PID: $server_pid) on :$port_server"
            echo "         Log: tail -f $LOG_DIR/server-${slot}.log"
            slot_active=1; found=1
        fi
        if [[ -n "$proxy_pid" ]] && ps -p "$proxy_pid" > /dev/null 2>&1; then
            echo "Slot $slot: proxy       (PID: $proxy_pid) on :$port_proxy"
            slot_active=1; found=1
        fi
    done
    [[ "$found" -eq 0 ]] && echo "Nothing running."
}

cmd_env() {
    local name="${1:-}"
    local slot="${2:-1}"
    [[ -z "$name" ]] && { echo "Usage: source $0 env <model-name> [slot]"; return 1; }
    [[ "$slot" != "1" && "$slot" != "2" ]] && { echo "Error: slot must be 1 or 2"; return 1; }

    local port_proxy=$(( PORT_BASE_PROXY + slot ))

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; return 1; }

    local config_dir=".claude-${_r_name}"
    [[ "$slot" -gt 1 ]] && config_dir=".claude-${_r_name}-${slot}"

    # When sourced, these exports take effect in the caller's shell.
    # When executed directly, they're printed for the user to see.
    export CLAUDE_CONFIG_DIR="$config_dir"
    export ANTHROPIC_BASE_URL="http://localhost:$port_proxy"
    export ANTHROPIC_AUTH_TOKEN="sk-no-key-required"
    export ANTHROPIC_MODEL="$_r_client"
    export ANTHROPIC_SMALL_FAST_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$_r_client"
    export API_TIMEOUT_MS="3000000"

    echo "Claude Code env set for $_r_label ($_r_client) [slot $slot]"
    echo "  CLAUDE_CONFIG_DIR=$config_dir"
    echo "  ANTHROPIC_BASE_URL=http://localhost:$port_proxy"
    echo "  ANTHROPIC_MODEL=$_r_client"
}

cmd_clear() {
    if [[ "$_IS_SOURCED" != true ]]; then
        echo "This command must be sourced:  source $0 clear"
        return 1
    fi

    unset CLAUDE_CONFIG_DIR
    unset ANTHROPIC_BASE_URL
    unset ANTHROPIC_AUTH_TOKEN
    unset ANTHROPIC_MODEL
    unset ANTHROPIC_SMALL_FAST_MODEL
    unset ANTHROPIC_DEFAULT_SONNET_MODEL
    unset ANTHROPIC_DEFAULT_OPUS_MODEL
    unset ANTHROPIC_DEFAULT_HAIKU_MODEL
    unset API_TIMEOUT_MS

    echo "Environment cleared."
    echo "Run 'source $0 env <name> [slot]' to set up a model again."
}

# ── Benchmark ────────────────────────────────────────────────
#
# ROCm env vars are now configured per-model in models.conf via _model_rocm_env.
# Use benchmark data + formula: score = tg128*5 + pp512*0.1 to determine optimal combo.
# Models using Vulkan (default bin) should have _model_rocm_env="" (empty).
#
# ── ROC env combos (all 8 permutations) ────────────────────

_roc_env_combos() {
    echo "ROC_ENABLE_PREFETCH=0;HSA_ENABLE_COMPRESSION=0;HSA_ENABLE_SDMA=0"
    echo "ROC_ENABLE_PREFETCH=0;HSA_ENABLE_COMPRESSION=0;HSA_ENABLE_SDMA=1"
    echo "ROC_ENABLE_PREFETCH=0;HSA_ENABLE_COMPRESSION=1;HSA_ENABLE_SDMA=0"
    echo "ROC_ENABLE_PREFETCH=0;HSA_ENABLE_COMPRESSION=1;HSA_ENABLE_SDMA=1"
    echo "ROC_ENABLE_PREFETCH=1;HSA_ENABLE_COMPRESSION=0;HSA_ENABLE_SDMA=0"
    echo "ROC_ENABLE_PREFETCH=1;HSA_ENABLE_COMPRESSION=0;HSA_ENABLE_SDMA=1"
    echo "ROC_ENABLE_PREFETCH=1;HSA_ENABLE_COMPRESSION=1;HSA_ENABLE_SDMA=0"
    echo "ROC_ENABLE_PREFETCH=1;HSA_ENABLE_COMPRESSION=1;HSA_ENABLE_SDMA=1"
}

_rocm_bin="/opt/llama.cpp-rocm/llama-bench"
_vulkan_bin="llama-bench"
_bench_common_args=(-ngl 999 -t 16 --mmap 0 -fa 1)

run_bench() {
    local binary="$1"
    local model="$2"
    local env_combo="$3"
    local backend_type="$4"
    local model_name="$5"
    local result_file="$6"

    local env_display="${env_combo:-<none>}"
    local timestamp_ms
    timestamp_ms=$(date +%s%3N)
    local tmp_out
    tmp_out=$(mktemp)

    printf "%s" "  $env_display ... "

    # env_combo uses ';' as separator; convert to spaces so env receives separate KEY=VAL args
    env ${env_combo//;/ } "$binary" "${_bench_common_args[@]}" -m "$model" > "$tmp_out" 2>&1
    local exit_code=$?

    if [[ "$exit_code" -eq 0 ]]; then
        echo "OK"
    else
        echo "FAILED"
    fi

    local env_json="{}"
    if [[ -n "$env_combo" ]]; then
        local env_parts=()
        IFS=';' read -ra pairs <<< "$env_combo"
        for pair in "${pairs[@]}"; do
            local key="${pair%%=*}"
            local val="${pair#*=}"
            env_parts+=("$key=\"$val\"")
        done
        env_json=$(printf '%s\n' "${env_parts[@]}" | jq -Rs 'split("\n") | map(select(length>0)) | map(split("=") | {key: .[0], value: .[1]}) | from_entries')
    fi

    local stdout_raw
    stdout_raw=$(jq -Rs 'split("\n") | map(select(length>0)) | join("\n")' "$tmp_out")
    rm -f "$tmp_out"

    local status_val exit_code_val
    if [[ "$exit_code" -eq 0 ]]; then
        status_val='"ok"'
        exit_code_val="null"
    else
        status_val='"error"'
        exit_code_val="$exit_code"
    fi

    local entry_json
    entry_json=$(jq -n \
        --arg ts "$timestamp_ms" \
        --arg backend "$backend_type" \
        --arg binary "$binary" \
        --arg model "$model" \
        --arg model_name "$model_name" \
        --argjson env "$env_json" \
        --argjson status "$status_val" \
        --argjson stdout "$stdout_raw" \
        --argjson exit_code "$exit_code_val" \
        '{
            timestamp: $ts,
            backend: $backend,
            binary: $binary,
            model: $model,
            model_name: $model_name,
            env: $env,
            status: $status,
            exit_code: $exit_code,
            stdout: $stdout
        }'
    )
    echo "$entry_json" >> "$result_file"
}

bench_model() {
    local m_name="$1"
    local m_model="$2"
    local m_label="$3"
    local full_mode="${4:-}"
    local result_file="$5"

    local model_path="${m_model//\~/$HOME}"
    if [[ ! -f "$model_path" ]]; then
        echo "  Skipping $m_label: model file not found"
        return
    fi

    echo "  $m_label"
    if [[ "$full_mode" == "full" ]]; then
        # Full test: all 8 ROCm env combos + Vulkan
        echo "    ROCM:"
        while IFS= read -r combo; do
            [[ -z "$combo" ]] && continue
            run_bench "$_rocm_bin" "$model_path" "$combo" "rocm" "$m_name" "$result_file"
        done < <(_roc_env_combos)

        echo "    Vulkan:"
        run_bench "$_vulkan_bin" "$model_path" "" "vulkan" "$m_name" "$result_file"
    else
        # Default: only test default ROCm env + Vulkan
        local default_rocm_env="ROC_ENABLE_PREFETCH=1;HSA_ENABLE_COMPRESSION=1;HSA_ENABLE_SDMA=0"
        echo "    ROCM (default):"
        run_bench "$_rocm_bin" "$model_path" "$default_rocm_env" "rocm" "$m_name" "$result_file"
        echo "    Vulkan:"
        run_bench "$_vulkan_bin" "$model_path" "" "vulkan" "$m_name" "$result_file"
    fi
}

cmd_benchmark() {
    local target=""
    local full=""

    while [[ "$1" == "--full" ]]; do
        full="full"
        shift
    done
    target="${1:-}"

    if [[ -z "$target" ]]; then
        echo "Usage: $0 bench [options] <model-name|all>"
        echo ""
        echo "  model-name   run benchmark for a single model"
        echo "  all          run benchmark for all models"
        echo ""
        echo "  Options:"
        echo "    --full      test all 8 ROCm env combos + Vulkan (default: only default ROCm)"
        echo ""
        echo "  Available models:"
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ _ _ _ m_label _ _ _ _ _ <<< "$entry"
            echo "    $m_name"
        done
        echo ""
        echo "  Result file: Benchmarks/benchmark-<model>-<timestamp>.jsonl"
        return
    fi

    _check_deps jq || return 1

    local result_file="$SCRIPT_DIR/Benchmarks/benchmark-${target}-$(date +%Y%m%d-%H%M%S).jsonl"
    mkdir -p "$(dirname "$result_file")"
    echo "Results → $result_file"
    echo ""

    if [[ "$target" == "all" ]]; then
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ m_model _ _ m_label _ _ _ _ _ <<< "$entry"
            echo "=== $m_label ==="
            bench_model "$m_name" "$m_model" "$m_label" "$full" "$result_file"
            echo ""
        done
    else
        local found=0
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ m_model _ _ m_label _ _ _ _ _ <<< "$entry"
            if [[ "$m_name" == "$target" ]]; then
                found=1
                echo "=== $m_label ==="
                bench_model "$m_name" "$m_model" "$m_label" "$full" "$result_file"
                break
            fi
        done
        if [[ "$found" -eq 0 ]]; then
            echo "Unknown model: $target"
            echo "Run '$0 bench' to see available models."
            return 1
        fi
    fi

    echo ""
    echo "Done. $(wc -l < "$result_file") entries written."
}

cmd_help() {
    echo ""
    printf "\033[1m$(basename "$0")\033[0m — LLM server manager\n"
    echo ""
    printf "  %-20s %s\n" "start <name> [slot]"   "start server + proxy (slot 1 or 2, default 1)"
    printf "  %-20s %s\n" "stop [slot]"            "stop slot (or all if omitted)"
    printf "  %-20s %s\n" "status"                 "show running state"
    printf "  %-20s %s\n" "bench [opts] <m>"       "run benchmark (model or 'all')"
    printf "  %-20s %s\n" "list"                   "show available models"
    printf "  %-20s %s\n" "env <name> [slot]"      "set Claude Code env vars (source!)"
    printf "  %-20s %s\n" "clear"                  "clear env vars (source!)"
    printf "  %-20s %s\n" "download <m>"           "download model(s) (or 'all')"
    echo ""
    echo "  Ports:  slot 1 → server :8001  proxy :8081"
    echo "          slot 2 → server :8002  proxy :8082"
    echo ""
}

# ── Download ────────────────────────────────────────────────

_download_model() {
    local repo="$1"
    local model_path="$2"
    local includes="$3"
    local force="${4:-}"

    echo "  Downloading $repo ..."
    mkdir -p "$model_path"

    local -a include_args=()
    read -ra include_pats <<< "$includes"
    for pat in "${include_pats[@]}"; do
        include_args+=(--include "$pat")
    done

    if [[ -n "$force" ]]; then
        HF_FORCE_DOWNLOAD=1 hf download "$repo" --local-dir "$model_path" "${include_args[@]}"
    else
        hf download "$repo" --local-dir "$model_path" "${include_args[@]}"
    fi
}

cmd_download() {
    local target=""
    local force=""

    while [[ "$1" == "--force" || "$1" == "-f" ]]; do
        force="1"
        shift
    done
    target="${1:-}"

    if [[ -z "$target" ]]; then
        echo "Usage: $0 download [--force] <model-name|all>"
        echo ""
        echo "  Options:"
        echo "    --force, -f   force re-download even if model exists"
        echo ""
        echo "  Available models:"
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ _ _ _ m_label _ _ _ _ _ <<< "$entry"
            echo "    $m_name"
        done
        return
    fi

    _check_deps hf || return 1

    if [[ "$target" == "all" ]]; then
        echo "Downloading all models..."
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ m_model _ _ m_label _ _ _ m_hf_repo m_hf_includes <<< "$entry"
            if [[ -z "$m_hf_repo" ]]; then
                echo "  Skipping $m_label: no download info configured"
                continue
            fi
            local model_dir
            model_dir="$(dirname "${m_model//\~/$HOME}")"
            echo "=== $m_label ==="
            _download_model "$m_hf_repo" "$model_dir" "$m_hf_includes" "$force"
        done
        return
    fi

    _resolve_model "$target" || { echo "Unknown model: $target"; echo "Run '$0 download' to see available models."; return 1; }

    if [[ -z "$_r_hf_repo" ]]; then
        echo "No download info configured for model: $target"
        return 1
    fi

    local model_dir
    model_dir="$(dirname "${_r_model//\~/$HOME}")"
    _download_model "$_r_hf_repo" "$model_dir" "$_r_hf_includes" "$force"
}

# ── Dispatch ────────────────────────────────────────────────
if [[ "$_IS_SOURCED" == true ]]; then
    if [[ "${1:-}" == "env" ]]; then
        shift
        cmd_env "$@"
    elif [[ "${1:-}" == "clear" ]]; then
        shift
        cmd_clear "$@"
    fi
else
    case "${1:-}" in
        start)      shift; cmd_start "$@" ;;
        stop)       shift; cmd_stop "$@" ;;
        status)     cmd_status ;;
        bench)      shift; cmd_benchmark "$@" ;;
        list)       cmd_list ;;
        help)       cmd_help ;;
        env)        echo "This command must be sourced:  source $0 env <name> [slot]" ;;
        download)   shift; cmd_download "$@" ;;
        *)          cmd_help; cmd_list ;;
    esac
fi
