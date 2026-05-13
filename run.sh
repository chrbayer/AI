#!/bin/bash
# Unified LLM server manager for Claude Code.
# Usage:
#   ./run.sh                 list available models
#   ./run.sh start <name>    start server + proxy (foreground)
#   ./run.sh stop            stop the proxy
#   ./run.sh status          show running state
#   source ./run.sh env <n>  export Claude Code env vars in this shell

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
PORT_SERVER=8001
PORT_PROXY=8081
PID_DIR="$SCRIPT_DIR/.pids"

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
    echo "  ./run.sh start <name>    start server + proxy"
    echo "  source ./run.sh env <name> set Claude Code env vars in this shell"
    echo ""
}

cmd_start() {
    local name="${1:-}"
    [[ -z "$name" ]] && { echo "Usage: $0 start <model-name>"; exit 1; }

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; exit 1; }
    _check_deps python3 || exit 1

    mkdir -p "$PID_DIR"

    # Expand tilde in model paths
    local model_path="${_r_model//\~/$HOME}"
    local mmproj_path=""
    if [[ -n "$_r_mmproj" ]]; then
        mmproj_path="${_r_mmproj//\~/$HOME}"
    fi

    # Build llama-server command
    local -a cmd=("${_r_binary}" \
        --model "$model_path" \
        --port "$PORT_SERVER")

    [[ -n "$mmproj_path" ]] && cmd+=(--mmproj "$mmproj_path")
    cmd+=(--alias "$_r_alias" "${_r_args[@]}")

    # Start proxy and poll until it listens on the port (max 10s)
    echo "Starting proxy on port $PORT_PROXY..."
    python3 "$PROXY_SCRIPT" > "$SCRIPT_DIR/proxy.log" 2>&1 &
    local proxy_pid=$!
    echo "$proxy_pid" > "$PID_DIR/proxy.pid"

    local max_wait=10
    for (( i=0; i<max_wait; i++ )); do
        sleep 1
        if ! ps -p "$proxy_pid" > /dev/null 2>&1; then
            echo "Error: Proxy failed to start. See proxy.log"
            exit 1
        fi
        if ss -tlnp 2>/dev/null | grep -q ":${PORT_PROXY} "; then
            echo "Proxy running (PID: $proxy_pid). Logs in proxy.log"
            break
        fi
        if (( i == max_wait - 1 )); then
            echo "Error: Proxy did not listen on port $PORT_PROXY within ${max_wait}s. See proxy.log"
            exit 1
        fi
    done

    cleanup() {
        echo ""
        local server_pid
        server_pid=$(_pid_on_port "$PORT_SERVER")
        if [[ -n "$server_pid" ]]; then
            kill "$server_pid" 2>/dev/null
            echo "llama-server (PID: $server_pid) stopped."
        fi
        echo "Stopping proxy (PID: $proxy_pid)..."
        kill "$proxy_pid" 2>/dev/null
        rm -rf "$PID_DIR"
    }
    trap cleanup SIGINT SIGTERM

    echo "Starting llama.cpp server on port $PORT_SERVER..."
    echo "Model: $_r_label ($_r_alias)"
    echo "ROCm env: ${_r_rocm_env:-—}"
    echo "----------------------------------------------------"

    # _r_rocm_env is intentionally unquoted — word-splits space-separated KEY=VAL pairs for env
    if [[ -n "$_r_rocm_env" ]]; then
        env $_r_rocm_env "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
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

cmd_stop() {
    local stopped=0

    # Stop server
    local server_pid
    server_pid=$(_pid_on_port "$PORT_SERVER")
    if [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null 2>&1; then
        kill "$server_pid" 2>/dev/null
        echo "llama-server (PID: $server_pid) stopped."
        stopped=$((stopped + 1))
    else
        echo "llama-server not running."
    fi

    # Stop proxy
    local pid=""
    if [[ -f "$PID_DIR/proxy.pid" ]]; then
        pid=$(cat "$PID_DIR/proxy.pid")
    fi

    if [[ -z "$pid" ]] || ! ps -p "$pid" > /dev/null 2>&1; then
        pid=$(_pid_on_port "$PORT_PROXY")
    fi

    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid" 2>/dev/null
        echo "Proxy (PID: $pid) stopped."
        stopped=$((stopped + 1))
    else
        echo "Proxy not running."
    fi

    rm -rf "$PID_DIR"
}

cmd_status() {
    local pid=""
    if [[ -f "$PID_DIR/proxy.pid" ]]; then
        pid=$(cat "$PID_DIR/proxy.pid")
    fi

    # Fallback: find the proxy by port if we have no PID file
    if [[ -z "$pid" ]] || ! ps -p "$pid" > /dev/null 2>&1; then
        pid=$(_pid_on_port "$PORT_PROXY")
    fi

    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
        echo "Proxy running (PID: $pid) on port $PORT_PROXY"
    else
        echo "Proxy not running."
    fi

    local server_pid=""
    server_pid=$(_pid_on_port "$PORT_SERVER")
    if [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null 2>&1; then
        echo "llama-server running (PID: $server_pid) on port $PORT_SERVER"
    else
        echo "llama-server not running."
    fi
}

cmd_env() {
    local name="${1:-}"
    [[ -z "$name" ]] && { echo "Usage: source $0 env <model-name>"; return 1; }

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; return 1; }

    local config_dir=".claude-${_r_name}"

    # When sourced, these exports take effect in the caller's shell.
    # When executed directly, they're printed for the user to see.
    export CLAUDE_CONFIG_DIR="$config_dir"
    export ANTHROPIC_BASE_URL="http://localhost:$PORT_PROXY"
    export ANTHROPIC_AUTH_TOKEN="sk-no-key-required"
    export ANTHROPIC_MODEL="$_r_client"
    export ANTHROPIC_SMALL_FAST_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$_r_client"
    export API_TIMEOUT_MS="3000000"

    echo "Claude Code env set for $_r_label ($_r_client)"
    echo "  CLAUDE_CONFIG_DIR=$config_dir"
    echo "  ANTHROPIC_BASE_URL=http://localhost:$PORT_PROXY"
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
    echo "Run 'source $0 env <name>' to set up a model again."
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
    printf "  %-14s %s\n" "start <name>"      "start server + proxy"
    printf "  %-14s %s\n" "stop"              "stop all processes"
    printf "  %-14s %s\n" "status"            "show running state"
    printf "  %-14s %s\n" "bench [opts] <m>"  "run benchmark (model or 'all')"
    printf "  %-14s %s\n" "list"              "show available models"
    printf "  %-14s %s\n" "env <name>"        "set Claude Code env vars (source!)"
    printf "  %-14s %s\n" "clear"             "clear env vars (source!)"
    printf "  %-14s %s\n" "download <m>"      "download model(s) (or 'all')"
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
        stop)       cmd_stop ;;
        status)     cmd_status ;;
        bench)      shift; cmd_benchmark "$@" ;;
        list)       cmd_list ;;
        help)       cmd_help ;;
        env)        echo "This command must be sourced:  source $0 env <name>" ;;
        download)   shift; cmd_download "$@" ;;
        *)          cmd_help; cmd_list ;;
    esac
fi
