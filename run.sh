#!/bin/bash
# Unified LLM server manager for Claude Code.
# Usage:
#   ./run.sh                                     list available models
#   ./run.sh start <name> [slot] [--proxy] [--public] [--max-predict N] [--reasoning off|on|low|medium|high|max|N] [--parallel N] [--ctx N] [--cache-ram N] [--verbose] [--clear-logs] [--host ADDR] [--gpu-priority low|medium|high|realtime] [--mmproj] [--spec on|off]  start server (+ proxy with --proxy) in background (slot 1-3, default 1)
#   ./run.sh stop [slot]                         stop slot (or all if omitted)
#   ./run.sh status                              show running state
#   ./run.sh clear-kv [slot]                     drop the KV cache without restarting (all slots, or one)
#   ./run.sh probe-reasoning [model]             show what each model's chat template supports
#   ./run.sh gen-certs <public-hostname>         create CA + certificates for --public
#   source ./run.sh env <name> [slot] [--proxy|--direct]   export Claude Code env vars in this shell
#
# Ports:  slot 1 → server :8001  proxy :8081
#         slot 2 → server :8002  proxy :8082
#         slot 3 → server :8003  proxy :8083

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
# --slot-save-path is what enables POST /slots/{id}?action=erase, the only way
# to drop a slot's KV cache without restarting the server (see clear-kv). The
# same flag also enables the save/restore actions, which write into this dir.
SLOT_STATE_DIR="$SCRIPT_DIR/.slots"

# ── Public exposure (--public) ───────────────────────────────
# Internet traffic never reaches llama-server or the proxy directly: stunnel
# terminates a mutually authenticated TLS connection from the VPS and forwards
# to 127.0.0.1. That keeps --public fully independent of --host, which stays
# what it always was — plain LAN exposure.
PORT_BASE_TLS_SERVER=8440   # slot N → port 8440+N  (TLS front for the server)
PORT_BASE_TLS_PROXY=8450    # slot N → port 8450+N  (TLS front for the proxy)
LLM_CONF_DIR="${LLM_CONF_DIR:-$HOME/.config/llm}"
TOKEN_FILE="${LLM_TOKEN_FILE:-$LLM_CONF_DIR/tokens}"
TLS_DIR="$LLM_CONF_DIR/tls"
STUNNEL_DIR="$SCRIPT_DIR/.stunnel"
PUBLIC_MAX_PREDICT_DEFAULT=8192   # --public caps generation length unless --max-predict says otherwise

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
        IFS='|' read -r m_name m_binary m_model m_mmproj m_alias m_label m_args m_client m_rocm_env m_hf_repo m_hf_includes m_hf_dir m_no_reasoning_args m_ctx m_reasoning m_reasoning_levels m_spec_args m_hf_draft m_hf_mmproj <<< "$entry"
        if [[ "$m_name" == "$name" ]]; then
            _r_name="$m_name"
            _r_binary="$m_binary"
            _r_model="$m_model"
            _r_mmproj="$m_mmproj"
            _r_alias="$m_alias"
            _r_label="$m_label"
            _r_args=($m_args)
            _r_client="${m_client:-$m_name}"
            _r_rocm_env="${m_rocm_env:-}"
            _r_hf_repo="${m_hf_repo:-}"
            _r_hf_includes="${m_hf_includes:-}"
            _r_hf_dir="${m_hf_dir:-}"
            _r_no_reasoning_args=($m_no_reasoning_args)
            _r_ctx="${m_ctx:-}"
            _r_reasoning="${m_reasoning:-unknown}"
            _r_reasoning_levels="${m_reasoning_levels:-}"
            _r_spec_args=($m_spec_args)
            _r_hf_draft="${m_hf_draft:-}"
            _r_hf_mmproj="${m_hf_mmproj:-}"
            return 0
        fi
    done
    return 1
}

# Start a long-lived process in its own session, so it survives the shell that
# started it: no controlling terminal means no SIGHUP when that terminal closes.
# Prints the PID and leaves it in $pidfile.
#
# The child writes its own PID immediately before exec rather than the caller
# taking $!. With job control on, setsid finds itself a process group leader and
# forks instead of exec-ing, so $! would name the wrapper — and stop/status,
# which read the PID file, would then be pointing at a process that is already
# gone. Any KEY=VAL environment has to be passed via `env`, since shell-level
# assignments would not survive being handed on as arguments.
_spawn_detached() {
    local pidfile="$1" logfile="$2" mode="$3"; shift 3
    rm -f "$pidfile"
    if [[ "$mode" == append ]]; then
        setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pidfile" "$@" \
            >> "$logfile" 2>&1 < /dev/null &
    else
        setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pidfile" "$@" \
            > "$logfile" 2>&1 < /dev/null &
    fi
    local i
    for i in $(seq 1 50); do
        [[ -s "$pidfile" ]] && { cat "$pidfile"; return 0; }
        sleep 0.1
    done
    return 1
}

# Guard for value-taking options: bash's `shift 2` quietly fails when only the
# flag itself is left, and the option loop would then spin forever. Call as
# `_need_value "$@"` so $1 is the flag and $2 its value.
_need_value() {
    if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value"
        return 1
    fi
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
        IFS='|' read -r m_name m_binary m_model m_mmproj m_alias m_label m_args m_client m_rocm_env _ _ _ <<< "$entry"
        printf "  \033[36m%-14s\033[0m %-32s %-40s %s\n" "$m_name" "$m_label" "$m_binary" "${m_rocm_env:-—}"
    done
    echo ""
    echo "  ./run.sh start <name> [slot]     start server + proxy (slot 1 or 2)"
    echo "  source ./run.sh env <name> [slot]  set Claude Code env vars in this shell"
    echo ""
}

# ── Reasoning ───────────────────────────────────────────────
#
# One switch for every model: --reasoning off|on|low|medium|high|max|N. What each
# model can actually do lives in models.conf (`reasoning`, `reasoning_levels`),
# because the mechanism is the chat template, not the server: --reasoning sets the
# template kwarg enable_thinking, --reasoning-effort sets the template variables
# reasoning_effort / reasoning_strength. A template that reads neither cannot be
# steered, and templates raise on level names they do not know — so a request the
# model cannot honour is refused here instead of failing per request later.
#
# Fills _reasoning_args (llama-server flags) and _reasoning_note (one line for the
# start output). Returns 1 with an explanation on stderr if the model cannot comply.
REASONING_LEVELS="low medium high max"

_translate_reasoning() {
    local spec="$1" mode="$2" level_map="$3"
    _reasoning_args=()
    _reasoning_note=""

    # A token budget is spelled as a number; 0 means off, -1 unrestricted.
    if [[ "$spec" =~ ^(-1|[0-9]+)$ ]]; then
        [[ "$spec" == "0" ]] && spec="off"
    elif [[ ! " off on $REASONING_LEVELS " == *" $spec "* ]]; then
        echo "Error: --reasoning takes off, on, a level ($REASONING_LEVELS), or a token budget (N, -1); got '$spec'" >&2
        return 1
    fi

    local is_level=false
    [[ " $REASONING_LEVELS " == *" $spec "* ]] && is_level=true

    case "$mode" in
        none)
            if [[ "$spec" == "off" ]]; then
                _reasoning_note="not applicable — this model has no thinking mode"
                return 0
            fi
            echo "Error: model '$_r_name' has no thinking mode (its chat template has no reasoning at all)" >&2
            return 1
            ;;
        locked-off)
            if [[ "$spec" == "off" ]]; then
                _reasoning_note="already off — the chat template locks thinking closed"
                return 0
            fi
            echo "Error: the chat template of '$_r_name' locks thinking closed, so '--reasoning $spec' cannot work." >&2
            echo "       Only a different template (--chat-template-file with the stock one) would restore it." >&2
            return 1
            ;;
        toggle)
            if [[ "$is_level" == true ]]; then
                echo "Error: model '$_r_name' supports on/off and a token budget, but no reasoning levels." >&2
                echo "       Use --reasoning on, off, or a token budget (e.g. --reasoning 2048)." >&2
                return 1
            fi
            ;;
        effort|unknown) ;;
        *)
            echo "Error: model '$_r_name' has an unknown reasoning mode '$mode' in models.conf" >&2
            return 1
            ;;
    esac

    if [[ "$spec" == "off" ]]; then
        # The model's own sampler tweaks for non-thinking mode come first, so an
        # explicit flag later in the command line still wins.
        [[ ${#_r_no_reasoning_args[@]} -gt 0 ]] && _reasoning_args+=("${_r_no_reasoning_args[@]}")
        _reasoning_args+=(--reasoning off --reasoning-budget 0)
        _reasoning_note="off (--reasoning off --reasoning-budget 0)"
        return 0
    fi

    if [[ "$spec" == "on" ]]; then
        _reasoning_args+=(--reasoning on)
        _reasoning_note="on, template default depth"
        return 0
    fi

    if [[ "$is_level" == true ]]; then
        local mapped=""
        if [[ "$mode" == "effort" ]]; then
            # "low=low medium=medium high=high max=xhigh" → the name this template takes
            local pair
            for pair in $level_map; do
                [[ "${pair%%=*}" == "$spec" ]] && mapped="${pair#*=}"
            done
            if [[ -z "$mapped" ]]; then
                echo "Error: model '$_r_name' does not map the level '$spec' (has: ${level_map:-none})" >&2
                return 1
            fi
        else
            mapped="$spec"   # unknown mode: pass the canonical name through
        fi
        _reasoning_args+=(--reasoning on --reasoning-effort "$mapped")
        _reasoning_note="level $spec (--reasoning-effort $mapped)"
        return 0
    fi

    # Remaining case: a token budget.
    _reasoning_args+=(--reasoning on --reasoning-budget "$spec")
    if [[ "$spec" == "-1" ]]; then
        _reasoning_note="on, unrestricted budget"
    else
        _reasoning_note="on, budget ${spec} tokens"
    fi
    return 0
}

# ── Public exposure helpers ─────────────────────────────────

# Count real tokens in the token file (blank lines and # comments don't count).
_token_count() {
    # grep -c prints 0 *and* exits 1 when nothing matches, so capture first and
    # only then fall back — piping the failure into `echo 0` yields "0\n0".
    local n
    n=$(grep -cvE '^[[:space:]]*(#|$)' "$TOKEN_FILE" 2>/dev/null) || n=0
    echo "${n:-0}"
}

# Everything --public needs must exist *before* a single port opens.
_check_public_prereqs() {
    local ok=true

    if ! command -v stunnel > /dev/null; then
        echo "Error: --public needs stunnel (sudo dnf install stunnel)"
        ok=false
    fi

    if [[ ! -f "$TOKEN_FILE" ]]; then
        echo "Error: no token file at $TOKEN_FILE"
        echo "       Create one (0600, one token per line), e.g.:"
        echo "         mkdir -p $LLM_CONF_DIR && umask 077"
        echo "         openssl rand -hex 32 > $TOKEN_FILE"
        ok=false
    else
        local perms
        perms=$(stat -c %a "$TOKEN_FILE")
        if [[ ! "$perms" =~ ^[0-7]00$ ]]; then
            echo "Error: $TOKEN_FILE is mode $perms — must not be readable by group or others."
            echo "       chmod 600 $TOKEN_FILE"
            ok=false
        elif [[ "$(_token_count)" -eq 0 ]]; then
            echo "Error: $TOKEN_FILE contains no tokens (blank lines and # comments don't count)"
            ok=false
        fi
    fi

    local f
    for f in server.pem server.key ca.pem; do
        if [[ ! -f "$TLS_DIR/$f" ]]; then
            echo "Error: missing $TLS_DIR/$f — run: $0 gen-certs <public-hostname>"
            ok=false
        fi
    done

    [[ "$ok" == true ]]
}

# Generate the stunnel config for one slot. verifyPeer + CAfile mean only a
# client holding a certificate from our own CA (i.e. the VPS) can complete the
# handshake — a scanner hitting the port never gets past the TLS layer.
_write_stunnel_conf() {
    local slot="$1" with_proxy="$2" conf="$3"
    local port_server=$(( PORT_BASE_SERVER + slot ))
    local port_proxy=$(( PORT_BASE_PROXY + slot ))
    local tls_server=$(( PORT_BASE_TLS_SERVER + slot ))
    local tls_proxy=$(( PORT_BASE_TLS_PROXY + slot ))

    {
        echo "foreground = yes"
        echo "pid ="
        echo "sslVersionMin = TLSv1.2"
        echo "TIMEOUTidle = 900"
        echo ""
        echo "[server-${slot}]"
        echo "accept = 0.0.0.0:${tls_server}"
        echo "connect = 127.0.0.1:${port_server}"
        echo "cert = $TLS_DIR/server.pem"
        echo "key = $TLS_DIR/server.key"
        echo "CAfile = $TLS_DIR/ca.pem"
        echo "verifyPeer = yes"
        if [[ "$with_proxy" == true ]]; then
            echo ""
            echo "[proxy-${slot}]"
            echo "accept = 0.0.0.0:${tls_proxy}"
            echo "connect = 127.0.0.1:${port_proxy}"
            echo "cert = $TLS_DIR/server.pem"
            echo "key = $TLS_DIR/server.key"
            echo "CAfile = $TLS_DIR/ca.pem"
            echo "verifyPeer = yes"
        fi
    } > "$conf"
}

cmd_start() {
    local name=""
    local slot="1"
    local reasoning=""          # "" = model/template default; off|on|<level>|N (see _translate_reasoning)
    local mlock=false
    local parallel=""
    local ctx=""                # "" = model default; N>0 = override context size
    local cache_ram=""          # "" = server default (8192 MiB); 0 = disable prompt cache; -1 = no limit; N>0 = MiB cap
    local verbose=false         # true = -lv 4 (TRACE): reveals ggml/backend + buffer-size startup logs
    local clear_logs=false      # true = truncate this slot's server/proxy logs before starting
    local host=""               # "" = llama-server default (127.0.0.1); e.g. 0.0.0.0 to expose on LAN
    local gpu_priority=""       # "" = driver default; low|medium|high|realtime (needs patched ggml-vulkan)
    local use_mmproj=false      # true = load the model's mmproj (multimodal projector), if defined
    local start_proxy=false     # true = also start proxy.py (timestamp normalization for prompt caching)
    local public=false          # true = token auth + hardening + stunnel TLS front for the VPS
    local max_predict=""        # "" = model/server default; N = cap tokens per generation
    local spec=""               # "" = on when the model declares spec_args; on|off forces it

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --proxy) start_proxy=true; shift ;;
            --public) public=true; shift ;;
            --max-predict) _need_value "$@" || exit 1; max_predict="$2"; shift 2 ;;
            --reasoning) _need_value "$@" || exit 1; reasoning="$2"; shift 2 ;;
            --reasoning-budget) _need_value "$@" || exit 1; reasoning="$2"; shift 2 ;;   # alias: a budget is a valid --reasoning spec
            --no-reasoning) reasoning=off; shift ;;          # alias for --reasoning off
            --mlock) mlock=true; shift ;;
            --parallel) _need_value "$@" || exit 1; parallel="$2"; shift 2 ;;
            --ctx) _need_value "$@" || exit 1; ctx="$2"; shift 2 ;;
            --cache-ram) _need_value "$@" || exit 1; cache_ram="$2"; shift 2 ;;
            --verbose) verbose=true; shift ;;
            --clear-logs) clear_logs=true; shift ;;
            --host) _need_value "$@" || exit 1; host="$2"; shift 2 ;;
            --gpu-priority) _need_value "$@" || exit 1; gpu_priority="$2"; shift 2 ;;
            --mmproj) use_mmproj=true; shift ;;
            --spec) _need_value "$@" || exit 1; spec="$2"; shift 2 ;;
            --no-spec) spec=off; shift ;;                     # alias for --spec off
            -*) echo "Unknown option: $1"; exit 1 ;;
            *) if [[ -z "$name" ]]; then name="$1"; elif [[ "$slot" == "1" ]]; then slot="$1"; fi; shift ;;
        esac
    done

    [[ -z "$name" ]] && { echo "Usage: $0 start <model-name> [slot] [--proxy] [--public] [--max-predict N] [--reasoning off|on|low|medium|high|max|N] [--mlock] [--parallel N] [--ctx N] [--cache-ram N] [--verbose] [--clear-logs] [--host ADDR] [--gpu-priority low|medium|high|realtime] [--mmproj] [--spec on|off]"; exit 1; }
    [[ "$slot" != "1" && "$slot" != "2" && "$slot" != "3" ]] && { echo "Error: slot must be 1, 2, or 3"; exit 1; }
    if [[ -n "$parallel" && ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --parallel requires a positive integer (got '$parallel')"; exit 1
    fi
    if [[ -n "$ctx" && ! "$ctx" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --ctx requires a positive integer (got '$ctx')"; exit 1
    fi
    if [[ -n "$cache_ram" && ! "$cache_ram" =~ ^(-1|0|[1-9][0-9]*)$ ]]; then
        echo "Error: --cache-ram requires -1 (no limit), 0 (disable prompt cache), or N>0 (MiB cap); got '$cache_ram'"; exit 1
    fi
    if [[ -n "$gpu_priority" && ! "$gpu_priority" =~ ^(low|medium|high|realtime)$ ]]; then
        echo "Error: --gpu-priority requires low, medium, high, or realtime; got '$gpu_priority'"; exit 1
    fi
    if [[ -n "$max_predict" && ! "$max_predict" =~ ^(-1|[1-9][0-9]*)$ ]]; then
        echo "Error: --max-predict requires -1 (no limit) or N>0 (token cap); got '$max_predict'"; exit 1
    fi
    if [[ "$public" == true ]]; then
        _check_public_prereqs || exit 1
        # --public without a cap would let one request generate until --timeout.
        [[ -z "$max_predict" ]] && max_predict="$PUBLIC_MAX_PREDICT_DEFAULT"
    fi

    local port_server=$(( PORT_BASE_SERVER + slot ))
    local port_proxy=$(( PORT_BASE_PROXY + slot ))

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; exit 1; }
    [[ "$start_proxy" == true ]] && { _check_deps python3 || exit 1; }

    # Needs the resolved model, since what --reasoning can mean depends on it.
    _reasoning_args=(); _reasoning_note=""
    if [[ -n "$reasoning" ]]; then
        _translate_reasoning "$reasoning" "$_r_reasoning" "$_r_reasoning_levels" || exit 1
    fi

    # Guard: abort if ports are already in use
    if ss -tlnp 2>/dev/null | grep -q ":${port_server} "; then
        echo "Error: Port $port_server already in use. Is slot $slot already running?"
        exit 1
    fi
    if [[ "$start_proxy" == true ]] && ss -tlnp 2>/dev/null | grep -q ":${port_proxy} "; then
        echo "Error: Port $port_proxy already in use. Is a proxy for slot $slot already running?"
        exit 1
    fi

    mkdir -p "$PID_DIR" "$LOG_DIR" "$SLOT_STATE_DIR"

    local proxy_log="$LOG_DIR/proxy-${slot}.log"
    local server_log="$LOG_DIR/server-${slot}.log"

    # --clear-logs: start this slot's logs fresh (e.g. for a clean cache-stats run)
    if [[ "$clear_logs" == true ]]; then
        : > "$server_log"
        : > "$proxy_log"
        echo "Cleared logs for slot $slot"
    fi

    # Expand tilde in model paths
    local model_path="${_r_model//\~/$HOME}"
    # mmproj (multimodal projector) is only loaded on explicit --mmproj request.
    local mmproj_path=""
    if [[ "$use_mmproj" == true ]]; then
        if [[ -n "$_r_mmproj" ]]; then
            mmproj_path="${_r_mmproj//\~/$HOME}"
        else
            echo "Error: --mmproj requested but model '$name' defines no mmproj"; exit 1
        fi
    fi

    if [[ -n "$spec" && "$spec" != "on" && "$spec" != "off" ]]; then
        echo "Error: --spec takes on or off; got '$spec'" >&2
        exit 1
    fi
    if [[ "$spec" == "on" && ${#_r_spec_args[@]} -eq 0 ]]; then
        echo "Error: model '$_r_name' declares no speculative decoding in models.conf." >&2
        echo "       It needs a draft head in its own GGUF to speculate without a second model." >&2
        exit 1
    fi

    # Build llama-server command
    local -a cmd=("${_r_binary}" \
        --model "$model_path" \
        --port "$port_server")

    # --host exposes the server beyond localhost (e.g. 0.0.0.0 for LAN access).
    [[ -n "$host" ]] && cmd+=(--host "$host")
    [[ -n "$mmproj_path" ]] && cmd+=(--mmproj "$mmproj_path")
    # --parallel and --timeout are managed centrally here, not in models.conf.
    cmd+=(--alias "$_r_client" "${_r_args[@]}" --parallel "${parallel:-1}" --timeout 600)
    # --ctx overrides the model's default context size when given.
    [[ -n "$ctx" ]] && _r_ctx="$ctx"
    [[ -n "$_r_ctx" ]] && cmd+=(--ctx-size "$_r_ctx")
    # --cache-ram overrides the server's default prompt-cache size (8192 MiB).
    # 0 disables the host-RAM prompt cache (keeps system RAM flat during use).
    [[ -n "$cache_ram" ]] && cmd+=(--cache-ram "$cache_ram")
    # Unconditional, so `clear-kv` works on every slot. The trailing slash is
    # required: llama-server concatenates path + filename without a separator.
    cmd+=(--slot-save-path "$SLOT_STATE_DIR/")
    # --public authenticates at llama-server itself. proxy.py forwards the
    # Authorization header untouched, so this one key file covers the direct and
    # the proxied endpoint alike. --no-slots matters most: /slots is enabled by
    # default and exposes other clients' prompts.
    if [[ "$public" == true ]]; then
        cmd+=(--api-key-file "$TOKEN_FILE" --no-webui --no-slots)
    fi
    # A cap on generated tokens; without it a single request can hold the GPU
    # until --timeout expires.
    [[ -n "$max_predict" && "$max_predict" != "-1" ]] && cmd+=(--n-predict "$max_predict")
    # llama.cpp maps ggml/model-load INFO logs to TRACE (level 4); the default
    # verbosity (3=INFO) filters them, so backend detection and buffer-size
    # lines never reach the log. -lv 4 reveals them (also more runtime logging).
    [[ "$verbose" == true ]] && cmd+=(-lv 4)
    [[ "$mlock" == true ]] && cmd+=(--mlock)
    # Whatever --reasoning translated into for this model (may be empty).
    [[ ${#_reasoning_args[@]} -gt 0 ]] && cmd+=("${_reasoning_args[@]}")
    # Speculative decoding: on whenever the model declares it, unless --spec off.
    # The draft head ships inside the model's own GGUF, so this costs no extra file.
    [[ "$spec" != "off" && ${#_r_spec_args[@]} -gt 0 ]] && cmd+=("${_r_spec_args[@]}")

    # The proxy is opt-in (--proxy): it only rewrites time/date stamps to keep the
    # prompt cache warm, which a client that sends no such stamps does not need.
    # Without it, clients talk to llama-server directly (see cmd_env).
    if [[ "$start_proxy" == true ]]; then
        # Start proxy and poll until it listens on the port (max 10s)
        echo "Starting proxy [slot $slot] on port $port_proxy..."
        # -u: unbuffered stdout/stderr so every log line lands in $proxy_log
        # immediately (and nothing is lost when the proxy is killed on stop).
        # LLM_TOKEN_FILE switches the proxy into hardened mode: token check,
        # path allowlist, concurrency and body-size caps. Unset = pass-through.
        local proxy_pid=""
        proxy_pid=$(_spawn_detached "$PID_DIR/proxy-${slot}.pid" "$proxy_log" truncate \
            env LLM_BACKEND_URL="http://localhost:${port_server}" \
                LLM_PROXY_HOST="${host:-127.0.0.1}" \
                LLM_PROXY_PORT="${port_proxy}" \
                LLM_TOKEN_FILE="$([[ "$public" == true ]] && echo "$TOKEN_FILE")" \
                python3 -u "$PROXY_SCRIPT")
        if [[ -z "$proxy_pid" ]]; then
            echo "Error: the proxy did not report a PID. See $proxy_log"
            exit 1
        fi

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
    fi

    # Start llama-server in background
    echo "Starting llama.cpp server [slot $slot] on port $port_server..."

    # Behaviour-relevant llama.cpp defaults we do NOT pass, so they stay hidden.
    _has_flag() { local f; for f in "${cmd[@]}"; do [[ "$f" == "$1" ]] && return 0; done; return 1; }

    # Render the effective start config once. It is printed to the terminal AND
    # written to the top of the server log (below), so a saved log is self-describing.
    # The Command line is the ground truth of every flag (samplers, cache-type-k/v,
    # flash-attn, threads, ngl come from models.conf); the Defaults section shows
    # behaviour-relevant llama.cpp defaults we never pass.
    _render_config() {
        echo "Model: $_r_label ($_r_alias)"
        if [[ -n "$mmproj_path" ]]; then
            echo "mmproj: $mmproj_path (--mmproj)"
        elif [[ -n "$_r_mmproj" ]]; then
            echo "mmproj: available but not loaded (pass --mmproj)"
        fi
        echo "ROCm env: ${_r_rocm_env:-—}"
        echo "Host:     ${host:-127.0.0.1 (default)}"
        # The LAN listener carries no authentication unless --public added a key
        # file; the TLS front below is a separate, always-authenticated path.
        if [[ -n "$host" && "$host" != "127.0.0.1" && "$host" != "localhost" && "$host" != "::1" ]]; then
            local where="beyond localhost ($host)"
            [[ "$host" == "0.0.0.0" || "$host" == "::" ]] && where="to ALL interfaces"
            if [[ "$public" == true ]]; then
                echo "  NOTE: bound $where; token auth is active (--public)."
            else
                echo "  WARNING: bound $where without authentication —"
                echo "           expose only inside a trusted network."
            fi
        fi
        if [[ "$public" == true ]]; then
            echo "Public:   enabled (--public)"
            echo "  Tokens:  $(_token_count) from $TOKEN_FILE"
            echo "  TLS in:  0.0.0.0:$(( PORT_BASE_TLS_SERVER + slot )) → server :$port_server (mTLS, client cert required)"
            if [[ "$start_proxy" == true ]]; then
                echo "           0.0.0.0:$(( PORT_BASE_TLS_PROXY + slot )) → proxy  :$port_proxy (mTLS, client cert required)"
            fi
            echo "  Server:  --no-webui --no-slots, generation capped at ${max_predict} tokens"
            echo "  Proxy:   path allowlist, token check, concurrency cap"
        else
            echo "Public:   disabled — reachable only via localhost/LAN (pass --public to expose via the VPS)"
        fi
        if [[ "$start_proxy" == true ]]; then
            echo "Proxy:    enabled on :$port_proxy (--proxy) — normalizes time/date stamps for the prompt cache"
        else
            echo "Proxy:    disabled — clients talk to :$port_server directly (pass --proxy to enable)"
        fi
        echo "Context:  ${_r_ctx:-default}"
        echo "Parallel: ${parallel:-1}"
        echo "Timeout:  600s"
        if [[ -n "$cache_ram" ]]; then
            case "$cache_ram" in
                0)  echo "PromptCache: disabled (--cache-ram 0)" ;;
                -1) echo "PromptCache: no limit (--cache-ram -1)" ;;
                *)  echo "PromptCache: ${cache_ram} MiB (--cache-ram ${cache_ram})" ;;
            esac
        else
            echo "PromptCache: 8192 MiB (server default)"
        fi
        if [[ -n "$reasoning" ]]; then
            echo "Reasoning: ${_reasoning_note}"
            [[ "$_r_reasoning" == "unknown" ]] && \
                echo "  NOTE: this model's template support is unverified in models.conf —"
            [[ "$_r_reasoning" == "unknown" ]] && \
                echo "        the flags are passed through, run '$0 probe-reasoning' once it is downloaded."
        else
            echo "Reasoning: template default (model supports: $_r_reasoning)"
        fi
        if [[ ${#_r_spec_args[@]} -eq 0 ]]; then
            echo "Speculative: not available for this model"
        elif [[ "$spec" == "off" ]]; then
            echo "Speculative: off (--spec off)"
        else
            echo "Speculative: on (${_r_spec_args[*]})"
        fi
        [[ "$mlock" == true ]] && echo "mlock: enabled (--mlock)"
        [[ -n "$gpu_priority" ]] && echo "GPU priority: $gpu_priority (GGML_VK_QUEUE_PRIORITY; needs patched ggml-vulkan)"
        echo "----- effective config -----"
        echo "Command:"
        printf '  %s\n' "${cmd[*]}"
        echo "Defaults in effect (not overridden):"
        _has_flag --slot-prompt-similarity || echo "  slot-prompt-similarity: 0.1  (LCP slot routing; 0 = pure LRU)"
        _has_flag --batch-size            || echo "  batch-size:             2048"
        _has_flag --ubatch-size           || echo "  ubatch-size:            512"
        _has_flag --keep                  || echo "  keep:                   0"
        _has_flag -lv                     || echo "  verbosity:              3 (INFO)  — use --verbose for TRACE"
        echo "----------------------------"
    }
    local config_text
    config_text="$(_render_config)"
    echo "$config_text"

    # Preload jemalloc for the server (better allocation behaviour under load).
    # stdbuf -oL -eL line-buffers stdio so raw ggml/ROCm prints (llama.cpp only
    # flushes its own LOG_* lines) also reach $server_log promptly; it appends
    # libstdbuf to LD_PRELOAD, coexisting with the jemalloc preload.
    # _r_rocm_env is intentionally unquoted — word-splits space-separated KEY=VAL pairs for env
    # --gpu-priority sets GGML_VK_QUEUE_PRIORITY for the patched ggml-vulkan backend
    # (VK_EXT_global_priority). Empty var = not set = driver default. See patches/.
    local gpu_prio_env=""
    [[ -n "$gpu_priority" ]] && gpu_prio_env="GGML_VK_QUEUE_PRIORITY=$gpu_priority"
    # Write the effective config to the top of a fresh server log, then let the
    # server append below it (>>), so the saved log is self-describing. Each start
    # truncates first, preserving the fresh-log-per-start behaviour.
    : > "$server_log"
    printf '%s\n\n' "$config_text" >> "$server_log"
    # Detached, so the model outlives the terminal it was started from.
    # _r_rocm_env is unquoted on purpose: it word-splits into KEY=VAL pairs for env.
    local server_pid=""
    server_pid=$(_spawn_detached "$PID_DIR/server-${slot}.pid" "$server_log" append \
        env LD_PRELOAD=/lib64/libjemalloc.so.2 $gpu_prio_env $_r_rocm_env stdbuf -oL -eL "${cmd[@]}")
    if [[ -z "$server_pid" ]]; then
        echo "Error: the server did not report a PID. See $server_log" >&2
        return 1
    fi

    echo "Server running (PID: $server_pid, detached)"

    # Start the TLS front last, so the port only opens once there is a backend
    # behind it.
    local stunnel_log="$LOG_DIR/stunnel-${slot}.log"
    if [[ "$public" == true ]]; then
        mkdir -p "$STUNNEL_DIR"
        local stunnel_conf="$STUNNEL_DIR/slot-${slot}.conf"
        _write_stunnel_conf "$slot" "$start_proxy" "$stunnel_conf"
        local stunnel_pid=""
        stunnel_pid=$(_spawn_detached "$PID_DIR/stunnel-${slot}.pid" "$stunnel_log" truncate \
            stunnel "$stunnel_conf")
        if [[ -z "$stunnel_pid" ]]; then
            echo "Error: stunnel did not report a PID. See $stunnel_log"
            exit 1
        fi

        sleep 1
        if ! ps -p "$stunnel_pid" > /dev/null 2>&1; then
            echo "Error: stunnel failed to start. See $stunnel_log"
            exit 1
        fi
        echo "TLS front running (PID: $stunnel_pid) on :$(( PORT_BASE_TLS_SERVER + slot ))$(
            [[ "$start_proxy" == true ]] && echo ", :$(( PORT_BASE_TLS_PROXY + slot ))")"
    fi

    echo ""
    echo "  Logs:  tail -f $server_log"
    [[ "$start_proxy" == true ]] && echo "         tail -f $proxy_log"
    [[ "$public" == true ]] && echo "         tail -f $stunnel_log"
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

    # Stop the TLS front — closing the public port first would be nicer, but the
    # slot is going down either way and stunnel is the cheapest to restart.
    local stunnel_pid=""
    [[ -f "$PID_DIR/stunnel-${slot}.pid" ]] && stunnel_pid=$(cat "$PID_DIR/stunnel-${slot}.pid")
    if [[ -n "$stunnel_pid" ]] && ps -p "$stunnel_pid" > /dev/null 2>&1; then
        kill "$stunnel_pid" 2>/dev/null
        echo "Slot $slot: stunnel (PID: $stunnel_pid) stopped."
        stopped=$(( stopped + 1 ))
    fi

    rm -f "$PID_DIR/server-${slot}.pid" "$PID_DIR/proxy-${slot}.pid" "$PID_DIR/stunnel-${slot}.pid"
    rm -f "$STUNNEL_DIR/slot-${slot}.conf"
    rmdir "$STUNNEL_DIR" 2>/dev/null || true
    # Only succeeds while nothing was saved there via ?action=save.
    rmdir "$SLOT_STATE_DIR" 2>/dev/null || true
    return $(( stopped == 0 ))
}

cmd_stop() {
    local slot="${1:-}"

    if [[ -n "$slot" ]]; then
        [[ "$slot" != "1" && "$slot" != "2" && "$slot" != "3" ]] && { echo "Error: slot must be 1, 2, or 3"; return 1; }
        _stop_slot "$slot" || echo "Slot $slot: nothing was running."
    else
        local total=0
        for s in 1 2 3; do
            _stop_slot "$s" && total=$(( total + 1 ))
        done
        [[ "$total" -eq 0 ]] && echo "Nothing was running."
    fi

    rmdir "$PID_DIR" 2>/dev/null || true
}

cmd_status() {
    local found=0
    for slot in 1 2 3; do
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

        local stunnel_pid=""
        [[ -f "$PID_DIR/stunnel-${slot}.pid" ]] && stunnel_pid=$(cat "$PID_DIR/stunnel-${slot}.pid")
        if [[ -n "$stunnel_pid" ]] && ps -p "$stunnel_pid" > /dev/null 2>&1; then
            local tls_ports=":$(( PORT_BASE_TLS_SERVER + slot ))"
            ss -tlnp 2>/dev/null | grep -q ":$(( PORT_BASE_TLS_PROXY + slot )) " && \
                tls_ports="$tls_ports, :$(( PORT_BASE_TLS_PROXY + slot ))"
            echo "Slot $slot: stunnel     (PID: $stunnel_pid) on $tls_ports  [public, mTLS]"
            slot_active=1; found=1
        fi
    done
    [[ "$found" -eq 0 ]] && echo "Nothing running."
}

# Summarize prompt-cache effectiveness from the server log (works whether or
# not clients go through the proxy). Per finished task the INFO log carries
# processed prompt tokens ("prompt eval time = .. / N"), generated tokens
# ("eval time = .. / M") and the final context size ("stop processing:
# n_tokens = T"); prompt_total = T - M and cached = prompt_total - N, which
# equals llama-server's own n_prompt_tokens_cache. The log is truncated on
# each start, so this reflects the current server session.
# Usage: cache-stats [slot]  (default: all slots with a server log)
cmd_cache_stats() {
    local want_slot="${1:-}"
    local found=0
    for slot in 1 2 3; do
        [[ -n "$want_slot" && "$slot" != "$want_slot" ]] && continue
        local f="$LOG_DIR/server-${slot}.log"
        [[ -f "$f" ]] || continue
        found=1
        echo "Slot $slot: $f"
        python3 - "$f" <<'PY'
import re, sys
proc, gen, total = {}, {}, {}
re_pe = re.compile(r"task\s+(\d+) \| prompt eval time =.*?/\s*(\d+) tokens")
re_ev = re.compile(r"task\s+(\d+) \|\s+eval time =.*?/\s*(\d+) tokens")
re_st = re.compile(r"task\s+(\d+) \| stop processing: n_tokens = (\d+)")
for ln in open(sys.argv[1], errors="replace"):
    m = re_pe.search(ln);  m and proc.update({m.group(1): int(m.group(2))})
    m = re_ev.search(ln);  m and gen.update({m.group(1): int(m.group(2))})
    m = re_st.search(ln);  m and total.update({m.group(1): int(m.group(2))})
n = tp = tc = 0
for tid, T in total.items():
    ptotal = T - gen.get(tid, 0)          # prompt tokens = context minus generated
    if ptotal <= 0:
        continue
    cached = max(0, ptotal - proc.get(tid, 0))
    n += 1; tp += ptotal; tc += cached
if tp:
    print(f"  tasks={n}  prompt_tokens={tp}  cached={tc}  "
          f"reprocessed={tp-tc}  hit_rate={100*tc/tp:.1f}%")
else:
    print("  no finished completion tasks in the log yet")
PY
    done
    [[ "$found" -eq 0 ]] && echo "No server logs yet (start a server first)."
    return 0
}

# Drop the KV cache of the running slots without restarting the server:
# POST /slots/{id}?action=erase for every server slot (--parallel N creates
# ids 0..N-1). That action needs --slot-save-path, which cmd_start always passes.
#
# Two things this does *not* do: it never interrupts a running generation (the
# server defers the erase until the slot falls idle, so the call can block), and
# it leaves the host-RAM prompt cache alone (--cache-ram, 8192 MiB by default) —
# a matching prompt can be restored from there afterwards.
#
# Usage: clear-kv [slot]  (default: every running slot)
cmd_clear_kv() {
    local want_slot="${1:-}"
    if [[ -n "$want_slot" && "$want_slot" != "1" && "$want_slot" != "2" && "$want_slot" != "3" ]]; then
        echo "Error: slot must be 1, 2, or 3"; return 1
    fi
    _check_deps python3 || return 1

    # A slot started with --public requires a token; reuse the first one, exactly
    # as cmd_env does. A server without --api-key-file ignores the header.
    local token=""
    if [[ -r "$TOKEN_FILE" ]]; then
        token=$(grep -m1 -vE '^[[:space:]]*(#|$)' "$TOKEN_FILE" 2>/dev/null)
    fi

    local found=0
    for slot in 1 2 3; do
        [[ -n "$want_slot" && "$slot" != "$want_slot" ]] && continue
        local server_pid=""
        [[ -f "$PID_DIR/server-${slot}.pid" ]] && server_pid=$(cat "$PID_DIR/server-${slot}.pid")
        [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null 2>&1 || continue
        found=1

        echo "Slot $slot: clearing KV on :$(( PORT_BASE_SERVER + slot ))"
        LLM_KV_TOKEN="$token" python3 - "$(( PORT_BASE_SERVER + slot ))" <<'KVPY'
import json, os, sys, urllib.error, urllib.request

base    = f"http://127.0.0.1:{sys.argv[1]}"
token   = os.environ.get("LLM_KV_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"} if token else {}

def call(path, method="GET", timeout=30):
    req = urllib.request.Request(f"{base}{path}", method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

try:
    # /props reports the slot count and stays readable even with --no-slots.
    n_slots = int(call("/props").get("total_slots", 1))
except Exception as e:
    print(f"  Error: cannot read /props ({e})")
    sys.exit(1)

total = 0
for i in range(n_slots):
    try:
        # Generous timeout: a busy slot only frees up once its generation ends,
        # which the server bounds with --timeout 600.
        res = call(f"/slots/{i}?action=erase", method="POST", timeout=620)
    except urllib.error.HTTPError as e:
        print(f"  slot {i}: HTTP {e.code} — {e.read().decode(errors='replace').strip()}")
        continue
    except Exception as e:
        print(f"  slot {i}: {e}")
        continue
    n = int(res.get("n_erased", 0))
    total += n
    print(f"  slot {i}: {n} tokens erased")

print(f"  total: {total} tokens")
KVPY

        # The prompt could still come back from host RAM, which no endpoint clears.
        local pc
        pc=$(grep -m1 '^PromptCache:' "$LOG_DIR/server-${slot}.log" 2>/dev/null)
        if [[ -n "$pc" && "$pc" != *disabled* ]]; then
            echo "  Note: ${pc#PromptCache: } — a matching prompt may be restored from host RAM."
            echo "        Start the slot with --cache-ram 0 for a hard reset."
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "No running server${want_slot:+ in slot $want_slot}."
        return 1
    fi
    return 0
}

# Read the chat template each model actually runs with and report what it
# supports, next to what models.conf claims. This is the same signal llama.cpp
# probes at load time (jinja::caps_*): enable_thinking for on/off,
# reasoning_effort / reasoning_strength for levels. The template is the one baked
# into the GGUF unless extra_args override it with --chat-template-file, which is
# what the server would then load — probing the GGUF there would report the
# template the model ships with rather than the one it runs.
#
# Usage: probe-reasoning [model]
cmd_probe_reasoning() {
    local want="${1:-}"
    _check_deps python3 || return 1

    # gguf-py ships with the llama.cpp sources, not with the binary.
    local gguf_py=""
    if python3 -c "import gguf" 2>/dev/null; then
        gguf_py=""
    elif [[ -d "${LLAMA_SRC:-$HOME/src/llama.cpp}/gguf-py" ]]; then
        gguf_py="${LLAMA_SRC:-$HOME/src/llama.cpp}/gguf-py"
    else
        echo "Error: needs the gguf python module — either 'pip install gguf' or a"
        echo "       llama.cpp checkout (set LLAMA_SRC, default ~/src/llama.cpp)"
        return 1
    fi

    printf "  %-11s %-11s %-11s %s\n" "MODEL" "CONFIGURED" "TEMPLATE" "READS"
    printf "  %-11s %-11s %-11s %s\n" "-----" "----------" "--------" "-----"

    local entry
    for entry in "${_MODELS[@]}"; do
        IFS='|' read -r m_name _ m_model _ _ _ m_args _ _ _ _ _ _ _ m_reasoning _ <<< "$entry"
        [[ -n "$want" && "$m_name" != "$want" ]] && continue
        local path="${m_model//\~/$HOME}"

        if [[ ! -f "$path" ]]; then
            # ASCII dash: bash pads printf by bytes, so a multibyte one misaligns.
            printf "  %-11s %-11s %-11s %s\n" "$m_name" "$m_reasoning" "-" "not downloaded"
            continue
        fi
        # --chat-template-file <path> anywhere in extra_args wins over the GGUF.
        # read -ra rather than an unquoted expansion: args are split on spaces
        # here, but a stray glob in them must not hit the filesystem.
        local tmpl_file="" prev=""
        local -a args_arr=()
        read -ra args_arr <<< "$m_args"
        local word
        for word in "${args_arr[@]}"; do
            [[ "$prev" == "--chat-template-file" ]] && { tmpl_file="$word"; break; }
            prev="$word"
        done
        GGUF_PY="$gguf_py" python3 - "$m_name" "$m_reasoning" "$path" "$tmpl_file" <<'PROBEPY'
import os, sys
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader

name, configured, path, tmpl_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

tmpl, source = None, ""
if tmpl_file:
    source = f"  [{os.path.basename(tmpl_file)}]"
    if not os.path.isfile(tmpl_file):
        print(f"  {name:<11} {configured:<11} {'?':<11} --chat-template-file {tmpl_file} is missing")
        sys.exit(0)
    with open(tmpl_file, encoding="utf-8") as fh:
        tmpl = fh.read()
else:
    for f in GGUFReader(path).fields.values():
        if f.name == "tokenizer.chat_template":
            tmpl = f.contents()

if tmpl is None:
    print(f"  {name:<11} {configured:<11} {'—':<11} no chat template in the GGUF")
    sys.exit(0)

reads = [v for v in ("enable_thinking", "reasoning_effort", "reasoning_strength") if v in tmpl]
if "reasoning_effort" in reads or "reasoning_strength" in reads:
    seen = "effort"
elif "enable_thinking" in reads:
    seen = "toggle"
elif "<think>" in tmpl:
    # Think tags present but nothing to switch on them: hard-coded either way.
    seen = "locked-off"
else:
    seen = "none"

mark = "" if seen == configured else "   <-- differs"
print(f"  {name:<11} {configured:<11} {seen:<11} {', '.join(reads) or '—'}{source}{mark}")
PROBEPY
    done
    return 0
}

cmd_env() {
    local name=""
    local slot="1"
    local want=""               # "" = auto (proxy if one is listening); proxy|direct = forced

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --proxy) want="proxy"; shift ;;
            --direct) want="direct"; shift ;;
            -*) echo "Unknown option: $1"; return 1 ;;
            *) if [[ -z "$name" ]]; then name="$1"; elif [[ "$slot" == "1" ]]; then slot="$1"; fi; shift ;;
        esac
    done

    [[ -z "$name" ]] && { echo "Usage: source $0 env <model-name> [slot] [--proxy|--direct]"; return 1; }
    [[ "$slot" != "1" && "$slot" != "2" && "$slot" != "3" ]] && { echo "Error: slot must be 1, 2, or 3"; return 1; }

    local port_proxy=$(( PORT_BASE_PROXY + slot ))
    local port_server=$(( PORT_BASE_SERVER + slot ))

    # Since the proxy is opt-in (start --proxy), point at whichever endpoint is
    # actually up: the proxy when it listens, the server otherwise. --proxy /
    # --direct force the choice (e.g. to set env before starting the slot).
    local port_target="$port_server" via="server"
    case "$want" in
        proxy)  port_target="$port_proxy"; via="proxy" ;;
        direct) port_target="$port_server"; via="server" ;;
        *)      if ss -tlnp 2>/dev/null | grep -q ":${port_proxy} "; then
                    port_target="$port_proxy"; via="proxy"
                fi ;;
    esac

    _resolve_model "$name" || { echo "Unknown model: $name"; cmd_list; return 1; }

    local config_dir=".claude-${_r_name}"
    [[ "$slot" -gt 1 ]] && config_dir=".claude-${_r_name}-${slot}"

    # A slot started with --public demands a token. Reuse the first one from the
    # token file so the local workflow stays a plain `source ./run.sh env`; a
    # server started without --public ignores the header anyway.
    local token="sk-no-key-required" token_src="placeholder"
    if [[ -r "$TOKEN_FILE" ]]; then
        local first
        first=$(grep -m1 -vE '^[[:space:]]*(#|$)' "$TOKEN_FILE" 2>/dev/null)
        [[ -n "$first" ]] && { token="$first"; token_src="$TOKEN_FILE"; }
    fi

    # When sourced, these exports take effect in the caller's shell.
    # When executed directly, they're printed for the user to see.
    export CLAUDE_CONFIG_DIR="$config_dir"
    export ANTHROPIC_BASE_URL="http://localhost:$port_target"
    export ANTHROPIC_AUTH_TOKEN="$token"
    export ANTHROPIC_MODEL="$_r_client"
    export ANTHROPIC_SMALL_FAST_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$_r_client"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$_r_client"
    export API_TIMEOUT_MS="3000000"
    export PI_STREAM_FIRST_EVENT_TIMEOUT_MS="600000"
    export PI_OPENAI_STREAM_IDLE_TIMEOUT_MS="600000"
    export OPENAI_BASE_URL="http://localhost:$port_target/v1"
    export OPENAI_API_KEY="$token"

    echo "Claude Code env set for $_r_label ($_r_client) [slot $slot] via $via"
    echo "  CLAUDE_CONFIG_DIR=$config_dir"
    echo "  Token: $token_src"
    echo "  ANTHROPIC_BASE_URL=http://localhost:$port_target"
    echo "  OPENAI_BASE_URL=http://localhost:$port_target/v1"
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
    unset OPENAI_BASE_URL
    unset OPENAI_API_KEY

    echo "Environment cleared."
    echo "Run 'source $0 env <name> [slot]' to set up a model again."
}

# Create the private CA and the two certificates the public path needs:
#   server.pem/.key       — presented by stunnel here; SAN must be the hostname
#                           the VPS connects to, or SSLProxyCheckPeerName fails
#   vps-client-combined.pem — copied to the VPS; without it stunnel's verifyPeer
#                           refuses the handshake, so scanners never get through
cmd_gen_certs() {
    local hostname="" force=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force) force=true; shift ;;
            -*) echo "Unknown option: $1"; return 1 ;;
            *) [[ -z "$hostname" ]] && hostname="$1"; shift ;;
        esac
    done
    [[ -z "$hostname" ]] && { echo "Usage: $0 gen-certs <public-hostname> [--force]"; return 1; }
    _check_deps openssl || return 1

    if [[ -f "$TLS_DIR/ca.pem" && "$force" != true ]]; then
        echo "Error: $TLS_DIR/ca.pem already exists."
        echo "       Re-issuing invalidates the client cert already on the VPS. Pass --force if that's intended."
        return 1
    fi

    mkdir -p "$TLS_DIR"
    chmod 700 "$TLS_DIR"
    ( umask 077 && cd "$TLS_DIR" && \
      openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
          -keyout ca.key -out ca.pem -subj "/CN=llm-local-ca" 2>/dev/null && \
      openssl req -newkey rsa:4096 -nodes -keyout server.key -out server.csr \
          -subj "/CN=$hostname" 2>/dev/null && \
      openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
          -out server.pem -days 3650 -sha256 \
          -extfile <(printf 'subjectAltName=DNS:%s\nextendedKeyUsage=serverAuth\n' "$hostname") 2>/dev/null && \
      openssl req -newkey rsa:4096 -nodes -keyout vps-client.key -out vps-client.csr \
          -subj "/CN=vps" 2>/dev/null && \
      openssl x509 -req -in vps-client.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
          -out vps-client.pem -days 3650 -sha256 \
          -extfile <(printf 'extendedKeyUsage=clientAuth\n') 2>/dev/null && \
      cat vps-client.pem vps-client.key > vps-client-combined.pem && \
      rm -f server.csr vps-client.csr ) || { echo "Error: certificate generation failed"; return 1; }

    echo "Certificates written to $TLS_DIR (valid 10 years, CN/SAN = $hostname)"
    echo ""
    echo "  Stays here:      ca.pem  server.pem  server.key"
    echo "  Copy to the VPS: vps-client-combined.pem   → SSLProxyMachineCertificateFile"
    echo "                   ca.pem                    → SSLProxyCACertificateFile"
    echo ""
    echo "  scp $TLS_DIR/vps-client-combined.pem $TLS_DIR/ca.pem <vps>:/etc/apache2/llm/"
    echo ""
    echo "  Note: the VPS must reach $hostname (DynDNS → your router), and"
    echo "        vps-client-combined.pem holds a private key — chmod 600 it there."
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
            IFS='|' read -r m_name _ m_model _ _ m_label _ _ _ _ _ _ _ <<< "$entry"
            echo "=== $m_label ==="
            bench_model "$m_name" "$m_model" "$m_label" "$full" "$result_file"
            echo ""
        done
    else
        local found=0
        for entry in "${_MODELS[@]}"; do
            IFS='|' read -r m_name _ m_model _ _ m_label _ _ _ _ _ _ _ <<< "$entry"
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
    printf "  %-20s %s\n" "start <name> [slot]"   "start server (slot 1-3, default 1)"
    printf "  %-20s %s\n" ""                      "  --proxy: also start proxy.py (normalizes time/date stamps for the prompt cache)"
    printf "  %-20s %s\n" ""                      "  --reasoning off|on|low|medium|high|max|N: one switch for every model;"
    printf "  %-20s %s\n" ""                      "    levels and on/off are translated per model (see 'probe-reasoning'), N = token budget"
    printf "  %-20s %s\n" ""                      "  --no-reasoning / --reasoning-budget N: kept as aliases; --parallel N: server slots (default 1)"
    printf "  %-20s %s\n" ""                      "  --ctx N: override the model's default context size"
    printf "  %-20s %s\n" ""                      "  --cache-ram N: prompt-cache host-RAM cap in MiB (0=disable, -1=no limit; default 8192)"
    printf "  %-20s %s\n" ""                      "  --spec on|off / --no-spec: speculative decoding; on by default where the"
    printf "  %-20s %s\n" ""                      "    model's own GGUF carries a draft head (currently qwen3.8)"
    printf "  %-20s %s\n" ""                      "  --verbose: -lv 4, reveals ggml/backend + buffer-size startup logs (more runtime logging too)"
    printf "  %-20s %s\n" ""                      "  --public: token auth + hardening + mTLS front for the VPS (needs gen-certs)"
    printf "  %-20s %s\n" ""                      "  --max-predict N: cap tokens per generation (-1=no limit; --public defaults to 8192)"
    printf "  %-20s %s\n" "stop [slot]"            "stop slot (or all if omitted)"
    printf "  %-20s %s\n" "status"                 "show running state"
    printf "  %-20s %s\n" "cache-stats [slot]"     "show prompt-cache hit rate (from the server log)"
    printf "  %-20s %s\n" "clear-kv [slot]"        "drop the KV cache of every server slot (or one slot)"
    printf "  %-20s %s\n" "probe-reasoning [m]"    "what each downloaded model's chat template supports vs. models.conf"
    printf "  %-20s %s\n" "gen-certs <host>"       "create the CA + server/VPS certificates for --public"
    printf "  %-20s %s\n" "bench [opts] <m>"       "run benchmark (model or 'all')"
    printf "  %-20s %s\n" "list"                   "show available models"
    printf "  %-20s %s\n" "env <name> [slot]"      "set Claude Code env vars (source!); --proxy/--direct force the endpoint"
    printf "  %-20s %s\n" "clear"                  "clear env vars (source!)"
    printf "  %-20s %s\n" "download <m>"           "download model(s) (or 'all')"
    echo ""
    echo "  Ports:  slot 1 → server :8001  proxy :8081"
    echo "          slot 2 → server :8002  proxy :8082"
    echo "          slot 3 → server :8003  proxy :8083"
    echo ""
    echo "  --public adds a TLS front per slot: server :844N, proxy :845N (N = slot)"
    echo "          Forward only those at the router; :800N/:808N stay local/LAN."
    echo ""
}

# ── Download ────────────────────────────────────────────────

# Where a model's draft belongs: the directory of the file its spec_args point
# --model-draft at. Deriving it keeps one source of truth for that path, instead
# of a second field that silently disagrees after an edit.
_draft_dir_from_spec() {
    local prev="" word
    for word in "$@"; do
        case "$prev" in
            -md|--model-draft|--spec-draft-model) dirname "$word"; return 0 ;;
        esac
        prev="$word"
    done
    return 1
}

# Some files a model needs live in a different repo than the model itself: the
# draft it speculates with, and — for the uncensored gemmas — the vision projector,
# which only the stock repo ships. Both fields hold "<repo> <include-pattern>".
_download_draft() {
    local draft_spec="$1" force="${2:-}"
    shift 2 2>/dev/null || shift $#
    local repo pat
    read -r repo pat <<< "$draft_spec"
    [[ -z "$repo" ]] && return 0
    local dir
    dir=$(_draft_dir_from_spec "$@") || {
        echo "  Skipping draft: spec_args name no --model-draft path" >&2
        return 0
    }
    echo "  Draft model:"
    _download_model "$repo" "$dir" "$pat" "$force"
}

# Same idea for the mmproj, whose destination is simply where _model_mmproj says
# the file goes.
_download_mmproj() {
    local spec="$1" mmproj_path="$2" force="${3:-}"
    local repo pat
    read -r repo pat <<< "$spec"
    [[ -z "$repo" || -z "$mmproj_path" ]] && return 0
    echo "  Vision projector:"
    _download_model "$repo" "$(dirname "${mmproj_path//\~/$HOME}")" "$pat" "$force"
}


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
            IFS='|' read -r m_name _ m_model m_mmproj _ m_label _ _ _ m_hf_repo m_hf_includes m_hf_dir _ _ _ _ m_spec_args m_hf_draft m_hf_mmproj <<< "$entry"
            if [[ -z "$m_hf_repo" ]]; then
                echo "  Skipping $m_label: no download info configured"
                continue
            fi
            local model_dir
            if [[ -n "$m_hf_dir" ]]; then
                model_dir="${m_hf_dir//\~/$HOME}"
            else
                model_dir="$(dirname "${m_model//\~/$HOME}")"
            fi
            echo "=== $m_label ==="
            _download_model "$m_hf_repo" "$model_dir" "$m_hf_includes" "$force"
            if [[ -n "$m_hf_draft" ]]; then
                local -a sa=(); read -ra sa <<< "$m_spec_args"
                _download_draft "$m_hf_draft" "$force" "${sa[@]}"
            fi
            [[ -n "$m_hf_mmproj" ]] && _download_mmproj "$m_hf_mmproj" "$m_mmproj" "$force"
        done
        return
    fi

    _resolve_model "$target" || { echo "Unknown model: $target"; echo "Run '$0 download' to see available models."; return 1; }

    if [[ -z "$_r_hf_repo" ]]; then
        echo "No download info configured for model: $target"
        return 1
    fi

    local model_dir
    if [[ -n "$_r_hf_dir" ]]; then
        model_dir="${_r_hf_dir//\~/$HOME}"
    else
        model_dir="$(dirname "${_r_model//\~/$HOME}")"
    fi
    _download_model "$_r_hf_repo" "$model_dir" "$_r_hf_includes" "$force"
    [[ -n "$_r_hf_draft" ]] && _download_draft "$_r_hf_draft" "$force" "${_r_spec_args[@]}"
    [[ -n "$_r_hf_mmproj" ]] && _download_mmproj "$_r_hf_mmproj" "$_r_mmproj" "$force"
    return 0
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
        cache-stats) shift; cmd_cache_stats "$@" ;;
        clear-kv)   shift; cmd_clear_kv "$@" ;;
        probe-reasoning) shift; cmd_probe_reasoning "$@" ;;
        gen-certs)  shift; cmd_gen_certs "$@" ;;
        bench)      shift; cmd_benchmark "$@" ;;
        list)       cmd_list ;;
        help)       cmd_help ;;
        env)        echo "This command must be sourced:  source $0 env <name> [slot] [--proxy|--direct]" ;;
        download)   shift; cmd_download "$@" ;;
        *)          cmd_help; cmd_list ;;
    esac
fi
