#!/bin/bash
# Smoke tests for llmctl: every command that can answer without a GPU, a model
# or a network — which is most of what goes wrong. They run against a sandbox
# (its own config, data, state and models directories) with stub binaries on the
# PATH, so nothing here touches a real installation or starts a server.
#
#   tests/smoke.sh            all of them
#   tests/smoke.sh voice      only the tests whose name contains "voice"
#
# Each case names what it checks, runs a command, and looks for text in its
# output. A case that fails prints the whole output.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTER="${1:-}"
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

export LLMCTL_CONFIG_DIR="$SANDBOX/config"
export LLMCTL_DATA_DIR="$SANDBOX/data"
export LLMCTL_STATE_DIR="$SANDBOX/state"
export LLMCTL_MODELS_DIR="$SANDBOX/models"
mkdir -p "$LLMCTL_CONFIG_DIR" "$LLMCTL_DATA_DIR" "$LLMCTL_MODELS_DIR" "$SANDBOX/bin"
cp "$ROOT/examples/models.conf" "$ROOT/examples/presets.conf" "$LLMCTL_CONFIG_DIR/"

# Stubs: the commands llmctl checks for before it builds anything. podman
# answers "no such container", so no slot looks like a running halogen.
for b in llama-server llama-tts hf stunnel git cmake; do
    printf '#!/bin/sh\nexit 0\n' > "$SANDBOX/bin/$b"
done
printf '#!/bin/sh\nexit 1\n' > "$SANDBOX/bin/podman"
chmod +x "$SANDBOX"/bin/*
export PATH="$SANDBOX/bin:$PATH"

# The model files models.conf names: only their presence is checked here.
for f in $(grep -oE '\$(MODELS_DIR|DATA_DIR)[^"]*' "$LLMCTL_CONFIG_DIR/models.conf" | sort -u); do
    path="${f/\$MODELS_DIR/$LLMCTL_MODELS_DIR}"; path="${path/\$DATA_DIR/$LLMCTL_DATA_DIR}"
    case "$path" in *.gguf|*.hgn|*.safetensors) mkdir -p "$(dirname "$path")"; : > "$path" ;; esac
done

pass=0 fail=0 skipped=0

# check <name> <expected text> -- <command...>   (expected "!text": must NOT appear)
check() {
    local name="$1" want="$2"; shift 3
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then skipped=$(( skipped + 1 )); return; fi
    local out rc
    out=$("$@" 2>&1); rc=$?
    local ok=true
    if [[ "$want" == '!'* ]]; then
        [[ "$out" == *"${want#!}"* ]] && ok=false
    else
        [[ "$out" == *"$want"* ]] || ok=false
    fi
    if [[ "$ok" == true ]]; then
        printf '  ok    %s\n' "$name"; pass=$(( pass + 1 ))
    else
        printf '  FAIL  %s\n        wanted: %s\n        got (rc=%s):\n%s\n' "$name" "$want" "$rc" "${out//$'\n'/$'\n'          }"
        fail=$(( fail + 1 ))
    fi
}

L="$ROOT/llmctl"

echo "llmctl smoke tests (sandbox: $SANDBOX)"

# ── the program answers at all ───────────────────────────────
check "version prints one"              "llmctl 1."        -- "$L" version
check "help lists the speech backend"   "/v1/audio/speech" -- "$L" help
check "list shows every backend"        "ComfyUI (images)" -- "$L" list
check "list shows the speech model"     "Qwen3-TTS"        -- "$L" list
# status reads ports, which no sandbox can hide, so this only asks that it runs.
check "status runs"                     "Slot"             -- env -i PATH="$PATH" HOME="$HOME" sh -c "$L status; echo Slot"

# ── llama backend ────────────────────────────────────────────
check "llama start builds a command"    "llama-server --model" -- "$L" start qwen 1 --print-cmd
check "llama start takes --ctx"         "--ctx-size 4096"      -- "$L" start qwen 1 --ctx 4096 --print-cmd
check "llama start rejects --lang"      "for speech (tts)"     -- "$L" start qwen 1 --lang de --print-cmd
check "llama start rejects --output"    "is for comfyui"       -- "$L" start qwen 1 --output /tmp --print-cmd
check "llama start rejects a level the model lacks" "has no thinking mode" -- "$L" start qwen-vl 3 --reasoning low --print-cmd
check "asr start loads its mmproj"      "--mmproj"             -- "$L" start asr 6 --mmproj --print-cmd

# ── halogen backend ──────────────────────────────────────────
check "halogen start runs podman"       "podman run --rm --name llmctl-halogen-1" -- "$L" start flash 1 --print-cmd
check "halogen start sets its context"  "HALOGEN_CTX=262144"                      -- "$L" start flash 1 --print-cmd
check "halogen start turns YaRN on"     "HALOGEN_ROPE_YARN=2"                     -- "$L" start flash 1 --ctx 524288 --print-cmd
check "halogen start refuses --spec"    "not available with the halogen backend"  -- "$L" start flash 1 --spec off --print-cmd
check "halogen start maps --mmproj"     "HALOGEN_VISION_TOWER=1"                  -- "$L" start flash 1 --mmproj --print-cmd

# ── comfyui backend ──────────────────────────────────────────
check "comfyui start names its dirs"    "--base-directory"       -- "$L" start comfy 9 --print-cmd
check "comfyui start takes --output"    "--output-directory /tmp/x" -- "$L" start comfy 9 --output /tmp/x --print-cmd
check "comfyui start refuses --ctx"     "not an LLM"             -- "$L" start comfy 9 --ctx 4096 --print-cmd

# ── tts backend ──────────────────────────────────────────────
check "speech start runs tts_server"    "tts_server.py"          -- "$L" start speech 5 --print-cmd
check "speech start passes the voices"  "--voices"               -- "$L" start speech 5 --print-cmd
check "speech start takes --lang"       "--lang en"              -- "$L" start speech 5 --lang en --print-cmd
check "speech start takes --seed"       "--seed 7"               -- "$L" start speech 5 --seed 7 --print-cmd
check "speech start refuses --parallel" "have no meaning for it" -- "$L" start speech 5 --parallel 2 --print-cmd
check "speech start refuses a language" "--lang takes zh, en"    -- "$L" start speech 5 --lang xx --print-cmd

# ── slots and presets ────────────────────────────────────────
check "a slot past the ports is refused" "past the port layout"  -- "$L" start qwen 999 --print-cmd
check "presets list what they start"     "speech"                -- "$L" presets
check "preset dry-run plans its slots"   "slot 5"                -- "$L" preset voice --dry-run
# The memory budget needs a GTT pool (an AMD GPU); without one it is left out.
if compgen -G '/sys/class/drm/card*/device/mem_info_gtt_total' > /dev/null; then
    check "preset dry-run weighs the memory" "GPU memory"        -- "$L" preset voice --dry-run
else
    check "preset dry-run skips the memory without a GPU" "!GPU memory" -- "$L" preset voice --dry-run
fi
check "preset refuses --output without comfy" "no ComfyUI entry" -- "$L" preset c4f --output /tmp --dry-run
check "an unknown preset says so"        "Unknown preset"        -- "$L" preset nosuchpreset

# ── env ──────────────────────────────────────────────────────
check "env exports a base URL"           "ANTHROPIC_BASE_URL"    -- "$L" env qwen 1 --direct
check "env refuses ComfyUI"              "not an LLM"            -- "$L" env comfy 9
check "env refuses the speech slot"      "not an LLM"            -- "$L" env speech 5

# ── voices ───────────────────────────────────────────────────
printf 'RIFFxxxxWAVE' > "$SANDBOX/sample.wav"
check "voice list is empty at first"     "none yet"              -- "$L" voice list
check "voice add takes a recording"      "added"                 -- "$L" voice add test "$SANDBOX/sample.wav" "a test"
check "voice list shows it"              "test"                  -- "$L" voice list
check "voice add refuses a name"         "lowercase letters"     -- "$L" voice add BAD "$SANDBOX/sample.wav"
check "voice add refuses a format"       "wav, mp3 or flac"      -- "$L" voice add other "$ROOT/llmctl"
check "voice export writes an archive"   "Wrote"                 -- "$L" voice export "$SANDBOX/voices.tar.gz"
check "voice rm removes it"              "removed"               -- "$L" voice rm test
check "voice rm says when there is none" "No voice"              -- "$L" voice rm test
check "voice import brings them back"    "Imported 1 voice"      -- "$L" voice import "$SANDBOX/voices.tar.gz"
check "voice import keeps what is there" "kept 1"                -- "$L" voice import "$SANDBOX/voices.tar.gz"

# Import decides per voice: the .txt follows its audio, whatever the format here.
mkdir -p "$SANDBOX/arc"
printf 'RIFFxxxxWAVE' > "$SANDBOX/arc/test.wav"; echo "new text" > "$SANDBOX/arc/test.txt"
echo "lonely" > "$SANDBOX/arc/solo.txt"; printf 'x' > "$SANDBOX/arc/Bad Name.wav"
tar -czf "$SANDBOX/two.tar.gz" -C "$SANDBOX/arc" .
V="$LLMCTL_DATA_DIR/tts/voices"
check "voice import keeps a kept voice's text" "a test"          -- sh -c "'$L' voice import '$SANDBOX/two.tar.gz' >/dev/null; cat '$V/test.txt'"
check "voice import skips a bad name"    "Skipped 'Bad Name'"    -- "$L" voice import "$SANDBOX/two.tar.gz"
check "voice import --force takes the text too" "new text"       -- sh -c "'$L' voice import '$SANDBOX/two.tar.gz' --force >/dev/null; cat '$V/test.txt'"
check "voice import fills a missing text" "lonely"               -- cat "$V/solo.txt"
check "voice import replaces another format" "!test.mp3"         -- sh -c "mv '$V/test.wav' '$V/test.mp3'; '$L' voice import '$SANDBOX/two.tar.gz' --force >/dev/null; ls '$V'"

# ── things that must not happen ──────────────────────────────
check "no command leaks the sandbox"     "!$HOME/.local/share/llmctl/tts" -- "$L" start speech 5 --print-cmd
check "cache-stats without logs says so" "No server logs yet"  -- "$L" cache-stats
check "logs without logs says so"       "No logs yet"           -- "$L" logs
mkdir -p "$LLMCTL_STATE_DIR/logs"; printf 'one\ntwo\nthree\n' > "$LLMCTL_STATE_DIR/logs/server-4.log"
check "logs lists what there is"         "server-4.log"          -- "$L" logs
check "logs -n shows the last lines"     "!one"                  -- "$L" logs 4 -n 2
check "logs names a missing proxy log"   "No proxy log for slot 4" -- "$L" logs 4 --proxy
check "logs refuses a bad -n"            "takes a number"        -- "$L" logs 4 -n x
check "update refuses a non-comfyui"     "nothing to update"     -- "$L" update qwen

echo ""
printf '%d passed, %d failed%s\n' "$pass" "$fail" "$( (( skipped )) && printf ', %d skipped' "$skipped")"
(( fail == 0 ))
