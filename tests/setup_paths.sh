#!/bin/bash
# The setup paths, from nothing (issue #5): what `download` and `update` do on
# a machine without ComfyUI, without the voice-design venv and without the
# speech server — the parts the smoke tests cannot reach.
#
#   tests/setup_paths.sh [comfy] [speech]      (default: both)
#
# Only what setup builds goes to a scratch directory (SETUP_TEST_DIR, default
# /tmp/llmctl-setup-test): the data directory with the checkout, the venvs and
# the llama.cpp build, and the state directory. The models, the config and
# pip's cache are the real ones, so nothing big is downloaded again. The two
# parts need ~17 GB each and run one after the other, each deleted before the
# next, so 24 GB of scratch are enough.
#
# Nothing is written into the real models directory: each part first asks
# `download --check` and stops unless every model is already there.
# Needs a machine with no slot running, and takes it over while it runs.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="$ROOT/llmctl"
T="${SETUP_TEST_DIR:-/tmp/llmctl-setup-test}"
PARTS=("$@"); (( ${#PARTS[@]} )) || PARTS=(comfy speech)

REAL_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/llmctl"
export LLMCTL_MODELS_DIR="${LLMCTL_MODELS_DIR:-$REAL_DATA/models}"
export LLMCTL_DATA_DIR="$T/data"
export LLMCTL_STATE_DIR="$T/state"
# build-tts-server.sh then clones llama.cpp into the scratch data directory
# instead of adding a worktree to a checkout of yours.
export LLAMA_SRC="$T/no-llama-checkout"

pass=0 fail=0
ok()   { printf '  ok    %s\n' "$1"; pass=$(( pass + 1 )); }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$(( fail + 1 )); }
step() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$1" "$(date +%T)"; }
room() { printf '  scratch: %s used, %s free\n' "$(du -sh "$T" 2>/dev/null | cut -f1)" "$(df -h --output=avail "$T" | tail -1 | tr -d ' ')"; }

# Up and answering within $2 seconds: URL $1 returns 200.
wait_http() {
    local i
    for (( i=0; i<$2; i+=5 )); do
        curl -sf -o /dev/null "$1" && return 0
        sleep 5
    done
    return 1
}

if ss -tln | grep -E ':(80(0[1-9])|808[1-9]) ' > /dev/null; then
    echo "A slot is running (ports 8001-8009 / 8081-8089) — stop it first: llmctl stop"
    exit 1
fi
mkdir -p "$T"
echo "Scratch: $T   models (real, read only): $LLMCTL_MODELS_DIR"
room

for part in "${PARTS[@]}"; do
    case "$part" in
    comfy)
        step "ComfyUI: nothing to download but the setup itself"
        out=$("$L" download comfy --check 2>&1)
        if grep -q ' 0 to download' <<< "$out"; then ok "every workflow model is already here"
        else echo "$out" | grep -E 'missing|models:'; bad "models missing — not writing into $LLMCTL_MODELS_DIR"; continue; fi

        step "ComfyUI: download from nothing"
        if "$L" download comfy 2>&1 | tail -25; then ok "download comfy"; else bad "download comfy"; fi
        [[ -x "$T/data/comfyui/app/.venv/bin/python" ]] && ok "checkout and venv in place" || bad "no venv"
        [[ -d "$T/data/comfyui/custom_nodes/ComfyUI-Qwen-TTS/.git" ]] && ok "custom nodes cloned" || bad "custom nodes missing"
        room

        step "ComfyUI: update"
        if "$L" update comfy 2>&1 | tail -15; then ok "update comfy"; else bad "update comfy"; fi

        step "ComfyUI: list"
        out=$("$L" list 2>&1 | sed -n '/workflows (/,/^$/p')
        echo "$out"
        if grep -q 'missing:' <<< "$out"; then bad "a workflow lacks models"; else ok "every workflow ready"; fi

        step "ComfyUI: start, and an image through the API"
        "$L" start comfy 9 --proxy 2>&1 | grep -E 'API:|Error'
        if wait_http http://127.0.0.1:8009/system_stats 180; then ok "ComfyUI answers"; else bad "ComfyUI does not answer"; fi
        if grep -qiE 'Traceback|IMPORT FAILED' "$T/state/logs/server-9.log"; then
            grep -iE -A3 'Traceback|IMPORT FAILED' "$T/state/logs/server-9.log" | head -12; bad "errors while loading"
        else ok "custom nodes load without a traceback"; fi
        size=$(curl -s -m 900 http://127.0.0.1:8089/v1/images/generations -H 'Content-Type: application/json' \
            -d '{"model":"flux2-klein-9b","prompt":"a red lighthouse at dusk","size":"512x512","seed":1}' |
            python3 -c 'import sys, json, base64; d = base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]); print(int.from_bytes(d[16:20], "big"), "x", int.from_bytes(d[20:24], "big"))' 2>/dev/null)
        [[ "$size" == "512 x 512" ]] && ok "klein renders 512x512 through the API" || bad "no image from the API (${size:-nothing})"
        "$L" stop 9 > /dev/null 2>&1
        room
        rm -rf "$T/data/comfyui"
        ;;
    speech)
        step "Speech: nothing to download but the setup itself"
        for m in speech asr; do
            out=$("$L" download "$m" --check 2>&1)
            # The venv is what this test builds; any model file missing stops it.
            if grep -qE 'file\(s\) to download' <<< "$out"; then
                echo "$out"; bad "$m: model files missing — not writing into $LLMCTL_MODELS_DIR"; continue 2
            fi
            ok "$m: models already here"
        done

        step "Speech: download from nothing"
        if "$L" download speech 2>&1 | tail -12; then ok "download speech"; else bad "download speech"; fi
        [[ -x "$T/data/tts/venv/bin/python" ]] && ok "voice-design venv in place" || bad "no voice-design venv"
        ls "$T/data/tts/voices/"*.wav > /dev/null 2>&1 && ok "starter voices copied" || bad "no starter voices"
        room

        step "Speech: build the streaming server (llama.cpp PR #26603)"
        if "$ROOT/patches/build-tts-server.sh" 2>&1 | tail -4; then ok "build-tts-server.sh"; else bad "build-tts-server.sh"; fi
        [[ -x "$T/data/llama.cpp-tts/build/bin/llama-server" ]] && ok "llama-server with /tts built" || bad "no llama-server built"
        room

        step "Speech: design a voice with the new venv"
        if "$L" voice design testvoice "Eine ruhige Frauenstimme mittleren Alters, klar und deutlich." 2>&1 | tail -2; then
            ok "voice design"; else bad "voice design"; fi

        step "Speech: speak, and hear it back"
        "$L" start speech 5 2>&1 | grep -E 'Error' ; "$L" start asr 6 --mmproj --proxy 2>&1 | grep -E 'Error'
        wait_http http://127.0.0.1:8005/health 300 && ok "speech answers" || bad "speech does not answer"
        wait_http http://127.0.0.1:8006/health 300 && ok "asr answers" || bad "asr does not answer"
        text="Heute ist ein schöner Tag für einen Spaziergang am Meer."
        curl -s -m 300 http://127.0.0.1:8005/v1/audio/speech -H 'Content-Type: application/json' \
            -d "{\"input\": \"$text\", \"voice\": \"frau\"}" -o "$T/said.wav"
        [[ "$(head -c 4 "$T/said.wav" 2>/dev/null)" == RIFF ]] && ok "speech: a WAV" || bad "speech: no WAV"
        heard=$(curl -s -m 300 http://127.0.0.1:8086/v1/audio/transcriptions -F file=@"$T/said.wav" -F model=asr |
                python3 -c 'import sys, json; print(json.load(sys.stdin).get("text", ""))' 2>/dev/null)
        echo "  said:  $text"; echo "  heard: $heard"
        [[ "${heard,,}" == *"spaziergang"* && "${heard,,}" == *"meer"* ]] && ok "asr hears what speech said" || bad "asr heard something else"
        "$L" stop 5 > /dev/null 2>&1; "$L" stop 6 > /dev/null 2>&1
        room
        rm -rf "$T/data/tts" "$T/data/llama.cpp-tts" "$T/data/llama.cpp"
        ;;
    *) echo "Unknown part: $part (comfy, speech)"; exit 1 ;;
    esac
done

echo ""
printf '%d passed, %d failed — scratch left in %s (rm -rf it when done)\n' "$pass" "$fail" "$T"
(( fail == 0 ))
