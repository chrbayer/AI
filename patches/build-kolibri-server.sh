#!/bin/bash
# Build llama-server with Kolibri-1 support (Aleph Alpha, "kolibri1" architecture),
# which stock llama.cpp does not have yet (feature request #29922), for llmctl's
# "kolibri" model.
#
# Usage: build-kolibri-server.sh [target dir]
#   default target: ~/.local/share/llmctl/llama.cpp-kolibri (what models.conf names)
#
# As build-tts-server.sh: it uses ~/src/llama.cpp (or $LLAMA_SRC) when that is a
# checkout, else clones llama.cpp into ~/.local/share/llmctl/llama.cpp, makes a git
# worktree at the commit the patch was made for and applies it:
# llama.cpp-kolibri1.patch, MIT, by Seraphiel102 / Hob-forge
# (https://huggingface.co/Hob-forge/Kolibri-1-GGUF) — the architecture, its
# router (selects on logits + bias, weights by the unbiased sigmoid), the
# tokenizer, the converter. Vulkan; llama-server and llama-bench (llmctl bench). Nothing is installed, and
# a checkout it uses is not changed.
#
# To rebuild: git worktree remove --force <target>, then run again. Once
# llama.cpp supports Kolibri, the ordinary llama-server runs it and this goes.
set -euo pipefail

BASE_COMMIT=836d57176dc699a726c55418e4f96b8ca628e1bf    # 2026-10-03, what the patch was made on
REPO=https://github.com/ggml-org/llama.cpp.git

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${LLMCTL_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/llmctl}"
DST="${1:-$DATA_DIR/llama.cpp-kolibri}"

missing=()
for c in git cmake c++ glslc; do command -v "$c" > /dev/null || missing+=("$c"); done
if (( ${#missing[@]} )); then
    echo "Missing: ${missing[*]}"
    echo "  Fedora: sudo dnf install git cmake gcc-c++ glslc vulkan-headers vulkan-loader-devel"
    echo "  Debian: sudo apt install git cmake g++ glslc libvulkan-dev"
    exit 1
fi

if [[ -e "$DST" ]]; then
    echo "$DST exists — building what is there (remove it to start over)."
else
    SRC="${LLAMA_SRC:-$HOME/src/llama.cpp}"
    if [[ ! -d "$SRC/.git" ]]; then
        SRC="$DATA_DIR/llama.cpp"
        if [[ ! -d "$SRC/.git" ]]; then
            echo "Cloning llama.cpp into $SRC (without file history, ~200 MB) ..."
            git clone -q --filter=blob:none "$REPO" "$SRC"
        fi
    fi
    echo "llama.cpp from $SRC"
    git -C "$SRC" cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch -q "$REPO" "$BASE_COMMIT"
    mkdir -p "$(dirname "$DST")"
    git -C "$SRC" worktree add -q --detach "$DST" "$BASE_COMMIT"
    git -C "$DST" -c user.name=llmctl -c user.email=llmctl@localhost am -q "$PATCH_DIR/llama.cpp-kolibri1.patch"
    echo "$DST: llama.cpp ${BASE_COMMIT:0:9} + the kolibri1 patch"
fi

cmake -S "$DST" -B "$DST/build" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build "$DST/build" -j "$(nproc)" --target llama-server llama-bench
echo "Built: $DST/build/bin/llama-server"
