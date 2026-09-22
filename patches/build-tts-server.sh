#!/bin/bash
# Build llama-server with a /tts endpoint — llama.cpp PR #26603, not merged yet —
# for llmctl's speech slot: the model stays loaded and the audio streams.
#
# Usage: build-tts-server.sh [target dir]
#   default target: ~/.local/share/llmctl/llama.cpp-tts (what models.conf names)
#
# No llama.cpp checkout needed. It uses ~/src/llama.cpp (or $LLAMA_SRC) when
# that is one, and otherwise clones llama.cpp into ~/.local/share/llmctl/llama.cpp.
# From there it makes a git worktree at a pinned, tested llama.cpp commit, merges
# the pinned PR commit, applies llama.cpp-pr26603-mtmd-init-opt.patch (the PR
# predates an extra argument to mtmd_helper_bitmap_init_from_buf) and builds
# llama-server and llama-tts with Vulkan. Nothing is installed, and a checkout
# it uses is not changed: the worktree has files of its own and shares only the
# git objects.
#
#   LLAMA_BASE=HEAD   build on the checkout's current commit instead of the pinned
#                     one — newer llama.cpp may no longer merge with the PR
#
# To rebuild: delete the target (git worktree remove --force <target>) and run
# again. Once the PR is merged, the ordinary llama-server has /tts and this goes.
set -euo pipefail

PR=26603
PR_COMMIT=435d4116eda2abbcdc3d63d85204470070f32705
BASE_COMMIT=b29c606e28a01b1bc8c1351026a0fa6e616bf6c4    # 2026-09-14, llama.cpp 0.4.1: tested
REPO=https://github.com/ggml-org/llama.cpp.git

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${LLMCTL_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/llmctl}"
DST="${1:-$DATA_DIR/llama.cpp-tts}"

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
    base="$BASE_COMMIT"
    [[ "${LLAMA_BASE:-}" == HEAD ]] && base="$(git -C "$SRC" rev-parse HEAD)"
    git -C "$SRC" cat-file -e "$base^{commit}" 2>/dev/null || git -C "$SRC" fetch -q "$REPO" "$base"
    git -C "$SRC" fetch -q "$REPO" "pull/$PR/head"
    git -C "$SRC" cat-file -e "$PR_COMMIT^{commit}" || { echo "PR commit $PR_COMMIT not found"; exit 1; }
    mkdir -p "$(dirname "$DST")"
    git -C "$SRC" worktree add -q --detach "$DST" "$base"
    git -C "$DST" -c user.name=llmctl -c user.email=llmctl@localhost merge -q --no-edit "$PR_COMMIT"
    git -C "$DST" apply "$PATCH_DIR/llama.cpp-pr26603-mtmd-init-opt.patch"
    echo "$DST: llama.cpp ${base:0:9} + PR #$PR (${PR_COMMIT:0:9}) + fix"
fi

cmake -S "$DST" -B "$DST/build" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build "$DST/build" -j "$(nproc)" --target llama-server llama-tts
echo "Built: $DST/build/bin/llama-server"
