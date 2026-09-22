#!/bin/bash
# Build llama-server with a /tts endpoint — llama.cpp PR #26603, not merged yet —
# for llmctl's speech slot: the model stays loaded and the audio streams.
#
# Usage: ./patches/build-tts-server.sh [llama.cpp checkout] [target dir]
#        (defaults: ~/src/llama.cpp, ~/src/llama.cpp-tts)
#
# A git worktree of the checkout's current commit, with the PR (pinned below)
# merged in and llama.cpp-pr26603-mtmd-init-opt.patch on top — the PR predates
# an extra argument to mtmd_helper_bitmap_init_from_buf. Built like the Vulkan
# build of the checkout; nothing is installed: models.conf points at the build.
# Once the PR is merged, the ordinary llama-server does this and this goes.
set -euo pipefail

PR=26603
PR_COMMIT=435d4116eda2abbcdc3d63d85204470070f32705
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HOME/src/llama.cpp}"
DST="${2:-$HOME/src/llama.cpp-tts}"

[[ -d "$SRC/.git" ]] || { echo "Not a llama.cpp checkout: $SRC"; exit 1; }
if [[ ! -d "$DST" ]]; then
    git -C "$SRC" fetch -q origin "pull/$PR/head"
    git -C "$SRC" cat-file -e "$PR_COMMIT^{commit}" || { echo "PR commit $PR_COMMIT not found"; exit 1; }
    git -C "$SRC" worktree add -q --detach "$DST" HEAD
    git -C "$DST" -c user.name=llmctl -c user.email=llmctl@localhost merge -q --no-edit "$PR_COMMIT"
    git -C "$DST" apply "$PATCH_DIR/llama.cpp-pr26603-mtmd-init-opt.patch"
    echo "$DST: $(git -C "$SRC" log -1 --format='%h %cs') + PR #$PR (${PR_COMMIT:0:9})"
else
    echo "$DST exists — building what is there (remove it to start over)"
fi
cmake -S "$DST" -B "$DST/build" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build "$DST/build" -j "$(nproc)" --target llama-server llama-tts
echo "Built: $DST/build/bin/llama-server"
