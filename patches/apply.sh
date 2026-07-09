#!/bin/bash
# Apply the ggml-vulkan queue-priority patch to a llama.cpp checkout and rebuild.
#
# Usage: ./patches/apply.sh [path-to-llama.cpp]   (default: ~/src/llama.cpp)
#
# Idempotent: skips if already applied. Tolerates line-number drift across
# llama.cpp versions via `git apply` fuzz, then a `patch` fallback.
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/ggml-vulkan-queue-priority.patch"
LLAMA="${1:-$HOME/src/llama.cpp}"
TARGET="ggml/src/ggml-vulkan/ggml-vulkan.cpp"

[[ -f "$PATCH" ]] || { echo "Patch not found: $PATCH"; exit 1; }
[[ -f "$LLAMA/$TARGET" ]] || { echo "Not a llama.cpp checkout: $LLAMA"; exit 1; }

cd "$LLAMA"

if grep -q "GGML_VK_QUEUE_PRIORITY" "$TARGET"; then
    echo "Already applied in $LLAMA — nothing to do."
    exit 0
fi

echo "Applying patch to $LLAMA ..."
if git apply --recount --whitespace=nowarn "$PATCH" 2>/dev/null; then
    echo "Applied cleanly (git apply)."
elif patch -p1 --forward --fuzz=3 < "$PATCH"; then
    echo "Applied with fuzz (patch). Verify the hunk landed correctly."
else
    echo "Patch did not apply automatically — the surrounding code changed upstream."
    echo "Apply it by hand: insert the block right after"
    echo "    std::vector<const char *> device_extensions;"
    echo "in $TARGET (see the patch file)."
    exit 1
fi

echo
echo "Now rebuild the Vulkan backend, e.g.:"
echo "    cmake --build $LLAMA/build --target llama-server -j\$(nproc)"
