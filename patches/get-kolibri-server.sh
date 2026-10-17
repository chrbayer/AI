#!/bin/bash
# Fetch the llama.cpp build llmctl's "kolibri" model runs on until a stable
# llama.cpp release knows Kolibri-1 (Aleph Alpha, "kolibri1" architecture).
#
# Usage: get-kolibri-server.sh [target dir]
#   default target: ~/.local/share/llmctl/llama.cpp-kolibri (what models.conf names)
#
# Kolibri-1 came into llama.cpp master with PR #30673 (2026-10-17); v0.7.0 does
# not have it. This takes ggml-org's prebuilt Vulkan build of master just after
# the merge, the same package `llmctl download llama` takes for a release, into
# <target>/<build> and points <target>/current at it. `download llama` keeps its
# own builds apart and does not touch this one. Once a stable release has the
# merge, `outdated` says so: the entry moves to LLAMA_DEFAULT_BIN and this goes.
set -euo pipefail

BUILD=b11710    # 2026-10-17, master one commit past the merge (62318cd)
ASSET="llama-$BUILD-bin-ubuntu-vulkan-x64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/$ASSET"

DATA_DIR="${LLMCTL_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/llmctl}"
DST="${1:-$DATA_DIR/llama.cpp-kolibri}"

if [[ -x "$DST/$BUILD/llama-server" ]]; then
    ln -sfn "$BUILD" "$DST/current"
    echo "$DST/current -> $BUILD (there already)"
    exit 0
fi
if [[ -e "$DST" && ! -L "$DST/current" && ! -d "$DST/$BUILD" && -n "$(ls -A "$DST" 2>/dev/null)" ]]; then
    echo "$DST holds something else (the old patched build?) — remove it first:"
    echo "  git -C ~/src/llama.cpp worktree remove --force $DST   (or rm -rf it)"
    exit 1
fi

mkdir -p "$DST"
tmp="$DST/.$BUILD.part"; rm -rf "$tmp"; mkdir -p "$tmp"
echo "Downloading $ASSET ..."
curl -fsSL -o "$tmp/$ASSET" "$URL"
tar xzf "$tmp/$ASSET" -C "$tmp"
top=$(dirname "$(find "$tmp" -name llama-server -type f | head -1)")
"$top/llama-server" --version 2>&1 | grep -q "build ${BUILD#b}" || { echo "unexpected build in $ASSET"; rm -rf "$tmp"; exit 1; }
mv "$top" "$DST/$BUILD"; rm -rf "$tmp"
ln -sfn "$BUILD" "$DST/current"
echo "Installed: $DST/current/llama-server ($BUILD)"
