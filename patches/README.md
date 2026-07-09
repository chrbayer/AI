# llama.cpp patches

## ggml-vulkan-queue-priority.patch

Adds an env-gated **per-process GPU scheduling priority** to the Vulkan backend
via `VK_EXT_global_priority`. Upstream ggml-vulkan hardcodes the queue priority
and offers no knob, so two `llama-server` instances sharing one GPU compete
equally. With this patch a background server can be told to yield GPU time to a
foreground one.

### What it does

At device creation, if `GGML_VK_QUEUE_PRIORITY` is set and the driver exposes
`VK_EXT_global_priority`, the compute/transfer queues are created with the
requested global priority:

| Value      | Meaning                                             |
|------------|-----------------------------------------------------|
| `low`      | Yields GPU time to higher-priority contexts         |
| `medium`   | Driver default (same as unset)                      |
| `high`     | Preempts lower-priority contexts (needs CAP_SYS_NICE on RADV) |
| `realtime` | Highest (needs CAP_SYS_NICE)                         |

Unset = unchanged upstream behaviour. If the extension is unavailable the value
is ignored (logged, no failure). A `ggml_vulkan: GGML_VK_QUEUE_PRIORITY=... applied`
line appears in the server log at startup.

### Apply

```bash
./patches/apply.sh [~/src/llama.cpp]     # default: ~/src/llama.cpp
cmake --build ~/src/llama.cpp/build --target llama-server -j$(nproc)
```

The script is idempotent and tolerates upstream line-number drift (git apply
fuzz, then `patch --fuzz=3`). If both fail, the hunk is a single self-contained
block inserted right after `std::vector<const char *> device_extensions;` — paste
it by hand from the `.patch` file.

### Use via run.sh

```bash
./run.sh start <model> 1 --gpu-priority high     # foreground
./run.sh start <model> 2 --gpu-priority low      # background, yields GPU
```

`run.sh` sets `GGML_VK_QUEUE_PRIORITY` from `--gpu-priority`. Without the flag,
nothing is set and behaviour is identical to stock llama.cpp.

### Note on realtime/high

On RADV, `high` and `realtime` require `CAP_SYS_NICE` (or root); without it queue
creation for that priority fails. `low` needs no special rights and is the useful
lever for "make this server get out of the way".

### Refreshing after a llama.cpp update

The patch targets a stable anchor line, so it usually re-applies across versions.
If ggml-vulkan restructures device creation, regenerate the patch:

```bash
cd ~/src/llama.cpp
# re-add the block, then:
git diff -- ggml/src/ggml-vulkan/ggml-vulkan.cpp > ~/AI/patches/ggml-vulkan-queue-priority.patch
```
