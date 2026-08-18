# AI — LLM Server Manager

Script-based tool to manage local LLM inference servers and proxies for Claude Code.

## Setup

```bash
# Install Python dependencies
pip install flask requests
pip install waitress   # optional, recommended for production proxy

# Download a model
./run.sh download <model-name>
./run.sh download all   # all models
```

## Usage

```bash
./run.sh list                          # Show available models
./run.sh start <name> [slot] [--proxy] # Start server in background (slot 1 or 2, default 1)
./run.sh stop [slot]                   # Stop slot, or all if omitted
./run.sh status                        # Show running state (all slots)
./run.sh clear-kv [slot]               # Drop the KV cache without restarting
./run.sh probe-reasoning [model]       # What each model's chat template supports
./run.sh bench [--full] <model|all>    # Run benchmark (default: default ROCm + Vulkan)
./run.sh bench --full all              # Full test: all 8 ROCm combos + Vulkan
source ./run.sh env <name> [slot]      # Set Claude Code env vars
source ./run.sh clear                  # Clear env vars
./run.sh download <model>              # Download model(s)
```

### The proxy is opt-in

`start` runs only llama-server. Pass `--proxy` to also start `proxy.py`, which
rewrites time/date stamps in prompts so the prompt cache stays warm — worth it
for clients that stamp every request, pointless for those that don't.

`env` follows suit: it points at the proxy when one is listening for that slot,
at llama-server otherwise. `--proxy` / `--direct` force the choice, e.g. when
setting the env before starting the slot.

```bash
./run.sh start qwen 1 --proxy          # server :8001 + proxy :8081
source ./run.sh env qwen 1             # → :8081 (proxy detected)
source ./run.sh env qwen 1 --direct    # → :8001 (bypass the proxy)
```

Logs are written to `logs/server-<slot>.log` and `logs/proxy-<slot>.log`.

### Running two models in parallel

```bash
./run.sh start qwen 1 --proxy  # slot 1 → server :8001, proxy :8081
./run.sh start gemma 2         # slot 2 → server :8002 (no proxy)

# In terminal A:
source ./run.sh env qwen 1
claude

# In terminal B:
source ./run.sh env gemma 2
claude

./run.sh stop 1                # stop only slot 1
./run.sh stop                  # stop everything
```

### Reasoning

One switch for every model:

```bash
./run.sh start <name> [slot] --reasoning off        # no thinking
./run.sh start <name> [slot] --reasoning on         # thinking, template default depth
./run.sh start <name> [slot] --reasoning high       # low | medium | high | max
./run.sh start <name> [slot] --reasoning 2048       # thinking, capped at N tokens (-1 = uncapped)
```

`--no-reasoning` and `--reasoning-budget N` still work as aliases.

What reaches llama-server depends on the model, because the mechanism is the chat
template rather than the server: `--reasoning` sets the template kwarg
`enable_thinking`, `--reasoning-effort` sets the template variables
`reasoning_effort` / `reasoning_strength`. So each entry in `models.conf` declares
what its template actually reads:

| `reasoning` | Meaning | `--reasoning` accepts |
| --- | --- | --- |
| `none` | no thinking in the template at all | `off` (as a no-op) |
| `toggle` | template reads `enable_thinking` | `off`, `on`, budget |
| `effort` | plus `reasoning_effort` / `reasoning_strength` | `off`, `on`, level, budget |
| `locked-off` | template hard-codes thinking closed | `off` (as a no-op) |
| `unknown` | not verified yet — flags pass through, `start` says so | everything |

Level names differ per model, so `reasoning_levels` maps the unified scale onto
what the template accepts — templates raise on names they do not know (stock
Qwen3.8 knows only `low`, `medium`, `xhigh`). `muse` therefore maps `max` → `xhigh`
while keeping `high` → `high`; `qwen3.8` has no `high` at all and folds both onto
`xhigh`. Asking for something a model cannot do fails immediately, before any
port opens or any weight is read:

```
$ ./run.sh start qwen 1 --reasoning low
Error: model 'qwen' supports on/off and a token budget, but no reasoning levels.
       Use --reasoning on, off, or a token budget (e.g. --reasoning 2048).
```

**A client can override all of this per request.** `--reasoning` is a default, not
a lock: `chat_template_kwargs: {"enable_thinking": true}` in a request beats
`--reasoning off --reasoning-budget 0` and reopens `<think>`. A bare
`reasoning_effort` field does not, nor does an Anthropic `thinking` block — only
`chat_template_kwargs`, because it goes straight into the template. llama.cpp's
own web UI sends it whenever its **Reasoning** dropdown is on a level rather than
on `Default`, which is the usual reason a slot started with `--reasoning off`
still thinks. `Off` there works too; only `Default` leaves the decision to the
server. Check what the server really builds with `/apply-template`, which renders
the prompt without generating:

```
$ curl -s localhost:8001/apply-template -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}]}' | jq -r .prompt | tail -6
<|im_start|>assistant
<think>

</think>

```

`probe-reasoning` reads the chat template every downloaded model actually runs
with and shows what it really supports next to what `models.conf` claims — the
same signal llama.cpp probes at load time. Where `extra_args` override the baked
template with `--chat-template-file`, it reads that file instead and names it:

```
$ ./run.sh probe-reasoning
  MODEL       CONFIGURED  TEMPLATE    READS
  qwen        toggle      toggle      enable_thinking
  qwen3.8     effort      effort      enable_thinking, reasoning_effort  [qwen3.8-unc.jinja]
  muse        effort      effort      reasoning_strength
  minimax     unknown     -           not downloaded
```

It needs the `gguf` python module — either installed, or a llama.cpp checkout
(`LLAMA_SRC`, default `~/src/llama.cpp`).

### Clearing the KV cache

llama-server keeps each conversation in a server slot; a new prompt reuses the
longest common prefix that is still there. To start from nothing without
restarting the model:

```bash
./run.sh clear-kv          # every running slot
./run.sh clear-kv 2        # only slot 2
```

The underlying endpoint is llama-server's own, and the proxy forwards it
unchanged:

```bash
curl -X POST 'http://localhost:8001/slots/0?action=erase'   # server slot 0
curl -X POST 'http://localhost:8081/slots/0?action=erase'   # same, via the proxy
```

Slot ids run from 0 to `--parallel N` minus one — `clear-kv` reads the count from
`/props` and walks all of them. The action needs `--slot-save-path`, which `start`
always passes (`.slots/`, gitignored).

Two limits worth knowing:

- **It never interrupts a running generation.** The server defers the erase until
  the slot falls idle, so the call can block for as long as the generation takes.
- **It does not touch the host-RAM prompt cache** (`--cache-ram`, 8192 MiB by
  default), and no endpoint does. Measured on a cleared slot: of a 53-token
  prompt sent again afterwards, 52 tokens came straight back from host RAM. For a
  hard reset, start the slot with `--cache-ram 0`.

## Exposing models on the internet (`--public`)

`--host` is and stays plain LAN exposure without authentication. `--public` is
the separate, always-authenticated path to the internet, and the two do not
interfere:

```
Client ──443/LE──► VPS Apache ──mTLS──► router ──► stunnel :844N ──► 127.0.0.1:800N
                   /sN/direct/                                       llama-server
                   /sN/cached/ ─────────────────► stunnel :845N ──► 127.0.0.1:808N
                                                                     proxy.py
```

Only the stunnel ports are forwarded at the router; `:800N` and `:808N` never
leave the machine. stunnel requires a client certificate from a private CA, so a
scanner hitting the port fails at the TLS handshake — before reaching any HTTP.

**Setup**

```bash
mkdir -p ~/.config/llm && (umask 077 && openssl rand -hex 32 > ~/.config/llm/tokens)
./run.sh gen-certs llm-home.example.org      # SAN must be the name the VPS connects to
# copy ca.pem + vps-client-combined.pem to the VPS (the command prints the scp line)
# put deploy/vps-llm-vhost.conf on the VPS and adjust the two Define lines
# forward 8441 (and 8451 with --proxy) at the router to this machine

./run.sh start qwen 1 --proxy --public
```

`--public` implies authentication — there is no way to open the port without it.
It also passes `--no-webui --no-slots` to llama-server (`GET /slots` is enabled by
default and shows other clients' prompts) and caps generation at 8192 tokens
(`--max-predict N` to change, `-1` for no limit). The proxy switches to a path
allowlist, checks tokens itself, and caps concurrent requests.

The one `/slots` request that stays open from outside is
`POST /slots/{id}?action=erase` (see above) — clearing your own KV cache from
away is the point. Its siblings `save` and `restore` write files here, so the
proxy rejects them and the vhost blocks them on the direct path too.

**Tokens** live in `~/.config/llm/tokens`, one per line, `#` comments allowed,
mode 0600 (enforced). Revoking one means deleting the line and restarting the
slot. `source ./run.sh env` picks up the first token automatically, so the local
workflow is unchanged.

## Models

Order and names follow `models.conf`; `./run.sh list` prints the same set. All of
them are served by the Vulkan build — the ROCm build is opt-in per model
(`LLAMA_ROCM_BIN`) and is what `bench` compares against.

- **qwen-moe** — Qwen3.6-35B-A3B MoE uncensored, Q8_K_P, 64K ctx, mmproj available
- **qwen** — Qwen3.6-27B uncensored, Q8_K_P, 64K ctx, mmproj available
- **qwen3.8** — Qwen3.8-27B uncensored v4, Q6_K, 64K ctx, mmproj available; runs `templates/qwen3.8-unc.jinja` instead of the baked template, which restores thinking
- **qwen-vl** — Qwen3-VL-8B vision-language uncensored, Q8_0, 8K ctx, mmproj available
- **muse** — Muse-Glimmer-30B abliterated aggressive (Meta base, agentic), Q8_0, 128K ctx, mmproj available
- **gemma** — Gemma-4-31B-it uncensored, Q8_0, 128K ctx
- **gemma-moe** — Gemma-4-26B-A4B-it MoE uncensored, Q8_0, 128K ctx
- **minimax** — MiniMax-M2.7, UD-IQ3_S, 64K ctx
- **llama3.3** — Llama-3.3-70B-Instruct abliterated, Q6_K, 32K ctx
- **r1** — DeepSeek-R1-Distill-Llama-70B Uncensored v2 Unbiased Reasoner, i1-Q5_K_M, 128K ctx
- **mistral** — Mistral-Medium-3.5-128B, UD-Q5_K_XL, 32K ctx
- **diamond** — L3.3-70B Magnum Diamond, i1-Q5_K_M, 32K ctx

Multimodal projectors are only loaded on an explicit `--mmproj`.

## Architecture

- `proxy.py` — optional Flask proxy (`start --proxy`) that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`
- `models.conf` — Model definitions (paths, binaries, ROCm env vars)
- `run.sh` — Main entry point for all commands
- `deploy/vps-llm-vhost.conf` — Apache vhost for the VPS in front of `--public`
