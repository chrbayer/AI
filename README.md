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
It also passes `--no-webui --no-slots` to llama-server (`/slots` is enabled by
default and shows other clients' prompts) and caps generation at 8192 tokens
(`--max-predict N` to change, `-1` for no limit). The proxy switches to a path
allowlist, checks tokens itself, and caps concurrent requests.

**Tokens** live in `~/.config/llm/tokens`, one per line, `#` comments allowed,
mode 0600 (enforced). Revoking one means deleting the line and restarting the
slot. `source ./run.sh env` picks up the first token automatically, so the local
workflow is unchanged.

## Models

- **qwen-moe** — Qwen3.6-35B-A3B MoE uncensored (ROCm)
- **qwen-vl** — Qwen3-VL-8B vision-language uncensored (ROCm)
- **qwen** — Qwen3.6-27B uncensored (ROCm)
- **gemma** — Gemma-4-31B-it (ROCm)
- **gemma-unc** — Gemma-4-31B-it uncensored (ROCm)
- **gemma-moe** — Gemma-4-26B-A4B-it MoE (Vulkan)
- **gemma-moe-unc** — Gemma-4-26B-A4B-it uncensored MoE (Vulkan)
- **minimax** — MiniMax-M2.7 (Vulkan)
- **r1-unc** — DeepSeek-R1-Distill-Llama-70B-Uncensored-v2 Q8_0 (ROCm)
- **r1-reason** — DeepSeek-R1-Distill-Llama-70B Unbiased Reasoner i1-Q6K (ROCm)
- **mistral** — Mistral-Medium-3.5-128B (Vulkan)

## Architecture

- `proxy.py` — optional Flask proxy (`start --proxy`) that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`
- `models.conf` — Model definitions (paths, binaries, ROCm env vars)
- `run.sh` — Main entry point for all commands
- `deploy/vps-llm-vhost.conf` — Apache vhost for the VPS in front of `--public`
