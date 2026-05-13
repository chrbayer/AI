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
./run.sh start <name> [slot]           # Start server + proxy in background (slot 1 or 2, default 1)
./run.sh stop [slot]                   # Stop slot, or all if omitted
./run.sh status                        # Show running state (all slots)
./run.sh bench [--full] <model|all>    # Run benchmark (default: default ROCm + Vulkan)
./run.sh bench --full all              # Full test: all 8 ROCm combos + Vulkan
source ./run.sh env <name> [slot]      # Set Claude Code env vars
source ./run.sh clear                  # Clear env vars
./run.sh download <model>              # Download model(s)
```

Logs are written to `logs/server-<slot>.log` and `logs/proxy-<slot>.log`.

### Running two models in parallel

```bash
./run.sh start qwen 1          # slot 1 → server :8001, proxy :8081
./run.sh start gemma 2         # slot 2 → server :8002, proxy :8082

# In terminal A:
source ./run.sh env qwen 1
claude

# In terminal B:
source ./run.sh env gemma 2
claude

./run.sh stop 1                # stop only slot 1
./run.sh stop                  # stop everything
```

## Models

- **qwen-moe** — Qwen3.6-35B-A3B MoE (ROCm)
- **qwen** — Qwen3.6-27B (ROCm)
- **gemma** — Gemma-4-31B-it (ROCm)
- **gemma-moe** — Gemma-4-26B-A4B-it MoE (Vulkan)
- **minimax** — MiniMax-M2.7 (Vulkan)
- **llama3.3** — Llama-3.3-70B (ROCm)
- **sauerkraut** — SauerkrautLM-104B (Vulkan)
- **mistral** — Mistral-Medium-3.5-128B (Vulkan)

## Architecture

- `proxy.py` — Flask proxy that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`
- `models.conf` — Model definitions (paths, binaries, ROCm env vars)
- `run.sh` — Main entry point for all commands
