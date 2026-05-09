# AI — LLM Server Manager

Script-based tool to manage local LLM inference servers and proxies for Claude Code.

## Setup

```bash
# Install Python dependencies
pip install flask requests

# Download a model
./run.sh download <model-name>
./run.sh download all   # all models
```

## Usage

```bash
./run.sh list                         # Show available models
./run.sh start <name>                 # Start server + proxy (foreground)
./run.sh stop                         # Stop all processes
./run.sh status                       # Show running state
./run.sh bench [--full] <model|all>   # Run benchmark (default: default ROCm + Vulkan)
./run.sh bench --full all             # Full test: all 8 ROCm combos + Vulkan
source ./run.sh env <name>            # Set Claude Code env vars
source ./run.sh clear                 # Clear env vars
./run.sh download <model>             # Download model(s)
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

- `proxy.py` — Flask proxy that forwards requests to the local llama-server and optimizes prompts for caching
- `models.conf` — Model definitions (paths, binaries, ROCm env vars)
- `run.sh` — Main entry point for all commands
