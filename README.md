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
while keeping `high` → `high`; `qwen` has no `high` at all and folds both onto
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
  qwen        effort      effort      enable_thinking, reasoning_effort  [qwen3.8-unc.jinja]
  muse        effort      effort      reasoning_strength
  minimax     unknown     -           not downloaded
```

It needs the `gguf` python module — either installed, or a llama.cpp checkout
(`LLAMA_SRC`, default `~/src/llama.cpp`).

### Speculative decoding

Generation on this box is bound by memory bandwidth, not compute. Every token
drags the whole model through memory once, so the token rate is simply
*bandwidth ÷ file size* — two measured points, both landing on ~220 GB/s:

```
Q8 27B   35.3 GB × 6.29 t/s = 222 GB/s
Q6_K 27B 27.5 GB × 7.98 t/s = 219 GB/s
```

Meanwhile `pp512` runs at 281 t/s against `tg128`'s 6.3 — about 45× more compute
sitting idle. Speculative decoding spends that idle compute: a cheap draft head
proposes N tokens and the big model verifies all of them in **one** pass, for
roughly the price of a single token.

Qwen3.8 ships its own draft head — the multi-token-prediction layer in `blk.64`,
which llama.cpp otherwise skips at load ("unused tensor ... ignoring"). No second
model, no extra download. `models.conf` declares it per model:

```bash
_model_spec_args=(--spec-type draft-mtp --spec-draft-n-max 4)
```

It is on by default wherever it is declared; `--spec off` (or `--no-spec`) turns
it off, `--spec on` errors out on a model that has no draft head. Measured on
qwen at its production context size:

| | tokens/s | acceptance | VRAM |
| --- | --- | --- | --- |
| off | 7.22 | — | 33.4 GB |
| on | **15.48** | 58 % | 36.9 GB |

That is 2.14× for 3.5 GB. The output distribution is unchanged by construction:
the big model verifies every token, it only does so in batches.

How deep to draft is the one thing worth tuning, and the answer depends on the
draft head's precision rather than on taste. Both quants of this model, swept:

| n-max | Q6_K head | Q8_K_P head |
| --- | --- | --- |
| 3 | **14.85** (47 %) | 14.36 (61 %) |
| 4 | 14.60 (38 %) | **15.48** (58 %) |
| 5 | — | 15.20 (51 %) |
| 6 | 13.06 (28 %) | 14.87 (46 %) |
| 8 | — | 10.95 (34 %) |

Drafting deeper stacks guesses on guesses, so acceptance decays with every step
— the question is only whether the saved verification pass outweighs it. The Q6
head starts at 47 % and cannot afford a fourth step; the Q8 head starts at 61 %
and can. This is why the entry runs the larger Q8 file even though it is 4 GB
more to read per token: on raw bandwidth Q8 is the slower choice (7.22 against
8.07 t/s), but the better head buys back more than it costs.

Only Qwen3.8 carries such a head. Qwen3.6 (`qwen-moe`) is an MTP-capable
architecture in llama.cpp but its GGUF contains no `nextn` tensors, and
llama/gemma4/muse-glimmer are not MTP architectures at all.

The two 70B models get there the other way, with `--spec-type draft-simple` and a
separate Llama-3.2-1B whose tokenizer is identical (checked by hashing all 128256
tokens). They are the slowest models here and gain the most:

| | without | with | acceptance |
| --- | --- | --- | --- |
| `llama3.3` (Q6_K, 57.9 GB) | 3.85 | **7.59** | 31 % |
| `diamond` (Q5_K_M, 49.9 GB) | 4.49 | **8.78** | 34 % |
| `gemma` (Q8_0, 32.6 GB) | 6.62 | **11.91** | 36 % |

`gemma` drafts from a gemma-4-E2B; the two 70B share one Llama-3.2-1B. The draft's
quant is its own tradeoff, since its cost is bandwidth and its benefit is accuracy:
on German prose Q6_K won (7.59) over both Q8_0 (7.32) and Q4_K_M (7.51).

That `llama3.3` figure is measured on **German** prose, which is what the model is
kept for. The same prompt in English accepts 54 % and reaches 10.34 t/s — language
alone nearly doubles the gain, so an English benchmark would badly overstate what
this box does in practice.

Size ratio is what decides it, not the quant level. Muse has neither an MTP head
nor a small sibling, so the only draft available is a coarser copy of itself —
Q2_K at 10.7 GB against 29.6 GB, a ratio of 1:2.8 where the working cases sit at
1:10 to 1:44. It loses: 7.73 t/s plain against 7.47 (n-max 2) and 7.13 (n-max 3),
despite 58 % acceptance, because a 30B draft also runs a full attention pass per
proposal on top of its bandwidth cost. Muse therefore declares no `spec_args`, and
the 10.7 GB file was deleted rather than kept.

**Never on an MoE.** Measured on `qwen-moe`: 44.58 t/s plain against 4.55 t/s with a
draft, a tenfold loss. An MoE reads only its active experts per token, so it is
already fast and a draft costs more than the tokens it saves; batch-verifying N
tokens then routes each to its own experts, widening the read instead of sharing
it. `qwen-moe` and `gemma-moe` declare no `spec_args`, which makes `--spec on`
refuse outright.

Diamond gains less because Magnum is heavily retrained, so a stock draft predicts
it worse. Acceptance also depends on the text: prose and dialogue run 15-20 points
below technical writing, the widest spread in these measurements — wider than
between any two draft models.

An abliterated draft was measured against both targets and lost 5 of 6 runs, which
is worth recording because the opposite sounds obvious. A refusal only ever affects
the first token of an answer, and speculation never re-decides it: the draft is
always fed the accepted prefix, so once the target has declined to refuse, a stock
draft follows the context like any other.

Unlike an MTP head, that draft is a second file from a second repo, so
`_model_hf_draft` holds `"<repo> <include-pattern>"` and `download` fetches it
after the model. Its destination is not configured — it is the directory of the
`--model-draft` path in `_model_spec_args`, so the two cannot drift apart.

### The server outlives its shell

`start` detaches the server with `setsid`, so it runs in its own session with no
controlling terminal: closing the terminal you started it from no longer sends it
SIGHUP, and it survives that shell exiting. `stop` and `status` are unaffected —
they go by the PID file, which the server writes itself just before exec rather
than taking `$!`, since under job control `$!` names the `setsid` wrapper instead
of `llama-server`.

The proxy (`--proxy`) and the TLS front (`--public`) are still plain background
jobs of the calling shell.

### Idle CPU (`--poll 0`)

llama.cpp's worker threads busy-wait on GPU completions by default, which shows up
as 700-1400 % CPU while the GPU does the actual work — enough to make the machine
feel loaded when it is only waiting. `--poll 0` is in `_common`, measured across
every model with and without speculation:

| | tokens/s | CPU |
| --- | --- | --- |
| polling | 3.85 - 44.54 | 733 - 1400 % |
| `--poll 0` | within 2 % of it | 22 - 49 % |

Throughput moves by at most 2 % and usually by nothing; two of the ten runs came
out marginally faster, which is the size of the noise. Waiting is worst *without*
speculation, since each token leaves the CPU idle longer.

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
- **qwen** — Qwen3.8-27B uncensored (HauhauCS), Q8_K_P, 64K ctx, mmproj available; drafts its own tokens from the MTP head in the GGUF (~2.1×), and runs `templates/qwen3.8-unc.jinja` for the uncensored system default. Replaced the 3.6-27B entry, which it supersedes outright
- **qwen-vl** — Qwen3-VL-8B vision-language uncensored, Q8_0, 8K ctx, mmproj available
- **muse** — Muse-Glimmer-30B abliterated aggressive (Meta base, agentic), Q8_0, 128K ctx, mmproj available
- **gemma** — Gemma-4-31B-it uncensored, Q8_0, 128K ctx (uncensored); vision via the stock repo's mmproj, drafted by gemma-4-E2B (~1.8×)
- **gemma-moe** — Gemma-4-26B-A4B-it MoE uncensored, Q8_0, 128K ctx (uncensored); vision via the stock repo's mmproj, no speculation (MoE)
- **minimax** — MiniMax-M2.7, UD-IQ3_S, 64K ctx
- **llama3.3** — Llama-3.3-70B-Instruct abliterated, Q6_K, 32K ctx; drafted by Llama-3.2-1B (~2.7×)
- **r1** — DeepSeek-R1-Distill-Llama-70B Uncensored v2 Unbiased Reasoner, i1-Q5_K_M, 128K ctx
- **mistral** — Mistral-Medium-3.5-128B, UD-Q5_K_XL, 32K ctx
- **diamond** — L3.3-70B Magnum Diamond, i1-Q5_K_M, 32K ctx; drafted by the same Llama-3.2-1B (~2.0×)

Multimodal projectors are only loaded on an explicit `--mmproj`.

## Architecture

- `proxy.py` — optional Flask proxy (`start --proxy`) that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`
- `models.conf` — Model definitions (paths, binaries, ROCm env vars)
- `run.sh` — Main entry point for all commands
- `deploy/vps-llm-vhost.conf` — Apache vhost for the VPS in front of `--public`
