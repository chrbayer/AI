# llmctl — LLM Server Manager

Script-based tool to manage local LLM inference servers and proxies for Claude Code.

## Installation & directories

```bash
sudo make install          # program to /usr/local (bin/llmctl, lib/llmctl, share/llmctl)
sudo make install-link     # instead: /usr/local/bin/llmctl → this checkout, edits apply at once
sudo make uninstall

pip install flask requests # for proxy.py
pip install waitress       # optional, recommended for production proxy
```

The program is read-only; everything you own or the servers produce lives in
your home directory:

| Where | What | Override |
| --- | --- | --- |
| `~/.config/llmctl/` | `models.conf`, `presets.conf`, `tokens`, `tls/` | `LLMCTL_CONFIG_DIR` (tokens/tls also `LLM_CONF_DIR`, `LLM_TOKEN_FILE`) |
| `~/.local/share/llmctl/models/` | the GGUF files (`$MODELS_DIR` in `models.conf`), ComfyUI's models in `comfyui/` | `LLMCTL_MODELS_DIR`, or set `MODELS_DIR` in `models.conf` |
| `~/.local/share/llmctl/` | `benchmarks/`, `claude/<model>[-<slot>]` (Claude Code profiles set by `env`), `comfyui/` (ComfyUI's checkout in `app/`, its workflows, input and output), `tts/` (voices, voice-design venv) | `LLMCTL_DATA_DIR` |
| `~/.local/state/llmctl/` | `logs/`, `pids/`, `slots/`, `stunnel/` | `LLMCTL_STATE_DIR` |

`examples/` holds a complete `models.conf` and `presets.conf`. Copy them to edit
freely, or link them to a checkout so changes — including `preset-save` — stay
under version control:

```bash
mkdir -p ~/.config/llmctl
cp /usr/local/share/llmctl/examples/*.conf ~/.config/llmctl/
# or
ln -s ~/AI/examples/{models,presets}.conf ~/.config/llmctl/
```

A command cannot change the environment of the shell that runs it, so `env` and
`clear` print shell code to evaluate. Once in `~/.zshrc` (or `~/.bashrc`)

```bash
eval "$(llmctl shell-init)"
```

and `llmctl env <name> [slot]` / `llmctl clear` apply to the current shell
directly — that is the form used below. Without it: `eval "$(llmctl env qwen 1)"`.

```bash
llmctl download <model-name>
llmctl download all   # all models
```

## Usage

```bash
llmctl list                            # Show available models
llmctl start <name> [slot] [--proxy] # Start server in background (any slot, default 1)
llmctl presets                         # Show the defined presets
llmctl preset <name>                   # Bring the machine to a whole configuration (see below)
llmctl preset-save <name>              # Record what is running now as a preset
llmctl stop [slot]                     # Stop slot, or all if omitted
llmctl status                          # Running state, model and key parameters (all slots)
llmctl cache-stats [slot]              # Prompt-cache hit rate, read from the server log
llmctl clear-kv [slot]                 # Drop the KV cache without restarting
llmctl probe-reasoning [model]         # What each model's chat template supports
llmctl gen-certs <host>                # CA + server/VPS certificates for --public
llmctl bench [--full] <model|all>      # Run benchmark (default: default ROCm + Vulkan)
llmctl bench --full all                # Full test: all 8 ROCm combos + Vulkan
llmctl bench flash                     # halogen: prefill + decode of the running slot
llmctl env <name> [slot]               # Set Claude Code env vars
llmctl clear                           # Clear env vars
llmctl download <model>                # Download model(s)
llmctl version                         # Print the version
```

### `start` options

| Option | Effect |
| --- | --- |
| `--proxy` | also start `proxy.py` (see below) |
| `--reasoning off\|on\|low\|medium\|high\|max\|N` | one switch for every model; `N` is a token budget |
| `--no-reasoning`, `--reasoning-budget N` | aliases for the above |
| `--parallel N` | server slots (default 1) |
| `--ctx N` | override the model's default context size |
| `--cache-ram N` | prompt-cache host-RAM cap in MiB (0 = disable, -1 = no limit; default 8192) |
| `--similarity F` | prefix share (0..1) a slot must already hold to be reused — llama-server's `--slot-prompt-similarity` (default 0.1; 0 = pure LRU slot pick) |
| `--spec on\|off`, `--no-spec` | speculative decoding; on by default for every model that declares a draft (`qwen`, `gemma`, `llama3.3`, `diamond`, `magnum`) |
| `--temp F`, `--top-p F` | override the samplers `models.conf` sets for this model. Passed last, so they also win over the non-thinking sampler set that `--reasoning off` brings with it — the same model can run at `--temp 0.1` for code and at its own 0.6 for prose |
| `--mmproj` | load the model's multimodal projector (vision), where `models.conf` defines one |
| `--host ADDR` | bind address (default 127.0.0.1; `0.0.0.0` exposes the server on the LAN) |
| `--gpu-priority low\|medium\|high\|realtime` | Vulkan queue priority (needs the patched ggml-vulkan) |
| `--verbose` | `-lv 4`, reveals ggml/backend + buffer-size startup logs |
| `--clear-logs` | truncate this slot's server/proxy log before starting |
| `--public` | token auth + hardening + mTLS front for the VPS (needs `gen-certs`) |
| `--max-predict N` | cap tokens per generation (-1 = no limit; `--public` defaults to 8192) |
| `--print-cmd` | build and validate everything, print the llama-server command, start nothing (this is how `preset` compares a slot against what it should run) |
| `--output DIR` | ComfyUI only: where generated images go (default `~/.local/share/llmctl/comfyui/output`) |

### The proxy is opt-in

`start` runs only llama-server. Pass `--proxy` to also start `proxy.py`, which
rewrites time/date stamps in prompts so the prompt cache stays warm — worth it
for clients that stamp every request, pointless for those that don't. (A
[halogen](#the-halogen-backend) model is the exception: its proxy always runs.)

It waits as long for the server as llama-server's own `--timeout` (600 s), so a
long prefill is not cut off halfway. With `--public` it also fetches image URLs
itself — public hosts only, redirects included — and passes them on inline:
llama-server would otherwise fetch any URL a token holder names, `127.0.0.1` and
the LAN included. That covers the proxied path; the direct one is not.

`env` follows suit: it points at the proxy when one is listening for that slot,
at llama-server otherwise. `--proxy` / `--direct` force the choice, e.g. when
setting the env before starting the slot.

```bash
llmctl start qwen 1 --proxy            # server :8001 + proxy :8081
llmctl env qwen 1                      # → :8081 (proxy detected)
llmctl env qwen 1 --direct             # → :8001 (bypass the proxy)
```

Logs are written to `~/.local/state/llmctl/logs/server-<slot>.log` and `proxy-<slot>.log`.

### Slots

A slot is just a number. Nothing enumerates slots and none is preallocated: the
ports follow from the number (`:800N` for the server, `:808N` for the proxy),
and commands that work on "all slots" find them from the PID files in
`~/.local/state/llmctl/pids/` plus whatever is listening on those ports. Start
slot 7 without ever having used 4, 5 or 6.

The only limit is where the port ranges meet, so it moves with the bases:

| | Range | Bound by | Override |
| --- | --- | --- | --- |
| server / proxy | slots 1–79 | `:8080` is where the proxy ports start | `LLMCTL_PORT_BASE_SERVER`, `LLMCTL_PORT_BASE_PROXY` |
| `--public` | slots 1–9 | `:8450` is where the TLS proxy ports start | `LLMCTL_PORT_BASE_TLS_SERVER`, `LLMCTL_PORT_BASE_TLS_PROXY` |

Both are checked before anything starts, and `llmctl help` prints the ranges that
are actually in effect.

### Running two models in parallel

```bash
llmctl start qwen 1 --proxy    # slot 1 → server :8001, proxy :8081
llmctl start gemma 2           # slot 2 → server :8002 (no proxy)

# In terminal A:
llmctl env qwen 1
claude

# In terminal B:
llmctl env gemma 2
claude

llmctl stop 1                  # stop only slot 1
llmctl stop                    # stop everything
```

### Presets

A preset is a whole configuration under one name: which models run, on which
slot, with which flags. `presets.conf` holds them, in the same shape as
`models.conf`:

```bash
_preset_name="c4f"
_preset_label="Llama-3.3-70B with Qwen3-VL-8B next to it for images"
_preset_entries=(
    "llama3.3 2 --parallel 5 --ctx 61440 --similarity 0.92"
    "qwen-vl  3 --mmproj"
)
add_preset
```

Each entry is exactly what would follow `llmctl start` — model, slot, then any
`start` flag. They are passed on untouched, so a preset can do whatever `start`
can do and there is no second option dialect to keep in sync.

```bash
llmctl presets                   # what is defined, and what each one starts
llmctl preset c4f                # bring the machine to that configuration
llmctl preset c4f --dry-run      # show the plan, change nothing
llmctl preset c4f --force        # reload everything, even what already matches
```

Some start options say how a slot is run and reached rather than what the model
does, and those apply to a whole configuration equally. They are given on the
command instead of in the file, and appended to every entry:

```bash
llmctl preset wrs --host 0.0.0.0               # the whole set on the LAN
llmctl preset wrs --host 0.0.0.0 --public      # …with token auth and the TLS front
llmctl preset wrs --clear-logs --verbose       # a fresh, loud run of the same set
```

`--proxy`, `--public`, `--host ADDR`, `--clear-logs`, `--verbose`,
`--gpu-priority L` and `--output DIR` are the set. Each entry gets those that
mean something to its backend: a ComfyUI entry takes `--output` (where its
images go) but no proxy, public front or GPU priority, and `--output` reaches
ComfyUI entries only — a preset without one refuses it. They come after the entry's own flags, so a
`--host` given here overrides one written into the entry. Everything that shapes
the model itself — `--ctx`, `--reasoning`, `--temp`, `--parallel`, `--cache-ram`,
`--spec` — belongs to its entry in `presets.conf` and is refused here, since it
would mean something different for each model in the set. The same options also
take part in the comparison, so `preset wrs` and `preset wrs --host 0.0.0.0`
describe two different configurations and each will restart what the other left.
`--clear-logs` is the exception that cannot reach a slot which keeps running; the
plan says so when that happens.

`preset` does not start blindly — it brings the machine to the named
configuration, keeping whatever already fits:

| Slot state | What happens |
| --- | --- |
| runs exactly what the preset asks for | kept, untouched |
| runs the same model with different flags, or a different model | stopped, started again as the preset wants it |
| empty | started |
| runs something the preset does not name | stopped — a preset describes the whole machine |
| model matches, only the proxy is missing or surplus | just the proxy is started or stopped, the weights stay loaded |

"Exactly" is meant literally: the target command is compared flag by flag against
the running process's own argv (`/proc/<pid>/cmdline`), so a differing `--ctx` or
`--parallel` counts as a mismatch. The comparison uses `start --print-cmd`, which
builds and validates the whole command and starts nothing — so `start` stays the
single authority on what an entry means.

```
Preset 'c4f' — Llama-3.3-70B with Qwen3-VL-8B next to it for images

  slot 1  -              Gemma-4-31B-it Unc       stop — not part of this preset
  slot 2  llama3.3       Llama-3.3-70B Abl        already running like this — keep
  slot 3  qwen-vl        Qwen3-VL-8B Unc          already running like this — keep
```

Everything that has to go is stopped before the first new model loads — the new
weights need the GPU memory the old ones hold. The models are then started one after
another, and each one is waited for until llama-server writes its own
`listening on http://…:800N` line — the line it prints after `model loaded`. That
serializes GPU memory allocation, and it makes the command return only once the last
model is actually ready to answer. The wait ends early if the server reports an
error (a severity `E` line) or exits, and gives up after `--wait N` seconds
(default 300). The limit is there for a server that hangs, not for one that is
merely slow: the dense 27B and the 35B MoE measured here took 2:11 and 3:15 to
load, so a big set stays well inside it.
If one does not come up, the slots *this run* started are stopped again; slots
that were already running and matched are left alone.

Before it loads anything, a preset adds up the GPU memory its set needs and
prints the balance under the plan, `--dry-run` included:

```
GPU memory (GTT pool 104.0 GiB, 0.0 GiB used outside llmctl, 2.0 GiB kept free):
  slot 2   llama3.3        54.9 GiB  (weights only — not measured yet)
  slot 3   qwen-vl         10.3 GiB  (measured)
  total                    65.2 GiB  of 102.0 GiB available
```

The numbers are measured, not modelled: KV cache per context and slot, hybrid
and sliding-window layers, cache types, drafts and projectors make a formula
unreliable. Once a slot has loaded, what its server holds on the GPU (from its
DRM fdinfo; for halogen, which runs alone, the GTT total) is written to
`~/.local/state/llmctl/footprints.tsv` under its exact start command. `preset`
records it after each load and `status` refreshes it. A command never measured
counts with the size of the files it loads — a lower bound, and marked as one;
halogen counts ~98 GiB until measured. The budget is the GTT pool minus what
other programs hold on the GPU and 2 GiB of headroom. When the set does not fit,
`preset` warns and loads anyway: an estimate can be off, and a model that fails
to load is rolled back as before. With ComfyUI in the set it also says which
workflows fit into what is left, by the size of their weights (ComfyUI's models
are unloaded before each LLM start, see [the comfyui backend](#the-comfyui-backend)).

On this machine that memory is GTT: the BIOS reserves only 1 GiB of VRAM, and
the GPU takes up to 104 GiB (`ttm.pages_limit=27262976`) out of the 124.4 GiB the
system has — the same pool the system and the host-RAM prompt caches live in.

### Recording what runs as a preset

`preset-save` is the other direction: it reads the running slots and writes them
back into `presets.conf`.

```bash
llmctl preset-save c4f                        # append what runs now as preset "c4f"
llmctl preset-save c4f --dry-run              # print the block, write nothing
llmctl preset-save c4f --force                # replace an existing preset of that name
llmctl preset-save c4f --label "for images" # description for the listing
```

The flags are derived from the running process — `--ctx` only when it differs
from the model's default, `--spec off` only when a model that could speculate
does not, `--reasoning` read back through the level map of that model's template,
`--proxy` when a proxy is up, `--gpu-priority` from the process environment.
Samplers are the one thing that cannot be read off directly — `models.conf` sets
them for every model — so the capture tries the smallest set of `--temp`/`--top-p`
that reproduces the command, and writes none when the model's own values are in
force. None
of that is trusted on derivation alone: each entry is fed back through
`start --print-cmd` and has to reproduce the server's own argv flag for flag.
Only then is it written. A slot that cannot be expressed in `start` options — a
model missing from `models.conf`, or a server started by hand — stops the whole
save with a diff showing where it parts ways, and nothing is written.

Two harmless normalizations follow from that: a flag that only restates a default
is dropped, and where a model maps two levels onto the same template value (qwen
sends both `high` and `max` as `xhigh`) the first of them is written. Both
produce the identical command, which is what the verification checks.

`status` names the preset a slot came from; `stop` clears that marker, as does
starting something else on the slot by hand.

```
Slot 2: llama-server (PID: 1826416) on :8002  [Preset: c4f]
```

Every slot stays an ordinary slot, so `llmctl env <model> <slot>` picks
one of the running models per shell as before.

### Reasoning

One switch for every model:

```bash
llmctl start <name> [slot] --reasoning off          # no thinking
llmctl start <name> [slot] --reasoning on           # thinking, template default depth
llmctl start <name> [slot] --reasoning high         # low | medium | high | max
llmctl start <name> [slot] --reasoning 2048         # thinking, capped at N tokens (-1 = uncapped)
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
$ llmctl start qwen 1 --reasoning low
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
$ llmctl probe-reasoning
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

| | tokens/s | acceptance | GPU memory |
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

The 70B-class models get there the other way, with `--spec-type draft-simple` and a
separate small model of the same family. They are the slowest models here and gain
the most:

| | without | with | acceptance |
| --- | --- | --- | --- |
| `llama3.3` (Q6_K, 57.9 GB) | 3.85 | **7.59** | 31 % |
| `diamond` (Q5_K_M, 49.9 GB) | 4.49 | **8.78** | 34 % |
| `magnum` (Q6_K, 64.4 GB) | 3.49 | **6.93** | 32 % |
| `gemma` (Q8_0, 32.6 GB) | 6.62 | **11.91** | 36 % |

`gemma` drafts from a gemma-4-E2B, `llama3.3` and `diamond` share one Llama-3.2-1B
whose tokenizer is identical to theirs (checked by hashing all 128256 tokens), and
`magnum` — a Qwen2.5-72B fulltune, not a Llama derivative — drafts from a
Qwen2.5-1.5B-Instruct. That last pair is the one case here where the vocabularies
are not identical, 152064 against 151936. llama.cpp allows a difference of at most
`SPEC_VOCAB_MAX_SIZE_DIFFERENCE` = 128 and then compares the token texts from id 5
up, so this pair fits with nothing to spare — a draft one step further from the
target would be refused at load.

The draft's quant is its own tradeoff, since its cost is bandwidth and its benefit
is accuracy. `magnum` was measured across the whole ladder, mean of German and
English prose:

| draft | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
| --- | --- | --- | --- | --- | --- | --- |
| size | 0.63 GB | 0.77 GB | 0.92 GB | 1.05 GB | 1.19 GB | 1.53 GB |
| t/s | 5.27 | 5.88 | **6.35** | 6.15 | 5.89 | 5.70 |
| acceptance | 18 % | 24 % | 27 % | 27 % | 25 % | 24 % |

The peak sits in the middle and both ends fall away for different reasons: below
Q4_K_M the draft stops guessing well (acceptance drops to 18 %), above it the extra
bandwidth per proposal is no longer repaid. `llama3.3` peaked one step higher, at
Q6_K (7.59) over Q8_0 (7.32) and Q4_K_M (7.51) — so the optimum is per model, not a
constant, and worth measuring rather than assuming.

Draft depth was swept the same way and lands on the same `--spec-draft-n-max 4` the
other 70B use, again as an interior peak:

| n-max | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| t/s | 5.88 | 6.18 | **6.35** | 5.86 | 5.45 |
| acceptance | 44 % | 37 % | 32 % | 23 % | 20 % |

Acceptance only ever falls as the draft lengthens, because a short proposal is
likelier to be taken whole. Up to 4 the longer accepted runs still outweigh that
decay; past it they no longer do, and the rejected tail is decoded for nothing.

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

**Never on a vision turn.** `qwen-vl` gains on text — 25.4 t/s plain against 34-39
with a Qwen3-0.6B draft, whose vocabulary is identical, so roughly 1.3-1.5x. Any
request carrying an image then fails outright:

```
HTTP 500  decode() failed: failed to process speculative batch
```

The server hands the draft only the text tokens (`get_text_tokens()` in
server-context.cpp drops the image placeholders), so the two contexts drift apart
the moment an image is in the prompt — the draft's KV cache stands at position 4
while the target continues at 52, past 48 image tokens, and the batch is refused
because positions must stay consecutive. It is not a quality tradeoff that costs
acceptance, it is a hard failure, and it is independent of the draft's quant.
`qwen-vl` therefore declares no `spec_args`: a gain on text turns is not worth
losing the one thing the model is kept for.

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

`start` detaches everything it launches — the server, the proxy under `--proxy`,
the TLS front under `--public` — through `_spawn_detached`. Each runs in its own
session with no controlling terminal, so closing the terminal you started it from
no longer sends it SIGHUP and it survives that shell exiting.

`stop` and `status` are unaffected: they go by the PID file, which the child
writes itself just before exec rather than the caller taking `$!`. Under job
control `setsid` finds itself a process group leader and forks instead of
exec-ing, so `$!` would name the wrapper and the PID file would point at a
process that has already gone.

### What `status` reports

Besides the PIDs and ports, `status` names the model each slot serves and the
parameters that decide how it behaves:

```
Slot 1: llama-server (PID: 330778) on :8001
         Model:     Qwen3.6-35B-A3B MoE Unc (qwen-moe)
         Params:    ctx 65536, parallel 1, prompt-cache off, host 0.0.0.0
         Reasoning: off
         Spec:      off
         Log: tail -f ~/.local/state/llmctl/logs/server-1.log
```

It reads that from `/proc/<pid>/cmdline`, the argv of the process that is
actually running, not from the config `start` wrote to the log — so it stays
right for a slot that was started by hand, and it cannot describe an older run
whose log is still lying around. The label comes from the `models.conf` entry
whose model file matches; a file no entry knows is shown by its own name.
`Params` lists only what departs from the defaults, apart from `ctx` and
`parallel`, which are always named.

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
llmctl clear-kv            # every running slot
llmctl clear-kv 2          # only slot 2
```

The underlying endpoint is llama-server's own, and the proxy forwards it
unchanged:

```bash
curl -X POST 'http://localhost:8001/slots/0?action=erase'   # server slot 0
curl -X POST 'http://localhost:8081/slots/0?action=erase'   # same, via the proxy
```

Slot ids run from 0 to `--parallel N` minus one — `clear-kv` reads the count from
`/props` and walks all of them. The action needs `--slot-save-path`, which `start`
always passes (`~/.local/state/llmctl/slots/`).

Two limits worth knowing:

- **It never interrupts a running generation.** The server defers the erase until
  the slot falls idle, so the call can block for as long as the generation takes.
- **It does not touch the host-RAM prompt cache** (`--cache-ram`, 8192 MiB by
  default), and no endpoint does. Measured on a cleared slot: of a 53-token
  prompt sent again afterwards, 52 tokens came straight back from host RAM. For a
  hard reset, start the slot with `--cache-ram 0`.

## The halogen backend

A models.conf entry with `_model_backend="halogen"` runs
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
instead of llama-server: a closed-source engine for one model family
(Qwen3.8-Flash-Next, 125B MoE) on Strix Halo, shipped as a container and
reading its own `.hgn` weights. The slot model stays the same — `start`, `stop`,
`status`, `env`, `preset`, `preset-save` all work — with these differences:

- **It runs in podman.** The server PID is the `podman run` client (its argv is
  the full command, which is what `preset` compares), the container's output is
  the server log, and the container is named `llmctl-halogen-<slot>`. `stop` uses
  `podman stop`, which waits for the container to exit and removes it.
- **It runs exclusively.** It pins ~68 GiB of weights and reserves ~30 GiB of KV
  pool and working memory — most of a 128 GB machine. `start` refuses it beside
  any other slot and refuses any other model beside it; a preset that names it
  can name nothing else. `preset` switching between a halogen and a llama
  configuration works, since it stops before it starts.
- **Its proxy always runs.** The server speaks OpenAI Chat Completions and
  Responses, not Anthropic Messages, so `proxy.py` translates `/v1/messages`
  (see `anthropic_compat.py`): system, text, images, tool use and results,
  thinking, streaming with pings during prefill. It also clamps token budgets to
  the server's cap — halogen answers a larger `max_tokens` with HTTP 400 instead
  of shortening it, and Claude Code asks for 32000 — and waits up to an hour for
  the backend, since a full 256K prefill alone takes minutes. Images given as
  http(s) URLs — which halogen refuses — are fetched by the proxy and passed on
  inline (Messages, Chat Completions and Responses alike; images only, at most
  20 MiB). Under `--public` only hosts that resolve to public addresses are
  fetched, redirects included, so a token cannot reach this machine or the LAN
  through it. `env` always points
  at the proxy, and also exports `CLAUDE_CODE_MAX_CONTEXT_TOKENS` with the slot's
  context, which Claude Code cannot know for a model outside its catalog.
- **Settings are container environment variables.** `extra_args` holds
  `HALOGEN_*=VALUE` pairs; start options map onto the same variables:

  | Option | halogen |
  | --- | --- |
  | `--ctx N` | `HALOGEN_CTX`; above the native 262144 also `HALOGEN_ROPE_YARN` (⌈N/262144⌉), and the KV pool grows to hold one full request |
  | `--parallel N` | `HALOGEN_KV_SLOTS` (conversations generating at once, sharing one pool) |
  | `--reasoning off\|level\|N` | `HALOGEN_ENABLE_THINKING=0`, `HALOGEN_REASONING_EFFORT`, `HALOGEN_MAX_THINKING_TOKENS` — server defaults a request may override |
  | `--temp`, `--top-p` | `HALOGEN_TEMPERATURE`, `HALOGEN_TOP_P` (unset = greedy) |
  | `--max-predict N` | `HALOGEN_MAX_TOKENS_CAP`, and the proxy clamps to it |
  | `--cache-ram 0` | `HALOGEN_PROMPT_CACHE=0`; no other value exists |
  | `--mmproj` | `HALOGEN_VISION_TOWER=1` (the vision file must sit beside the checkpoint) |
  | `--host ADDR` | where podman publishes the port |
  | `--public` | a TLS front for the **proxy only** — the server has no authentication, so its own port never gets one, and `--public` with a LAN `--host` is refused |

  `--similarity`, `--gpu-priority`, `--verbose` and `--spec off` have no
  equivalent and are refused. The MTP draft is always on; it is byte-identical
  to plain decoding.
- **`--compact`** (any model, but made for this one) has the kernel defragment
  free memory before the start (`sudo`). The server needs its ~30 GiB in 2 MiB
  pieces; with fewer free, the kernel compacts on the fly and startup and
  every prefill can stall for minutes. `start` measures this and suggests
  `--compact` below 40 GiB.
- `download` also pulls the container image, and offers to remove older tags of
  it that no models.conf entry names and no container runs (a few GB each);
  without a terminal it prints the `podman rmi` command instead.
- `clear-kv` has nothing to call (restart with `--cache-ram 0` for a hard reset),
  `cache-stats` reads the server's own `/cache` counters, `probe-reasoning` reads
  `tokenizer/chat_template.jinja` beside the checkpoint, and `bench` measures
  the running slot over HTTP (`halogen_bench.py`), since llama-bench cannot read
  `.hgn`: prefill at 1K/8K/32K tokens (`--full`: up to 128K) with the prompt
  cache defeated, decode on prose and code. The runs land in the same
  `benchmarks/*.jsonl` as llama-bench's, with a `results` list per run. While a
  halogen server runs, `bench` skips the llama models.

**Memory as `free` shows it is misleading.** The weights are pinned by the GPU
driver straight out of the file cache, not `mlock`ed, so `free` and
`MemAvailable` count them as reclaimable cache: ~80 GiB "available" while
~13 GiB really are. `status` prints the real figure.

**The host matters more than for llama-server.** Measured on this machine:
prefill 1100–1300 t/s and decode 35 t/s (prose) / 48 t/s (code) after a fresh
boot with `amd_iommu=off amdgpu.noretry=0` on the kernel line; the same server
with fragmented memory managed 8–370 t/s of prefill, most requests stalling
40–200 s in kernel compaction. Keep `vm.compaction_proactiveness` at the kernel
default (20) — 0 does not prevent the stalls.

## The comfyui backend

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) generates and edits images. It is
no LLM, but it draws on the same GPU memory: FLUX.2 klein 9B or Qwen-Image 2.1
need ~30 GB of weights, FLUX.2 dev took ~105 GB while it ran. So it takes a slot
like any server, and `start`, `stop`, `status`, `preset` and `preset-save` all
handle it. A models.conf entry with `_model_backend="comfyui"` describes it
(`examples/models.conf` has `comfy`):

| Field | For comfyui |
| --- | --- |
| `binary` | the ComfyUI checkout, with its `.venv` beside `main.py` (`~/.local/share/llmctl/comfyui/app`) |
| `model` | the directory of the image models (`~/.local/share/llmctl/models/comfyui`) |
| `extra_args` | additional ComfyUI flags (`--disable-pinned-memory --use-pytorch-cross-attention`) |
| `rocm_env` | its environment (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) |

The checkout's parent directory holds what ComfyUI writes: `user/` (settings,
saved workflows), `input/`, `output/`, `temp/` and `custom_nodes/`
(`--base-directory`).

```bash
llmctl download comfy                 # set it up, fetch every model its workflows name
llmctl download comfy klein           # …only what the workflows with "klein" in the name need
llmctl start comfy 9                  # web UI on http://127.0.0.1:8009
llmctl update comfy                   # git pull, Python dependencies, new workflows
llmctl update comfy --torch           # …and torch itself
```

- **Setup.** `download` does whatever is still missing: a git checkout of
  ComfyUI, a venv with torch built for ROCm 7.2 (`COMFYUI_PYTHON` picks the
  interpreter, default `python3.13`), `requirements.txt` held to that torch, the
  CUDA packages a dependency drags in removed, the patches from
  `comfyui/patches/` applied, the bundled workflows copied, and the models.
  `--from DIR` moves an existing checkout into place instead of building a new
  one: its models, `user/`, `input/` and `output/` go to where llmctl keeps
  them, the checkout becomes `app/`, and the venv's scripts are rewritten to the
  new path. On one filesystem all of it is a rename.
- **Workflows.** `comfyui/workflows/` holds the workflows, and `download`/`update`
  copy the ones ComfyUI does not have yet into
  `~/.local/share/llmctl/comfyui/user/default/workflows/`. A workflow you
  changed in the UI stays as it is; they only report that it differs. To keep
  the workflows under version control instead, link the directory to a checkout
  — what the UI saves then lands there:
  `ln -sfn ~/AI/comfyui/workflows ~/.local/share/llmctl/comfyui/user/default/workflows`
- **Models come from the workflows.** Each loader node in a saved workflow names
  its file with source and folder (`properties.models`: name, url, directory).
  `download` collects these from every workflow, fetches each file once — the
  FLUX.2 dev text encoder serves all four dev workflows — and never fetches a
  file that is there. Where workflows name one file with different URLs, the
  URLs are tried in order and `list` reports the disagreement. Gated repos
  (black-forest-labs) need their licence accepted and `hf auth login`. `list`
  shows per workflow how many of its models are present, and the files no
  workflow names any more; nothing is ever deleted.
- **Models that exist only in another layout** are built instead of fetched.
  `comfyui/sources.json` holds a recipe for each: the repo pinned to a commit,
  its safetensors shards, and how to rename the tensors. `download` fetches the
  shards and writes one file with the new names, copying the tensor data byte
  for byte; tensor count and size are checked first. The Heretic text encoder
  for Qwen-Image 2.1 is such a case: its repo is a `transformers` checkpoint,
  and dropping the `language_model` level gives a file whose header is
  byte-identical to Comfy-Org's `qwen3vl_8b_bf16.safetensors`.
- **Civitai.** Workflows can name models on civitai.com, which hands most files
  out only to a logged-in account. `download` takes the API key from
  `~/.config/llmctl/civitai-token` (or `CIVITAI_TOKEN`). When there is none, or
  Civitai refuses it, `download` asks for one on the terminal, saying where to
  create it (civitai.com → Account settings → API Keys), and stores it there
  with mode 600. The key goes into the download URL's query and is masked in
  every message.
- **NSFW workflows.** `FLUX.2 klein 9B NSFW T2I/Edit` and `FLUX.2 dev NSFW
  T2I/Edit` add a LoRA. For klein it is what makes nudity possible at all:
  without it klein dresses a figure the prompt describes as nude. FLUX.2 dev
  renders nudity on its own; there the LoRA shapes the style — a warmer film
  look, different poses and bodies. klein uses
  [Flux Klein – NSFW v2](https://huggingface.co/diroverflo/FLux_Klein_9B_NSFW)
  (Hugging Face), dev
  [SexGod Flux.2 D Female Nudity](https://civitai.com/models/2604891)
  (Civitai; trigger word `femalenudestyle`). Each LoRA sits in the subgraph
  right after the model loader; its strength is the **LoRA strength** slider
  on the workflow's node (1.0 by default, 0 to compare against the plain
  model). In the dev workflows it comes before the Turbo switch, so Turbo
  still toggles. An abliterated klein text
  encoder was measured too and left out: on its own it changes nothing, since
  the reluctance lives in the image model, not in the encoder. Adults only.
- **Custom nodes.** `comfyui/custom_nodes.txt` names the node packs llmctl
  installs, each pinned to a commit: `download` clones them into
  `~/.local/share/llmctl/comfyui/custom_nodes/`, `update` moves them to the
  commit the file names, and their requirements go into ComfyUI's venv with
  torch held to its ROCm build (`-name` after the commit drops a requirement,
  `+name` adds one). Restart ComfyUI after either. Node packs that load whole
  model directories name them as Hugging Face repos
  (`https://huggingface.co/<org>/<repo>`), and `download` fetches the snapshot
  into `<directory>/<name>`. Because such packs look for their models under
  ComfyUI's base directory rather than `--models-directory`, llmctl links
  `~/.local/share/llmctl/comfyui/models` to the models directory.
- **Speech in ComfyUI.** [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)
  brings Qwen3-TTS; two workflows use it. `Qwen3-TTS Stimme entwerfen` designs
  a voice from a description (VoiceDesign), `Qwen3-TTS Stimme klonen` speaks in
  the voice of a recording (Base, `x_vector_only`, so the clone speaks
  accent-free German). The speech slot's voices appear in ComfyUI's input as
  `voice-<name>.wav` — hard links, since LoadAudio refuses a symlink that leads
  outside its input directory — and `llmctl voice` keeps them in step. The
  VoiceDesign model `voice design` already has is reflinked, not downloaded
  again. Both run through PyTorch at about 1.6× real time; for speech on
  demand the speech slot (llama.cpp) is faster.
- **Patches.** `comfyui/patches/qwen35-rocm-conv3d.patch` works around a
  segfault of PyTorch's Conv3d fallback on ROCm in the Qwen3-VL vision encoder
  (the Qwen-Image 2.1 edit workflow). `update` takes the patches out before
  pulling and puts them back after; one that no longer applies is reported, and
  ComfyUI runs without it.
- **Memory.** ComfyUI loads models per workflow and keeps them until something
  inside it needs the room — it cannot see an LLM coming. So every llama
  `start`, a preset's included, first asks each running ComfyUI to unload
  (`POST /free`) and waits until the memory is gone. A ComfyUI with a job in its
  queue is left alone, with a warning. `clear-kv` on its slot unloads too.
  `status` shows the GPU and RAM the process holds, read from its DRM fdinfo.
- **Exclusive with halogen**, as every other slot is: a preset that switches to
  halogen stops ComfyUI.
- **Images** go to `~/.local/share/llmctl/comfyui/output/`, or wherever
  `--output DIR` says (`llmctl start comfy 9 --output ~/Bilder/comfy`; also in a
  preset entry, where `~` is expanded too). `preset-save` records it.
- **No LLM options.** `--host`, `--verbose` (`--verbose DEBUG`), `--clear-logs`
  and `--output` apply; everything else is refused. `env`, `bench`, `cache-stats` and
  `probe-reasoning` have nothing to do for it. ComfyUI has no authentication, so
  `--host` beyond localhost warns.

## The tts backend (speech)

A models.conf entry with `_model_backend="tts"` (`speech` in the examples)
speaks: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B through llama.cpp,
behind `tts_server.py`, which answers the OpenAI speech API on the slot's server
port.

```bash
llmctl download speech        # GGUFs, voice-design venv and model, two voices
llmctl start speech 5
curl -s localhost:8005/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"input": "Guten Morgen!", "voice": "frau"}' -o morgen.wav
```

- **API.** `POST /v1/audio/speech` takes `input` (up to 4096 characters),
  `voice`, `response_format` (`wav` or `pcm`: 24 kHz mono, 16 bit), `stream`
  and, as an extension, `language` (ISO 639-1: de, en, fr, es, it, pt, ru, zh,
  ja, ko). `GET /v1/audio/voices` lists the voices. `--lang` sets the slot's
  default (`de`); `--host` and `--clear-logs` apply too, nothing else.
- **Streaming.** As with OpenAI's API, the audio is sent as it is generated
  unless the request says `"stream": false` — the openai package expects that
  and sends no flag. A streamed `wav` has an unknown length in its header,
  which players read to the end, and `pcm` is raw. `"stream": false` returns one
  whole file, levelled to −20 dBFS. The text is spoken sentence by sentence, the first
  sentence on its own, so the first sound comes after that sentence: 1.0 s for
  a 16 s answer here, the rest generated at 1.7× real time, faster than it
  plays. Qwen3-TTS' vocoder works in windows of 72 frames (6 s), so a single
  long sentence arrives in 6 s pieces.
- **Engine.** Streaming needs llama-server with a `/tts` endpoint, which is
  llama.cpp PR #26603, not merged yet. `tts_server.py` starts it as a child on
  a private port that dies with it; the model stays loaded (5.0 GiB with one
  slot of 4096 tokens). Without it, llmctl falls back to `llama-tts`, run once
  per request: ~1.5 s of loading each time, no streaming, the model on the GPU
  only while it speaks. For comparison, the same model in PyTorch ran at 1.56×
  real time — slower than real time, bound by 90,000 kernel launches per
  sentence.
- **Building that llama-server.** No llama.cpp checkout is needed:

  ```bash
  /usr/local/share/llmctl/patches/build-tts-server.sh    # or patches/ in a checkout of this repo
  ```

  It uses `~/src/llama.cpp` (or `$LLAMA_SRC`) if that is a llama.cpp checkout
  and otherwise clones llama.cpp into `~/.local/share/llmctl/llama.cpp`. From
  there it makes a git worktree in `~/.local/share/llmctl/llama.cpp-tts` —
  where models.conf looks — at a pinned llama.cpp commit that is known to work,
  merges the pinned PR commit, applies a one-line fix for an API change since
  (`patches/llama.cpp-pr26603-mtmd-init-opt.patch`) and builds `llama-server`
  and `llama-tts` with Vulkan. Nothing is installed, and a checkout it uses is
  left as it is: the worktree shares only its git objects. It checks for git,
  cmake, a C++ compiler and glslc first. `LLAMA_BASE=HEAD` builds on the
  checkout's current commit instead, which may no longer merge with the PR. To
  rebuild, remove the worktree (`git worktree remove --force
  ~/.local/share/llmctl/llama.cpp-tts`) and run it again; once the PR is
  merged, the ordinary llama-server does all this.
- **Voices.** Qwen3-TTS Base speaks in the voice of a short reference recording,
  kept in `~/.local/share/llmctl/tts/voices/`, one file per voice, named by
  the file. The clone keeps the timbre and speaks accent-free German. Qwen3-TTS'
  own preset speakers (PyTorch CustomVoice) are English and Chinese speakers
  and keep their accent in German, so they are not used.

```bash
llmctl voice list
llmctl voice design erzaehler "Ein deutscher Muttersprachler um die sechzig, warme Stimme, ruhiges Erzähltempo, hochdeutsch ohne Akzent"
llmctl voice add ich ~/aufnahme.wav "meine eigene Stimme"
llmctl voice rm erzaehler
```

  `voice design` runs Qwen3-TTS VoiceDesign (PyTorch, in its own venv under
  `~/.local/share/llmctl/tts/venv`) once to speak a sample in the described
  voice, ~15 s; describe the speaker as a native speaker of the language, since
  the clone keeps any accent. `voice add` takes a recording of your own (wav,
  mp3 or flac); a few seconds of clean speech are enough.
- **Loudness.** A clone speaks as loud as its reference was recorded — a quiet
  recording measured 11 dB below the designed voices, and so did its clone.
  `voice add` and `voice design` therefore store every voice as mono 16-bit WAV
  at −20 dBFS RMS, and the speech server brings every answer to the same level
  (`--loudness`, −20 by default, `off` to disable) when it is not streamed.
  Both hold peaks below −1 dBFS, so a recording with strong plosives ends up a
  little quieter; a stream cannot be levelled, but clones of the stored voices
  land within a few dB of that level.

## Building blocks for a voice agent

llmctl does not contain an agent, but everything one needs to listen and speak:

| Step | Slot | Endpoint |
| --- | --- | --- |
| hear | `asr` (Qwen3-ASR 1.7B, `--mmproj --proxy`) | `POST :8086/v1/audio/transcriptions` |
| think | any LLM, e.g. `qwen-moe --proxy` | `POST :808N/v1/chat/completions` (`stream`) |
| speak | `speech` (Qwen3-TTS 1.7B) | `POST :8005/v1/audio/speech` (`stream`) |

`llmctl preset voice` starts the two speech slots (~10 GB together); add the
LLM to a preset of your own. The endpoints follow OpenAI's API, so its client
libraries work unchanged:

```python
from openai import OpenAI
asr = OpenAI(base_url="http://localhost:8086/v1", api_key="-")
tts = OpenAI(base_url="http://localhost:8005/v1", api_key="-")
text = asr.audio.transcriptions.create(model="asr", file=open("frage.wav", "rb")).text
with tts.audio.speech.with_streaming_response.create(
        model="qwen3-tts", voice="frau", input=answer, response_format="pcm") as r:
    for chunk in r.iter_bytes():          # 24 kHz mono s16le, as it is generated
        player.write(chunk)
```

- **Speech recognition.** Qwen3-ASR recognizes 30 languages including German
  and names the language. It writes "language German<asr_text>…" before the
  text; the proxy of the `asr` slot strips that and returns `language` as a
  field of its own (streamed: a `transcript.language` event first). Measured:
  a 7.6 s recording in 1.0 s, word for word. With `stream=true` the text comes
  as `transcript.text.delta` events. The recording has to be complete — cutting
  the microphone into utterances (voice activity detection) is the agent's job.
- **Latency.** From the end of an utterance: ~1 s to its text, then the LLM's
  time to its first sentence, then ~1 s to the first sound of that sentence.
  Feed the LLM's streamed answer to `/v1/audio/speech` sentence by sentence, or
  wait for the first sentence and send the rest as one request.
- **Round trip.** A German question spoken by a cloned voice and transcribed
  back came out identical, punctuation included.

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
leave the machine. The two TLS bases sit ten apart, so `--public` covers slots
1–9 — plain slots go far higher (see [Slots](#slots)). stunnel requires a client certificate from a private CA, so a
scanner hitting the port fails at the TLS handshake — before reaching any HTTP.

**Setup**

```bash
mkdir -p ~/.config/llmctl && (umask 077 && openssl rand -hex 32 > ~/.config/llmctl/tokens)
llmctl gen-certs llm-home.example.org        # SAN must be the name the VPS connects to
# copy ca.pem + vps-client-combined.pem to the VPS (the command prints the scp line)
# put deploy/vps-llm-vhost.conf on the VPS and adjust the two Define lines
# forward 8441 (and 8451 with --proxy) at the router to this machine

llmctl start qwen 1 --proxy --public
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

**Tokens** live in `~/.config/llmctl/tokens`, one per line, `#` comments allowed,
mode 0600 (enforced). Revoking one means deleting the line and restarting the
slot. `llmctl env` picks up the first token automatically, so the local
workflow is unchanged.

## Models

Order and names follow `models.conf`; `llmctl list` prints the same set. All of
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
- **magnum** — Magnum-v4-72B, Q6_K, 32K ctx; a Qwen2.5-72B fulltune, drafted by Qwen2.5-1.5B-Instruct Q4_K_M (~2.0×)
- **flash** — Qwen3.8-Flash-Next 125B MoE on the [halogen backend](#the-halogen-backend), 4-bit `.hgn`, 256K ctx (512K with YaRN via `--ctx 524288`), vision via `--mmproj` (on in preset `flash`); runs exclusively

Multimodal projectors are only loaded on an explicit `--mmproj`.

## Architecture

- `llmctl` — Main entry point for all commands
- `proxy.py` — optional Flask proxy (`start --proxy`) that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`. For halogen it also answers `/v1/messages` (`LLM_TRANSLATE_MESSAGES=1`), clamps token budgets (`LLM_MAX_TOKENS_CAP`) and waits longer (`LLM_PROXY_TIMEOUT`)
- `anthropic_compat.py` — the Messages ↔ Chat Completions translation `proxy.py` uses for backends without a Messages API
- `halogen_bench.py` — `bench` for halogen models: prefill and decode speed of a running slot
- `examples/models.conf` — Model definitions (paths, binaries, ROCm env vars); read from `~/.config/llmctl/`
- `examples/presets.conf` — Named configurations: which models run together, on which slots, with which flags (`llmctl preset <name>`)
- `templates/` — chat templates referenced from `models.conf` as `$SHARE_DIR/templates/…`
- `Makefile` — `install`, `install-link`, `uninstall`
- `deploy/vps-llm-vhost.conf` — Apache vhost for the VPS in front of `--public`
