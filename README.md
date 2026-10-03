# llmctl — LLM Server Manager

One command for a local AI machine (AMD Strix Halo): language models on
llama.cpp, halogen and gufo behind OpenAI- and Anthropic-compatible proxies (for
Claude Code and other clients), speech in and out, and pictures, video, music and
3D on ComfyUI with an image API. It starts, stops and combines them in slots and
presets, checks that they fit in memory, downloads models, measures them, and
tells what has a newer version.

## Installation & directories

```bash
sudo make install          # program to /usr/local (bin/llmctl, lib/llmctl, share/llmctl)
sudo make install-link     # instead: /usr/local/bin/llmctl → this checkout, edits apply at once
sudo make uninstall

pip install flask requests pillow # for proxy.py and the image API
pip install waitress       # optional, recommended for production proxy
```

The program is read-only; everything you own or the servers produce lives in
your home directory:

| Where | What | Override |
| --- | --- | --- |
| `~/.config/llmctl/` | `models.conf`, `presets.conf`, `tokens`, `tls/` | `LLMCTL_CONFIG_DIR` (tokens/tls also `LLM_CONF_DIR`, `LLM_TOKEN_FILE`) |
| `~/.local/share/llmctl/models/` | the GGUF files (`$MODELS_DIR` in `models.conf`), ComfyUI's models in `comfyui/` | `LLMCTL_MODELS_DIR`, or set `MODELS_DIR` in `models.conf` |
| `~/.local/share/llmctl/` | `llama.cpp/` (prebuilt llama-server, see below), `benchmarks/`, `claude/<model>[-<slot>]` (Claude Code profiles set by `env`), `comfyui/` (ComfyUI's checkout in `app/`, its workflows, input and output), `tts/` (voices; a voice-design venv only without ComfyUI) | `LLMCTL_DATA_DIR` |
| `~/.local/state/llmctl/` | `logs/`, `pids/`, `slots/`, `stunnel/` | `LLMCTL_STATE_DIR` |

**llama-server** itself: `llmctl download llama` installs prebuilt builds at
llama.cpp's newest stable release (a `vX.Y.Z` tag, not every nightly) into
`~/.local/share/llmctl/llama.cpp/` — `vulkan/` from ggml-org (its
`ubuntu-vulkan-x64` package of the build made of the release's own commit; it
needs only the system's Vulkan loader) and `rocm/` from
[lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)
(the package for this GPU's target, read from KFD — gfx1151 here — with its
ROCm runtime inside, so no system ROCm; the build is lemonade's oldest one at or
past the release commit). Each sits in `<kind>/<build>/`, `<kind>/current`
points at the one in use. A new build becomes current only after `--version`
names its commit and it loads the smallest llama.cpp model on disk and answers
`/health`; the one before stays for a rollback. `llmctl update llama` moves on
when a new release is out, `--check` tells first, `outdated` lists both.
`--master` (`download llama --master`, `update llama --master`) takes the
newest builds instead: ggml-org's newest bNNNNN with a Vulkan package and
lemonade's newest for the GPU target. Use
them in `models.conf`:

```bash
LLAMA_DEFAULT_BIN="$HOME/.local/share/llmctl/llama.cpp/vulkan/current/llama-server"
LLAMA_ROCM_BIN="$HOME/.local/share/llmctl/llama.cpp/rocm/current/llama-server"
```

Builds of your own (`/usr/local/bin`, `/opt`) are never touched. ggml-org's own
ROCm package is not used: it takes ROCm from the system.

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
llmctl download all           # all models
llmctl download all --check   # only tell what is missing, with sizes
```

`--check` fetches nothing and creates nothing. For each model it lists what a
download would bring — files from Hugging Face (asked of `hf download
--dry-run`), a halogen container image not pulled yet, and for ComfyUI the
checkout and venv, custom nodes not at their commit and every model a
workflow names that is not there (`download comfy klein --check`, too). It
ends with a verdict, and its exit status says the same: 0 nothing to
download, 1 something, 2 could not tell.

## Usage

```bash
llmctl list                            # Show available models
llmctl start <name> [slot] [--proxy] # Start server in background (any slot, default 1)
llmctl presets                         # Show the defined presets
llmctl preset <name>                   # Bring the machine to a whole configuration (see below)
llmctl preset-save <name>              # Record what is running now as a preset
llmctl stop [slot]                     # Stop slot, or all if omitted
llmctl service enable <preset>         # Run a preset at login, restart dead slots
llmctl status                          # Running state, model and key parameters (all slots)
llmctl cache-stats [slot]              # Prompt-cache hit rate, read from the server log
llmctl logs [slot] [-f] [--proxy|--tls] # A slot's log (last 50 lines, -n N); no slot: all logs
llmctl clear-kv [slot]                 # Drop the KV cache without restarting
llmctl probe-reasoning [model]         # What each model's chat template supports
llmctl gen-certs <host>                # CA + server/VPS certificates for --public
llmctl bench [--full] <model|all>      # Run benchmark (default: default ROCm + Vulkan)
llmctl bench --full all                # Full test: all 8 ROCm combos + Vulkan
llmctl bench flash                     # halogen: prefill + decode of the running slot
llmctl env <name> [slot]               # Set Claude Code env vars
llmctl clear                           # Clear env vars
llmctl download <model>                # Download model(s); --check: only say what is missing
llmctl prune [--dry-run]               # Models nothing names any more; deletes after a typed yes
llmctl outdated [--quick]              # What has a newer version (images, builds, ComfyUI, models)
llmctl version                         # Print the version
```

`make install` also installs completion for bash and zsh: commands, model
and preset names, slots, voices and option values complete with Tab. zsh picks
it up from `/usr/local/share/zsh/site-functions` once `compinit` runs (a new
shell; `rm ~/.zcompdump*` if an old cache hides it), bash through the
bash-completion package.

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

### A preset as a service

```bash
llmctl service enable voice     # at every login: llmctl preset voice
llmctl service start            # now, without logging in again
llmctl service status           # preset, state, next repair, last log lines
llmctl service stop             # stop the slots the service holds
llmctl service disable          # no more at login; what runs keeps running
```

`enable` writes three systemd user units to `~/.config/systemd/user/`:
`llmctl.service` runs the preset (and stops its slots at logout),
`llmctl-repair.timer` looks every two minutes for a slot whose server died and
starts it again. The repair is `llmctl preset <name> --repair`, which only
restarts slots this preset still holds: `start`, `stop` and another preset
take a slot away from it, so a machine switched to something else by hand is
never switched back behind your back. It stops nothing.

- The servers run inside the service (systemd's cgroup for it), the repaired
  ones too — the timer asks for them through `systemctl reload`, not as a
  process of its own that systemd would clean up after.
- A user service starts at login. For boot, before anyone logs in:
  `loginctl enable-linger $USER`.
- `enable` records the `PATH` and any `LLMCTL_*_DIR` of the shell it runs in,
  since a service does not see your shell's environment; run it again after
  changing those.
- Use `llmctl service start|stop` rather than `systemctl` — an alias such as
  `systemctl='sudo systemctl'` would send `--user` to root's service manager.

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

### A server that dies while starting

`start` watches the new server for 20 s (`LLMCTL_START_WATCH`, in seconds; 0
turns it off). One that exits in that time — a model or vision projector it
cannot load, memory it cannot get — makes `start` fail with the lines of its log
that say why, and the slot is cleared, rather than "Server running" over a
process that is gone and that only `status` or the log would show. One that
answers in that time is reported up; one still loading is left to load. `preset`
already waits for each slot and rolls back a half-started preset; it now hears
of such a failure at once instead of after its wait.

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
reading its own `.hgn` weights. The same backend also runs
[halogen-server](#halogen-server-qwen38-27b), its sibling for the dense
Qwen3.8-27B. The slot model stays the same — `start`, `stop`,
`status`, `env`, `preset`, `preset-save` all work — with these differences:

- **It runs in podman.** The server PID is the `podman run` client (its argv is
  the full command, which is what `preset` compares), the container's output is
  the server log, and the container is named `llmctl-halogen-<slot>`. `stop` uses
  `podman stop`, which waits for the container to exit and removes it.
- **Other slots beside it, when they fit.** It pins ~62 GiB of weights and
  reserves ~20 GiB of KV pool and working memory, ~82 GiB in all, which leaves
  ~20 GiB of the GTT pool for other slots — the voice slots (~11 GiB) or
  embeddings and a reranker (~14 GiB), not a second big LLM. While halogen runs
  or starts, `start` weighs what the new slot takes (measured, else estimated)
  plus what runs against the GTT pool minus what programs outside llmctl hold
  and 2 GiB, and refuses what does not fit, with the numbers; `--ignore-memory`
  starts it anyway. halogen's pinned weights count, though they lie outside the
  pool: the pool is carved from the same RAM, and what lies beyond it is the
  system's and the page cache's, through which the n-gram table is read. A
  preset with halogen is weighed the same way as a whole and not started when
  it does not fit (without halogen an overcommitted preset still only warns).
  Measured with flash, embeddings, a reranker and the speech slot together:
  11.9 GiB really available, no swap on the SSD, all four answering. Start
  halogen first where you can: it wants contiguous 2 MiB blocks, and memory
  others have fragmented makes its startup compact.
- **Its proxy always runs.** Since 0.15.1 the server speaks the Anthropic
  Messages API itself — thinking blocks replayed exactly through their
  signature, `cache_control` as a resume point for the prompt cache — so the
  proxy passes `/v1/messages` through. Its own translation into Chat
  Completions (`anthropic_compat.py`: system, text, images, tool use and
  results, thinking, streaming with pings during prefill) is still there for an
  older image: `LLMCTL_HALOGEN_TRANSLATE=1` when starting. The proxy clamps token budgets to
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
- **`HALOGEN_CACHE_BRANCHES`** (0.13.3, default 2, 3 in models.conf here) is
  how many branches of one conversation the prompt cache keeps resume points
  for. Two is a conversation and one side turn; three is a parent forked into
  two children, which is what Claude Code running subagents does — on two, the
  parent's resume point is the one dropped. It costs ~0.8 GiB of host memory
  and raises the entry cap with it (16 entries here, ~111 MiB each).
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

**`HALOGEN_WEIGHTS_LOCK=1` (0.13.2) is deliberately not set.** Measured against
a co-tenant growing in 5 GiB steps, with it off: prefill 1084 tok/s alone, 947
at 10 GiB, and at 15 GiB the engine wedged — no answer for 180 s, then its own
watchdog took it down (the failure of issues #83 and #85, reproduced). With it
on, prefill held (1179 tok/s at 15 GiB) and the co-tenant was killed instead.
It stays off because of what each costs the disk: the weights are clean file
pages, so reclaiming them writes nothing and costs re-reads, while locking them
leaves only anonymous memory to reclaim, which goes to swap — on this host a
file on the SSD, which filled during that test. A killed server starts again;
SSD writes do not come back. Other slots start beside it only when they fit,
so the pressure can only come from programs outside llmctl. `status` follows the setting: with
the weights locked, `MemAvailable` has already dropped by them and subtracting
them again would count them twice.

**v2 against w4b** (0.15.1 with v2 against 0.14.2 with w4b and its sidecar,
two runs each, one session):

| | 0.14.2 + w4b | 0.15.1 + v2 |
| --- | --- | --- |
| halogen's own count | 96.3 GiB (68.0 weights, 7.2 pool, 21.1 working) | **81.7 GiB** (62.1, 7.2, 12.4) |
| GTT (GPU) | 34.7 GiB | **19.9 GiB** |
| `MemAvailable` drop on loading | 34.7 GiB | **19.1 GiB** |
| really available (`status`) | 8.4 GiB | **30.6 GiB** |
| prefill 850 / 6.6k / 26k tokens | 981–991 / 1344–1401 / 1351–1354 t/s | 1049–1051 / 1431–1435 / 1389–1396 t/s |
| decode prose / code | 29.5–37.5 / 46.9–53.9 t/s | 32.8–38.4 / 51.8–53.2 t/s |
| first streamed token | 0.6–0.7 s | 0.3 s |

14.6 GiB less memory — 5.9 from the checkpoint, 8.7 from 0.15.1's working
memory fit — for the same speed: prefill 3–7% faster, decode within the spread
between runs, which is wide (the first w4b run, with host memory fragmented
after a 118 GB download, decoded prose at 29.5). The two checkpoints are
different weights, so their greedy texts differ and so do their drafts.

**The host matters more than for llama-server.** Measured on 0.14.2 with w4b:
prefill 1310–1370 t/s (6.6k–26k tokens) and decode 39 / 59 t/s after a fresh
boot with `amd_iommu=off amdgpu.noretry=0` on the kernel line; the same server
with fragmented memory managed 8–370 t/s of prefill, most requests stalling
40–200 s in kernel compaction. Keep `vm.compaction_proactiveness` at the kernel
default (20) — 0 does not prevent the stalls.

### halogen-server (Qwen3.8-27B)

`qwen-halogen` runs [halogen-server](https://github.com/peonist-ai/halogen-server)
(image `ghcr.io/peonist-ai/halogen`) with
[peonist-ai/halogen-qwen3.8-27b](https://huggingface.co/peonist-ai/halogen-qwen3.8-27b):
the stock (not uncensored) Qwen3.8-27B, FFN in 4 bit for prefill (unsloth's
NVFP4 values), the rest FP8, with an MTP head and a DFlash2 block drafter. Text
only. llmctl tells it from halogen-flash-server by the image and handles it like
the above, except:

| Option | halogen-server |
| --- | --- |
| `--ctx N` | `HALOGEN_SLOT_CTX`, at most 262144 (no YaRN) |
| `--parallel N` | `HALOGEN_KV_SLOTS`; each slot holds its full context (64 KiB per position), and more than one turns speculation off |
| `--cache-ram N` | `HALOGEN_CACHE_MB` (MiB, 0 = off): the prompt cache, in host memory at ~70 KB per token. Unset, the server takes most of what is free; models.conf sets 20000 |
| `--reasoning off\|level`, `--temp`, `--top-p` | request defaults: the server has no setting for them, so the container carries them (label `llmctl.defaults`) and the proxy fills them into chat requests that set none themselves. A thinking budget does not exist. Without, the template thinks at effort xhigh |
| `--mmproj` | refused |

The checkpoint's `tokenizer/` is mounted as `/tokenizer`. The server has no
Messages API, so the proxy translates `/v1/messages`, and since it ignores
`thinking`, Claude Code's "thinking off" goes out as `reasoning_effort: none`.
The default drafter is DFlash2 (`HALOGEN_DRAFTER=2`), which suits code; a
request may pick another (`"drafter"`). `status` shows GPU memory and the
container's RAM (the mapped checkpoint and the prompt cache); `outdated` reads
the changelog of halogen-server.

Measured against `qwen` (llama.cpp) and `qwen-gufo` (both HauhauCS Q8), greedy,
thinking off:

| | qwen-halogen | qwen-gufo | qwen |
| --- | --- | --- | --- |
| prefill 850 / 6.6K / 26K / 53K tokens | **690 / 677 / 580 / 491** | 380 / 378 / 366 / 343 | 298 / 283 / 245 / 204 |
| decode prose / code (t/s) | 14.6 / **45.3** (DFlash2), **17.0** / 38.1 (MTP) | 15.3 / 37.2 | 11.9 / 18.7 |
| memory | 20.5 GiB GPU + 33.4 GiB checkpoint + prompt cache | 44.1 GiB GPU + 31.4 GiB RAM | ~47 GiB |

## The gufo backend

[gufo](https://github.com/gufo-org/gufo) (MIT) is an inference engine written
for Strix Halo alone, shipped as a container like halogen. Where halogen runs one
checkpoint of its own, gufo reads ordinary GGUFs — here the same HauhauCS
Qwen3.8-27B Q8 that `qwen` runs on llama.cpp, as `qwen-gufo`, with z-lab's
DFlash2 draft (1.1 GB) for speculation.

```bash
llmctl download qwen-gufo            # the GGUF, the draft and the image
llmctl start qwen-gufo 1 --mmproj    # or: llmctl preset qwen-gufo / qwen-gufo-256k
eval "$(llmctl env qwen-gufo 1)"
```

**Measured** against `qwen` on llama.cpp, same GGUF, same bench
(`halogen_bench.py`, which also measures any OpenAI server by `--url`), greedy:

| | prefill 850 / 6.6K / 26K tokens | decode prose | decode code | first token |
| --- | --- | --- | --- | --- |
| llama.cpp `qwen` (MTP head) | 312 / 286 / 247 t/s | 11.8 t/s | 19.0 t/s | 0.6–1.0 s |
| gufo, no draft | 402 / 406 / 381 | 7.0 | 7.0 | 0.38 s |
| **gufo + DFlash2** (`qwen-gufo`) | 389 / 406 / 379 | 15.3 | 37.4 | 0.39 s |

Prefill 30–55 % faster, code decoding twice as fast; without a draft gufo
decodes at 7 t/s, the bandwidth limit for 29 GiB of Q8. unsloth's UD-Q8_K_XL,
gufo's own target, ran no faster than HauhauCS's Q8_K_P. gufo runs MTP for
Flash-Next only, so the 27B speculates with DFlash2.

What llmctl does for it:

- **The container**: `podman run` as for halogen (`llmctl-gufo-<slot>`, the model
  directory read-only on `/models`, the draft's on `/draft`), with
  `--security-opt label=disable` — under SELinux it could otherwise neither
  read the mounted models nor map `/dev/kfd`. That concerns this container
  only; no `setsebool` for the whole host.
- **Settings are flags** of `gufo serve llm`: models.conf's extra_args hold
  them (`--temperature 0.6 …`), `--ctx` becomes `--context` (per session),
  `--parallel` `--sessions` (2 by default), `--reasoning off|on|<level>`
  `--think` / `--reasoning-effort`, `--temp`, `--top-p` and `--max-predict`
  replace the entry's values, `--spec off` drops the draft, `--verbose`
  logs at debug. `--served-model-name` is the entry's name: gufo refuses a
  request for any other model, and `llmctl env` hands Claude Code exactly
  that one.
- **Its proxy always runs and translates `/v1/messages`** to chat completions.
  gufo 0.4.0's own Messages API returns the model's thinking inside the text
  block, `</think>` and all; its chat completions split it off, and the
  translation turns that into a proper thinking block. `LLMCTL_GUFO_NATIVE_MESSAGES=1`
  passes the Messages API through instead, once gufo fixes it.
- **Vision with `--mmproj`**: gufo looks for `mmproj-BF16.gguf` beside the
  model and does not find HauhauCS's name on its own, so llmctl passes it;
  it is uploaded at the first image.
- **Memory**: the weights sit in the container's RAM (31 GiB), the KV of every
  session in GTT, reserved at start — 41 / 45 / 54 / 68 GiB in all at
  32K / 64K / 128K / 256K per session, two sessions. The memory check counts
  both, as it counts halogen's pinned weights: `flash` (84 GiB) is refused
  beside it, ComfyUI fits. gufo also keeps conversation snapshots in RAM for
  its prompt cache, sized by itself from what is free at start (~80 GB at
  most); 45 GiB at start grew to 50 GiB after a bench and an image.
- `status`, `stop`, `logs`, `bench`, `preset`, `preset-save`, `prune`,
  `download` and `outdated` know it; `cache-stats` and `clear-kv` have nothing
  to ask it. `--public` is not wired up for gufo yet.
- **Updates**: `outdated` asks the registry for newer tags of the image and
  shows the version headings of gufo's changelog in between, as for halogen.
  gufo releases nearly every day; move the tag in models.conf after a bench,
  and `download` pulls it and offers to remove the old one.

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
llmctl update comfy                   # to the newest release, Python dependencies, new workflows
llmctl update comfy --torch           # …and torch itself
llmctl update comfy --master          # master instead of the releases (download comfy --master too)
```

- **Releases.** ComfyUI tags a release about once a week (vX.Y.0 on master,
  patch releases vX.Y.1, .2 … with backported fixes on a `release/vX.Y`
  branch). The checkout follows the newest tag: `download` checks it out,
  `update` moves onto a newer one, and `outdated` counts ComfyUI as current
  when it is on the newest tag or past it — not every commit on master.
  `update --master` follows master instead, for a fix or template that is
  only there (`download comfy --master` sets a new one up on master); a
  checkout on master is reported with its version (`v0.38.0+28, master`) and
  how far origin/master is ahead. The custom node packs stay on the commits
  `comfyui/custom_nodes.txt` pins, unless a pack tags a release made after
  its pin — `update` takes that and `outdated` reports it; new commits are
  only shown. (Most packs tag rarely: Impact-Pack's newest tag, 8.28, lies
  8 commits behind its pin, rgthree's 407.)
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
- **Steering Qwen-Image 2.1: ControlNet and inpainting.** alibaba-pai's
  [Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)
  (7.6 GB, `model_patches/`) is one model for eight kinds of control image —
  canny, depth, pose, scribble, lineart, HED, MLSD, grayscale — and for
  inpainting. ComfyUI loads it as a model patch ("Load Model Patch" + "Apply Fun
  ControlNet"). One workflow, `Qwen-Image 2.1 Control`, whose *Steuerung*
  picks the kind and whose *Nur in der Maske* repaints only the masked part;
  as the image API's models (and the workflows they were):
  - `Qwen-Image 2.1 Control`: a control image of any of those kinds, the
    prompt paints the rest. Strength 0.5 by default: at 1.0 a scribble came
    back as flat vector art whatever the prompt asked, at 0.5 as the asked-for
    photograph on the scribble's layout. Raise it for a pose or depth map that
    has to be followed closely.
  - `Qwen-Image 2.1 Canny Control`: a photo's edges (ComfyUI's own Canny) as
    the control — a new picture on the old one's layout, e.g. as a watercolour.
  - `Qwen-Image 2.1 Pose Control`: the body pose of a person in a photo
    (DWPose) — someone else, somewhere else, standing the same way. Tried: a
    man waving in a park became a dancer on a stage, arm raised and hand on
    the hip exactly as his.
  - `Qwen-Image 2.1 Depth Control`: the spatial layout of a photo (Depth
    Anything V2 Large) — the same room or landscape in other materials and
    style, freer than Canny, which holds every edge. Tried: a modern living
    room became an alpine cabin, sofa, table, lamp, window and shelves where
    they were.
  - `Qwen-Image 2.1 Canny/Pose/Depth Inpaint`: repaint only inside the mask,
    while the whole photo's edges, pose or depth steer what is painted there
    (the control branch takes the control image and the mask at once). Tried
    against plain Inpaint, Turbo, two seeds each:
    - **Canny keeps the cut**: a black coat masked and asked to be red
      leather stayed the same coat — buttons, collar, folds — in red leather;
      plain Inpaint made a different jacket, a zipped biker one.
    - **Depth keeps the shape**: a tablecloth masked and asked to be a wooden
      table kept the cloth's drape, now dark; plain Inpaint made a wooden
      table (and once left a white rest). Use it to change a material, not a
      thing.
    - **Pose adds little** where the person stays: a cyclist in a new rain
      jacket sat the same way with plain Inpaint, which follows what is
      around the mask. It is for putting someone else into the pose.

    Turbo 25 s each; through the image API as
    `qwen-image-21-{canny,pose,depth}-inpaint[-turbo]` with `mask` or `boxes`.
  - `Qwen-Image 2.1 Colorize`: a black-and-white photo in colour. Its
    luminance (Image Luminance, from comfyui_controlnet_aux) is the control,
    so every line and texture stays; the prompt says what colours to give —
    the default asks for natural colour film, name the dress, car or sky
    for more. Strength 1.0, the default, is the faithful one: faces, ages
    and grey hair kept, colours a little reserved; at 0.8 livelier and
    mostly true, at 0.6 it invents (a bouquet in a 1950s family's hands,
    pink on a grey cat). Tried on four pictures turned grey, whose real
    colours were known, and an old family photograph: skin, sky, trees,
    jeans and the dog's coat came out close to the originals. Turbo 25 s,
    14 steps 79 s, alike enough that Turbo is the one to use.
    It colours what is there and mends nothing: a face already smeared in
    the photo comes back smeared, in colour. The Detailer does not save one
    either — on a face 80 px tall it repaints the smear (at 0.75 a hint of
    eyes, while the sharp faces around turn into other people). Inpaint
    does: a box over the face and a description ("a smiling middle-aged
    woman with light wavy hair") gave a clear face in 25 s — a new one,
    plausible, a little sharper than the grainy photo around it.
  - `Qwen-Image 2.1 Outpaint`: extend a picture beyond its edges. The canvas
    is padded (256 px left and right by default, set per side on the "New
    border" node) and the new border inpainted; the prompt describes the whole
    wider scene. The picture is scaled to ~0.8 MP first, so the result is
    ~1.3 MP (1424×912 from a square). Tried: a bench cut off at the left edge
    was completed, with more trees, lamps and benches around; a man in a park
    got more park on both sides; no visible seam. 112–124 s, Turbo 36 s.
  - `Qwen-Image 2.1 Inpaint`: paint the mask over what to redraw (the image
    node's mask editor); the prompt describes the whole picture, and only the
    masked part changes — for **replacing or adding** something (a cap
    swapped, a figure put in). It is not for removing: it fills the mask with
    an object of the mask's shape. A backpack masked on a bench came back as a
    wooden box and, with a tight mask and a prompt describing the bench's
    slats, as a grey seat shell. The Fun patch blanks the masked pixels itself
    (grey), so the shape comes from the mask, not from what was there. To
    **remove** something, use an Edit workflow ("Entferne den roten Rucksack,
    sonst nichts ändern"): Qwen-Image 2.1 Edit (97 s, 47 s as Turbo) and FLUX.2
    klein Edit (55 s) both took the backpack away and ran the bench's slats
    through; outside it they shift the picture slightly (7.7/255 on average,
    not visible), where Inpaint keeps it exact (3.8/255).

  Pose and depth come from the custom node pack
  [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux),
  installed like the others (`comfyui/custom_nodes.txt`, without its CUDA
  onnxruntime and mediapipe); each workflow shows the map it made as a
  preview. DWPose runs its torchscript models, on the GPU. Its detector models
  (1.6 GB) come from Hugging Face on first use into the pack's own `ckpts/`.
  Depth Anything V2 Large is licensed CC-BY-NC — fine at home, not for
  commercial work; `depth_anything_v2_vits.pth` (Apache-2.0, coarser) is one
  dropdown away. There is no official template for 2.1 yet; `comfyui/build_control_workflows.py` writes
  the graphs in API format and has ComfyUI's frontend turn them into UI
  workflows. Measured: ~80–95 s per 1024² image. Each has a **Turbo** variant
  (`… Turbo (bf16, 4 Schritte)`) with the 4-step Acc LoRA below: 25–38 s,
  control and quality held — pose, depth, canny and inpaint came out as good
  as at 14 steps; the scribble's cabin a little busier.
- **Detailer.** `Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14)` finds faces and
  hands (YOLO detectors from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer)),
  outlines each with SAM ViT-H, crops it, repaints it at up to 1024 px with
  Qwen-Image 2.1 and pastes it back — so a small hand gets many more pixels
  than in the whole picture. Faces at denoise 0.45, hands at 0.6. It runs on
  the custom node packs [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
  (GPL-3.0) and its Subpack (AGPL-3.0, with ultralytics), installed like the
  others; SAM 2 (a CUDA extension built from git) is left out, SAM 1 is pure
  Python. The detectors take 6–50 MB, SAM ViT-H 2.5 GB. Through the image API
  as `qwen-image-21-detailer`; the request's prompt is ignored, the detailer
  has its own (nodes titled "(fixed)").

  Tried on seven pictures, the machine to itself, 55 s for one hand or face,
  ~100 s for two, 6 min for a crowd:
  - **Hands at 0.6 get their shape mended**: a hand around a cup, two on
    piano keys, a fretting hand, hands on a bicycle chain came out better
    formed. At 0.45 the detailer only adds detail (wrinkles, nails) to the
    same shape.
  - **Faces drift.** On a clean portrait, brown eyes turned grey-green at
    0.6, 0.45 and 0.35 alike, with or without "eyes" in its prompt; the
    face is otherwise the same person. For a picture whose faces are fine,
    bypass the face pass (Ctrl+B on "Gesichter nachbessern").
  - **It is no rescue**: a crowd of melted faces stays melted — at 0.6 it
    grew new ones. Use a new seed for those.
  - **SAM keeps the surroundings**: without it 1.4–18 % of the picture
    changed, with it 0.5–12 % — piano keys around the hands stayed as
    they were. It costs 5–10 % more time.
  - **No Turbo**: with the 4-step LoRA skin came out coarse and aged, at 0.45
    and at 0.6, where 14 steps keep it natural.
  - Missed: hands knitting, partly hidden behind the needles and the wool.
- **Background removal.** `Qwen-Image 2.1 Background Removal`
  is ComfyUI's own template on the bf16 models already here: an edit with
  the instruction "Remove the background, and output a PNG image", and
  Qwen-Image 2.1 answers with a real alpha channel. Tried: a man in a park cut
  out cleanly to the fingertips; in a living room it chose the sofa and the
  coffee table as the subject and dropped the room. 150–165 s per image (25
  steps); the Turbo variant (4 steps with the Acc LoRA) 47 s, as clean and as
  transparent. Through the image API as `qwen-image-21-background-removal`: the
  prompt is the instruction, so send that sentence; the PNG comes back with
  transparency.
- **BiRefNet background removal** (#34). `BiRefNet Background Removal`
  (general; *Feine Kanten* switches to HR matting) runs
  ZhengPeng7's BiRefNet through
  [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG): no diffusion, the
  photo's own pixels with an alpha channel, in 2–3.5 s instead of 52–65 s
  with Qwen-Image Turbo. 885 MB (general) and 445 MB (HR matting), from
  1038lab/BiRefNet into `models/comfyui/RMBG/BiRefNet/`. Tried on six
  pictures against Qwen-Image 2.1 Background Removal Turbo:
  - Qwen dropped the woman on the street entirely, made the coat of a woman
    with curly hair half transparent, left out the carafe beside a wine glass
    and **redrew** a bicycle's spokes — it repaints, BiRefNet cuts out.
  - *general* keeps solid things solid: both glasses whole, the dog with
    some grass at its paws, flyaway hair with a few specks of background.
  - *HR matting* is the one for fine edges: hair strands with soft alpha,
    the spokes clean and thin (general left them dark with the brick wall
    in them), fur without the grass — and glass truly transparent, so a
    carafe nearly vanishes.

  ComfyUI-RMBG's `RMBG-2.0` model looked the same as *general* but is
  BRIA's, non-commercial; BiRefNet is MIT. The pack also brings face,
  clothes and body segmentation, SAM 2/3 and GroundingDINO masks, not used
  by a workflow here yet. Image API: `birefnet-background-removal`,
  `birefnet-matting-background-removal` — the prompt is ignored, the PNG
  comes back with transparency.
- **Text to mask (SAM 3) and captions (Florence-2)** (#36, #37). Two helper
  workflows the image API runs itself, usable in the UI as well:
  `SAM 3 Select` (Meta's SAM 3.1 through ComfyUI-RMBG's SAM3Segment, 1.7 GB
  from 1038lab's ungated repack, SAM License) outlines what a few words name,
  one mask per match, in ~5 s with loading; `Florence-2 Caption`
  ([kijai's ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2),
  Florence-2 large, MIT, 1.5 GB) describes a picture in detail in 1–3 s.
  Masks from words, tried on six pictures:

  | words | Florence-2 | SAM 3 |
  |---|---|---|
  | the sofa | the big sofa | both sofas, cushions left out |
  | the floor lamp | ✓ | ✓ |
  | the woman | her head only | all of her, exactly |
  | the guitar | ✓ | ✓ |
  | the left hand | the right hand | both hands |
  | hair | patchy, face included | clean |

  So SAM 3 makes the masks and Florence-2 the captions; GroundingDINO +
  SAM 2 (also in ComfyUI-RMBG) failed on its first call here. Neither knows
  left from right — a box drawn by hand does.
- **Z-Image Turbo.** `Z-Image Turbo T2I (bf16, 8 Schritte)` is ComfyUI's own
  template for Tongyi-MAI's Z-Image-Turbo (6B, Apache-2.0): photorealistic,
  8 steps, text-to-image only. 20.7 GB of models (12.3 the DiT, 8.0 the
  Qwen3-4B encoder, 0.3 the VAE). 18 s per 1024² image, 21 s with loading —
  on a par with Qwen-Image 2.1 Turbo, with a look of its own (a lighthouse,
  an old fisherman, a bookshop, a spice market came out clean, faces and
  hands included). **It runs beside flash**: 19–20 s per image there, no
  swap to the SSD, but it leaves flash 11.5 GiB instead of 31 for the page
  cache its n-gram table is read through — prefill held (1036/1385/1396 t/s
  at 850/6.6K/26K tokens), decode fell 20–30 % (prose 31.5 t/s, code 44)
  while ComfyUI kept the model loaded. Image API: `z-image-turbo`.
- **Qwen-Image Layered.** `Qwen-Image Layered (bf16, 20 Schritte)` is ComfyUI's
  template for Qwen's Qwen-Image-Layered (20B, Apache-2.0): it splits a
  picture into RGBA layers — the background with what stood in front of it
  filled in, and each thing on a layer of its own with transparency, to move,
  recolour, replace or reuse without touching the rest. 50 GB of models (40.9
  the DiT, 9 the Qwen2.5-VL-7B encoder it needs, 0.3 the VAE). It works at 640
  px; four layers take ~6.5 min, three ~5.5. The prompt describes the picture.
  Tried, four layers each:
  - a concert poster: the dark background, the stars and small print, the
    title complete — also where the guitar had covered it — and the guitar
    cut out;
  - a living room: the room without lamp, armchair and sofa (wall and floor
    behind them filled in), then each of the three on its own;
  - a woman on a street: the street without her, completed, and her cut out
    cleanly — but layers three and four were a white blob and her coat once
    more. **More layers than the scene holds give junk**: two suit a person in
    front of a scene, four a poster or a room.

  The template makes two; the image API takes `layers` (1–8) and answers with
  one transparent PNG per layer, background first. The model wants the
  picture described: **sent without a prompt, the image API has Florence-2
  describe it** and uses that (a woman on a street, two layers: the street
  without her and her cut out, 262 s with the caption). In the UI the
  workflow `Florence-2 Caption` gives the text to paste. Image API:
  `qwen-image-layered`.
- **Qwen-Image Layered Control.** `Qwen-Image Layered Control (bf16, 20
  Schritte)` is the sibling that pulls **one named thing** out of a picture:
  the prompt names it ("the red electric guitar", "the woman"), the first
  layer comes back as that thing alone on transparency. ComfyUI's template
  (`image_qwen_image_layered_control`) loads the DiT as fp8; here it reads
  the bf16 file (`qwen_image_layered_control_bf16.safetensors`, another 40.9
  GB, DiffSynth-Studio, Apache-2.0) with the same encoder and VAE. At 640 px
  with two layers a run takes ~4 min. Tried: the poster's title text, a
  sofa and a floor lamp in a living room, the woman on the street — each cut
  out cleanly; the guitar on the poster came back only at 1024 px (~13 min).
  The second layer is mostly junk — take the first. One layer alone is not
  the cheaper way: at 1024 px it gave the guitar but an empty layer for the
  woman. The model card names posters and graphics as its strength. Image
  API: `qwen-image-layered-control`, the prompt is the thing to extract,
  `layers` as above.
- **Z-Image Turbo 2K Upscale.** `Z-Image Turbo 2K Upscale (bf16, 5 Schritte)` is
  ComfyUI's own template: RealESRGAN ×4 (67 MB) on the picture scaled to 1 MP,
  halved, then a light Z-Image Turbo pass (denoise 0.33) that adds detail — a
  1024² picture comes back at 2048² in ~106 s. Image API:
  `z-image-turbo-2k-upscale`; it keeps its own prompt ("masterpiece, 8k"), as
  the request's is usually a placeholder. Compared on three pictures shrunk to
  384 px, against the originals: it stays faithful and sharp; SeedVR2 is the
  most faithful and the fastest (30 s, but ×4 from the input: 1536 px here);
  alibaba-pai's Z-Image **Tile ControlNet** (the same Union 2.1 repo, 6.7 GB),
  repainting at 2048² from the enlarged picture, invented the most — another
  face with new wrinkles, a wall in a different wood — in 160 s, so it is not
  a workflow here.
- **Z-Image Turbo NSFW.** `Z-Image Turbo T2I` with *Unzensiert* runs Turbo on tewea's NSFW finetune of its DiT
  (`z_image_turbo_bf16_nsfw_v2`, 12.6 GB, Apache-2.0, pinned; its model card
  says nothing about the training). Side by side with the original, same
  seed: a fox and a fisherman came out as good — the finetune costs no general
  quality — and on two artistic nudes the original already rendered nudity;
  the finetune showed more, in other poses. Where it differs is for prompts
  the original avoids: judge it on your own. An abliterated Qwen3-4B encoder
  (BennyDaBall's, same tensors as Comfy-Org's) changed next to nothing on top,
  as on klein — an image model's encoder translates the prompt, it does not
  refuse — so the workflow keeps the stock one. Image API: `z-image-turbo-nsfw`.
- **Z-Image, the base model.** `Z-Image T2I (bf16, 25 Schritte)` is ComfyUI's
  template for Tongyi-MAI's undistilled Z-Image: 25 steps at cfg 4, so a
  real negative prompt — the image API takes it as `negative_prompt` (only
  for workflows whose guidance uses one). Only the DiT is new (12.3 GB); the
  encoder and VAE are Turbo's. 106–109 s an image, six times Turbo's 18–19 s.
  Against Turbo, same prompts and seeds:
  - **Variety**: four seeds of "a woman sitting in a café" gave four
    different women, clothes and cafés; Turbo gave the same young woman in the
    same jumper four times.
  - **Style**: "Van Gogh: a harbour at night" painted a harbour at night in
    his brushwork; Turbo copied *The Starry Night*, cypress and all.
  - **Negative prompt**: "cars, traffic" turned a car-filled street into a
    busy pedestrian one, a few cars left far off.
  - Fox and fisherman as good (Turbo a little more polished); the shop sign
    misspelt by both ("Sonnensschin", "Sonnenschen").

  So: Turbo for speed, the base model when variety, a style or a negative
  prompt matter. Image API: `z-image`.
- **Z-Image Turbo with ControlNet.** `Z-Image Turbo Control / Canny Control /
  Pose Control / Depth Control / Inpaint / Colorize (bf16, 8 Schritte)` are
  the Qwen-Image 2.1 control workflows on Z-Image Turbo, with alibaba-pai's
  [Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)
  (2602, 8-step, 6.7 GB, Apache-2.0: canny, depth, pose, MLSD, HED,
  scribble, gray, and inpaint) through the same node. Side by side with the
  Qwen-Image Turbo variants on the same inputs and seed: pose (a man waving
  in a park → a dancer in red, arm, hand on hip and stance exact), depth (a
  living room → an alpine cabin), inpaint (a cake on an empty plate) and
  colourising came out as good, crisper and more photographic; ~28 s each
  warm. **Canny needs a lower strength**: at 1.0 it kept the photo whatever
  the prompt asked, at 0.65–0.8 it painted the asked watercolour on the
  photo's layout, at 0.5 it lost the layout — the workflow uses 0.65, the
  others 1.0 (control 0.8). Image API: `z-image-turbo-{control,canny-control,
  pose-control,depth-control,inpaint,colorize}`; `strength`, and `mask` or
  `boxes` for inpaint, as for Qwen-Image.
- **Qwen-Image 2.1 in 4 steps.** `Qwen-Image 2.1 [Heretic] T2I/Edit Turbo (bf16,
  4 Schritte)` add alibaba-pai's official 4-step Acc LoRA and sample 4 steps
  with `euler`: a 1024² image in 16 s instead of 70, an edit in 47 s instead
  of 77, with a composition of its own for a given seed. The LoRA is published
  for diffusers only — its keys lack `.weight`, each block's MLP has separate
  `gate_layer`/`proj` where ComfyUI's model fuses them into `gate_up`, and it
  carries one `proj_out` per sampling step — so ComfyUI loads nothing of it
  and the result is the bare model at 4 steps, a smear. `download comfy`
  converts it (`comfyui/sources.json`, `convert_qwenimage21_lora`): modules
  renamed, each MLP pair stacked into one LoRA on `gate_up` (exact), `proj_out`
  as the mean of its four steps (they differ by ~1%). Checked: bit-identical
  to a torch conversion, every key loaded, sharp images.
- **NSFW.** *Unzensiert* in `FLUX.2 klein 9B T2I/Edit` and `FLUX.2 dev
  T2I/Edit` adds a LoRA. For klein it is what makes nudity possible at all:
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
  still toggles. The klein NSFW workflows also use an abliterated text encoder
  ([ponpoke/flux2-klein-9b-uncensored-text-encoder](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder),
  16.4 GB, same tensors as the stock one). On its own it changes nothing —
  klein's reluctance lives in the image model, which is what the LoRA is for.
  Beside the LoRA it changes little: 8 pairs with the same prompt and seed, one
  encoder each, were near-identical in 7; in the eighth only the abliterated
  one left out a garment the prompt had not asked for. The encoder is a
  dropdown on the workflow's node, so the stock one is a click away in the
  UI. Adults only.
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
- **rgthree-comfy.** [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
  (MIT) adds comfort in the UI and nothing else: a progress bar on top, bypass
  or mute a whole group with one click, seed control (random, last, fixed),
  an image comparer with a slider, context nodes that carry model, prompts and
  latent as one wire, and the power lora loader. No models, no requirements.
  None of the bundled workflows uses its nodes, and the API export
  (`export_api.py`) and the workflow builder give the same files with it
  loaded. Its settings are under the rgthree entry of ComfyUI's settings.
- **Music in ComfyUI (ACE-Step 1.5).** Two of ComfyUI's own templates, no
  node pack, as one workflow `ACE-Step 1.5`: XL Turbo (*XL* on, DiT 10 GB)
  or the 2B Turbo (DiT 4.8 GB), both with the 0.6B
  and 4B Qwen encoders (1.2 + 8.4 GB) and the VAE — 25 GB together,
  Apache-2.0. A song from a style description (tags), lyrics with `[Verse]`
  and `[Chorus]` markers, BPM, key, time signature and language; `[Instrumental]`
  as lyrics for none. Saved as MP3 in `output/audio/`. 90 s of music take
  33–34 s with either model (loading 5–9 s): the time goes into the 4B
  language model that writes the audio codes, not into the DiT, so the
  bigger one costs only disk. Tried with the same seed: a German pop ballad
  and an English indie rock song came out without a flaw in the singing
  with both. For an orchestral instrumental XL was clearly better — one
  piece — where the 2B Turbo made unrelated snippets with pauses between
  them: XL for instrumentals.

  Two more of ComfyUI's music templates earned their place against it (#41),
  tried with the same lyrics and styles:
  - `Stable Audio 3 Medium (8 Schritte)` (Stability AI, Stability Community
    License, free below $1M revenue; 15 GB with its T5Gemma and Qwen3.5-2B
    encoders): instrumentals, loops and sound design, no singing. A short
    idea is written out by Qwen3.5-2B first (*Enable_Reprompt*; categories
    Music, Instrument, SFX, One-shot). **90 s of orchestra in 14 s**, a
    10.7 s drum loop in 5 s — strong on the instrumental.
  - `YuE2 Text2Music (int8, 32 Schritte)` (CC BY-NC 4.0, 4 GB): songs with
    vocals from style and lyrics, with an optional symbolic plan (ABC) of
    the melody first. 81–100 s for 90 s; strong in German and English and
    on the instrumental.

  MiniMax Music 3 did not: 530–780 s for 90 s here (it plans ~2250 tokens,
  then its DiT takes 7–11 s a step), the singing unintelligible, the
  instrumental weaker and 62 s instead of 90. Not kept.

  **Music through the image API** (#33): `POST /v1/audio/music` on the
  slot's image API (port 808N, with `--proxy`) runs these workflows from
  `comfyui/api/audio/` and answers with the audio itself (MP3, or FLAC with
  `"response_format": "flac"`; the seed in `X-Seed`). `model`
  `ace-step-15-xl-turbo` (the default), `ace-step-15-turbo`,
  `yue2-text2music` or `stable-audio-3-medium`; `prompt` the style or the
  description; `lyrics` with `[Verse]`/`[Chorus]` (none: an instrumental;
  Stable Audio refuses lyrics); `duration` 5–300 s (60 by default); for
  ACE-Step also `bpm`, `key` ("C major") and `language` ("de"). Results go
  to ComfyUI's temp, not `output/`. Measured: 30 s of German song from ACE
  XL in 19 s, a 10.7 s loop from Stable Audio in 10 s.

  ```bash
  curl -s localhost:8089/v1/audio/music -H 'Content-Type: application/json' -o lied.mp3 \
    -d '{"model": "ace-step-15-xl-turbo", "prompt": "German pop ballad, piano, female vocals",
         "lyrics": "[Verse]\nDer Morgen hängt noch grau im Fenster", "duration": 60, "language": "de"}'
  ```
- **3D models from a picture** (#38). `Pixal3D TRELLIS.2 Image to Model (int8)`
  is ComfyUI's own template: a photo becomes a textured mesh (GLB in
  `output/3d/`, PBR: base colour, metallic/roughness, normal map, ambient
  occlusion). The background is cut away first (BiRefNet), then either
  **Pixal3D** (the default; it also estimates the photo's geometry and field
  of view with MoGe) or **TRELLIS.2** (Microsoft, 4B; *Switch to Trellis2*)
  builds the shape and its colours as voxels, which are meshed, decimated,
  UV-unwrapped and baked. All MIT; 15.2 GB of models. Tried on a product
  shot of a handbag, a ceramic fox figurine and a model of a half-timbered
  house, all made with Z-Image Turbo:

  | | Pixal3D | TRELLIS.2 |
  |---|---|---|
  | fox | 437 s | 620 s |
  | handbag | 735 s | 3788 s |
  | house | 2308 s | 4734 s |

  Both are kept: on the fox and the house both did well; on the plain white
  handbag both failed, TRELLIS.2 a little less. TRELLIS.2's texture stage is
  what takes the time (100–215 s a step here). Files are 23–50 MB at 200,000
  triangles with 4096 px textures.

  **The triangle count** (`target_face_count` on *DecimateMesh*, 200,000 here
  instead of the template's 700,000) hardly changes the time: the fine
  surface is baked into the normal map from the full mesh, and only the
  steps after the decimation run again. Changing just that number and
  running once more takes ~45 s while ComfyUI still holds the rest in its
  cache — a fox at 700,000, 200,000, 100,000 and 50,000 triangles came out
  in 437 s, then 48, 47 and 45 s, 42 to 17 MB, and all looked very good on
  a phone (more triangles a little better). After a restart a new count
  costs the whole run again.

  On ROCm the UV unwrap's batched fp64 solve fails in hipBLAS
  (`HIPBLAS_STATUS_ALLOC_FAILED`); `comfyui/patches/mesh-uv-unwrap-rocm-cpu.patch`
  sends it to the CPU, where it takes ~40 s for 670,000 faces. Hunyuan3D 2.1,
  the other native 3D template, is not set up: its licence excludes the
  European Union. Not served through the image API.
- **Video (LTX-2.5)** (#40). Two of ComfyUI's own templates for Lightricks'
  LTX-2.5, no node pack: `LTX-2.5 Video (int8, distilled)` turns a picture
  into a clip — or, with *Switch to Text to Video?* on, a prompt alone — and
  `LTX-2.5 First-Last Frame (int8, distilled)` fills the clip between a first
  and a last picture. Both make the sound with it (an audio VAE beside the
  video one). 45 GB: the distilled 22B DiT (int8, 21.5 GB), Gemma-4 12B as
  its text encoder (15.4 GB), Gemma-4 E2B for the optional prompt
  enhancement (5.2 GB), the VAEs and a 2× latent upscaler. LTX-2.x Community
  License: free for personal use and below $10M revenue; the Hugging Face
  repo is gated — accept the licence there once with the account whose token
  `download` uses.

  The template samples in two stages: 8 steps at half the size, the latent
  upscaled 2×, then 3 steps at full size. Width and height go in steps of
  32, the length is frame rate × seconds + 1. Measured at 1280×704, 5 s at
  24 fps with sound: a beach scene from a picture in 409 s, a tram in a rainy
  street from text alone in 450 s (stage 1 ~40 s, stage 2 ~260 s), the beach
  from a first to a last picture in 895 s — the last one the most natural
  movement of all. For a quick look, 1024×576 with the second stage left out
  (its output decoded straight away; sampling at 512×288) gives 2 s in 55 s.
  A portrait picture is cut to the template's 16:9 unless the size is set.
  Not served through the image API.
- **Talking and singing pictures** (#43). `LTX-2.5 Talking (int8, distilled)`
  makes a picture speak or sing along to an audio file: the voice is encoded
  with LTX's audio VAE and held fixed by a noise mask while the video is made
  to it — ComfyUI's picture+audio template does this for LTX-2.3; here it runs
  on the LTX-2.5 models already in place. llmctl's *Audio to whole seconds*
  node pads the audio with silence to the video's length — whole seconds, then
  exactly N frames / 24, to the sample — and makes the video that long. That is
  what makes it work: with a voice of 6.1 s under a 7 s video the lips ran on
  for 20 frames after the voice had stopped and drifted from it; padded, they
  follow it closely and stop with it. The sound under the finished video is the
  original audio, not what comes back through LTX's audio VAE: that round trip
  moved the voice 60 ms ahead of the lips and cut 70 ms off its end. A close-up, the face filling the
  frame, follows better than a head-and-shoulders shot. A 7 s clip at
  1280×704 takes ~12 min. Voices from the speech slot or the Qwen3-TTS
  workflows, songs from ACE-Step or YuE2 fit straight in.
- **Music videos** (#47): `llmctl musicvideo <song> <picture>...` makes one on
  the running ComfyUI. The song is cut into sections (`--seconds`, 5 by
  default); in each the pictures take turns to sing it (LTX-2.5 Talking), or a
  sections file has a scene play instead — `scene: <prompt>` from the text
  alone, `scene <picture>: <prompt>` from a picture, `sing <picture>: <prompt>`
  for a section of its own. Every clip is cut to exactly the frames of its
  section and the whole song goes underneath, so the picture stays on the sound
  from the first frame to the last (a 15 s test: 360 frames, 15.000 s).
  `--continuous` starts each section from the last frame of the one before.
  Finished sections are kept beside the output (`<out>.parts/`, with what they
  were made from): a run that stops, or one with a few sections changed, renders
  only what is missing. `--dry-run` shows the plan and the time it takes —
  about 6–7 min per 5 s section at 1280×704. Songs from ACE-Step or YuE2
  (`/v1/audio/music`), singers from Z-Image and Qwen-Image Edit fit straight in.
- **Smoother video** (#44). `Video Frame Interpolation (FILM)` is ComfyUI's own
  template: Google's FILM (69 MB) puts frames between the frames of a video,
  2× by default (up to 16×), the frame rate raised with it and the sound kept.
  A 5 s LTX clip at 1280×704 went from 24 to 48 fps (241 frames) in 64 s,
  visibly smoother in waves and hair. Takes any video file in `input/`.
  `SeedVR2 3B Video Upscale (int8)` (ComfyUI's template, 3.5 GB) sharpens and
  enlarges a video: the 5 s beach clip from 1280×704 to 1920×1056 came out with
  real detail in hair, coat and surf, not just a softer enlargement — but it
  is slow here: 595 s for one second of video, 2864 s for all five with *Split
  Latent* on. Without that switch the whole clip went into one step that had
  not finished after 40 minutes; leave it on for anything longer than a second.
  Best kept for a finished clip.

  The three LTX-2.5 workflows (Video, First-Last Frame, Talking) have both as
  switches before the video is put together, off by default:
  *Hochskalieren (SeedVR2)* (1.5×, Split Latent on) and *48 fps (FILM)* —
  upscaling first, so SeedVR2 sees half the frames. A switched-off branch
  loads nothing. 1 s at 1024×576 with both on: 1536×864 at 48 fps in 408 s.
  For a clip made already, or a whole music video, the two workflows above
  take the video file.
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
- **Beside halogen only what is left.** ComfyUI starts beside halogen when its
  idle footprint fits, but its workflows need 5–53 GB each once they run, and
  beside halogen's ~82 GiB little of that is there: `preset --dry-run` lists
  which workflows still fit.
- **Images** go to `~/.local/share/llmctl/comfyui/output/`, or wherever
  `--output DIR` says (`llmctl start comfy 9 --output ~/Bilder/comfy`; also in a
  preset entry, where `~` is expanded too). `preset-save` records it.
- **No authentication.** ComfyUI has none of its own, and llmctl adds none, so
  `--host` beyond localhost hands anyone on that network the ability to run
  workflows, read the input and output directories and install nodes. The same
  holds for the speech (`tts`) and speech-recognition (`asr`) slots: their
  servers answer whoever reaches them. Keep them on 127.0.0.1, or put something
  in front that authenticates — `--public` does that for LLM slots (token plus
  mTLS), but it does not cover these.
- **No LLM options.** `--host`, `--verbose` (`--verbose DEBUG`), `--clear-logs`,
  `--output` and `--proxy` (the image API, below) apply; everything else is refused. `env`, `bench`, `cache-stats` and
  `probe-reasoning` have nothing to do for it. ComfyUI has no authentication, so
  `--host` beyond localhost warns.

### Workflows with switches

Variants that differ in a model file, a LoRA or a few steps are one workflow
with switches rather than a workflow each — 31 workflows instead of 69. A
switch is a boolean on the canvas; ComfyUI runs only the branch it picks (the
If/Else switch's inputs are lazy), so the other branch loads no model and
costs nothing.

| Workflow | Switches |
|---|---|
| `Qwen-Image 2.1 T2I`, `Qwen-Image 2.1 Edit` | Turbo (Acc LoRA, 4 euler steps instead of 14 dpmpp_2m), Unzensiert (Heretic encoder) |
| `Qwen-Image 2.1 Control` | Steuerung (Bild, Kanten, Pose, Tiefe — llmctl's *Control image* node, each with its strength), Nur in der Maske, Turbo |
| `Qwen-Image 2.1 Inpaint` | Ausschnitt (crop and stitch), Turbo |
| `Qwen-Image 2.1 Outpaint`, `… Colorize`, `… Background Removal` | Turbo |
| `Z-Image Turbo T2I` | Unzensiert (the NSFW finetune) |
| `Z-Image Turbo Control` | Steuerung |
| `Z-Image Turbo Inpaint` | Ausschnitt |
| `FLUX.2 dev T2I`, `FLUX.2 dev Edit` | Turbo (its 8-step LoRA), Unzensiert (NSFW LoRA) |
| `FLUX.2 klein 9B T2I`, `FLUX.2 klein 9B Edit` | Unzensiert (uncensored encoder and NSFW LoRA) |
| `BiRefNet Background Removal` | Feine Kanten (HR matting) |
| `ACE-Step 1.5` | XL |

The image API keeps a model per variant, under the names it had: export_api.py
writes each variant's API graph with its switches set and the unused branch
cut away (`comfyui/variants.py` says which is which), so
`qwen-image-21-inpaint-crop-turbo` is `Qwen-Image 2.1 Inpaint` with Ausschnitt
and Turbo on. Each of the 62 image variants was checked against the workflow
it replaced: the same nodes with the same settings. `llmctl update comfy`
removes the replaced workflows from ComfyUI's list where they are still as
bundled (`comfyui/retired_workflows.txt`) and names the ones you changed.

### ComfyUI from the internet (`--tunnel`)

ComfyUI, its image API and the repair page can be reached from outside
through a server of your own that is on the internet, without opening a
port here:

```
Browser ──443 (Let's Encrypt) + Basic Auth──► Apache on the server ──► 127.0.0.1:8009  ComfyUI
                                                                   └─► 127.0.0.1:8089  /repair, /v1/images
                         autossh -R 8009, -R 8089 ◄── this machine (connects out)
```

`llmctl start comfy 9 --proxy --vision 2 --tunnel` starts autossh beside the
slot: `-R 8009:127.0.0.1:8009` for the UI and, with `--proxy`, `-R 8089:…`
for the image API with its repair page — on the server bound to its loopback
only. The server comes from `LLMCTL_TUNNEL_HOST` or
`~/.config/llmctl/tunnel-host` (`user@host`, a key that logs in without a
prompt). `stop` ends the tunnel with the slot, `status` shows it. A port the
far side still holds from a session that died unnoticed makes the first try
fail (`ExitOnForwardFailure`); llmctl clears its own leftovers and tries again
after a pause.

As a preset: `llmctl preset comfy-remote` (`comfy 9 --proxy --vision 2
--tunnel`). On a ComfyUI that already runs, `preset` adds or stops the image
API and the tunnel on their own instead of restarting it, restarts the image
API when it asks another vision model, and renews the tunnel when the image
API comes anew; `preset --repair` (the watchdog of `llmctl service`) opens a
tunnel again that went down. Tried: switching `comfy` → `comfy-remote` →
`comfy` stopped and added image API and tunnel with ComfyUI kept running, and
a killed tunnel came back with `--repair`.

`deploy/vps-comfy-vhost.conf` is the Apache side: its own name with a Let's
Encrypt certificate, **Basic Auth in front of everything** — ComfyUI has no
login, and whoever reaches it runs workflows and uploads files — the
websocket, `/repair` and `/v1/images/` to the image API, one-hour timeouts
for the long jobs and 200 MB request bodies for pictures. The repair page
calls the image API from the browser with the same credentials; OpenAI
clients from outside send none, so the API stays closed to them. Its
*Vision starten/stoppen* buttons work from outside too: they run llmctl here.

Tried: the tunnel up in 3 s, 8009 and 8089 listening on the server's
loopback, UI, repair page and its status answering there; `stop` freed both
ports on the server and left no ssh behind; a restart came up at once. Through
the server's Apache with this vhost: 401 with a Basic Auth prompt on every
path without credentials, and with them ComfyUI and the repair page load in
the browser.

### Which model for a picture

The text-to-image models side by side, one session, the machine to itself,
1024×1024, seed 42, each model's own workflow: a red fox in snow at morning
light, an old fisherman's portrait at a harbour, and a shop sign reading
"Bäckerei Sonnenschein – frisch seit 1952". Times are per image; the fox
includes loading the model.

| Model | Fox | Fisherman | Sign | Spelling of the sign | Look |
| --- | --- | --- | --- | --- | --- |
| FLUX.2 klein 9B | 40 s | 20 s | 41 s | "Sonnenschin" | vivid, warm light, a little polished |
| FLUX.2 dev Turbo | 170 s | 149 s | 157 s | right | close to dev; coloured specks on the fisherman's cap |
| FLUX.2 dev | 305 s | 288 s | 302 s | right | the best light and the sharpest faces |
| Qwen-Image 2.1 Turbo | 36 s | 17 s | 36 s | unreadable | grainy, darker, documentary |
| Qwen-Image 2.1 | 52 s | 51 s | 51 s | right | clean, flat light, bare backgrounds |
| Qwen-Image 2.1 Heretic | 57 s | 51 s | 57 s | right ("frischseit" tight) | as Qwen-Image for these prompts |
| Z-Image Turbo | 29 s | 18 s | 30 s | "Sonnenschen" | the most natural photographs, fine skin, soft background |
| Z-Image (base) | 109 s | 106 s | 106 s | "Sonnensschin" | like Turbo, less polished; far more variety between seeds, follows a style |

- **Photographs, fast:** Z-Image Turbo — the best look per second here.
- **Variety, styles, negative prompts:** Z-Image base — four seeds, four
  different pictures, where Turbo repeats itself.
- **The best picture, time no matter:** FLUX.2 dev, five minutes an image.
- **Text in the picture, German with umlauts:** Qwen-Image 2.1 at 14 steps
  (51 s) or FLUX.2 dev; Z-Image and klein drop a letter, Qwen-Image Turbo's
  4-step LoRA breaks the lettering altogether.
- **Editing, control, inpainting, colouring, details:** Qwen-Image 2.1 — the
  ControlNet, Inpaint, Detailer and Colorize workflows are built on it.

### The image API (`--proxy`)

`llmctl start comfy 9 --proxy` also starts `images_server.py` on the slot's
proxy port (:8089): OpenAI's image API, so an agent makes and edits images
without knowing a ComfyUI workflow.

```python
from openai import OpenAI
img = OpenAI(base_url="http://localhost:8089/v1", api_key="-", timeout=3600, max_retries=0)
r = img.images.generate(model="flux2-klein-9b", prompt="Ein roter Leuchtturm bei Sonnenuntergang",
                        size="1024x768", extra_body={"seed": 42})
png = base64.b64decode(r.data[0].b64_json)
r = img.images.edit(model="flux2-klein-9b", image=open("turm.png", "rb"),
                    prompt="Mache den Leuchtturm blau-weiß gestreift")
```

- **Models are the bundled workflows**, named after them: `flux2-klein-9b`,
  `flux2-klein-9b-nsfw`, `flux2-dev`, `flux2-dev-turbo`, `flux2-dev-nsfw`,
  `qwen-image-21`, `qwen-image-21-heretic`, `qwen-image-21-turbo`,
  `qwen-image-21-heretic-turbo` for generations and edits, `z-image-turbo`
  for generations,
  `qwen-image-21-control`, `qwen-image-21-canny-control`,
  `qwen-image-21-pose-control`, `qwen-image-21-depth-control`,
  `qwen-image-21-colorize` and
  `qwen-image-21-inpaint`, `qwen-image-21-outpaint` (256 px left and right)
  and `qwen-image-21-background-removal` for edits (each also as `…-turbo`)
  (the image is the control image, the photo or the picture to inpaint —
  transparent where to redraw, as OpenAI's edits take a mask),
  `qwen-image-21-detailer` for edits (faces and hands; its own prompts, no
  Turbo), and `seedvr2-7b-upscale` for edits (4× upscaling, no prompt). `GET /v1/models`
  lists those whose model files ComfyUI has, with the endpoints each serves.
  Without `model`, it is klein.
- **What a request sets:** `prompt`, `size` (generations; `WxH` rounded to 16,
  256–2048, or `auto` for the workflow's own), `n` (1–8, one after another),
  `seed` (not in OpenAI's API; the response names the one used, so a result
  can be made again) and `response_format` (`b64_json`, or `url` — a ComfyUI
  `/view` link, valid until ComfyUI restarts). Always PNG. Edits take one
  image per reference the workflow has (klein 1, dev and Qwen-Image 2), as
  `image`/`image[]` form fields or as data URLs in a JSON `images` list;
  fewer are fine — the unused references drop out of the workflow. An edit's
  size follows its first image.
- **Extra fields for the ControlNet workflows** (#19), so an agent steers them
  as finely as the UI; `GET /v1/models` names the ones each model takes under
  `parameters`, and a workflow refuses a field it has no use for:
  - `mask` (Inpaint): a PNG the size of the image, transparent where to
    redraw — OpenAI's edits mask, as a form file or a data URL. It takes the
    place of the image's own transparency, so the image can stay a plain photo.
  - `strength` (all Qwen-Image 2.1 ControlNet workflows, 0–2): how closely the
    result follows the control image; the workflows use 0.5 (Control) and 1.
    For Canny at 0.3 a watercolour of the photo keeps only its layout, at 1.0
    every edge.
  - `boxes` (Inpaint): instead of a mask, rectangles `[x1, y1, x2, y2]` in the
    image's pixels, each grown by `margin` (16 px by default) — what
    `/v1/images/check` returns.
  - `select` (Inpaint): instead of a mask, a few English words — `"the floor
    lamp"`, `"hair"` — and SAM 3 outlines every match; the mask is their
    outline grown by `margin` (64 px by default here: at 16 the repaint kept
    the outline's shape, a lamp asked to become a plant came back a lamp).
    `POST /v1/images/select` with `image` and `prompt` answers with the
    `segments` alone, a `box` and the `share` of the picture for each — places
    for `boxes`. See *Text to mask* below.
  - `pad` (Outpaint): pixels to add, one number for every side or
    `{"left", "top", "right", "bottom", "feathering"}` (steps of 8, up to
    2048; feathering 40 by default).

  ```python
  r = img.images.edit(model="qwen-image-21-inpaint-turbo", image=open("tisch.png", "rb"),
                      mask=open("maske.png", "rb"), prompt="ein Stück Schokoladenkuchen")
  r = img.images.edit(model="qwen-image-21-outpaint-turbo", image=open("tisch.png", "rb"),
                      prompt="Küche mit Hängelampen", extra_body={"pad": {"top": 256}})
  ```
- **Checking a picture for artifacts** (#23): `POST /v1/images/check` with one
  `image` (and the `prompt` it was made from, which helps) asks a vision model
  for the flaws in it — extra fingers, melted faces, ghosted objects, garbled
  text — and answers with `artifacts`: an `id`, what is wrong, a `fix` (what
  the place should show, a prompt to repaint it) and a `box` in pixels, ready
  for `boxes`. A flaw whose box covers more than `max_area` of the picture
  (a share, 0.4 by default) comes as a remark in `remarks` — with its box, to
  use by hand — since repainting that much tends to make new flaws; `0.05`
  keeps to small glitches. `"preview": true` adds the picture with the
  numbered boxes drawn in. The vision model is a slot of its own: start the
  image API with `--vision SLOT` (`llmctl start comfy 9 --proxy --vision 2`)
  and that slot with `llmctl start flash 2 --mmproj`.

  ```python
  import requests
  c = requests.post("http://localhost:8089/v1/images/check", timeout=3600,
                    files={"image": open("bild.png", "rb")}, data={"prompt": prompt}).json()
  pick = [a for a in c["artifacts"] if a["id"] in (1, 3)]
  r = img.images.edit(model="qwen-image-21-inpaint-crop-turbo", image=open("bild.png", "rb"),
                      prompt=prompt + ". " + "; ".join(a["fix"] for a in pick),
                      extra_body={"boxes": [a["box"] for a in pick]})
  ```

  Measured on 24 Qwen-Image images with hand-judged flaws (22 flawed, 2 clean):
  flash finds 18 of the 22 at least partly (13 fully), puts 3 at the wrong
  place, misses 1, and flags one of the two clean images; ~40 s per image
  (7–170 s). Qwen3-VL-8B found 1 of 22; Qwen3.8-27B was no better than flash
  and ten times slower. So the check proposes and someone picks — repainting
  every box also repaints healthy places.

  **What repair is for:** small, local glitches in an otherwise good picture —
  a hand on a toothbrush, a guitar head, a milk jug, a doubled clock came out
  right in one go (26 s each with Inpaint Turbo). A picture that is flawed
  all over (a crowd of melted faces, tangled yoga bodies) has no sound
  surroundings to repaint from; there a new seed does better. Unattended,
  with every box: 9 of 21 pictures clearly better, 7 hardly changed, 2 worse
  (a cyclist lost his shorts, a new upside-down figure). Box size does not
  tell them apart — the harmful boxes held 10–13 % of the picture, good big
  repairs 10–29 %; `max_area` 0.05 would have spared both. flash (82 GiB) and the Inpaint
  workflow (~37 GiB of models: DiT, ControlNet, encoder) do not fit into the
  104 GiB together: check first, stop flash, then repair.

  **Repair on a crop** (#35): `qwen-image-21-inpaint-crop-turbo` (and
  `-crop`, `z-image-turbo-inpaint-crop`) use
  [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch):
  the mask's box, 2.5 times as wide and high, is cut out, scaled to 1024²,
  repainted there and stitched back with a 32 px blended seam. The plain
  Inpaint workflows scale the whole picture to ~1 MP and send every pixel
  through the VAE — a 2048² photo came back at 1024² — the crop ones return
  the picture at its size, untouched outside the patch (on a 2048² piano
  photo every pixel away from the box was identical), and a small flaw gets
  the model's whole resolution. Inpaint Crop Turbo 27 s, Inpaint Crop 79 s.
  Context matters: at 1.5 times the box a face in profile came back frontal
  and a hand grew fingers; at 2.5 both held, and hands on piano keys and
  knitting came out sharper than from the whole picture. Z-Image Turbo on a
  crop did faces and the piano well, but made a hand on a coffee cup into
  glowing orange nails at every context — Qwen-Image is the one to use.
- **The repair page:** `http://localhost:8089/repair` does all of the above by
  hand. Drop in a picture (or pick one of ComfyUI's latest outputs), give the
  prompt it came from, press *Prüfen*: the flaws appear as numbered boxes over
  the picture, each with a checkbox and its `fix` text to edit. Boxes move,
  grow at their corner and go with ✕ or Delete; dragging on the picture adds
  one of your own. *Reparieren* repaints the ticked ones — the prompt is made
  from the picture's prompt and the fixes, and stays as you edit it — and
  shows the result beside the picture (hold it for the original). It uses
  Inpaint Crop Turbo by default: boxes whose crops overlap are repainted in
  one run, the others one after the other, each with its own fixes in the
  prompt (a prompt edited by hand goes to every run). The whole-picture
  Inpaint workflows stay in the list. *Stelle per Wort* adds boxes by name:
  SAM 3 finds what the words name ("the left hand" gives both hands — it
  knows things, not left and right), each match becomes a box to keep,
  move or drop, its fix left to write. *Nochmal*
  tries another seed, *Übernehmen & weiter* makes the result the picture to
  check again, *Herunterladen* saves it.

  The page starts and stops the vision model on request (*Vision starten*,
  *Vision stoppen*), since it and Inpaint do not fit together; starting first
  has ComfyUI let go of its models. That needs `--vision` as a slot number:
  the image API then calls `llmctl start flash N --mmproj` and `llmctl stop N`
  (`LLMCTL_VISION_MODEL` names another model). Nothing starts by itself. With
  `--host`, whoever reaches the page can press those buttons, as they can run
  workflows in ComfyUI.
- **The same in ComfyUI:** the workflow `Artifact Check (flash)` holds
  llmctl's own node *Artifact Check (llmctl)* (from `comfyui/nodes/`, linked
  into ComfyUI's `custom_nodes/` at every start). Load a picture, give its
  prompt, run: the node asks the image API's check and shows the picture with
  the numbered boxes, a report and the prompt for the repair. It also leaves
  the picture in `input/` as `artifacts_<hash>.png`, the flaws transparent.
  Load that file in `Qwen-Image 2.1 Inpaint` with *Ausschnitt* (and *Turbo*): the boxes are its mask,
  to edit in the MaskEditor (right-click the picture → *Open in MaskEditor*);
  paste the repair prompt and run. Its `vision` switch, *start if needed, stop
  after*, starts flash for the check (after unloading ComfyUI's models) and
  stops it afterwards, so the Inpaint run has the memory; *leave as is* (the
  default) touches nothing. The node reaches the image API through
  `LLMCTL_IMAGES_API`, which llmctl sets when it starts ComfyUI — so start it
  with `--proxy --vision SLOT`. The workflow is not served through the image
  API itself; after an install, `llmctl update comfy` adds it to the UI's list.
- **Results stay out of the gallery.** They go to ComfyUI's temp directory,
  not to `output/`; the caller has them.
- **Measured** at 1024×1024, the machine to itself, each including loading
  its model: Qwen-Image 2.1 51–77 s, an edit with one image 98 s, with two
  160 s; FLUX.2 dev 330 s, an edit with one image 630 s, with two ~20 min,
  FLUX.2 dev Turbo 137 s; SeedVR2 upscales 384 → 1536 px in 33 s.
  klein, beside a 70B and an 8B LLM with 37 GB free: 39–55 s, an edit 62 s —
  and once 350 s, when its ~33 GB met that memory and the system went into
  zram. The official `openai` Python client works unchanged.
- **Give the client time, and no retries.** The `openai` client gives up
  after 10 minutes and then sends the request again, twice — for FLUX.2 dev
  that is the same 20 minutes of work three times. Hence `timeout=3600,
  max_retries=0` above. The server waits up to an hour itself and stops the
  job in ComfyUI when that runs out.
- **The API format.** POST /prompt takes a workflow in ComfyUI's API format,
  which only its frontend can make from a saved one. `comfyui/api/` holds the
  bundled workflows in that format, made by `comfyui/export_api.py`: it opens
  the UI of a running ComfyUI in Chrome without a window and exports each
  workflow as "Export (API)" would. After changing a workflow, run it again
  (`comfyui/export_api.py 8009`) and commit what it writes; `make test` fails
  while a workflow has no API version.

## The tts backend (speech)

A models.conf entry with `_model_backend="tts"` (`speech` in the examples)
speaks: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B through llama.cpp,
behind `tts_server.py`, which answers the OpenAI speech API on the slot's server
port.

```bash
llmctl download speech        # GGUFs, VoiceDesign model, two voices (+ a venv, without ComfyUI)
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
- **Variation.** Generation samples, so the same text sounds a little
  different each time, and so does how much the clone sounds like its
  reference: 0.78–0.89 by a speaker model (Resemblyzer) for one voice here,
  from draw to draw. The seed is therefore random by default. A request's
  `seed`, or the slot's `--seed N`, fixes it for output that must repeat.
  Measured and ruled out: a fixed seed keeps one draw for good (1234 happened
  to be a below-average one), and a "best seed per voice" does not carry over
  — the seeds best for one text were no better, or worse, for the next. The
  streaming engine is also only reproducible as a sequence: some state inside
  the PR's server outlives a request, one cause of which, the audio RNG not
  reseeding, `patches/llama.cpp-pr26603-reseed-per-generation.patch` fixes.
  Qwen3-TTS' cloning with the reference's transcript (ICL, PyTorch only) was
  no closer than the speaker embedding llama.cpp uses: 0.842 against 0.851.
  What helps is the reference itself: 10–15 s of clean speech. Replacing a
  7.6 s recording with a 21.6 s one raised the resemblance of its clones from
  0.866 to 0.898 (best draw 0.932).
- **Comparing two recordings** is worth measuring rather than guessing. A
  speaker model scores how close a clone comes to its reference; nothing of
  this belongs to llmctl, so it lives in a venv of its own:

  ```bash
  python3 -m venv ~/.cache/spk && ~/.cache/spk/bin/pip install \
      torch --index-url https://download.pytorch.org/whl/cpu
  ~/.cache/spk/bin/pip install resemblyzer soundfile
  ```

  Speak the same text with each voice (`curl … -o clone-<voice>.wav`), then:

  ```python
  import numpy as np
  from resemblyzer import VoiceEncoder, preprocess_wav
  enc = VoiceEncoder("cpu")
  ref = enc.embed_utterance(preprocess_wav("aufnahme.wav"))
  for f in ("clone-a.wav", "clone-b.wav"):
      print(f, float(np.dot(ref, enc.embed_utterance(preprocess_wav(f)))))
  ```

  Scores here: 0.85–0.93 for a clone of the speaker, 0.47–0.54 for a
  different voice. Take several draws per voice — one says little.
- **Chunking.** Each sentence is generated without the ones before it, so
  tone and pace can shift a little at sentence boundaries. `"chunking": false`
  (or the slot server's `--no-chunking`) speaks groups of sentences up to
  ~280 characters instead — smoother, but the first sound waits for the whole
  first group: 3.3 s instead of 1.5 s for two sentences here.
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
  merges the pinned PR commit, applies the fixes in
  `patches/llama.cpp-pr26603-*.patch` (an API change since; the audio RNG's
  reseeding) and builds `llama-server`
  and `llama-tts` with Vulkan. Nothing is installed, and a checkout it uses is
  left as it is: the worktree shares only its git objects. It checks for git,
  cmake, a C++ compiler and glslc first. `LLAMA_BASE=HEAD` builds on the
  checkout's current commit instead, which may no longer merge with the PR. To
  rebuild, remove the worktree (`git worktree remove --force
  ~/.local/share/llmctl/llama.cpp-tts`) and run it again; once the PR is
  merged, the ordinary llama-server does all this.
- **Moving to a newer PR or llama.cpp.** Both commits are pinned in
  `patches/build-tts-server.sh` (`PR_COMMIT`, `BASE_COMMIT`), so a rebuild
  reproduces the same thing until they are changed. To follow the PR:

  ```bash
  gh pr view 26603 -R ggml-org/llama.cpp --json headRefOid -q .headRefOid   # its commit now
  git -C ~/src/llama.cpp log -1 --format=%H                                 # a llama.cpp to build on
  ```

  Put them into the script, remove the worktree and run it. Three things can
  go wrong, each with an obvious answer: the merge conflicts (the PR is too
  far behind that llama.cpp — pin an older `BASE_COMMIT`, e.g. the merge
  base), a `llama.cpp-pr26603-*.patch` no longer applies (the PR took the fix
  over — drop that patch file), or the build fails on an API that moved
  (adjust the call and regenerate the patch with `git -C
  ~/.local/share/llmctl/llama.cpp-tts diff`). After building, restart the
  speech slot: it runs the binary it started with.
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

  `voice design` runs Qwen3-TTS VoiceDesign (PyTorch) once to speak a sample
  in the described voice, ~15 s. Where ComfyUI is set up it runs in ComfyUI's
  venv, with the `qwen_tts` its Qwen-TTS node ships as source — the same ROCm
  torch, not installed a second time (16 GB); the pip package `qwen-tts` would
  pin transformers 4.57 and break ComfyUI. Without ComfyUI, `download speech`
  builds a venv of its own under `~/.local/share/llmctl/tts/venv`, and `prune`
  offers it for deletion once ComfyUI's can take over; describe the speaker as a native speaker of the language, since
  the clone keeps any accent. `voice add` takes a recording of your own (wav,
  mp3 or flac); a few seconds of clean speech are enough.
- **Backing them up.** The voices live in `~/.local/share/llmctl/tts/voices/`
  and nowhere else (the two bundled ones aside), so a lost home directory means
  recording and designing them again. `llmctl voice export [FILE]` writes them
  all into one archive (default `~/llmctl-voices-<date>.tar.gz`), `llmctl voice
  import FILE` brings them back — existing voices of the same name stay unless
  `--force`.
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
| hear, live | `asr-live` (Nemotron 3.5 ASR on audio.cpp) | `POST :800N/v1/audio/transcriptions/live` |
| think | any LLM, e.g. `qwen-moe --proxy` | `POST :808N/v1/chat/completions` (`stream`) |
| speak | `speech` (Qwen3-TTS 1.7B) | `POST :8005/v1/audio/speech` (`stream`) |

`llmctl preset voice` starts the two speech slots (~10 GB together); add the
LLM to a preset of your own. All three listen on 127.0.0.1 without
authentication — an agent runs beside them, not across a network. The endpoints follow OpenAI's API, so its client
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

- **Live speech recognition.** `asr-live` runs Nemotron 3.5 ASR (0.6B, 35
  languages, German found by itself) on [audio.cpp](https://github.com/0xShug0/audio.cpp),
  a ggml runtime for speech models, as backend `audiocpp`. It transcribes
  while the audio is still coming in: send raw 16 kHz mono s16le PCM as it is
  captured, in a chunked request body, and read the text as SSE deltas on the
  same connection —

  ```bash
  ffmpeg -f pulse -i default -ar 16000 -ac 1 -f s16le - |
    curl -sN -T - -H 'Expect:' 'http://127.0.0.1:8007/v1/audio/transcriptions/live?model=nemotron-asr-live'
  ```

  Measured on a 25 s German passage: first text after 1 s, a delta every
  320 ms, and the whole text 0.06 s after the last audio, where `asr` needs
  2.6 s once the recording is complete. A whole recording also works
  (`POST /v1/audio/transcriptions`, `model=nemotron-asr`, 0.34 s for 25 s). It
  is a little less exact than `asr`: three slips in that passage (a missing
  "Prozent", "vier, zwei" for "vier Komma zwei"), one in 22 s of free speech,
  and numbers stay words. ~2.4 GiB on the GPU. `llmctl download asr-live`
  clones audio.cpp at its newest release tag and builds it with Vulkan and only
  the model families `models.conf` names (85 s, 1.8 GB); when a newer release
  is out, `outdated` says so and `download asr-live` rebuilds at it
  (`--master`: at main; v0.9.0 is the fallback when GitHub cannot be asked).
  New commits on main are only shown. The server takes its models from a JSON file llmctl writes
  per slot under `~/.local/state/llmctl/audiocpp/`.

  **When the speaker has finished** follows from the same stream: while
  someone talks, a delta arrives every 320 ms; when none has come for about
  700 ms, the utterance is over. audio.cpp's Silero VAD would say so directly,
  but its server does not deliver VAD over the live route yet (it waits for
  text and fails at the end), only its CLI does.

  Tried on audio.cpp and not taken: Qwen3-ASR (5.2 s for the passage, and its
  "streaming" only buffers) and Qwen3-TTS, which runs offline only — the
  passage above in 20.3 s with nothing before, where `speech` has the first
  sound after 3.5 s and all of it after 16.0 s.
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

## Building blocks for retrieval and memory

An agent that has to find something again — notes, documents, what was said
last week — needs two more slots beside its LLM:

| Step | Slot | Endpoint |
| --- | --- | --- |
| index and search | `embed` (Qwen3-Embedding 8B) | `POST :800N/v1/embeddings` |
| sort the hits | `rerank` (Qwen3-Reranker 0.6B) | `POST :800N/v1/rerank` |

`llmctl download embed` and `llmctl download rerank`, then start them on free slots
(`llmctl start embed 7`, `llmctl start rerank 8`) or add `"embed 7"` and
`"rerank 8"` to a preset. Neither generates text, so `env`, `bench` and
`probe-reasoning` leave them out, and `status` names the endpoint instead of
reasoning and drafting. Like the voice slots they have no authentication.

```python
from openai import OpenAI
import requests
emb = OpenAI(base_url="http://localhost:8007/v1", api_key="-")
docs = [...]                                                      # passages, as they are
query = "Instruct: Given a question, retrieve passages that answer it\nQuery: " + question
vectors = emb.embeddings.create(model="embed", input=docs + [query]).data  # 4096 dims
# ... nearest by cosine similarity, then the top 20 or so into the reranker:
hits = requests.post("http://localhost:8008/v1/rerank",
                     json={"query": question, "documents": top20}).json()["results"]
```

- **Queries carry an instruction, documents do not.** Qwen3-Embedding was
  trained that way; the instruction may be English whatever the language of
  the text.
- **Measured.** Three German questions against four unrelated passages: both
  slots put the right passage first every time, the reranker with scores of
  0.94–1.00 against ≤ 0.001 for the rest. 64 passages (8,200 tokens) embed in
  9.6 s (~850 tokens/s), rerank in 2.0 s. On the GPU: 11.9 GB and 2.6 GB.
- **Smaller.** `Qwen/Qwen3-Embedding-4B-GGUF` (Q8_0, 4.1 GB) roughly doubles
  the speed for a little retrieval quality: change the entry's `model` and
  `hf_includes`. Vectors from different models are not comparable — an index
  is rebuilt when the model changes.
- **One input, one batch.** A pooled input has to fit into one physical batch,
  so both entries set `-ub`/`-b` to the context size (8,192 tokens). Longer
  passages are refused; split them.

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
- **scout** — Llama 4 Scout (109B MoE, 17B active) abliterated v2, i1-Q5_K_M, 64K ctx, vision via `--mmproj` (llama.cpp's own projector); prompts read 2.6–3.4× and written ~1.3× as fast as `llama3.3` (271–283 / 15.2 t/s), but its German is sloppy — typos in most answers — where llama3.3's is clean
- **r1** — DeepSeek-R1-Distill-Llama-70B Uncensored v2 Unbiased Reasoner, i1-Q5_K_M, 128K ctx
- **mistral** — Mistral-Medium-3.5-128B, UD-Q5_K_XL, 32K ctx
- **diamond** — L3.3-70B Magnum Diamond, i1-Q5_K_M, 32K ctx; drafted by the same Llama-3.2-1B (~2.0×)
- **magnum** — Magnum-v4-72B, Q6_K, 32K ctx; a Qwen2.5-72B fulltune, drafted by Qwen2.5-1.5B-Instruct Q4_K_M (~2.0×)
- **qwen-gufo** — the same Qwen3.8-27B uncensored Q8 on the [gufo backend](#the-gufo-backend), DFlash2 draft, 64K per session (256K in preset `qwen-gufo-256k`), vision via `--mmproj`; prefill 30–55 % and code decoding ~2× faster than `qwen`
- **qwen-halogen** — Qwen3.8-27B stock (not uncensored) on [halogen-server](#halogen-server-qwen38-27b), ~6.3 bpw, DFlash2 draft, 256K ctx, text only; prefill ~1.8× `qwen-gufo`, code decoding 45 t/s — meant for code
- **flash** — Qwen3.8-Flash-Next 125B MoE on the [halogen backend](#the-halogen-backend), 4-bit `.hgn`, 256K ctx (512K with YaRN via `--ctx 524288`), vision via `--mmproj` (on in preset `flash`); ~82 GiB, other slots beside it only as they fit

- **embed** — Qwen3-Embedding-8B, Q8_0, 8K ctx; `/v1/embeddings` for [retrieval](#building-blocks-for-retrieval-and-memory)
- **rerank** — Qwen3-Reranker-0.6B, Q8_0, 8K ctx; `/v1/rerank`
- **asr-live** — Nemotron 3.5 ASR 0.6B, f16, on audio.cpp (backend `audiocpp`); [live speech recognition](#building-blocks-for-a-voice-agent)

Multimodal projectors are only loaded on an explicit `--mmproj`.

### Updating llmctl itself

`llmctl self-update` brings llmctl to its newest release (a `vX.Y.Z` tag on
GitHub; `--master` takes master instead, installed whenever asked, as its
version number stays the last release's until the next). Where it came from
decides how:

- run from a checkout (`make install-link`): the checkout is fast-forwarded to
  the release if it is behind, and that is all;
- installed with `make install`, which notes the checkout it was run from in
  `share/llmctl/source`: the checkout is fast-forwarded if it is behind the
  release, then `sudo make install` from it — so a checkout that is ahead of
  the release (your own commits) is installed as it is;
- installed without a checkout: the release's archive is downloaded and
  installed with `sudo make install`.

A checkout with uncommitted changes, or one that has diverged from the release,
is left alone. `--check` only tells what it would do. Afterwards it names the
steps that may apply (`update comfy`, `download llama`, `download asr-live`)
without running them. `outdated` lists llmctl first: the newest release and
the first lines of its notes, how far master is past it, and whether the
checkout is newer than the installed copy.

### What has a newer version

`llmctl outdated` asks, and changes nothing (exit 0: all current, 1: something
newer). Everything is measured against releases; `--master` measures llmctl,
the llama.cpp builds, audio.cpp and ComfyUI against master instead — what
`self-update --master`, `update llama --master`, `download asr-live --master`
and `update comfy --master` would bring — with the newest release beside it:

- **halogen**: the image tag in `models.conf` against the registry's newest,
  with the changelog headings in between;
- **llama.cpp**: the commit each installed llama-server reports in `--version`
  against the newest stable release (a `v` tag, e.g. v0.5.0) — at or past it
  is current; how far `master` has moved on is only shown;
- **the speech server's PR**: merged yet, or commits past the one we build;
- **ComfyUI** against its newest release tag (not master), and each custom node's pinned commit;
- **installed models against their Hugging Face repos, by content.** GGUFs and
  halogen through hf's own record of each file's SHA-256, taken at download.
  ComfyUI's models are read and hashed once — ~140 GB in 42 s here, four files
  at a time — and cached in `~/.local/state/llmctl/checksums.tsv` by size and
  mtime, so later runs take seconds; `--quick` compares size and the date of
  the repo's last change instead. The date is shown either way. Built models
  (`comfyui/sources.json`) come from a pinned revision and cannot drift;
  Civitai's cannot be asked.

A changed model is fetched again by deleting it and running `download`.

### Watching for new versions

`outdated` ends with a *Watch* section from `comfyui/watch.conf`: for each model
family in use (LTX, Qwen-Image, Z-Image, FLUX, ACE-Step, TRELLIS, Stable Audio)
whether its author on Hugging Face has published a higher version than the one
in use — a new version is a new repo (`Lightricks/LTX-2.6`), which the checks
of the files in use would never see — and, for open issues that wait for a
ComfyUI template (an LTX-2.5 picture+audio workflow for #43, say), whether
ComfyUI's template list has it now. A line in `watch.conf` adds a family or a
template to wait for.

### Pruning what nothing names

`llmctl prune` lists what no longer belongs to anything, with its size on
disk and what deleting it would actually free, and deletes it after `yes` is
typed at a terminal (`--dry-run`, or no terminal: list only):

- a directory under the models directory no `models.conf` entry reaches into;
- inside one that an entry does reach, a `.gguf` no entry names — the further
  shards of a split GGUF count as named by its first, drafts named in
  `spec_args` count, halogen directories are left whole (the server mounts
  them), and so is the VoiceDesign model llmctl uses itself;
- ComfyUI models no installed workflow names (what `list` shows as such).

Never listed: a joined file's `X.gguf.parts.json` (see below), and the
joined file itself counts as named like any other. On btrfs, `prune` reports
what deleting would really free rather than the size of the files.

### Models split into byte parts

Some repos hold a large GGUF as plain byte parts — `X.gguf.part1of2`,
`X.gguf.part2of2` (mradermacher's Llama-3.3-70B Q6_K, for one) — which only
`cat` makes a model of. `download` handles them (`hf_parts.py`):

1. it fetches the parts, then reads each once and checks it against the
   SHA-256 the Hub keeps for it (its LFS hash);
2. joins them with `cat` — on btrfs a reflink, no second copy — and checks
   the size; a joined file already there is kept when its SHA-256 equals
   that of the parts in a row;
3. writes `X.gguf.parts.json` beside it: repo, revision, and each part's
   name, size and SHA-256, plus the joined file's;
4. deletes the parts, and what hf keeps about them in `.cache`.

From then on the companion stands for the parts. `download` does not fetch
them again while the Hub still has exactly those (`--exclude`); when the
Hub's hashes or sizes change, it fetches the new ones and joins anew.
`download --check` reports a joined file as the same on the Hub or changed,
from the companion — no local parts needed, no download. Without the Hub
(offline) the companion is trusted rather than fetching tens of GB.
gguf-split shards (`X-00001-of-00002.gguf`) are not parts: llama.cpp loads
those itself, and they stay as they are.

## Architecture

- `llmctl` — Main entry point for all commands
- `proxy.py` — optional Flask proxy (`start --proxy`) that forwards requests to the local llama-server and optimizes prompts for caching; port and backend configurable via `LLM_PROXY_PORT` / `LLM_BACKEND_URL`. For halogen it also answers `/v1/messages` (`LLM_TRANSLATE_MESSAGES=1`), clamps token budgets (`LLM_MAX_TOKENS_CAP`) and waits longer (`LLM_PROXY_TIMEOUT`)
- `anthropic_compat.py` — the Messages ↔ Chat Completions translation `proxy.py` uses for backends without a Messages API
- `halogen_bench.py` — `bench` for halogen models: prefill and decode speed of a running slot
- `hf_parts.py` — joins models the Hub holds as byte parts (`*.partNofM`), checks them against its hashes and keeps those in `X.gguf.parts.json`
- `images_server.py` — OpenAI image API in front of a ComfyUI slot (`start comfy N --proxy`); `comfyui/export_api.py` makes the API-format workflows in `comfyui/api/` it runs
- `examples/models.conf` — Model definitions (paths, binaries, ROCm env vars); read from `~/.config/llmctl/`
- `examples/presets.conf` — Named configurations: which models run together, on which slots, with which flags (`llmctl preset <name>`)
- `templates/` — chat templates referenced from `models.conf` as `$SHARE_DIR/templates/…`
- `Makefile` — `install`, `install-link`, `uninstall`, `test`
- `tests/` — `smoke.sh` and `test_python.py`, see [Tests](#tests)
- `deploy/vps-llm-vhost.conf` — Apache vhost for the VPS in front of `--public`

## Tests

    make test                    # both suites
    tests/smoke.sh voice         # only the smoke cases whose name contains "voice"
    tests/test_python.py -k Loudness

Neither needs a GPU, a model or the network, and neither touches your
installation. `tests/smoke.sh` runs `llmctl` against a sandbox — its own
config, data, state and models directories through the `LLMCTL_*_DIR`
overrides, empty stand-ins for the model files `examples/models.conf` names, and
stub binaries on the `PATH` — and checks what each command prints: the command
lines `--print-cmd` builds for every backend, the options each backend refuses,
slots, presets, `env`, and voices from `add` to `export`/`import`.
`tests/test_python.py` covers the Python helpers' own rules: which models a
workflow needs, how text is split into pieces to speak, loudness levelling, the
streamed WAV header and the transcript clean-up in the proxy. A helper whose
imports (flask, numpy, requests) are missing is skipped, not failed.

What they cannot reach is everything that needs the real thing: a server that
starts and answers, a download, `update comfy`. `tests/setup_paths.sh` covers
the setup paths from nothing — ComfyUI (`download`, `update`, custom nodes,
an image through the API), the voice-design venv, the streaming speech build,
speech read back by asr — with only what setup builds in a scratch directory
(`SETUP_TEST_DIR`, default `/tmp/llmctl-setup-test`) and the real models, which
it never writes to. The two parts take ~17 GB each, one after the other, and
the machine to themselves.
