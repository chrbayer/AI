#!/usr/bin/env python3
"""The models ComfyUI workflows need — `llmctl list` and `llmctl download` for a
comfyui entry.

A workflow saved by ComfyUI names the files its loader nodes need, with where
they come from: every loader carries properties.models = [{name, url,
directory}], `directory` being the models subfolder (diffusion_models,
text_encoders, vae, loras, ...). So the workflows themselves are the list of
what to download, and a new workflow brings its models along.

A file is identified by <directory>/<name>. Workflows share files — the text
encoder of FLUX.2 dev sits in all four of its workflows — and each is fetched
once. The same file may also be named with different URLs; they are tried in
the order they appear, and the disagreement is reported.

    comfyui_models.py list     WORKFLOWS_DIR MODELS_DIR
    comfyui_models.py sizes    WORKFLOWS_DIR MODELS_DIR
    comfyui_models.py download WORKFLOWS_DIR MODELS_DIR [PATTERN] [--sources FILE]

PATTERN restricts download to the workflows whose file name contains it
(case-insensitive). Files already present are never fetched again, and nothing
is ever deleted: `list` only reports files no workflow names any more.

Some files exist nowhere in the form ComfyUI loads — a checkpoint published only
in the transformers layout, say. --sources names a JSON file of recipes for
them (comfyui/sources.json): the repo and commit to take the shards from, and
how to rename the tensors. Such a file is built rather than fetched: the shards
are downloaded, then written out as one safetensors file with the new names.
The tensor data is copied byte for byte, so nothing is converted or rounded.
"""
import json
import os
import struct
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}
HF_URL = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
# A whole repo, for node packs that load a model directory (Qwen3-TTS): the
# workflow names it as <directory>/<name> and the repo's snapshot goes there.
HF_REPO = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+?)(?:/tree/([^/]+))?/?$")
STAGING = ".download"


def workflows(wf_dir, pattern=None):
    """(file name, parsed JSON) for every workflow, sorted by name."""
    out = []
    for path in sorted(Path(wf_dir).glob("**/*.json")):
        if pattern and pattern.lower() not in path.name.lower():
            continue
        try:
            out.append((str(path.relative_to(wf_dir)), json.loads(path.read_text())))
        except (OSError, ValueError) as e:
            print(f"  skipping {path.name}: {e}", file=sys.stderr)
    return out


def nodes(graph):
    """All nodes, including those inside subgraphs (definitions.subgraphs)."""
    yield from graph.get("nodes", []) or []
    for sub in (graph.get("definitions") or {}).get("subgraphs", []) or []:
        yield from nodes(sub)


def refs(wf):
    """{directory/name: [url, ...]} for one workflow, URLs in order of appearance."""
    out = {}
    for node in nodes(wf):
        for m in (node.get("properties") or {}).get("models") or []:
            name, directory = m.get("name"), m.get("directory")
            if not name or not directory:
                continue
            urls = out.setdefault(f"{directory}/{name}", [])
            if m.get("url") and m["url"] not in urls:
                urls.append(m["url"])
    return out


def collect(wfs):
    """Merge per-workflow refs: {key: [urls]} plus {key: [workflow names]}."""
    urls, users = {}, {}
    for name, wf in wfs:
        for key, u in refs(wf).items():
            have = urls.setdefault(key, [])
            have.extend(x for x in u if x not in have)
            users.setdefault(key, []).append(name)
    return urls, users


def present(path):
    """A file, or a model directory with something in it."""
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def gib(n):
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


def orphans(wfs, models):
    """Model files under `models` that none of the workflows names."""
    referenced = set()
    for _, wf in wfs:
        referenced.update(refs(wf))
    out = []
    if models.is_dir():
        for sub in sorted(p for p in models.iterdir() if p.is_dir() and p.name != STAGING):
            for f in sorted(sub.rglob("*")):
                key = str(f.relative_to(models))
                inside = any(key.startswith(ref + "/") for ref in referenced)   # part of a repo snapshot
                if f.is_file() and f.suffix in MODEL_SUFFIXES and key not in referenced and not inside:
                    out.append(f)
    return out


def cmd_orphans(wf_dir, models_dir):
    """One path per line, for `llmctl prune`."""
    for f in orphans(workflows(wf_dir), Path(models_dir)):
        print(f)
    return 0


def cmd_list(wf_dir, models_dir):
    wfs = workflows(wf_dir)
    if not wfs:
        print(f"    no workflows in {wf_dir}")
        return 0
    models = Path(models_dir)
    width = max(len(n) for n, _ in wfs)
    for name, wf in wfs:
        r = refs(wf)
        have = [k for k in r if present(models / k)]
        total = sum(size(models / k) for k in have)
        missing = [k.split("/", 1)[1] for k in r if k not in have]
        state = "ready" if not missing else "missing: " + ", ".join(missing)
        print(f"    {name.removesuffix('.json'):<{width - 5}}  {len(have)}/{len(r)} models, "
              f"{gib(total):>8}  {state}")
    lone = orphans(wfs, models)
    if lone:
        print("    named by no workflow (kept; llmctl prune removes them):")
        for f in lone:
            print(f"      {f.relative_to(models)} ({gib(f.stat().st_size)})")
    urls, _ = collect(wfs)
    for key, u in urls.items():
        if len(u) > 1:
            print(f"    note: {key} has {len(u)} sources: " + ", ".join(u))
    return 0


def cmd_sizes(wf_dir, models_dir):
    """<bytes>\t<workflow> for every workflow whose models are all present —
    what `llmctl preset` weighs against the memory ComfyUI would have left."""
    models = Path(models_dir)
    for name, wf in workflows(wf_dir):
        r = refs(wf)
        if r and all(present(models / k) for k in r):
            print(f"{sum(size(models / k) for k in r)}\t{name.removesuffix('.json')}")
    return 0


def fetch_hf(url, dest, staging):
    m = HF_URL.match(url)
    assert m
    repo, rev, path = m.groups()
    path = urllib.parse.unquote(path)     # "Flux%20Klein.safetensors" is "Flux Klein.safetensors" in the repo
    local = staging / repo
    cmd = ["hf", "download", repo, path, "--revision", rev, "--local-dir", str(local)]
    if subprocess.run(cmd).returncode != 0:
        return False
    src = local / path
    if not src.is_file():
        print(f"      hf reported success, but {src} is not there", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)
    return True


CIVITAI_HELP = ("Civitai hands most files out only to a logged-in account. Create an API key at\n"
                "https://civitai.com/user/account → API Keys → Add API key, and paste it here.")


def civitai_token_file():
    return Path(os.environ.get("LLMCTL_CIVITAI_TOKEN_FILE",
                               os.path.expanduser("~/.config/llmctl/civitai-token")))


def ask_civitai_token(reason):
    """Ask for a key on the terminal, store it for next time. None when nobody
    can answer (no terminal) or the answer is empty."""
    print(f"      {reason}", file=sys.stderr)
    if not sys.stdin.isatty():
        print("      " + CIVITAI_HELP.replace("\n", "\n      ").replace("paste it here",
              f"put it into {civitai_token_file()}"), file=sys.stderr)
        return None
    import getpass
    print("      " + CIVITAI_HELP.replace("\n", "\n      "), file=sys.stderr)
    token = getpass.getpass("      Civitai API key (empty to skip): ").strip()
    if not token:
        return None
    f = civitai_token_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(mode=0o600)
    f.write_text(token)
    os.chmod(f, 0o600)
    print(f"      stored in {f}", file=sys.stderr)
    os.environ["CIVITAI_TOKEN"] = token
    return token


def with_token(url, token):
    """The key goes into the query — the documented way; a header would not
    survive the redirect to Civitai's storage."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query) + [("token", token)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def fetch_url(url, dest, secret=None):
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "llmctl"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 16 * 2**20)
    except OSError as e:
        part.unlink(missing_ok=True)
        # The error text can carry the URL, and with it the key.
        msg = str(e).replace(secret, "***") if secret else str(e)
        print(f"      {msg}", file=sys.stderr)
        return getattr(e, "code", None) or False
    os.replace(part, dest)
    return True


def fetch_repo(url, dest, staging):
    """A whole Hugging Face repo into the directory dest."""
    repo, rev = HF_REPO.match(url).groups()
    part = staging / repo
    cmd = ["hf", "download", repo, "--local-dir", str(part)] + (["--revision", rev] if rev else [])
    if subprocess.run(cmd).returncode != 0:
        return False
    shutil.rmtree(part / ".cache", ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.rmdir()                  # empty, or present() would have held
    os.replace(part, dest)
    return True


def fetch_plain(url, dest):
    if urllib.parse.urlsplit(url).hostname not in ("civitai.com", "civitai.red"):
        return fetch_url(url, dest) is True
    token = os.environ.get("CIVITAI_TOKEN", "").strip()
    if not token:
        token = ask_civitai_token("no Civitai API key yet")
        if not token:
            return False
    result = fetch_url(with_token(url, token), dest, token)
    if result in (401, 403):
        token = ask_civitai_token(f"Civitai refused the key (HTTP {result})")
        if not token:
            return False
        result = fetch_url(with_token(url, token), dest, token)
    return result is True


UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def hf_bytes(text):
    """hf's '18.0G' as bytes."""
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]?)B?", text.strip())
    return int(float(m.group(1)) * UNITS[m.group(2)]) if m else 0


def hf_dry_run(args):
    """(files to fetch, their bytes) — or None when hf cannot tell."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["hf", "download", *args, "--local-dir", tmp, "--dry-run"],
                           capture_output=True, text=True)
    m = re.search(r"Will download (\d+) files? \(out of \d+\) totalling ([\d.]+\s*\w*)", r.stdout + r.stderr)
    return (int(m.group(1)), hf_bytes(m.group(2).rstrip("."))) if m else None


def cmd_check(wf_dir, models_dir, pattern=None, sources=None):
    """What `download` would fetch, fetching nothing. Exit 1 when something is missing."""
    wfs = workflows(wf_dir, pattern)
    if not wfs:
        print(f"  no workflow in {wf_dir}" + (f" matches '{pattern}'" if pattern else ""))
        return 2
    urls, users = collect(wfs)
    recipes = {}
    if sources and Path(sources).is_file():
        recipes = {k: v for k, v in json.loads(Path(sources).read_text()).items() if not k.startswith("_")}
    models = Path(models_dir)
    missing, known, unknown = 0, 0, 0
    for key, u in urls.items():
        if present(models / key):
            continue
        missing += 1
        if key in recipes:
            n = recipes[key].get("bytes") or 0
            known += n
            print(f"  missing  {gib(n):>8}  {key}  (built from {recipes[key]['repo']})")
            continue
        got = None
        url = u[0] if u else ""
        m, r = HF_URL.match(url), HF_REPO.match(url)
        if m:
            repo, rev, path = m.groups()
            got = hf_dry_run([repo, urllib.parse.unquote(path), "--revision", rev])
        elif r:
            repo, rev = r.groups()
            got = hf_dry_run([repo] + (["--revision", rev] if rev else []))
        if got:
            known += got[1]
        else:
            unknown += 1
        size = gib(got[1]) if got else "?"
        where = urllib.parse.urlsplit(url).hostname or "no URL — place it by hand"
        who = users[key][0].removesuffix(".json") if len(users[key]) == 1 else f"{len(users[key])} workflows"
        print(f"  missing  {size:>8}  {key}  ({where}; for {who})")
    print(f"  {len(urls)} models: {len(urls) - missing} present, {missing} to download"
          + (f", {gib(known)}" if missing else "")
          + (f" plus {unknown} of unknown size" if unknown else ""))
    return 1 if missing else 0


def hf_api(url, body=None):
    """GET, or POST a JSON body, to the Hub API — with the hf login's token,
    which gated repos need."""
    token = os.environ.get("HF_TOKEN") or ""
    tf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    if not token and tf.is_file():
        token = tf.read_text().strip()
    headers = {"User-Agent": "llmctl"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def checksum_cache():
    return Path(os.environ.get("LLMCTL_CHECKSUMS",
                               os.path.expanduser("~/.local/state/llmctl/checksums.tsv")))


def sha256_files(files):
    """{path: sha256} for local files, read once and cached by path, size and
    mtime — a model file changes only when it is replaced."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    cache_file, cache = checksum_cache(), {}
    if cache_file.is_file():
        for line in cache_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                cache[parts[0]] = parts[1:]
    def key(f):
        st = f.stat()
        return [str(st.st_size), str(st.st_mtime_ns)]
    todo = [f for f in files if cache.get(str(f), [None, None])[:2] != key(f)]
    if todo:
        total = sum(f.stat().st_size for f in todo)
        print(f"  checksums: reading {len(todo)} file(s), {gib(total)} — once, then cached",
              file=sys.stderr, flush=True)
        def one(f):
            h = hashlib.sha256()
            with open(f, "rb", buffering=0) as fh:
                while chunk := fh.read(64 << 20):
                    h.update(chunk)
            return f, h.hexdigest()
        with ThreadPoolExecutor(4) as pool:
            for f, digest in pool.map(one, todo):
                cache[str(f)] = key(f) + [digest]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("".join(f"{p}\t{v[0]}\t{v[1]}\t{v[2]}\n" for p, v in sorted(cache.items())))
    return {f: cache[str(f)][2] for f in files}


def remote_files(repo, rev, paths=None):
    """{path: (size, sha256 or None, date of the last commit that changed it)}."""
    if paths is None:
        entries = hf_api(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true")
    else:
        entries = hf_api(f"https://huggingface.co/api/models/{repo}/paths-info/{rev}",
                         {"paths": sorted(paths), "expand": True})
    return {e["path"]: (e.get("size"), (e.get("lfs") or {}).get("oid"),
                        ((e.get("lastCommit") or {}).get("date") or "")[:10])
            for e in entries if e.get("type", "file") == "file"}


def cmd_outdated(wf_dir, models_dir, sources=None, quick=False):
    """Which present models are no longer what their Hugging Face source holds.
    Each file is compared by SHA-256 with the one the repo holds now (read once,
    then cached), or with --quick by size alone; the date of the last change in
    the repo is shown beside. A whole repo (a directory) is compared file by
    file. Built models (sources.json) come from a pinned revision and cannot
    drift; other hosts cannot be asked. Exit 1 when something changed."""
    import datetime
    urls, _ = collect(workflows(wf_dir))
    recipes = set()
    if sources and Path(sources).is_file():
        recipes = {k for k in json.loads(Path(sources).read_text()) if not k.startswith("_")}
    models = Path(models_dir)
    # key -> candidates: [(repo, rev, [(local file, remote path)])]; any candidate matching = current
    cands, unchecked = {}, []
    wanted = {}                                   # (repo, rev) -> remote paths, or None for a whole tree
    for key, u in urls.items():
        dest = models / key
        if not present(dest) or key in recipes:
            continue
        for x in u:
            m, r = HF_URL.match(x), HF_REPO.match(x)
            if m and dest.is_file():
                repo, rev, path = m.groups()
                path = urllib.parse.unquote(path)
                cands.setdefault(key, []).append((repo, rev, [(dest, path)]))
                if wanted.get((repo, rev), set()) is not None:
                    wanted.setdefault((repo, rev), set()).add(path)
            elif r and dest.is_dir():
                repo, rev = r.groups()
                files = [(f, str(f.relative_to(dest))) for f in sorted(dest.rglob("*"))
                         if f.is_file() and ".cache" not in f.parts]
                cands.setdefault(key, []).append((repo, rev or "main", files, "tree"))
                wanted[(repo, rev or "main")] = None
        if key not in cands:
            unchecked.append(f"{key} ({urllib.parse.urlsplit(u[0]).hostname if u else 'no URL'})")
    remote, failed = {}, {}
    for (repo, rev), paths in wanted.items():
        try:
            remote[(repo, rev)] = remote_files(repo, rev, paths)
        except OSError as e:
            failed[(repo, rev)] = str(e)
    local_sum = {}
    if not quick:
        need = [f for cs in cands.values() for c in cs if (c[0], c[1]) in remote
                for f, p in c[2] if remote[(c[0], c[1])].get(p, (0, None))[1]]
        local_sum = sha256_files(sorted(set(need)))
    changed, current = [], 0
    for key, cs in cands.items():
        notes, ok_any, asked = [], False, False
        for c in cs:
            repo, rev, files = c[0], c[1], c[2]
            if (repo, rev) in failed:
                notes.append(f"{repo}: {failed[(repo, rev)]}"); continue
            asked = True
            rem = remote[(repo, rev)]
            prefix = "" if len(c) < 4 else None     # a tree: its files are the repo's files
            diffs = []
            for f, p in files:
                size, sha, date = rem.get(p, (None, None, ""))
                st = f.stat()
                here = datetime.date.fromtimestamp(st.st_mtime).isoformat()
                if size is None:
                    diffs.append(f"{p}: gone from {repo}")
                elif not quick and sha and local_sum.get(f) != sha:
                    diffs.append(f"{p}: other content in {repo} (changed {date}, here since {here})")
                elif size != st.st_size:
                    diffs.append(f"{p}: {st.st_size:,} bytes here, {size:,} in {repo} (changed {date})")
                elif quick and date and date > here:
                    diffs.append(f"{p}: changed in {repo} on {date}, after this copy ({here}) — size alike; run without --quick")
            if len(c) == 4:
                missing = sorted(set(rem) - {p for _, p in files})
                diffs += [f"{p}: new in {repo}" for p in missing]
            if not diffs:
                ok_any = True; break
            notes.append("; ".join(diffs[:3]) + (" ..." if len(diffs) > 3 else ""))
        if ok_any:
            current += 1
        elif asked:
            changed.append(f"{key}: {notes[-1]}")
        else:
            unchecked.append(f"{key} ({notes[0]})")
    for c in changed:
        print(f"  changed   {c}")
    for u in unchecked:
        print(f"  unchecked {u}")
    how = "size and date" if quick else "SHA-256"
    print(f"  {current + len(changed)} models checked by {how}: {current} current, {len(changed)} changed"
          + (f", {len(unchecked)} not checkable" if unchecked else ""))
    return 1 if changed else 0


def read_header(path):
    """(header dict without __metadata__, byte offset where the data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def write_safetensors(dest, tensors, metadata):
    """tensors: [(name, dtype, shape, bytes-like or (path, pos, size))], written
    in name order, 8-byte aligned, to dest via a .part file."""
    header, offset = {"__metadata__": metadata}, 0
    for name, dtype, shape, data in tensors:
        size = data[2] if isinstance(data, tuple) else len(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)       # the data starts 8-byte aligned
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    writing {dest.name} ({gib(offset)})")
    with open(part, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for _, _, _, data in tensors:
            if not isinstance(data, tuple):
                out.write(data)
                continue
            src, pos, size = data
            with open(src, "rb") as f:
                f.seek(pos)
                while size:
                    chunk = f.read(min(size, 64 * 2**20))
                    out.write(chunk)
                    size -= len(chunk)
    os.replace(part, dest)


def bf16_to_f32(np, raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(np, x):
    """Round to nearest even, as torch does."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def convert_qwenimage21_lora(key, recipe, dest, staging):
    """alibaba-pai's Qwen-Image 2.1 LoRAs are for diffusers only: keys without
    ".weight", separate gate_layer/proj in each block's MLP, and one proj_out
    per sampling step. ComfyUI fuses the MLP into gate_up = [gate; up] and has
    one proj_out. So: the direct modules renamed; each gate_layer/proj pair
    stacked into one LoRA on gate_up (rank r+r, block-diagonal up); proj_out as
    the mean of its steps (they differ by ~1%); the norms, which it carries
    unchanged, left out."""
    import numpy as np
    local = staging / recipe["repo"]
    print(f"    converted from {recipe['repo']}@{recipe['revision'][:12]} (diffusers LoRA → ComfyUI)")
    cmd = ["hf", "download", recipe["repo"], recipe["file"], "--revision", recipe["revision"],
           "--local-dir", str(local)]
    if subprocess.run(cmd).returncode != 0:
        return False
    src = local / recipe["file"]
    header, start = read_header(src)
    if {t["dtype"] for t in header.values()} != {"BF16"}:
        print("    expected a bf16 LoRA — not converting it", file=sys.stderr)
        return False
    def raw(name):
        a, b = header[name]["data_offsets"]
        with open(src, "rb") as f:
            f.seek(start + a)
            return f.read(b - a)
    def ref(name):
        a, b = header[name]["data_offsets"]
        return (src, start + a, b - a)
    mods = sorted({k.rsplit(".lora_", 1)[0] for k in header if ".lora_" in k})
    out = []
    for m in mods:
        if m.endswith(".img_mlp.gate_layer") or m.endswith(".img_mlp.proj"):
            continue
        for part in ("down", "up"):
            k = f"{m}.lora_{part}"
            out.append((f"diffusion_model.{m}.lora_{part}.weight", "BF16", header[k]["shape"], ref(k)))
    for m in mods:
        if not m.endswith(".img_mlp.gate_layer"):
            continue
        b = m[:-len(".gate_layer")]
        g, p = header[f"{b}.gate_layer.lora_up"]["shape"], header[f"{b}.proj.lora_up"]["shape"]
        rg, rp = header[f"{b}.gate_layer.lora_down"]["shape"][0], header[f"{b}.proj.lora_down"]["shape"][0]
        down = raw(f"{b}.gate_layer.lora_down") + raw(f"{b}.proj.lora_down")        # rows stacked
        up = np.zeros((g[0] + p[0], rg + rp), dtype=np.uint16)                        # bf16 zero is 0x0000
        up[:g[0], :rg] = np.frombuffer(raw(f"{b}.gate_layer.lora_up"), dtype=np.uint16).reshape(g)
        up[g[0]:, rg:] = np.frombuffer(raw(f"{b}.proj.lora_up"), dtype=np.uint16).reshape(p)
        in_dim = header[f"{b}.gate_layer.lora_down"]["shape"][1]
        out.append((f"diffusion_model.{b}.gate_up.lora_down.weight", "BF16", [rg + rp, in_dim], down))
        out.append((f"diffusion_model.{b}.gate_up.lora_up.weight", "BF16", list(up.shape), up.tobytes()))
    shape = header["proj_out.weight"]["shape"]                                        # [steps, out, in]
    mean = bf16_to_f32(np, raw("proj_out.weight")).reshape(shape).mean(axis=0)
    out.append(("diffusion_model.proj_out.set_weight", "BF16", shape[1:], f32_to_bf16(np, mean).tobytes()))
    out.sort(key=lambda t: t[0])
    if len(out) != recipe["tensors"]:
        print(f"    expected {recipe['tensors']} tensors, made {len(out)} — not writing it", file=sys.stderr)
        return False
    write_safetensors(dest, out, {"converted_from": f"{recipe['repo']}@{recipe['revision']}/{recipe['file']}"})
    shutil.rmtree(local, ignore_errors=True)
    return True


def build(key, recipe, dest, staging):
    """Make a model from a recipe in sources.json: shards joined and renamed
    (the default), or a conversion named by "kind"."""
    if recipe.get("kind") == "qwenimage21_prefused_lora":
        return convert_qwenimage21_lora(key, recipe, dest, staging)
    local = staging / recipe["repo"]
    print(f"    built from {recipe['repo']}@{recipe['revision'][:12]} "
          f"({len(recipe['shards'])} shards, tensors renamed)")
    cmd = ["hf", "download", recipe["repo"], *recipe["shards"],
           "--revision", recipe["revision"], "--local-dir", str(local)]
    if subprocess.run(cmd).returncode != 0:
        return False

    def renamed(name):
        for old, new in recipe.get("rename", []):
            if name.startswith(old):
                return new + name[len(old):]
        return name

    # Every tensor: new name, source shard, source byte range.
    tensors = []
    for shard in recipe["shards"]:
        header, start = read_header(local / shard)
        for name, t in header.items():
            a, b = t["data_offsets"]
            tensors.append((renamed(name), t["dtype"], t["shape"], local / shard, start + a, b - a))
    tensors.sort()
    names = [t[0] for t in tensors]
    total = sum(t[5] for t in tensors)
    if len(set(names)) != len(names):
        print("    two tensors end up with the same name — the recipe's rename is wrong", file=sys.stderr)
        return False
    if len(tensors) != recipe["tensors"] or total != recipe["bytes"]:
        print(f"    expected {recipe['tensors']} tensors / {recipe['bytes']} bytes, the shards hold "
              f"{len(tensors)} / {total} — not building it", file=sys.stderr)
        return False

    write_safetensors(dest, [(n, dt, sh, (src, pos, size)) for n, dt, sh, src, pos, size in tensors],
                      {"format": "pt"})
    shutil.rmtree(local, ignore_errors=True)
    return True


def cmd_download(wf_dir, models_dir, pattern=None, sources=None):
    wfs = workflows(wf_dir, pattern)
    if not wfs:
        print(f"  no workflow in {wf_dir}" + (f" matches '{pattern}'" if pattern else ""))
        return 1
    print(f"  workflows: {', '.join(n.removesuffix('.json') for n, _ in wfs)}")
    urls, users = collect(wfs)
    recipes = {}
    if sources and Path(sources).is_file():
        recipes = {k: v for k, v in json.loads(Path(sources).read_text()).items() if not k.startswith("_")}
    models = Path(models_dir)
    staging = models / STAGING
    failed, fetched, n_present = [], 0, 0
    for key, u in urls.items():
        dest = models / key
        if present(dest):
            n_present += 1
            continue
        if key in recipes:
            print(f"  {key}")
            if build(key, recipes[key], dest, staging):
                fetched += 1
            else:
                failed.append(key)
            continue
        if not u:
            print(f"  {key}: no URL in the workflow ({', '.join(users[key])}) — place it by hand")
            failed.append(key)
            continue
        if len(u) > 1:
            print(f"  {key}: {len(u)} sources named, trying them in order")
        print(f"  {key}")
        for url in u:
            print(f"    from {url}")
            if HF_URL.match(url):
                ok = fetch_hf(url, dest, staging)
            elif HF_REPO.match(url):
                ok = fetch_repo(url, dest, staging)
            else:
                ok = fetch_plain(url, dest)
            if ok:
                fetched += 1
                break
        else:
            print(f"    failed. A gated repo (black-forest-labs, ...) needs its licence accepted on"
                  f" huggingface.co and `hf auth login`.")
            failed.append(key)
    if not failed:
        # hf keeps what it needs to resume in here; only a complete run may drop it.
        shutil.rmtree(staging, ignore_errors=True)
    print(f"  {len(urls)} models: {n_present} present, {fetched} downloaded, {len(failed)} failed")
    return 1 if failed else 0


def main(argv):
    sources = None
    if "--sources" in argv:
        i = argv.index("--sources")
        sources = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if len(argv) < 3 or argv[0] not in ("list", "sizes", "download", "orphans", "check", "outdated"):
        print("usage: comfyui_models.py list|sizes|orphans WORKFLOWS_DIR MODELS_DIR\n"
              "       comfyui_models.py download|check WORKFLOWS_DIR MODELS_DIR [PATTERN] [--sources FILE]",
              file=sys.stderr)
        return 2
    if argv[0] == "list":
        return cmd_list(argv[1], argv[2])
    if argv[0] == "outdated":
        quick = "--quick" in argv
        argv = [x for x in argv if x != "--quick"]
        return cmd_outdated(argv[1], argv[2], sources, quick)
    if argv[0] == "check":
        return cmd_check(argv[1], argv[2], argv[3] if len(argv) > 3 else None, sources)
    if argv[0] == "orphans":
        return cmd_orphans(argv[1], argv[2])
    if argv[0] == "sizes":
        return cmd_sizes(argv[1], argv[2])
    return cmd_download(argv[1], argv[2], argv[3] if len(argv) > 3 else None, sources)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
