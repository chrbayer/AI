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


def cmd_list(wf_dir, models_dir):
    wfs = workflows(wf_dir)
    if not wfs:
        print(f"    no workflows in {wf_dir}")
        return 0
    models = Path(models_dir)
    width = max(len(n) for n, _ in wfs)
    referenced = set()
    for name, wf in wfs:
        r = refs(wf)
        referenced.update(r)
        have = [k for k in r if present(models / k)]
        total = sum(size(models / k) for k in have)
        missing = [k.split("/", 1)[1] for k in r if k not in have]
        state = "ready" if not missing else "missing: " + ", ".join(missing)
        print(f"    {name.removesuffix('.json'):<{width - 5}}  {len(have)}/{len(r)} models, "
              f"{gib(total):>8}  {state}")
    orphans = []
    if models.is_dir():
        for sub in sorted(p for p in models.iterdir() if p.is_dir() and p.name != STAGING):
            for f in sorted(sub.rglob("*")):
                key = str(f.relative_to(models))
                inside = any(key.startswith(ref + "/") for ref in referenced)   # part of a repo snapshot
                if f.is_file() and f.suffix in MODEL_SUFFIXES and key not in referenced and not inside:
                    orphans.append(f"{key} ({gib(f.stat().st_size)})")
    if orphans:
        print("    named by no workflow (kept):")
        for o in orphans:
            print(f"      {o}")
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
    if len(argv) < 3 or argv[0] not in ("list", "sizes", "download"):
        print("usage: comfyui_models.py list|sizes WORKFLOWS_DIR MODELS_DIR\n"
              "       comfyui_models.py download WORKFLOWS_DIR MODELS_DIR [PATTERN] [--sources FILE]",
              file=sys.stderr)
        return 2
    if argv[0] == "list":
        return cmd_list(argv[1], argv[2])
    if argv[0] == "sizes":
        return cmd_sizes(argv[1], argv[2])
    return cmd_download(argv[1], argv[2], argv[3] if len(argv) > 3 else None, sources)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
