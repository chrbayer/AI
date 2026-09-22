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
    comfyui_models.py download WORKFLOWS_DIR MODELS_DIR [PATTERN]

PATTERN restricts download to the workflows whose file name contains it
(case-insensitive). Files already present are never fetched again, and nothing
is ever deleted: `list` only reports files no workflow names any more.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}
HF_URL = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
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
        present = [k for k in r if (models / k).is_file()]
        size = sum((models / k).stat().st_size for k in present)
        missing = [k.split("/", 1)[1] for k in r if k not in present]
        state = "ready" if not missing else "missing: " + ", ".join(missing)
        print(f"    {name.removesuffix('.json'):<{width - 5}}  {len(present)}/{len(r)} models, "
              f"{gib(size):>8}  {state}")
    orphans = []
    if models.is_dir():
        for sub in sorted(p for p in models.iterdir() if p.is_dir() and p.name != STAGING):
            for f in sorted(sub.rglob("*")):
                key = str(f.relative_to(models))
                if f.is_file() and f.suffix in MODEL_SUFFIXES and key not in referenced:
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


def fetch_hf(url, dest, staging):
    m = HF_URL.match(url)
    assert m
    repo, rev, path = m.groups()
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


def fetch_plain(url, dest):
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 16 * 2**20)
    except OSError as e:
        print(f"      {e}", file=sys.stderr)
        part.unlink(missing_ok=True)
        return False
    os.replace(part, dest)
    return True


def cmd_download(wf_dir, models_dir, pattern=None):
    wfs = workflows(wf_dir, pattern)
    if not wfs:
        print(f"  no workflow in {wf_dir}" + (f" matches '{pattern}'" if pattern else ""))
        return 1
    print(f"  workflows: {', '.join(n.removesuffix('.json') for n, _ in wfs)}")
    urls, users = collect(wfs)
    models = Path(models_dir)
    staging = models / STAGING
    failed, fetched, present = [], 0, 0
    for key, u in urls.items():
        dest = models / key
        if dest.is_file():
            present += 1
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
            ok = fetch_hf(url, dest, staging) if HF_URL.match(url) else fetch_plain(url, dest)
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
    print(f"  {len(urls)} models: {present} present, {fetched} downloaded, {len(failed)} failed")
    return 1 if failed else 0


def main(argv):
    if len(argv) < 3 or argv[0] not in ("list", "download"):
        print("usage: comfyui_models.py list WORKFLOWS_DIR MODELS_DIR\n"
              "       comfyui_models.py download WORKFLOWS_DIR MODELS_DIR [PATTERN]", file=sys.stderr)
        return 2
    if argv[0] == "list":
        return cmd_list(argv[1], argv[2])
    return cmd_download(argv[1], argv[2], argv[3] if len(argv) > 3 else None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
