#!/usr/bin/env python3
"""Models a Hugging Face repo holds as plain byte parts (X.gguf.part1of2, …).

download fetches the parts, `join` puts them together with cat — on btrfs a
reflink, no second copy — after checking each part against the SHA-256 the
Hub keeps for it (its LFS oid), writes X.gguf.parts.json beside the joined
file with the parts' names, sizes and hashes, and deletes the parts. From
then on the companion stands for them: `excludes` names the parts download
must not fetch again, `check` compares the companion with the Hub instead of
local parts. gguf-split shards (X-00001-of-00002.gguf) are not parts —
llama.cpp loads those itself — and are left alone.

    hf_parts.py excludes REPO DIR    parts whose companion still matches the Hub
    hf_parts.py check REPO DIR       one line per joined file: current or changed
    hf_parts.py join REPO DIR        join what is complete, verify, write, delete
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfyui_models import hf_api  # noqa: E402

PART = re.compile(r"^(?P<file>.+)\.part(?P<n>\d+)of(?P<m>\d+)$")
COMPANION = ".parts.json"
CHUNK = 64 << 20


def groups(names):
    """{joined name: [part names in order]} for the complete sets among `names`."""
    found = {}
    for name in names:
        m = PART.match(name)
        if m:
            found.setdefault((m["file"], int(m["m"])), {})[int(m["n"])] = name
    return {file: [parts[i] for i in range(1, total + 1)]
            for (file, total), parts in found.items() if sorted(parts) == list(range(1, total + 1))}


def hub_files(repo, revision="main"):
    """{path: {"size", "sha256"}} of the repo, and the commit it was read at."""
    info = hf_api(f"https://huggingface.co/api/models/{repo}/revision/{revision}")
    sha = info.get("sha", revision)
    tree = hf_api(f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
    files = {}
    for e in tree:
        if e.get("type") == "file":
            lfs = e.get("lfs") or {}
            files[e["path"]] = {"size": lfs.get("size", e.get("size")), "sha256": lfs.get("oid")}
    return files, sha


def companions(directory):
    """{companion path: its content} for the joined files under DIR."""
    out = {}
    for p in sorted(Path(directory).rglob("*" + COMPANION)):
        if ".cache" in p.parts:
            continue
        try:
            out[p] = json.loads(p.read_text())
        except ValueError:
            continue
    return out


def matches(companion, hub, rel_dir):
    """Whether the Hub still holds exactly the parts the companion names."""
    for part in companion.get("parts", []):
        remote = hub.get(f"{rel_dir}/{part['name']}" if rel_dir else part["name"])
        if not remote or remote["size"] != part["size"] or remote["sha256"] != part["sha256"]:
            return False
    return bool(companion.get("parts"))


def rel(p, directory):
    r = str(Path(p).parent.relative_to(directory))
    return "" if r == "." else r


def cmd_excludes(repo, directory):
    """Parts download should skip. Without the Hub there is no telling — then the
    companion is trusted, rather than fetching tens of GB on a network hiccup."""
    comp = companions(directory)
    if not comp:
        return 0
    try:
        hub, _ = hub_files(repo)
    except Exception:                                             # noqa: BLE001
        hub = None
    for path, c in comp.items():
        if hub is None or matches(c, hub, rel(path, directory)):
            for part in c.get("parts", []):
                print(f"{rel(path, directory) + '/' if rel(path, directory) else ''}{part['name']}")
    return 0


def cmd_check(repo, directory):
    comp = companions(directory)
    if not comp:
        return 0
    try:
        hub, _ = hub_files(repo)
    except Exception as e:                                        # noqa: BLE001
        print(f"unknown\t{repo}: cannot ask the Hub ({e})")
        return 0
    for path, c in comp.items():
        joined = path.name[: -len(COMPANION)]
        state = "current" if matches(c, hub, rel(path, directory)) else "changed"
        print(f"{state}\t{joined}\t{len(c.get('parts', []))} parts")
    return 0


def hash_parts(paths):
    """SHA-256 of each part and of them all in a row (the joined file), one read."""
    whole, each = hashlib.sha256(), []
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)
                whole.update(chunk)
        each.append(h.hexdigest())
    return each, whole.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def forget_parts(directory, rel_dir, names):
    """The parts and what hf keeps about them in .cache, so it does not count
    them as half-downloaded."""
    base = Path(directory) / rel_dir if rel_dir else Path(directory)
    meta = Path(directory) / ".cache" / "huggingface" / "download" / rel_dir if rel_dir else \
        Path(directory) / ".cache" / "huggingface" / "download"
    for n in names:
        for p in (base / n, meta / (n + ".metadata"), meta / (n + ".lock")):
            p.unlink(missing_ok=True)


def join_one(directory, rel_dir, joined, parts, hub, sha, repo):
    base = Path(directory) / rel_dir if rel_dir else Path(directory)
    paths = [base / p for p in parts]
    sizes = [p.stat().st_size for p in paths]
    print(f"  {joined}: {len(parts)} parts, {sum(sizes) / 1e9:.1f} GB — checking them against the Hub", flush=True)
    each, whole = hash_parts(paths)
    for name, size, digest in zip(parts, sizes, each):
        remote = hub.get(f"{rel_dir}/{name}" if rel_dir else name)
        if not remote:
            raise SystemExit(f"  {name}: not in {repo} — leaving the parts as they are")
        if remote["size"] != size or remote["sha256"] != digest:
            raise SystemExit(f"  {name}: differs from the Hub (size or SHA-256) — leaving the parts as they are")
    target = base / joined
    if target.exists() and target.stat().st_size == sum(sizes) and sha256(target) == whole:
        print(f"  {joined}: already joined from these parts")
    else:
        tmp = target.with_name(target.name + ".joining")
        with open(tmp, "wb") as out:                 # cat copies by copy_file_range: a reflink on btrfs
            subprocess.run(["cat", *map(str, paths)], stdout=out, check=True)
        if tmp.stat().st_size != sum(sizes):
            tmp.unlink()
            raise SystemExit(f"  {joined}: joined size does not add up — parts kept")
        os.replace(tmp, target)
        print(f"  {joined}: joined")
    companion = {"repo": repo, "revision": sha, "file": joined, "size": sum(sizes), "sha256": whole,
                 "parts": [{"name": n, "size": s, "sha256": d} for n, s, d in zip(parts, sizes, each)]}
    (base / (joined + COMPANION)).write_text(json.dumps(companion, indent=1) + "\n")
    forget_parts(directory, rel_dir, parts)
    print(f"  {joined}: parts deleted; {joined + COMPANION} keeps their hashes")


def cmd_join(repo, directory):
    directory = Path(directory)
    local = {}
    for p in directory.rglob("*"):
        if p.is_file() and ".cache" not in p.relative_to(directory).parts and PART.match(p.name):
            local.setdefault(rel(p, directory), []).append(p.name)
    todo = [(d, j, ps) for d, names in local.items() for j, ps in groups(names).items()]
    if not todo:
        return 0
    hub, sha = hub_files(repo)
    for rel_dir, joined, parts in sorted(todo):
        join_one(directory, rel_dir, joined, parts, hub, sha, repo)
    return 0


def main(argv):
    if len(argv) != 4 or argv[1] not in ("excludes", "check", "join"):
        print(__doc__, file=sys.stderr)
        return 2
    return {"excludes": cmd_excludes, "check": cmd_check, "join": cmd_join}[argv[1]](argv[2], argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
