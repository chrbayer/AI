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

A file made here from a repo's files — a GGUF quantized down with llama-quantize
from one only offered too big (Q8_0) — gets the same companion: its sources
take the parts' place, and `derive` records how it was made. Download skips
the sources while the Hub has them unchanged; when it changes, download fetches
them, and `join` makes the file again from them by the same recipe.

    hf_parts.py excludes REPO DIR    parts whose companion still matches the Hub
    hf_parts.py check REPO DIR       one line per joined or made file: current or changed
    hf_parts.py join REPO DIR        join what is complete, verify, write, delete;
                                     make again what was made, where its sources are back
    hf_parts.py derive REPO DIR OUT TYPE [--adopt]
                                     OUT from the repo's GGUF here (all its shards),
                                     llama-quantize --allow-requantize to TYPE ($LLAMA_QUANTIZE),
                                     the sources checked, recorded and deleted; --adopt takes an
                                     OUT already made from them instead of making it again
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


def companions(directory, repo=None):
    """{companion path: its content} for the joined files under DIR — those of
    REPO only, when given: a directory may hold files of a second repo too (a
    vision projector beside the model), which do not have these parts."""
    out = {}
    for p in sorted(Path(directory).rglob("*" + COMPANION)):
        if ".cache" in p.parts:
            continue
        try:
            c = json.loads(p.read_text())
        except ValueError:
            continue
        if repo is None or c.get("repo", repo) == repo:
            out[p] = c
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
    comp = companions(directory, repo)
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
    comp = companions(directory, repo)
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
        how = f"made ({c['derive']['quantize']}) from" if c.get("derive") else "joined from"
        print(f"{state}\t{joined}\t{len(c.get('parts', []))} file(s)\t{how}")
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


SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def quantize(sources, out, qtype):
    """llama-quantize the (first shard of the) sources to qtype, into out."""
    tool = os.environ.get("LLAMA_QUANTIZE", "llama-quantize")
    first = sorted(sources, key=lambda p: p.name)[0]
    tmp = out.with_name(out.name + ".making")
    print(f"  {out.name}: {tool} --allow-requantize {first.name} → {qtype}", flush=True)
    r = subprocess.run([tool, "--allow-requantize", str(first), str(tmp), qtype], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"  {out.name}: llama-quantize failed — sources kept\n{r.stderr[-800:]}")
    os.replace(tmp, out)


def derive_one(directory, repo, out_name, qtype, adopt=False, sources=None):
    """Check the sources against the Hub, make (or adopt) out, write the
    companion with the recipe, delete the sources."""
    directory = Path(directory)
    hub, sha = hub_files(repo)
    if sources is None:                              # the repo's GGUF files that are here
        sources = sorted(directory / n for n in hub if n.endswith(".gguf") and "mmproj" not in n
                         and (directory / n).is_file() and n != out_name)
    if not sources:
        raise SystemExit(f"  {out_name}: none of {repo}'s GGUF files are here to make it from")
    names = [p.name for p in sources]
    sizes = [p.stat().st_size for p in sources]
    print(f"  {out_name}: from {len(sources)} file(s), {sum(sizes) / 1e9:.1f} GB — checking them against the Hub",
          flush=True)
    each = [sha256(p) for p in sources]
    for name, size, digest in zip(names, sizes, each):
        remote = hub.get(name)
        if not remote or remote["size"] != size or remote["sha256"] != digest:
            raise SystemExit(f"  {name}: differs from {repo} (or is not there) — sources kept")
    out = directory / out_name
    if not (adopt and out.exists()):
        quantize(sources, out, qtype)
    else:
        print(f"  {out_name}: taken as it is (made from these sources before)")
    companion = {"repo": repo, "revision": sha, "file": out_name, "size": out.stat().st_size, "sha256": sha256(out),
                 "derive": {"quantize": qtype, "tool": "llama-quantize --allow-requantize"},
                 "parts": [{"name": n, "size": z, "sha256": d} for n, z, d in zip(names, sizes, each)]}
    (directory / (out_name + COMPANION)).write_text(json.dumps(companion, indent=1) + "\n")
    forget_parts(directory, "", names)
    print(f"  {out_name}: sources deleted; {out_name + COMPANION} keeps their hashes and the recipe")


def cmd_derive(repo, directory, out_name, qtype, adopt=False):
    derive_one(directory, repo, out_name, qtype, adopt)
    return 0


def cmd_join(repo, directory):
    directory = Path(directory)
    # Made files whose sources download fetched again (they changed on the Hub).
    for path, c in companions(directory, repo).items():
        d = c.get("derive")
        here = [path.parent / p["name"] for p in c.get("parts", [])]
        if d and here and all(p.is_file() for p in here):
            derive_one(directory, repo, c["file"], d["quantize"], sources=here)
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
    if len(argv) >= 6 and argv[1] == "derive":
        return cmd_derive(argv[2], argv[3], argv[4], argv[5], adopt="--adopt" in argv[6:])
    if len(argv) != 4 or argv[1] not in ("excludes", "check", "join"):
        print(__doc__, file=sys.stderr)
        return 2
    return {"excludes": cmd_excludes, "check": cmd_check, "join": cmd_join}[argv[1]](argv[2], argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
