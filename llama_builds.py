#!/usr/bin/env python3
"""Which prebuilt llama-server builds belong to llama.cpp's newest stable release.

    llama_builds.py resolve [GFX]     one JSON object: the release and the two builds
    llama_builds.py gfx               this machine's GPU target (gfx1151), from KFD

ggml-org tags a stable vX.Y.Z now and then and publishes a build (bNNNNN) of
every master commit with Linux packages, among them ubuntu-vulkan-x64: the
Vulkan build for a release is the one made of the release's own commit.

ggml-org's ROCm package takes ROCm from the system; lemonade-sdk/llamacpp-rocm
ships one per GPU target with its ROCm runtime inside. Its builds are master
nightlies that name their llama.cpp commit by five hex digits: the build for a
release is the oldest one whose commit is at the release or past it.
"""
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

GGML = "ggml-org/llama.cpp"
LEMONADE = "lemonade-sdk/llamacpp-rocm"


def gh(path):
    """GitHub's API, through gh when it is logged in (5000 requests an hour, not 60)."""
    if shutil.which("gh") and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0:
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise OSError(f"GitHub: {path}: {r.stderr.strip()[:200]}")
        return json.loads(r.stdout)
    with urllib.request.urlopen(f"https://api.github.com/{path}", timeout=30) as r:
        return json.load(r)


def gfx_name(version):
    """KFD's gfx_target_version (major, minor, stepping as decimal pairs) as a target:
    110501 -> gfx1151, 100300 -> gfx1030, 90010 -> gfx90a."""
    return f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"


def gfx(topology="/sys/class/kfd/kfd/topology/nodes"):
    """This machine's GPU target; the CPU node reports 0."""
    for p in sorted(Path(topology).glob("*/properties")):
        for line in p.read_text().splitlines():
            k, _, v = line.partition(" ")
            if k == "gfx_target_version" and int(v):
                return gfx_name(int(v))
    return None


def stable():
    rel = gh(f"repos/{GGML}/releases/latest")
    tag = rel["tag_name"]
    sha = gh(f"repos/{GGML}/commits/{tag}")["sha"]
    return tag, sha, rel["published_at"]


def vulkan_build(sha, since):
    """The bNNNNN release made of the commit, with its ubuntu-vulkan-x64 package."""
    stop = datetime.fromisoformat(since.replace("Z", "+00:00")) - timedelta(days=3)
    for page in range(1, 15):
        rels = gh(f"repos/{GGML}/releases?per_page=100&page={page}")
        for r in rels:
            if r.get("target_commitish") == sha and r["tag_name"].startswith("b"):
                for a in r["assets"]:
                    if a["name"].endswith("-bin-ubuntu-vulkan-x64.tar.gz"):
                        return {"build": r["tag_name"], "asset": a["name"], "url": a["browser_download_url"],
                                "size": a["size"]}
        if not rels or datetime.fromisoformat(rels[-1]["published_at"].replace("Z", "+00:00")) < stop:
            break
    return None


def lemonade_asset(assets, target):
    """The zip for this GPU: its own name (gfx1151), or its family (gfx103X for gfx1030)."""
    names = {a["name"]: a for a in assets if "-ubuntu-rocm-" in a["name"]}
    for want in (target, target[:-1] + "X"):
        for n, a in names.items():
            if n.endswith(f"-ubuntu-rocm-{want}-x64.zip"):
                return a
    return None


def rocm_build(tag, sha, target):
    """lemonade's oldest build at or past the release commit, for this GPU target."""
    rels = gh(f"repos/{LEMONADE}/releases?per_page=40")
    since = (datetime.fromisoformat(gh(f"repos/{GGML}/commits/{sha}")["commit"]["committer"]["date"]
                                    .replace("Z", "+00:00")) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    master = [c["sha"] for page in (1, 2, 3)
              for c in gh(f"repos/{GGML}/commits?sha=master&since={since}&per_page=100&page={page}")]
    found = []
    for r in rels:
        short = next((line.split(":", 1)[1].strip(" *") for line in (r.get("body") or "").splitlines()
                      if "Commit Hash" in line), "")
        full = next((s for s in master if short and s.startswith(short.lower())), None)
        if not full:
            continue
        status = gh(f"repos/{GGML}/compare/{tag}...{full}")["status"]
        if status in ("identical", "ahead"):
            found.append((r["published_at"], r, full))
    for _, r, full in sorted(found, key=lambda x: x[0]):
        a = lemonade_asset(r["assets"], target)
        if a:
            return {"build": r["tag_name"], "commit": full, "asset": a["name"], "url": a["browser_download_url"],
                    "size": a["size"], "gfx": target}
    return None


def main(argv):
    if len(argv) >= 2 and argv[1] == "gfx":
        print(gfx() or "")
        return 0
    if len(argv) < 2 or argv[1] != "resolve":
        print(__doc__, file=sys.stderr)
        return 2
    target = argv[2] if len(argv) > 2 else gfx()
    tag, sha, published = stable()
    out = {"tag": tag, "commit": sha, "vulkan": vulkan_build(sha, published),
           "rocm": rocm_build(tag, sha, target) if target else None}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
