#!/usr/bin/env python3
"""watch.py CONF TEMPLATES_DIR — new model versions and awaited ComfyUI templates (see watch.conf)."""
import json, re, shutil, subprocess, sys, urllib.request
from pathlib import Path


def version_key(v):
    return [int(x) for x in re.findall(r"\d+", v)]


def hf_models(author):
    url = f"https://huggingface.co/api/models?author={author}&limit=200&sort=createdAt&direction=-1"
    with urllib.request.urlopen(url, timeout=30) as r:
        return [m["id"].split("/", 1)[1] for m in json.load(r)]


def main(conf, templates):
    names = sorted(p.stem for p in Path(templates).glob("*.json")) if Path(templates).is_dir() else []
    found = 0
    for line in Path(conf).read_text().splitlines():
        parts = line.split(None, 4) if not line.lstrip().startswith("#") else []
        if not parts:
            continue
        if parts[0] == "model" and len(parts) == 5:
            _, author, rx, used, label = parts
            try:
                repos = hf_models(author)
            except OSError as e:
                print(f"  {label}: cannot ask Hugging Face ({e})")
                continue
            vers = {}
            for r in repos:
                m = re.match(rx, r)
                if m and m.group(1):
                    vers.setdefault(m.group(1), r)
            newer = sorted((v for v in vers if version_key(v) > version_key(used)), key=version_key)
            if newer:
                print(f"  {label}: {newer[-1]} is out ({author}/{vers[newer[-1]]}); in use {used}")
                found += 1
            else:
                print(f"  {label}: {used} in use — nothing newer from {author}")
        elif parts[0] == "template" and len(parts) >= 3:
            rx, issue = parts[1], parts[2]
            what = parts[3] + (" " + parts[4] if len(parts) > 4 else "") if len(parts) > 3 else ""
            hits = [n for n in names if re.search(rx, n)]
            if hits:
                print(f"  {issue}: ComfyUI has {', '.join(hits[:3])} — {what}")
                found += 1
            else:
                print(f"  {issue}: waiting — {what}")
    if not names:
        print("  (ComfyUI's templates not found — their checks need ComfyUI set up)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
