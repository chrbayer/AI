#!/usr/bin/env python3
"""A music video from a song and pictures of the singer, on a running ComfyUI —
`llmctl musicvideo` (#47).

The song is cut into sections (5 s by default). In each the singer sings it — a
picture made to sing along with LTX-2.5 Talking — or, where the sections file
says so, a scene plays from a prompt (LTX-2.5 Video, its own sound dropped).
Every clip is cut to exactly the frames of its section, the clips are joined,
and the whole song goes underneath, so the picture stays on the sound to the
frame from the first section to the last.

    musicvideo.py --url URL --workflows DIR SONG PICTURE [PICTURE ...]
                  [--sections FILE] [--prompt TEXT] [--seconds N] [--size WxH]
                  [--continuous] [--out FILE] [--dry-run]

The pictures take turns, one per singing section. With --continuous a section
starts from the last frame of the one before instead (the first from the first
picture): one continuous take, but faces and colours drift over many sections.

The sections file has one line per section, in order; past its end the
default applies (sing), and # starts a comment. A picture named in a line
(relative to the file) is that section's start, whatever else applies:

    sing                       the singer sings (the default)
    sing: <prompt>             ... with this prompt instead of --prompt
    sing <picture>: <prompt>   ... from this picture
    scene: <prompt>            a scene from the prompt alone (text to video;
                               with --continuous from the frame before), no singing
    scene <picture>: <prompt>  a scene that starts from this picture

Finished sections are kept in OUT.parts/ with what they were made from, so a
run that is interrupted, or run again with a few sections changed, only renders
what is missing. About 6-7 minutes per 5 s section at 1280x704.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

FPS = 24
SING = ("The singer sings this part of the song with feeling, lips moving clearly with the words, small natural "
        "head and body movements; music video.")
TALKING = "LTX-2.5 Talking (int8, distilled).json"
VIDEO = "LTX-2.5 Video (int8, distilled).json"
UPLOAD_DIR = "musicvideo"


def probe(path, *what):
    return subprocess.run(["ffprobe", "-v", "error", *what, "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def read_sections(path, count, prompt):
    """[(kind, picture or None, prompt)] for `count` sections."""
    plan = []
    for n, line in enumerate(Path(path).read_text().splitlines() if path else [], 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        head, _, text = line.partition(":")
        kind, _, picture = head.strip().partition(" ")
        kind, picture, text = kind.lower(), picture.strip() or None, text.strip()
        if kind not in ("sing", "scene") or (kind == "scene" and not text):
            sys.exit(f"{path}:{n}: want 'sing [picture][: prompt]' or 'scene [picture]: prompt', got: {line}")
        if picture:
            picture = str(Path(path).parent / picture)
            if not Path(picture).is_file():
                sys.exit(f"{path}:{n}: no such picture: {picture}")
        plan.append((kind, picture, text or prompt))
    plan = plan[:count]
    return plan + [("sing", None, prompt)] * (count - len(plan))


def plan_sections(plan, total, seconds, pictures, continuous):
    """Section i covers [start, end) of the song and keeps exactly the frames that
    fall in it — counted from the song's start, so they add up to the song's.
    Its start: its own picture, else (continuous) the frame before, else the next
    of the pictures for singing, nothing (text to video) for a scene."""
    sections, sung = [], 0
    for i, (kind, picture, prompt) in enumerate(plan):
        start, end = i * seconds, min((i + 1) * seconds, total)
        frames = round(end * FPS) - round(start * FPS)
        if picture is None and not (continuous and i) and kind == "sing":
            picture = pictures[sung % len(pictures)]
        if picture is None and i == 0 and continuous:
            picture = pictures[0]
        sung += kind == "sing"
        sections.append({"i": i, "kind": kind, "prompt": prompt, "start": start, "end": end, "frames": frames,
                         "picture": picture, "follow": picture is None and continuous})
    return sections


class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def upload(self, path, subfolder=UPLOAD_DIR):
        """The file into ComfyUI's input/<subfolder>/; its name as loaders take it."""
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image", files={"image": (Path(path).name, f)},
                              data={"subfolder": subfolder, "type": "input", "overwrite": "true"}, timeout=120)
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def run(self, graph, dest):
        """Queue a graph, wait, fetch what it made (a picture, video or sound) to dest."""
        r = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=60)
        d = r.json()
        if not d.get("prompt_id") or d.get("node_errors"):
            sys.exit(f"ComfyUI refused the workflow: {r.text[:2000]}")
        pid = d["prompt_id"]
        while True:
            h = requests.get(f"{self.url}/history/{pid}", timeout=60).json().get(pid)
            if h and h["status"].get("status_str") == "error":
                sys.exit(f"ComfyUI failed: {json.dumps(h['status']['messages'])[-2000:]}")
            if h and h["status"].get("completed"):
                break
            time.sleep(3)
        out = next(i for o in h["outputs"].values() for i in o.get("images", []) + o.get("videos", []) + o.get("audio", []))
        r = requests.get(f"{self.url}/view", params={"filename": out["filename"], "subfolder": out.get("subfolder", ""),
                                                      "type": out.get("type", "output")}, timeout=600)
        r.raise_for_status()
        Path(dest).write_bytes(r.content)


def node(g, cls, title=None):
    for n in g.values():
        if n["class_type"] == cls and (title is None or n.get("_meta", {}).get("title") == title):
            return n
    sys.exit(f"the workflow has no {cls} {title or ''}")


def graph(workflows, kind, picture, prompt, audio, seconds, size, prefix, placeholder):
    g = json.loads((Path(workflows) / (TALKING if kind == "sing" else VIDEO)).read_text())
    # A scene from text alone still names a picture: ComfyUI checks the loader's
    # file even where the switch passes it by.
    node(g, "LoadImage")["inputs"]["image"] = picture or placeholder
    if not picture:
        node(g, "PrimitiveBoolean", "Switch to Text to Video?")["inputs"]["value"] = True
    node(g, "PrimitiveStringMultiline", "Prompt")["inputs"]["value"] = prompt
    if kind == "sing":
        node(g, "LoadAudio")["inputs"]["audio"] = audio
    else:
        node(g, "PrimitiveInt", "Duration")["inputs"]["value"] = seconds
    if size:
        node(g, "PrimitiveInt", "Width")["inputs"]["value"] = size[0]
        node(g, "PrimitiveInt", "Height")["inputs"]["value"] = size[1]
    node(g, "SaveVideo")["inputs"]["filename_prefix"] = prefix
    return g


def main():
    ap = argparse.ArgumentParser(description="a music video from a song and pictures of the singer")
    ap.add_argument("song")
    ap.add_argument("pictures", nargs="+")
    # llmctl passes these: the ComfyUI slot and the API video workflows.
    ap.add_argument("--url", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--workflows", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--sections", help="what each section shows (see above)")
    ap.add_argument("--prompt", default=SING, help="the prompt of a singing section")
    ap.add_argument("--seconds", type=int, default=5, help="length of a section (1-10, default 5)")
    ap.add_argument("--size", help="WxH, multiples of 32 (default: 16:9 at ~0.9 MP, 1280x704)")
    ap.add_argument("--continuous", action="store_true", help="each section starts from the last frame of the one before")
    ap.add_argument("--out", help="the video (default: <song>.mp4 beside the song)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan only")
    a = ap.parse_args()

    for f in [a.song, *a.pictures, *([a.sections] if a.sections else [])]:
        if not Path(f).is_file():
            sys.exit(f"no such file: {f}")
    if not 1 <= a.seconds <= 10:
        sys.exit("--seconds: 1 to 10")
    size = None
    if a.size:
        try:
            size = tuple(int(x) for x in a.size.lower().split("x"))
            assert len(size) == 2 and all(x % 32 == 0 and 256 <= x <= 2048 for x in size)
        except (ValueError, AssertionError):
            sys.exit("--size: WxH, each a multiple of 32 between 256 and 2048")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is missing")
    total = float(probe(a.song, "-show_entries", "format=duration"))
    count = math.ceil(total / a.seconds - 1e-6)
    plan = read_sections(a.sections, count, a.prompt)
    out = Path(a.out or Path(a.song).with_suffix(".mp4"))
    parts = out.with_name(out.name + ".parts")

    sections = plan_sections(plan, total, a.seconds, a.pictures, a.continuous)
    print(f"{Path(a.song).name}: {total:.2f} s, {count} sections of {a.seconds} s"
          f"{' (continuous)' if a.continuous else ''} -> {out}")
    for s in sections:
        pic = Path(s["picture"]).name if s["picture"] else "last frame before" if s["follow"] else "text only"
        print(f"  {s['i'] + 1:2d}  {s['start']:6.2f}-{s['end']:6.2f} s  {s['kind']:5s}  {pic}  {s['prompt'][:60]}")
    print(f"  about {count * 6.5 * a.seconds / 5:.0f} min at 1280x704")
    if a.dry_run:
        return

    if not a.url:
        sys.exit("musicvideo renders on ComfyUI, which is not running — llmctl start comfy 9")
    comfy = Comfy(a.url)
    try:
        requests.get(f"{comfy.url}/system_stats", timeout=10).raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"no ComfyUI at {comfy.url} ({e})")
    parts.mkdir(parents=True, exist_ok=True)
    song_hash = hashlib.sha256(Path(a.song).read_bytes()).hexdigest()[:16]
    trimmed, previous, t_all = [], None, time.time()
    for s in sections:
        picture = s["picture"]
        if s["follow"]:                                            # --continuous: the frame before
            picture = parts / f"start_{s['i']:02d}.png"
            ffmpeg("-i", str(previous), "-vf", f"select=eq(n\\,{sections[s['i'] - 1]['frames'] - 1})",
                   "-frames:v", "1", str(picture))
        pic_hash = hashlib.sha256(Path(picture).read_bytes()).hexdigest()[:16] if picture else None
        key = json.dumps([song_hash, pic_hash, s["kind"], s["prompt"], s["start"], s["end"], a.size], sort_keys=True)
        clip, kept, meta = parts / f"clip_{s['i']:02d}.mp4", parts / f"part_{s['i']:02d}.mp4", parts / f"part_{s['i']:02d}.json"
        if kept.exists() and meta.exists() and meta.read_text() == key:
            print(f"  {s['i'] + 1:2d}/{count}  kept from before")
        else:
            t = time.time()
            audio = None
            if s["kind"] == "sing":
                seg = parts / f"song_{s['i']:02d}.flac"
                ffmpeg("-i", a.song, "-ss", f"{s['start']:.6f}", "-t", f"{s['end'] - s['start']:.6f}", str(seg))
                audio = comfy.upload(seg)
            g = graph(a.workflows, s["kind"], comfy.upload(picture) if picture else None, s["prompt"], audio,
                      math.ceil(s["end"] - s["start"] - 1e-6), size, f"video/musicvideo/{out.stem}_{s['i']:02d}",
                      placeholder=comfy.upload(a.pictures[0]))
            comfy.run(g, clip)
            have = int(probe(clip, "-select_streams", "v", "-count_frames", "-show_entries", "stream=nb_read_frames"))
            if have < s["frames"]:
                sys.exit(f"section {s['i'] + 1}: the clip has {have} frames, {s['frames']} needed")
            ffmpeg("-i", str(clip), "-an", "-frames:v", str(s["frames"]), "-c:v", "libx264", "-crf", "16",
                   "-pix_fmt", "yuv420p", "-r", str(FPS), str(kept))
            meta.write_text(key)
            done = s["i"] + 1
            print(f"  {done:2d}/{count}  {s['kind']} in {time.time() - t:.0f} s"
                  f"  (all so far {(time.time() - t_all) / 60:.0f} min)", flush=True)
        previous = kept
        trimmed.append(kept)

    listing = parts / "list.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in trimmed))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-i", a.song, "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{total:.6f}", str(out))
    frames = int(probe(out, "-select_streams", "v", "-count_frames", "-show_entries", "stream=nb_read_frames"))
    print(f"{out}: {frames} frames = {frames / FPS:.3f} s, song {total:.3f} s, "
          f"{(time.time() - t_all) / 60:.0f} min")


if __name__ == "__main__":
    main()
