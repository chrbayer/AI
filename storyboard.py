#!/usr/bin/env python3
"""A scene from a storyboard, on a running ComfyUI — `llmctl storyboard` (#48).

A plan file names a photo, the keyframes to make from it and the four clips
between them. Three steps, each run again only where its input changed:

1. Keyframes, each straight from the photo with Qwen-Image Edit Turbo (an edit
   of an edit gathers noise). With "object", that object is cut out of
   keyframe 1 (SAM 3) and given to every later edit as a second picture, so it
   looks the same in all of them.
2. A contact sheet of the keyframes (sheet.jpg), the object's size measured in
   each (SAM 3; it must stay the same within a shot) and, with --vision, a check by a
   vision model: per picture where the person and the object are, against what
   the plan expects there, and over all pictures whether they make one scene.
   Look before rendering: an hour of LTX is worth it only for keyframes that fit.
3. With --render: the four clips, LTX-2.5 Chain Keyframes — each from one
   keyframe to the next, or, where a clip has "cut", from a picture of its own —
   and the motion measured: stalls and jumps. --check-video looks at a rendered
   video again, with --vision for ghosts too (two people, a half-transparent
   one: LTX dissolving someone to a place it does not walk her to).

    storyboard.py --url URL --workflows DIR PLAN.json [--vision URL] [--render | --check-video] [--out FILE]

The plan (JSON):

    {"photo": "beach.png",                     relative to the plan file
     "object": "the pale spiral seashell",     optional: SAM 3's prompt for it
     "size": "1280x704", "seed": 1,            optional
     "pull": 0.5,                              optional: Stärke Zielbild, all clips
     "keys": [{"edit": "...", "check": "..."}, ... five],
     "clips": [{"prompt": "...", "seconds": 4, "pull": 0.7}, ...,   four; a clip with
               {"prompt": "...", "cut": {"edit": "...", "check": "..."}}]}

A clip's "seconds" (1-10, default 5; 1.5 is fine, rounded to LTX's grid of
1/3 s) is what its action needs: a clip done
early waits at its end picture (a stall), one too short rushes. Its "pull"
overrides the plan's.

"edit" is the Qwen Edit prompt: <image1> is the photo, <image2> the object; a
picture's own "seed" tries another take of it. "from": "key_3" edits it from
that keyframe instead of the photo (<image1> is then that keyframe): the person
stays where she was — LTX does not walk her to a place she does not face, it
dissolves her there. One edit of an edit is fine; a row of them gathers grain.
"object_stays": true pastes the
object in where it lies in key 1 — until it is picked up it must not move (Qwen
draws it somewhere else each time, and LTX then kicks or throws it there); its
edit then leaves the object out and places the person by it.
"check" says where things should be, in the picture's own terms ("the shell
lies front left, in the direction she walks"). What one keyframe has to reach
from the one before must be a natural movement in a clip's 5 s — the same shot
size or a plausible camera move; a jump from close to wide is a "cut".

Everything lands in PLAN.storyboard/ beside the plan: key_N.png (and
cut_N.png), object.png, sheet.jpg, check.md, the video. Delete a picture there
to have it made anew.
"""
import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from musicvideo import Comfy  # noqa: E402

EDIT = "Qwen-Image 2.1 Edit Turbo (bf16, 4 Schritte).json"
SAM = "SAM 3 Select.json"
CHAIN = "video/LTX-2.5 Chain Keyframes (int8, distilled, 4 clips).json"
CLIPS = 4
UPLOAD_DIR = "storyboard"
FPS = 24


def load_plan(path):
    if not Path(path).is_file():
        sys.exit(f"no such file: {path}")
    plan = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    plan["photo"] = str(base / plan["photo"])
    if not Path(plan["photo"]).is_file():
        sys.exit(f"no such photo: {plan['photo']}")
    if len(plan.get("keys", [])) != CLIPS + 1 or len(plan.get("clips", [])) != CLIPS:
        sys.exit(f"the plan wants {CLIPS + 1} keys and {CLIPS} clips")
    for i, clip in enumerate(plan["clips"], 1):
        seconds = float(clip.get("seconds", 5))
        if not 1 <= seconds <= 10:
            sys.exit(f"clip {i}: seconds 1 to 10")
        if clip_frames(seconds) != round(seconds * FPS) + 1:
            print(f"clip {i}: {seconds:g} s → {(clip_frames(seconds) - 1) / FPS:.2f} s "
                  f"({clip_frames(seconds)} frames; LTX takes 8·n + 1)")
    if plan["clips"][0].get("cut"):
        sys.exit("clip 1 starts on key 1; a cut makes sense from clip 2 on")
    for i, k in enumerate(plan["keys"] + [c["cut"] for c in plan["clips"] if c.get("cut")], 1):
        if not k.get("edit"):
            sys.exit(f"picture {i}: no 'edit' prompt")
    if plan.get("size"):
        try:
            w, h = (int(x) for x in plan["size"].lower().split("x"))
            assert w % 32 == 0 and h % 32 == 0
            plan["size"] = (w, h)
        except (ValueError, AssertionError):
            sys.exit("size: WxH, multiples of 32")
    return plan


def picture_part(p, kind="png"):
    """A picture as a chat message part."""
    data = p if isinstance(p, bytes) else Path(p).read_bytes()
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64," + base64.b64encode(data).decode()}}


def clip_frames(seconds):
    """The frames LTX makes for a clip of `seconds`: 8·n + 1, the nearest (a tie
    upwards) — as the chain workflows' Dauer Clip N rounds it."""
    return math.floor(seconds * FPS / 8 + 0.5) * 8 + 1


def pictures(plan):
    """[(name, edit, check)] in the order they are shown: key 1, ..., with each
    cut's start picture before the key its clip runs to."""
    out = [("key_1", plan["keys"][0])]
    for i, clip in enumerate(plan["clips"], 1):
        if clip.get("cut"):
            out.append((f"cut_{i}", clip["cut"]))
        out.append((f"key_{i + 1}", plan["keys"][i]))
    return out


def digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(Path(p).read_bytes() if isinstance(p, Path) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:16]


class Board:
    def __init__(self, a, plan):
        self.a, self.plan = a, plan
        self.dir = Path(a.plan).with_suffix(".storyboard")
        self.dir.mkdir(exist_ok=True)
        self.comfy = Comfy(a.url) if a.url else None
        self.workflows = Path(a.workflows)

    def need_comfy(self):
        if not self.comfy:
            sys.exit("ComfyUI is not running — start it: llmctl start comfy")
        return self.comfy

    def fresh(self, out, key):
        """Whether out was made from these inputs (its .json beside it says so)."""
        meta = out.with_suffix(".json")
        return out.is_file() and meta.is_file() and json.loads(meta.read_text()).get("from") == key

    def made(self, out, key):
        out.with_suffix(".json").write_text(json.dumps({"from": key}) + "\n")

    def edit(self, name, prompt, seed, ref=None, source=None):
        out = self.dir / f"{name}.png"
        source = Path(source or self.plan["photo"])
        key = digest(source, prompt, seed, *([ref] if ref else []))
        if self.fresh(out, key):
            print(f"  {name}: unchanged")
            return out
        comfy = self.need_comfy()
        g = json.loads((self.workflows / EDIT).read_text())
        loads = sorted((k for k, n in g.items() if n["class_type"] == "LoadImage"), key=int)
        g[loads[0]]["inputs"]["image"] = comfy.upload(source, UPLOAD_DIR)
        if ref:
            g[loads[1]]["inputs"]["image"] = comfy.upload(ref, UPLOAD_DIR)
        else:                                        # one picture: the second loader goes
            for n in g.values():
                for a_, x in list(n["inputs"].items()):
                    if isinstance(x, list) and x[0] == loads[1]:
                        del n["inputs"][a_]
            del g[loads[1]]
        g = {k: n for k, n in g.items() if n["class_type"] != "ImageCompare"}
        for n in g.values():
            if n["class_type"].startswith("TextEncodeQwenImage"):
                n["inputs"]["prompt"] = prompt
            if n["class_type"] == "KSampler":
                n["inputs"]["seed"] = seed
            if n["class_type"] in ("SaveImage", "SaveImageAdvanced"):
                n["inputs"]["filename_prefix"] = f"storyboard/{self.dir.stem}_{name}"
        comfy.run(g, out)
        self.made(out, key)
        print(f"  {name}: made")
        return out

    def object_box(self, picture, name):
        """Where SAM 3 finds the object in a picture: (x0, y0, x1, y1) in its pixels, or None."""
        from PIL import Image
        mask = self.dir / f"{name}_mask.png"
        key = digest(Path(picture), self.plan["object"])
        if not self.fresh(mask, key):
            comfy = self.need_comfy()
            g = json.loads((self.workflows / SAM).read_text())
            for n in g.values():
                if n["class_type"] == "LoadImage":
                    n["inputs"]["image"] = comfy.upload(picture, UPLOAD_DIR)
                if n["class_type"] == "SAM3Segment":
                    n["inputs"].update(prompt=self.plan["object"], max_segments=1)
            comfy.run(g, mask)
            self.made(mask, key)
        m = Image.open(mask).convert("L")
        box = m.point([0] * 128 + [255] * 128).getbbox()
        if not box:
            return None
        w, h = Image.open(picture).size
        sx, sy = w / m.width, h / m.height
        return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy

    def cut_object(self, key1):
        """The object from keyframe 1: SAM 3's mask, its box grown a little, cut out."""
        from PIL import Image
        out = self.dir / "object.png"
        key = digest(key1, self.plan["object"])
        if self.fresh(out, key):
            print("  object: unchanged")
            return out
        found = self.object_box(key1, "key_1")
        if not found:
            sys.exit(f"SAM 3 found no '{self.plan['object']}' in key_1.png — reword 'object' or the first edit")
        x0, y0, x1, y1 = found
        picture = Image.open(key1).convert("RGB")
        grow = 0.15 * max(x1 - x0, y1 - y0)
        box = (max(0, int(x0 - grow)), max(0, int(y0 - grow)),
               min(picture.width, int(x1 + grow)), min(picture.height, int(y1 + grow)))
        picture.crop(box).save(out)
        self.made(out, key)
        print(f"  object: cut out of key_1 at {box}")
        return out

    def paste_object(self, key1, picture, name):
        """The object as it lies in keyframe 1, pasted into a later picture at the
        same place (SAM 3's mask, its edge softened): until it is picked up it must
        not move, and Qwen draws it somewhere else in every edit — LTX then kicks,
        throws or blends it from one place to the other."""
        from PIL import Image, ImageFilter
        out = self.dir / f"{name}_pasted.png"
        key = digest(Path(key1), Path(picture), self.plan["object"])
        if self.fresh(out, key):
            return out
        mask = self.dir / "key_1_mask.png"
        self.object_box(key1, "key_1")                      # makes the mask if need be
        src = Image.open(key1).convert("RGB")
        dst = Image.open(picture).convert("RGB")
        if dst.size != src.size:
            sys.exit(f"{name}: {dst.size} is not the size of key_1 {src.size}; the object cannot be pasted")
        m = Image.open(mask).convert("L").resize(src.size).point([0] * 128 + [255] * 128)
        m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
        dst.paste(src, (0, 0), m)
        dst.save(out)
        self.made(out, key)
        print(f"  {name}: object pasted where it lies in key_1")
        return out

    def sizes(self, shown, made):
        """The object's size, measured: within a shot (from one cut to the next the
        camera stays where it is) it must stay about the same — a hand over it
        hides some, so within 0.6 to 1.6 times the first picture of the shot it
        was found in. Generated pictures drift here, and a vision model is no
        judge of it."""
        lines, ref, bad = [], None, 0
        for name, _ in shown:
            if name.startswith("cut_"):
                ref = None                                   # a new shot
            box = self.object_box(made[name], name)
            if not box:
                lines.append(f"{name}: not found")
                continue
            size = max(box[2] - box[0], box[3] - box[1])
            if ref is None:
                ref = (name, size)
                lines.append(f"{name}: {size:.0f} px")
                continue
            ratio = size / ref[1]
            ok = 0.6 <= ratio <= 1.6
            bad += not ok
            lines.append(f"{name}: {size:.0f} px, {ratio:.2f}× {ref[0]}" + ("" if ok else " — MISMATCH: size"))
        for line in lines:
            print(f"  size {line}")
        return lines, bad

    def keyframes(self):
        seed = int(self.plan.get("seed", 1))
        shown = pictures(self.plan)
        made = {}
        made["key_1"] = self.edit("key_1", shown[0][1]["edit"], seed)
        ref = self.cut_object(made["key_1"]) if self.plan.get("object") else None
        for i, (name, k) in enumerate(shown[1:], 2):
            if k.get("from") and k["from"] not in made:
                sys.exit(f"{name}: 'from' names {k['from']}, which comes later or does not exist")
            made[name] = self.edit(name, k["edit"], int(k.get("seed", seed + i)), ref, made.get(k.get("from")))
            if k.get("object_stays"):
                made[name] = self.paste_object(made["key_1"], made[name], name)
        self.sheet(shown, made)
        self.measured = self.sizes(shown, made) if self.plan.get("object") else ([], 0)
        return shown, made

    def sheet(self, shown, made):
        from PIL import Image, ImageDraw
        thumbs = [Image.open(made[n]).convert("RGB") for n, _ in shown]
        h = 360
        thumbs = [t.resize((round(t.width * h / t.height), h)) for t in thumbs]
        cols = 3
        rows = -(-len(thumbs) // cols)
        w = max(t.width for t in thumbs)
        sheet = Image.new("RGB", (cols * w + (cols + 1) * 12, rows * (h + 36) + 12), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (t, (name, _)) in enumerate(zip(thumbs, shown)):
            x, y = 12 + (i % cols) * (w + 12), 12 + (i // cols) * (h + 36)
            sheet.paste(t, (x, y + 24))
            label = name.replace("key_", "Bild ").replace("cut_", "Schnitt -> Clip ")
            draw.text((x, y + 4), label, fill=(230, 230, 230))
        sheet.save(self.dir / "sheet.jpg", quality=88)
        print(f"  contact sheet: {self.dir / 'sheet.jpg'}")

    def ask(self, content):
        """One question to the vision model (--vision), its answer."""
        url = self.a.vision.rstrip("/")
        url = url if url.endswith("/v1") else url + "/v1"
        try:
            r = requests.post(f"{url}/chat/completions", timeout=900, json={
                "messages": [{"role": "user", "content": content}], "max_tokens": 900, "temperature": 0.2})
        except requests.ConnectionError:
            sys.exit(f"no vision model at {url}")
        if not r.ok:
            sys.exit(f"the vision model at {url} refused: {r.text[:500]}\n"
                     "(a llama.cpp model sees pictures only when started with --mmproj)")
        return r.json()["choices"][0]["message"]["content"].strip()

    def check(self, shown, made):
        """A vision model's look at each picture against its 'check', then at all."""
        thing = self.plan.get("object") or "anything the scene is about"
        ask = self.ask

        image = picture_part

        report = [f"# Storyboard check: {Path(self.a.plan).name}\n"]
        for name, k in shown:
            q = (f"This is one keyframe of a short film. Describe in 3-4 short sentences, in the picture's own terms "
                 f"(left/right/front/back of the picture, not of the person): where the person is, where {thing} is, "
                 f"which way the person faces or moves, and the shot size (wide, medium, close). Say how many people "
                 f"are in the picture, and whether anyone or anything appears twice, half transparent, or "
                 f"cut off.")
            if k.get("check"):
                q += (f" The plan says: \"{k['check']}\" — and one person only, nothing doubled. End with one "
                      f"line, 'MATCH' or 'MISMATCH: <what differs>'.")
            answer = ask([{"type": "text", "text": q}, image(made[name])])
            report.append(f"## {name}\n\n{answer}\n")
            verdict = answer[answer.rfind("MISMATCH"):] if "MISMATCH" in answer else "MATCH" if "MATCH" in answer else "?"
            print(f"  {name}: {verdict.splitlines()[0][:160]}")
        q = ("These pictures are the keyframes of one continuous short film, in order. Is it believably the same "
             f"person, with the same clothes, and {thing} the same in size and look in every picture? Do the places "
             "and directions agree from one picture to the next (an object stays where it lay until it is picked "
             "up, the person keeps walking the way she went, the camera stays on the same side)? List each "
             "inconsistency with the pictures' numbers, or say that there is none.")
        answer = ask([{"type": "text", "text": q}, *[image(made[n]) for n, _ in shown]])
        report.append(f"## All pictures\n\n{answer}\n")
        if self.measured[0]:
            report.append("## Size of the object (measured)\n\n" + "\n".join(f"- {x}" for x in self.measured[0]) + "\n")
        (self.dir / "check.md").write_text("\n".join(report))
        print(f"  check: {self.dir / 'check.md'}")

    def render(self, made):
        comfy = self.need_comfy()
        g = json.loads((self.workflows / CHAIN).read_text())
        title = lambda n: n.get("_meta", {}).get("title", "")
        for n in g.values():
            t = title(n)
            if n["class_type"] == "LoadImage" and t.startswith("Bild "):
                n["inputs"]["image"] = comfy.upload(made[f"key_{t.split()[1]}"], UPLOAD_DIR)
            if n["class_type"] == "LoadImage" and t.startswith("Start Clip "):
                i = int(t.split()[2])
                cut = made.get(f"cut_{i}", made["key_1"])
                n["inputs"]["image"] = comfy.upload(cut, UPLOAD_DIR)
            if n["class_type"] == "PrimitiveFloat" and t.startswith("Stärke Zielbild "):
                clip = self.plan["clips"][int(t.split()[2]) - 1]
                if "pull" in clip or "pull" in self.plan:
                    n["inputs"]["value"] = float(clip.get("pull", self.plan.get("pull")))
            if n["class_type"] in ("PrimitiveFloat", "PrimitiveInt") and t.startswith("Dauer Clip "):
                n["inputs"]["value"] = float(self.plan["clips"][int(t.split()[2]) - 1].get("seconds", 5))
            if n["class_type"] == "PrimitiveBoolean" and t.startswith("Schnitt vor Clip "):
                n["inputs"]["value"] = f"cut_{t.split()[3]}" in made
            if n["class_type"] == "PrimitiveStringMultiline" and t.startswith("Prompt "):
                n["inputs"]["value"] = self.plan["clips"][int(t.split()[1]) - 1]["prompt"]
            if self.plan.get("size") and n["class_type"] == "PrimitiveInt" and t.lower() in ("width", "height"):
                n["inputs"]["value"] = self.plan["size"][t.lower() == "height"]
            if n["class_type"] == "SaveVideo":
                n["inputs"]["filename_prefix"] = f"video/storyboard_{self.dir.stem}"
        out = self.video()
        total = sum((clip_frames(float(c.get("seconds", 5))) - 1) / FPS for c in self.plan["clips"])
        print(f"  rendering 4 clips, {total:.1f} s — about {total * 3:.0f} min at 1280x704 …", flush=True)
        comfy.run(g, out)
        print(f"  video: {out}")
        self.motion(out)

    def video(self):
        return Path(self.a.out) if self.a.out else self.dir / f"{Path(self.a.plan).stem}.mp4"

    def ghosts(self, video):
        """Every half second a frame to the vision model: two people where there is
        one, or anyone half transparent — LTX's way of moving someone it does not
        walk (a dissolve), which the motion measurement does not see."""
        import subprocess
        frames = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='not(mod(n\\,{FPS // 2}))',scale=640:-2",
                                 "-vsync", "0", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                                capture_output=True, check=True).stdout
        pictures = [b"\xff\xd8" + p for p in frames.split(b"\xff\xd8")[1:]]
        found = []
        for i, jpg in enumerate(pictures):
            answer = self.ask([{"type": "text", "text":
                                "A frame of a video. How many people are in it, and is any of them half transparent, "
                                "a faint double, or a ghost of another? Answer in one line: 'PEOPLE <n>, GHOST yes' or "
                                "'PEOPLE <n>, GHOST no'."}, picture_part(jpg, "jpeg")])
            t = i * (FPS // 2) / FPS
            if "GHOST YES" in answer.upper().replace(":", "") or "PEOPLE 1" not in answer.upper().replace(":", ""):
                found.append(f"{t:.1f} s: {answer.splitlines()[0][:100]}")
        for line in found or ["none"]:
            print(f"  ghost or double {line}")
        with open(self.dir / "check.md", "a") as f:
            f.write(f"\n## Ghosts in {Path(video).name} (vision model, every 0.5 s)\n\n" +
                    "\n".join(f"- {x}" for x in found or ["none"]) + "\n")

    def motion(self, video):
        """Where the finished video stands still or jumps, measured: the mean change
        from frame to frame (grey, 160 px wide). A stall — half a second and more
        nearly still at a clip's end — is a clip done early, waiting for its end
        picture (shorter "seconds", lower "pull", more to do in its prompt). A jump
        at a join is a clip that did not get there (longer, higher pull)."""
        import shutil
        import subprocess
        import numpy as np
        if not shutil.which("ffmpeg"):
            return
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=160:88,format=gray",
                              "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, np.uint8).reshape(-1, 88, 160).astype(float)
        d = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2))         # d[i]: frame i -> i+1
        report, start = [], 0
        for i, clip in enumerate(self.plan["clips"], 1):
            frames = clip_frames(float(clip.get("seconds", 5))) - (0 if i == 1 or clip.get("cut") else 1)
            end = start + frames                                    # this clip's frames: [start, end)
            seg = d[start:end - 1]
            still = 0
            while still < len(seg) and seg[len(seg) - 1 - still] < 0.35:
                still += 1
            line = f"clip {i}: {frames} frames, motion {seg.mean():.2f}"
            if still >= FPS // 2:
                line += f" — STALL: the last {still / FPS:.1f} s nearly still"
            if end - 1 < len(d) and i < len(self.plan["clips"]) and not self.plan["clips"][i].get("cut"):
                jump, usual = d[end - 1], np.median(seg) + 1e-6
                if jump > 3 * usual and jump > 1.0:
                    line += f" — JUMP into clip {i + 1} ({jump / usual:.0f}× the usual change)"
            report.append(line)
            start = end
        for line in report:
            print(f"  {line}")
        with open(self.dir / "check.md", "a") as f:
            f.write(f"\n## Motion of {Path(video).name} (measured)\n\n" + "\n".join(f"- {x}" for x in report) + "\n")


def main():
    ap = argparse.ArgumentParser(description="a scene from a storyboard: keyframes, a check, four LTX clips")
    ap.add_argument("plan")
    # llmctl passes these: the ComfyUI slot and the API workflows.
    ap.add_argument("--url", default="", help=argparse.SUPPRESS)
    ap.add_argument("--workflows", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--vision", help="an OpenAI base URL of a model that sees pictures, for the check")
    ap.add_argument("--render", action="store_true", help="render the clips (else: keyframes, sheet, check only)")
    ap.add_argument("--check-video", action="store_true",
                    help="check the rendered video: motion (stalls, jumps), with --vision ghosts too")
    ap.add_argument("--out", help="the video (default: PLAN.storyboard/PLAN.mp4)")
    a = ap.parse_args()
    plan = load_plan(a.plan)
    board = Board(a, plan)
    if a.check_video:
        if not board.video().is_file():
            sys.exit(f"no video yet: {board.video()} — render it with --render")
        print("Video check:")
        board.motion(board.video())
        if a.vision:
            board.ghosts(board.video())
        return
    print("Keyframes:")
    shown, made = board.keyframes()
    if a.vision:
        print("Check:")
        board.check(shown, made)
    if a.render:
        print("Video:")
        board.render(made)
    elif not a.vision:
        print(f"Look at {board.dir / 'sheet.jpg'}; then run again with --render (and --vision URL for a check).")


if __name__ == "__main__":
    main()
