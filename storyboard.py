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
   --redo-clip N renders one clip again (--seed: another take) and splices it
   in at its frames, the rest of the video as it was; the seed goes into the
   plan, and --render makes every clip with a "seed" or "guides" that way after
   the chain, so the plan says the whole video.

    storyboard.py --url URL --workflows DIR PLAN.json [--vision URL]
                  [--render | --redo-clip N [--seed S] | --check-video] [--out FILE]

The plan (JSON):

    {"photo": "beach.png",                     relative to the plan file
     "object": "the pale spiral seashell",     optional: SAM 3's prompt for it
     "size": "1280x704", "seed": 1,            optional
     "pull": 0.5,                              optional: Stärke Zielbild, all clips
     "ambience": "steady surf and wind ...",   optional: one soundtrack for it all
     "soundtrack": "song.mp3",                 optional: a song under it instead
     "soundtrack_start": 12.5,                 (where in the song it begins)
     "smooth_joins": true,                     the default: see smooth_joins
     "keys": [{"edit": "...", "check": "..."}, ... five],
     "clips": [{"prompt": "...", "seconds": 4, "pull": 0.7}, ...,   four; a clip with
               {"prompt": "...", "cut": {"edit": "...", "check": "..."}}]}

A clip's "seconds" (1-10, default 5; 1.5 is fine, rounded to LTX's grid of
1/3 s) is what its action needs: a clip done
early waits at its end picture (a stall), one too short rushes. Its "pull"
overrides the plan's. "guides": [{"at": 2.5, "picture": "cut_4", "pull": 0.6}]
on a clip gives --redo-clip a picture the clip must pass at that time — a keyframe
in the middle, so that LTX does not hurry past what comes first. Not a copy of the
picture before: between two equal pictures LTX holds the frame still (2.5 s of a
still image in the close-up). A guide's own "edit" (with "from") makes a variant
of it — the same pose, the waves and hair moved on.

"edit" is the Qwen Edit prompt: <image1> is the photo, <image2> the object; a
picture's own "seed" tries another take of it. "from": "key_3" edits it from
that keyframe instead of the photo (<image1> is then that keyframe): the person
stays where she was — LTX does not walk her to a place she does not face, it
dissolves her there. One edit of an edit is fine; a row of them gathers grain.
"object_at": [x, y] on key 1
(with "object_height", and the plan's "object_image", a picture of the object
with transparency) pastes the object in exactly there instead of having Qwen
draw it, roughly where it likes; key 1 may then go without an "edit".
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
AMBIENCE = "audio/Stable Audio 3 Medium (8 Schritte).json"
CHAIN = "video/LTX-2.5 Chain Keyframes (int8, distilled, 4 clips).json"
SINGLE = "video/LTX-2.5 First-Last Frame (int8, distilled).json"
CLIPS = 4
UPLOAD_DIR = "storyboard"
DEFAULT_SEED = 4242
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
        if not k.get("edit") and not (i == 1 and k.get("object_at")):
            sys.exit(f"picture {i}: no 'edit' prompt")
    if plan["keys"][0].get("object_at"):
        if not plan.get("object_image") or not plan.get("object"):
            sys.exit("object_at wants an 'object_image' (with transparency) and the 'object' it shows")
        plan["object_image"] = str(base / plan["object_image"])
        if not Path(plan["object_image"]).is_file():
            sys.exit(f"no such object image: {plan['object_image']}")
    if plan.get("soundtrack"):
        plan["soundtrack"] = str(base / plan["soundtrack"])
        if not Path(plan["soundtrack"]).is_file():
            sys.exit(f"no such soundtrack: {plan['soundtrack']}")
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


def smooth_joins(src, dst, joins, tone=24, blend=3):
    """Where one clip goes on from the last frame of the one before, the new clip
    starts a little darker and softer (its first frame is that last frame through
    the VAE once more): 3 levels of brightness and a fifth of the sharpness in the
    beach scene, a jump there. Here the new clip takes the tone (mean and spread
    of each colour) of the frame before it, fading back to its own over `tone`
    frames, and its first `blend` frames are eased in from that frame. At a join
    nothing moves much (keyframes are resting states), so that leaves no ghost."""
    import subprocess
    import numpy as np
    w, h = (int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                            "stream=width,height", "-of", "csv=p=0", str(src)],
                                           capture_output=True, text=True, check=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()
    for j in joins:
        if not 0 < j < len(x):
            continue
        a = x[j - 1].astype(np.float32)
        ma, sa = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)
        b = x[j].astype(np.float32)
        mb, sb = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
        for k in range(min(tone, len(x) - j)):
            f = x[j + k].astype(np.float32)
            weight = 1 - k / tone
            f = weight * ((f - mb) * (sa / (sb + 1e-6)) + ma) + (1 - weight) * f
            if k < blend:
                v = (k + 1) / (blend + 1)
                f = (1 - v) * a + v * f
            x[j + k] = np.clip(f, 0, 255).astype(np.uint8)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                            "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    enc.stdin.write(x.tobytes())
    enc.stdin.close()
    if enc.wait():
        sys.exit("smoothing the joins failed")


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

    def place_object(self, picture, key):
        """Keyframe 1 with the object pasted in where the plan says ("object_at":
        its centre as fractions of width and height, "object_height": its height
        as a fraction of the picture's) from "object_image", a picture of it with
        transparency. Qwen puts an object roughly where a prompt asks, and the
        whole scene hangs on where it lies: whether her way to it crosses it,
        where she stops. The mask is the picture's own transparency."""
        from PIL import Image
        out, mask = self.dir / "key_1_placed.png", self.dir / "key_1_placed_mask.png"
        thing = Path(self.plan["object_image"])
        where = (key["object_at"], key.get("object_height", 0.06))
        k = digest(Path(picture), thing, where)
        if not self.fresh(out, k):
            scene = Image.open(picture).convert("RGB")
            obj = Image.open(thing).convert("RGBA")
            h = max(4, round(where[1] * scene.height))
            obj = obj.resize((max(4, round(obj.width * h / obj.height)), h), Image.LANCZOS)
            x = round(where[0][0] * scene.width - obj.width / 2)
            y = round(where[0][1] * scene.height - obj.height / 2)
            scene.paste(obj, (x, y), obj)
            scene.save(out)
            m = Image.new("L", scene.size, 0)
            m.paste(obj.getchannel("A"), (x, y))
            m.save(mask)
            self.made(out, k)
            print(f"  key_1: object placed at {where[0]}, {h} px high")
        # what SAM 3 would have found there: the mask is known, for the cut-out,
        # object_stays and the size check alike
        known = self.dir / "key_1_mask.png"
        known.write_bytes(mask.read_bytes())
        self.made(known, digest(out, self.plan["object"]))
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
        first = shown[0][1]
        made["key_1"] = self.edit("key_1", first["edit"], seed) if first.get("edit") else Path(self.plan["photo"])
        if first.get("object_at"):
            made["key_1"] = self.place_object(made["key_1"], first)
        ref = self.cut_object(made["key_1"]) if self.plan.get("object") else None
        for i, (name, k) in enumerate(shown[1:], 2):
            if k.get("from") and k["from"] not in made:
                sys.exit(f"{name}: 'from' names {k['from']}, which comes later or does not exist")
            made[name] = self.edit(name, k["edit"], int(k.get("seed", seed + i)), ref, made.get(k.get("from")))
            if k.get("object_stays"):
                made[name] = self.paste_object(made["key_1"], made[name], name)
        # a guide picture of its own (a variant of a keyframe: the same pose, the
        # waves and hair moved on — a copy of a keyframe would hold the clip still)
        for c, clip in enumerate(self.plan["clips"], 1):
            for j, guide in enumerate(clip.get("guides", []), 1):
                if guide.get("edit"):
                    name = f"guide_{c}_{j}"
                    made[name] = self.edit(name, guide["edit"], int(guide.get("seed", seed + 100 * c + j)), ref,
                                           made.get(guide.get("from")))
                    guide["picture"] = name
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
        self.write_layout(out, [clip_frames(float(c.get("seconds", 5))) for c in self.plan["clips"]])
        # the clips made apart from the chain, in order: a clip with its own seed (a
        # take chosen with --redo-clip) or with guides (the chain has start and end
        # alone) — so the plan says the whole video, not the chain and some redos
        for i, clip in enumerate(self.plan["clips"], 1):
            if "seed" in clip or clip.get("guides"):
                self.redo_clip(made, i, int(clip.get("seed", DEFAULT_SEED)), finish=False)
        self.motion(out)
        self.finish(out)

    def write_layout(self, video, frames):
        """What each clip was rendered with, beside the video: --redo-clip cuts the
        old video at these frames, whatever the plan says now."""
        Path(str(video) + ".layout.json").write_text(json.dumps({"frames": frames}) + "\n")

    def layout(self, video):
        f = Path(str(video) + ".layout.json")
        if f.is_file():
            return json.loads(f.read_text())["frames"]
        return [clip_frames(float(c.get("seconds", 5))) for c in self.plan["clips"]]

    def ambience(self, video):
        """One soundtrack for the whole video instead of LTX's, which it makes clip by
        clip: each clip a soundscape of its own, so the sound jumps at every join
        (8 dB in the beach scene, where one clip had the surf and the next the
        wind). Stable Audio 3 (SFX) makes "ambience" as long as the video, three
        takes, the most even one (the least spread of loudness from second to
        second) goes under the picture, which is copied as it is."""
        import subprocess
        import numpy as np
        comfy = self.need_comfy()
        length = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                       "stream=duration", "-of", "csv=p=0", str(video)],
                                      capture_output=True, text=True, check=True).stdout)
        takes = []
        for seed in (1, 2, 3):
            take = self.dir / f"ambience_{seed}.flac"
            key = digest(self.plan["ambience"], seed, round(length, 2))
            if not self.fresh(take, key):
                g = json.loads((self.workflows / AMBIENCE).read_text())
                for n in g.values():
                    t = n.get("_meta", {}).get("title", "")
                    if t.startswith("User: short description"):
                        n["inputs"]["value"] = self.plan["ambience"]
                    if n["class_type"] == "CustomCombo":
                        n["inputs"].update(choice="SFX", index=2)
                    if t == "Float (Duration)":
                        n["inputs"]["value"] = math.ceil(length + 1)
                    if n["class_type"] == "KSampler":
                        n["inputs"]["seed"] = seed
                    if n["class_type"] == "SaveAudioAdvanced":
                        n["inputs"].update(filename_prefix=f"audio/storyboard_{self.dir.stem}", format="flac")
                comfy.run(g, take)
                self.made(take, key)
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(take), "-t", str(length), "-ac", "1", "-ar", "48000",
                                  "-f", "s16le", "-"], capture_output=True, check=True).stdout
            x = np.frombuffer(raw, np.int16).astype(float)
            level = [20 * np.log10(np.sqrt(np.mean(x[i:i + 48000] ** 2)) / 32768 + 1e-9)
                     for i in range(0, len(x) - 48000, 48000)]
            takes.append((max(level) - min(level), take))
        spread, best = min(takes)
        print(f"  ambience: {best.name}, loudness within {spread:.0f} dB")
        return best

    def finish(self, video):
        """The video to watch, <plan>_final.mp4, from the rendered one (which stays
        as it is, for --redo-clip): its joins smoothed (smooth_joins) and its
        sound — the "soundtrack" (a song: a scene that runs through it), else the
        "ambience", else LTX's own."""
        import subprocess
        length = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                       "stream=duration", "-of", "csv=p=0", str(video)],
                                      capture_output=True, text=True, check=True).stdout)
        out = video.with_name(video.stem + "_final.mp4")
        picture = video
        if self.plan.get("smooth_joins", True):
            joins = [j for j, cut in self.joins(video) if not cut]
            if joins:
                picture = video.with_name(video.stem + "_smoothed.mp4")
                smooth_joins(video, picture, joins)
                print(f"  joins smoothed at frames {', '.join(map(str, joins))}")
        if self.plan.get("soundtrack"):
            song, start = self.plan["soundtrack"], float(self.plan.get("soundtrack_start", 0))
            fade = min(2.0, length / 4)
            audio = ["-ss", str(start), "-i", song]
            shape = f"atrim=0:{length},afade=t=out:st={length - fade}:d={fade},loudnorm=I=-16:TP=-1.5,aresample=48000"
            what = f"soundtrack {Path(song).name} from {start:g} s"
        elif self.plan.get("ambience"):
            audio = ["-i", str(self.ambience(video))]
            shape = f"atrim=0:{length},afade=t=in:d=0.3,afade=t=out:st={length - 0.8}:d=0.8,loudnorm=I=-20:TP=-2,aresample=48000"
            what = "ambience"
        else:
            audio, shape, what = ["-i", str(video)], "anull", "LTX's sound"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(picture), *audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-af", shape, "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
        print(f"  final: {out} ({what})")

    def joins(self, video):
        """[(frame, cut)] where each later clip begins in the video."""
        out, at = [], 0
        for i, (made, clip) in enumerate(zip(self.layout(video), self.plan["clips"]), 1):
            if i > 1:
                out.append((at, bool(clip.get("cut"))))
            at += made - (0 if i == 1 or clip.get("cut") else 1)
        return out

    def redo_clip(self, made, i, seed, finish=True):
        """Clip i alone again, spliced into the finished video at its frames — for
        the one clip that failed in a video otherwise good. It starts on the
        video's own last frame before it (or, at a cut, on its picture) and runs
        to its keyframe, as in the chain; everything else is copied as it is."""
        import shutil
        import subprocess
        comfy = self.need_comfy()
        video = self.video()
        clips = self.plan["clips"]
        if not 1 <= i <= len(clips):
            sys.exit(f"--redo-clip: 1 to {len(clips)}")
        clip, frames = clips[i - 1], clip_frames(float(clips[i - 1].get("seconds", 5)))
        old = self.layout(video)                                  # as the video was made, not as the plan says now
        start = sum(f - (0 if j == 1 or c.get("cut") else 1)
                    for j, (f, c) in enumerate(zip(old[:i - 1], clips[:i - 1]), 1))   # this clip's first new frame
        follows = i > 1 and not clip.get("cut")                   # starts on the frame before it
        first = self.dir / f"redo_{i}_start.png"
        if follows:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select='eq(n\\,{start - 1})'",
                            "-frames:v", "1", str(first)], check=True)
        else:
            shutil.copy(made[f"cut_{i}"] if clip.get("cut") else made["key_1"], first)
        g = json.loads((self.workflows / SINGLE).read_text())
        pull = float(clip.get("pull", self.plan.get("pull", 0.5)))
        for n in g.values():
            t = n.get("_meta", {}).get("title", "")
            if n["class_type"] == "LoadImage":
                end = made[f"key_{i + 1}"]
                n["inputs"]["image"] = comfy.upload(first if t == "Load First Frame" else end, UPLOAD_DIR)
            if n["class_type"] == "LTXVAddGuide":
                n["inputs"]["strength"] = (1.0 if follows else n["inputs"]["strength"]) if n["inputs"]["frame_idx"] == 0 else pull
            if n["class_type"] == "ComfyMathExpression" and n["inputs"].get("expression", "").strip() == "a * b + 1":
                n["inputs"]["expression"] = str(frames)
            if n["class_type"] == "PrimitiveStringMultiline":
                n["inputs"]["value"] = clip["prompt"]
            if self.plan.get("size") and n["class_type"] == "PrimitiveInt" and t.lower() in ("width", "height"):
                n["inputs"]["value"] = self.plan["size"][t.lower() == "height"]
            if n["class_type"] == "RandomNoise":
                n["inputs"]["noise_seed"] = seed
            if n["class_type"] == "SaveVideo":
                n["inputs"]["filename_prefix"] = f"video/storyboard_{self.dir.stem}_clip{i}"
        for guide in clip.get("guides", []):
            self.add_guide(g, guide, made, frames)
        new = self.dir / f"redo_{i}.mp4"
        print(f"  clip {i} again ({frames} frames, seed {seed}) …", flush=True)
        comfy.run(g, new)
        # Lossless: a lossy pass would change the other clips' frames a little,
        # and a later redo starting on one of them would come out another clip
        # (the distilled sampler takes a start frame 0.15 off to a clip 10 off).
        keep = video.with_name(video.stem + ".before_redo.mp4")
        shutil.copy(video, keep)
        skip = 1 if follows else 0
        stop = start + old[i - 1] - skip                          # where the old clip ended
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(keep), "-i", str(new), "-filter_complex",
                        f"[0:v]trim=start_frame=0:end_frame={start},setpts=PTS-STARTPTS[a];"
                        f"[1:v]trim=start_frame={skip}:end_frame={frames},setpts=PTS-STARTPTS[b];"
                        f"[0:v]trim=start_frame={stop},setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]",
                        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-r", str(FPS),
                        "-c:a", "copy", str(video)], check=True)
        old[i - 1] = frames
        self.write_layout(video, old)
        print(f"  video: {video} (clip {i} now frames {start}–{start + frames - skip - 1}; the one before: {keep.name})")
        if finish:
            self.motion(video)
            self.finish(video)

    def record_seed(self, i, seed):
        """The take chosen, into the plan file: a --render then makes it again."""
        raw = json.loads(Path(self.a.plan).read_text())
        raw["clips"][i - 1]["seed"] = seed
        Path(self.a.plan).write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        self.plan["clips"][i - 1]["seed"] = seed
        print(f"  plan: clip {i} keeps seed {seed}")

    def add_guide(self, g, guide, made, frames):
        """One more picture a clip must pass through, at "at" seconds — a keyframe
        in the middle (only --redo-clip: the chain has start and end alone). What
        happens before it, LTX may not hurry past: a close-up that is to listen
        first, with eyes closed, and only then open them. Its picture goes the way
        the end picture does (resized, compressed), its guide after the end's."""
        end = next(k for k, n in g.items() if n["class_type"] == "LTXVAddGuide" and n["inputs"]["frame_idx"] == -1)
        pre = g[end]["inputs"]["image"][0]
        resize = g[pre]["inputs"]["image"][0]
        ids = iter(str(max(int(k) for k in g if k.isdigit()) + j) for j in range(1, 5))
        load, rs, pp, gd = next(ids), next(ids), next(ids), next(ids)
        name = guide["picture"]
        if name not in made:
            sys.exit(f"guide: no picture {name} (key_N or cut_N)")
        g[load] = {"class_type": "LoadImage", "inputs": {"image": self.need_comfy().upload(made[name], UPLOAD_DIR)},
                   "_meta": {"title": f"Guide {name}"}}
        g[rs] = {"class_type": g[resize]["class_type"], "inputs": {**g[resize]["inputs"], "input": [load, 0]}}
        g[pp] = {"class_type": g[pre]["class_type"], "inputs": {**g[pre]["inputs"], "image": [rs, 0]}}
        at = min(frames - 9, max(8, round(float(guide["at"]) * FPS / 8) * 8))   # on LTX's grid of 8, inside the clip
        for n in g.values():                                      # whoever read the end guide reads this one
            for a, x in n["inputs"].items():
                if isinstance(x, list) and x[0] == end:
                    n["inputs"][a] = [gd, x[1]]
        g[gd] = {"class_type": "LTXVAddGuide", "inputs": {**g[end]["inputs"], "positive": [end, 0], "negative": [end, 1],
                                                           "latent": [end, 2], "image": [pp, 0], "frame_idx": at,
                                                           "strength": float(guide.get("pull", 0.6))}}
        print(f"  guide {name} at frame {at} ({at / FPS:.2f} s)")

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
        """Jumps, halts and freezes, measured by optical flow (videocheck.py): a jump
        is a clip that did not get there or a background snapping over (longer,
        another seed, its keyframe nearer the one before); a halt or freeze a
        clip done early (shorter, a lower pull, idle motion in its prompt). Cuts
        the plan has are left out."""
        import videocheck
        joins = [j for j, _ in self.joins(video)]
        cuts = [j for j, cut in self.joins(video) if cut]
        found, _, n = videocheck.check(video, FPS, cuts, joins)
        report = [videocheck.describe(e, FPS, joins).strip() for e in found] or ["no jump, halt or freeze"]
        for line in report:
            print(f"  {line}")
        with open(self.dir / "check.md", "a") as f:
            f.write(f"\n## Motion of {Path(video).name} (optical flow)\n\n" + "\n".join(f"- {x}" for x in report) + "\n")


def main():
    ap = argparse.ArgumentParser(description="a scene from a storyboard: keyframes, a check, four LTX clips")
    ap.add_argument("plan")
    # llmctl passes these: the ComfyUI slot and the API workflows.
    ap.add_argument("--url", default="", help=argparse.SUPPRESS)
    ap.add_argument("--workflows", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--vision", help="an OpenAI base URL of a model that sees pictures, for the check")
    ap.add_argument("--render", action="store_true", help="render the clips (else: keyframes, sheet, check only)")
    ap.add_argument("--redo-clip", type=int, metavar="N", help="render clip N alone again and splice it into the video")
    ap.add_argument("--seed", type=int, help="the seed of --redo-clip (another take: another seed); written into the plan")
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
    if a.redo_clip:
        if not board.video().is_file():
            sys.exit(f"no video yet: {board.video()} — render it with --render")
        print("Video:")
        clip = plan["clips"][a.redo_clip - 1] if 1 <= a.redo_clip <= len(plan["clips"]) else {}
        seed = a.seed if a.seed is not None else int(clip.get("seed", DEFAULT_SEED))
        board.redo_clip(made, a.redo_clip, seed)
        board.record_seed(a.redo_clip, seed)
    elif a.render:
        print("Video:")
        board.render(made)
    elif not a.vision:
        print(f"Look at {board.dir / 'sheet.jpg'}; then run again with --render (and --vision URL for a check).")


if __name__ == "__main__":
    main()
