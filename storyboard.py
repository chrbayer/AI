#!/usr/bin/env python3
"""A scene from a storyboard, on a running ComfyUI — `llmctl storyboard` (#48).

A plan file names a photo, the keyframes to make from it and the four clips
between them. Three steps, each run again only where its input changed:

1. Keyframes, each straight from the photo with Qwen-Image Edit Turbo (an edit
   of an edit gathers noise). With "object", that object is cut out of
   keyframe 1 (SAM 3) and given to every later edit as a second picture, so it
   looks the same in all of them.
2. A contact sheet of the keyframes (sheet.jpg) and, with --vision, a check by a
   vision model: per picture where the person and the object are, against what
   the plan expects there, and over all pictures whether they make one scene.
   Look before rendering: an hour of LTX is worth it only for keyframes that fit.
3. With --render: the four clips, LTX-2.5 Chain Keyframes — each from one
   keyframe to the next, or, where a clip has "cut", from a picture of its own.

    storyboard.py --url URL --workflows DIR PLAN.json [--vision URL] [--render] [--out FILE]

The plan (JSON):

    {"photo": "beach.png",                     relative to the plan file
     "object": "the pale spiral seashell",     optional: SAM 3's prompt for it
     "size": "1280x704", "seed": 1,            optional
     "keys": [{"edit": "...", "check": "..."}, ... five],
     "clips": [{"prompt": "..."}, ...,        four; a clip with
               {"prompt": "...", "cut": {"edit": "...", "check": "..."}}]}

"edit" is the Qwen Edit prompt: <image1> is the photo, <image2> the object; a
picture's own "seed" tries another take of it.
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

    def edit(self, name, prompt, seed, ref=None):
        out = self.dir / f"{name}.png"
        key = digest(Path(self.plan["photo"]), prompt, seed, *([ref] if ref else []))
        if self.fresh(out, key):
            print(f"  {name}: unchanged")
            return out
        comfy = self.need_comfy()
        g = json.loads((self.workflows / EDIT).read_text())
        loads = sorted((k for k, n in g.items() if n["class_type"] == "LoadImage"), key=int)
        g[loads[0]]["inputs"]["image"] = comfy.upload(self.plan["photo"], UPLOAD_DIR)
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

    def cut_object(self, key1):
        """The object from keyframe 1: SAM 3's mask, its box grown a little, cut out."""
        from PIL import Image
        out, mask = self.dir / "object.png", self.dir / "object_mask.png"
        key = digest(key1, self.plan["object"])
        if self.fresh(out, key):
            print("  object: unchanged")
            return out
        comfy = self.need_comfy()
        g = json.loads((self.workflows / SAM).read_text())
        for n in g.values():
            if n["class_type"] == "LoadImage":
                n["inputs"]["image"] = comfy.upload(key1, UPLOAD_DIR)
            if n["class_type"] == "SAM3Segment":
                n["inputs"].update(prompt=self.plan["object"], max_segments=1)
        comfy.run(g, mask)
        m = Image.open(mask).convert("L")
        box = m.point([0] * 128 + [255] * 128).getbbox()
        if not box:
            sys.exit(f"SAM 3 found no '{self.plan['object']}' in key_1.png — reword 'object' or the first edit")
        picture = Image.open(key1).convert("RGB")
        sx, sy = picture.width / m.width, picture.height / m.height
        x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        grow = 0.15 * max(x1 - x0, y1 - y0)
        box = (max(0, int(x0 - grow)), max(0, int(y0 - grow)),
               min(picture.width, int(x1 + grow)), min(picture.height, int(y1 + grow)))
        picture.crop(box).save(out)
        self.made(out, key)
        print(f"  object: cut out of key_1 at {box}")
        return out

    def keyframes(self):
        seed = int(self.plan.get("seed", 1))
        shown = pictures(self.plan)
        made = {}
        made["key_1"] = self.edit("key_1", shown[0][1]["edit"], seed)
        ref = self.cut_object(made["key_1"]) if self.plan.get("object") else None
        for i, (name, k) in enumerate(shown[1:], 2):
            made[name] = self.edit(name, k["edit"], int(k.get("seed", seed + i)), ref)
        self.sheet(shown, made)
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

    def check(self, shown, made):
        """A vision model's look at each picture against its 'check', then at all."""
        url = self.a.vision.rstrip("/")
        url = url if url.endswith("/v1") else url + "/v1"
        thing = self.plan.get("object") or "anything the scene is about"

        def ask(content):
            try:
                r = requests.post(f"{url}/chat/completions", timeout=900, json={
                "messages": [{"role": "user", "content": content}], "max_tokens": 900, "temperature": 0.2})
            except requests.ConnectionError:
                sys.exit(f"no vision model at {url}")
            if not r.ok:
                sys.exit(f"the vision model at {url} refused: {r.text[:500]}\n"
                         "(a llama.cpp model sees pictures only when started with --mmproj)")
            return r.json()["choices"][0]["message"]["content"].strip()

        def image(p):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                       base64.b64encode(Path(p).read_bytes()).decode()}}

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
            if n["class_type"] == "PrimitiveBoolean" and t.startswith("Schnitt vor Clip "):
                n["inputs"]["value"] = f"cut_{t.split()[3]}" in made
            if n["class_type"] == "PrimitiveStringMultiline" and t.startswith("Prompt "):
                n["inputs"]["value"] = self.plan["clips"][int(t.split()[1]) - 1]["prompt"]
            if self.plan.get("size") and n["class_type"] == "PrimitiveInt" and t.lower() in ("width", "height"):
                n["inputs"]["value"] = self.plan["size"][t.lower() == "height"]
            if n["class_type"] == "SaveVideo":
                n["inputs"]["filename_prefix"] = f"video/storyboard_{self.dir.stem}"
        out = Path(self.a.out) if self.a.out else self.dir / f"{Path(self.a.plan).stem}.mp4"
        print(f"  rendering 4 clips — about an hour at 1280x704 …", flush=True)
        comfy.run(g, out)
        print(f"  video: {out}")


def main():
    ap = argparse.ArgumentParser(description="a scene from a storyboard: keyframes, a check, four LTX clips")
    ap.add_argument("plan")
    # llmctl passes these: the ComfyUI slot and the API workflows.
    ap.add_argument("--url", default="", help=argparse.SUPPRESS)
    ap.add_argument("--workflows", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--vision", help="an OpenAI base URL of a model that sees pictures, for the check")
    ap.add_argument("--render", action="store_true", help="render the clips (else: keyframes, sheet, check only)")
    ap.add_argument("--out", help="the video (default: PLAN.storyboard/PLAN.mp4)")
    a = ap.parse_args()
    plan = load_plan(a.plan)
    board = Board(a, plan)
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
