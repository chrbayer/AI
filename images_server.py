#!/usr/bin/env python3
"""OpenAI's image API in front of ComfyUI — for llmctl's comfyui slots.

    POST /v1/images/generations   {"model", "prompt", "size", "n", "seed", "response_format"}
                                  extra: "negative_prompt" (workflows with real guidance)
    POST /v1/images/edits         multipart: image (or image[]), mask, prompt, model, n, seed
                                  or JSON: "images": ["data:image/png;base64,...", ...], "mask"
                                  extra: "strength" (ControlNet), "pad" (Outpaint),
                                  "boxes" + "margin" (Inpaint: a mask made of rectangles),
                                  "layers" (Layered: how many RGBA layers come back),
                                  "select" (masked workflows: the mask from a few words, SAM 3);
                                  Qwen-Image Layered without a prompt is described by Florence-2
    POST /v1/images/select        image + prompt: a box for each thing SAM 3 finds that the words name
    POST /v1/audio/music          {"model", "prompt" (style), "lyrics", "duration", "seed", "bpm",
                                  "key", "language", "response_format": mp3|flac}: a song or an
                                  instrumental from the music workflows (comfyui/api/audio/)
    POST /v1/video/clips          multipart or JSON: "prompt", "image" (or two in image[]: first
                                  and last), "model", "seconds" (1-10), "size", "quality":
                                  fast|full, "seed": an MP4 with sound (LTX-2.5, api/video/);
                                  without an image from the prompt alone; with "audio" (up
                                  to 10 s) the picture speaks or sings it (ltx-25-talking). It stays in
                                  ComfyUI's output/llmctl-api/: there is no temp node for video
    POST /v1/images/check         image (+ prompt, max_area): the artifacts a vision model sees,
                                  with boxes to repaint and a prompt for each (needs --vision)
    GET  /v1/models               the workflows whose models ComfyUI has, and the
                                  extra fields each takes
    GET  /repair                  a page to check a picture, pick and edit the boxes, and
                                  repaint them; it starts and stops the vision slot on request

A model is a bundled workflow, in the API format comfyui/export_api.py makes
of it: "FLUX.2 klein 9B T2I (bf16, 4 Schritte).json" is flux2-klein-9b for
generations, its Edit sibling the same name for edits. The request's prompt,
size, seed and images are written into the workflow's nodes, the workflow is
queued on ComfyUI, and the images it saves come back — as PNG, base64 unless
response_format is "url". Results go to ComfyUI's temp directory, not its
output: the caller has them.

Started by `llmctl start comfy N --proxy` on the slot's proxy port.
"""
import argparse
import base64
import hashlib
import io
import json
import random
import re
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request

try:                              # the mask from boxes, /check's sizes and preview
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                                               # pragma: no cover
    Image = None

app = Flask(__name__)
ARGS = argparse.Namespace()

# Loader inputs that name a model file: ComfyUI lists the files it has for each.
MODEL_INPUTS = {"unet_name", "clip_name", "vae_name", "lora_name", "ckpt_name",
                "clip_name1", "clip_name2", "model_name"}
# Nodes that only show something in the UI; through the API they are work for nothing.
UI_ONLY = {"ImageCompare", "PreviewImage", "PreviewAny", "Note", "MarkdownNote"}
SAVE = {"SaveImage", "SaveImageAdvanced"}
LATENT_SIZE = {"EmptyFlux2LatentImage", "EmptyLatentImage", "EmptySD3LatentImage",
               "EmptyHunyuanLatent", "Flux2Scheduler"}
SIZE_MIN, SIZE_MAX, SIZE_STEP = 256, 2048, 16
STRENGTH_MAX = 2.0
PAD_MAX, PAD_STEP, FEATHER_MAX = 2048, 8, 512
MARGIN_DEFAULT, MARGIN_MAX = 16, 256
# Around a thing `select` outlines: its own outline gives the repaint its shape
# back (a floor lamp asked to become a plant came back a lamp at 16 px, a plant at 64).
SELECT_MARGIN = 64
# Music: seconds a request may ask for, and the default.
DURATION_MIN, DURATION_MAX, DURATION_DEFAULT = 5, 300, 60
# Helper workflows the image API runs itself; no models of their own.
TOOLS = {"Florence-2 Caption", "SAM 3 Select"}
# A box over this share of the picture is no place to repaint but a remark on
# the whole ("four dishes, not five"); /check lists it apart.
WHOLE_SHARE = 0.4


class Refused(Exception):
    """A request this server cannot fulfil; the message goes back as a 400."""


class Unavailable(Exception):
    """Something this server needs does not answer; a 503."""


def error(message, status=400, kind="invalid_request_error"):
    return jsonify({"error": {"message": message, "type": kind}}), status


# ── the workflows ────────────────────────────────────────────

def slug(stem):
    """'FLUX.2 klein 9B T2I (bf16, 4 Schritte)' -> ('flux2-klein-9b', 'generations')."""
    name = re.sub(r"\(.*?\)", "", stem)
    # Workflows that start from an image: edits, the upscaler, the detailer, and
    # the ControlNet ones (a control image, a photo's edges, an image to inpaint).
    kind = "edits" if re.search(r"\b(Edit|Upscale|Control|Canny|Inpaint|Outpaint|Removal|Detailer|Colorize|Layered)\b", name) else "generations"
    name = re.sub(r"\b(T2I|Edit)\b", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name.lower().replace(".", "")).strip("-")
    return name, kind


def music_workflows():
    """{name: path} for the music workflows (api/audio/), named as the image ones."""
    return {slug(p.stem)[0]: p for p in sorted((Path(ARGS.api) / "audio").glob("*.json"))}


def workflows():
    """{(name, kind): path} for every API workflow bundled."""
    out = {}
    for path in sorted(Path(ARGS.api).glob("*.json")):
        if path.stem not in TOOLS:
            out.setdefault(slug(path.stem), path)
    return out


_have = {"at": 0.0, "files": {}}
_have_lock = threading.Lock()


def model_files():
    """{input name: set of files ComfyUI offers}, fresh every 30 s."""
    with _have_lock:
        if time.time() - _have["at"] > 30:
            info = requests.get(f"{ARGS.comfy}/object_info", timeout=30).json()
            files = {}
            for node in info.values():
                for section in ("required", "optional"):
                    for key, spec in (node.get("input", {}).get(section) or {}).items():
                        if key not in MODEL_INPUTS or not spec:
                            continue
                        if isinstance(spec[0], list):                  # ["a.safetensors", …]
                            files.setdefault(key, set()).update(spec[0])
                        elif spec[0] == "COMBO" and len(spec) > 1:      # newer nodes: options apart
                            files.setdefault(key, set()).update((spec[1] or {}).get("options") or [])
            _have.update(at=time.time(), files=files)
        return _have["files"]


def missing_models(graph):
    files = model_files()
    return sorted({v for n in graph.values() for k, v in n["inputs"].items()
                   if k in MODEL_INPUTS and isinstance(v, str) and v not in files.get(k, set())})


def load(name, kind):
    path = workflows().get((name, kind))
    if path is None:
        known = sorted(n for n, k in workflows() if k == kind)
        raise Refused(f"no {kind} model '{name}' — there are: {', '.join(known)}")
    graph = json.loads(path.read_text())
    lacking = missing_models(graph)
    if lacking:
        raise Refused(f"'{name}' needs models ComfyUI does not have: {', '.join(lacking)} "
                      f"— llmctl download comfy \"{path.stem}\"")
    return graph


# ── filling a workflow in ────────────────────────────────────

def title(node):
    return node.get("_meta", {}).get("title", "")


def set_prompt(graph, prompt):
    done = False
    for node in graph.values():
        if "(fixed)" in title(node):              # a prompt of its own (the detailer's)
            continue
        if node["class_type"] == "CLIPTextEncode" and "Positive" in title(node):
            node["inputs"]["text"] = prompt; done = True
        elif node["class_type"].startswith("TextEncodeQwenImage"):
            node["inputs"]["prompt"] = prompt; done = True
    if not done:
        plain = [n for n in graph.values() if n["class_type"] == "CLIPTextEncode"
                 and "Negative" not in title(n) and "(fixed)" not in title(n)
                 and isinstance(n["inputs"].get("text"), str)]
        if plain:
            plain[0]["inputs"]["text"] = prompt; done = True
    return done


def set_size(graph, size):
    if not size or size == "auto":
        return
    m = re.fullmatch(r"(\d+)x(\d+)", size)
    if not m:
        raise Refused(f"size is WIDTHxHEIGHT or auto, not '{size}'")
    w, h = (min(SIZE_MAX, max(SIZE_MIN, int(int(v) / SIZE_STEP + 0.5) * SIZE_STEP)) for v in m.groups())
    for node in graph.values():
        if node["class_type"] == "PrimitiveInt" and title(node) in ("Width", "Height"):
            node["inputs"]["value"] = w if title(node) == "Width" else h
        elif node["class_type"] in LATENT_SIZE and "width" in node["inputs"]:
            node["inputs"]["width"], node["inputs"]["height"] = w, h


def set_seed(graph, seed):
    for node in graph.values():
        for key in ("noise_seed", "seed"):
            if isinstance(node["inputs"].get(key), int):
                node["inputs"][key] = seed


def image_nodes(graph):
    """The workflow's image inputs, in the order its Load Image nodes were made."""
    return sorted((k for k, n in graph.items() if n["class_type"] == "LoadImage"),
                  key=lambda k: [int(p) for p in k.split(":")])


def refers(value, gone):
    return isinstance(value, list) and len(value) == 2 and value[0] in gone


def prune(graph, gone):
    """Remove the nodes in `gone` and whatever only served them. A reference
    image that is not there drops out of a ReferenceLatent chain and out of a
    Qwen-Image encoder's optional images; anything else that needed it goes too."""
    gone = set(gone)
    changed = True
    while changed:
        changed = False
        for key, node in list(graph.items()):
            if key in gone:
                continue
            for name, value in list(node["inputs"].items()):
                if not refers(value, gone):
                    continue
                if name.startswith("images."):
                    del node["inputs"][name]
                elif node["class_type"] == "ReferenceLatent" and name == "latent":
                    through = node["inputs"]["conditioning"]
                    for other in graph.values():
                        for n2, v2 in other["inputs"].items():
                            if v2 == [key, 0]:
                                other["inputs"][n2] = through
                    gone.add(key)
                else:
                    gone.add(key)
                changed = True
                break
    for key in gone:
        graph.pop(key, None)


def mask_users(graph):
    """Where the workflow reads the mask its first image carries as transparency
    to redraw there. The upscaler only puts that transparency back on its result."""
    slots = image_nodes(graph)
    if not slots:
        return []
    return [(node, name) for node in graph.values() for name, value in node["inputs"].items()
            if value == [slots[0], 1] and node["class_type"] != "JoinImageWithAlpha"]


def negative_nodes(graph):
    """A workflow's negative prompt, where it does something: a text encoder titled
    Negative, and a sampler with real guidance (cfg 1 ignores the negative)."""
    sampled = any(n["class_type"] == "KSampler" and isinstance(n["inputs"].get("cfg"), (int, float))
                  and n["inputs"]["cfg"] > 1 for n in graph.values())
    return [n for n in graph.values() if sampled and n["class_type"] == "CLIPTextEncode"
            and "Negative" in title(n) and isinstance(n["inputs"].get("text"), str)]


def set_negative(graph, negative):
    nodes = negative_nodes(graph)
    if not nodes:
        raise Refused("this workflow takes no negative prompt (no guidance to steer away with)")
    if not isinstance(negative, str):
        raise Refused("negative_prompt is text")
    for node in nodes:
        node["inputs"]["text"] = negative


def extras(graph):
    """The extra request fields this workflow takes."""
    out = []
    if negative_nodes(graph):
        out.append("negative_prompt")
    if mask_users(graph):
        out += ["mask", "boxes", "select"]
    kinds = {n["class_type"] for n in graph.values()}
    if "ZImageFunControlnet" in kinds:
        out.append("strength")
    if "ImagePadForOutpaint" in kinds:
        out.append("pad")
    if "EmptyQwenImageLayeredLatentImage" in kinds:
        out.append("layers")
    return out


def png_info(data):
    """(width, height, has alpha) of a PNG, or None for anything else."""
    if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return w, h, data[25] in (4, 6) or b"tRNS" in data


def set_mask(graph, mask, image):
    """OpenAI's mask: transparent where the image is to be redrawn. It takes the
    place of the mask the image's own transparency gives."""
    users = mask_users(graph)
    if not users:
        raise Refused("this workflow takes no mask — the Inpaint ones do")
    info = png_info(mask)
    if info is None:
        raise Refused("the mask is a PNG")
    if not info[2]:
        raise Refused("the mask needs transparency: transparent is redrawn, opaque is kept")
    size = png_info(image)
    if size and size[:2] != info[:2]:
        raise Refused(f"the mask is {info[0]}x{info[1]}, the image {size[0]}x{size[1]}")
    key = str(max(int(k.split(":")[0]) for k in graph) + 1)
    graph[key] = {"class_type": "LoadImageMask", "inputs": {"image": upload(mask), "channel": "alpha"},
                  "_meta": {"title": "Mask (API)"}}
    for node, name in users:
        node["inputs"][name] = [key, 0]


def set_strength(graph, strength):
    nodes = [n for n in graph.values() if n["class_type"] == "ZImageFunControlnet"]
    if not nodes:
        raise Refused("this workflow has no ControlNet to set a strength for")
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        raise Refused(f"strength is a number, not '{strength}'") from None
    if not 0 <= strength <= STRENGTH_MAX:
        raise Refused(f"strength is 0 to {STRENGTH_MAX:g}")
    for node in nodes:
        node["inputs"]["strength"] = strength


LAYERS_MAX = 8


def set_layers(graph, layers):
    """How many RGBA layers Qwen-Image-Layered splits the picture into: the
    background and one per thing. More than the scene holds gives empty or
    doubled ones; two suit a person in front of a scene, four a poster."""
    nodes = [n for n in graph.values() if n["class_type"] == "EmptyQwenImageLayeredLatentImage"]
    if not nodes:
        raise Refused("this workflow does not split into layers — the Layered one does")
    try:
        layers = int(layers)
    except (TypeError, ValueError):
        raise Refused(f"layers is a number, not '{layers}'") from None
    if not 1 <= layers <= LAYERS_MAX:
        raise Refused(f"layers is 1 to {LAYERS_MAX}")
    for node in nodes:
        node["inputs"]["layers"] = layers


def set_pad(graph, pad):
    """pad: pixels on every side, or {"left", "top", "right", "bottom", "feathering"}."""
    nodes = [n for n in graph.values() if n["class_type"] == "ImagePadForOutpaint"]
    if not nodes:
        raise Refused("this workflow does not pad — the Outpaint ones do")
    if isinstance(pad, str):
        try:
            pad = json.loads(pad)
        except ValueError:
            raise Refused(f"pad is a number or an object, not '{pad}'") from None
    sides = ("left", "top", "right", "bottom")
    if isinstance(pad, int) and not isinstance(pad, bool):
        pad = dict.fromkeys(sides, pad)
    if not isinstance(pad, dict) or set(pad) - {*sides, "feathering"}:
        raise Refused('pad is a number or {"left", "top", "right", "bottom", "feathering"}')
    values = {}
    for side in sides:
        v = pad.get(side, 0)
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= PAD_MAX or v % PAD_STEP:
            raise Refused(f"pad {side} is 0 to {PAD_MAX} in steps of {PAD_STEP}, not {v!r}")
        values[side] = v
    if not any(values.values()):
        raise Refused("pad adds nothing on any side")
    if "feathering" in pad:
        f = pad["feathering"]
        if not isinstance(f, int) or isinstance(f, bool) or not 0 <= f <= FEATHER_MAX:
            raise Refused(f"pad feathering is 0 to {FEATHER_MAX}, not {f!r}")
        values["feathering"] = f
    for node in nodes:
        node["inputs"].update(values)


def pillow():
    if Image is None:
        raise Refused("this needs Pillow beside the image API (pip install pillow)")


def image_size(data):
    pillow()
    try:
        return Image.open(io.BytesIO(data)).size
    except Exception:                                             # noqa: BLE001
        raise Refused("that is no image Pillow can read") from None


def boxes_mask(boxes, size, margin=None):
    """OpenAI's mask from rectangles in the image's pixels: transparent inside
    each, grown by `margin` on every side, opaque elsewhere."""
    pillow()
    if isinstance(boxes, str):
        try:
            boxes = json.loads(boxes)
        except ValueError:
            raise Refused("boxes is a JSON list of [x1, y1, x2, y2]") from None
    if not isinstance(boxes, list) or not boxes or not all(
            isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)
            for b in boxes):
        raise Refused("boxes is a non-empty list of [x1, y1, x2, y2] in the image's pixels")
    try:
        margin = MARGIN_DEFAULT if margin in (None, "") else int(margin)
    except (TypeError, ValueError):
        raise Refused(f"margin is a number of pixels, not '{margin}'") from None
    if not 0 <= margin <= MARGIN_MAX:
        raise Refused(f"margin is 0 to {MARGIN_MAX}")
    w, h = size
    mask = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        draw.rectangle([max(0, x1 - margin), max(0, y1 - margin),
                        min(w - 1, x2 + margin), min(h - 1, y2 + margin)], fill=(0, 0, 0, 0))
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return buf.getvalue()


def upload(data, suffix=".png"):
    name = hashlib.sha256(data).hexdigest()[:24] + suffix
    r = requests.post(f"{ARGS.comfy}/upload/image",
                      files={"image": (name, data)},
                      data={"subfolder": "llmctl-api", "type": "input", "overwrite": "true"},
                      timeout=60)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def prepare(graph, prompt, size, images, mask=None, strength=None, pad=None, boxes=None, margin=None,
            negative_prompt=None, layers=None, select=None):
    for key in [k for k, n in graph.items() if n["class_type"] in UI_ONLY]:
        del graph[key]
    for node in graph.values():               # into temp, not the user's output
        if node["class_type"] in SAVE:
            node["class_type"] = "PreviewImage"
            node["inputs"] = {"images": node["inputs"]["images"]}
    # A workflow without a prompt (the upscaler) ignores one: OpenAI's clients
    # always send it.
    if prompt is not None:
        set_prompt(graph, prompt)
    if negative_prompt not in (None, ""):
        set_negative(graph, negative_prompt)
    set_size(graph, size)
    if images is not None:
        slots = image_nodes(graph)
        if not images:
            raise Refused("an edit needs at least one image")
        if len(images) > len(slots):
            raise Refused(f"this workflow takes at most {len(slots)} image(s), got {len(images)}")
        for key, data in zip(slots, images):
            graph[key]["inputs"]["image"] = upload(data)
        if sum(x not in (None, "") for x in (mask, boxes, select)) > 1:
            raise Refused("one of mask, boxes or select")
        if select not in (None, ""):
            if not mask_users(graph):
                raise Refused("this workflow takes no mask — the Inpaint ones do")
            mask = select_mask(images[0], select, margin)
        if boxes is not None:
            if not mask_users(graph):
                raise Refused("this workflow takes no mask — the Inpaint ones do")
            mask = boxes_mask(boxes, image_size(images[0]), margin)
        if mask is not None:
            set_mask(graph, mask, images[0])
        prune(graph, slots[len(images):])
    if strength is not None:
        set_strength(graph, strength)
    if pad is not None:
        set_pad(graph, pad)
    if layers not in (None, ""):
        set_layers(graph, layers)
    if not any(n["class_type"] == "PreviewImage" for n in graph.values()):
        raise Refused("with these images the workflow has nothing left to produce")


# ── running it ───────────────────────────────────────────────

def execute(graph):
    """Queue the workflow, wait for it, return its outputs as ComfyUI's history has them."""
    r = requests.post(f"{ARGS.comfy}/prompt",
                      json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=60)
    if r.status_code != 200:
        try:
            j = r.json()
            detail = j.get("error", {}).get("message", "")
            for nid, ne in (j.get("node_errors") or {}).items():
                for e in ne.get("errors", []):
                    detail += f"; node {nid} ({ne.get('class_type')}): {e.get('details') or e.get('message')}"
        except ValueError:
            detail = r.text[:300]
        raise Refused(f"ComfyUI refused the workflow: {detail}")
    pid = r.json()["prompt_id"]
    deadline = time.time() + ARGS.timeout
    while time.time() < deadline:
        h = requests.get(f"{ARGS.comfy}/history/{pid}", timeout=30).json().get(pid)
        if h and h.get("status", {}).get("completed"):
            break
        if h and h.get("status", {}).get("status_str") == "error":
            msgs = [m[1].get("exception_message", "") for m in h["status"].get("messages", [])
                    if m[0] == "execution_error"]
            raise RuntimeError("ComfyUI failed: " + ("; ".join(msgs) or "see its log"))
        time.sleep(0.5)
    else:
        # Out of the queue, or stopped if it already runs: nobody waits for it
        # any more, and a client that retries would otherwise add a second one.
        requests.post(f"{ARGS.comfy}/queue", json={"delete": [pid]}, timeout=10)
        requests.post(f"{ARGS.comfy}/interrupt", json={"prompt_id": pid}, timeout=10)
        raise TimeoutError(f"no result within {ARGS.timeout} s — stopped it in ComfyUI")
    return h["outputs"]


def run(graph):
    """Queue the workflow, wait for it, return its images as (bytes, view URL)."""
    out = []
    for node in execute(graph).values():
        for img in node.get("images", []):
            q = {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                 "type": img.get("type", "temp")}
            v = requests.get(f"{ARGS.comfy}/view", params=q, timeout=60)
            v.raise_for_status()
            out.append((v.content, requests.Request("GET", f"{ARGS.public_comfy}/view",
                                                    params=q).prepare().url))
    return out


def tool(name):
    path = Path(ARGS.api) / f"{name}.json"
    if not path.is_file():
        raise Unavailable(f"the helper workflow '{name}' is missing from {ARGS.api}")
    graph = json.loads(path.read_text())
    lacking = missing_models(graph)
    if lacking:
        raise Refused(f"'{name}' needs models ComfyUI does not have: {', '.join(lacking)} "
                      f"— llmctl download comfy \"{name}\"")
    return graph


def caption(data):
    """Florence-2's detailed description of the picture."""
    graph = tool("Florence-2 Caption")
    graph[image_nodes(graph)[0]]["inputs"]["image"] = upload(data)
    for out in execute(graph).values():
        if out.get("text"):
            return out["text"][0].strip()
    raise RuntimeError("Florence-2 gave no caption")


def segments(data, words):
    """One mask (a PIL "L" image, white where it is) for each thing SAM 3 finds."""
    pillow()
    if not isinstance(words, str) or not words.strip():
        raise Refused("select names what to find, in a few words")
    graph = tool("SAM 3 Select")
    graph[image_nodes(graph)[0]]["inputs"]["image"] = upload(data)
    sam = next(n for n in graph.values() if n["class_type"] == "SAM3Segment")
    sam["inputs"]["prompt"] = words.strip()
    masks = [Image.open(io.BytesIO(png)).convert("L").point(lambda v: 255 if v > 127 else 0) for png, _ in run(graph)]
    return [m for m in masks if m.getbbox()]


def select_mask(data, words, margin=None):
    """OpenAI's mask from words: transparent over all SAM 3 finds, grown by `margin`."""
    from PIL import ImageChops, ImageFilter
    try:
        margin = SELECT_MARGIN if margin in (None, "") else int(margin)
    except (TypeError, ValueError):
        raise Refused(f"margin is a number of pixels, not '{margin}'") from None
    if not 0 <= margin <= MARGIN_MAX:
        raise Refused(f"margin is 0 to {MARGIN_MAX}")
    found = segments(data, words)
    if not found:
        raise Refused(f"SAM 3 finds no '{words.strip()}' in the picture")
    union = found[0]
    for m in found[1:]:
        union = ImageChops.lighter(union, m)
    if margin:                     # grown: blurred, and whatever the blur reaches
        union = union.filter(ImageFilter.GaussianBlur(margin / 2)).point(lambda v: 255 if v > 4 else 0)
    size = image_size(data)
    if union.size != size:
        union = union.resize(size)
    mask = Image.new("RGBA", size, (0, 0, 0, 255))
    mask.putalpha(ImageChops.invert(union))
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return buf.getvalue()


def respond(name, kind, prompt, size, n, seed, fmt, images=None, **extra):
    if fmt not in ("b64_json", "url"):
        raise Refused("response_format is b64_json or url")
    if not 1 <= n <= 8:
        raise Refused("n is 1 to 8")
    template = load(name, kind)
    # Qwen-Image Layered wants the picture described; without a prompt Florence-2 does it.
    if name == "qwen-image-layered" and images and not (prompt or "").strip():
        prompt = caption(images[0])
        app.logger.info("layered: Florence-2 caption %r", prompt[:120])
    seed = random.randrange(2**48) if seed is None else int(seed)
    data = []
    t0 = time.time()
    for i in range(n):
        graph = json.loads(json.dumps(template))
        prepare(graph, prompt, size, images, **extra)
        set_seed(graph, seed + i)
        for png, url in run(graph):
            data.append({"b64_json": base64.b64encode(png).decode()} if fmt == "b64_json"
                        else {"url": url})
    app.logger.info("%s %s: %d image(s) in %.1f s, seed %d", kind, name, len(data), time.time() - t0, seed)
    return jsonify({"created": int(time.time()), "data": data, "seed": seed})


def default_model(kind):
    names = sorted(n for n, k in workflows() if k == kind)
    for n in names:
        if "klein" in n and "nsfw" not in n and not missing_models(json.loads(workflows()[(n, kind)].read_text())):
            return n
    return names[0] if names else ""


def handle(fn):
    try:
        return fn()
    except Refused as e:
        return error(str(e))
    except TimeoutError as e:
        return error(str(e), 504, "timeout")
    except Unavailable as e:
        return error(str(e), 503, "unavailable")
    except requests.ConnectionError:
        return error(f"ComfyUI does not answer at {ARGS.comfy} (still starting?)", 503, "unavailable")
    except Exception as e:                                        # noqa: BLE001
        app.logger.exception("request failed")
        return error(str(e), 500, "server_error")


# ── endpoints ────────────────────────────────────────────────

@app.post("/v1/images/generations")
def generations():
    def go():
        j = request.get_json(silent=True) or {}
        if not isinstance(j.get("prompt"), str) or not j["prompt"].strip():
            raise Refused("prompt is required")
        return respond(j.get("model") or default_model("generations"), "generations", j["prompt"],
                       j.get("size"), int(j.get("n") or 1), j.get("seed"),
                       j.get("response_format") or "b64_json", negative_prompt=j.get("negative_prompt"))
    return handle(go)


def data_url(s):
    if not isinstance(s, str):
        s = (s or {}).get("image_url") or ""
    return base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)


@app.post("/v1/images/edits")
def edits():
    def go():
        if request.files:
            f = request.form
            images = [x.read() for key in ("image", "image[]") for x in request.files.getlist(key)]
            prompt, model, size = f.get("prompt", ""), f.get("model"), f.get("size")
            n, seed, fmt = int(f.get("n") or 1), f.get("seed"), f.get("response_format") or "b64_json"
            mask = request.files["mask"].read() if "mask" in request.files else None
            strength, pad = f.get("strength") or None, f.get("pad") or None
            negative = f.get("negative_prompt")
            layers = f.get("layers")
            boxes, margin = f.get("boxes") or None, f.get("margin")
            select = f.get("select")
        else:
            j = request.get_json(silent=True) or {}
            raw = j.get("images") or j.get("image") or []
            images = [data_url(x) for x in (raw if isinstance(raw, list) else [raw])]
            prompt, model, size = j.get("prompt", ""), j.get("model"), j.get("size")
            n, seed, fmt = int(j.get("n") or 1), j.get("seed"), j.get("response_format") or "b64_json"
            mask = data_url(j["mask"]) if j.get("mask") else None
            strength, pad = j.get("strength"), j.get("pad")
            negative = j.get("negative_prompt")
            layers = j.get("layers")
            boxes, margin = j.get("boxes"), j.get("margin")
            select = j.get("select")
        return respond(model or default_model("edits"), "edits", prompt, size, n,
                       seed if seed in (None, "") else int(seed), fmt, images,
                       mask=mask, strength=strength, pad=pad, boxes=boxes, margin=margin,
                       negative_prompt=negative, layers=layers, select=select)
    return handle(go)


@app.post("/v1/images/select")
def select_endpoint():
    """A box [x1, y1, x2, y2] in pixels for each thing SAM 3 finds that the
    prompt names — places for `boxes`, or for the repair page."""
    def go():
        if request.files:
            data, words = request.files["image"].read(), request.form.get("prompt", "")
        else:
            j = request.get_json(silent=True) or {}
            if not j.get("image"):
                raise Refused("image is required")
            data, words = data_url(j["image"]), j.get("prompt", "")
        t0 = time.time()
        found = segments(data, words)
        w, h = image_size(data)
        out = []
        for m in found:
            if m.size != (w, h):
                m = m.resize((w, h))
            x1, y1, x2, y2 = m.getbbox()
            out.append({"box": [x1, y1, x2, y2], "share": round(sum(m.histogram()[255:]) / (w * h), 4)})
        return jsonify({"segments": out, "seconds": round(time.time() - t0, 1)})
    return handle(go)


# ── checking a picture ───────────────────────────────────────

CHECK_PROMPT = """You are a strict quality inspector for AI-generated images.{about}
Find generation artifacts: extra, missing or fused fingers, hands or limbs; malformed or melted faces; duplicated or ghosted objects; body parts that merge with objects or other people; physically impossible geometry; garbled or misspelled text; objects that make no sense.
Do not report style, lighting or composition choices. Only report real defects you can see.
Answer only with JSON: {{"artifacts": [{{"what": "short description", "fix": "what this region should show instead, as a short image description to repaint it", "bbox_2d": [x1, y1, x2, y2]}}]}} with coordinates on a 0-1000 scale for both axes. Use an empty list if the image is clean."""


def parse_artifacts(text):
    """The artifacts in a vision model's answer: an object with "artifacts",
    or a bare list; "label" for "what". None when there is no JSON in it."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None
    j = j.get("artifacts") if isinstance(j, dict) else j
    if not isinstance(j, list):
        return None
    out = []
    for a in j:
        b = a.get("bbox_2d") if isinstance(a, dict) else None
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            out.append({"what": str(a.get("what") or a.get("label") or ""),
                        "fix": str(a.get("fix") or ""), "bbox_2d": b})
    return out


def area_limit(value):
    if value in (None, ""):
        return WHOLE_SHARE
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise Refused(f"max_area is a share of the picture, not '{value}'") from None
    if not 0 < value <= 1:
        raise Refused("max_area is a share of the picture above 0, up to 1")
    return value


def to_pixels(artifacts, size, max_area=WHOLE_SHARE):
    """0–1000 boxes to the image's pixels. A box over `max_area` of the picture
    is a remark, not a place to repaint: repainting that much makes new flaws."""
    w, h = size
    places, remarks = [], []
    for a in artifacts:
        x1, y1, x2, y2 = a["bbox_2d"]
        x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
        y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
        box = [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)]
        if (x2 - x1) * (y2 - y1) > max_area * 1e6:
            remarks.append({"what": a["what"], "fix": a["fix"], "box": box})
            continue
        places.append({"id": len(places) + 1, "what": a["what"], "fix": a["fix"], "box": box})
    return places, remarks


def preview(data, places):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(im)
    width = max(3, im.width // 300)
    font = ImageFont.load_default(max(20, im.width // 30))
    for a in places:
        x1, y1, x2, y2 = a["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=width)
        draw.text((x1 + 2 * width, y1 + width), str(a["id"]), fill="red", font=font)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def vision_model():
    try:
        return requests.get(f"{ARGS.vision}/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception:                                             # noqa: BLE001
        return "vision"


def check(data, prompt=None, want_preview=False, max_area=None):
    if not getattr(ARGS, "vision", ""):
        raise Refused("no vision model — start the image API with --vision SLOT (flash with --mmproj)")
    max_area = area_limit(max_area)
    size = image_size(data)
    about = f' It was generated from the prompt: "{prompt}".' if prompt else ""
    t0 = time.time()
    try:
        r = requests.post(f"{ARGS.vision}/v1/chat/completions", timeout=ARGS.timeout, json={
            "model": vision_model(), "temperature": 0, "max_tokens": 12000,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                {"type": "text", "text": CHECK_PROMPT.format(about=about)}]}]})
    except requests.ConnectionError:
        raise Unavailable(f"the vision model does not answer at {ARGS.vision}") from None
    j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or "choices" not in j:
        detail = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else r.text[:300]
        raise RuntimeError(f"the vision model failed: {detail} (loaded with --mmproj?)")
    artifacts = parse_artifacts(j["choices"][0]["message"].get("content"))
    if artifacts is None:
        raise RuntimeError("the vision model gave no artifact list")
    places, remarks = to_pixels(artifacts, size, max_area)
    out = {"size": list(size), "artifacts": places, "remarks": remarks,
           "seconds": round(time.time() - t0, 1)}
    if want_preview:
        out["preview"] = preview(data, places)
    app.logger.info("check: %d artifact(s), %d remark(s) in %.1f s", len(places), len(remarks), out["seconds"])
    return out


@app.post("/v1/images/check")
def check_endpoint():
    def go():
        if request.files:
            f = request.form
            files = [x.read() for key in ("image", "image[]") for x in request.files.getlist(key)]
            prompt, want = f.get("prompt"), f.get("preview", "").lower() in ("1", "true", "yes")
            max_area = f.get("max_area")
        else:
            j = request.get_json(silent=True) or {}
            raw = j.get("image") or j.get("images") or []
            files = [data_url(x) for x in (raw if isinstance(raw, list) else [raw])]
            prompt, want = j.get("prompt"), bool(j.get("preview"))
            max_area = j.get("max_area")
        if len(files) != 1:
            raise Refused("a check takes exactly one image")
        return jsonify(check(files[0], prompt or None, want, max_area))
    return handle(go)


# ── the repair page ──────────────────────────────────────────

REPAIR_PAGE = Path(__file__).with_name("images_repair.html")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_vision = {"state": None, "message": ""}
_vision_lock = threading.Lock()


def vision_up():
    if not getattr(ARGS, "vision", ""):
        return False
    try:
        return requests.get(f"{ARGS.vision}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def comfy_up():
    try:
        return requests.get(f"{ARGS.comfy}/system_stats", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def controllable():
    return bool(getattr(ARGS, "llmctl", None) and getattr(ARGS, "vision_slot", None)
                and getattr(ARGS, "vision_model", None))


def run_vision(action):
    """Start or stop the vision slot through llmctl, in the background. Before
    a start ComfyUI lets go of its models: beside flash they do not fit."""
    slot = str(ARGS.vision_slot)
    try:
        if action == "start":
            try:
                requests.post(f"{ARGS.comfy}/free", json={"unload_models": True, "free_memory": True}, timeout=10)
                time.sleep(3)
            except requests.RequestException:
                pass
            cmd = [ARGS.llmctl, "start", ARGS.vision_model, slot, "--mmproj"]
        else:
            cmd = [ARGS.llmctl, "stop", slot]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        if r.returncode != 0:
            _vision.update(state=None, message=f"llmctl {action} failed: {tail}")
            return
        if action == "start":
            deadline = time.time() + 900
            while time.time() < deadline and not vision_up():
                time.sleep(3)
            ok = vision_up()
            _vision.update(state=None, message="" if ok else "the vision slot did not come up; see llmctl logs " + slot)
        else:
            _vision.update(state=None, message="")
    except Exception as e:                                        # noqa: BLE001
        _vision.update(state=None, message=f"llmctl {action}: {e}")


@app.get("/repair")
def repair_page():
    return Response(REPAIR_PAGE.read_text(), mimetype="text/html")


@app.get("/repair/status")
def repair_status():
    up = vision_up()
    state = _vision["state"] or ("up" if up else "down")
    return jsonify({"vision": {"configured": bool(getattr(ARGS, "vision", "")), "state": state,
                               "controllable": controllable(), "model": getattr(ARGS, "vision_model", None),
                               "slot": getattr(ARGS, "vision_slot", None), "message": _vision["message"]},
                    "comfy": {"up": comfy_up()}})


@app.post("/repair/vision")
def repair_vision():
    def go():
        action = (request.get_json(silent=True) or {}).get("action")
        if action not in ("start", "stop"):
            raise Refused("action is start or stop")
        if not controllable():
            raise Refused("this image API was not started with a vision slot llmctl can start and stop")
        with _vision_lock:
            if _vision["state"]:
                raise Refused(f"the vision slot is already {_vision['state']}")
            _vision.update(state="starting" if action == "start" else "stopping", message="")
        threading.Thread(target=run_vision, args=(action,), daemon=True).start()
        return jsonify({"state": _vision["state"]})
    return handle(go)


@app.get("/repair/recent")
def repair_recent():
    def go():
        r = requests.get(f"{ARGS.comfy}/internal/files/output", timeout=10)
        r.raise_for_status()
        names = [n.rsplit(" [", 1)[0] for n in r.json()]
        return jsonify([n for n in names if n.lower().endswith(IMAGE_SUFFIXES)][:60])
    return handle(go)


@app.get("/repair/image")
def repair_image():
    def go():
        name = request.args.get("name", "")
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise Refused("name is a file in ComfyUI's output")
        v = requests.get(f"{ARGS.comfy}/view", params={"filename": name, "type": "output"}, timeout=60)
        if v.status_code != 200:
            raise Refused(f"ComfyUI has no output '{name}'")
        if request.args.get("thumb"):
            pillow()
            im = Image.open(io.BytesIO(v.content)).convert("RGB")
            im.thumbnail((256, 256))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            return Response(buf.getvalue(), mimetype="image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})
        return Response(v.content, mimetype=v.headers.get("content-type", "image/png"))
    return handle(go)


# ── music ────────────────────────────────────────────────────

def set_music(graph, prompt, lyrics=None, duration=None, seed=None, bpm=None, key=None, language=None):
    """The request into a music workflow: ACE-Step's text encoder, YuE2's style and
    lyrics, Stable Audio's description; the length wherever a node holds it."""
    duration = DURATION_DEFAULT if duration in (None, "") else float(duration)
    if not DURATION_MIN <= duration <= DURATION_MAX:
        raise Refused(f"duration is {DURATION_MIN} to {DURATION_MAX} seconds")
    kinds = {n["class_type"] for n in graph.values()}
    sings = bool(kinds & {"TextEncodeAceStepAudio1.5", "YuE2GenerateMusic"})
    if lyrics and not sings:
        raise Refused("this model makes instrumentals and sounds only — no lyrics")
    for n in graph.values():
        c, i, t = n["class_type"], n["inputs"], title(n)
        if c == "TextEncodeAceStepAudio1.5":
            i.update(tags=prompt, lyrics=lyrics or "[Instrumental]", duration=duration)
            if bpm not in (None, ""):
                i["bpm"] = int(bpm)
            if key:
                i["keyscale"] = key
            if language:
                i["language"] = language
        elif c == "EmptyAceStep1.5LatentAudio" and not isinstance(i.get("seconds"), list):
            i["seconds"] = duration
        elif c == "YuE2GenerateMusic":
            i["max_duration"] = int(duration)
        elif c == "PrimitiveStringMultiline" and "Style" in t:
            i["value"] = prompt
        elif c == "PrimitiveStringMultiline" and "Lyrics" in t:
            i["value"] = lyrics or "[Instrumental]"
        elif c == "PrimitiveStringMultiline" and "description" in t:
            i["value"] = prompt
        elif c == "PrimitiveFloat" and "Duration" in t:
            i["value"] = duration
    if seed is not None:
        for n in graph.values():
            for k in ("seed", "noise_seed"):
                if isinstance(n["inputs"].get(k), int):
                    n["inputs"][k] = int(seed)
    # Into temp, not the gallery: PreviewAudio writes FLAC there. PreviewAny stays —
    # Stable Audio and YuE2 pass their text on through it.
    for n in graph.values():
        if n["class_type"].startswith("SaveAudio"):
            n["class_type"], n["inputs"] = "PreviewAudio", {"audio": n["inputs"]["audio"]}


def to_mp3(flac):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "0",
                        "-f", "mp3", "pipe:1"], input=flac, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-300:])
    return r.stdout


@app.post("/v1/audio/music")
def music():
    def go():
        j = request.get_json(silent=True) or {}
        if not isinstance(j.get("prompt"), str) or not j["prompt"].strip():
            raise Refused("prompt (the style or the description) is required")
        fmt = j.get("response_format") or "mp3"
        if fmt not in ("mp3", "flac"):
            raise Refused("response_format is mp3 or flac")
        known = music_workflows()
        name = j.get("model") or ("ace-step-15-xl-turbo" if "ace-step-15-xl-turbo" in known else next(iter(known), ""))
        if name not in known:
            raise Refused(f"no music model '{name}' — there are: {', '.join(known)}")
        graph = json.loads(known[name].read_text())
        lacking = missing_models(graph)
        if lacking:
            raise Refused(f"'{name}' needs models ComfyUI does not have: {', '.join(lacking)} "
                          f"— llmctl download comfy \"{known[name].stem}\"")
        seed = random.randrange(2**48) if j.get("seed") in (None, "") else int(j["seed"])
        set_music(graph, j["prompt"].strip(), j.get("lyrics"), j.get("duration"), seed,
                  j.get("bpm"), j.get("key"), j.get("language"))
        t0 = time.time()
        clips = [a for out in execute(graph).values() for a in out.get("audio", [])]
        if not clips:
            raise RuntimeError("the workflow gave no audio")
        q = {"filename": clips[0]["filename"], "subfolder": clips[0].get("subfolder", ""), "type": clips[0].get("type", "temp")}
        v = requests.get(f"{ARGS.comfy}/view", params=q, timeout=120)
        v.raise_for_status()
        body = v.content if fmt == "flac" else to_mp3(v.content)
        app.logger.info("music %s: %.0f s of audio in %.1f s, seed %d", name, float(j.get("duration") or DURATION_DEFAULT),
                        time.time() - t0, seed)
        return Response(body, mimetype="audio/flac" if fmt == "flac" else "audio/mpeg",
                        headers={"X-Seed": str(seed), "X-Model": name})
    return handle(go)


# ── video clips ──────────────────────────────────────────────

# The clip workflows (api/video/) this API serves: a picture comes to life, or
# a prompt alone; or the clip between a first and a last picture; or a picture
# that speaks or sings along to a voice, as long as the voice. The chains are
# `llmctl musicvideo`'s.
CLIP_WORKFLOWS = {"ltx-25-video": "LTX-2.5 Video (int8, distilled)",
                  "ltx-25-first-last-frame": "LTX-2.5 First-Last Frame (int8, distilled)",
                  "ltx-25-talking": "LTX-2.5 Talking (int8, distilled)"}
CLIP_SECONDS_MIN, CLIP_SECONDS_MAX, CLIP_SECONDS_DEFAULT = 1, 10, 5
# The templates' 1280×704: ~0.9 MP, in steps of 64 — the first stage samples at
# half the size, and LTX's VAE wants multiples of 32 there.
CLIP_PIXELS, CLIP_STEP = 1280 * 704, 64
# Fast without a first stage to stop at (First-Last Frame): this share of the pixels.
FAST_SHARE = 0.36


def clip_workflows():
    """{name: path} of the clip workflows ComfyUI's bundle has."""
    folder = Path(ARGS.api) / "video"
    return {name: folder / f"{stem}.json" for name, stem in CLIP_WORKFLOWS.items()
            if (folder / f"{stem}.json").is_file()}


def clip_size(size, image=None, pixels=None):
    """(width, height) of the clip: the size asked for, else the picture's shape
    at ~0.9 MP (the template would cut a portrait to 16:9), else 16:9 — at
    `pixels` if given, in steps of 64."""
    if size and size != "auto":
        m = re.fullmatch(r"(\d+)x(\d+)", size)
        if not m:
            raise Refused(f"size is WIDTHxHEIGHT or auto, not '{size}'")
        w, h = int(m[1]), int(m[2])
        pixels = pixels or w * h
    elif image is not None:
        w, h = image_size(image)
    else:
        w, h = 16, 9
    k = ((pixels or CLIP_PIXELS) / (w * h)) ** 0.5
    w, h = (min(1920, max(256, round(v * k / CLIP_STEP) * CLIP_STEP)) for v in (w, h))
    return w, h


def clip_node(graph, cls, name):
    for node in graph.values():
        if node["class_type"] == cls and title(node).lower() == name.lower():
            return node
    raise Refused(f"the workflow has no {cls} '{name}'")


def first_stage(graph):
    """Fast: the clip straight from the first stage, at half the size — the
    decoders take its latent and the second stage is never run. Only the
    two-stage workflows (Video); the others are made smaller instead."""
    seps = [k for k, n in graph.items() if n["class_type"] == "LTXVSeparateAVLatent"]
    decoders = [n for n in graph.values() if n["class_type"] in ("VAEDecodeTiled", "LTXVAudioVAEDecode")
                and refers(n["inputs"].get("samples"), set(seps))]
    used = {n["inputs"]["samples"][0] for n in decoders}
    if len(seps) != 2 or len(used) != 1:
        return False
    other = next(k for k in seps if k not in used)
    for n in decoders:
        n["inputs"]["samples"] = [other, n["inputs"]["samples"][1]]
    return True


def voice_wav(data):
    """The voice as WAV, as LoadAudio reads it whatever the phone recorded
    (WebM/Opus, MP4/AAC), and its length in seconds."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
                       input=data, capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise Refused("that is no audio ffmpeg can read")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(r.stdout)
    return buf.getvalue(), len(r.stdout) / (2 * 48000)


def set_clip(graph, name, prompt, images, seconds, size, fast, seed, audio=None):
    """The request into a clip workflow; returns the clip's (width, height)."""
    for key in [k for k, n in graph.items() if n["class_type"] in UI_ONLY]:
        del graph[key]
    talks = any(n["class_type"] == "LoadAudio" for n in graph.values())
    if talks != (audio is not None):
        raise Refused("the talking clip takes an audio file — and only it does")
    if talks:
        if not images:
            raise Refused("the talking clip takes a picture of who speaks")
        wav, length = voice_wav(audio)
        if not 0.3 <= length <= CLIP_SECONDS_MAX:
            raise Refused(f"the voice is {length:.1f} s; it may be up to {CLIP_SECONDS_MAX} s")
        clip_node(graph, "LoadAudio", "Voice or song")["inputs"]["audio"] = upload(wav, ".wav")
        seconds = None
    try:
        seconds = None if talks else CLIP_SECONDS_DEFAULT if seconds in (None, "") else int(seconds)
    except (TypeError, ValueError):
        raise Refused(f"seconds is a whole number, not '{seconds}'") from None
    if not talks and not CLIP_SECONDS_MIN <= seconds <= CLIP_SECONDS_MAX:
        raise Refused(f"seconds is {CLIP_SECONDS_MIN} to {CLIP_SECONDS_MAX}")
    slots = image_nodes(graph)
    if name == "ltx-25-first-last-frame" and len(images) != 2:
        raise Refused("the first-last-frame clip takes two images: the first and the last")
    if len(images) > len(slots):
        raise Refused(f"this workflow takes at most {len(slots)} image(s), got {len(images)}")
    if images:
        for key, data in zip(slots, images):
            graph[key]["inputs"]["image"] = upload(data)
    else:
        # From the prompt alone: the loader still has to name a file ComfyUI has,
        # even where the switch passes it by.
        clip_node(graph, "PrimitiveBoolean", "Switch to Text to Video?")["inputs"]["value"] = True
        pillow()
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, "PNG")
        graph[slots[0]]["inputs"]["image"] = upload(buf.getvalue())
    clip_node(graph, "PrimitiveStringMultiline", "Prompt")["inputs"]["value"] = prompt
    if not talks:                                 # a voice is as long as it is
        clip_node(graph, "PrimitiveInt", "Duration")["inputs"]["value"] = seconds
    w, h = clip_size(size, images[0] if images else None)
    out = (w, h)
    if fast and first_stage(graph):
        out = (w // 2, h // 2)                    # what the first stage samples at
    elif fast:                                    # one stage only: fewer pixels instead
        w, h = out = clip_size(f"{w}x{h}", pixels=CLIP_PIXELS * FAST_SHARE)
    clip_node(graph, "PrimitiveInt", "Width")["inputs"]["value"] = w
    clip_node(graph, "PrimitiveInt", "Height")["inputs"]["value"] = h
    set_seed(graph, seed)
    for node in graph.values():
        if node["class_type"] == "SaveVideo":     # no temp node for videos: output/llmctl-api/
            node["inputs"]["filename_prefix"] = "llmctl-api/clip"
    return out


@app.post("/v1/video/clips")
def clips():
    def go():
        if request.files or request.form:
            f = request.form
            images = [x.read() for key in ("image", "image[]") for x in request.files.getlist(key)]
            prompt, model, size, seconds = f.get("prompt", ""), f.get("model"), f.get("size"), f.get("seconds")
            quality, seed = f.get("quality"), f.get("seed")
            audio = request.files["audio"].read() if "audio" in request.files else None
        else:
            j = request.get_json(silent=True) or {}
            raw = j.get("images") or j.get("image") or []
            images = [data_url(x) for x in (raw if isinstance(raw, list) else [raw])]
            prompt, model, size, seconds = j.get("prompt", ""), j.get("model"), j.get("size"), j.get("seconds")
            quality, seed = j.get("quality"), j.get("seed")
            audio = data_url(j["audio"]) if j.get("audio") else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise Refused("prompt is required: what happens in the clip")
        if quality not in (None, "", "fast", "full"):
            raise Refused("quality is fast or full")
        known = clip_workflows()
        name = model or ("ltx-25-talking" if audio is not None else
                         "ltx-25-first-last-frame" if len(images) == 2 else "ltx-25-video")
        if name not in known:
            raise Refused(f"no video model '{name}' — there are: {', '.join(known) or 'none'}")
        graph = json.loads(known[name].read_text())
        lacking = missing_models(graph)
        if lacking:
            raise Refused(f"'{name}' needs models ComfyUI does not have: {', '.join(lacking)} "
                          f"— llmctl download comfy \"{known[name].stem}\"")
        seed = random.randrange(2**48) if seed in (None, "") else int(seed)
        t0 = time.time()
        w, h = set_clip(graph, name, prompt.strip(), images, seconds, size, quality == "fast", seed, audio)
        made = [v for out in execute(graph).values() for v in out.get("images", []) + out.get("videos", [])
                if str(v.get("filename", "")).endswith((".mp4", ".webm", ".mkv"))]
        if not made:
            raise RuntimeError("the workflow gave no video")
        q = {"filename": made[0]["filename"], "subfolder": made[0].get("subfolder", ""), "type": made[0].get("type", "output")}
        v = requests.get(f"{ARGS.comfy}/view", params=q, timeout=300)
        v.raise_for_status()
        app.logger.info("clip %s: %dx%d, %s s in %.1f s, seed %d", name, w, h, seconds or "voice-long",
                        time.time() - t0, seed)
        return Response(v.content, mimetype="video/mp4",
                        headers={"X-Seed": str(seed), "X-Model": name, "X-Size": f"{w}x{h}"})
    return handle(go)


@app.get("/v1/models")
def models():
    def go():
        data = {}
        for (name, kind), path in workflows().items():
            graph = json.loads(path.read_text())
            if missing_models(graph):
                continue
            m = data.setdefault(name, {"id": name, "object": "model", "owned_by": "comfyui",
                                       "endpoints": [], "parameters": []})
            m["endpoints"].append(f"/v1/images/{kind}")
            m["parameters"] += [x for x in extras(graph) if x not in m["parameters"]]
        for name, path in music_workflows().items():
            if not missing_models(json.loads(path.read_text())):
                data[name] = {"id": name, "object": "model", "owned_by": "comfyui",
                              "endpoints": ["/v1/audio/music"], "parameters": []}
        for name, path in clip_workflows().items():
            if not missing_models(json.loads(path.read_text())):
                data[name] = {"id": name, "object": "model", "owned_by": "comfyui",
                              "endpoints": ["/v1/video/clips"], "parameters": ["seconds", "quality"]}
        return jsonify({"object": "list", "data": list(data.values())})
    return handle(go)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--comfy", required=True, help="ComfyUI's URL, e.g. http://127.0.0.1:8009")
    p.add_argument("--public-comfy", help="ComfyUI's URL as a client reaches it (for response_format=url)")
    p.add_argument("--api", required=True, help="directory of the API-format workflows")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    # FLUX.2 dev editing with two references takes ~20 min here.
    p.add_argument("--timeout", type=int, default=3600, help="seconds one image may take")
    p.add_argument("--vision", help="URL of an OpenAI chat server with vision, for /v1/images/check")
    p.add_argument("--vision-slot", type=int, help="llmctl slot of the vision model, for /repair to start and stop")
    p.add_argument("--vision-model", help="llmctl model that /repair starts on that slot")
    p.add_argument("--llmctl", help="llmctl itself, for /repair to start and stop the vision slot")
    p.parse_args(namespace=ARGS)
    ARGS.comfy = ARGS.comfy.rstrip("/")
    ARGS.public_comfy = (ARGS.public_comfy or ARGS.comfy).rstrip("/")
    ARGS.vision = (ARGS.vision or "").rstrip("/")
    print(f"Image API on http://{ARGS.host}:{ARGS.port}/v1/images/* for ComfyUI at {ARGS.comfy}; "
          f"{len(workflows())} workflows in {ARGS.api}", flush=True)
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)


if __name__ == "__main__":
    main()
