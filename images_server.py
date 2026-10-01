#!/usr/bin/env python3
"""OpenAI's image API in front of ComfyUI — for llmctl's comfyui slots.

    POST /v1/images/generations   {"model", "prompt", "size", "n", "seed", "response_format"}
                                  extra: "negative_prompt" (workflows with real guidance)
    POST /v1/images/edits         multipart: image (or image[]), mask, prompt, model, n, seed
                                  or JSON: "images": ["data:image/png;base64,...", ...], "mask"
                                  extra: "strength" (ControlNet), "pad" (Outpaint),
                                  "boxes" + "margin" (Inpaint: a mask made of rectangles)
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
    kind = "edits" if re.search(r"\b(Edit|Upscale|Control|Canny|Inpaint|Outpaint|Removal|Detailer|Colorize)\b", name) else "generations"
    name = re.sub(r"\b(T2I|Edit)\b", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name.lower().replace(".", "")).strip("-")
    return name, kind


def workflows():
    """{(name, kind): path} for every API workflow bundled."""
    out = {}
    for path in sorted(Path(ARGS.api).glob("*.json")):
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
                        if key in MODEL_INPUTS and spec and isinstance(spec[0], list):
                            files.setdefault(key, set()).update(spec[0])
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
                 and "Negative" not in title(n) and isinstance(n["inputs"].get("text"), str)]
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
        out += ["mask", "boxes"]
    kinds = {n["class_type"] for n in graph.values()}
    if "ZImageFunControlnet" in kinds:
        out.append("strength")
    if "ImagePadForOutpaint" in kinds:
        out.append("pad")
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


def upload(data):
    name = hashlib.sha256(data).hexdigest()[:24] + ".png"
    r = requests.post(f"{ARGS.comfy}/upload/image",
                      files={"image": (name, data)},
                      data={"subfolder": "llmctl-api", "type": "input", "overwrite": "true"},
                      timeout=60)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def prepare(graph, prompt, size, images, mask=None, strength=None, pad=None, boxes=None, margin=None,
            negative_prompt=None):
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
        if mask is not None and boxes is not None:
            raise Refused("a mask or boxes, not both")
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
    if not any(n["class_type"] == "PreviewImage" for n in graph.values()):
        raise Refused("with these images the workflow has nothing left to produce")


# ── running it ───────────────────────────────────────────────

def run(graph):
    """Queue the workflow, wait for it, return its images as (bytes, view URL)."""
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
    out = []
    for node in h["outputs"].values():
        for img in node.get("images", []):
            q = {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                 "type": img.get("type", "temp")}
            v = requests.get(f"{ARGS.comfy}/view", params=q, timeout=60)
            v.raise_for_status()
            out.append((v.content, requests.Request("GET", f"{ARGS.public_comfy}/view",
                                                    params=q).prepare().url))
    return out


def respond(name, kind, prompt, size, n, seed, fmt, images=None, **extra):
    if fmt not in ("b64_json", "url"):
        raise Refused("response_format is b64_json or url")
    if not 1 <= n <= 8:
        raise Refused("n is 1 to 8")
    template = load(name, kind)
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
            boxes, margin = f.get("boxes") or None, f.get("margin")
        else:
            j = request.get_json(silent=True) or {}
            raw = j.get("images") or j.get("image") or []
            images = [data_url(x) for x in (raw if isinstance(raw, list) else [raw])]
            prompt, model, size = j.get("prompt", ""), j.get("model"), j.get("size")
            n, seed, fmt = int(j.get("n") or 1), j.get("seed"), j.get("response_format") or "b64_json"
            mask = data_url(j["mask"]) if j.get("mask") else None
            strength, pad = j.get("strength"), j.get("pad")
            negative = j.get("negative_prompt")
            boxes, margin = j.get("boxes"), j.get("margin")
        return respond(model or default_model("edits"), "edits", prompt, size, n,
                       seed if seed in (None, "") else int(seed), fmt, images,
                       mask=mask, strength=strength, pad=pad, boxes=boxes, margin=margin,
                       negative_prompt=negative)
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
