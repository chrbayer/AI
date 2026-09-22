#!/usr/bin/env python3
"""OpenAI's image API in front of ComfyUI — for llmctl's comfyui slots.

    POST /v1/images/generations   {"model", "prompt", "size", "n", "seed", "response_format"}
    POST /v1/images/edits         multipart: image (or image[]), prompt, model, n, seed
                                  or JSON: "images": ["data:image/png;base64,...", ...]
    GET  /v1/models               the workflows whose models ComfyUI has

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
import json
import random
import re
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, request

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


class Refused(Exception):
    """A request this server cannot fulfil; the message goes back as a 400."""


def error(message, status=400, kind="invalid_request_error"):
    return jsonify({"error": {"message": message, "type": kind}}), status


# ── the workflows ────────────────────────────────────────────

def slug(stem):
    """'FLUX.2 klein 9B T2I (bf16, 4 Schritte)' -> ('flux2-klein-9b', 'generations')."""
    name = re.sub(r"\(.*?\)", "", stem)
    kind = "edits" if re.search(r"\b(Edit|Upscale)\b", name) else "generations"
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


def upload(data):
    name = hashlib.sha256(data).hexdigest()[:24] + ".png"
    r = requests.post(f"{ARGS.comfy}/upload/image",
                      files={"image": (name, data)},
                      data={"subfolder": "llmctl-api", "type": "input", "overwrite": "true"},
                      timeout=60)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def prepare(graph, prompt, size, images):
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
    set_size(graph, size)
    if images is not None:
        slots = image_nodes(graph)
        if not images:
            raise Refused("an edit needs at least one image")
        if len(images) > len(slots):
            raise Refused(f"this workflow takes at most {len(slots)} image(s), got {len(images)}")
        for key, data in zip(slots, images):
            graph[key]["inputs"]["image"] = upload(data)
        prune(graph, slots[len(images):])
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
        requests.post(f"{ARGS.comfy}/queue", json={"delete": [pid]}, timeout=10)
        raise TimeoutError(f"no result within {ARGS.timeout} s")
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


def respond(name, kind, prompt, size, n, seed, fmt, images=None):
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
        prepare(graph, prompt, size, images)
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
                       j.get("response_format") or "b64_json")
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
        else:
            j = request.get_json(silent=True) or {}
            raw = j.get("images") or j.get("image") or []
            images = [data_url(x) for x in (raw if isinstance(raw, list) else [raw])]
            prompt, model, size = j.get("prompt", ""), j.get("model"), j.get("size")
            n, seed, fmt = int(j.get("n") or 1), j.get("seed"), j.get("response_format") or "b64_json"
        return respond(model or default_model("edits"), "edits", prompt, size, n,
                       seed if seed in (None, "") else int(seed), fmt, images)
    return handle(go)


@app.get("/v1/models")
def models():
    def go():
        data = {}
        for (name, kind), path in workflows().items():
            if missing_models(json.loads(path.read_text())):
                continue
            m = data.setdefault(name, {"id": name, "object": "model", "owned_by": "comfyui",
                                       "endpoints": []})
            m["endpoints"].append(f"/v1/images/{kind}")
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
    p.add_argument("--timeout", type=int, default=900, help="seconds one image may take")
    p.parse_args(namespace=ARGS)
    ARGS.comfy = ARGS.comfy.rstrip("/")
    ARGS.public_comfy = (ARGS.public_comfy or ARGS.comfy).rstrip("/")
    print(f"Image API on http://{ARGS.host}:{ARGS.port}/v1/images/* for ComfyUI at {ARGS.comfy}; "
          f"{len(workflows())} workflows in {ARGS.api}", flush=True)
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)


if __name__ == "__main__":
    main()
