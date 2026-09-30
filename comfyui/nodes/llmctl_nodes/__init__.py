"""llmctl's own ComfyUI nodes, linked into custom_nodes/ at every start.

Artifact Check (llmctl) asks the image API's /v1/images/check — a vision
model, flash by default — for the flaws in a picture. It gives the picture
with numbered boxes, a mask over them, a prompt to repaint them with and a
report, and it leaves the picture in input/ with the boxes transparent: load
that in the Inpaint workflow and the boxes are its mask, to edit in the
MaskEditor. The image API is the slot's --proxy; llmctl hands its address in
as LLMCTL_IMAGES_API when it starts ComfyUI with --proxy.
"""
import base64
import hashlib
import io
import os
import time

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths

VISION_CHOICES = ["leave as is", "start if needed, stop after"]


def to_pil(image):
    """The first picture of an IMAGE batch."""
    return Image.fromarray((image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))


def to_image(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0)[None]


def grown(box, margin, size):
    w, h = size
    x1, y1, x2, y2 = box
    return max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin), min(h, y2 + margin)


def mask_of(boxes, margin, size):
    """1 inside the boxes (grown by margin), 0 elsewhere — ComfyUI's MASK."""
    w, h = size
    m = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in grown(box, margin, size))
        m[y1:y2, x1:x2] = 1.0
    return m


def preview(pil, artifacts):
    im = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    width = max(3, im.width // 300)
    font = ImageFont.load_default(max(20, im.width // 30))
    for a in artifacts:
        x1, y1, x2, y2 = a["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 59, 48), width=width)
        draw.text((x1 + 2 * width, y1 + width), str(a["id"]), fill=(255, 59, 48), font=font)
    return im


def repair_prompt(prompt, artifacts):
    fixes = list(dict.fromkeys(a["fix"].strip().rstrip(".").strip() for a in artifacts
                               if a.get("fix", "").strip(" .")))
    return ". ".join(p for p in (prompt.strip().rstrip("."), "; ".join(fixes)) if p)


class ImageApi:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def get(self, path):
        try:
            r = requests.get(self.url + path, timeout=10)
        except requests.ConnectionError:
            raise RuntimeError(f"the image API does not answer at {self.url} — "
                               "start ComfyUI with llmctl start comfy N --proxy --vision SLOT") from None
        r.raise_for_status()
        return r.json()

    def post(self, path, body, timeout=60):
        try:
            r = requests.post(self.url + path, json=body, timeout=timeout)
        except requests.ConnectionError:
            raise RuntimeError(f"the image API does not answer at {self.url} — "
                               "start ComfyUI with llmctl start comfy N --proxy --vision SLOT") from None
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200:
            raise RuntimeError((j.get("error") or {}).get("message") or r.text[:300])
        return j

    def vision_state(self):
        return self.get("/repair/status")["vision"]

    def vision(self, action, want, limit=900):
        self.post("/repair/vision", {"action": action})
        deadline = time.time() + limit
        while time.time() < deadline:
            v = self.vision_state()
            if v["state"] == want:
                return
            if v["state"] not in ("starting", "stopping") and v.get("message"):
                raise RuntimeError(v["message"])
            time.sleep(2)
        raise RuntimeError(f"the vision model did not {action} within {limit} s")


def free_comfy_memory():
    """Beside flash, ComfyUI's loaded models do not fit: let go of them first."""
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()


class LlmctlArtifactCheck:
    CATEGORY = "llmctl"
    FUNCTION = "check"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("preview", "image", "mask", "repair_prompt", "report")
    DESCRIPTION = ("Asks a vision model (the image API's /v1/images/check) for the flaws in a picture. "
                   "Leaves the picture in input/ with the flaws transparent — load it in the Inpaint "
                   "workflow, where they are the mask to edit in the MaskEditor.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What the picture was made from — helps the check."}),
                "max_area": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
                                       "tooltip": "Larger flaws are reported, not boxed: repainting that much makes new ones."}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 256,
                                   "tooltip": "Pixels each box grows by in the mask."}),
                "vision": (VISION_CHOICES, {"default": VISION_CHOICES[0],
                                            "tooltip": "Start the vision model when it is off and stop it after the check, "
                                                       "so the Inpaint workflow has the memory."}),
            },
            "optional": {
                "api_url": ("STRING", {"default": "",
                                       "tooltip": "The image API; empty: the one llmctl started beside this ComfyUI (--proxy)."}),
            },
        }

    def check(self, image, prompt, max_area, margin, vision, api_url=""):
        url = api_url.strip() or os.environ.get("LLMCTL_IMAGES_API", "")
        if not url:
            raise RuntimeError("no image API: start ComfyUI with llmctl start comfy N --proxy --vision SLOT, or set api_url")
        api = ImageApi(url)
        manage = vision == VISION_CHOICES[1]
        started = False
        if manage and api.vision_state()["state"] != "up":
            free_comfy_memory()
            api.vision("start", "up")
            started = True
        try:
            pil = to_pil(image)
            buf = io.BytesIO()
            pil.save(buf, "PNG")
            data = buf.getvalue()
            c = api.post("/v1/images/check", {"image": "data:image/png;base64," + base64.b64encode(data).decode(),
                                              "prompt": prompt.strip() or None, "max_area": max_area}, timeout=3600)
        finally:
            if manage and (started or api.vision_state()["state"] == "up"):
                api.vision("stop", "down")
        artifacts = c["artifacts"]
        m = mask_of([a["box"] for a in artifacts], margin, pil.size)
        # The picture with the flaws transparent: LoadImage makes that its mask.
        rgba = pil.convert("RGBA")
        rgba.putalpha(Image.fromarray(((1.0 - m) * 255).astype(np.uint8)))
        name = f"artifacts_{hashlib.sha256(data).hexdigest()[:10]}.png"
        rgba.save(os.path.join(folder_paths.get_input_directory(), name))
        lines = [f"{a['id']}: {a['what']} → {a['fix']}" for a in artifacts] or ["no flaws found"]
        lines += [f"remark: {r['what']}" for r in c.get("remarks", [])]
        lines += ["", f"checked in {c['seconds']} s; in input/ as {name} — load it in "
                      "Qwen-Image 2.1 Inpaint (Turbo), edit the mask in the MaskEditor, use repair_prompt"]
        report = "\n".join(lines)
        return {"ui": {"text": [report]},
                "result": (to_image(preview(pil, artifacts)), image, torch.from_numpy(m)[None],
                           repair_prompt(prompt, artifacts), report)}


NODE_CLASS_MAPPINGS = {"LlmctlArtifactCheck": LlmctlArtifactCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"LlmctlArtifactCheck": "Artifact Check (llmctl)"}
