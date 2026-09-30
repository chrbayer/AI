#!/usr/bin/env python3
"""Make the Qwen-Image 2.1 ControlNet workflows from their API graphs.

    comfyui/build_control_workflows.py [PORT]      (a running ComfyUI, default 8009)

alibaba-pai's Qwen-Image-2.1-Fun-Controlnet-Union steers Qwen-Image 2.1 with a
control image (canny, depth, pose, scribble, lineart, …, one model for all) or
repaints a masked region. ComfyUI loads it as a model patch ("Load Model Patch"
+ "Apply Fun ControlNet"); there is no official template for 2.1 yet. The
graphs are written here in API format, which is short and exact, and ComfyUI's
own frontend turns them into UI workflows (app.loadApiJson, then
graph.serialize), which get the model download entries llmctl reads. Run it
again after changing a graph, then comfyui/export_api.py.
"""
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8009
CDP_PORT = 9334

HF = "https://huggingface.co"
MODELS = {   # file -> (directory, url), for the loader nodes' download entries
    "qwen_image_2.1_bf16.safetensors": ("diffusion_models",
        f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_bf16.safetensors"),
    "qwen3vl_8b_bf16.safetensors": ("text_encoders",
        f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors"),
    "qwen_image_2.1_vae_bf16.safetensors": ("vae",
        f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"),
    "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors": ("model_patches",
        f"{HF}/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/"
        "8a4702014d4dabb5f896fcba917e2ee0a961465f/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"),
}
LOADER_INPUT = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name",
                "ModelPatchLoader": "name"}


def base(prompt, prefix):
    """Qwen-Image 2.1 as the bundled T2I workflow runs it, with the ControlNet
    patched into the model and the latent sized from image 5 (scaled to ~1 MP)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "ModelPatchLoader", "inputs": {"name": "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"}},
        "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
        "7": {"class_type": "GetImageSize", "inputs": {"image": ["6", 0]}},
        "8": {"class_type": "ZImageFunControlnet", "inputs": {"model": ["1", 0], "model_patch": ["4", 0], "vae": ["3", 0], "strength": 1.0}},
        "9": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": prompt, "negative_prompt": "", "resolution": 1024, "clip": ["2", 0]}},
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": ["7", 0], "height": ["7", 1], "batch_size": 1}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["8", 0], "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["10", 0],
                                                    "seed": 42, "control_after_generate": "randomize", "steps": 14, "cfg": 1.0,
                                                    "sampler_name": "dpmpp_2m", "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def control():
    """A ready control image of any kind the union model knows."""
    g = base("A knight in ornate silver armour standing in a misty forest clearing, cinematic light, "
             "photorealistic", "Qwen_image_2.1_control")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "control.png"}, "_meta": {"title": "Control image (pose, depth, scribble, lineart, canny …)"}}
    g["8"]["inputs"]["image"] = ["6", 0]
    # At 1.0 a scribble comes back as flat vector art whatever the prompt says;
    # at 0.5 the prompt's photograph, on the scribble's layout. More for a pose
    # or depth map that has to be followed closely.
    g["8"]["inputs"]["strength"] = 0.5
    return g


def canny():
    """Edges of a photo as the control: a new picture on the old one's layout."""
    g = base("The same scene as a watercolour painting, soft washes, visible paper texture", "Qwen_image_2.1_canny")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "photo.png"}, "_meta": {"title": "Photo whose layout to keep"}}
    g["14"] = {"class_type": "Canny", "inputs": {"image": ["6", 0], "low_threshold": 0.4, "high_threshold": 0.8}}
    g["15"] = {"class_type": "PreviewImage", "inputs": {"images": ["14", 0]}, "_meta": {"title": "Edges"}}
    g["8"]["inputs"]["image"] = ["14", 0]
    return g


def inpaint():
    """Repaint where the mask is (drawn in the mask editor, or transparent in
    the PNG); the prompt describes the whole picture."""
    g = base("A red vintage bicycle leaning against the wall, the rest of the street unchanged", "Qwen_image_2.1_inpaint")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "inpaint.png"}, "_meta": {"title": "Image — paint the mask over what to redraw"}}
    g["16"] = {"class_type": "MaskToImage", "inputs": {"mask": ["5", 1]}}
    g["17"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["16", 0], "upscale_method": "bilinear", "megapixels": 1.0, "resolution_steps": 16}}
    g["18"] = {"class_type": "ImageToMask", "inputs": {"image": ["17", 0], "channel": "red"}}
    g["8"]["inputs"].update({"inpaint_image": ["6", 0], "mask": ["18", 0]})
    return g


WORKFLOWS = {
    "Qwen-Image 2.1 Control (bf16, dpmpp_2m 14)": control,
    "Qwen-Image 2.1 Canny Control (bf16, dpmpp_2m 14)": canny,
    "Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)": inpaint,
}


async def cdp(ws, method, params=None, _id=[0]):
    _id[0] += 1
    await ws.send(json.dumps({"id": _id[0], "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _id[0]:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["result"]


async def evaluate(ws, expr):
    r = await cdp(ws, "Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
    if "exceptionDetails" in r:
        raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r))
    return r["result"].get("value")


def with_downloads(wf):
    """The download entry each loader node carries — what llmctl's download reads."""
    for n in wf["nodes"]:
        key = LOADER_INPUT.get(n.get("type"))
        if not key:
            continue
        name = (n.get("widgets_values") or [None])[0]
        if name in MODELS:
            directory, url = MODELS[name]
            n.setdefault("properties", {})["models"] = [{"name": name, "url": url, "directory": directory}]
    return wf


async def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or sys.exit("no Chrome")
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
                             f"--user-data-dir={profile}", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            sys.exit("Chrome did not come up")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            await cdp(ws, "Page.navigate", {"url": f"http://127.0.0.1:{PORT}/"})
            for _ in range(120):
                if await evaluate(ws, "!!(window.app && window.app.graph && window.app.loadApiJson)"):
                    break
                await asyncio.sleep(0.5)
            else:
                sys.exit("the ComfyUI frontend did not load")
            await asyncio.sleep(2)
            for name, make in WORKFLOWS.items():
                api = make()
                wf = await evaluate(ws, f"""(async () => {{
                    await app.loadApiJson({json.dumps(api)}, {json.dumps(name)});
                    return app.graph.serialize();
                }})()""")
                out = ROOT / "workflows" / f"{name}.json"
                out.write_text(json.dumps(with_downloads(wf), indent=1, ensure_ascii=False) + "\n")
                print(f"{len(wf['nodes']):3d} nodes  {out.relative_to(ROOT.parent)}")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


asyncio.run(main())
