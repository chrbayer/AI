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
    "Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors": ("loras",
        f"{HF}/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs/resolve/"
        "f7545234760e1847cd8e89e52bd951cb0b7e327f/models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors"),
    "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors": ("model_patches",
        f"{HF}/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/"
        "8a4702014d4dabb5f896fcba917e2ee0a961465f/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"),
    # Detailer: YOLO face and hand detectors (Bingsu/adetailer, Apache-2.0) and
    # SAM 1 ViT-H (Meta, Apache-2.0; a Hugging Face mirror, pinned).
    "bbox/face_yolov8m.pt": ("ultralytics/bbox",
        f"{HF}/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov8m.pt"),
    "bbox/hand_yolov8s.pt": ("ultralytics/bbox",
        f"{HF}/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/hand_yolov8s.pt"),
    "sam_vit_h_4b8939.pth": ("sams",
        f"{HF}/ybelkada/segment-anything/resolve/7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"),
}
LOADER_INPUT = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name",
                "ModelPatchLoader": "name", "LoraLoaderModelOnly": "lora_name",
                "UltralyticsDetectorProvider": "model_name", "SAMLoader": "model_name"}


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


def pose():
    """The body pose of a person in a photo (DWPose, from comfyui_controlnet_aux)
    as the control: someone else, somewhere else, standing the same way."""
    g = base("A dancer in a flowing red dress on a theatre stage, spotlight, photograph", "Qwen_image_2.1_pose")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "pose.png"}, "_meta": {"title": "Photo with the pose to take over"}}
    # torchscript, not onnx: runs through torch on the GPU; onnxruntime here is CPU only.
    g["14"] = {"class_type": "DWPreprocessor", "inputs": {"image": ["6", 0], "detect_hand": "enable", "detect_body": "enable",
               "detect_face": "enable", "resolution": 1024, "bbox_detector": "yolox_l.torchscript.pt",
               "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable"}}
    g["15"] = {"class_type": "PreviewImage", "inputs": {"images": ["14", 0]}, "_meta": {"title": "Pose"}}
    g["8"]["inputs"]["image"] = ["14", 0]
    return g


def depth():
    """The spatial layout of a photo (Depth Anything V2 Large, from
    comfyui_controlnet_aux; CC-BY-NC) as the control: the same room or
    landscape, freely re-made in its materials and style."""
    g = base("The same room as a cosy wooden alpine cabin interior, warm evening light, photograph", "Qwen_image_2.1_depth")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "room.png"}, "_meta": {"title": "Photo whose space to keep"}}
    g["14"] = {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["6", 0], "ckpt_name": "depth_anything_v2_vitl.pth",
               "resolution": 1024}}
    g["15"] = {"class_type": "PreviewImage", "inputs": {"images": ["14", 0]}, "_meta": {"title": "Depth"}}
    g["8"]["inputs"]["image"] = ["14", 0]
    return g


def colorize():
    """A black-and-white photo in colour: its luminance (comfyui_controlnet_aux's
    Image Luminance) is the control, so every detail stays; the colours come
    from the prompt."""
    g = base("The same photograph in natural colour, realistic skin tones, colour film", "Qwen_image_2.1_colorize")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "bw.png"}, "_meta": {"title": "Black-and-white photo"}}
    g["14"] = {"class_type": "ImageLuminanceDetector", "inputs": {"image": ["6", 0], "gamma_correction": 1.0, "resolution": 1024}}
    g["15"] = {"class_type": "PreviewImage", "inputs": {"images": ["14", 0]}, "_meta": {"title": "Luminance"}}
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


def outpaint():
    """Extend a picture beyond its edges: the canvas padded (ImagePadForOutpaint,
    256 px left and right by default, a 40 px feathered seam), the new border
    inpainted. The picture is scaled to ~0.8 MP first, in steps of 16, so the
    padded canvas stays a multiple of 16 and ~1.3 MP."""
    g = base("A wide panoramic photograph of the whole scene, continuing naturally beyond the frame", "Qwen_image_2.1_outpaint")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "outpaint.png"}, "_meta": {"title": "Picture to extend"}}
    g["6"]["inputs"]["megapixels"] = 0.8
    g["21"] = {"class_type": "ImagePadForOutpaint", "inputs": {"image": ["6", 0], "left": 256, "top": 0, "right": 256,
               "bottom": 0, "feathering": 40}, "_meta": {"title": "New border (px per side)"}}
    g["7"]["inputs"]["image"] = ["21", 0]
    g["8"]["inputs"].update({"inpaint_image": ["21", 0], "mask": ["21", 1]})
    return g


def turbo(make):
    """The same graph with the 4-step Acc LoRA (converted for ComfyUI by
    download comfy) between the model and the ControlNet, and 4 euler steps:
    ~25-38 s instead of 80-95, control and quality held (tried on all five)."""
    def made():
        g = make()
        g["20"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0,
                   "lora_name": "Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors"}, "_meta": {"title": "4-step Acc LoRA (alibaba-pai)"}}
        g["8"]["inputs"]["model"] = ["20", 0]
        g["11"]["inputs"].update(steps=4, sampler_name="euler")
        g["13"]["inputs"]["filename_prefix"] += "_turbo"
        return g
    return made


FACE_PROMPT = "a natural, detailed human face with clear eyes, nose and mouth"
HAND_PROMPT = "a natural human hand with five well-formed fingers"


def detailer(sam=True):
    """Faces, then hands: each found one is cropped, repainted at up to 1024 px
    with Qwen-Image 2.1 and pasted back; SAM outlines it for the mask. Faces at
    denoise 0.45, which keeps the person (0.6 turned brown eyes blue-green);
    hands at 0.6, which mends their shape (0.45 only added detail). Tried on
    seven pictures; the 4-step LoRA made skin coarse and aged, so no Turbo."""
    def one(image, detector, cond, denoise):
        return {"class_type": "FaceDetailer", "inputs": {
            "image": image, "model": ["1", 0], "clip": ["2", 0], "vae": ["3", 0],
            "guide_size": 1024, "guide_size_for": True, "max_size": 1024, "seed": 42,
            "steps": 14, "cfg": 1.0, "sampler_name": "dpmpp_2m", "scheduler": "simple",
            "positive": [cond, 0], "negative": [cond, 1], "denoise": denoise, "feather": 5,
            "noise_mask": True, "force_inpaint": True, "bbox_threshold": 0.5, "bbox_dilation": 10,
            "bbox_crop_factor": 3.0, "sam_detection_hint": "center-1", "sam_dilation": 0,
            "sam_threshold": 0.93, "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False", "drop_size": 10, "bbox_detector": [detector, 0],
            "wildcard": "", "cycle": 1, "inpaint_model": False, "noise_mask_feather": 20,
            "tiled_encode": False, "tiled_decode": False,
            **({"sam_model_opt": ["32", 0]} if sam else {})}}
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "detail.png"}, "_meta": {"title": "Bild"}},
        "9": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": FACE_PROMPT, "negative_prompt": "", "resolution": 1024, "clip": ["2", 0]},
              "_meta": {"title": "Gesicht (fixed)"}},
        "10": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": HAND_PROMPT, "negative_prompt": "", "resolution": 1024, "clip": ["2", 0]},
               "_meta": {"title": "Hand (fixed)"}},
        "30": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}, "_meta": {"title": "Gesichter finden"}},
        "31": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/hand_yolov8s.pt"}, "_meta": {"title": "Hände finden"}},
        "40": {**one(["5", 0], "30", "9", 0.45), "_meta": {"title": "Gesichter nachbessern"}},
        "41": {**one(["40", 0], "31", "10", 0.6), "_meta": {"title": "Hände nachbessern"}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["41", 0], "filename_prefix": "Qwen_image_2.1_detailer"}},
    }
    if sam:
        g["32"] = {"class_type": "SAMLoader", "inputs": {"model_name": "sam_vit_h_4b8939.pth", "device_mode": "AUTO"}, "_meta": {"title": "SAM ViT-H"}}
    return g


def artifact_check():
    """A vision model looks for flaws; the picture lands in input/ with them transparent."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "check.png"}, "_meta": {"title": "Bild"}},
        "2": {"class_type": "LlmctlArtifactCheck", "_meta": {"title": "Artifact Check (llmctl)"},
              "inputs": {"image": ["1", 0], "prompt": "", "max_area": 0.2, "margin": 16,
                         "vision": "leave as is", "api_url": ""}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}, "_meta": {"title": "Fundstellen"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 4]}, "_meta": {"title": "Bericht"}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["2", 3]}, "_meta": {"title": "Prompt für die Reparatur"}},
    }


WORKFLOWS = {
    "Qwen-Image 2.1 Control (bf16, dpmpp_2m 14)": control,
    "Qwen-Image 2.1 Canny Control (bf16, dpmpp_2m 14)": canny,
    "Qwen-Image 2.1 Pose Control (bf16, dpmpp_2m 14)": pose,
    "Qwen-Image 2.1 Depth Control (bf16, dpmpp_2m 14)": depth,
    "Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)": inpaint,
    "Qwen-Image 2.1 Colorize (bf16, dpmpp_2m 14)": colorize,
    "Qwen-Image 2.1 Outpaint (bf16, dpmpp_2m 14)": outpaint,
    "Qwen-Image 2.1 Control Turbo (bf16, 4 Schritte)": turbo(control),
    "Qwen-Image 2.1 Canny Control Turbo (bf16, 4 Schritte)": turbo(canny),
    "Qwen-Image 2.1 Pose Control Turbo (bf16, 4 Schritte)": turbo(pose),
    "Qwen-Image 2.1 Depth Control Turbo (bf16, 4 Schritte)": turbo(depth),
    "Qwen-Image 2.1 Inpaint Turbo (bf16, 4 Schritte)": turbo(inpaint),
    "Qwen-Image 2.1 Colorize Turbo (bf16, 4 Schritte)": turbo(colorize),
    "Qwen-Image 2.1 Outpaint Turbo (bf16, 4 Schritte)": turbo(outpaint),
    "Artifact Check (flash)": artifact_check,
    "Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14)": detailer,
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
            # "bbox/face_yolov8m.pt" lies in ultralytics/bbox as face_yolov8m.pt
            n.setdefault("properties", {})["models"] = [{"name": name.rsplit("/", 1)[-1], "url": url, "directory": directory}]
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
                wf = with_downloads(wf)
                # The frontend gives each load a new id; an unchanged graph keeps its old one.
                if out.exists():
                    old = json.loads(out.read_text())
                    if {**old, "id": None} == {**wf, "id": None}:
                        wf["id"] = old["id"]
                out.write_text(json.dumps(wf, indent=1, ensure_ascii=False) + "\n")
                print(f"{len(wf['nodes']):3d} nodes  {out.relative_to(ROOT.parent)}")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


asyncio.run(main())
