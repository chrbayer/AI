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

Variants are switches in one workflow (Turbo, Unzensiert, Steuerung, Nur in der
Maske, Ausschnitt, Feine Kanten, XL); the template-made families start from
their API graphs frozen in comfyui/bases/. variants.py says which switch
setting is which of the image API's models; export_api.py writes those.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8009
CDP_PORT = 9334

HF = "https://huggingface.co"
BIREFNET = "4d000788a9698c7f8d67c8c6ce2b40c768f5b909"
FLORENCE = "21a599d414c4d928c9032694c424fb94458e3594"
SAM3 = "ea8e153c669a0284a496c0ec65a53b8e4f5ca7e7"
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
    # Z-Image Turbo (Comfy-Org's repack, pinned) and alibaba-pai's Union 2.1 for it,
    # 2602 8-step: canny, depth, pose, mlsd, hed, scribble, gray, and inpaint.
    "z_image_turbo_bf16.safetensors": ("diffusion_models",
        f"{HF}/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    "qwen_3_4b.safetensors": ("text_encoders",
        f"{HF}/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/text_encoders/qwen_3_4b.safetensors"),
    "ae.safetensors": ("vae",
        f"{HF}/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/vae/ae.safetensors"),
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors": ("model_patches",
        f"{HF}/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/"
        "5155fc56d17821007d6f62ac192c09e0f0e72016/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"),
    # Detailer: YOLO face and hand detectors (Bingsu/adetailer, Apache-2.0) and
    # SAM 1 ViT-H (Meta, Apache-2.0; a Hugging Face mirror, pinned).
    "bbox/face_yolov8m.pt": ("ultralytics/bbox",
        f"{HF}/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov8m.pt"),
    "bbox/hand_yolov8s.pt": ("ultralytics/bbox",
        f"{HF}/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/hand_yolov8s.pt"),
    "sam_vit_h_4b8939.pth": ("sams",
        f"{HF}/ybelkada/segment-anything/resolve/7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"),
    # Background removal (ComfyUI-RMBG's BiRefNet node; ZhengPeng7's BiRefNet, MIT,
    # in 1038lab's repack, pinned). birefnet.py, the model code, is left to the
    # node: it rewrites one import in it on every load, so it never matches the Hub.
    "BiRefNet-general": [("RMBG/BiRefNet", f"{HF}/1038lab/BiRefNet/resolve/{BIREFNET}/{f}")
                         for f in ("BiRefNet-general.safetensors", "config.json", "BiRefNet_config.py")],
    "BiRefNet-HR-matting": [("RMBG/BiRefNet", f"{HF}/1038lab/BiRefNet/resolve/{BIREFNET}/{f}")
                            for f in ("BiRefNet-HR-matting.safetensors", "config.json", "BiRefNet_config.py")],
    # Florence-2 large (Microsoft, MIT) for kijai's ComfyUI-Florence2, which reads
    # LLM/<name>/ — only what it loads: the repo holds the weights twice
    # (safetensors and .bin) and its model code, which the node brings itself.
    "microsoft/Florence-2-large": [("LLM/Florence-2-large", f"{HF}/microsoft/Florence-2-large/resolve/{FLORENCE}/{f}")
                                   for f in ("model.safetensors", "config.json", "generation_config.json",
                                             "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                                             "vocab.json")],
    # SAM 3.1 (Meta, SAM License) for ComfyUI-RMBG's SAM3Segment, from 1038lab's
    # ungated repack.
    "sam3.1_multiplex_fp16": [("sam3", f"{HF}/1038lab/sam3/resolve/{SAM3}/sam3.1_multiplex_fp16.safetensors")],
}
LOADER_INPUT = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name",
                "ModelPatchLoader": "name", "LoraLoaderModelOnly": "lora_name",
                "UltralyticsDetectorProvider": "model_name", "SAMLoader": "model_name",
                "BiRefNetRMBG": "model", "DownloadAndLoadFlorence2Model": "model", "SAM3Segment": "model_name"}


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


def zimage(make, strength=None):
    """The same control graph on Z-Image Turbo: its DiT, Qwen3-4B encoder and VAE,
    AuraFlow sampling shift 3, 8 res_multistep steps at cfg 1 (the template's),
    and alibaba-pai's Union 2.1 (2602, 8-step) as the model patch. The node that
    applies it is the one the Qwen-Image graphs use."""
    def made():
        g = make()
        prompt = g["9"]["inputs"]["prompt"]
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}}
        g["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}}
        g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}
        g["4"]["inputs"]["name"] = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
        g["21"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": g["8"]["inputs"]["model"], "shift": 3}}
        g["8"]["inputs"]["model"] = ["21", 0]
        g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}, "_meta": {"title": "Positive prompt"}}
        g["19"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["9", 0]}}
        g["10"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": ["7", 0], "height": ["7", 1], "batch_size": 1}}
        g["11"]["inputs"].update(positive=["9", 0], negative=["19", 0], steps=8, cfg=1.0,
                                 sampler_name="res_multistep", scheduler="simple")
        if strength is not None:
            g["8"]["inputs"]["strength"] = strength
        g["13"]["inputs"]["filename_prefix"] = g["13"]["inputs"]["filename_prefix"].replace("Qwen_image_2.1", "Z_image_turbo")
        return g
    return made


def masked(make, prompt, prefix):
    """A control workflow that repaints only inside the mask: the edges, pose
    or depth of the whole photo steer what is painted there (#24)."""
    def made():
        g = make()
        g["9"]["inputs"]["prompt"] = prompt
        g["13"]["inputs"]["filename_prefix"] = prefix
        g["5"] = {"class_type": "LoadImage", "inputs": {"image": "inpaint.png"},
                  "_meta": {"title": "Photo — paint the mask over what to redraw"}}
        g["16"] = {"class_type": "MaskToImage", "inputs": {"mask": ["5", 1]}}
        g["17"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["16", 0], "upscale_method": "bilinear", "megapixels": 1.0, "resolution_steps": 16}}
        g["18"] = {"class_type": "ImageToMask", "inputs": {"image": ["17", 0], "channel": "red"}}
        g["8"]["inputs"].update({"inpaint_image": ["6", 0], "mask": ["18", 0]})
        return g
    return made


canny_inpaint = masked(canny, "The same photo; the masked jacket is now bright red leather", "Qwen_image_2.1_canny_inpaint")
pose_inpaint = masked(pose, "The same photo; the person now wears a yellow rain jacket and dark trousers", "Qwen_image_2.1_pose_inpaint")
depth_inpaint = masked(depth, "The same photo; the masked part of the room is now a wooden bookshelf", "Qwen_image_2.1_depth_inpaint")


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


def crop(make):
    """Inpaint on a crop (ComfyUI-Inpaint-CropAndStitch): the masked part and some
    context around it cut out, scaled to ~1 MP, repainted there and stitched back
    with a blended seam. The rest of the picture keeps its pixels and its size;
    a small flaw gets the model's whole resolution instead of a few dozen pixels."""
    def made():
        g = make()
        g["30"] = {"class_type": "InpaintCropImproved", "inputs": {
            "image": ["5", 0], "mask": ["5", 1], "downscale_algorithm": "bilinear", "upscale_algorithm": "bicubic",
            "preresize": False, "preresize_mode": "ensure minimum resolution", "preresize_min_width": 1024,
            "preresize_min_height": 1024, "preresize_max_width": 16384, "preresize_max_height": 16384,
            "mask_fill_holes": True, "mask_expand_pixels": 0, "mask_invert": False, "mask_blend_pixels": 32,
            "mask_hipass_filter": 0.1, "extend_for_outpainting": False, "extend_up_factor": 1.0,
            "extend_down_factor": 1.0, "extend_left_factor": 1.0, "extend_right_factor": 1.0,
            "context_from_mask_extend_factor": CROP_CONTEXT, "output_resize_to_target_size": True,
            "output_target_width": 1024, "output_target_height": 1024, "output_padding": "32",
            "device_mode": "gpu (much faster)"}, "_meta": {"title": "Crop around the mask"}}
        for k in ("6", "16", "17", "18"):
            g.pop(k)
        g["7"]["inputs"]["image"] = ["30", 1]
        g["8"]["inputs"].update({"inpaint_image": ["30", 1], "mask": ["30", 2]})
        g["31"] = {"class_type": "InpaintStitchImproved", "inputs": {"stitcher": ["30", 0], "inpainted_image": ["12", 0]},
                   "_meta": {"title": "Stitch back"}}
        g["13"]["inputs"]["images"] = ["31", 0]
        g["13"]["inputs"]["filename_prefix"] += "_crop"
        return g
    return made


def background(model, prefix):
    """The subject kept, the background transparent (ComfyUI-RMBG's BiRefNet
    node): seconds, and the subject's own pixels — nothing is repainted."""
    def made():
        return {
            "5": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}, "_meta": {"title": "Photo — the subject stays, the background goes"}},
            "30": {"class_type": "BiRefNetRMBG", "inputs": {
                "image": ["5", 0], "model": model, "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                "invert_output": False, "refine_foreground": False, "unload_model": False,
                "background": "Alpha", "background_color": "#222222"}},
            "13": {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": prefix}},
        }
    return made


def caption():
    """Florence-2 describes the picture — the prompt Qwen-Image Layered wants.
    The image API runs it for a Layered edit that comes without a prompt."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}, "_meta": {"title": "Picture"}},
        "2": {"class_type": "DownloadAndLoadFlorence2Model", "inputs": {"model": "microsoft/Florence-2-large", "precision": "fp16",
              "convert_to_safetensors": False}},
        "3": {"class_type": "Florence2Run", "inputs": {"image": ["1", 0], "florence2_model": ["2", 0], "text_input": "",
              "task": "more_detailed_caption", "fill_mask": True, "keep_model_loaded": False, "max_new_tokens": 1024,
              "num_beams": 3, "do_sample": False, "output_mask_select": "", "seed": 1}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 2]}, "_meta": {"title": "Caption"}},
    }


def select():
    """SAM 3 finds what a few words name ("the sofa", "hair"), one mask per
    match — the image API's `select` and /v1/images/select."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}, "_meta": {"title": "Picture"}},
        "2": {"class_type": "SAM3Segment", "inputs": {
            "image": ["1", 0], "model_name": "sam3.1_multiplex_fp16", "prompt": "the sofa", "output_mode": "Separate",
            "confidence_threshold": 0.5, "max_segments": 0, "segment_pick": 0, "mask_blur": 0, "mask_offset": 0,
            "device": "Auto", "invert_output": False, "unload_model": True, "background": "Alpha",
            "background_color": "#222222"}, "_meta": {"title": "What to find"}},
        "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}, "_meta": {"title": "Masks"}},
    }


FACE_PROMPT = "a natural, detailed human face with clear eyes, nose and mouth"
HAND_PROMPT = "a natural human hand with five well-formed fingers"
# The crop: the mask's box, 2.5 times as wide and high. At 1.5 a face seen in
# profile came back frontal and a hand grew fingers — too little around it to
# match; at 2.5 both held, and the patch is still painted at ~1 MP.
CROP_CONTEXT = 2.5


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


# ── switches: one workflow, its variants behind them ─────────

def _new_id(g):
    return str(max(int(k) for k in g if k.isdigit()) + 1)


def _flag(g, name, value=False):
    """The PrimitiveBoolean titled `name`, made once per graph."""
    for k, n in g.items():
        if n["class_type"] == "PrimitiveBoolean" and variants.title(n) == name:
            return k
    k = _new_id(g)
    g[k] = {"class_type": "PrimitiveBoolean", "inputs": {"value": value}, "_meta": {"title": name}}
    return k


def _switch(g, flag, off, on, label):
    """An If/Else switch between two refs; whoever read `off` reads the switch."""
    k = _new_id(g)
    g[k] = {"class_type": "ComfySwitchNode", "inputs": {"switch": [flag, 0]}, "_meta": {"title": label}}
    variants._replace_refs(g, off, [k, 0])
    g[k]["inputs"].update(on_false=off, on_true=on)
    return k


def _find(g, cls, which=0):
    return [k for k, n in g.items() if n["class_type"] == cls][which]


def toggle_loader(g, name, loader, **alt):
    """`name` on: a second copy of the loader with other inputs (another file)."""
    k = _new_id(g)
    g[k] = json.loads(json.dumps(g[loader]))
    g[k]["inputs"].update(alt)
    g[k].setdefault("_meta", {})["title"] = f"{variants.title(g[loader]) or g[loader]['class_type']} ({name})"
    _switch(g, _flag(g, name), [loader, 0], [k, 0], name)
    return g


def toggle_insert(g, name, source, node):
    """`name` on: `node` (a LoRA) between `source` and whatever read it."""
    k = _new_id(g)
    g[k] = node
    _switch(g, _flag(g, name), [source, 0], [k, 0], name)
    g[k]["inputs"]["model"] = [source, 0]
    return g


def toggle_turbo(g, sampler, root, steps=4, sampler_name="euler"):
    """Turbo on: the Acc LoRA after `root` (the diffusion model), a copy of every
    node between it and `sampler`, and a second sampler with 4 euler steps."""
    flag = _flag(g, "Turbo")
    lora = next((k for k, n in g.items() if n["class_type"] == "LoraLoaderModelOnly"
                 and n["inputs"].get("lora_name") == "Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors"), None)
    if lora is None:
        lora = _new_id(g)
        g[lora] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [root, 0], "strength_model": 1.0,
                   "lora_name": "Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors"}, "_meta": {"title": "4-step Acc LoRA (alibaba-pai)"}}
    # the nodes on a path from root to the sampler's model input
    up = {}

    def reaches(k):
        if k not in up:
            up[k] = k == root or any(variants.is_ref(v) and reaches(v[0]) for v in g[k]["inputs"].values())
        return up[k]
    chain, todo = [], [g[sampler]["inputs"]["model"][0]]
    while todo:
        k = todo.pop()
        if k == root or k in chain or not reaches(k):
            continue
        chain.append(k)
        todo += [v[0] for v in g[k]["inputs"].values() if variants.is_ref(v)]
    copy = {}
    for k in chain:
        copy[k] = _new_id(g)
        g[copy[k]] = json.loads(json.dumps(g[k]))
    for k in chain:
        for a, v in g[copy[k]]["inputs"].items():
            if variants.is_ref(v) and v[0] == root:
                g[copy[k]]["inputs"][a] = [lora, 0]
            elif variants.is_ref(v) and v[0] in copy:
                g[copy[k]]["inputs"][a] = [copy[v[0]], v[1]]
    s2 = _new_id(g)
    g[s2] = json.loads(json.dumps(g[sampler]))
    m = g[sampler]["inputs"]["model"]
    g[s2]["inputs"]["model"] = [lora, 0] if m[0] == root else [copy[m[0]], m[1]]
    g[s2]["inputs"].update(steps=steps, sampler_name=sampler_name)
    g[s2]["_meta"] = {"title": "Sampler (Turbo)"}
    g[sampler].setdefault("_meta", {})["title"] = "Sampler"
    _switch(g, flag, [sampler, 0], [s2, 0], "Turbo")
    return g


def toggle_crop(g):
    """Ausschnitt on: the crop around the mask repainted and stitched back (see crop)."""
    flag = _flag(g, "Ausschnitt")
    c = crop(lambda: json.loads(json.dumps(g)))()
    # A node is shared only if it and everything it reads are unchanged.
    same = {}

    def unchanged(k):
        if k not in same:
            same[k] = k in g and c[k] == g[k] and all(
                unchanged(v[0]) for v in c[k]["inputs"].values() if variants.is_ref(v) and v[0] in c)
        return same[k]
    ren = {k: (k if unchanged(k) else None) for k in c}
    for k in c:                                     # what the crop branch changed gets new ids
        if ren[k] is None:
            ren[k] = _new_id({**g, **{v: None for v in ren.values() if v}})
    for k, n in c.items():
        if ren[k] != k:
            n = json.loads(json.dumps(n))
            for a, v in n["inputs"].items():
                if variants.is_ref(v):
                    n["inputs"][a] = [ren[v[0]], v[1]]
            g[ren[k]] = n
    save = _find(g, "SaveImage")
    del g[ren[save]]
    _switch(g, flag, g[save]["inputs"]["images"], [ren["31"], 0], "Ausschnitt")
    return g


def control_all(masked=True):
    """The control image as it is, or the edges, pose or depth of a photo — chosen
    with Steuerung; with Nur in der Maske only the masked part is repainted."""
    g = base("A knight in ornate silver armour standing in a misty forest clearing, cinematic light, "
             "photorealistic", "Qwen_image_2.1_control")
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": "control.png"},
              "_meta": {"title": "Steuerbild — oder ein Foto, aus dem Kanten, Pose oder Tiefe kommen"}}
    g["40"] = canny()["14"]
    g["41"] = pose()["14"]
    g["42"] = depth()["14"]
    # Strength 0.5 for a ready control image (a scribble at 1.0 came back as flat
    # vector art), 1.0 for the made ones.
    g["43"] = {"class_type": "LlmctlControlImage", "inputs": {
        "steuerung": "Bild", "strength_bild": 0.5, "strength_kanten": 1.0, "strength_pose": 1.0, "strength_tiefe": 1.0,
        "bild": ["6", 0], "kanten": ["40", 0], "pose": ["41", 0], "tiefe": ["42", 0]}, "_meta": {"title": "Steuerung"}}
    g["44"] = {"class_type": "PreviewImage", "inputs": {"images": ["43", 0]}, "_meta": {"title": "Steuerbild"}}
    g["8"]["inputs"].update(image=["43", 0], strength=["43", 1])
    if masked:
        g["16"] = {"class_type": "MaskToImage", "inputs": {"mask": ["5", 1]}}
        g["17"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["16", 0], "upscale_method": "bilinear", "megapixels": 1.0, "resolution_steps": 16}}
        g["18"] = {"class_type": "ImageToMask", "inputs": {"image": ["17", 0], "channel": "red"}}
        g["45"] = json.loads(json.dumps(g["8"]))
        g["45"]["inputs"].update({"inpaint_image": ["6", 0], "mask": ["18", 0]})
        _switch(g, _flag(g, "Nur in der Maske"), ["8", 0], ["45", 0], "Nur in der Maske")
    return g


def zimage_control():
    g = zimage(lambda: control_all(masked=False))()
    # Z-Image: 0.8 for a ready control (Qwen-Image's 0.5 lets it drift), canny 0.65
    # (1.0 kept the photo whatever the prompt asked, 0.5 lost the layout).
    g["43"]["inputs"].update(strength_bild=0.8, strength_kanten=0.65)
    return g


def from_base(name):
    """A frozen API graph of a ComfyUI template (comfyui/bases/), ids renumbered."""
    src = json.loads((ROOT / "bases" / f"{name}.json").read_text())
    src = {k: v for k, v in src.items() if v["class_type"] not in ("ImageCompare", "MarkdownNote", "Note")}
    ids = {k: str(i + 1) for i, k in enumerate(src)}
    g = {}
    for k, n in src.items():
        n = json.loads(json.dumps(n))
        for a, v in n["inputs"].items():
            if variants.is_ref(v) and v[0] in ids:
                n["inputs"][a] = [ids[v[0]], v[1]]
        g[ids[k]] = n
    return g


def qwen_template(name, uncensored=True):
    def made():
        g = from_base(name)
        toggle_turbo(g, _find(g, "KSampler"), _find(g, "UNETLoader"))
        if uncensored:
            toggle_loader(g, "Unzensiert", _find(g, "CLIPLoader"), clip_name="qwen3vl_8b_bf16_heretic.safetensors")
        return g
    return made


def flux_dev(name):
    def made():
        g = from_base(name)
        for n in g.values():
            if n["class_type"] == "PrimitiveBoolean" and variants.title(n).startswith("Enable") and "LoRA" in variants.title(n):
                n["_meta"]["title"] = "Turbo"
        return toggle_insert(g, "Unzensiert", _find(g, "UNETLoader"), {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "flux2_dev_female_nude_style_v1.safetensors", "strength_model": 1.0}, "_meta": {"title": "NSFW LoRA"}})
    return made


def flux_klein(name):
    def made():
        g = from_base(name)
        toggle_loader(g, "Unzensiert", _find(g, "CLIPLoader"), clip_name="qwen_3_8b_uncensored.safetensors")
        return toggle_insert(g, "Unzensiert", _find(g, "UNETLoader"), {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "flux2_klein_9b_nsfw_v2.safetensors", "strength_model": 1.0}, "_meta": {"title": "NSFW LoRA"}})
    return made


def zimage_t2i():
    g = from_base("zimage_t2i")
    return toggle_loader(g, "Unzensiert", _find(g, "UNETLoader"), unet_name="z_image_turbo_bf16_nsfw_v2.safetensors")


def ace():
    g = from_base("ace")
    return toggle_loader(g, "XL", _find(g, "UNETLoader"), unet_name="acestep_v1.5_xl_turbo_bf16.safetensors")


def birefnet():
    g = background("BiRefNet-general", "BiRefNet_background")()
    return toggle_loader(g, "Feine Kanten", "30", model="BiRefNet-HR-matting")


def with_turbo(make):
    def made():
        g = make()
        return toggle_turbo(g, "11", "1")
    return made


def inpaint_all():
    g = inpaint()
    toggle_crop(g)
    for k in [k for k, n in list(g.items()) if n["class_type"] == "KSampler"]:
        toggle_turbo(g, k, "1")
    return g


def video_finish(make):
    """Two switches before the video is put together, both off: *Hochskalieren
    (SeedVR2)* — SeedVR2 3B, 1.5×, Split Latent on (~10 min per second of video:
    for a clip to keep) — and *48 fps (FILM)*, frames in between, the rate doubled
    (~1 min). Upscaling first: the other way round SeedVR2 would have twice the
    frames. The sound goes through untouched."""
    def made():
        g = make()
        create = _find(g, "CreateVideo")
        images, fps = g[create]["inputs"]["images"], g[create]["inputs"]["fps"]
        # SeedVR2: ComfyUI's video template without its file input and output
        sv = json.loads((ROOT / "bases" / "seedvr2_video.json").read_text())
        drop = {k for k, n in sv.items() if n["class_type"] in ("LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo",
                                                                "Video Slice")}
        drop |= {k for k, n in sv.items() if n["class_type"] == "ComfySwitchNode"
                 and any(variants.is_ref(x) and x[0] in drop for x in n["inputs"].values())}
        frames_from = next(k for k, n in sv.items() if n["class_type"] == "GetVideoComponents")
        ids = {}
        for k in sv:
            if k not in drop:
                ids[k] = _new_id({**g, **{v: None for v in ids.values()}})
        for k, n in sv.items():
            if k in drop:
                continue
            n = json.loads(json.dumps(n))
            for a, x in n["inputs"].items():
                if variants.is_ref(x):
                    n["inputs"][a] = images if x[0] == frames_from else [ids[x[0]], x[1]]
            if n["class_type"] == "ResizeImageMaskNode":
                n["inputs"]["resize_type.multiplier"] = 1.5
            if n["class_type"] == "PrimitiveBoolean":                    # Split Latent: on
                n["inputs"]["value"] = True
            g[ids[k]] = n
        post = ids[next(k for k, n in sv.items() if n["class_type"] == "SeedVR2PostProcessing")]
        resize = ids[next(k for k, n in sv.items() if n["class_type"] == "ResizeImageMaskNode")]
        up = _switch(g, _flag(g, "Hochskalieren (SeedVR2)"), images, [post, 0], "Hochskalieren (SeedVR2)")
        g[resize]["inputs"]["input"] = images            # the switch took this read over too; SeedVR2 reads the frames
        # FILM
        loader, interp, rate = (str(int(_new_id(g)) + i) for i in range(3))
        g[loader] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "film_net_fp16.safetensors"}}
        g[rate] = {"class_type": "ComfyMathExpression", "inputs": {"expression": "a * 2", "values.a": fps}, "_meta": {"title": "fps × 2"}}
        flag = _flag(g, "48 fps (FILM)")
        _switch(g, flag, [up, 0], [interp, 0], "48 fps (FILM)")
        g[interp] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": [loader, 0], "images": [up, 0], "multiplier": 2}}
        k = _new_id(g)
        g[k] = {"class_type": "ComfySwitchNode", "inputs": {"switch": [flag, 0], "on_false": fps, "on_true": [rate, 0]},
                "_meta": {"title": "48 fps (FILM): rate"}}
        g[create]["inputs"]["fps"] = [k, 0]
        return g
    return made


def talking():
    """LTX-2.5 picture + audio to video (#43): the voice (or song) is encoded with
    LTX's audio VAE and held fixed by a noise mask, and the video is made to it —
    the person speaks or sings along. The audio is padded to whole seconds and
    the video made exactly that long: with a shorter audio the lips ran on after
    the voice had stopped. The audio is made exactly as long as the video — N
    frames / rate, to the sample — not just to the second. ComfyUI's ia2v
    template does this for LTX-2.3."""
    g = from_base("ltx_i2v")
    find = lambda cls, t=None: next(k for k, n in g.items() if n["class_type"] == cls and (t is None or variants.title(n) == t))
    empty = find("LTXVEmptyLatentAudio")
    audio_vae = g[empty]["inputs"]["audio_vae"]
    a, fit, enc, mask, held = (str(int(_new_id(g)) + i) for i in range(5))
    g[a] = {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}, "_meta": {"title": "Voice or song"}}
    rate = find("PrimitiveInt", "Frame Rate")
    g[fit] = {"class_type": "LlmctlAudioFit", "inputs": {"audio": [a, 0], "frame_rate": [rate, 0], "extra_seconds": 0}}
    g[enc] = {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": [fit, 0], "audio_vae": audio_vae}}
    g[mask] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}}
    g[held] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [enc, 0], "mask": [mask, 0]},
               "_meta": {"title": "Audio held as it is"}}
    variants._replace_refs(g, [empty, 0], [held, 0])
    del g[empty]
    duration = find("PrimitiveInt", "Duration")
    variants._replace_refs(g, [duration, 0], [fit, 1])
    del g[duration]
    g[find("PrimitiveStringMultiline", "Prompt")]["inputs"]["value"] = (
        "The person speaks to the camera, the lips moving clearly with every word, natural small head movements "
        "and blinks; static close-up shot.")
    g[find("SaveVideo")]["inputs"]["filename_prefix"] = "video/LTX-2.5_talking"
    # The sound under the video is the original, not what comes back through the
    # audio VAE: that round trip moved the voice 60 ms earlier than the lips (which
    # follow the encoded original) and cut 70 ms off its end.
    create = find("CreateVideo")
    decode = g[create]["inputs"]["audio"][0]
    g[create]["inputs"]["audio"] = [fit, 0]
    del g[decode]
    return g


def chain(clips=4, keyframes=False):
    """LTX-2.5 clips in a row (#48): each starts on the last frame of the one before
    and has a prompt of its own, so the story moves on over clips × Duration
    seconds. The clips' pictures and sound are joined by LlmctlJoinClips, which
    drops each repeated start frame, keeps the sound on its pictures to the
    sample and crossfades each join within the frame two clips share. The models load once; each clip costs what
    one LTX-2.5 Video (or First-Last Frame) does.

    keyframes: on First-Last Frame, each clip runs from the last frame of the one
    before to a picture of its own (Bild 2 … Bild N+1) — a storyboard, each made
    from the one before with Qwen-Image Edit say, so the person and whatever is
    in the scene carry on. Over a plain chain a face drifts a little with every
    clip and the drift adds up; here every clip ends on the storyboard again,
    pulled to it with *Stärke Zielbild* (0.5).

    *Ton fortsetzen* (on): a later clip's sound starts with the last second of the
    one before, held as it is, and LTX goes on from there — the same wind and
    surf rather than a new soundscape at every join."""
    g = from_base("ltx_flf" if keyframes else "ltx_i2v")
    find = lambda cls, t=None: next(k for k, n in g.items() if n["class_type"] == cls and (t is None or variants.title(n) == t))
    if keyframes:
        start, end = find("LoadImage", "Load First Frame"), find("LoadImage", "Load Last Frame")
        g[start]["_meta"]["title"], g[end]["_meta"]["title"] = "Bild 1", "Bild 2"
    else:
        start, end = find("LoadImage"), None
    prompt, create = find("PrimitiveStringMultiline", "Prompt"), find("CreateVideo")
    sources = {start, prompt} | ({end} if end else set())
    # A clip: what reads its pictures or its prompt, up to the finished video, and
    # its own noise.
    per, grew = set(), True
    while grew:
        grew = False
        for k, n in g.items():
            if k in per or k in sources | {create} or n["class_type"] == "SaveVideo":
                continue
            if n["class_type"] == "RandomNoise" or any(variants.is_ref(x) and x[0] in per | sources
                                                       for x in n["inputs"].values()):
                per.add(k)
                grew = True
    # The first stage's audio: an empty latent, which a later clip can replace
    # with the end of the sound before it.
    first_av = next(k for k in per if g[k]["class_type"] == "LTXVConcatAVLatent"
                    and g[g[k]["inputs"]["audio_latent"][0]]["class_type"] == "LTXVEmptyLatentAudio")
    audio_vae = g[g[first_av]["inputs"]["audio_latent"][0]]["inputs"]["audio_vae"]
    go_on = _flag(g, "Ton fortsetzen", True)
    held_mask = _new_id(g)
    g[held_mask] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}}
    if keyframes:
        # How hard a clip is pulled to its picture at the end. The template's 0.7
        # had LTX reach the picture early and hold it for a second; less leaves it
        # free to move into it.
        pull = _new_id(g)
        g[pull] = {"class_type": "PrimitiveFloat", "inputs": {"value": 0.5}, "_meta": {"title": "Stärke Zielbild"}}
        for k in per:
            if g[k]["class_type"] == "LTXVAddGuide" and g[k]["inputs"]["frame_idx"] == -1:
                g[k]["inputs"]["strength"] = [pull, 0]
    g[prompt]["_meta"]["title"] = "Prompt 1"
    images, audio = [g[create]["inputs"]["images"]], [g[create]["inputs"]["audio"]]
    for i in range(2, clips + 1):
        last, text = (str(int(_new_id(g)) + j) for j in range(2))
        g[last] = {"class_type": "ImageFromBatch", "inputs": {"image": images[-1], "batch_index": -1, "length": 1},
                   "_meta": {"title": f"Last frame of clip {i - 1}"}}
        g[text] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": ""}, "_meta": {"title": f"Prompt {i}"}}
        target = None
        if keyframes:
            target = _new_id(g)
            g[target] = {"class_type": "LoadImage", "inputs": dict(g[end]["inputs"]), "_meta": {"title": f"Bild {i + 1}"}}
        swap = {start: last, prompt: text, **({end: target} if end else {})}
        ids = {}
        for k in sorted(per, key=int):
            ids[k] = _new_id({**g, **{v: None for v in ids.values()}})
        for k in per:
            n = json.loads(json.dumps(g[k]))
            for a, x in n["inputs"].items():
                if a == "bypass":                       # a later clip always starts from its picture
                    n["inputs"][a] = False
                elif variants.is_ref(x) and x[0] in per:
                    n["inputs"][a] = [ids[x[0]], x[1]]
                elif variants.is_ref(x) and x[0] in swap:
                    n["inputs"][a] = [swap[x[0]], x[1]]
            if n["class_type"] == "RandomNoise":
                n["inputs"]["noise_seed"] += i - 1
            if n["class_type"] == "LTXVAddGuide" and n["inputs"]["frame_idx"] == 0:
                n["inputs"]["strength"] = 1.0           # it starts exactly where the clip before ended
            g[ids[k]] = n
        # Ton fortsetzen: the last second of the sound before, encoded and held at
        # the start of this clip's audio; the rest LTX makes.
        av = ids[first_av]
        tail, enc, held, av2 = (str(int(_new_id(g)) + j) for j in range(4))
        g[tail] = {"class_type": "TrimAudioDuration", "inputs": {"audio": audio[-1], "start_index": -1.0, "duration": 1.0}}
        g[enc] = {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": [tail, 0], "audio_vae": audio_vae}}
        g[held] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [enc, 0], "mask": [held_mask, 0]},
                   "_meta": {"title": f"Sound of clip {i - 1} held"}}
        g[av2] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": [av, 0], "audio_latent": [held, 0]}}
        sw = _switch(g, go_on, [av, 0], [av2, 0], "Ton fortsetzen")
        g[av2]["inputs"]["video_latent"] = [av, 0]      # the switch took this read over too
        images.append([ids[images[0][0]], images[0][1]])
        audio.append([ids[audio[0][0]], audio[0][1]])
        assert sw
    join = _new_id(g)
    g[join] = {"class_type": "LlmctlJoinClips", "_meta": {"title": "Clips joined"},
               "inputs": {"frame_rate": g[g[create]["inputs"]["fps"][0]]["inputs"]["values.a"], "crossfade": 0.04,
                          **{f"images_{i + 1}": x for i, x in enumerate(images)},
                          **{f"audio_{i + 1}": x for i, x in enumerate(audio)}}}
    g[create]["inputs"].update(images=[join, 0], audio=[join, 1])
    g[find("SaveVideo")]["inputs"]["filename_prefix"] = "video/LTX-2.5_chain" + ("_keyframes" if keyframes else "")
    return g


def three_d():
    """Pixal3D / TRELLIS.2 from a picture (#38), plus *Mehrere Ansichten
    (TRELLIS.2)*: up to three more pictures of the object (Ansicht 2-4 — the
    side, the back, from above), each cut free and encoded like the main one,
    their conditionings combined (the sampler averages the views' predictions)
    for the structure and shape stages, where the form is decided. The
    upsampling at 1536 takes the main picture alone: with every view it took
    445 s a step instead of a few. Measured: a fox from three pictures in 190 s
    against 355 s from one, a handbag's side profile closer to its side view.
    Each view has its own switch (Ansicht N verwenden), for fewer than three.
    Off, the workflow is the template's; with the template's switch on Pixal3D
    the views go nowhere either way."""
    g = from_base("pixal3d_trellis")
    find = lambda cls, t=None: next(k for k, n in g.items() if n["class_type"] == cls and (t is None or variants.title(n) == t))
    main = find("LoadImage")
    bg_model = find("LoadBackgroundRemovalModel")
    clip_vision = find("CLIPVisionLoader")
    cond = find("Trellis2Conditioning")
    crop = find("ImageCropToMask")
    shape = find("Trellis2ShapeStage")
    upsample = find("Trellis2UpsampleStage")
    template_switches = [k for k, n in g.items() if n["class_type"] == "ComfySwitchNode"]
    flag = _flag(g, "Mehrere Ansichten (TRELLIS.2)", False)
    pos, neg = [cond, 0], [cond, 1]
    for i in (2, 3, 4):
        load, rb, cut, enc, cp, cn = (str(int(_new_id(g)) + j) for j in range(6))
        g[load] = {"class_type": "LoadImage", "inputs": {"image": g[main]["inputs"]["image"]},
                   "_meta": {"title": f"Ansicht {i}"}}
        g[rb] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": [bg_model, 0], "image": [load, 0]}}
        g[cut] = {"class_type": "ImageCropToMask",
                  "inputs": {**{k: v for k, v in g[crop]["inputs"].items() if not variants.is_ref(v)},
                             "images": [load, 0], "masks": [rb, 0]}}
        g[enc] = {"class_type": "Trellis2Conditioning", "inputs": {"clip_vision_model": [clip_vision, 0], "image": [cut, 0]}}
        g[cp] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": pos, "conditioning_2": [enc, 0]}}
        g[cn] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": neg, "conditioning_2": [enc, 1]}}
        # A view of its own switch: an unused slot (holding the main picture,
        # so it validates) would count the main picture twice.
        use = _flag(g, f"Ansicht {i} verwenden", True)
        sp, sn = (str(int(_new_id(g)) + j) for j in range(2))
        for sw, on, off in ((sp, [cp, 0], pos), (sn, [cn, 0], neg)):
            g[sw] = {"class_type": "ComfySwitchNode", "inputs": {"switch": [use, 0], "on_false": off, "on_true": on},
                     "_meta": {"title": f"Ansicht {i}"}}
        pos, neg = [sp, 0], [sn, 0]
    # Structure and shape: the views together, where the switch is on.
    sw_pos, sw_neg = (str(int(_new_id(g)) + j) for j in range(2))
    for sw, off, on in ((sw_pos, [cond, 0], pos), (sw_neg, [cond, 1], neg)):
        g[sw] = {"class_type": "ComfySwitchNode", "inputs": {"switch": [flag, 0], "on_false": off, "on_true": on},
                 "_meta": {"title": "Mehrere Ansichten"}}
    # The template's Pixal3D/TRELLIS.2 switches: one copy keeps the main picture
    # alone (for the upsampling), the original takes the views.
    single = {}
    for k in template_switches:
        for a, x in g[k]["inputs"].items():
            if x in ([cond, 0], [cond, 1]):
                copy = _new_id(g)
                g[copy] = json.loads(json.dumps(g[k]))
                single[x[1]] = [copy, 0]
                g[k]["inputs"][a] = [(sw_pos, sw_neg)[x[1]], 0]
                break
    assert set(single) == {0, 1}
    # The upsampling: the main picture alone, through a shape stage of its own.
    alone, up_pos, up_neg = (str(int(_new_id(g)) + j) for j in range(3))
    g[alone] = {"class_type": "Trellis2ShapeStage", "inputs": {"positive": single[0], "negative": single[1],
                                                                "voxel": g[shape]["inputs"]["voxel"]},
                "_meta": {"title": "Main picture alone (upsampling)"}}
    for sw, a, i in ((up_pos, "positive", 0), (up_neg, "negative", 1)):
        g[sw] = {"class_type": "ComfySwitchNode",
                 "inputs": {"switch": [flag, 0], "on_false": g[upsample]["inputs"][a], "on_true": [alone, i]},
                 "_meta": {"title": "Mehrere Ansichten"}}
        g[upsample]["inputs"][a] = [sw, 0]
    return g


WORKFLOWS = {
    "LTX-2.5 Video (int8, distilled)": video_finish(lambda: from_base("ltx_i2v")),
    "Pixal3D TRELLIS.2 Image to Model (int8)": three_d,
    "LTX-2.5 Chain (int8, distilled, 4 clips)": video_finish(chain),
    "LTX-2.5 Chain Keyframes (int8, distilled, 4 clips)": video_finish(lambda: chain(keyframes=True)),
    "LTX-2.5 First-Last Frame (int8, distilled)": video_finish(lambda: from_base("ltx_flf")),
    "LTX-2.5 Talking (int8, distilled)": video_finish(talking),
    "Qwen-Image 2.1 T2I": qwen_template("qwen_t2i"),
    "Qwen-Image 2.1 Edit": qwen_template("qwen_edit"),
    "Qwen-Image 2.1 Background Removal": qwen_template("qwen_bg", uncensored=False),
    "Qwen-Image 2.1 Control": lambda: toggle_turbo(control_all(), "11", "1"),
    "Qwen-Image 2.1 Inpaint": inpaint_all,
    "Qwen-Image 2.1 Outpaint": with_turbo(outpaint),
    "Qwen-Image 2.1 Colorize": with_turbo(colorize),
    "Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14)": detailer,
    "Z-Image Turbo T2I": zimage_t2i,
    "Z-Image Turbo Control": zimage_control,
    "Z-Image Turbo Inpaint": lambda: toggle_crop(zimage(inpaint)()),
    "Z-Image Turbo Colorize (bf16, 8 Schritte)": zimage(colorize),
    "FLUX.2 dev T2I": flux_dev("flux_dev_t2i"),
    "FLUX.2 dev Edit": flux_dev("flux_dev_edit"),
    "FLUX.2 klein 9B T2I": flux_klein("flux_klein_t2i"),
    "FLUX.2 klein 9B Edit": flux_klein("flux_klein_edit"),
    "BiRefNet Background Removal": birefnet,
    "ACE-Step 1.5": ace,
    "Artifact Check (flash)": artifact_check,
    "Florence-2 Caption": caption,
    "SAM 3 Select": select,
}

# The workflows each variant once was, for checking the switches (tests, --check).
SINGLE = {
    "Qwen-Image 2.1 Control (bf16, dpmpp_2m 14)": control,
    "Qwen-Image 2.1 Canny Control (bf16, dpmpp_2m 14)": canny,
    "Qwen-Image 2.1 Pose Control (bf16, dpmpp_2m 14)": pose,
    "Qwen-Image 2.1 Depth Control (bf16, dpmpp_2m 14)": depth,
    "Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)": inpaint,
    "Qwen-Image 2.1 Colorize (bf16, dpmpp_2m 14)": colorize,
    "Qwen-Image 2.1 Outpaint (bf16, dpmpp_2m 14)": outpaint,
    "Qwen-Image 2.1 Canny Inpaint (bf16, dpmpp_2m 14)": canny_inpaint,
    "Qwen-Image 2.1 Pose Inpaint (bf16, dpmpp_2m 14)": pose_inpaint,
    "Qwen-Image 2.1 Depth Inpaint (bf16, dpmpp_2m 14)": depth_inpaint,
    "Qwen-Image 2.1 Control Turbo (bf16, 4 Schritte)": turbo(control),
    "Qwen-Image 2.1 Canny Control Turbo (bf16, 4 Schritte)": turbo(canny),
    "Qwen-Image 2.1 Pose Control Turbo (bf16, 4 Schritte)": turbo(pose),
    "Qwen-Image 2.1 Depth Control Turbo (bf16, 4 Schritte)": turbo(depth),
    "Qwen-Image 2.1 Inpaint Turbo (bf16, 4 Schritte)": turbo(inpaint),
    "Qwen-Image 2.1 Colorize Turbo (bf16, 4 Schritte)": turbo(colorize),
    "Qwen-Image 2.1 Outpaint Turbo (bf16, 4 Schritte)": turbo(outpaint),
    "Qwen-Image 2.1 Canny Inpaint Turbo (bf16, 4 Schritte)": turbo(canny_inpaint),
    "Qwen-Image 2.1 Pose Inpaint Turbo (bf16, 4 Schritte)": turbo(pose_inpaint),
    "Qwen-Image 2.1 Depth Inpaint Turbo (bf16, 4 Schritte)": turbo(depth_inpaint),
    "Artifact Check (flash)": artifact_check,
    # Strength 0.8 for the controls: the card's 0.65-1.00, where Qwen-Image's
    # scribble-friendly 0.5 lets Z-Image drift.
    "Z-Image Turbo Control (bf16, 8 Schritte)": zimage(control, 0.8),
    # Canny at 1.0 kept the photo whatever the prompt asked; 0.65 and 0.8 gave the
    # watercolour on its layout, 0.5 lost the layout.
    "Z-Image Turbo Canny Control (bf16, 8 Schritte)": zimage(canny, 0.65),
    "Z-Image Turbo Pose Control (bf16, 8 Schritte)": zimage(pose),
    "Z-Image Turbo Depth Control (bf16, 8 Schritte)": zimage(depth),
    "Z-Image Turbo Inpaint (bf16, 8 Schritte)": zimage(inpaint),
    "Z-Image Turbo Colorize (bf16, 8 Schritte)": zimage(colorize),
    "Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14)": detailer,
    "Qwen-Image 2.1 Inpaint Crop (bf16, dpmpp_2m 14)": crop(inpaint),
    "Qwen-Image 2.1 Inpaint Crop Turbo (bf16, 4 Schritte)": crop(turbo(inpaint)),
    "Z-Image Turbo Inpaint Crop (bf16, 8 Schritte)": crop(zimage(inpaint)),
    "BiRefNet Background Removal (general, MIT)": background("BiRefNet-general", "BiRefNet_background"),
    "BiRefNet Matting Background Removal (HR, MIT)": background("BiRefNet-HR-matting", "BiRefNet_background_matting"),
    "Florence-2 Caption": caption,
    "SAM 3 Select": select,
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


# The download entries the template-made workflows carried, by file name.
DOWNLOADS = json.loads((ROOT / "bases" / "downloads.json").read_text())


def with_downloads(wf):
    """The download entry each loader node carries — what llmctl's download reads."""
    for n in wf["nodes"]:
        known = [v for v in (n.get("widgets_values") or []) if isinstance(v, str) and v in DOWNLOADS and v not in MODELS]
        if known:
            n.setdefault("properties", {})["models"] = [{"name": v, **e} for v in known for e in DOWNLOADS[v]]
            continue
        key = LOADER_INPUT.get(n.get("type"))
        if not key:
            continue
        name = (n.get("widgets_values") or [None])[0]
        if name in MODELS:
            # "bbox/face_yolov8m.pt" lies in ultralytics/bbox as face_yolov8m.pt;
            # a list is a model of several files (the BiRefNet ones).
            files = MODELS[name] if isinstance(MODELS[name], list) else [MODELS[name]]
            n.setdefault("properties", {})["models"] = [
                {"name": (url.rsplit("/", 1)[-1] if isinstance(MODELS[name], list) else name.rsplit("/", 1)[-1]),
                 "url": url, "directory": directory} for directory, url in files]
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


if __name__ == "__main__":
    asyncio.run(main())
