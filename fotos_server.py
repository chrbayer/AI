#!/usr/bin/env python3
"""Photo editing for the phone — a page in front of llmctl's image API.

    GET  /fotos                    the page (fotos.html), made for a phone
    GET  /fotos/api/status         ComfyUI and the image API: up, down, starting, stopping
    POST /fotos/api/comfy          {"action": "start"|"stop"}: llmctl starts or stops ComfyUI
    GET  /fotos/api/actions        what the page offers, and which of it ComfyUI can do now
    GET  /fotos/api/prompts        the library of prompts (fotos_prompts.json), searched on the page
    POST /fotos/api/jobs           multipart: image (or from=<job id>), action, quality, text
    GET  /fotos/api/jobs           the jobs, newest first
    GET  /fotos/api/jobs/<id>      one job
    GET  /fotos/api/jobs/<id>/<input|result>[?thumb=1][&download=1]
    DELETE /fotos/api/jobs/<id>

Unlike the image API it does not belong to the ComfyUI slot: it runs on its own
(`llmctl fotos enable`, a user service with its own tunnel), so the page is
there with ComfyUI stopped and can start it. A job runs here, not in the
browser: the picture goes up once, the server sends it to the image API and
keeps the result, so a phone that locks its screen or loses the network finds
it done when it comes back. One job at a time, as ComfyUI works through them
anyway.
"""
import argparse
import base64
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

from PIL import Image, ImageOps

app = Flask(__name__)
ARGS = argparse.Namespace()
PAGE = Path(__file__).with_name("fotos.html")
PROMPTS = Path(__file__).with_name("fotos_prompts.json")

# Qwen-Image Edit works at the picture's own size, and its time grows faster
# than the pixels: a weather change took 151 s at 1.6 MP, 62 s at 1.0 MP and
# 33 s at 0.7 MP (warm) — 1.0 MP as sharp as 1.6, 0.7 MP softer and its
# framing shifted. The ControlNet workflows scale to 1 MP themselves.
EDIT_PIXELS = 1_050_000

# What the page offers. model: (fast, best) — the image API's names; max_side /
# max_pixels: what the picture is brought down to first (SeedVR2 enlarges by 4,
# Qwen-Image edits at the picture's own size). text: whether the user writes
# something, and what it becomes.
ACTIONS = {
    "colorize": {
        "label": "Kolorieren", "hint": "Schwarzweiß wird Farbe",
        "model": ("qwen-image-21-colorize-turbo", "qwen-image-21-colorize"),
        "prompt": "The same photograph in natural colour, realistic skin tones, colour film",
        "text": "optional", "text_label": "Farben vorgeben (optional, englisch) — wirkt aufs ganze Bild",
        "text_placeholder": "z. B. blue sky, green trees",
        "max_side": 2048, "seconds": (30, 90),
    },
    "restore": {
        "label": "Restaurieren", "hint": "Kratzer, Flecken, Rauschen weg",
        "model": ("qwen-image-21-turbo", "qwen-image-21"),
        "prompt": ("Restore this old photograph: remove scratches, dust, stains, creases and noise, "
                   "sharpen it gently. Keep the people, their faces, the composition and the colours "
                   "exactly as they are."),
        "text": None, "max_pixels": EDIT_PIXELS, "seconds": (40, 120),
    },
    # Thorough: FLUX.2 klein's detailer, which redraws small faces (a group
    # photo) where Qwen-Image's repaints a smear.
    "detail": {
        "label": "Gesichter", "hint": "Gesichter und Hände nachzeichnen",
        "model": ("qwen-image-21-detailer", "flux2-klein-9b-detailer"),
        "prompt": "", "text": None, "max_pixels": 2_400_000, "seconds": (60, 90),
    },
    "upscale": {
        "label": "Vergrößern", "hint": "4× größer und schärfer",
        "model": ("seedvr2-7b-upscale", "seedvr2-7b-upscale"),
        "prompt": "", "text": None, "max_side": 1024, "seconds": (40, 40),
    },
    "background": {
        "label": "Freistellen", "hint": "Hintergrund entfernen",
        "model": ("birefnet-background-removal", "birefnet-matting-background-removal"),
        "prompt": "", "text": None, "max_side": 2048, "png": True, "seconds": (10, 20),
    },
    # An instruction edit, not Inpaint: a mask in the shape of a person gets
    # another person painted into it; told to remove one, Qwen-Image Edit fills
    # in the beach behind (tried: 45 s, German words as well as English).
    "remove": {
        "label": "Entfernen", "hint": "Etwas aus dem Bild nehmen",
        "model": ("qwen-image-21-turbo", "qwen-image-21"),
        "prompt": "Entferne {} aus dem Foto. Alles andere bleibt genau so, wie es ist.",
        "text": "fill", "text_label": "Was soll weg?",
        "text_placeholder": "z. B. die Person, den Mülleimer, die Stromleitungen",
        "max_pixels": EDIT_PIXELS, "seconds": (45, 120),
    },
    # The model is the user's (EDIT_MODELS), and a second picture may come along.
    "edit": {
        "label": "Ändern", "hint": "In eigenen Worten, auch mit 2 Fotos",
        "model": ("qwen-image-21-turbo", "qwen-image-21"),
        "prompt": "", "text": "prompt", "text_label": "Was soll sich ändern?",
        "text_placeholder": "z. B. Mach den Himmel abendrot – oder mit 2 Fotos: "
                            "Setze die Person aus Bild 1 in die Szene aus Bild 2",
        "max_pixels": EDIT_PIXELS, "seconds": (45, 150), "models": True, "second": True,
    },
    # Outpaint: the picture is scaled to ~0.8 MP and padded; the prompt describes
    # the whole wider scene (its own, unless the user writes one).
    "expand": {
        "label": "Erweitern", "hint": "Mehr Rand ums Bild",
        "model": ("qwen-image-21-outpaint-turbo", "qwen-image-21-outpaint"),
        "prompt": "A wide panoramic photograph of the whole scene, continuing naturally beyond the frame",
        "text": "replace", "text_label": "Die ganze Szene beschreiben (optional)",
        "text_placeholder": "z. B. Ein Strand bei Sonnenuntergang mit Dünen",
        "max_side": 2048, "seconds": (30, 80), "sides": True,
    },
    # Inpaint inside a mask painted with a finger: transparent where to redraw.
    "paint": {
        "label": "Übermalen", "hint": "Stelle markieren, neu malen",
        "model": ("qwen-image-21-inpaint-crop-turbo", "qwen-image-21-inpaint-crop"),
        "prompt": "", "text": "prompt", "text_label": "Was soll an die markierte Stelle?",
        "text_placeholder": "z. B. eine Vase mit Sonnenblumen",
        "max_side": 2048, "seconds": (35, 90), "mask": True,
        # Form behalten: the photo's edges steer the repaint — a coat masked to be
        # red leather stays the same coat, buttons and folds, in red leather. Only
        # if all of it is marked: half a coat stayed black.
        "keep": ("qwen-image-21-canny-inpaint-turbo", "qwen-image-21-canny-inpaint"),
    },
    # Pulls one named thing out of the picture (Qwen-Image Layered Control at
    # 640 px): the first layer, the thing alone on transparency.
    "cutout": {
        "label": "Ausschneiden", "hint": "Ein Ding als PNG, ohne Rest",
        "model": ("qwen-image-layered-control", "qwen-image-layered-control"),
        "prompt": "", "text": "prompt", "text_label": "Was soll ausgeschnitten werden?",
        "text_placeholder": "z. B. die Gitarre, die Frau, das rote Auto",
        "max_side": 1024, "png": True, "layers": 2, "seconds": (120, 120),
    },
    # The ControlNet workflows: something new on the photo's pose, layout or
    # lines. Z-Image Turbo's came out crisper than Qwen-Image's Turbo; the
    # thorough one is Qwen-Image at 14 steps. Both want the new scene described
    # in full: "als Raumschiff-Quartier" alone left the living room as it was.
    "pose": {
        "label": "Pose übernehmen", "hint": "Jemand anderes, gleiche Haltung",
        "model": ("z-image-turbo-pose-control", "qwen-image-21-pose-control"),
        "prompt": "", "text": "prompt", "text_label": "Wer oder was soll so dastehen, und wo?",
        "text_placeholder": "z. B. eine Tänzerin im roten Kleid auf einer Bühne",
        "max_pixels": EDIT_PIXELS, "seconds": (45, 75),
    },
    # Depth on Qwen-Image Turbo: Z-Image's made an untidy garden neon-green
    # mush and a living room's windows black panels, Qwen-Image's a tended
    # garden and the bright windows kept (27 s).
    "restyle": {
        "label": "Neu gestalten", "hint": "Gleicher Raum, anderer Stil",
        "model": ("qwen-image-21-depth-control-turbo", "qwen-image-21-depth-control"),
        "prompt": "", "text": "prompt", "text_label": "Wie soll es jetzt aussehen?",
        "text_placeholder": "z. B. dasselbe Wohnzimmer als gemütliche Almhütte aus altem Holz, "
                            "Kamin, rote Teppiche – je genauer, desto besser",
        "max_pixels": EDIT_PIXELS, "seconds": (30, 75),
    },
    # Canny on Qwen-Image: Z-Image's (strength 0.65) lost the layout without a
    # description — a beach as watercolour came back an empty landscape.
    "art": {
        "label": "Als Kunstwerk", "hint": "Aquarell, Zeichnung, Comic …",
        "model": ("qwen-image-21-canny-control-turbo", "qwen-image-21-canny-control"),
        "prompt": "", "text": "optional", "text_label": "Was ist zu sehen? (optional, hilft dem Modell)",
        "text_placeholder": "z. B. zwei Kinder am Strand mit einem Hund",
        "max_pixels": EDIT_PIXELS, "seconds": (25, 75), "styles": True,
    },
    # LTX-2.5: the picture comes to life as a clip with sound; with a second
    # picture the clip runs from the first to it (First-Last Frame). Fast takes
    # the first stage alone, at half the size. seconds: for 5 s of clip.
    "animate": {
        "label": "Animieren", "hint": "Kurzes Video mit Ton",
        "model": ("ltx-25-video", "ltx-25-video"), "video": True,
        "prompt": ("The scene comes to life with natural, gentle movement; the people move naturally "
                   "and keep their faces. The camera holds still."),
        "text": "replace", "text_label": "Was soll passieren? (optional)",
        "text_placeholder": "z. B. Sie lacht und winkt in die Kamera, die Wellen rollen heran",
        "max_side": 1536, "seconds": (90, 380), "end_seconds": (160, 900), "second": True,
        "second_label": "Endbild (optional) – das Video läuft dorthin",
    },
    # LTX-2.5 Talking: the picture speaks or sings along to a voice recorded on
    # the phone, the clip as long as the voice. seconds: for 5 s of voice.
    "talk": {
        "label": "Sprechen lassen", "hint": "Das Foto spricht deine Aufnahme",
        "model": ("ltx-25-talking", "ltx-25-talking"), "video": True, "audio": True,
        "prompt": ("The person speaks the words clearly and naturally, lips moving exactly with the voice, "
                   "small natural head movements and facial expressions. The camera holds still."),
        "text": "replace", "text_label": "Wie soll es wirken? (optional)",
        "text_placeholder": "z. B. Sie singt mit geschlossenen Augen, ganz gefühlvoll",
        "max_side": 1536, "seconds": (105, 380),
    },
}
# How the page groups the actions.
GROUPS = [("Verbessern", ["colorize", "restore", "detail", "upscale"]),
          ("Verändern", ["background", "cutout", "remove", "paint", "edit", "expand"]),
          ("Neu erschaffen", ["pose", "restyle", "art"]),
          ("Video", ["animate", "talk"])]
# Als Kunstwerk: the style, and the prompt it makes (the photo's edges keep the layout).
STYLES = {
    "watercolour": ("Aquarell", "A delicate watercolour painting, soft washes of colour on textured paper"),
    "pencil": ("Bleistift", "A detailed pencil drawing, graphite on white paper, fine hatching"),
    "oil": ("Ölgemälde", "An impressionist oil painting, visible brush strokes, rich colours"),
    "comic": ("Comic", "A comic book illustration, bold ink outlines, flat vivid colours"),
    "anime": ("Anime", "An anime illustration, clean line art, soft cel shading"),
}
# Sprechen lassen: the longest voice, in seconds.
VOICE_MAX = 10
# Animieren with an end picture.
FIRST_LAST = "ltx-25-first-last-frame"
# Animieren and the video model of Erzeugen: how long a clip may be.
CLIP_SECONDS = (3, 5, 8)
# Ändern: which model. images: how many pictures its workflow takes.
EDIT_MODELS = {
    "qwen-image-21-turbo": {"label": "Qwen-Image Turbo", "hint": "schnell, 2 Fotos", "images": 2, "seconds": 60},
    "qwen-image-21": {"label": "Qwen-Image", "hint": "gründlicher, 2 Fotos", "images": 2, "seconds": 150},
    "flux2-klein-9b": {"label": "FLUX.2 klein", "hint": "lebendig, 1 Foto", "images": 1, "seconds": 65},
    "flux2-dev-turbo": {"label": "FLUX.2 dev Turbo", "hint": "beste Qualität, langsam", "images": 2, "seconds": 300},
}
# Erzeugen: the text-to-image workflows, as the README compares them.
GEN_MODELS = {
    "z-image-turbo": {"label": "Z-Image Turbo", "hint": "natürliche Fotos", "seconds": 30},
    "flux2-klein-9b": {"label": "FLUX.2 klein", "hint": "lebendig, warmes Licht", "seconds": 40},
    "qwen-image-21": {"label": "Qwen-Image", "hint": "Schrift im Bild", "seconds": 55},
    "z-image": {"label": "Z-Image", "hint": "mehr Abwechslung, Stile", "seconds": 110},
    "flux2-dev-turbo": {"label": "FLUX.2 dev Turbo", "hint": "fast wie dev", "seconds": 150},
    "flux2-dev": {"label": "FLUX.2 dev", "hint": "beste Qualität", "seconds": 300},
    # A clip from a prompt: Z-Image Turbo makes the first frame, LTX-2.5 animates
    # it. LTX from text alone drifts with a short prompt — two seeds of three made
    # a film still of a man instead of the tram asked for.
    "ltx-25-video": {"label": "LTX-2.5 Video", "hint": "Startbild, dann Clip mit Ton", "seconds": 430, "fast_seconds": 130,
                     "video": True, "frame_model": "z-image-turbo"},
}
# Unzensiert: the same workflow with the uncensored encoder or NSFW finetune.
# What has none (Z-Image base, FLUX.2 dev Turbo) is not offered then.
UNCENSORED = {
    "z-image-turbo": "z-image-turbo-nsfw",
    "flux2-klein-9b": "flux2-klein-9b-nsfw",
    "flux2-dev": "flux2-dev-nsfw",
    "qwen-image-21": "qwen-image-21-heretic",
    "qwen-image-21-turbo": "qwen-image-21-heretic-turbo",
}
# About 1 MP in the usual shapes, in steps of 16.
ASPECTS = {"1:1": "1024x1024", "4:3": "1152x864", "3:4": "864x1152", "16:9": "1344x768", "9:16": "768x1344"}
# A clip: ~0.9 MP in steps of 64, as the image API makes them.
VIDEO_ASPECTS = {"1:1": "960x960", "4:3": "1088x832", "3:4": "832x1088", "16:9": "1280x704", "9:16": "704x1280"}
# Erweitern: pixels added to each side (of the ~0.8 MP picture), steps of 8.
SIDES = {
    "wide": {"left": 256, "right": 256, "top": 0, "bottom": 0},
    "tall": {"left": 0, "right": 0, "top": 256, "bottom": 256},
    "all": {"left": 192, "right": 192, "top": 192, "bottom": 192},
    "top": {"left": 0, "right": 0, "top": 384, "bottom": 0},
}
KEEP_JOBS = 100
# What ComfyUI should find free: Qwen-Image 2.1 with its text encoder takes
# 25-30 GB while it works. Below this the page asks before starting it — beside
# Flash-Next (~82 GiB) the machine would go into zram or worse.
FREE_NEEDED_GIB = 40

_jobs_lock = threading.Lock()
_queue = queue.Queue()
_comfy = {"state": None, "message": ""}       # state: starting / stopping while llmctl runs
_comfy_lock = threading.Lock()
_last_done = {"at": 0.0, "freed": True}


class Refused(Exception):
    pass


def error(message, status=400):
    return jsonify({"error": message}), status


# ── jobs on disk ─────────────────────────────────────────────

def jobs_dir():
    return Path(ARGS.data) / "jobs"


def job_path(jid):
    if not jid or not all(c in "0123456789abcdef-" for c in jid):
        raise Refused("no such job")
    return jobs_dir() / jid


def read_job(jid):
    try:
        return json.loads((job_path(jid) / "job.json").read_text())
    except (OSError, ValueError):
        return None


def write_job(job):
    d = job_path(job["id"])
    tmp = d / "job.json.new"
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    tmp.replace(d / "job.json")


def update_job(jid, **fields):
    with _jobs_lock:
        job = read_job(jid)
        if job is None:
            return None
        job.update(fields)
        write_job(job)
        return job


def all_jobs():
    out = []
    for d in jobs_dir().glob("*/job.json"):
        try:
            out.append(json.loads(d.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda j: j["created"], reverse=True)


def prune_jobs():
    for job in all_jobs()[KEEP_JOBS:]:
        if job["state"] in ("done", "failed"):
            shutil.rmtree(job_path(job["id"]), ignore_errors=True)


def public(job):
    """A job as the page sees it, with its place in the queue."""
    j = dict(job)
    if j["state"] == "queued":
        waiting = [x for x in all_jobs() if x["state"] == "queued"]
        waiting.sort(key=lambda x: x["created"])
        j["position"] = next((i + 1 for i, x in enumerate(waiting) if x["id"] == j["id"]), None)
    return j


# ── the picture ──────────────────────────────────────────────

def fit(data, action, name="input"):
    """The upload as the workflow should get it: turned upright by its EXIF,
    brought down to the action's size, as PNG (or JPEG for a photo without
    transparency, to keep the upload to the image API small)."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:                                             # noqa: BLE001
        raise Refused("das ist kein Bild, das der Server lesen kann") from None
    im = ImageOps.exif_transpose(im)
    alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    im = im.convert("RGBA" if alpha else "RGB")
    spec = ACTIONS[action]
    w, h = im.size
    scale = 1.0
    if spec.get("max_side"):
        scale = min(scale, spec["max_side"] / max(w, h))
    if spec.get("max_pixels"):
        scale = min(scale, (spec["max_pixels"] / (w * h)) ** 0.5)
    if scale < 1.0:
        im = im.resize((max(16, round(w * scale)), max(16, round(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if alpha:
        im.save(buf, "PNG")
        return buf.getvalue(), f"{name}.png", im.size
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue(), f"{name}.jpg", im.size


def fit_mask(data, size):
    """The painted mask at the picture's size: transparent where to redraw.
    The page sends it at its own scale of the picture."""
    try:
        m = Image.open(io.BytesIO(data))
        m.load()
    except Exception:                                             # noqa: BLE001
        raise Refused("die Markierung ist kein Bild") from None
    alpha = m.convert("RGBA").split()[3].resize(tuple(size), Image.Resampling.NEAREST)
    if alpha.getextrema()[0] > 127:
        raise Refused("es ist keine Stelle markiert")
    out = Image.new("RGBA", tuple(size), (0, 0, 0, 255))
    out.putalpha(alpha.point(lambda v: 0 if v < 128 else 255))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def input_file(d, name="input"):
    for f in (f"{name}.jpg", f"{name}.png"):
        if (d / f).is_file():
            return d / f
    return None


def resolve(model, uncensored):
    """The image API's name for a model, its uncensored one when asked and there
    is one — colouring or enlarging has nothing to refuse. The page offers no
    model to generate or edit with that has none."""
    return UNCENSORED.get(model, model) if uncensored else model


# ── the worker ───────────────────────────────────────────────

def api_error(r):
    try:
        e = r.json().get("error")
        return e.get("message") if isinstance(e, dict) else str(e)
    except ValueError:
        return r.text[:300] or f"HTTP {r.status_code}"


def edit_form(job):
    """The form fields an edit job sends: model, prompt, and pad where it pads."""
    spec = ACTIONS[job["action"]]
    form = {"model": job["model"], "response_format": "b64_json", "n": "1"}
    text = (job.get("text") or "").strip()
    if spec.get("styles"):
        form["prompt"] = STYLES[job.get("style") or "watercolour"][1] + (" of " + text if text else "")
    elif spec["text"] == "prompt":
        form["prompt"] = text
    elif spec["text"] == "optional":
        form["prompt"] = spec["prompt"] + (", " + text if text else "")
    elif spec["text"] == "fill":
        form["prompt"] = spec["prompt"].format(text)
    elif spec["text"] == "replace":
        form["prompt"] = text or spec["prompt"]
    elif spec["prompt"]:
        form["prompt"] = spec["prompt"]
    if spec.get("sides"):
        form["pad"] = json.dumps(SIDES[job.get("sides") or "wide"])
    if spec.get("layers"):
        form["layers"] = str(spec["layers"])
    return form


def edit_files(d):
    """The pictures (image[] in order) and the mask, as requests wants them."""
    files = []
    for name in ("input", "input2"):
        src = input_file(d, name)
        if src is not None:
            files.append(("image[]", (src.name, src.read_bytes(),
                                      "image/png" if src.suffix == ".png" else "image/jpeg")))
    if not files:
        raise RuntimeError("das Ausgangsbild fehlt")
    if (d / "mask.png").is_file():
        files.append(("mask", ("mask.png", (d / "mask.png").read_bytes(), "image/png")))
    if (d / "voice.wav").is_file():
        files.append(("audio", ("voice.wav", (d / "voice.wav").read_bytes(), "audio/wav")))
    return files


def run_job(jid):
    job = update_job(jid, state="running", started=time.time())
    if job is None:
        return
    d = job_path(jid)
    try:
        model = job["model"]
        if job.get("video"):
            return run_clip(job, d)
        if job["action"] == "generate":
            r = requests.post(f"{ARGS.api}/v1/images/generations", timeout=ARGS.timeout,
                              json={"model": model, "prompt": job["text"], "size": job["size"],
                                    "n": 1, "response_format": "b64_json"})
        else:
            r = requests.post(f"{ARGS.api}/v1/images/edits", timeout=ARGS.timeout,
                              data=edit_form(job), files=edit_files(d))
        if r.status_code != 200:
            raise RuntimeError(api_error(r))
        png = base64.b64decode(r.json()["data"][0]["b64_json"])
        (d / "result.png").write_bytes(png)
        size = Image.open(io.BytesIO(png)).size
        update_job(jid, state="done", finished=time.time(), result_size=list(size),
                   seed=r.json().get("seed"))
        app.logger.info("job %s %s (%s): %.0f s", jid, job["action"], job["model"], time.time() - job["started"])
    except requests.ConnectionError:
        update_job(jid, state="failed", finished=time.time(), error="ComfyUI läuft nicht (mehr)")
    except Exception as e:                                        # noqa: BLE001
        update_job(jid, state="failed", finished=time.time(), error=str(e))
    finally:
        _last_done.update(at=time.time(), freed=False)


def run_clip(job, d):
    """A video job: the clip from the image API, and a still of it for the list."""
    jid = job["id"]
    form = {"model": job["model"], "prompt": clip_prompt(job), "seconds": str(min(10, job["seconds"])),
            "quality": "full" if job.get("quality") == "best" else "fast"}
    if job["action"] == "generate" and input_file(d) is None:
        r = requests.post(f"{ARGS.api}/v1/images/generations", timeout=ARGS.timeout,
                          json={"model": job["frame_model"], "prompt": job["text"], "size": job["size"],
                                "n": 1, "response_format": "b64_json"})
        if r.status_code != 200:
            raise RuntimeError("Startbild: " + api_error(r))
        (d / "input.png").write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
    r = requests.post(f"{ARGS.api}/v1/video/clips", timeout=ARGS.timeout, data=form, files=edit_files(d))
    if r.status_code != 200:
        raise RuntimeError(api_error(r))
    (d / "result.mp4").write_bytes(r.content)
    poster(d)
    size = [int(v) for v in r.headers.get("X-Size", "0x0").split("x")]
    update_job(jid, state="done", finished=time.time(), result_size=size, seed=r.headers.get("X-Seed"))
    app.logger.info("job %s %s (%s): %.0f s", jid, job["action"], job["model"], time.time() - job["started"])


def clip_prompt(job):
    spec = ACTIONS.get(job["action"])
    text = (job.get("text") or "").strip()
    return text or (spec["prompt"] if spec else "")


def poster(d):
    """A still from the clip's first second, for the list and before it plays."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(d / "result.mp4"),
                        "-frames:v", "1", "-q:v", "3", str(d / "result.poster.jpg")],
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        app.logger.warning("poster: %s", r.stderr.decode(errors="replace")[-200:])


def worker():
    while True:
        jid = _queue.get()
        try:
            run_job(jid)
        except Exception:                                         # noqa: BLE001
            app.logger.exception("job %s", jid)
        prune_jobs()


def janitor():
    """ComfyUI keeps its models in memory after a job; when nothing has run for
    a while, it lets them go, so an LLM started beside it finds the room."""
    while True:
        time.sleep(30)
        if (ARGS.free_after and not _last_done["freed"] and _queue.empty()
                and time.time() - _last_done["at"] > ARGS.free_after):
            try:
                requests.post(f"{ARGS.comfy}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
                app.logger.info("ComfyUI: models unloaded after %d s idle", ARGS.free_after)
            except requests.RequestException:
                pass
            _last_done["freed"] = True


# ── ComfyUI itself ───────────────────────────────────────────

def api_status():
    """(image API up, ComfyUI up)."""
    try:
        r = requests.get(f"{ARGS.api}/repair/status", timeout=3)
        return True, bool(r.json().get("comfy", {}).get("up"))
    except (requests.RequestException, ValueError):
        return False, False


def mem_available_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return None


def run_llmctl(action):
    try:
        if action == "start":
            cmd = [ARGS.llmctl, "start", ARGS.comfy_model, str(ARGS.comfy_slot), "--proxy"]
        else:
            cmd = [ARGS.llmctl, "stop", str(ARGS.comfy_slot)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-4:])
        if r.returncode != 0:
            _comfy.update(state=None, message=tail or f"llmctl {action} failed")
            return
        if action == "start":
            deadline = time.time() + 300
            while time.time() < deadline and api_status() != (True, True):
                time.sleep(2)
            _comfy.update(state=None, message="" if api_status() == (True, True)
                          else "ComfyUI ist nicht hochgekommen — llmctl logs " + str(ARGS.comfy_slot))
        else:
            _comfy.update(state=None, message="")
    except Exception as e:                                        # noqa: BLE001
        _comfy.update(state=None, message=f"llmctl {action}: {e}")


# ── endpoints ────────────────────────────────────────────────

@app.get("/fotos")
def page():
    return Response(PAGE.read_text(), mimetype="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/fotos/")
def page_slash():
    return page()


@app.get("/fotos/manifest.webmanifest")
def manifest():
    return jsonify({"name": "Fotos", "short_name": "Fotos", "start_url": "/fotos", "scope": "/fotos",
                    "display": "standalone", "background_color": "#161618", "theme_color": "#161618",
                    "icons": [{"src": "/fotos/icon.svg", "sizes": "any", "type": "image/svg+xml"}]})


ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="14" fill="#2f6fdb"/>
<rect x="12" y="18" width="40" height="30" rx="5" fill="none" stroke="#fff" stroke-width="4"/>
<circle cx="32" cy="33" r="8" fill="none" stroke="#fff" stroke-width="4"/>
<rect x="24" y="12" width="16" height="8" rx="3" fill="#fff"/></svg>"""


@app.get("/fotos/icon.svg")
def icon():
    return Response(ICON, mimetype="image/svg+xml", headers={"Cache-Control": "max-age=86400"})


@app.get("/fotos/api/status")
def status():
    api, comfy = api_status()
    state = _comfy["state"] or ("up" if api and comfy else "starting" if api else "down")
    busy = [j for j in all_jobs() if j["state"] in ("queued", "running")]
    return jsonify({"comfy": state, "message": _comfy["message"], "busy": len(busy),
                    "slot": ARGS.comfy_slot})


@app.post("/fotos/api/comfy")
def comfy():
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in ("start", "stop"):
        return error("action is start or stop")
    free = mem_available_gib()
    if action == "start" and not body.get("force") and free is not None and free < FREE_NEEDED_GIB:
        return jsonify({"error": f"Nur {free:.0f} GiB Speicher frei, ComfyUI braucht beim Bearbeiten "
                                 f"25–30 GB, für Videos über 40 GB. Läuft ein großes Sprachmodell? Erst das stoppen – "
                                 f"oder trotzdem starten.", "code": "memory"}), 409
    with _comfy_lock:
        if _comfy["state"]:
            return error(f"ComfyUI wird gerade schon {'gestartet' if _comfy['state'] == 'starting' else 'gestoppt'}", 409)
        if action == "stop" and any(j["state"] == "running" for j in all_jobs()):
            return error("Es läuft gerade eine Bearbeitung", 409)
        # Stop only what is ComfyUI with its image API — not whatever else holds the slot.
        if action == "stop" and not api_status()[0]:
            return error(f"Auf Slot {ARGS.comfy_slot} läuft keine ComfyUI mit Bild-API", 409)
        _comfy.update(state="starting" if action == "start" else "stopping", message="")
    threading.Thread(target=run_llmctl, args=(action,), daemon=True).start()
    return jsonify({"comfy": _comfy["state"]})


@app.get("/fotos/api/prompts")
def prompts():
    """The library of prompts the page searches (fotos_prompts.json)."""
    try:
        data = json.loads(PROMPTS.read_text())
    except (OSError, ValueError):
        data = {"topics": [], "prompts": []}
    data.pop("_comment", None)
    return jsonify(data)


@app.get("/fotos/api/actions")
def actions():
    have = None
    try:
        r = requests.get(f"{ARGS.api}/v1/models", timeout=10)
        have = {m["id"] for m in r.json()["data"]}
    except (requests.RequestException, ValueError, KeyError):
        pass
    def avail(model):
        return None if have is None else model in have

    def unc(model):
        return None if model not in UNCENSORED else avail(UNCENSORED[model])

    out = []
    for key, a in ACTIONS.items():
        fast, best = a["model"]
        out.append({"id": key, "label": a["label"], "hint": a["hint"], "text": a["text"],
                    "text_label": a.get("text_label"), "text_placeholder": a.get("text_placeholder"),
                    "qualities": (fast != best or bool(a.get("video"))) and not a.get("models"),
                    "seconds": a["seconds"], "end_seconds": a.get("end_seconds"),
                    "models": bool(a.get("models")), "second": bool(a.get("second")),
                    "sides": bool(a.get("sides")), "mask": bool(a.get("mask")),
                    "video": bool(a.get("video")), "second_label": a.get("second_label"),
                    "audio": bool(a.get("audio")), "styles": bool(a.get("styles")),
                    "keep": bool(a.get("keep")), "keep_available": avail(a["keep"][0]) if a.get("keep") else None,
                    "group": next(g for g, ids in GROUPS for i in ids if i == key),
                    "second_available": avail(FIRST_LAST) if a.get("video") else True,
                    "available": avail(fast), "best_available": avail(best),
                    "uncensored": [unc(fast), unc(best)]})
    def models(table):
        # a clip from a prompt needs its first frame's model too, and is as uncensored as that
        return [dict(v, id=k, available=avail(k) if avail(k) is False else avail(v.get("frame_model", k)),
                     uncensored=unc(v.get("frame_model", k))) for k, v in table.items()]

    return jsonify({"actions": out, "edit_models": models(EDIT_MODELS), "gen_models": models(GEN_MODELS),
                    "aspects": list(ASPECTS), "clip_seconds": list(CLIP_SECONDS), "voice_max": VOICE_MAX,
                    "groups": [g for g, _ in GROUPS], "styles": [[k, v[0]] for k, v in STYLES.items()]})


def new_job(fields, files=()):
    """A job directory with its files and job.json, queued."""
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = job_path(jid)
    d.mkdir(parents=True)
    for name, body in files:
        (d / name).write_bytes(body)
    job = dict(fields, id=jid, state="queued", created=time.time())
    with _jobs_lock:
        write_job(job)
    _queue.put(jid)
    return job


def voice(data):
    """The recording as WAV, cut to VOICE_MAX seconds to the sample, and its
    length. The phone records WebM/Opus or MP4/AAC; ffmpeg reads both."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
                       input=data, capture_output=True, timeout=120)
    pcm = r.stdout[:VOICE_MAX * 48000 * 2]
    if r.returncode != 0 or len(pcm) < 48000:                     # under half a second
        raise Refused("die Aufnahme lässt sich nicht lesen" if r.returncode or not pcm else "die Aufnahme ist zu kurz")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(pcm)
    return buf.getvalue(), len(pcm) / (2 * 48000)


def clip_seconds(value):
    try:
        n = int(value or 5)
    except ValueError:
        raise Refused("ungültige Länge") from None
    if n not in CLIP_SECONDS:
        raise Refused("ein Video ist " + ", ".join(map(str, CLIP_SECONDS)) + " Sekunden lang")
    return n


def picture(field, from_field):
    """(bytes, parent job) of an uploaded picture or of an earlier result."""
    if field in request.files:
        return request.files[field].read(), None
    parent = request.form.get(from_field)
    if parent:
        src = job_path(parent) / "result.png"
        if not src.is_file():
            raise Refused("dieses Ergebnis gibt es nicht mehr")
        return src.read_bytes(), parent
    return None, None


@app.post("/fotos/api/jobs")
def create_job():
    try:
        f = request.form
        action = f.get("action", "")
        # A template's [placeholder] left as it is counts with its words.
        text = re.sub(r"\[([^\]]*)\]", r"\1", (f.get("text") or "")).strip()
        if len(text) > 2000:
            raise Refused("der Text ist zu lang")
        uncensored = f.get("uncensored") in ("1", "true", "on")
        if action == "generate":
            model = f.get("model") or "z-image-turbo"
            if model not in GEN_MODELS:
                raise Refused("unbekanntes Modell")
            if not text:
                raise Refused("Was soll auf dem Bild sein? – bitte beschreiben")
            aspect = f.get("aspect") if f.get("aspect") in ASPECTS else "1:1"
            n = min(4, max(1, int(f.get("n") or 1)))
            video = bool(GEN_MODELS[model].get("video"))
            fields = {"action": "generate", "label": "Erzeugt", "text": text, "aspect": aspect,
                      "size": (VIDEO_ASPECTS if video else ASPECTS)[aspect], "model": resolve(model, uncensored),
                      "model_label": GEN_MODELS[model]["label"],
                      "uncensored": uncensored and model in UNCENSORED, "png": False}
            if video:
                frame = GEN_MODELS[model]["frame_model"]
                fields.update(video=True, label="Video", seconds=clip_seconds(f.get("seconds")),
                              quality="best" if f.get("quality") == "best" else "fast",
                              frame_model=resolve(frame, uncensored), uncensored=uncensored and frame in UNCENSORED)
            jobs = [new_job(fields) for _ in range(n)]       # one job per picture: each its own card
            return jsonify([public(j) for j in jobs])
        if action not in ACTIONS:
            raise Refused("unbekannte Aktion")
        spec = ACTIONS[action]
        if spec["text"] in ("prompt", "fill") and not text:
            raise Refused(f"{spec['text_label']} – bitte ausfüllen")
        quality = "best" if f.get("quality") == "best" else "fast"
        if spec.get("models"):
            model = f.get("model") or spec["model"][0]
            if model not in EDIT_MODELS:
                raise Refused("unbekanntes Modell")
            label = EDIT_MODELS[model]["label"]
        elif spec.get("keep") and f.get("keep") == "1":
            model, label = spec["keep"][1 if quality == "best" else 0], "Form behalten"
        else:
            model = spec["model"][1 if quality == "best" else 0]
            label = None
        style = f.get("style") if f.get("style") in STYLES else "watercolour"
        if spec.get("styles"):
            label = STYLES[style][0]
        data, parent = picture("image", "from")
        if data is None:
            raise Refused("kein Bild")
        body, name, size = fit(data, action)
        files = [(name, body)]
        data2, parent2 = picture("image2", "from2") if spec.get("second") else (None, None)
        if data2 is not None and spec.get("video"):
            model, label = FIRST_LAST, "mit Endbild"
            body2, name2, _ = fit(data2, action, "input2")
            files.append((name2, body2))
        elif data2 is not None:
            if EDIT_MODELS[model]["images"] < 2:
                raise Refused(f"{label} nimmt nur ein Foto – für zwei Qwen-Image oder FLUX.2 dev wählen")
            body2, name2, _ = fit(data2, action, "input2")
            files.append((name2, body2))
        if spec.get("mask"):
            if "mask" not in request.files:
                raise Refused("bitte erst die Stelle markieren")
            files.append(("mask.png", fit_mask(request.files["mask"].read(), size)))
        seconds = clip_seconds(f.get("seconds")) if spec.get("video") and not spec.get("audio") else None
        if spec.get("audio"):
            if "audio" not in request.files:
                raise Refused("bitte erst etwas aufnehmen oder eine Audiodatei wählen")
            wav, length = voice(request.files["audio"].read())
            files.append(("voice.wav", wav))
            seconds = max(1, math.ceil(length - 1e-6))
        sides = f.get("sides") if f.get("sides") in SIDES else "wide"
        job = new_job({"action": action, "label": spec["label"], "text": text, "quality": quality,
                       "model": resolve(model, uncensored), "model_label": label,
                       "uncensored": uncensored and model in UNCENSORED,
                       "parent": parent, "parent2": parent2, "two": data2 is not None,
                       "sides": sides if spec.get("sides") else None,
                       "input_size": list(size), "png": bool(spec.get("png")),
                       "style": style if spec.get("styles") else None,
                       **({"video": True, "seconds": seconds} if spec.get("video") else {})},
                      files)
        return jsonify([public(job)])
    except Refused as e:
        return error(str(e))
    except ValueError:
        return error("ungültige Angabe")


@app.get("/fotos/api/jobs")
def list_jobs():
    return jsonify([public(j) for j in all_jobs()[:60]])


@app.get("/fotos/api/jobs/<jid>")
def get_job(jid):
    try:
        job_path(jid)
    except Refused:
        return error("no such job", 404)
    job = read_job(jid)
    return jsonify(public(job)) if job else error("no such job", 404)


@app.get("/fotos/api/jobs/<jid>/<which>")
def job_image(jid, which):
    try:
        d = job_path(jid)
    except Refused:
        return error("no such job", 404)
    job = read_job(jid)
    if job is None or which not in ("input", "input2", "result", "poster"):
        return error("no such job", 404)
    name = f"{job['label'].lower()}-{jid}"
    if job.get("video") and which in ("result", "poster"):
        # The clip as it came (Range requests for the phone's player), and its still.
        if which == "result" and not request.args.get("thumb"):
            return send_file(d / "result.mp4", mimetype="video/mp4", max_age=86400, conditional=True,
                             as_attachment=bool(request.args.get("download")), download_name=name + ".mp4")
        which = "result.poster"
    if which == "poster":
        return error("not there", 404)
    src = d / "result.png" if which == "result" else d / "result.poster.jpg" if which == "result.poster" \
        else input_file(d, which)
    if src is None or not src.is_file():
        return error("not there", 404)
    if request.args.get("thumb"):
        cache = d / f"{which}.thumb.jpg"
        if not cache.is_file():
            im = Image.open(src)
            im = (im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB"))
            im.thumbnail((360, 360))
            if im.mode == "RGBA":                       # transparent: on light grey
                bg = Image.new("RGB", im.size, (225, 225, 225))
                bg.paste(im, mask=im.split()[3])
                im = bg
            im.save(cache, "JPEG", quality=82)
        return send_file(cache, mimetype="image/jpeg", max_age=86400)
    if which == "result.poster":
        return send_file(src, mimetype="image/jpeg", max_age=86400)
    if which == "result" and not job.get("png"):
        # A photo as JPEG: a quarter of the PNG, and what the phone's gallery expects.
        cache = d / "result.jpg"
        if not cache.is_file():
            Image.open(src).convert("RGB").save(cache, "JPEG", quality=92)
        return send_file(cache, mimetype="image/jpeg", max_age=86400,
                         as_attachment=bool(request.args.get("download")), download_name=name + ".jpg")
    return send_file(src, max_age=86400, as_attachment=bool(request.args.get("download")),
                     download_name=name + src.suffix)


@app.delete("/fotos/api/jobs/<jid>")
def delete_job(jid):
    try:
        d = job_path(jid)
    except Refused:
        return error("no such job", 404)
    job = read_job(jid)
    if job is None:
        return error("no such job", 404)
    if job["state"] == "running":
        return error("läuft gerade", 409)
    if job["state"] == "queued":
        update_job(jid, state="failed", error="abgebrochen")
    shutil.rmtree(d, ignore_errors=True)
    return jsonify({"deleted": jid})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--api", required=True, help="the image API, e.g. http://127.0.0.1:8089")
    p.add_argument("--comfy", required=True, help="ComfyUI itself, e.g. http://127.0.0.1:8009 (to free its memory)")
    p.add_argument("--comfy-slot", type=int, required=True)
    p.add_argument("--comfy-model", default="comfy")
    p.add_argument("--llmctl", required=True, help="llmctl itself, to start and stop ComfyUI")
    p.add_argument("--data", required=True, help="where the jobs are kept")
    p.add_argument("--timeout", type=int, default=3700)
    p.add_argument("--free-after", type=int, default=900,
                   help="seconds without a job after which ComfyUI unloads its models (0: never)")
    p.parse_args(namespace=ARGS)
    ARGS.api, ARGS.comfy = ARGS.api.rstrip("/"), ARGS.comfy.rstrip("/")
    jobs_dir().mkdir(parents=True, exist_ok=True)
    # What was queued or running when the server last stopped is not coming back.
    for job in all_jobs():
        if job["state"] in ("queued", "running"):
            update_job(job["id"], state="failed", error="vom Neustart des Servers unterbrochen")
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=janitor, daemon=True).start()
    print(f"Fotos on http://{ARGS.host}:{ARGS.port}/fotos — image API {ARGS.api}, "
          f"ComfyUI on slot {ARGS.comfy_slot}, jobs in {jobs_dir()}", flush=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        from waitress import serve
        serve(app, host=ARGS.host, port=ARGS.port, threads=8)
    except ImportError:
        app.run(host=ARGS.host, port=ARGS.port, threaded=True)


if __name__ == "__main__":
    main()
