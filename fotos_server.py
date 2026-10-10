#!/usr/bin/env python3
"""Photo editing for the phone — a page in front of llmctl's image API.

    GET  /fotos                    the page (fotos.html), made for a phone
    GET  /fotos/api/status         ComfyUI and the image API: up, down, starting, stopping
    POST /fotos/api/comfy          {"action": "start"|"stop"}: llmctl starts or stops ComfyUI
    GET  /fotos/api/actions        what the page offers, and which of it ComfyUI can do now
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
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

from PIL import Image, ImageOps

app = Flask(__name__)
ARGS = argparse.Namespace()
PAGE = Path(__file__).with_name("fotos.html")

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
        "text": None, "max_pixels": 1_600_000, "seconds": (40, 120),
    },
    "detail": {
        "label": "Gesichter", "hint": "Gesichter und Hände nachzeichnen",
        "model": ("qwen-image-21-detailer", "qwen-image-21-detailer"),
        "prompt": "", "text": None, "max_pixels": 2_400_000, "seconds": (60, 60),
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
        "max_pixels": 1_600_000, "seconds": (45, 120),
    },
    "edit": {
        "label": "Ändern", "hint": "In eigenen Worten",
        "model": ("qwen-image-21-turbo", "qwen-image-21"),
        "prompt": "", "text": "prompt", "text_label": "Was soll sich ändern?",
        "text_placeholder": "z. B. Mach den Himmel abendrot",
        "max_pixels": 1_600_000, "seconds": (40, 120),
    },
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

def fit(data, action):
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
        return buf.getvalue(), "input.png", im.size
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue(), "input.jpg", im.size


def input_file(d):
    for name in ("input.jpg", "input.png"):
        if (d / name).is_file():
            return d / name
    return None


# ── the worker ───────────────────────────────────────────────

def api_error(r):
    try:
        e = r.json().get("error")
        return e.get("message") if isinstance(e, dict) else str(e)
    except ValueError:
        return r.text[:300] or f"HTTP {r.status_code}"


def run_job(jid):
    job = update_job(jid, state="running", started=time.time())
    if job is None:
        return
    d = job_path(jid)
    spec = ACTIONS[job["action"]]
    src = input_file(d)
    if src is None:
        update_job(jid, state="failed", finished=time.time(), error="das Ausgangsbild fehlt")
        return
    model = spec["model"][1 if job.get("quality") == "best" else 0]
    form = {"model": model, "response_format": "b64_json", "n": "1"}
    text = (job.get("text") or "").strip()
    if spec["text"] == "prompt":
        form["prompt"] = text
    elif spec["text"] == "optional":
        form["prompt"] = spec["prompt"] + (", " + text if text else "")
    elif spec["text"] == "fill":
        form["prompt"] = spec["prompt"].format(text)
    elif spec["prompt"]:
        form["prompt"] = spec["prompt"]
    try:
        with open(src, "rb") as f:
            r = requests.post(f"{ARGS.api}/v1/images/edits", data=form,
                              files={"image": (src.name, f, "image/png" if src.suffix == ".png" else "image/jpeg")},
                              timeout=ARGS.timeout)
        if r.status_code != 200:
            raise RuntimeError(api_error(r))
        png = base64.b64decode(r.json()["data"][0]["b64_json"])
        (d / "result.png").write_bytes(png)
        size = Image.open(io.BytesIO(png)).size
        update_job(jid, state="done", finished=time.time(), model=model,
                   result_size=list(size))
        app.logger.info("job %s %s (%s): %.0f s", jid, job["action"], model, time.time() - job["started"])
    except requests.ConnectionError:
        update_job(jid, state="failed", finished=time.time(), error="ComfyUI läuft nicht (mehr)")
    except Exception as e:                                        # noqa: BLE001
        update_job(jid, state="failed", finished=time.time(), error=str(e))
    finally:
        _last_done.update(at=time.time(), freed=False)


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
                                 f"25–30 GB. Läuft ein großes Sprachmodell? Erst das stoppen – "
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


@app.get("/fotos/api/actions")
def actions():
    have = None
    try:
        r = requests.get(f"{ARGS.api}/v1/models", timeout=10)
        have = {m["id"] for m in r.json()["data"]}
    except (requests.RequestException, ValueError, KeyError):
        pass
    out = []
    for key, a in ACTIONS.items():
        fast, best = a["model"]
        out.append({"id": key, "label": a["label"], "hint": a["hint"], "text": a["text"],
                    "text_label": a.get("text_label"), "text_placeholder": a.get("text_placeholder"),
                    "qualities": fast != best, "seconds": a["seconds"],
                    "available": None if have is None else fast in have,
                    "best_available": None if have is None else best in have})
    return jsonify(out)


@app.post("/fotos/api/jobs")
def create_job():
    try:
        action = request.form.get("action", "")
        if action not in ACTIONS:
            raise Refused("unbekannte Aktion")
        spec = ACTIONS[action]
        text = (request.form.get("text") or "").strip()
        if spec["text"] in ("prompt", "fill") and not text:
            raise Refused(f"{spec['text_label']} — bitte ausfüllen")
        if len(text) > 2000:
            raise Refused("der Text ist zu lang")
        if "image" in request.files:
            data = request.files["image"].read()
            parent = None
        elif request.form.get("from"):
            parent = request.form["from"]
            src = job_path(parent) / "result.png"
            if not src.is_file():
                raise Refused("dieses Ergebnis gibt es nicht mehr")
            data = src.read_bytes()
        else:
            raise Refused("kein Bild")
        body, name, size = fit(data, action)
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = job_path(jid)
        d.mkdir(parents=True)
        (d / name).write_bytes(body)
        job = {"id": jid, "action": action, "label": spec["label"], "text": text,
               "quality": "best" if request.form.get("quality") == "best" else "fast",
               "state": "queued", "created": time.time(), "parent": parent,
               "input_size": list(size), "png": bool(spec.get("png"))}
        with _jobs_lock:
            write_job(job)
        _queue.put(jid)
        return jsonify(public(job))
    except Refused as e:
        return error(str(e))


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
    if job is None or which not in ("input", "result"):
        return error("no such job", 404)
    src = input_file(d) if which == "input" else d / "result.png"
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
    name = f"{job['label'].lower()}-{jid}"
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
