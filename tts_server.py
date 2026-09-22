#!/usr/bin/env python3
"""Speech for a llmctl slot — an OpenAI-compatible /v1/audio/speech in front of
llama.cpp's llama-tts.

llama.cpp runs Qwen3-TTS (since August 2026) but only in its llama-tts binary;
llama-server has no speech endpoint yet (PR #26603). So this server takes the
request, runs llama-tts once per request, and returns the WAV it writes. Loading
costs ~1.5 s per request on top of generation, which runs at about twice real
time on this machine. When llama-server gains the endpoint, the backend can
move there.

Qwen3-TTS "Base" speaks in the voice of a short reference recording, and with
it, accent-free German: a voice is nothing but an audio file in the voices
directory (`llmctl voice …` manages them), selected by its file name.

    POST /v1/audio/speech   {"input": "...", "voice": "mann", "response_format": "wav"|"pcm",
                             "language": "de"}          → audio (24 kHz mono, 16 bit)
    GET  /v1/audio/voices   {"voices": ["frau", "mann", ...]}
    GET  /v1/models         the one model this slot serves
    GET  /health

`language` is an extension (ISO 639-1; the slot's --lang otherwise). `model` is
accepted and ignored, as a slot serves one model; `speed` other than 1 is refused
rather than ignored.
"""
import argparse
import io
import logging
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from flask import Flask, Response, jsonify, request

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")
LANGUAGES = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
MAX_INPUT = 4096            # characters per request, as OpenAI's API has it

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts_server")
app = Flask(__name__)
args = argparse.Namespace()     # filled in by main()
gpu = threading.Lock()      # one generation at a time: they would only fight over the GPU


def voices():
    d = Path(args.voices)
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_SUFFIXES}


def error(status, message, kind="invalid_request_error"):
    return jsonify({"error": {"message": message, "type": kind}}), status


@app.get("/health")
def health():
    return jsonify({"status": "ok", "voices": len(voices())})


@app.get("/v1/models")
def models():
    return jsonify({"object": "list", "data": [{"id": args.alias, "object": "model", "owned_by": "llmctl"}]})


@app.get("/v1/audio/voices")
def list_voices():
    return jsonify({"voices": list(voices())})


@app.post("/v1/audio/speech")
def speech():
    body = request.get_json(silent=True) or {}
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        return error(400, "'input' must be non-empty text")
    if len(text) > MAX_INPUT:
        return error(400, f"'input' is limited to {MAX_INPUT} characters")
    fmt = body.get("response_format", "wav")
    if fmt not in ("wav", "pcm"):
        return error(400, f"response_format '{fmt}' is not supported; use wav or pcm")
    if float(body.get("speed", 1) or 1) != 1:
        return error(400, "'speed' is not supported by this model")
    lang = body.get("language", args.lang)
    if lang not in LANGUAGES:
        return error(400, f"language '{lang}' is not one of {', '.join(sorted(LANGUAGES))}")
    known = voices()
    name = body.get("voice") or args.default_voice or next(iter(known), "")
    if name not in known:
        return error(400, f"unknown voice '{name}'; known: {', '.join(known) or 'none — add one with llmctl voice'}")

    with tempfile.TemporaryDirectory(prefix="llmctl-tts-") as tmp:
        out = Path(tmp) / "out.wav"
        cmd = [args.llama_tts, "-m", args.model, "-mm", args.mmproj, "-p", text,
               "--tts-lang", lang, "--tts-speaker-file", str(known[name]), "-o", str(out),
               *args.extra]
        with gpu:
            t = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                return error(504, f"generation took longer than {args.timeout} s", "server_error")
            dt = time.time() - t
        if r.returncode != 0 or not out.is_file():
            tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
            log.error("llama-tts failed (%s): %s", r.returncode, " | ".join(tail))
            return error(500, "llama-tts failed: " + (tail[-1] if tail else f"exit {r.returncode}"), "server_error")
        data = out.read_bytes()

    with wave.open(io.BytesIO(data)) as w:
        seconds = w.getnframes() / w.getframerate()
        pcm = w.readframes(w.getnframes())
    log.info("voice=%s lang=%s chars=%d audio=%.1fs took=%.1fs (%.2fx real time)",
             name, lang, len(text), seconds, dt, seconds / dt if dt else 0)
    if fmt == "pcm":
        return Response(pcm, mimetype="audio/pcm")
    return Response(data, mimetype="audio/wav")


def main():
    global args
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True, help="Qwen3-TTS Base GGUF")
    p.add_argument("--mmproj", required=True, help="its mmproj GGUF (speaker encoder, code predictor, vocoder)")
    p.add_argument("--voices", required=True, help="directory of reference recordings, one voice per file")
    p.add_argument("--llama-tts", default="llama-tts")
    p.add_argument("--lang", default="de")
    p.add_argument("--default-voice", default="")
    p.add_argument("--alias", default="qwen3-tts")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("extra", nargs="*", help="further llama-tts arguments (after --)")
    args = p.parse_args()
    log.info("model %s, voices in %s (%d), language %s", os.path.basename(args.model), args.voices,
             len(voices()), args.lang)
    # llmctl waits for this line before it calls the slot up.
    log.info("tts_server listening on http://%s:%d", args.host, args.port)
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=4)
    except ImportError:
        app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
