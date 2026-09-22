#!/usr/bin/env python3
"""Speech for a llmctl slot — an OpenAI-compatible /v1/audio/speech in front of
llama.cpp's Qwen3-TTS.

Two engines, chosen by the binary llmctl hands over:

  server  llama-server with a /tts endpoint (llama.cpp PR #26603, not merged
          yet): started once as a child on a private port, model resident,
          audio streamed as it is generated.
  cli     llama-tts, run once per request: ~1.5 s of loading each time, and the
          audio only when the whole utterance is done.

Either way the text is spoken sentence by sentence: the first sound comes after
the first sentence rather than after the whole text, and no single generation
runs into llama.cpp's frame limit (512 frames, ~42 s). Each piece is generated
without the ones before it, so tone and pace can shift a little at sentence
boundaries; "chunking": false (or the server's --no-chunking) speaks groups of
sentences up to ~280 characters instead — smoother, but the first sound waits
for the whole first group.

Generation samples, so the same text sounds a little different every time.
A request's "seed" (or the server's --seed) fixes it; -1, the default, draws
a new one each time. The default is random on purpose: how much a clone
sounds like its reference varies from draw to draw (0.78-0.89 by a speaker
model here), a fixed seed keeps one draw for good, and the best seed for one
text is no better than any other for the next.

Qwen3-TTS "Base" speaks in the voice of a short reference recording, and with
it, accent-free German: a voice is nothing but an audio file in the voices
directory (`llmctl voice …` manages them), selected by its file name.

    POST /v1/audio/speech   {"input": "...", "voice": "mann",
                             "response_format": "wav"|"pcm", "stream": true,
                             "language": "de", "seed": -1, "chunking": true}
                                                   → 24 kHz mono 16-bit audio
    GET  /v1/audio/voices   {"voices": ["frau", "mann", ...]}
    GET  /v1/models         the one model this slot serves
    GET  /health

`pcm` is raw little-endian 16-bit, as OpenAI's API has it. Like OpenAI's, the
audio is streamed as it is generated unless the request says "stream": false —
clients such as the openai package expect that and send no flag; a streamed
`wav` carries an unknown length in its header, which players read to the end.
`stream_format` "audio" is that; "sse" is not offered. `language` is an extension (ISO
639-1; the slot's --lang otherwise). `model` is accepted and ignored; `speed`
other than 1 is refused rather than ignored.

The clone speaks as loud as its reference was recorded, so every whole answer
("stream": false) is brought to one loudness (--loudness, -20 dBFS RMS, peaks
below -1 dBFS; off to disable). A stream cannot be levelled without knowing its end; the voices are
stored at that level already (`llmctl voice`), and their clones land within a
few dB of it.
"""
import argparse
import base64
import ctypes
import json
import logging
import os
import re
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")
LANGUAGES = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
MAX_INPUT = 4096            # characters per request, as OpenAI's API has it
SAMPLE_RATE = 24000
PEAK_CEILING_DBFS = -1.0
CHUNK_CHARS = 280           # a sentence group per generation; far below the frame limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts_server")
app = Flask(__name__)
args = argparse.Namespace()     # filled in by main()
gpu = threading.Lock()          # one generation at a time: they would only fight over the GPU
engine = None


# ── voices, text, audio ─────────────────────────────────────

def voices():
    d = Path(args.voices)
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_SUFFIXES}


def sentences(text, first_alone=True):
    """Groups of whole sentences, each at most CHUNK_CHARS unless one sentence is
    longer; with first_alone, the first is a single sentence, so the first sound
    comes early."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…:;])\s+|\n+", text) if p.strip()]
    if not parts:
        return []
    chunks, cur = ([parts[0]], "") if first_alone else ([], "")
    for p in (parts[1:] if first_alone else parts):
        if cur and len(cur) + 1 + len(p) > CHUNK_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def to_int16(samples):
    return (np.clip(samples, -1.0, 32767 / 32768) * 32768).astype("<i2")


def normalize(x, target_dbfs):
    """float32 mono at target_dbfs RMS, peaks at most PEAK_CEILING_DBFS."""
    if x.size == 0:
        return x, 0.0
    rms, peak = float(np.sqrt(np.mean(x ** 2))), float(np.abs(x).max())
    if rms < 1e-6:
        return x, 0.0
    gain = min(10 ** (target_dbfs / 20) / rms, 10 ** (PEAK_CEILING_DBFS / 20) / peak)
    return x * gain, 20 * np.log10(gain)


def wav_header(n_bytes=None):
    """16-bit mono WAV header; without a length, one that says "unknown" (0xFFFFFFFF)."""
    data = 0xFFFFFFFF if n_bytes is None else n_bytes
    riff = 0xFFFFFFFF if n_bytes is None else 36 + n_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16) +
            b"data" + struct.pack("<I", data))


# ── engines ─────────────────────────────────────────────────

class CliEngine:
    """llama-tts once per piece of text."""

    def pieces(self, text, lang, voice, seed, first_alone):
        for chunk in sentences(text, first_alone):
            with tempfile.TemporaryDirectory(prefix="llmctl-tts-") as tmp:
                out = Path(tmp) / "out.wav"
                cmd = [args.binary, "-m", args.model, "-mm", args.mmproj, "-p", chunk,
                       "--tts-lang", lang, "--tts-speaker-file", str(voice), "-o", str(out),
                       "--seed", str(seed), *args.extra]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                if r.returncode != 0 or not out.is_file():
                    tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                    raise RuntimeError("llama-tts failed: " + (tail[-1] if tail else f"exit {r.returncode}"))
                with wave.open(str(out)) as w:
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            yield pcm.astype(np.float32) / 32768.0


class ServerEngine:
    """llama-server with /tts, started as a child that dies with this process."""

    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cmd = [args.binary, "-m", args.model, "--mmproj", args.mmproj, "--host", "127.0.0.1",
               "--port", str(self.port), *args.extra]
        libc = ctypes.CDLL("libc.so.6", use_errno=True)

        def die_with_parent():          # PR_SET_PDEATHSIG: gone when we are, even on SIGKILL
            libc.prctl(1, signal.SIGTERM)
        log.info("engine: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, preexec_fn=die_with_parent)
        self.refs = {}
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} while loading")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server did not come up")

    def ref(self, voice):
        st = voice.stat().st_mtime
        cached = self.refs.get(voice)
        if not cached or cached[0] != st:
            self.refs[voice] = (st, base64.b64encode(voice.read_bytes()).decode())
        return self.refs[voice][1]

    def pieces(self, text, lang, voice, seed, first_alone):
        """float32 blocks as the engine streams them, piece by piece."""
        for chunk in sentences(text, first_alone):
            body = json.dumps({"input": chunk, "lang": lang, "speaker_ref_b64": self.ref(voice),
                               "response_format": "pcm", "stream": True, "seed": seed}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/tts", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                rest = b""
                while True:
                    block = r.read1(16384)
                    if not block:
                        break
                    block = rest + block
                    cut = len(block) - len(block) % 4
                    rest = block[cut:]
                    if cut:
                        yield np.frombuffer(block[:cut], dtype="<f4")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()


# ── API ─────────────────────────────────────────────────────

def error(status, message, kind="invalid_request_error"):
    return jsonify({"error": {"message": message, "type": kind}}), status


@app.get("/health")
def health():
    return jsonify({"status": "ok", "engine": args.engine, "voices": len(voices())})


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
    voice = known[name]
    if body.get("stream_format", "audio") != "audio":
        return error(400, "stream_format 'sse' is not supported; audio is streamed as it is")
    seed = body.get("seed", args.seed)
    if not isinstance(seed, int) or seed < -1:
        return error(400, "'seed' must be an integer, -1 for a random one")
    first_alone = bool(body.get("chunking", args.chunking))
    t0 = time.time()

    if body.get("stream", True):
        def gen():
            first, n = None, 0
            with gpu:
                if fmt == "wav":
                    yield wav_header()
                try:
                    for block in engine.pieces(text, lang, voice, seed, first_alone):
                        if first is None:
                            first = time.time() - t0
                        n += block.size
                        yield to_int16(block).tobytes()
                except Exception as e:           # headers are gone; the log has to say it
                    log.error("stream broke off: %s", e)
            log.info("voice=%s lang=%s chars=%d audio=%.1fs first_audio=%.2fs took=%.1fs (stream)",
                     name, lang, len(text), n / SAMPLE_RATE, first or 0, time.time() - t0)
        mime = "audio/wav" if fmt == "wav" else "audio/pcm"
        return Response(stream_with_context(gen()), mimetype=mime)

    try:
        with gpu:
            x = np.concatenate(list(engine.pieces(text, lang, voice, seed, first_alone)) or [np.zeros(0, np.float32)])
    except Exception as e:
        log.error("%s", e)
        return error(500, str(e), "server_error")
    dt = time.time() - t0
    gain = 0.0
    if args.loudness is not None:
        x, gain = normalize(x, args.loudness)
    pcm = to_int16(x).tobytes()
    log.info("voice=%s lang=%s chars=%d audio=%.1fs took=%.1fs (%.2fx real time) gain=%+.1f dB",
             name, lang, len(text), x.size / SAMPLE_RATE, dt, x.size / SAMPLE_RATE / dt if dt else 0, gain)
    if fmt == "pcm":
        return Response(pcm, mimetype="audio/pcm")
    return Response(wav_header(len(pcm)) + pcm, mimetype="audio/wav")


def main():
    global args, engine
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True, help="Qwen3-TTS Base GGUF")
    p.add_argument("--mmproj", required=True, help="its mmproj GGUF (speaker encoder, code predictor, vocoder)")
    p.add_argument("--voices", required=True, help="directory of reference recordings, one voice per file")
    p.add_argument("--binary", default="llama-tts",
                   help="llama-tts, or a llama-server that has /tts (the engine follows the name)")
    p.add_argument("--lang", default="de")
    p.add_argument("--default-voice", default="")
    p.add_argument("--alias", default="qwen3-tts")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--chunking", action=argparse.BooleanOptionalAction, default=True,
                   help="speak the first sentence on its own when a request does not say "
                        "(earlier first sound, but tone may shift between sentences)")
    p.add_argument("--seed", type=int, default=-1,
                   help="sampling seed when a request names none; -1 (default) = random every time")
    p.add_argument("--loudness", default="-20",
                   type=lambda v: None if v == "off" else float(v),
                   help="RMS level of every whole answer in dBFS, or off (default -20)")
    p.add_argument("extra", nargs="*", help="further llama-tts / llama-server arguments (after --)")
    args = p.parse_args()
    args.engine = "server" if os.path.basename(args.binary).startswith("llama-server") else "cli"
    log.info("model %s, voices in %s (%d), language %s, engine %s", os.path.basename(args.model),
             args.voices, len(voices()), args.lang, args.engine)
    engine = ServerEngine() if args.engine == "server" else CliEngine()

    def on_term(*_):
        if isinstance(engine, ServerEngine):
            engine.stop()
        os._exit(0)
    signal.signal(signal.SIGTERM, on_term)
    # llmctl waits for this line before it calls the slot up.
    log.info("tts_server listening on http://%s:%d", args.host, args.port)
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=4)
    except ImportError:
        app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
