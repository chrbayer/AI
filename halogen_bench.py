#!/usr/bin/env python3
"""Prefill and decode speed of a running halogen server — `llmctl bench` for
halogen models, where llama-bench cannot read the checkpoint.

Prefill: time to the first token for prompts of several lengths. Each prompt
starts with a unique random prefix, so the prompt cache cannot help.
Decode: tokens/s after the first token, streamed, greedy with the MTP draft on,
once on prose and once on code (code drafts better: it copies its own context).

    halogen_bench.py [--slot N | --url URL] [--record FILE ...] [sizes ...]

The server is measured on its slot port (8000+N), not through the proxy, so the
numbers are the server's own. With --record the run is appended to FILE as one
JSON line in the shape of llmctl's llama-bench entries, plus a `results` list.
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.request

WORDS = ("the of and to in is that for it as was with be by on not he this are or his "
         "from at which but have an they you were her she there been one all we their "
         "memory river engine window garden theory signal market winter number system "
         "history station village kernel bridge letter summer silver process fabric").split()

DECODE_TASKS = [
    ("prose", "Schreibe einen ausführlichen Essay (mindestens 800 Wörter) über die "
              "Geschichte der Eisenbahn in Europa."),
    ("code", "Schreibe ein vollständiges Python-Modul mit einer LRU-Cache-Klasse, "
             "Typannotationen, Docstrings und ausführlichen pytest-Tests."),
]


def filler(n_words, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 17 == 16:
            out.append(".\n" if i % 85 == 84 else ".")
    return " ".join(out)


def nonce():
    return "".join(random.choices(string.ascii_lowercase, k=12))


class Bench:
    def __init__(self, base):
        self.url = f"{base}/v1/chat/completions"

    def run(self, messages, max_tokens):
        """One streamed greedy request; returns (seconds to first token,
        seconds after it, usage)."""
        body = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        t_first = None
        usage = {}
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    if t_first is None and ch.get("delta", {}).get("content"):
                        t_first = time.perf_counter()
        t_end = time.perf_counter()
        t_first = t_first or t_end
        return t_first - t0, t_end - t_first, usage

    def prefill(self, target_tokens):
        n = nonce()
        # ~1.3 tokens per filler word; the real count comes back in usage.
        text = f"[{n}]\n" + filler(int(target_tokens / 1.3), n)
        ttft, _, usage = self.run([{"role": "user", "content": text + "\n\nAntworte nur mit: OK"}], 4)
        tokens = usage.get("prompt_tokens", 0)
        return {"test": "prefill", "tokens": tokens, "seconds": round(ttft, 3),
                "tok_s": round(tokens / ttft, 1) if ttft else None}

    def decode(self, task, prompt, max_tokens):
        ttft, gen, usage = self.run([{"role": "user", "content": f"[{nonce()}] {prompt}"}], max_tokens)
        tokens = usage.get("completion_tokens", 0)
        return {"test": f"decode-{task}", "tokens": tokens, "seconds": round(gen, 3),
                "tok_s": round((tokens - 1) / gen, 1) if gen else None, "ttft": round(ttft, 3)}


def main():
    ap = argparse.ArgumentParser(description="prefill and decode speed of a running halogen server")
    ap.add_argument("--slot", type=int, default=1, help="llmctl slot (server port 8000+N), default 1")
    ap.add_argument("--url", help="server base URL instead, e.g. http://127.0.0.1:8001")
    ap.add_argument("--decode-tokens", type=int, default=1024, help="tokens per decode run (default 1024)")
    ap.add_argument("--record", metavar="FILE", help="append the run to FILE as one JSON line")
    ap.add_argument("--model-name", default="", help="for --record: the models.conf name")
    ap.add_argument("--model", default="", help="for --record: the checkpoint path")
    ap.add_argument("--binary", default="", help="for --record: the container image")
    ap.add_argument("sizes", nargs="*", type=int,
                    help="prompt sizes in tokens (default 1000 8000 32000 64000)")
    args = ap.parse_args()

    base = (args.url or f"http://127.0.0.1:{8000 + args.slot}").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            health = json.load(r)
    except Exception as e:
        sys.exit(f"no server answers on {base} ({e}) — is the slot running? llmctl status")
    if health.get("busy") or health.get("in_flight"):
        print("  NOTE: the server is busy with other requests; these numbers will be low.", flush=True)

    sizes = args.sizes or [1000, 8000, 32000, 64000]
    b = Bench(base)
    started = int(time.time() * 1000)
    results, error = [], None
    try:
        print(f"    server {base}, context {health.get('context')}, warmup ...", flush=True)
        b.run([{"role": "user", "content": "Hallo"}], 8)
        print("    prefill (input):", flush=True)
        for s in sizes:
            r = b.prefill(s)
            results.append(r)
            print(f"      {r['tokens']:>7} tokens  {r['seconds']:8.2f} s  {r['tok_s']:8.1f} tok/s", flush=True)
        print("    decode (output), greedy, MTP on:", flush=True)
        for task, prompt in DECODE_TASKS:
            r = b.decode(task, prompt, args.decode_tokens)
            results.append(r)
            print(f"      {task:<6} {r['tokens']:>5} tokens  {r['seconds']:6.2f} s  {r['tok_s']:6.1f} tok/s"
                  f"   (first token after {r['ttft']:.2f} s)", flush=True)
    except Exception as e:                     # recorded, like a failed llama-bench run
        error = str(e)
        print(f"    FAILED: {e}", flush=True)

    if args.record:
        entry = {
            "timestamp": str(started),
            "backend": "halogen",
            "binary": args.binary,
            "model": args.model,
            "model_name": args.model_name,
            "env": {},
            "status": "error" if error else "ok",
            "exit_code": 1 if error else None,
            "stdout": error or "",
            "server": {k: health.get(k) for k in ("context", "slots", "kv_pool_positions", "version")},
            "results": results,
        }
        with open(args.record, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    sys.exit(1 if error else 0)


if __name__ == "__main__":
    main()
