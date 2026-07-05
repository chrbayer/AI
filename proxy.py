import json
import logging
import os
import threading
import time
import requests
from flask import Flask, request, Response
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
TARGET_URL = os.environ.get('LLM_BACKEND_URL', 'http://localhost:8001')
PROXY_PORT  = int(os.environ.get('LLM_PROXY_PORT', '8081'))
REQUEST_TIMEOUT = 300

# ── Prompt-cache effectiveness accounting ──────────────────────────────
# llama-server reports, per completion, how many prompt tokens were served
# from cache (usage.prompt_tokens_details.cached_tokens / timings.cache_n).
# We tee the response, tally it, and log per-request + cumulative hit rate.
# Optional JSONL dump via LLM_STATS_FILE for later analysis.
_STATS_FILE = os.environ.get('LLM_STATS_FILE')
_stats_lock = threading.Lock()
_stats = {"requests": 0, "prompt": 0, "cached": 0}


def _extract_cache_stats(body_bytes, content_type):
    """Return (prompt_tokens, cached_tokens) from a completion response, or None."""
    text = body_bytes.decode('utf-8', 'replace')
    objs = []
    if 'text/event-stream' in (content_type or ''):
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('data:'):
                payload = line[5:].strip()
                if payload and payload != '[DONE]':
                    try:
                        objs.append(json.loads(payload))
                    except ValueError:
                        pass
    else:
        try:
            objs.append(json.loads(text))
        except ValueError:
            return None
    prompt = cached = None
    for o in objs:  # last chunk carrying the numbers wins
        if not isinstance(o, dict):
            continue
        u = o.get('usage')
        if isinstance(u, dict) and u.get('prompt_tokens') is not None:
            prompt = u['prompt_tokens']
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', cached)
        t = o.get('timings')
        if isinstance(t, dict) and t.get('cache_n') is not None:
            cached = t['cache_n']
            if prompt is None and t.get('prompt_n') is not None:
                prompt = t['prompt_n'] + t['cache_n']
    if prompt is None:
        return None
    return prompt, (cached or 0)


def _record_cache_stats(body_bytes, content_type, path):
    res = _extract_cache_stats(body_bytes, content_type)
    if not res:
        return
    prompt, cached = res
    if prompt <= 0:
        return
    with _stats_lock:
        _stats["requests"] += 1
        _stats["prompt"] += prompt
        _stats["cached"] += cached
        n, tp, tc = _stats["requests"], _stats["prompt"], _stats["cached"]
    log.info("cache: req prompt=%d cached=%d (%.1f%%) | session reqs=%d prompt=%d cached=%d (%.1f%%)",
             prompt, cached, 100.0 * cached / prompt,
             n, tp, tc, (100.0 * tc / tp) if tp else 0.0)
    if _STATS_FILE:
        try:
            with open(_STATS_FILE, 'a') as f:
                f.write(json.dumps({"t": time.time(), "path": path,
                                    "prompt": prompt, "cached": cached}) + "\n")
        except OSError:
            pass

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
])

def _filter_headers(headers):
    return {k: v for k, v in headers if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}

def optimize_prompt(content):
    if not isinstance(content, str):
        return content
    return re.sub(r"(Current time:|Date:|Time:)\s+[^\n]+", r"\1 CONSTANT_TIME", content, flags=re.IGNORECASE)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
def proxy(path):
    url = f"{TARGET_URL}/{path}"
    data = request.get_json(silent=True)

    if data and "messages" in data:
        for msg in data["messages"]:
            if msg.get("role") == "user" and "content" in msg:
                msg["content"] = optimize_prompt(msg["content"])

    headers = _filter_headers(request.headers)

    try:
        if data is not None:
            resp = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                json=data,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        else:
            resp = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                data=request.get_data(),
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        resp_headers = _filter_headers(resp.headers.items())
        ctype = resp.headers.get('Content-Type', '')
        # Only tee completion endpoints (they carry the cache counters); pass
        # everything else straight through untouched.
        if 'completion' in path or 'responses' in path:
            def tee():
                chunks = []
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        chunks.append(chunk)
                        yield chunk
                finally:
                    try:
                        _record_cache_stats(b"".join(chunks), ctype, path)
                    except Exception as e:  # never let accounting break the proxy
                        log.debug("cache-stats parse failed: %s", e)
            return Response(tee(), status=resp.status_code, headers=resp_headers)
        return Response(resp.iter_content(chunk_size=1024), status=resp.status_code, headers=resp_headers)
    except Exception as e:
        log.error("Proxy error: %s", e)
        return Response(json.dumps({"error": str(e)}), status=500, content_type="application/json")

if __name__ == '__main__':
    log.info("Backend: %s  Proxy port: %d", TARGET_URL, PROXY_PORT)
    try:
        from waitress import serve
        serve(app, host='127.0.0.1', port=PROXY_PORT)
    except ImportError:
        log.warning("waitress not installed, falling back to Flask dev server")
        app.run(port=PROXY_PORT, host='127.0.0.1')
