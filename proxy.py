import hmac
import json
import logging
import os
import threading
import requests
from flask import Flask, request, Response
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
TARGET_URL = os.environ.get('LLM_BACKEND_URL', 'http://localhost:8001')
PROXY_HOST  = os.environ.get('LLM_PROXY_HOST', '127.0.0.1')
PROXY_PORT  = int(os.environ.get('LLM_PROXY_PORT', '8081'))
REQUEST_TIMEOUT = 300

# Hardened mode. run.sh sets LLM_TOKEN_FILE only for `start --public`; when it is
# set, every request must carry a known token and may only reach an allowlisted
# path, and concurrency/body size are capped. Unset (the LAN/localhost default)
# leaves the proxy a plain pass-through, exactly as before.
TOKEN_FILE      = os.environ.get('LLM_TOKEN_FILE', '')
MAX_CONCURRENCY = int(os.environ.get('LLM_MAX_CONCURRENCY', '4'))
MAX_BODY_BYTES  = int(os.environ.get('LLM_MAX_BODY_BYTES', str(32 * 1024 * 1024)))

# Endpoints a public client legitimately needs. Everything else llama-server
# offers — GET /slots (leaks other clients' prompts), /props, the Web UI,
# /metrics — stays unreachable through the proxy.
ALLOWED_PATHS = frozenset([
    "/health",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/messages",
    "/v1/messages/count_tokens",
])

# The one exception to the allowlist: POST /slots/{id}?action=erase frees a
# slot's KV cache, which a remote client legitimately needs. The sibling actions
# on the same path write files on this machine (`save`/`restore`, enabled by the
# --slot-save-path that erase itself requires), so the action is matched
# explicitly rather than the path.
_SLOT_PATH = re.compile(r"^slots/\d+$")

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
])


def _load_tokens(path):
    """Read one token per line; blank lines and # comments are ignored."""
    tokens = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.append(line)
    if not tokens:
        raise SystemExit(f"Token file {path} contains no tokens — refusing to start in public mode")
    return tokens


TOKENS = _load_tokens(TOKEN_FILE) if TOKEN_FILE else []
_sem = threading.BoundedSemaphore(MAX_CONCURRENCY)


def _client_addr():
    """Real client IP for the log: behind the VPS the peer is the local tunnel."""
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "-")


def _presented_token():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _authorized():
    presented = _presented_token()
    if not presented:
        return False
    # Compare against every token so the time taken does not reveal which
    # prefix matched.
    ok = False
    for token in TOKENS:
        if hmac.compare_digest(presented, token):
            ok = True
    return ok


def _path_allowed(path):
    """Public mode: allowlisted path, or a KV erase on one slot."""
    if f"/{path}" in ALLOWED_PATHS:
        return True
    # Exactly one action parameter: llama-server copies the query multimap into a
    # std::map, so the *last* occurrence wins there, while Werkzeug reports the
    # first — ?action=erase&action=save would pass here and save on the server.
    return (request.method == "POST"
            and _SLOT_PATH.match(path) is not None
            and request.args.getlist("action") == ["erase"])


def _deny(status, message):
    return Response(json.dumps({"error": message}), status=status,
                    content_type="application/json")


def _filter_headers(headers):
    return {k: v for k, v in headers if k.lower() not in _HOP_BY_HOP and k.lower() != "host"}


def optimize_prompt(content):
    if not isinstance(content, str):
        return content
    return re.sub(r"(Current time:|Date:|Time:)\s+[^\n]+", r"\1 CONSTANT_TIME", content, flags=re.IGNORECASE)


def _relay(resp):
    """Stream the backend response, then release the concurrency slot."""
    try:
        for chunk in resp.iter_content(chunk_size=1024):
            yield chunk
    finally:
        resp.close()
        _sem.release()


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
def proxy(path):
    if TOKENS:
        if not _authorized():
            log.warning("Rejected request from %s: bad or missing token (%s /%s)",
                        _client_addr(), request.method, path)
            return _deny(401, "unauthorized")
        if not _path_allowed(path):
            log.warning("Rejected request from %s: path not allowed (/%s)", _client_addr(), path)
            return _deny(404, "not found")
        if request.content_length and request.content_length > MAX_BODY_BYTES:
            return _deny(413, "request body too large")

    # Forward the query string verbatim; llama-server reads parameters from it
    # (?action=erase on /slots/{id}, ?fail_on_no_slot=1 on /slots).
    qs = request.query_string.decode()
    url = f"{TARGET_URL}/{path}" + (f"?{qs}" if qs else "")
    data = request.get_json(silent=True)

    if data and "messages" in data:
        for msg in data["messages"]:
            if msg.get("role") == "user" and "content" in msg:
                msg["content"] = optimize_prompt(msg["content"])

    headers = _filter_headers(request.headers)

    # One generation can occupy the GPU for minutes, so the cap is on concurrent
    # requests, not on request rate. Refuse rather than queue: a queued client
    # just times out further down the line.
    if not _sem.acquire(blocking=False):
        log.warning("Rejected request from %s: %d concurrent requests in flight",
                    _client_addr(), MAX_CONCURRENCY)
        return _deny(429, "too many concurrent requests")

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
    except Exception as e:
        _sem.release()
        log.error("Proxy error: %s", e)
        return Response(json.dumps({"error": str(e)}), status=500, content_type="application/json")

    # From here on _relay owns the slot and releases it when the stream ends.
    resp_headers = _filter_headers(resp.headers.items())
    return Response(_relay(resp), status=resp.status_code, headers=resp_headers)


if __name__ == '__main__':
    log.info("Backend: %s  Proxy: %s:%d", TARGET_URL, PROXY_HOST, PROXY_PORT)
    if TOKENS:
        log.info("Public mode: %d token(s) from %s, max %d concurrent requests, "
                 "body limit %d MiB, %d allowlisted paths",
                 len(TOKENS), TOKEN_FILE, MAX_CONCURRENCY,
                 MAX_BODY_BYTES // (1024 * 1024), len(ALLOWED_PATHS))
    else:
        log.info("Open mode: no authentication (LLM_TOKEN_FILE unset)")
    try:
        from waitress import serve
        serve(app, host=PROXY_HOST, port=PROXY_PORT)
    except ImportError:
        log.warning("waitress not installed, falling back to Flask dev server")
        app.run(port=PROXY_PORT, host=PROXY_HOST)
