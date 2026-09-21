import hmac
import json
import logging
import os
import queue
import threading
import requests
from flask import Flask, request, Response
import re

import anthropic_compat as ac

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
TARGET_URL = os.environ.get('LLM_BACKEND_URL', 'http://localhost:8001')
PROXY_HOST  = os.environ.get('LLM_PROXY_HOST', '127.0.0.1')
PROXY_PORT  = int(os.environ.get('LLM_PROXY_PORT', '8081'))
# Seconds the backend may stay silent — for a streamed answer, that includes the
# whole prefill before its first token. llmctl raises it for halogen, whose
# prefill of a full 256K prompt alone takes about four minutes.
REQUEST_TIMEOUT = int(os.environ.get('LLM_PROXY_TIMEOUT', '300'))

# Backends without /v1/messages (halogen): answer it here by translating to and
# from /v1/chat/completions (see anthropic_compat.py). Off = pass-through, as
# llama-server speaks the Messages API itself.
TRANSLATE_MESSAGES = os.environ.get('LLM_TRANSLATE_MESSAGES', '') == '1'
# Largest token budget the backend accepts. halogen refuses a request above its
# cap outright (HTTP 400) instead of shortening it, and Claude Code asks for
# 32000 as a matter of course; clamping here keeps such requests alive.
MAX_TOKENS_CAP = int(os.environ.get('LLM_MAX_TOKENS_CAP', '0')) or None
# While the backend prefills, a streamed Messages answer sends a ping this often,
# so neither the client nor anything in between takes the silence for a hang.
PING_INTERVAL = 10

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
    "/v1/responses",
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


_STAMP = re.compile(r"(Current time:|Date:|Time:)\s+[^\n]+", flags=re.IGNORECASE)


def optimize_prompt(content):
    """Replace time/date stamps in a string, or in the text parts of a list of
    content parts / blocks (OpenAI and Anthropic shapes alike)."""
    if isinstance(content, str):
        return _STAMP.sub(r"\1 CONSTANT_TIME", content)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                part["text"] = _STAMP.sub(r"\1 CONSTANT_TIME", part["text"])
    return content


def normalize_request(data):
    """Stamps in user and system turns, and in a Messages request's system."""
    for msg in data.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "system") and "content" in msg:
            msg["content"] = optimize_prompt(msg["content"])
    if "system" in data:
        data["system"] = optimize_prompt(data["system"])


def clamp_max_tokens(data):
    if not MAX_TOKENS_CAP:
        return
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if isinstance(data.get(key), int) and data[key] > MAX_TOKENS_CAP:
            data[key] = MAX_TOKENS_CAP


def _relay(resp):
    """Stream the backend response, then release the concurrency slot."""
    try:
        for chunk in resp.iter_content(chunk_size=1024):
            yield chunk
    finally:
        resp.close()
        _sem.release()


def _json_response(status, body):
    return Response(json.dumps(body, ensure_ascii=False), status=status,
                    content_type="application/json")


def _relay_messages_stream(resp, model):
    """Translate a Chat Completions stream into a Messages stream, pinging while
    the backend is silent, then release the concurrency slot."""
    tr = ac.StreamTranslator(model)
    lines = queue.Queue()

    def pump():
        try:
            for line in resp.iter_lines():
                lines.put(line)
        except Exception as e:          # closed under us, or the backend died
            lines.put(e)
        finally:
            lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        yield tr.start()
        while True:
            try:
                item = lines.get(timeout=PING_INTERVAL)
            except queue.Empty:
                yield ac.sse("ping", {"type": "ping"})
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                log.error("Backend stream failed: %s", item)
                yield ac.sse("error", ac.error_body(502, f"backend stream failed: {item}"))
                return
            line = item.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("error"):
                yield ac.sse("error", ac.error_body(500, ac.openai_error_message(payload)))
                return
            yield tr.feed(chunk)
        yield tr.end()
    finally:
        resp.close()
        _sem.release()


def _messages(path, data):
    """/v1/messages and /v1/messages/count_tokens for a backend that only
    speaks Chat Completions. Owns the concurrency slot the caller acquired."""
    if not isinstance(data, dict):
        _sem.release()
        return _json_response(400, ac.error_body(400, "request body must be a JSON object"))
    if path == "v1/messages/count_tokens":
        _sem.release()
        return _json_response(200, ac.estimate_tokens(data))

    model = data.get("model") or "unknown"
    chat = ac.messages_to_chat(data, MAX_TOKENS_CAP)
    stream = bool(chat.get("stream"))
    try:
        resp = requests.post(f"{TARGET_URL}/v1/chat/completions", json=chat, stream=stream,
                             timeout=REQUEST_TIMEOUT)
    except Exception as e:
        _sem.release()
        log.error("Proxy error: %s", e)
        return _json_response(502, ac.error_body(502, str(e)))

    if resp.status_code >= 400:
        try:
            message = ac.openai_error_message(resp.text)
        finally:
            resp.close()
            _sem.release()
        log.warning("Backend refused a translated request (%d): %s", resp.status_code, message)
        return _json_response(resp.status_code, ac.error_body(resp.status_code, message))

    if stream:
        return Response(_relay_messages_stream(resp, model), status=200,
                        content_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    try:
        body = ac.chat_to_message(resp.json(), model)
    except ValueError as e:
        return _json_response(502, ac.error_body(502, f"backend sent no JSON: {e}"))
    finally:
        resp.close()
        _sem.release()
    return _json_response(200, body)


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

    if isinstance(data, dict):
        normalize_request(data)
        clamp_max_tokens(data)

    headers = _filter_headers(request.headers)

    # One generation can occupy the GPU for minutes, so the cap is on concurrent
    # requests, not on request rate. Refuse rather than queue: a queued client
    # just times out further down the line.
    if not _sem.acquire(blocking=False):
        log.warning("Rejected request from %s: %d concurrent requests in flight",
                    _client_addr(), MAX_CONCURRENCY)
        return _deny(429, "too many concurrent requests")

    if TRANSLATE_MESSAGES and request.method == "POST" and path in ("v1/messages", "v1/messages/count_tokens"):
        return _messages(path, data)

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
    log.info("Backend: %s  Proxy: %s:%d  Timeout: %ds", TARGET_URL, PROXY_HOST, PROXY_PORT, REQUEST_TIMEOUT)
    if TRANSLATE_MESSAGES:
        log.info("Translating /v1/messages to /v1/chat/completions (backend has no Messages API)")
    if MAX_TOKENS_CAP:
        log.info("Token budgets above %d are clamped to it", MAX_TOKENS_CAP)
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
