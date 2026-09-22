import base64
import hmac
import ipaddress
import json
import logging
import os
import queue
import socket
import threading
import urllib.parse
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
# Backends that take images only inline (halogen refuses http(s) URLs): fetch
# such images here and pass them on as data: URLs.
INLINE_IMAGE_URLS = os.environ.get('LLM_INLINE_IMAGE_URLS', '') == '1'
IMAGE_MAX_BYTES = 20 * 1024 * 1024
IMAGE_FETCH_TIMEOUT = 20
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


# Speech recognition models write their result as "language German<asr_text>…"
# (Qwen3-ASR), and llama-server's /v1/audio/transcriptions passes that on. A
# client wants the text; the language goes into its own field, where OpenAI's
# verbose_json has it.
ASR_PREFIX = re.compile(r"^\s*language\s+([A-Za-z]+)\s*<asr_text>")


def _split_transcript(text):
    m = ASR_PREFIX.match(text or "")
    if not m:
        return text, None
    lang = m.group(1)
    return text[m.end():], (None if lang.lower() == "none" else lang)


def _clean_transcription(resp):
    """A transcription answer without the model's language header."""
    ctype = resp.headers.get("Content-Type", "")
    try:
        if "text/event-stream" not in ctype:
            body = resp.content
            try:
                obj = json.loads(body)
                if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                    obj["text"], lang = _split_transcript(obj["text"])
                    if lang:
                        obj["language"] = lang
                    body = json.dumps(obj, ensure_ascii=False).encode()
            except ValueError:
                pass
            yield body
            return
        # Streamed: hold the deltas back until the header is complete, then pass
        # the rest on as it comes.
        held, open_ = "", False
        resp.encoding = "utf-8"          # requests would guess ISO-8859-1 for event streams
        for line in resp.iter_lines(decode_unicode=True):
            if not line.startswith("data: "):
                yield (line + "\n").encode()
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                yield (line + "\n").encode()
                continue
            if ev.get("type") == "transcript.text.delta" and not open_:
                held += ev.get("delta", "")
                if "<asr_text>" not in held and len(held) < 64:
                    continue
                rest, lang = _split_transcript(held)
                open_ = True
                if lang:
                    yield f"data: {json.dumps({'type': 'transcript.language', 'language': lang})}\n\n".encode()
                if not rest:
                    continue
                ev["delta"] = rest
            elif ev.get("type") == "transcript.text.done" and isinstance(ev.get("text"), str):
                ev["text"], lang = _split_transcript(ev["text"])
                if lang:
                    ev["language"] = lang
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n".encode()
    finally:
        resp.close()
        _sem.release()


class ImageFetchError(Exception):
    pass


def _check_public_host(url):
    """In public mode the proxy must not become a way into this machine or its
    network: a token holder could otherwise have it fetch http://127.0.0.1:8001
    or a LAN address. Every address the host resolves to has to be global."""
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise ImageFetchError(f"no host in image URL {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ImageFetchError(f"cannot resolve {host}: {e}")
    for info in infos:
        addr = ipaddress.ip_address(info[4][0].split("%")[0])
        if not addr.is_global:
            raise ImageFetchError(f"image host {host} resolves to a non-public address ({addr})")


def _fetch_image(url):
    """Download an http(s) image and return it as a data: URL."""
    for _ in range(5):                  # redirects, each hop checked on its own
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise ImageFetchError(f"unsupported image URL scheme: {url[:40]!r}")
        if TOKENS:
            _check_public_host(url)
        try:
            resp = requests.get(url, stream=True, timeout=IMAGE_FETCH_TIMEOUT, allow_redirects=False,
                                headers={"User-Agent": "llmctl-proxy/1 (image fetch)"})
        except requests.RequestException as e:
            raise ImageFetchError(f"cannot fetch image {url}: {e}")
        with resp:
            if resp.is_redirect:
                url = urllib.parse.urljoin(url, resp.headers.get("Location", ""))
                continue
            if resp.status_code != 200:
                raise ImageFetchError(f"image {url} answered HTTP {resp.status_code}")
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                raise ImageFetchError(f"{url} is not an image (Content-Type {ctype or 'missing'})")
            data = bytearray()
            for chunk in resp.iter_content(65536):
                data += chunk
                if len(data) > IMAGE_MAX_BYTES:
                    raise ImageFetchError(f"image {url} is larger than {IMAGE_MAX_BYTES // (1024 * 1024)} MiB")
        log.info("Inlined image %s (%s, %d KiB)", url, ctype, len(data) // 1024)
        return f"data:{ctype};base64,{base64.b64encode(bytes(data)).decode()}"
    raise ImageFetchError(f"too many redirects for image {url}")


def _inline(url, cache):
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return url
    if url not in cache:
        cache[url] = _fetch_image(url)
    return cache[url]


def inline_image_urls(data):
    """Replace http(s) image URLs by data: URLs, in Chat Completions messages
    (image_url parts), Responses input (input_image items) and Messages content
    (image blocks with a url source, also inside tool results). Raises
    ImageFetchError when an image cannot be fetched."""
    cache = {}

    def parts(content):
        if not isinstance(content, list):
            return
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                src = part.get("source")
                if isinstance(src, dict) and src.get("type") == "url":
                    inlined = _inline(src.get("url"), cache)
                    if inlined.startswith("data:"):
                        head, b64 = inlined.split(",", 1)
                        part["source"] = {"type": "base64", "data": b64,
                                          "media_type": head[5:].split(";")[0]}
            elif part.get("type") == "tool_result":
                parts(part.get("content"))
            elif part.get("type") == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict):
                    iu["url"] = _inline(iu.get("url"), cache)
                else:
                    part["image_url"] = _inline(iu, cache)
            elif part.get("type") == "input_image":
                part["image_url"] = _inline(part.get("image_url"), cache)

    for msg in data.get("messages") or []:
        if isinstance(msg, dict):
            parts(msg.get("content"))
    inp = data.get("input")
    if isinstance(inp, list):
        parts(inp)
        for item in inp:
            if isinstance(item, dict):
                parts(item.get("content"))


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
    if INLINE_IMAGE_URLS:
        try:
            inline_image_urls(chat)
        except ImageFetchError as e:
            _sem.release()
            log.warning("Rejected request from %s: %s", _client_addr(), e)
            return _json_response(400, ac.error_body(400, str(e)))
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

    if INLINE_IMAGE_URLS and isinstance(data, dict):
        try:
            inline_image_urls(data)
        except ImageFetchError as e:
            _sem.release()
            log.warning("Rejected request from %s: %s", _client_addr(), e)
            return _deny(400, str(e))

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
    if path == "v1/audio/transcriptions" and resp.status_code == 200:
        resp_headers = {k: v for k, v in dict(resp_headers).items() if k.lower() != "content-length"}
        return Response(_clean_transcription(resp), status=200, headers=resp_headers)
    return Response(_relay(resp), status=resp.status_code, headers=resp_headers)


if __name__ == '__main__':
    log.info("Backend: %s  Proxy: %s:%d  Timeout: %ds", TARGET_URL, PROXY_HOST, PROXY_PORT, REQUEST_TIMEOUT)
    if TRANSLATE_MESSAGES:
        log.info("Translating /v1/messages to /v1/chat/completions (backend has no Messages API)")
    if MAX_TOKENS_CAP:
        log.info("Token budgets above %d are clamped to it", MAX_TOKENS_CAP)
    if INLINE_IMAGE_URLS:
        log.info("Fetching http(s) image URLs and passing them on inline%s",
                 " (public hosts only)" if TOKENS else "")
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
