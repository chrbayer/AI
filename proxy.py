import json
import logging
import requests
from flask import Flask, request, Response
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
TARGET_URL = "http://localhost:8001"
REQUEST_TIMEOUT = 300

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
        return Response(resp.iter_content(chunk_size=1024), status=resp.status_code, headers=resp_headers)
    except Exception as e:
        log.error("Proxy error: %s", e)
        return Response(json.dumps({"error": str(e)}), status=500, content_type="application/json")

if __name__ == '__main__':
    try:
        from waitress import serve
        log.info("Starting on port 8081 (waitress)")
        serve(app, host='127.0.0.1', port=8081)
    except ImportError:
        log.warning("waitress not installed, falling back to Flask dev server")
        app.run(port=8081, host='127.0.0.1')
