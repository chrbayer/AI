import requests
from flask import Flask, request, Response
import re

app = Flask(__name__)
TARGET_URL = "http://localhost:8001" # llama-server Port

def optimize_prompt(content):
    if not isinstance(content, str): return content
    # Normalisiert Zeitstempel für maximales Caching
    return re.sub(r"(Current time:|Date:|Time:)\s+[^\n]+", r"\1 CONSTANT_TIME", content, flags=re.IGNORECASE)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{TARGET_URL}/{path}"
    data = request.get_json(silent=True)

    # Prompt-Optimierung
    if data and "messages" in data:
        for msg in data["messages"]:
            if "content" in msg:
                msg["content"] = optimize_prompt(msg["content"])

    # Header-Forwarding (Wichtig für Claude Code)
    headers = {k: v for k, v in request.headers if k.lower() != 'host'}

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            json=data,
            stream=True
        )
        return Response(resp.iter_content(chunk_size=1024), status=resp.status_code, headers=dict(resp.headers))
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500)

if __name__ == '__main__':
    app.run(port=8081, host='127.0.0.1')

