#!/usr/bin/env python3
"""Turn the bundled workflows into ComfyUI's API format, for images_server.py.

    comfyui/export_api.py [PORT]      (a running ComfyUI, default 8009)

The UI format a workflow is saved in (nodes, links, subgraphs, widget values
by position) is not what POST /prompt takes, and only ComfyUI's own frontend
knows how to turn one into the other — it is what "Export (API)" does. So this
runs the frontend: Chrome without a window and with a throwaway profile opens
the UI, loads each image workflow and writes app.graphToPrompt() to
comfyui/api/<workflow>.json. Run it when a workflow changes; nothing at run
time needs a browser.
"""
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8009
UI = f"http://127.0.0.1:{PORT}/"
CDP_PORT = 9333


async def cdp(ws, method, params=None, _id=[0]):
    _id[0] += 1
    await ws.send(json.dumps({"id": _id[0], "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _id[0]:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["result"]


async def evaluate(ws, expr):
    r = await cdp(ws, "Runtime.evaluate", {"expression": expr, "awaitPromise": True,
                                            "returnByValue": True})
    if "exceptionDetails" in r:
        raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r))
    return r["result"].get("value")


async def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or sys.exit("no Chrome")
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
                             f"--user-data-dir={profile}", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            sys.exit("Chrome did not come up")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            await cdp(ws, "Page.navigate", {"url": UI})
            for _ in range(120):
                if await evaluate(ws, "!!(window.app && window.app.graph && window.app.graphToPrompt)"):
                    break
                await asyncio.sleep(0.5)
            else:
                sys.exit("the ComfyUI frontend did not load")
            await asyncio.sleep(2)
            for path in sorted((ROOT / "workflows").glob("*.json")):
                wf = json.loads(path.read_text())
                if any("TTS" in n.get("type", "") for n in wf.get("nodes", [])):
                    continue                                   # speech has its own server
                if any(n.get("type", "") in ("Save3DAdvanced", "SaveGLB", "SaveWEBM", "SaveAnimatedWEBP")
                       for n in wf.get("nodes", [])):
                    continue                                   # 3D: no image to answer with
                # music goes to api/audio/, for /v1/audio/music, video to api/video/, for
                # `llmctl musicvideo`; the image list stays images
                audio = any(n.get("type", "").startswith("SaveAudio") for n in wf.get("nodes", []))
                video = any(n.get("type", "") == "SaveVideo" for n in wf.get("nodes", []))
                if any(n.get("type", "") == "LlmctlArtifactCheck" for n in wf.get("nodes", [])):
                    continue                                   # it calls the image API itself
                prompt = await evaluate(ws, f"""(async () => {{
                    await app.loadGraphData({json.dumps(wf)}, true, true, {json.dumps(path.stem)});
                    const p = await app.graphToPrompt();
                    return p.output;
                }})()""")
                # A workflow with switches gives one API workflow per variant (variants.py).
                for name, settings in variants.VARIANTS.get(path.stem, [(path.stem, None)]):
                    graph = variants.resolve(prompt, settings) if settings is not None else prompt
                    name = name.removeprefix("audio/")
                    out = ROOT / "api" / ("audio" if audio else "video" if video else "") / f"{name}.json"
                    out.parent.mkdir(exist_ok=True)
                    out.write_text(json.dumps(graph, indent=1, ensure_ascii=False) + "\n")
                    print(f"{len(graph):3d} nodes  {out.relative_to(ROOT.parent)}")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


asyncio.run(main())
