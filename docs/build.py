#!/usr/bin/env python3
"""The llmctl gallery: examples and results as HTML pages (#53).

docs/examples.json lists sections and their examples — what was made, how, and
from which files. For each picture, video or sound the ComfyUI graph stored in
the file (PNG text chunk, MP4/FLAC tag) is read and shown: models, prompts,
seeds, steps, size — and offered as the API graph to run again.

    build.py --local [--out DIR]     every example, the media in full, where they
                                     lie (default ~/.local/share/llmctl/docs)
    build.py --public                the examples marked public, their media made
                                     small into docs/media/, the pages into docs/
                                     (GitHub Pages); refused above --limit MB

A section with "de" (title, summary, intro; its examples' title and text, its
media's caption_de) is also built in German, as <id>.de.html. A section with
"help": true is Darkroom's help as well: --public writes it, in English and
German, into docs/help/ — the page without the gallery around it, its media
under /darkroom/help/media/ — and lists the media it needs in
docs/help/media.txt; `make install` takes those along, darkroom_server.py
serves them at /darkroom/help.

Media are named by a root and a path: out:video/x.mp4 (ComfyUI's output),
in:… (its input), bench:… (~/.local/share/llmctl/benchmarks), repo:… (here).
"""
import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SHARE = Path.home() / ".local/share/llmctl"
ROOTS = {"out": SHARE / "comfyui/output", "in": SHARE / "comfyui/input", "bench": SHARE / "benchmarks", "repo": REPO}
IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO = {".mp4", ".webm", ".mov"}
AUDIO = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}
MODEL = {".glb"}


def resolve(ref):
    root, _, rest = ref.partition(":")
    if root not in ROOTS:
        sys.exit(f"unknown root in {ref!r}: {', '.join(ROOTS)}")
    return ROOTS[root] / rest


# --- what a file says about how it was made --------------------------------

def graph_of(path):
    """The ComfyUI API graph stored in a file, or None."""
    try:
        if path.suffix.lower() == ".png":
            from PIL import Image
            text = Image.open(path).info.get("prompt")
        elif path.suffix.lower() in VIDEO | AUDIO:
            tags = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                                              str(path)], capture_output=True, text=True).stdout or "{}")
            text = {k.lower(): v for k, v in tags.get("format", {}).get("tags", {}).items()}.get("prompt")
        else:
            return None
        return json.loads(text) if text else None
    except (OSError, ValueError):
        return None


def facts(graph):
    """What a reader wants from a graph: models, prompts, seeds, sampling, size."""
    out = {"models": [], "prompts": [], "seeds": [], "sampling": [], "size": None, "inputs": []}
    if not graph:
        return out
    for n in graph.values():
        c, i = n.get("class_type", ""), n.get("inputs", {})
        title = n.get("_meta", {}).get("title", "")
        val = lambda k: i.get(k) if not isinstance(i.get(k), list) else None
        for k in ("unet_name", "ckpt_name", "clip_name", "lora_name", "vae_name", "model_name"):
            if val(k) and str(val(k)) not in out["models"]:
                out["models"].append(str(val(k)))
        for k in ("text", "prompt", "value"):
            v = val(k)
            if isinstance(v, str) and len(v) > 20 and ("Encode" in c or "Prompt" in title or "prompt" in k
                                                       or c == "PrimitiveStringMultiline"):
                if v.strip() and v not in out["prompts"]:
                    out["prompts"].append(v.strip())
        for k in ("seed", "noise_seed"):
            if val(k) is not None:
                out["seeds"].append(str(val(k)))
        if c.startswith("KSampler") or c == "SamplerCustomAdvanced":
            s = {k: val(k) for k in ("steps", "cfg", "sampler_name", "scheduler") if val(k) is not None}
            if s:
                out["sampling"].append(s)
        if title.lower() in ("width", "height") or c.startswith("EmptyLatent") or c.startswith("EmptySD3"):
            w, h = val("width"), val("height")
            if w and h:
                out["size"] = f"{w}×{h}"
        if c == "LoadImage" and val("image"):
            out["inputs"].append(str(val("image")))
    out["seeds"] = sorted(set(out["seeds"]))
    return out


# --- media for the public pages ----------------------------------------------

def small(src, media, limit_px=1280):
    """A copy for the public pages, made small; its name in docs/media/."""
    ext = src.suffix.lower()
    key = hashlib.sha1(f"{src}:{src.stat().st_mtime_ns}:{limit_px}".encode()).hexdigest()[:12]
    if ext in IMAGE:
        dst = media / f"{src.stem[:40]}-{key}.jpg"
        if not dst.exists():
            from PIL import Image
            im = Image.open(src)
            if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                # JPEG has no transparency: show it as a checkerboard, as editors do
                im = im.convert("RGBA")
                board = Image.new("RGBA", im.size, (236, 236, 236, 255))
                tile = max(8, im.width // 64)
                dark = Image.new("RGBA", (tile, tile), (200, 200, 200, 255))
                for y in range(0, im.height, tile):
                    for x in range((y // tile) % 2 * tile, im.width, 2 * tile):
                        board.paste(dark, (x, y))
                im = Image.alpha_composite(board, im)
            im = im.convert("RGB")
            im.thumbnail((limit_px, limit_px))
            im.save(dst, quality=82, optimize=True)
    elif ext in VIDEO:
        dst = media / f"{src.stem[:40]}-{key}.mp4"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale='min(960,iw)':-2",
                            "-c:v", "libx264", "-crf", "27", "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-map_metadata", "-1", str(dst)],
                           check=True)
    elif ext in AUDIO:
        dst = media / f"{src.stem[:40]}-{key}.mp3"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
                            "-map_metadata", "-1", str(dst)], check=True)
    elif ext in MODEL:
        # decimated to 40,000 triangles, textures to 1024 px, Draco and JPEG
        # (docs/tools/glb_small.py in Blender): a 27 MB model came out 0.6 MB
        # and looked the same
        dst = media / f"{src.stem[:40]}-{key}{ext}"
        if not dst.exists():
            if not shutil.which("blender"):
                sys.exit(f"{src.name}: making a GLB small takes Blender")
            subprocess.run(["blender", "-b", "-P", str(HERE / "tools/glb_small.py"), "--", str(src), str(dst),
                            "40000", "1024"], check=True, capture_output=True)
    else:
        dst = media / f"{src.stem[:40]}-{key}{ext}"
        if not dst.exists():
            shutil.copy(src, dst)
    return dst


# --- the pages ---------------------------------------------------------------

CSS = """
:root{--bg:#fbfaf7;--fg:#1f1d1a;--muted:#6b665e;--card:#fff;--line:#e4e0d8;--accent:#2f6f8f;--code:#f3f0ea}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--fg:#ebe7e0;--muted:#a39d93;--card:#201f1d;--line:#34312d;--accent:#7fb6d1;--code:#2a2826}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{border-bottom:1px solid var(--line);padding:18px 16px}header a{color:var(--fg);text-decoration:none;font-weight:600}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:8px;font-size:14px}nav a{color:var(--muted);text-decoration:none}nav a.on,nav a:hover{color:var(--accent)}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:28px;margin:12px 0 4px}h2{font-size:21px;margin:34px 0 6px}
.lead{color:var(--muted);max-width:760px}p{max-width:760px}a{color:var(--accent)}
.ex{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0}.ex>h2{margin-top:0}
.media{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin:12px 0}
.media.wide{grid-template-columns:1fr}figure{margin:0}figure img,figure video{width:100%;height:auto;border-radius:6px;display:block;background:#0003}
figure audio{width:100%}figcaption{font-size:13px;color:var(--muted);margin-top:4px}
model-viewer{width:100%;height:360px;background:#0001;border-radius:6px}
details{margin-top:10px;font-size:14px}details h3{font-size:14px;margin:14px 0 2px}summary{cursor:pointer;color:var(--accent)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:8px 0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
pre,code{background:var(--code);border-radius:5px;font:13px/1.45 ui-monospace,Menlo,monospace}pre{padding:10px;overflow-x:auto;white-space:pre-wrap}code{padding:1px 4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}.cards a{display:block;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;color:var(--fg);text-decoration:none}
.cards a:hover{border-color:var(--accent)}.cards b{display:block;margin-bottom:4px}.cards span{color:var(--muted);font-size:14px}
table{border-collapse:collapse;font-size:14px;margin:10px 0}td,th{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left}
footer{color:var(--muted);font-size:13px;padding:30px 16px;text-align:center}
@media (max-width:600px){.media{grid-template-columns:1fr}h1{font-size:24px}}
"""

MODEL_VIEWER = '<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>'


def inline(text):
    """A little markdown: `code`, **bold**, *italic*, [link](url), paragraphs and lists."""
    def one(s):
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = "".join(part if part.startswith("<code>") else
                    re.sub(r"(?<![*\w])\*([^*\s][^*]*?)\*(?![*\w])", r"<i>\1</i>", part)
                    for part in re.split(r"(<code>.*?</code>)", s))
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if all(l.lstrip().startswith(("- ", "* ")) for l in lines):
            out.append("<ul>" + "".join(f"<li>{one(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        elif all(re.match(r"\s*\d+\. ", l) for l in lines):
            out.append("<ol>" + "".join(f"<li>{one(re.sub(r'^\s*\d+\. ', '', l))}</li>" for l in lines) + "</ol>")
        else:
            out.append(f"<p>{one(' '.join(lines))}</p>")
    return "\n".join(out)


def page(title, body, sections, current, public, lang="en", other=None):
    nav = "".join(f'<a href="{s["id"]}.html"{" class=on" if s["id"] == current else ""}>{html.escape(s["title"])}</a>'
                  for s in sections)
    if other:                                  # the same page in the other language
        nav += f'<a href="{other[0]}" lang="{other[1]}">{"Deutsch" if other[1] == "de" else "English"}</a>'
    note = "Examples and results, made with llmctl on one Strix Halo machine." + ("" if public else " Local edition: every example, the media in full.")
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style>{MODEL_VIEWER}</head><body>
<header><a href="index.html">llmctl gallery</a><nav>{nav}</nav></header>
<main>{body}</main><footer>{note} <a href="https://github.com/chrbayer/AI">github.com/chrbayer/AI</a></footer></body></html>"""


GALLERY = "https://chrbayer.github.io/AI/"


def help_page(title, body, lang):
    """Darkroom's help: the section alone, a way back to the page, the gallery
    for more."""
    back, more = ("← Zurück zu Darkroom", "Mehr Beispiele in der Galerie") if lang == "de" \
        else ("← Back to Darkroom", "More examples in the gallery")
    # before and after side by side, also on a phone
    css = CSS + ("header{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}"
                 "@media (max-width:600px){.media{grid-template-columns:1fr 1fr;gap:8px}}")
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{css}</style></head><body>
<header><a href="/darkroom">{back}</a><a href="{GALLERY}" style="font-weight:400">{more}</a></header>
<main>{body}</main><footer><a href="https://github.com/chrbayer/AI">github.com/chrbayer/AI</a></footer></body></html>"""


def german(item):
    """A section, example or medium as the German page has it."""
    de = item.get("de") or {}
    out = dict(item, **de)
    if item.get("caption_de"):
        out["caption"] = item["caption_de"]
    if "media" in item:
        out["media"] = [german(m) for m in item["media"]]
    if "examples" in item:
        out["examples"] = [german(e) for e in item["examples"]]
    return out


class Builder:
    def __init__(self, out, public, limit_mb):
        self.out, self.public, self.limit = out, public, limit_mb
        self.media = out / "media"
        self.graphs = out / "graphs"
        if public:
            self.media.mkdir(parents=True, exist_ok=True)
        self.graphs.mkdir(parents=True, exist_ok=True)
        self.used = set()
        self.prefix = "media/"                 # where a page finds its media; the help's elsewhere
        self.help_used = None                  # the media a help page takes, while one is built

    def src(self, path):
        """Where the page finds a medium."""
        if self.public:
            dst = small(path, self.media)
            self.used.add(dst.name)
            if self.help_used is not None:
                self.help_used.add(dst.name)
            return f"{self.prefix}{dst.name}"
        return path.resolve().as_uri()

    def figure(self, m):
        path = resolve(m["file"])
        if not path.is_file():
            print(f"  missing: {m['file']}", file=sys.stderr)
            return ""
        cap = f"<figcaption>{inline(m['caption'])[3:-4]}</figcaption>" if m.get("caption") else ""
        ext, url = path.suffix.lower(), self.src(path)
        if ext in IMAGE:
            el = f'<a href="{url}"><img src="{url}" loading="lazy" alt="{html.escape(m.get("caption", path.name))}"></a>'
        elif ext in VIDEO:
            el = f'<video src="{url}" controls preload="metadata" playsinline></video>'
        elif ext in AUDIO:
            el = f'<audio src="{url}" controls preload="none"></audio>'
        elif ext in MODEL:
            el = f'<model-viewer src="{url}" camera-controls auto-rotate shadow-intensity="1" alt="{html.escape(path.name)}"></model-viewer>'
        else:
            el = f'<a href="{url}">{html.escape(path.name)}</a>'
        return f"<figure>{el}{cap}</figure>"

    def made(self, ex):
        """The 'how it was made' block: the example's own facts, then what the graph
        in each medium says — once, if they all share one graph."""
        graphs = []
        for m in ex.get("media", []):
            p = resolve(m["file"])
            g = graph_of(p) if p.is_file() else None
            if g and all(g != h for _, h in graphs):
                graphs.append((m.get("caption") or p.name, g))
        out = []
        rows = list(ex.get("facts", {}).items())
        if rows:
            out.append(self.rows(rows))
        if ex.get("command"):
            out.append(f"<pre>{html.escape(ex['command'])}</pre>")
        for label, g in graphs:
            if len(graphs) > 1:
                out.append(f"<h3>{inline(label)[3:-4]}</h3>")
            out.append(self.graph_block(g, ex))
        if not out:
            return ""
        return f"<details{' open' if ex.get('open') else ''}><summary>How it was made</summary>{''.join(out)}</details>"

    def rows(self, rows):
        return "<dl>" + "".join(f"<dt>{html.escape(k)}</dt><dd>{inline(str(v))[3:-4]}</dd>" for k, v in rows) + "</dl>"

    def graph_block(self, g, ex):
        f = facts(g)
        rows = []
        if f["models"]:
            rows.append(("Models", ", ".join(f["models"])))
        if f["size"]:
            rows.append(("Size", f["size"]))
        if f["sampling"]:
            rows.append(("Sampling", ", ".join(f"{k} {v}" for k, v in f["sampling"][0].items())))
        if f["seeds"] and not ex.get("hide_seed"):
            rows.append(("Seed", ", ".join(f["seeds"][:4])))
        out = [self.rows(rows)] if rows else []
        out += [f"<pre>{html.escape(p)}</pre>" for p in f["prompts"][:3]]
        name = hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:12] + ".json"
        (self.graphs / name).write_text(json.dumps(g, indent=1, ensure_ascii=False))
        out.append(f'<p><a href="graphs/{name}" download>The ComfyUI API graph</a> — '
                   f'queue it on a running ComfyUI to make it again.</p>')
        return "".join(out)

    def example(self, ex):
        media = [m for m in ex.get("media", []) if not self.public or m.get("public", ex.get("public", False))]
        figs = "".join(self.figure(m) for m in media)
        wide = " wide" if ex.get("wide") else ""
        table = ""
        if ex.get("table"):
            head, *rows = ex["table"]
            table = ("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr>" +
                     "".join("<tr>" + "".join(f"<td>{inline(str(c))[3:-4]}</td>" for c in r) + "</tr>" for r in rows) +
                     "</table>")
        made = "" if self.help_used is not None else self.made(ex)
        return (f'<section class="ex" id="{ex.get("id", "")}"><h2>{html.escape(ex["title"])}</h2>'
                f'{inline(ex.get("text", ""))}{table}<div class="media{wide}">{figs}</div>{made}</section>')

    def help(self, s, exs):
        """Darkroom's help, from the same section: docs/help/<id>.<lang>.html."""
        d = self.out / "help"
        d.mkdir(exist_ok=True)
        self.prefix, self.help_used = "/darkroom/help/media/", set()
        try:
            for lang, sec in (("en", dict(s, examples=exs)), ("de", german(dict(s, examples=exs)))):
                body = f'<h1>{html.escape(sec["title"])}</h1><div class="lead">{inline(sec.get("intro", ""))}</div>' + \
                       "".join(self.example(e) for e in sec["examples"])
                (d / f"{s['id']}.{lang}.html").write_text(help_page(sec["title"], body, lang))
            (d / "media.txt").write_text("".join(f"{n}\n" for n in sorted(self.help_used)))
            print(f"  help/{s['id']}.en.html, .de.html: {len(self.help_used)} media")
        finally:
            self.prefix, self.help_used = "media/", None

    def build(self, data):
        sections = [s for s in data["sections"] if not self.public or any(
            e.get("public") or any(m.get("public") for m in e.get("media", [])) for e in s["examples"])]
        for s in sections:
            exs = [e for e in s["examples"] if not self.public or e.get("public") or any(m.get("public") for m in e.get("media", []))]
            body = f'<h1>{html.escape(s["title"])}</h1><div class="lead">{inline(s.get("intro", ""))}</div>' + \
                   "".join(self.example(e) for e in exs)
            other = (f"{s['id']}.de.html", "de") if s.get("de") else None
            (self.out / f"{s['id']}.html").write_text(page(f"{s['title']} — llmctl gallery", body, sections, s["id"],
                                                            self.public, "en", other))
            print(f"  {s['id']}.html: {len(exs)} examples")
            if s.get("de"):
                g = german(dict(s, examples=exs))
                body = f'<h1>{html.escape(g["title"])}</h1><div class="lead">{inline(g.get("intro", ""))}</div>' + \
                       "".join(self.example(e) for e in g["examples"])
                (self.out / f"{s['id']}.de.html").write_text(page(f"{g['title']} — llmctl gallery", body, sections,
                                                                   s["id"], self.public, "de", (f"{s['id']}.html", "en")))
                print(f"  {s['id']}.de.html")
            if s.get("help") and self.public:
                self.help(s, exs)
        cards = "".join(f'<a href="{s["id"]}.html"><b>{html.escape(s["title"])}</b><span>{html.escape(s.get("summary", ""))}</span></a>'
                        for s in sections)
        body = f'<h1>{html.escape(data["title"])}</h1><div class="lead">{inline(data.get("intro", ""))}</div><div class="cards">{cards}</div>'
        (self.out / "index.html").write_text(page(data["title"], body, sections, "", self.public))
        if self.public:
            for f in self.media.iterdir():                       # what no page uses any more
                if f.name not in self.used:
                    f.unlink()
            size = sum(f.stat().st_size for f in self.out.rglob("*") if f.is_file()) / 1e6
            print(f"  public site: {size:.0f} MB")
            if size > self.limit:
                sys.exit(f"the public site is {size:.0f} MB, above --limit {self.limit} MB: mark fewer media public")


def main():
    ap = argparse.ArgumentParser(description="the llmctl gallery: examples and results as HTML")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="every example, the media in full, where they lie")
    mode.add_argument("--public", action="store_true", help="the public examples, media made small, into docs/")
    ap.add_argument("--out", help="where the local pages go (default ~/.local/share/llmctl/docs)")
    ap.add_argument("--limit", type=float, default=100, help="MB the public site may take (default 100)")
    a = ap.parse_args()
    data = json.loads((HERE / "examples.json").read_text())
    out = HERE if a.public else Path(a.out or SHARE / "docs")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'public' if a.public else 'local'} gallery → {out}")
    Builder(out, a.public, a.limit).build(data)
    print(f"  open {out / 'index.html'}")


if __name__ == "__main__":
    main()
