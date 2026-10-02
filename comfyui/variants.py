"""One UI workflow with switches, several API workflows: which is which.

A workflow in comfyui/workflows/ may hold its variants behind switches — a
PrimitiveBoolean titled "Turbo", "Unzensiert", "Nur in der Maske",
"Ausschnitt", "Feine Kanten" or "XL" feeding If/Else switches, and the
"Steuerung" of an llmctl Control image node. ComfyUI runs only the branch a
switch picks (its inputs are lazy), so the other branch costs nothing.

The image API keeps a model per variant (qwen-image-21-inpaint-crop-turbo, …):
export_api.py writes each variant's API graph with its switches set and the
unused branches cut away (`resolve`). `canonical` reduces a graph to what it
computes, for comparing a variant with the workflow it replaced.
"""
import json

# UI workflow -> [(API workflow name, settings)]; a workflow not named here has no variants.
VARIANTS = {
    "Qwen-Image 2.1 T2I": [
        ("Qwen-Image 2.1 T2I (bf16, dpmpp_2m 14)", {"Turbo": False, "Unzensiert": False}),
        ("Qwen-Image 2.1 T2I Turbo (bf16, 4 Schritte)", {"Turbo": True, "Unzensiert": False}),
        ("Qwen-Image 2.1 Heretic T2I (bf16, dpmpp_2m 14)", {"Turbo": False, "Unzensiert": True}),
        ("Qwen-Image 2.1 Heretic T2I Turbo (bf16, 4 Schritte)", {"Turbo": True, "Unzensiert": True}),
    ],
    "Qwen-Image 2.1 Edit": [
        ("Qwen-Image 2.1 Edit (bf16, dpmpp_2m 14)", {"Turbo": False, "Unzensiert": False}),
        ("Qwen-Image 2.1 Edit Turbo (bf16, 4 Schritte)", {"Turbo": True, "Unzensiert": False}),
        ("Qwen-Image 2.1 Heretic Edit (bf16, dpmpp_2m 14)", {"Turbo": False, "Unzensiert": True}),
        ("Qwen-Image 2.1 Heretic Edit Turbo (bf16, 4 Schritte)", {"Turbo": True, "Unzensiert": True}),
    ],
    "Qwen-Image 2.1 Background Removal": [
        ("Qwen-Image 2.1 Background Removal (bf16, euler 25)", {"Turbo": False}),
        ("Qwen-Image 2.1 Background Removal Turbo (bf16, 4 Schritte)", {"Turbo": True}),
    ],
    "Qwen-Image 2.1 Control": [
        (f"Qwen-Image 2.1 {name}Control{t} ({spec})", {"Turbo": turbo, "Steuerung": kind, "Nur in der Maske": False})
        for kind, name in (("Bild", ""), ("Kanten", "Canny "), ("Pose", "Pose "), ("Tiefe", "Depth "))
        for turbo, t, spec in ((False, "", "bf16, dpmpp_2m 14"), (True, " Turbo", "bf16, 4 Schritte"))
    ] + [
        (f"Qwen-Image 2.1 {name}Inpaint{t} ({spec})", {"Turbo": turbo, "Steuerung": kind, "Nur in der Maske": True})
        for kind, name in (("Kanten", "Canny "), ("Pose", "Pose "), ("Tiefe", "Depth "))
        for turbo, t, spec in ((False, "", "bf16, dpmpp_2m 14"), (True, " Turbo", "bf16, 4 Schritte"))
    ],
    "Qwen-Image 2.1 Inpaint": [
        ("Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)", {"Turbo": False, "Ausschnitt": False}),
        ("Qwen-Image 2.1 Inpaint Turbo (bf16, 4 Schritte)", {"Turbo": True, "Ausschnitt": False}),
        ("Qwen-Image 2.1 Inpaint Crop (bf16, dpmpp_2m 14)", {"Turbo": False, "Ausschnitt": True}),
        ("Qwen-Image 2.1 Inpaint Crop Turbo (bf16, 4 Schritte)", {"Turbo": True, "Ausschnitt": True}),
    ],
    "Qwen-Image 2.1 Outpaint": [
        ("Qwen-Image 2.1 Outpaint (bf16, dpmpp_2m 14)", {"Turbo": False}),
        ("Qwen-Image 2.1 Outpaint Turbo (bf16, 4 Schritte)", {"Turbo": True}),
    ],
    "Qwen-Image 2.1 Colorize": [
        ("Qwen-Image 2.1 Colorize (bf16, dpmpp_2m 14)", {"Turbo": False}),
        ("Qwen-Image 2.1 Colorize Turbo (bf16, 4 Schritte)", {"Turbo": True}),
    ],
    "Z-Image Turbo T2I": [
        ("Z-Image Turbo T2I (bf16, 8 Schritte)", {"Unzensiert": False}),
        ("Z-Image Turbo NSFW T2I (bf16, 8 Schritte)", {"Unzensiert": True}),
    ],
    "Z-Image Turbo Control": [
        (f"Z-Image Turbo {name}Control (bf16, 8 Schritte)", {"Steuerung": kind})
        for kind, name in (("Bild", ""), ("Kanten", "Canny "), ("Pose", "Pose "), ("Tiefe", "Depth "))
    ],
    "Z-Image Turbo Inpaint": [
        ("Z-Image Turbo Inpaint (bf16, 8 Schritte)", {"Ausschnitt": False}),
        ("Z-Image Turbo Inpaint Crop (bf16, 8 Schritte)", {"Ausschnitt": True}),
    ],
    "FLUX.2 dev T2I": [
        ("FLUX.2 dev T2I (fp8, 20 Schritte)", {"Turbo": False, "Unzensiert": False}),
        ("FLUX.2 dev Turbo T2I (fp8, 8 Schritte)", {"Turbo": True, "Unzensiert": False}),
        ("FLUX.2 dev NSFW T2I (fp8, 20 Schritte)", {"Turbo": False, "Unzensiert": True}),
    ],
    "FLUX.2 dev Edit": [
        ("FLUX.2 dev Edit (fp8, 20 Schritte)", {"Turbo": False, "Unzensiert": False}),
        ("FLUX.2 dev Turbo Edit (fp8, 8 Schritte)", {"Turbo": True, "Unzensiert": False}),
        ("FLUX.2 dev NSFW Edit (fp8, 20 Schritte)", {"Turbo": False, "Unzensiert": True}),
    ],
    "FLUX.2 klein 9B T2I": [
        ("FLUX.2 klein 9B T2I (bf16, 4 Schritte)", {"Unzensiert": False}),
        ("FLUX.2 klein 9B NSFW T2I (bf16, 4 Schritte)", {"Unzensiert": True}),
    ],
    "FLUX.2 klein 9B Edit": [
        ("FLUX.2 klein 9B Edit (bf16)", {"Unzensiert": False}),
        ("FLUX.2 klein 9B NSFW Edit (bf16)", {"Unzensiert": True}),
    ],
    "BiRefNet Background Removal": [
        ("BiRefNet Background Removal (general, MIT)", {"Feine Kanten": False}),
        ("BiRefNet Matting Background Removal (HR, MIT)", {"Feine Kanten": True}),
    ],
    "ACE-Step 1.5": [
        ("audio/ACE-Step 1.5 Turbo (bf16, 8 Schritte)", {"XL": False}),
        ("audio/ACE-Step 1.5 XL Turbo (bf16, 8 Schritte)", {"XL": True}),
    ],
}
SWITCH_TITLES = {"Turbo", "Unzensiert", "Nur in der Maske", "Ausschnitt", "Feine Kanten", "XL"}
CHOICE = "LlmctlControlImage"
OUTPUTS = {"SaveImage", "SaveImageAdvanced", "SaveAudio", "SaveAudioMP3", "SaveAudioAdvanced", "SaveAudioOpus",
           "PreviewImage", "PreviewAny", "PreviewAudio", "Save3DAdvanced", "SaveGLB", "LlmctlArtifactCheck",
           "SaveVideo", "SaveWEBM", "SaveAnimatedWEBP"}


def is_ref(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def title(n):
    return (n.get("_meta") or {}).get("title", "")


def _replace_refs(g, old, new):
    """Every input that reads output `old` ([id, slot]) reads `new` instead (a ref or a value)."""
    for n in g.values():
        for k, v in list(n["inputs"].items()):
            if is_ref(v) and v[0] == old[0] and v[1] == old[1]:
                n["inputs"][k] = json.loads(json.dumps(new))


def prune(g):
    """Drop what no output needs."""
    need, todo = set(), [k for k, n in g.items() if n["class_type"] in OUTPUTS]
    while todo:
        k = todo.pop()
        if k in need or k not in g:
            continue
        need.add(k)
        todo += [v[0] for v in g[k]["inputs"].values() if is_ref(v)]
    for k in [k for k in g if k not in need]:
        del g[k]
    return g


def resolve(graph, settings):
    """The graph with its switches set as `settings` says and the unused branches gone."""
    g = json.loads(json.dumps(graph))
    flags = {k: n for k, n in g.items() if n["class_type"] == "PrimitiveBoolean" and title(n) in settings}
    unknown = set(settings) - {title(n) for n in flags.values()} - {"Steuerung"}
    if unknown:
        have = sorted({title(n) for n in g.values() if n["class_type"] == "PrimitiveBoolean"})
        raise KeyError(f"no switch titled {sorted(unknown)} (switches: {have})")
    for k, n in list(g.items()):
        if n["class_type"] == "ComfySwitchNode" and is_ref(n["inputs"].get("switch")) and n["inputs"]["switch"][0] in flags:
            on = settings[title(flags[n["inputs"]["switch"][0]])]
            _replace_refs(g, [k, 0], n["inputs"]["on_true" if on else "on_false"])
            del g[k]
        elif n["class_type"] == CHOICE and "Steuerung" in settings:
            kind = settings["Steuerung"].lower()
            _replace_refs(g, [k, 0], n["inputs"][kind])
            _replace_refs(g, [k, 1], n["inputs"][f"strength_{kind}"])
            del g[k]
    return prune(g)


IGNORED = {"seed", "noise_seed", "text", "prompt", "filename_prefix", "image", "control_after_generate",
           "tags", "lyrics", "value"}


def canonical(graph):
    """What the graph computes, without ids, titles, prompts, seeds or file names:
    the sorted signatures of its outputs, each the node's class and inputs with every
    link replaced by the signature of what it reads. Primitive nodes and switches set
    to a value read as that value."""
    g = json.loads(json.dumps(graph))
    for k, n in list(g.items()):                      # primitives as their values
        if n["class_type"] in ("PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean") and "value" in n["inputs"]:
            _replace_refs(g, [k, 0], n["inputs"]["value"])
    for k, n in list(g.items()):                      # switches already set
        if n["class_type"] == "ComfySwitchNode" and isinstance(n["inputs"].get("switch"), bool):
            _replace_refs(g, [k, 0], n["inputs"]["on_true" if n["inputs"]["switch"] else "on_false"])
    for k in [k for k, n in g.items() if n["class_type"] in ("ImageCompare", "MarkdownNote", "Note")]:
        del g[k]
    prune(g)
    memo = {}

    def num(v):
        return int(v) if isinstance(v, float) and v.is_integer() else v

    def sig(k):
        if k not in memo:
            n = g[k]
            memo[k] = json.dumps([n["class_type"], {a: (["@", sig(v[0]), v[1]] if is_ref(v) else num(v))
                                                    for a, v in sorted(n["inputs"].items()) if a not in IGNORED}],
                                 sort_keys=True)
        return memo[k]
    return sorted(sig(k) for k, n in g.items() if n["class_type"] in OUTPUTS and n["class_type"] != "PreviewImage")
