#!/usr/bin/env python3
"""Unit tests for llmctl's Python helpers — the parts with rules of their own:
which models a workflow needs, how text is split into pieces to speak, how audio
is levelled, and how a transcript is cleaned on its way through the proxy.

    tests/test_python.py            all of them
    tests/test_python.py -k voice   the usual unittest options

Nothing here touches a GPU, a model or the network. tts_server needs flask and
numpy, proxy.py needs flask and requests; a helper whose imports are missing is
skipped rather than failed, so the rest still runs.
"""
import io
import contextlib
import shutil
import json
import os
import re
import warnings
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import comfyui_models as cm                                        # stdlib only
import anthropic_compat as ac                                       # stdlib only
sys.path.insert(0, str(ROOT / "comfyui"))
import variants                                                     # noqa: E402

ts: Any = None          # typed loosely: a missing import skips its tests below
proxy: Any = None
images: Any = None
try:
    import images_server as images                                 # flask, requests
except ImportError:
    pass
try:
    import tts_server as ts                                        # flask, numpy
except ImportError:
    pass
try:
    import proxy                                                   # flask, requests
except ImportError:
    pass


def workflow(*models) -> dict[str, Any]:
    """A workflow in ComfyUI's shape, with one loader per model entry."""
    return {"nodes": [{"id": i, "type": "Loader", "properties": {"models": [m]}}
                      for i, m in enumerate(models)]}


class Workflows(unittest.TestCase):
    def test_models_are_keyed_by_directory_and_name(self):
        wf = workflow({"name": "a.safetensors", "url": "https://x/a", "directory": "vae"})
        self.assertEqual(list(cm.refs(wf)), ["vae/a.safetensors"])

    def test_a_model_without_a_directory_is_ignored(self):
        wf = workflow({"name": "a.safetensors", "url": "https://x/a"})
        self.assertEqual(cm.refs(wf), {})

    def test_models_inside_subgraphs_count(self):
        wf = workflow()
        wf["definitions"] = {"subgraphs": [workflow(
            {"name": "b.safetensors", "url": "https://x/b", "directory": "loras"})]}
        self.assertEqual(list(cm.refs(wf)), ["loras/b.safetensors"])

    def test_workflows_sharing_a_model_name_it_once(self):
        m = {"name": "a.safetensors", "url": "https://x/a", "directory": "vae"}
        urls, users = cm.collect([("one.json", workflow(m)), ("two.json", workflow(m))])
        self.assertEqual(list(urls), ["vae/a.safetensors"])
        self.assertEqual(users["vae/a.safetensors"], ["one.json", "two.json"])

    def test_two_urls_for_one_file_are_both_kept(self):
        a = {"name": "a.safetensors", "url": "https://x/a", "directory": "vae"}
        b = dict(a, url="https://y/a")
        urls, _ = cm.collect([("one.json", workflow(a)), ("two.json", workflow(b))])
        self.assertEqual(urls["vae/a.safetensors"], ["https://x/a", "https://y/a"])


class Urls(unittest.TestCase):
    def test_a_file_url_gives_repo_revision_and_path(self):
        m = cm.HF_URL.match("https://huggingface.co/org/repo/resolve/main/dir/f.safetensors")
        assert m
        self.assertEqual(m.groups(), ("org/repo", "main", "dir/f.safetensors"))

    def test_a_repo_url_is_not_a_file_url(self):
        url = "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz"
        self.assertIsNone(cm.HF_URL.match(url))
        repo = cm.HF_REPO.match(url)
        assert repo
        self.assertEqual(repo.groups(), ("Qwen/Qwen3-TTS-Tokenizer-12Hz", None))

    def test_a_repo_url_may_pin_a_revision(self):
        pinned = cm.HF_REPO.match("https://huggingface.co/a/b/tree/abc123")
        assert pinned
        self.assertEqual(pinned.groups(), ("a/b", "abc123"))

    def test_a_civitai_token_goes_into_the_query(self):
        out = cm.with_token("https://civitai.com/api/download/models/1?fileId=2", "SECRET")
        self.assertIn("token=SECRET", out)
        self.assertIn("fileId=2", out)


class PresentAndSize(unittest.TestCase):
    def test_a_file_is_present_a_missing_one_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.safetensors"
            self.assertFalse(cm.present(f))
            f.write_bytes(b"x" * 10)
            self.assertTrue(cm.present(f))
            self.assertEqual(cm.size(f), 10)

    def test_a_model_directory_counts_when_it_has_something_in_it(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            repo.mkdir()
            self.assertFalse(cm.present(repo))          # an empty one is a failed download
            (repo / "model.safetensors").write_bytes(b"x" * 5)
            (repo / "config.json").write_bytes(b"{}")
            self.assertTrue(cm.present(repo))
            self.assertEqual(cm.size(repo), 7)


@unittest.skipIf(ts is None, "tts_server needs flask and numpy")
class Speaking(unittest.TestCase):
    def test_the_first_sentence_goes_out_on_its_own(self):
        text = "Erster Satz. Zweiter Satz. Dritter Satz."
        self.assertEqual(ts.sentences(text)[0], "Erster Satz.")

    def test_without_chunking_sentences_are_grouped(self):
        text = "Erster Satz. Zweiter Satz. Dritter Satz."
        self.assertEqual(ts.sentences(text, first_alone=False), [text.replace(". ", ". ")])

    def test_a_group_stays_under_the_limit(self):
        text = " ".join(f"Satz Nummer {i} steht hier." for i in range(60))
        for chunk in ts.sentences(text):
            self.assertLessEqual(len(chunk), max(ts.CHUNK_CHARS, len("Satz Nummer 59 steht hier.")))

    def test_every_word_survives_the_split(self):
        text = "Hallo! Wie geht es dir? Mir geht es gut: wirklich gut; danke."
        self.assertEqual(" ".join(ts.sentences(text)).split(), text.split())

    def test_empty_text_gives_no_pieces(self):
        self.assertEqual(ts.sentences("   "), [])


@unittest.skipIf(ts is None, "tts_server needs flask and numpy")
class Loudness(unittest.TestCase):
    def test_a_quiet_signal_is_brought_up_to_the_target(self):
        import numpy as np
        x = np.full(24000, 0.01, dtype=np.float32)
        y, gain = ts.normalize(x, -20.0)
        self.assertAlmostEqual(float(20 * np.log10(np.sqrt(np.mean(y ** 2)))), -20.0, places=3)
        self.assertGreater(gain, 0)

    def test_peaks_stay_under_the_ceiling(self):
        import numpy as np
        x = np.zeros(24000, dtype=np.float32)
        x[::100] = 0.9                                   # loud peaks, quiet overall
        y, _ = ts.normalize(x, -20.0)
        self.assertLessEqual(float(np.abs(y).max()),
                             10 ** (ts.PEAK_CEILING_DBFS / 20) + 1e-6)

    def test_silence_is_left_alone(self):
        import numpy as np
        x = np.zeros(1000, dtype=np.float32)
        y, gain = ts.normalize(x, -20.0)
        self.assertEqual(gain, 0.0)
        self.assertTrue((y == 0).all())

    def test_a_streamed_wav_header_says_the_length_is_unknown(self):
        self.assertIn(b"\xff\xff\xff\xff", ts.wav_header())
        self.assertNotIn(b"\xff\xff\xff\xff", ts.wav_header(1000))

    def test_the_header_carries_the_sample_rate(self):
        import struct
        self.assertEqual(struct.unpack("<I", ts.wav_header(0)[24:28])[0], ts.SAMPLE_RATE)


@unittest.skipIf(proxy is None, "proxy.py needs flask and requests")
class Transcripts(unittest.TestCase):
    def test_the_language_header_is_taken_off(self):
        text, lang = proxy._split_transcript("language German<asr_text>Guten Tag.")
        self.assertEqual(text, "Guten Tag.")
        self.assertEqual(lang, "German")

    def test_text_without_a_header_is_left_alone(self):
        text, lang = proxy._split_transcript("Guten Tag.")
        self.assertEqual(text, "Guten Tag.")
        self.assertIsNone(lang)

    def test_an_unknown_language_is_reported_as_none(self):
        text, lang = proxy._split_transcript("language None<asr_text>...")
        self.assertEqual(text, "...")
        self.assertIsNone(lang)


@unittest.skipIf(proxy is None, "proxy.py needs flask and requests")
class RequestDefaults(unittest.TestCase):
    """halogen-server has no server-wide reasoning or sampler setting: the proxy
    fills them into requests that say nothing themselves."""
    def setUp(self):
        self.saved = proxy.REQUEST_DEFAULTS
        proxy.REQUEST_DEFAULTS = {"reasoning_effort": "medium", "temperature": 0.6}

    def tearDown(self):
        proxy.REQUEST_DEFAULTS = self.saved

    def test_a_silent_request_gets_them(self):
        chat = {"messages": []}
        proxy.apply_defaults(chat)
        self.assertEqual(chat, {"messages": [], "reasoning_effort": "medium", "temperature": 0.6})

    def test_a_request_keeps_its_own(self):
        chat = {"reasoning_effort": "low", "temperature": 0}
        proxy.apply_defaults(chat)
        self.assertEqual(chat, {"reasoning_effort": "low", "temperature": 0})

    def test_thinking_set_another_way_takes_no_effort(self):
        # The server refuses two thinking controls that disagree.
        for chat in ({"enable_thinking": False}, {"chat_template_kwargs": {"reasoning_effort": "high"}}):
            proxy.apply_defaults(chat)
            self.assertNotIn("reasoning_effort", chat)
            self.assertEqual(chat["temperature"], 0.6)


class ThinkingOff(unittest.TestCase):
    BODY = {"model": "m", "max_tokens": 100, "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "Hi"}]}

    def test_by_default_it_travels_as_thinking(self):
        chat = ac.messages_to_chat(self.BODY)
        self.assertEqual(chat["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", chat)

    def test_halogen_server_gets_an_effort_instead(self):
        chat = ac.messages_to_chat(self.BODY, None, "none")
        self.assertNotIn("thinking", chat)
        self.assertEqual(chat["reasoning_effort"], "none")

    def test_an_effort_still_passes_with_thinking_on(self):
        body = dict(self.BODY, thinking={"type": "enabled"})
        self.assertEqual(ac.messages_to_chat(body, None, "none")["reasoning_effort"], "high")


try:
    import musicvideo                                              # requests
except ImportError:
    musicvideo = None


@unittest.skipIf(musicvideo is None, "musicvideo.py needs requests")
class MusicVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "city.png").write_bytes(b"x")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def plan(self, text, count=4):
        f = self.tmp / "sections.txt"
        f.write_text(text)
        return musicvideo.read_sections(str(f), count, "sing it")

    def test_the_sections_file_is_read_and_filled_up(self):
        plan = self.plan("sing\n# a comment\nscene: a city at night\nsing city.png: eyes closed\n")
        self.assertEqual(plan, [("sing", None, "sing it"), ("scene", None, "a city at night"),
                                ("sing", str(self.tmp / "city.png"), "eyes closed"), ("sing", None, "sing it")])

    def test_a_scene_needs_a_prompt_and_a_picture_must_exist(self):
        for bad in ("scene\n", "dance: x\n", "sing nowhere.png: x\n"):
            with self.assertRaises(SystemExit):
                self.plan(bad)

    def test_the_frames_add_up_to_the_song(self):
        plan = [("sing", None, "p")] * 7
        sections = musicvideo.plan_sections(plan, 31.27, 5, ["a.png", "b.png"], False)
        self.assertEqual(sum(s["frames"] for s in sections), round(31.27 * 24))
        self.assertEqual([s["picture"] for s in sections[:3]], ["a.png", "b.png", "a.png"])
        self.assertEqual(sections[-1]["end"], 31.27)

    def test_scenes_without_a_picture_are_text_only_or_follow(self):
        plan = [("sing", None, "p"), ("scene", None, "city"), ("sing", None, "p")]
        apart = musicvideo.plan_sections(plan, 15, 5, ["a.png", "b.png"], False)
        self.assertEqual([(s["picture"], s["follow"]) for s in apart], [("a.png", False), (None, False), ("b.png", False)])
        along = musicvideo.plan_sections(plan, 15, 5, ["a.png", "b.png"], True)
        self.assertEqual([(s["picture"], s["follow"]) for s in along], [("a.png", False), (None, True), (None, True)])


class Sources(unittest.TestCase):
    def test_the_bundled_recipes_are_complete(self):
        data = json.loads((ROOT / "comfyui" / "sources.json").read_text())
        for key, recipe in data.items():
            if key.startswith("_"):
                continue
            with self.subTest(recipe=key):
                self.assertIn("/", key)                  # <directory>/<name>
                source = "file" if recipe.get("kind") else "shards"
                for field in ("repo", "revision", source, "tensors", "bytes"):
                    self.assertIn(field, recipe)
                self.assertEqual(len(recipe["revision"]), 40)   # a pinned commit

    def test_a_recipe_joins_and_renames_shards(self):
        """build() once went missing unnoticed: every model it would make was
        already there. Two fake shards, joined and renamed without a download."""
        import struct
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            staging = Path(d) / "staging"
            shard_dir = staging / "org" / "repo"
            shard_dir.mkdir(parents=True)
            def shard(path, tensors):
                header, blob = {}, b""
                for name, data in tensors:
                    header[name] = {"dtype": "BF16", "shape": [len(data) // 2], "data_offsets": [len(blob), len(blob) + len(data)]}
                    blob += data
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)
            shard(shard_dir / "a.safetensors", [("model.x", b"\x01\x00\x02\x00")])
            shard(shard_dir / "b.safetensors", [("model.y", b"\x03\x00")])
            recipe = {"repo": "org/repo", "revision": "0" * 40, "shards": ["a.safetensors", "b.safetensors"],
                      "rename": [["model.", "enc."]], "tensors": 2, "bytes": 6}
            dest = Path(d) / "out.safetensors"
            with mock.patch.object(cm.subprocess, "run", return_value=mock.Mock(returncode=0)):
                self.assertTrue(cm.build("text_encoders/out.safetensors", recipe, dest, staging))
            header, start = cm.read_header(dest)
            self.assertEqual(sorted(header), ["enc.x", "enc.y"])
            raw = dest.read_bytes()
            x = header["enc.x"]["data_offsets"]
            self.assertEqual(raw[start + x[0]:start + x[1]], b"\x01\x00\x02\x00")

    def test_bf16_rounds_to_nearest_even(self):
        import numpy as np
        x = np.array([1.0, 1.00390625, 1.01171875, -2.5], dtype=np.float32)   # halfway cases included
        back = cm.bf16_to_f32(np, cm.f32_to_bf16(np, x).tobytes())
        self.assertEqual(back.tolist(), [1.0, 1.0, 1.015625, -2.5])

    def test_every_bundled_workflow_parses_and_names_its_models(self):
        for path in sorted((ROOT / "comfyui" / "workflows").glob("*.json")):
            with self.subTest(workflow=path.name):
                wf = json.loads(path.read_text())
                if any(n.get("type", "").startswith("Llmctl") for n in wf["nodes"]):
                    continue                                  # llmctl's nodes load no model
                self.assertTrue(cm.refs(wf), "names no models at all")
                for key, urls in cm.refs(wf).items():
                    self.assertTrue(urls, f"{key} has no URL")


class Checksums(unittest.TestCase):
    def test_a_file_is_read_once_and_again_only_when_it_changes(self):
        import hashlib, os
        with tempfile.TemporaryDirectory() as d:
            os.environ["LLMCTL_CHECKSUMS"] = str(Path(d) / "sums.tsv")
            f = Path(d) / "m.safetensors"
            f.write_bytes(b"one")
            self.assertEqual(cm.sha256_files([f])[f], hashlib.sha256(b"one").hexdigest())
            # A cached entry is used as long as size and mtime hold: planted, it is returned.
            cache = Path(d) / "sums.tsv"
            cache.write_text(cache.read_text().replace(hashlib.sha256(b"one").hexdigest(), "cached"))
            self.assertEqual(cm.sha256_files([f])[f], "cached")
            f.write_bytes(b"other")
            self.assertEqual(cm.sha256_files([f])[f], hashlib.sha256(b"other").hexdigest())
            del os.environ["LLMCTL_CHECKSUMS"]


API = ROOT / "comfyui" / "api"


def api(name):
    return json.loads((API / name).read_text())


@unittest.skipIf(images is None, "images_server needs flask and requests")
class ImageApi(unittest.TestCase):
    def test_every_image_workflow_has_an_api_version(self):
        for path in sorted((ROOT / "comfyui" / "workflows").glob("*.json")):
            if "TTS" in path.name:
                continue
            if any(n.get("type", "") == "LlmctlArtifactCheck" for n in json.loads(path.read_text())["nodes"]):
                continue                                          # calls the image API itself
            nodes = json.loads(path.read_text())["nodes"]
            # a workflow with switches is exported as its variants
            names = [n for n, _ in variants.VARIANTS.get(path.stem, [(path.stem, None)])]
            if any(n.get("type", "").startswith("SaveAudio") for n in nodes):
                for name in names:
                    self.assertTrue((API / "audio" / f"{name.removeprefix('audio/')}.json").exists(),
                                    "run comfyui/export_api.py: music goes to api/audio/")
                continue
            if any(n.get("type", "") in ("Save3DAdvanced", "SaveGLB", "SaveVideo", "SaveWEBM", "SaveAnimatedWEBP") for n in nodes):
                continue                                          # 3D and video, not for the image API
            for name in names:
                with self.subTest(workflow=path.name, api=name):
                    self.assertTrue((API / f"{name}.json").exists(),
                                    "run comfyui/export_api.py and commit what it writes")

    def test_workflow_names(self):
        self.assertEqual(images.slug("FLUX.2 klein 9B T2I (bf16, 4 Schritte)"), ("flux2-klein-9b", "generations"))
        self.assertEqual(images.slug("FLUX.2 klein 9B NSFW Edit (bf16)"), ("flux2-klein-9b-nsfw", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Heretic T2I (bf16, dpmpp_2m 14)"),
                         ("qwen-image-21-heretic", "generations"))
        self.assertEqual(images.slug("SeedVR2 7B Upscale (fp16)"), ("seedvr2-7b-upscale", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Canny Control (bf16, dpmpp_2m 14)"), ("qwen-image-21-canny-control", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)"), ("qwen-image-21-inpaint", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Outpaint Turbo (bf16, 4 Schritte)"), ("qwen-image-21-outpaint-turbo", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Background Removal (bf16, euler 25)"),
                         ("qwen-image-21-background-removal", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14)"),
                         ("qwen-image-21-detailer", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Colorize Turbo (bf16, 4 Schritte)"),
                         ("qwen-image-21-colorize-turbo", "edits"))
        self.assertEqual(images.slug("Z-Image Turbo T2I (bf16, 8 Schritte)"), ("z-image-turbo", "generations"))
        self.assertEqual(images.slug("Z-Image Turbo Canny Control (bf16, 8 Schritte)"), ("z-image-turbo-canny-control", "edits"))
        self.assertEqual(images.slug("Z-Image T2I (bf16, 25 Schritte)"), ("z-image", "generations"))
        self.assertEqual(images.slug("Z-Image Turbo NSFW T2I (bf16, 8 Schritte)"), ("z-image-turbo-nsfw", "generations"))

    def test_layered_takes_a_layer_count(self):
        self.assertEqual(images.slug("Qwen-Image Layered (bf16, 20 Schritte)"), ("qwen-image-layered", "edits"))
        g = api("Qwen-Image Layered (bf16, 20 Schritte).json")
        self.assertIn("layers", images.extras(g))
        images.set_layers(g, "4")
        self.assertEqual({n["inputs"]["layers"] for n in g.values() if n["class_type"] == "EmptyQwenImageLayeredLatentImage"}, {4})
        for bad in (0, 9, "many"):
            with self.subTest(bad), self.assertRaises(images.Refused):
                images.set_layers(g, bad)
        with self.assertRaises(images.Refused):
            images.set_layers(api("Qwen-Image 2.1 Edit (bf16, dpmpp_2m 14).json"), 3)

    def test_layered_control_names_its_object(self):
        name = "Qwen-Image Layered Control (bf16, 20 Schritte)"
        self.assertEqual(images.slug(name), ("qwen-image-layered-control", "edits"))
        g = api(name + ".json")
        self.assertIn("layers", images.extras(g))
        images.set_prompt(g, "the red electric guitar")
        self.assertEqual(g["83:6"]["inputs"]["text"], "the red electric guitar")
        self.assertEqual({n["inputs"]["unet_name"] for n in g.values() if n["class_type"] == "UNETLoader"},
                         {"qwen_image_layered_control_bf16.safetensors"})

    def test_inpaint_crop_repaints_a_crop_and_stitches_it_back(self):
        for name, sl in (("Qwen-Image 2.1 Inpaint Crop (bf16, dpmpp_2m 14)", "qwen-image-21-inpaint-crop"),
                         ("Qwen-Image 2.1 Inpaint Crop Turbo (bf16, 4 Schritte)", "qwen-image-21-inpaint-crop-turbo"),
                         ("Z-Image Turbo Inpaint Crop (bf16, 8 Schritte)", "z-image-turbo-inpaint-crop")):
            with self.subTest(name):
                self.assertEqual(images.slug(name), (sl, "edits"))
                g = api(name + ".json")
                kinds = {n["class_type"] for n in g.values()}
                self.assertIn("InpaintCropImproved", kinds)
                self.assertNotIn("ImageScaleToTotalPixels", kinds)     # the picture keeps its size
                crop = next(n for n in g.values() if n["class_type"] == "InpaintCropImproved")
                self.assertEqual(crop["inputs"]["context_from_mask_extend_factor"], 2.5)
                save = next(n for n in g.values() if n["class_type"] == "SaveImage")
                self.assertEqual(g[save["inputs"]["images"][0]]["class_type"], "InpaintStitchImproved")
                # an API mask replaces the one the picture's transparency gives the crop
                self.assertEqual([(n["class_type"], k) for n, k in images.mask_users(g)], [("InpaintCropImproved", "mask")])

    def test_birefnet_removes_the_background_without_a_prompt(self):
        for name, sl, model in (("BiRefNet Background Removal (general, MIT)", "birefnet-background-removal", "BiRefNet-general"),
                                ("BiRefNet Matting Background Removal (HR, MIT)", "birefnet-matting-background-removal", "BiRefNet-HR-matting")):
            with self.subTest(name):
                self.assertEqual(images.slug(name), (sl, "edits"))
                g = api(name + ".json")
                node = next(n for n in g.values() if n["class_type"] == "BiRefNetRMBG")
                self.assertEqual((node["inputs"]["model"], node["inputs"]["background"]), (model, "Alpha"))
                self.assertFalse(images.set_prompt(g, "anything"))
                wf = json.loads((ROOT / "comfyui" / "workflows" / "BiRefNet Background Removal.json").read_text())
                node = next(n for n in wf["nodes"] if n.get("type") == "BiRefNetRMBG" and n["widgets_values"][0] == model)
                entries = (node.get("properties") or {}).get("models") or []
                # birefnet.py is the node's: it rewrites an import in it on every load
                self.assertEqual({m["name"] for m in entries}, {f"{model}.safetensors", "config.json", "BiRefNet_config.py"})
                self.assertEqual({m["directory"] for m in entries}, {"RMBG/BiRefNet"})
                self.assertTrue(all("/resolve/4d000788a9698c7f8d67c8c6ce2b40c768f5b909/" in m["url"] for m in entries))

    def test_switches_and_their_variants(self):
        wfs = {p.stem: json.loads(p.read_text()) for p in (ROOT / "comfyui" / "workflows").glob("*.json")}
        for ui, vs in variants.VARIANTS.items():
            with self.subTest(ui):
                self.assertIn(ui, wfs)
                titles = {n.get("title") for n in wfs[ui]["nodes"] if n.get("type") == "PrimitiveBoolean"}
                if any(n.get("type") == variants.CHOICE for n in wfs[ui]["nodes"]):
                    titles.add("Steuerung")
                for name, settings in vs:
                    self.assertLessEqual(set(settings), titles, name)
        for ui, wf in wfs.items():                 # a switch without variants would never reach the API
            if ui not in variants.VARIANTS:
                self.assertFalse({n.get("title") for n in wf["nodes"] if n.get("type") == "PrimitiveBoolean"} & variants.SWITCH_TITLES, ui)
        retired = [line.split()[1:] for line in (ROOT / "comfyui" / "retired_workflows.txt").read_text().splitlines()
                   if line and not line.startswith("#")]
        for parts in retired:
            self.assertNotIn(" ".join(parts).removesuffix(".json"), wfs)

    def test_resolving_a_switch_cuts_the_other_branch(self):
        g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "a"}},
             "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "b"}},
             "3": {"class_type": "PrimitiveBoolean", "inputs": {"value": False}, "_meta": {"title": "XL"}},
             "4": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["3", 0], "on_false": ["1", 0], "on_true": ["2", 0]}},
             "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": 1}},
             "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}}}
        on = variants.resolve(g, {"XL": True})
        self.assertEqual(on["5"]["inputs"]["model"], ["2", 0])
        self.assertEqual(sorted(on), ["2", "5", "6"])
        self.assertEqual(variants.canonical(variants.resolve(g, {"XL": False})), variants.canonical(
            {k: g[k] for k in ("1", "6")} | {"5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 9}}}))
        with self.assertRaises(KeyError):
            variants.resolve(g, {"Turbo": True})

    def test_tools_are_no_models(self):
        old = vars(images.ARGS).get("api")
        images.ARGS.api = str(API)
        try:
            names = {p.stem for p in images.workflows().values()}
        finally:
            images.ARGS.api = old
        self.assertTrue(images.TOOLS)
        self.assertFalse(images.TOOLS & names)
        self.assertIn("Qwen-Image 2.1 Inpaint (bf16, dpmpp_2m 14)", names)
        for name, entries in (("Florence-2 Caption", {"model.safetensors", "config.json", "generation_config.json",
                                                      "preprocessor_config.json", "tokenizer.json",
                                                      "tokenizer_config.json", "vocab.json"}),
                              ("SAM 3 Select", {"sam3.1_multiplex_fp16.safetensors"})):
            with self.subTest(name):
                wf = json.loads((ROOT / "comfyui" / "workflows" / f"{name}.json").read_text())
                got = [m for n in wf["nodes"] for m in (n.get("properties") or {}).get("models") or []]
                self.assertEqual({m["name"] for m in got}, entries)       # not the .bin twin of the weights
                self.assertTrue(all("/resolve/" in m["url"] and "/main/" not in m["url"] for m in got))
        sam = next(n for n in api("SAM 3 Select.json").values() if n["class_type"] == "SAM3Segment")
        self.assertEqual(sam["inputs"]["output_mode"], "Separate")          # a box per match

    def test_select_makes_the_mask_from_words(self):
        Image = images.Image
        if Image is None:
            self.skipTest("needs Pillow")
        pic = Image.new("RGB", (200, 100), "white"); buf = io.BytesIO(); pic.save(buf, "PNG"); data = buf.getvalue()
        m1 = Image.new("L", (200, 100), 0); m1.paste(255, (20, 20, 40, 40))
        m2 = Image.new("L", (200, 100), 0); m2.paste(255, (150, 50, 160, 60))
        old = images.segments
        try:
            images.segments = lambda d, w: [m1, m2]
            mask = Image.open(io.BytesIO(images.select_mask(data, "the cups", 0))).getchannel("A")
            self.assertEqual((mask.getpixel((30, 30)), mask.getpixel((155, 55)), mask.getpixel((100, 80))), (0, 0, 255))
            grown = Image.open(io.BytesIO(images.select_mask(data, "the cups"))).getchannel("A")
            self.assertEqual(grown.getpixel((40 + images.SELECT_MARGIN // 2, 30)), 0)   # the default margin reaches out
            self.assertEqual(grown.getpixel((100, 5)), 255)
            images.segments = lambda d, w: []
            with self.assertRaises(images.Refused):
                images.select_mask(data, "a giraffe")
        finally:
            images.segments = old
        with self.assertRaises(images.Refused):
            images.segments(data, "  ")
        g = api("Qwen-Image 2.1 Inpaint Crop Turbo (bf16, 4 Schritte).json")
        old_upload = images.upload
        images.upload = lambda d: "x.png"
        try:
            with self.assertRaises(images.Refused):
                images.prepare(g, "p", None, [data], boxes=[[1, 1, 5, 5]], select="the sofa")
        finally:
            images.upload = old_upload

    def test_layered_without_a_prompt_is_described(self):
        seen = {}
        olds = images.load, images.caption, images.run
        images.load = lambda name, kind: api("Qwen-Image Layered (bf16, 20 Schritte).json")
        images.caption = lambda data: "a woman on a street"
        def run(graph):
            seen["prompt"] = [n["inputs"].get("text") for n in graph.values() if n["class_type"] == "CLIPTextEncode"]
            return []
        images.run = run
        old_upload = images.upload
        images.upload = lambda d: "x.png"
        try:
            with images.app.app_context():
                images.respond("qwen-image-layered", "edits", "", None, 1, 1, "b64_json", [b"png"])
            self.assertIn("a woman on a street", seen["prompt"])
            with images.app.app_context():
                images.respond("qwen-image-layered", "edits", "my own", None, 1, 1, "b64_json", [b"png"])
            self.assertIn("my own", seen["prompt"])
        finally:
            images.load, images.caption, images.run = olds
            images.upload = old_upload

    def test_music_requests_fill_each_music_workflow(self):
        old = vars(images.ARGS).get("api")
        images.ARGS.api = str(API)
        try:
            known = images.music_workflows()
        finally:
            images.ARGS.api = old
        self.assertEqual(set(known), {"ace-step-15-xl-turbo", "ace-step-15-turbo", "stable-audio-3-medium", "yue2-text2music"})
        g = json.loads(known["ace-step-15-xl-turbo"].read_text())
        images.set_music(g, "german pop ballad", "[Verse]\nla la", 90, 7, 88, "C major", "de")
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeAceStepAudio1.5")["inputs"]
        self.assertEqual((enc["tags"], enc["lyrics"], enc["duration"], enc["bpm"], enc["keyscale"], enc["language"]),
                         ("german pop ballad", "[Verse]\nla la", 90, 88, "C major", "de"))
        lat = next(n for n in g.values() if n["class_type"] == "EmptyAceStep1.5LatentAudio")["inputs"]
        self.assertTrue(isinstance(lat["seconds"], list) or lat["seconds"] == 90)
        kinds = {n["class_type"] for n in g.values()}
        self.assertIn("PreviewAudio", kinds)                        # into temp, not the gallery
        self.assertFalse({k for k in kinds if k.startswith("SaveAudio")})
        self.assertTrue(all(n["inputs"]["seed"] == 7 for n in g.values() if isinstance(n["inputs"].get("seed"), int)))
        y = json.loads(known["yue2-text2music"].read_text())
        images.set_music(y, "indie rock", None, 30)
        vals = {images.title(n): n["inputs"].get("value") for n in y.values() if n["class_type"] == "PrimitiveStringMultiline"}
        self.assertIn("indie rock", vals.values())
        self.assertIn("[Instrumental]", vals.values())
        self.assertEqual(next(n for n in y.values() if n["class_type"] == "YuE2GenerateMusic")["inputs"]["max_duration"], 30)
        a = json.loads(known["stable-audio-3-medium"].read_text())
        images.set_music(a, "a dusty drum loop", None, 10.7)
        self.assertIn(10.7, [n["inputs"].get("value") for n in a.values() if n["class_type"] == "PrimitiveFloat"])
        with self.assertRaises(images.Refused):                      # it does not sing
            images.set_music(json.loads(known["stable-audio-3-medium"].read_text()), "x", "[Verse] words", 30)
        for bad in (2, 301):
            with self.subTest(bad), self.assertRaises(images.Refused):
                images.set_music(json.loads(known["ace-step-15-turbo"].read_text()), "x", None, bad)

    def test_model_lists_in_both_object_info_formats(self):
        info = {"UNETLoader": {"input": {"required": {"unet_name": [["a.safetensors"], {}]}}},
                "UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", {"options": ["x4.safetensors"]}]}}}}
        class R:
            def json(self): return info
        old_get, images.requests.get = images.requests.get, lambda *a, **k: R()
        old_comfy = getattr(images.ARGS, "comfy", None)
        images.ARGS.comfy = "http://127.0.0.1:9"
        images._have.update(at=0.0, files={})
        try:
            files = images.model_files()
        finally:
            images.requests.get = old_get
            images.ARGS.comfy = old_comfy
            images._have.update(at=0.0, files={})
        self.assertEqual(files["unet_name"], {"a.safetensors"})
        self.assertEqual(files["model_name"], {"x4.safetensors"})

    def test_the_2k_upscaler_keeps_its_own_prompt(self):
        self.assertEqual(images.slug("Z-Image Turbo 2K Upscale (bf16, 5 Schritte)"), ("z-image-turbo-2k-upscale", "edits"))
        g = api("Z-Image Turbo 2K Upscale (bf16, 5 Schritte).json")
        images.set_prompt(g, "PROMPT")
        self.assertNotIn("PROMPT", json.dumps(g))
        self.assertIn("masterpiece, 8k", json.dumps(g))

    def test_a_negative_prompt_only_where_guidance_uses_it(self):
        g = api("Z-Image T2I (bf16, 25 Schritte).json")
        self.assertIn("negative_prompt", images.extras(g))
        images.set_negative(g, "NEGATIVE")
        self.assertIn("NEGATIVE", json.dumps(g))
        images.set_prompt(g, "POSITIVE")
        neg = [n for n in g.values() if "Negative" in images.title(n)]
        self.assertEqual(neg[0]["inputs"]["text"], "NEGATIVE")          # the prompt went elsewhere
        turbo = api("Z-Image Turbo T2I (bf16, 8 Schritte).json")       # cfg 1: no negative to steer by
        self.assertNotIn("negative_prompt", images.extras(turbo))
        with self.assertRaises(images.Refused):
            images.set_negative(turbo, "x")
        self.assertEqual(images.extras(api("Z-Image Turbo Inpaint (bf16, 8 Schritte).json")), ["mask", "boxes", "select", "strength"])
        self.assertEqual(images.slug("Qwen-Image 2.1 Pose Inpaint Turbo (bf16, 4 Schritte)"),
                         ("qwen-image-21-pose-inpaint-turbo", "edits"))
        for kind in ("Canny", "Pose", "Depth"):
            with self.subTest(kind=kind):
                g = api(f"Qwen-Image 2.1 {kind} Inpaint Turbo (bf16, 4 Schritte).json")
                self.assertEqual(images.extras(g), ["mask", "boxes", "select", "strength"])

    def test_the_detailer_keeps_its_own_prompts(self):
        g = api("Qwen-Image 2.1 Detailer (bf16, dpmpp_2m 14).json")
        before = sorted(n["inputs"]["prompt"] for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")
        images.set_prompt(g, "PROMPT")
        self.assertNotIn("PROMPT", json.dumps(g))
        self.assertEqual(sorted(n["inputs"]["prompt"] for n in g.values() if n["class_type"] == "TextEncodeQwenImage21"), before)

    def test_each_generation_takes_prompt_size_and_seed(self):
        for path in sorted(API.glob("*T2I*.json")):
            with self.subTest(workflow=path.name):
                g = json.loads(path.read_text())
                self.assertTrue(images.set_prompt(g, "PROMPT"))
                images.set_size(g, "1000x600")
                images.set_seed(g, 1234)
                text = json.dumps(g)
                self.assertIn("PROMPT", text)
                self.assertIn("1008", text)           # rounded to 16
                self.assertIn("608", text)
                self.assertIn(": 1234", text)

    def test_a_size_that_is_no_size_is_refused(self):
        with self.assertRaises(images.Refused):
            images.set_size({}, "large")

    def _prepared(self, name, n_images):
        g = api(name)
        images.upload = lambda data: "llmctl-api/x.png"          # no ComfyUI here
        images.prepare(g, "PROMPT", None, [b"png"] * n_images)
        return g

    def _refs_ok(self, g):
        for key, node in g.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], g, f"{key} points at a node that is gone")

    def test_a_missing_reference_drops_out_of_a_reference_chain(self):
        g = self._prepared("FLUX.2 dev Edit (fp8, 20 Schritte).json", 1)
        self.assertEqual(sum(n["class_type"] == "LoadImage" for n in g.values()), 1)
        self.assertEqual(sum(n["class_type"] == "ReferenceLatent" for n in g.values()), 1)
        self._refs_ok(g)

    def test_a_missing_reference_drops_out_of_qwen_images(self):
        g = self._prepared("Qwen-Image 2.1 Edit (bf16, dpmpp_2m 14).json", 1)
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")
        self.assertIn("images.image_1", enc["inputs"])
        self.assertNotIn("images.image_2", enc["inputs"])
        self._refs_ok(g)

    def test_all_references_used_leaves_the_chain_whole(self):
        g = self._prepared("FLUX.2 dev Edit (fp8, 20 Schritte).json", 2)
        self.assertEqual(sum(n["class_type"] == "ReferenceLatent" for n in g.values()), 2)
        self._refs_ok(g)

    def test_too_many_images_are_refused(self):
        with self.assertRaises(images.Refused):
            self._prepared("FLUX.2 klein 9B Edit (bf16).json", 5)

    def test_results_go_to_temp_not_the_gallery(self):
        for path in sorted(API.glob("*.json")):
            if path.stem in images.TOOLS:
                continue                                  # run as they are, outputs read directly
            with self.subTest(workflow=path.name):
                g = json.loads(path.read_text())
                n = len(images.image_nodes(g))
                images.upload = lambda data: "llmctl-api/x.png"
                images.prepare(g, "PROMPT", None, [b"png"] * n if n else None)
                kinds = {x["class_type"] for x in g.values()}
                self.assertIn("PreviewImage", kinds)
                self.assertFalse(kinds & images.SAVE)
                self.assertFalse(kinds & (images.UI_ONLY - {"PreviewImage"}))
                self._refs_ok(g)


def png(w, h, alpha=True):
    """Enough of a PNG for the header checks: signature and IHDR."""
    return (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
            + w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 6 if alpha else 2, 0, 0, 0]))


@unittest.skipIf(images is None, "images_server needs flask and requests")
class ImageApiExtras(unittest.TestCase):
    INPAINT = "Qwen-Image 2.1 Inpaint Turbo (bf16, 4 Schritte).json"
    OUTPAINT = "Qwen-Image 2.1 Outpaint (bf16, dpmpp_2m 14).json"
    CONTROL = "Qwen-Image 2.1 Control (bf16, dpmpp_2m 14).json"
    EDIT = "Qwen-Image 2.1 Edit (bf16, dpmpp_2m 14).json"

    def setUp(self):
        images.upload = lambda data: "llmctl-api/x.png"

    def _prepared(self, name, **extra):
        g = api(name)
        images.prepare(g, "PROMPT", None, [png(64, 48)], **extra)
        return g

    def test_each_workflow_names_the_extras_it_takes(self):
        self.assertEqual(images.extras(api(self.INPAINT)), ["mask", "boxes", "select", "strength"])
        self.assertEqual(images.extras(api(self.OUTPAINT)), ["strength", "pad"])
        self.assertEqual(images.extras(api(self.CONTROL)), ["strength"])
        self.assertEqual(images.extras(api(self.EDIT)), [])
        self.assertEqual(images.extras(api("SeedVR2 7B Upscale (fp16).json")), [])  # keeps alpha, redraws nothing

    def test_a_mask_replaces_the_images_transparency(self):
        g = self._prepared(self.INPAINT, mask=png(64, 48))
        key, node = next((k, n) for k, n in g.items() if n["class_type"] == "LoadImageMask")
        self.assertEqual(node["inputs"]["channel"], "alpha")
        slot = images.image_nodes(g)[0]
        self.assertFalse([v for n in g.values() for v in n["inputs"].values() if v == [slot, 1]])
        self.assertTrue([v for n in g.values() for v in n["inputs"].values() if v == [key, 0]])

    def test_a_mask_that_does_not_fit_is_refused(self):
        for mask, why in ((png(64, 64), "size"), (png(64, 48, alpha=False), "alpha"), (b"GIF89a", "not PNG")):
            with self.subTest(why), self.assertRaises(images.Refused):
                self._prepared(self.INPAINT, mask=mask)
        with self.assertRaises(images.Refused):
            self._prepared(self.EDIT, mask=png(64, 48))

    def test_strength_goes_into_the_controlnet(self):
        g = self._prepared(self.CONTROL, strength="0.8")
        self.assertEqual({n["inputs"]["strength"] for n in g.values()
                          if n["class_type"] == "ZImageFunControlnet"}, {0.8})
        for bad in (-0.1, 2.5, "strong"):
            with self.subTest(bad), self.assertRaises(images.Refused):
                self._prepared(self.CONTROL, strength=bad)
        with self.assertRaises(images.Refused):
            self._prepared(self.EDIT, strength=0.5)

    def test_pad_sets_the_sides(self):
        pad = lambda g: next(n["inputs"] for n in g.values() if n["class_type"] == "ImagePadForOutpaint")
        self.assertEqual({pad(self._prepared(self.OUTPAINT, pad=64))[k] for k in ("left", "top", "right", "bottom")}, {64})
        p = pad(self._prepared(self.OUTPAINT, pad='{"top": 128, "feathering": 20}'))
        self.assertEqual((p["left"], p["top"], p["right"], p["bottom"], p["feathering"]), (0, 128, 0, 0, 20))
        for bad in (0, 12, {"up": 64}, {"left": 4096}, {"left": 64, "feathering": -1}, True):
            with self.subTest(bad), self.assertRaises(images.Refused):
                self._prepared(self.OUTPAINT, pad=bad)
        with self.assertRaises(images.Refused):
            self._prepared(self.INPAINT, pad=64)


@unittest.skipIf(images is None or images.Image is None, "images_server needs flask, requests and pillow")
class ImageCheck(unittest.TestCase):
    def test_boxes_make_a_mask_transparent_inside_them(self):
        from PIL import Image
        m = Image.open(io.BytesIO(images.boxes_mask([[10, 10, 20, 20]], (100, 50), margin=5)))
        self.assertEqual(m.size, (100, 50))
        self.assertEqual(m.getpixel((15, 15))[3], 0)        # inside
        self.assertEqual(m.getpixel((6, 6))[3], 0)          # in the margin
        self.assertEqual(m.getpixel((40, 40))[3], 255)      # kept
        m = Image.open(io.BytesIO(images.boxes_mask("[[-50, -50, 500, 10]]", (100, 50), margin=0)))
        self.assertEqual(m.getpixel((99, 0))[3], 0)         # clipped to the picture

    def test_boxes_that_are_no_boxes_are_refused(self):
        for boxes in ([], [[1, 2, 3]], "nonsense", [[1, 2, 3, True]]):
            with self.subTest(boxes), self.assertRaises(images.Refused):
                images.boxes_mask(boxes, (100, 100))
        with self.assertRaises(images.Refused):
            images.boxes_mask([[1, 2, 3, 4]], (100, 100), margin=1000)

    def test_boxes_go_into_the_inpaint_mask_and_not_elsewhere(self):
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (64, 48)).save(buf, "PNG")
        images.upload = lambda data: "llmctl-api/x.png"
        g = api(ImageApiExtras.INPAINT)
        images.prepare(g, "P", None, [buf.getvalue()], boxes=[[1, 1, 9, 9]])
        self.assertIn("LoadImageMask", {n["class_type"] for n in g.values()})
        for extra in ({"boxes": [[1, 1, 9, 9]], "mask": b"x"},):
            with self.assertRaises(images.Refused):
                images.prepare(api(ImageApiExtras.INPAINT), "P", None, [buf.getvalue()], **extra)
        with self.assertRaises(images.Refused):
            images.prepare(api(ImageApiExtras.EDIT), "P", None, [buf.getvalue()], boxes=[[1, 1, 9, 9]])

    def test_a_vision_answer_is_read_in_its_shapes(self):
        obj = '<think>hm</think>```json\n{"artifacts": [{"what": "hand", "fix": "a hand", "bbox_2d": [1, 2, 3, 4]}]}\n```'
        bare = '[{"label": "text", "bbox_2d": [5, 6, 7, 8]}, {"what": "no box"}]'
        self.assertEqual(images.parse_artifacts(obj), [{"what": "hand", "fix": "a hand", "bbox_2d": [1, 2, 3, 4]}])
        self.assertEqual(images.parse_artifacts(bare), [{"what": "text", "fix": "", "bbox_2d": [5, 6, 7, 8]}])
        self.assertEqual(images.parse_artifacts('{"artifacts": []}'), [])
        self.assertIsNone(images.parse_artifacts("I see no problems."))

    def test_boxes_become_pixels_and_whole_picture_ones_remarks(self):
        places, remarks = images.to_pixels([
            {"what": "a", "fix": "", "bbox_2d": [100, 200, 300, 400]},
            {"what": "b", "fix": "", "bbox_2d": [0, 0, 1000, 1000]},
            {"what": "c", "fix": "", "bbox_2d": [900, 900, 1200, 800]}], (2000, 1000))
        self.assertEqual([p["box"] for p in places], [[200, 200, 600, 400], [1800, 800, 2000, 900]])
        self.assertEqual([p["id"] for p in places], [1, 2])
        self.assertEqual([r["what"] for r in remarks], ["b"])
        self.assertEqual(remarks[0]["box"], [0, 0, 2000, 1000])       # still there to use by hand
        places, remarks = images.to_pixels([{"what": "a", "fix": "", "bbox_2d": [100, 200, 300, 400]}],
                                           (2000, 1000), max_area=0.03)
        self.assertEqual((len(places), len(remarks)), (0, 1))         # 4 % of the picture

    def test_max_area_is_a_share(self):
        self.assertEqual(images.area_limit(None), images.WHOLE_SHARE)
        self.assertEqual(images.area_limit("0.1"), 0.1)
        for bad in (0, 1.5, "much"):
            with self.subTest(bad), self.assertRaises(images.Refused):
                images.area_limit(bad)

    def test_without_a_vision_model_a_check_is_refused(self):
        with self.assertRaises(images.Refused):
            images.check(b"png")


@unittest.skipIf(images is None, "images_server needs flask and requests")
class RepairPage(unittest.TestCase):
    def setUp(self):
        self.args = vars(images.ARGS).copy()
        images.ARGS.comfy = "http://127.0.0.1:9"          # nothing listens there
        images.ARGS.vision = ""
        for k in ("vision_slot", "vision_model", "llmctl"):
            setattr(images.ARGS, k, None)
        self.client = images.app.test_client()

    def tearDown(self):
        images.ARGS.__dict__.clear(); images.ARGS.__dict__.update(self.args)

    def test_the_page_groups_boxes_by_the_workflows_crop(self):
        page = (ROOT / "images_repair.html").read_text()
        builder = (ROOT / "comfyui" / "build_control_workflows.py").read_text()
        self.assertEqual(re.search(r"const CROP_CONTEXT = ([\d.]+);", page)[1],
                         re.search(r"^CROP_CONTEXT = ([\d.]+)$", builder, re.M)[1])
        first = re.search(r'<select id="model"><option value="([^"]+)"', page)[1]
        self.assertEqual(first, "qwen-image-21-inpaint-crop-turbo")
        for m in re.findall(r'<option value="(qwen-image-21-inpaint[^"]*)"', page):
            self.assertIn(m, {images.slug(p.stem)[0] for p in (ROOT / "comfyui" / "api").glob("*.json")})

    def test_the_page_finds_places_by_word(self):
        page = (ROOT / "images_repair.html").read_text()
        self.assertIn('id="word"', page)
        self.assertIn('jsonPost("v1/images/select"', page)
        self.assertIn("/v1/images/select", [r.rule for r in images.app.url_map.iter_rules()])

    def test_the_page_is_served(self):
        r = self.client.get("/repair")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Bildreparatur", r.data)

    def test_status_without_a_vision_slot(self):
        j = self.client.get("/repair/status").get_json()
        self.assertEqual(j["vision"]["state"], "down")
        self.assertFalse(j["vision"]["configured"])
        self.assertFalse(j["vision"]["controllable"])
        self.assertFalse(j["comfy"]["up"])

    def test_vision_cannot_be_started_without_llmctl_and_a_slot(self):
        r = self.client.post("/repair/vision", json={"action": "start"})
        self.assertEqual(r.status_code, 400)
        images.ARGS.llmctl, images.ARGS.vision_slot, images.ARGS.vision_model = "/bin/false", 2, "flash"
        self.assertEqual(self.client.post("/repair/vision", json={"action": "reboot"}).status_code, 400)

    def test_only_plain_output_names_are_fetched(self):
        for name in ("../secret.png", "a/b.png", ".hidden.png", ""):
            with self.subTest(name):
                self.assertEqual(self.client.get("/repair/image", query_string={"name": name}).status_code, 400)


NODES = ROOT / "comfyui" / "nodes"


class ArtifactNode(unittest.TestCase):
    """The helpers of llmctl's ComfyUI node, without ComfyUI."""
    @classmethod
    def setUpClass(cls):
        import importlib.util
        import types
        sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
        spec = importlib.util.spec_from_file_location("llmctl_nodes", NODES / "llmctl_nodes" / "__init__.py")
        cls.n = importlib.util.module_from_spec(spec)
        try:
            with warnings.catch_warnings():                   # torch's ROCm build is chatty
                warnings.simplefilter("ignore")
                spec.loader.exec_module(cls.n)                # numpy, torch, pillow, requests
        except ImportError as e:
            raise unittest.SkipTest(f"the node needs {e.name}")

    def test_the_mask_covers_the_grown_boxes(self):
        m = self.n.mask_of([[10, 10, 20, 20]], 5, (100, 50))
        self.assertEqual(m.shape, (50, 100))
        self.assertEqual((m[5, 5], m[24, 24], m[26, 26], m[40, 90]), (1.0, 1.0, 0.0, 0.0))

    def test_the_repair_prompt_joins_the_fixes_once(self):
        a = [{"fix": "a hand"}, {"fix": "a hand"}, {"fix": " a cup "}, {"fix": ""}]
        self.assertEqual(self.n.repair_prompt("a kitchen", a), "a kitchen. a hand; a cup")
        self.assertEqual(self.n.repair_prompt("", a), "a hand; a cup")
        self.assertEqual(self.n.repair_prompt("A kitchen.", [{"fix": "A hand."}, {"fix": "A cup."}]),
                         "A kitchen. A hand; A cup")

    def test_joined_clips_take_comfyuis_types(self):
        types = self.n.LlmctlJoinClips.INPUT_TYPES()
        inputs = {**types["required"], **types["optional"]}
        self.assertEqual({v[0] for k, v in inputs.items() if k.startswith("images_")}, {"IMAGE"})
        self.assertEqual({v[0] for k, v in inputs.items() if k.startswith("audio_")}, {"AUDIO"})

    def test_joins_are_crossfaded_without_a_dip(self):
        import torch
        rate, fps = 48000, 24
        n = round(121 * rate / fps)
        frame = rate // fps
        # Two clips of the same level, clip 2 a sign flip of clip 1: a hard cut
        # steps from +1 to -1; a crossfade in the shared frame passes smoothly.
        clips = {"images_1": torch.zeros(121, 4, 4, 3), "audio_1": {"waveform": torch.ones(1, 2, n), "sample_rate": rate},
                 "images_2": torch.zeros(121, 4, 4, 3), "audio_2": {"waveform": -torch.ones(1, 2, n), "sample_rate": rate}}
        hard = self.n.LlmctlJoinClips().join(fps, 0.0, **clips)[1]["waveform"]
        soft = self.n.LlmctlJoinClips().join(fps, 0.04, **clips)[1]["waveform"]
        self.assertEqual(hard.shape, soft.shape)                       # the length, and with it the sync, stays
        at = n                                                         # clip 2's kept sound begins here
        self.assertEqual((hard[0, 0, at - 1].item(), hard[0, 0, at].item()), (1.0, -1.0))
        self.assertTrue(torch.equal(hard[..., :at - frame], soft[..., :at - frame]))   # only the shared frame changes
        self.assertTrue(torch.equal(hard[..., at:], soft[..., at:]))
        self.assertLess(soft[0, 0, at - frame - 1:at + 1].diff().abs().max().item(), 0.01)
        # equal power: two sounds of the same level do not dip
        same = dict(clips, audio_2={"waveform": torch.ones(1, 2, n), "sample_rate": rate})
        level = self.n.LlmctlJoinClips().join(fps, 0.04, **same)[1]["waveform"][0, 0, at - frame:at]
        self.assertGreaterEqual(level.min().item(), 0.99)

    def test_joined_clips_keep_the_sound_on_the_pictures(self):
        import torch
        rate, fps = 48000, 24
        # Three 5 s clips (121 frames each), their sound a little long, short and
        # in mono; each sample holds its clip's number, so a join shows up.
        lengths = [round(121 * rate / fps) + 700, round(121 * rate / fps) - 900, round(121 * rate / fps)]
        clips = {}
        for i, (n, ch) in enumerate(zip(lengths, (2, 2, 1)), 1):
            clips[f"images_{i}"] = torch.full((121, 8, 8, 3), float(i))
            clips[f"audio_{i}"] = {"waveform": torch.full((1, ch, n), float(i)), "sample_rate": rate}
        images, audio = self.n.LlmctlJoinClips().join(fps, **clips)
        wave = audio["waveform"]
        self.assertEqual(images.shape[0], 121 + 120 + 120)            # the repeated start frames dropped
        self.assertEqual(wave.shape, (1, 2, round(images.shape[0] * rate / fps)))
        clip2 = round(121 * rate / fps)                                # where clip 2's second frame begins
        self.assertEqual((wave[0, 0, clip2 - 1].item(), wave[0, 0, clip2].item()), (1.0, 2.0))
        clip3 = round(241 * rate / fps)
        # clip 2 came out 900 samples short: silence up to clip 3, which is mono made stereo
        self.assertEqual((wave[0, 1, clip3 - 1].item(), wave[0, 1, clip3].item()), (0.0, 3.0))


import hashlib
import hf_parts                                                   # comfyui_models: stdlib only


class Parts(unittest.TestCase):
    """Byte parts on the Hub: joined, checked, recorded, deleted — without the Hub."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = [b"A" * 1000, b"B" * 700]
        self.names = ["m.Q6_K.gguf.part1of2", "m.Q6_K.gguf.part2of2"]
        self.hub = {n: {"size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in zip(self.names, self.data)}
        hf_parts.hub_files = lambda repo, revision="main": (self.hub, "c0ffee")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_parts(self):
        for n, d in zip(self.names, self.data):
            (self.tmp / n).write_bytes(d)
        meta = self.tmp / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        for n in self.names:
            (meta / (n + ".metadata")).write_text("x")

    def test_complete_sets_only_and_never_split_ggufs(self):
        g = hf_parts.groups(["a.gguf.part1of2", "a.gguf.part2of2", "b.gguf.part1of3", "b.gguf.part3of3",
                             "c-00001-of-00002.gguf", "c-00002-of-00002.gguf"])
        self.assertEqual(g, {"a.gguf": ["a.gguf.part1of2", "a.gguf.part2of2"]})

    def test_join_checks_writes_the_companion_and_deletes_the_parts(self):
        self.write_parts()
        with contextlib.redirect_stdout(io.StringIO()):
            hf_parts.cmd_join("org/repo", str(self.tmp))
        joined = self.tmp / "m.Q6_K.gguf"
        self.assertEqual(joined.read_bytes(), b"".join(self.data))
        c = json.loads((self.tmp / "m.Q6_K.gguf.parts.json").read_text())
        self.assertEqual(c["sha256"], hashlib.sha256(b"".join(self.data)).hexdigest())
        self.assertEqual([p["name"] for p in c["parts"]], self.names)
        self.assertEqual(c["revision"], "c0ffee")
        self.assertFalse(any((self.tmp / n).exists() for n in self.names))
        self.assertFalse(list((self.tmp / ".cache").rglob("*.metadata")))

    def test_a_made_file_records_its_sources_and_is_made_again(self):
        # Two Q8_0 shards on the Hub, quantized down here; a stand-in for
        # llama-quantize copies its input, so the output is known.
        shards = {"m-Q8_0-00001-of-00002.gguf": b"Q" * 900, "m-Q8_0-00002-of-00002.gguf": b"R" * 400}
        self.hub = {n: {"size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in shards.items()}
        for n, d in shards.items():
            (self.tmp / n).write_bytes(d)
        tool = self.tmp / "fake-quantize"
        tool.write_text('#!/bin/sh\ncp "$2" "$3"\n')
        tool.chmod(0o755)
        os.environ["LLAMA_QUANTIZE"] = str(tool)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hf_parts.cmd_derive("org/repo", str(self.tmp), "m-Q5_K_M.gguf", "Q5_K_M")
            c = json.loads((self.tmp / "m-Q5_K_M.gguf.parts.json").read_text())
            self.assertEqual(c["derive"]["quantize"], "Q5_K_M")
            self.assertEqual([p["name"] for p in c["parts"]], sorted(shards))
            self.assertEqual((self.tmp / "m-Q5_K_M.gguf").read_bytes(), b"Q" * 900)   # from the first shard
            self.assertFalse(any((self.tmp / n).exists() for n in shards))              # the sources are gone
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                hf_parts.cmd_excludes("org/repo", str(self.tmp))
                hf_parts.cmd_check("org/repo", str(self.tmp))
            self.assertIn("m-Q8_0-00001-of-00002.gguf", out.getvalue())                 # not fetched again
            self.assertIn("current\tm-Q5_K_M.gguf\t2 file(s)\tmade (Q5_K_M) from", out.getvalue())
            # The Hub changes; download fetches the new sources; join makes it again.
            shards = {"m-Q8_0-00001-of-00002.gguf": b"S" * 950, "m-Q8_0-00002-of-00002.gguf": b"T" * 400}
            self.hub = {n: {"size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in shards.items()}
            for n, d in shards.items():
                (self.tmp / n).write_bytes(d)
            with contextlib.redirect_stdout(io.StringIO()):
                hf_parts.cmd_join("org/repo", str(self.tmp))
            self.assertEqual((self.tmp / "m-Q5_K_M.gguf").read_bytes(), b"S" * 950)
            self.assertFalse(any((self.tmp / n).exists() for n in shards))
        finally:
            del os.environ["LLAMA_QUANTIZE"]

    def test_a_second_repo_in_the_directory_does_not_see_the_parts(self):
        # A vision projector from another repo lands beside the joined model;
        # checking that repo must not report the model's parts as changed.
        self.write_parts()
        with contextlib.redirect_stdout(io.StringIO()):
            hf_parts.cmd_join("org/repo", str(self.tmp))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            hf_parts.cmd_check("org/mmproj", str(self.tmp))
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(len(hf_parts.companions(self.tmp, "org/repo")), 1)

    def test_an_existing_joined_file_is_adopted_when_it_matches(self):
        self.write_parts()
        joined = self.tmp / "m.Q6_K.gguf"
        joined.write_bytes(b"".join(self.data))
        ino = joined.stat().st_ino
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hf_parts.cmd_join("org/repo", str(self.tmp))
        self.assertIn("already joined", out.getvalue())
        self.assertEqual(joined.stat().st_ino, ino)                 # not rewritten
        self.assertFalse((self.tmp / self.names[0]).exists())

    def test_a_part_that_differs_from_the_hub_is_kept(self):
        self.write_parts()
        self.hub[self.names[1]]["sha256"] = "0" * 64
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            hf_parts.cmd_join("org/repo", str(self.tmp))
        self.assertTrue(all((self.tmp / n).exists() for n in self.names))
        self.assertFalse((self.tmp / "m.Q6_K.gguf.parts.json").exists())

    def test_excludes_and_check_follow_the_hub(self):
        self.write_parts()
        with contextlib.redirect_stdout(io.StringIO()):
            hf_parts.cmd_join("org/repo", str(self.tmp))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hf_parts.cmd_excludes("org/repo", str(self.tmp))
        self.assertEqual(out.getvalue().split(), self.names)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hf_parts.cmd_check("org/repo", str(self.tmp))
        self.assertTrue(out.getvalue().startswith("current\tm.Q6_K.gguf\t2"))
        self.hub[self.names[0]]["sha256"] = "1" * 64              # a new upload on the Hub
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hf_parts.cmd_excludes("org/repo", str(self.tmp))
        self.assertEqual(out.getvalue(), "")                       # download fetches them again
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hf_parts.cmd_check("org/repo", str(self.tmp))
        self.assertTrue(out.getvalue().startswith("changed"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


import llama_builds as lb                                          # noqa: E402


class LlamaBuilds(unittest.TestCase):
    def test_gpu_targets(self):
        self.assertEqual([lb.gfx_name(v) for v in (110501, 100300, 90010, 120001)],
                         ["gfx1151", "gfx1030", "gfx90a", "gfx1201"])
        with tempfile.TemporaryDirectory() as d:
            for node, v in (("0", 0), ("1", 110501)):
                Path(d, node).mkdir()
                Path(d, node, "properties").write_text(f"cpu_cores_count 16\ngfx_target_version {v}\n")
            self.assertEqual(lb.gfx(d), "gfx1151")

    def test_the_rocm_zip_for_a_target_or_its_family(self):
        assets = [{"name": f"llama-b1330-ubuntu-rocm-{t}-x64.zip"} for t in ("gfx103X", "gfx110X", "gfx1150", "gfx1151", "gfx120X")]
        assets.append({"name": "llama-b1330-windows-rocm-gfx1151-x64.zip"})
        pick = lambda t: (lb.lemonade_asset(assets, t) or {}).get("name")
        self.assertEqual(pick("gfx1151"), "llama-b1330-ubuntu-rocm-gfx1151-x64.zip")
        self.assertEqual(pick("gfx1032"), "llama-b1330-ubuntu-rocm-gfx103X-x64.zip")
        self.assertIsNone(pick("gfx942"))


import watch                                                        # noqa: E402


class Watch(unittest.TestCase):
    def test_versions_and_awaited_templates(self):
        self.assertGreater(watch.version_key("2.10"), watch.version_key("2.9"))
        self.assertGreater(watch.version_key("3"), watch.version_key("2.5"))
        with tempfile.TemporaryDirectory() as d:
            t = Path(d, "templates"); t.mkdir()
            for n in ("video_ltx2_5_i2v", "video_ltx2_5_ia2v"):
                (t / f"{n}.json").write_text("{}")
            conf = Path(d, "watch.conf")
            conf.write_text("# comment\ntemplate ^video_ltx2_5_.*ia2v #43 talking avatar\n"
                            "template ^video_ltx2_5_.*id_lora #45 identity\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = watch.main(str(conf), str(t))
            self.assertEqual(rc, 1)
            self.assertIn("#43: ComfyUI has video_ltx2_5_ia2v", out.getvalue())
            self.assertIn("#45: waiting", out.getvalue())
