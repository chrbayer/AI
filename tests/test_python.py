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
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import comfyui_models as cm                                        # stdlib only

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


class Sources(unittest.TestCase):
    def test_the_bundled_recipes_are_complete(self):
        data = json.loads((ROOT / "comfyui" / "sources.json").read_text())
        for key, recipe in data.items():
            if key.startswith("_"):
                continue
            with self.subTest(recipe=key):
                self.assertIn("/", key)                  # <directory>/<name>
                for field in ("repo", "revision", "shards", "tensors", "bytes"):
                    self.assertIn(field, recipe)
                self.assertEqual(len(recipe["revision"]), 40)   # a pinned commit

    def test_every_bundled_workflow_parses_and_names_its_models(self):
        for path in sorted((ROOT / "comfyui" / "workflows").glob("*.json")):
            with self.subTest(workflow=path.name):
                wf = json.loads(path.read_text())
                self.assertTrue(cm.refs(wf), "names no models at all")
                for key, urls in cm.refs(wf).items():
                    self.assertTrue(urls, f"{key} has no URL")


API = ROOT / "comfyui" / "api"


def api(name):
    return json.loads((API / name).read_text())


@unittest.skipIf(images is None, "images_server needs flask and requests")
class ImageApi(unittest.TestCase):
    def test_every_image_workflow_has_an_api_version(self):
        for path in sorted((ROOT / "comfyui" / "workflows").glob("*.json")):
            if "TTS" in path.name:
                continue
            with self.subTest(workflow=path.name):
                self.assertTrue((API / path.name).exists(),
                                "run comfyui/export_api.py and commit what it writes")

    def test_workflow_names(self):
        self.assertEqual(images.slug("FLUX.2 klein 9B T2I (bf16, 4 Schritte)"), ("flux2-klein-9b", "generations"))
        self.assertEqual(images.slug("FLUX.2 klein 9B NSFW Edit (bf16)"), ("flux2-klein-9b-nsfw", "edits"))
        self.assertEqual(images.slug("Qwen-Image 2.1 Heretic T2I (bf16, dpmpp_2m 14)"),
                         ("qwen-image-21-heretic", "generations"))
        self.assertEqual(images.slug("SeedVR2 7B Upscale (fp16)"), ("seedvr2-7b-upscale", "edits"))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
