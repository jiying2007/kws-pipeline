#!/usr/bin/env python3
"""No model downloads: test sample accounting, immutable cache and audio binding."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_zipformer_system as target


class Engine:
    def __init__(self):
        self.keyword_spotter = self
        self.chunks = []
        self.ready = False
        self.eof = False
        self.finish_calls = 0
        self.resets = 0

    def create_stream(self):
        return self

    def accept_waveform(self, rate, data):
        assert rate == 16000
        self.chunks.append(list(data))
        self.ready = True

    def input_finished(self):
        self.finish_calls += 1
        self.eof = True
        self.ready = True

    def is_ready(self, stream):
        return self.ready

    def decode_stream(self, stream):
        self.ready = False

    def get_result(self, stream):
        return SimpleNamespace(keyword="kw2" if self.eof else "kw1",
                               tokens=["x", "iǎo"], timestamps=[0.0, 0.02])

    def reset_stream(self, stream):
        self.resets += 1


class NativeSystemTests(unittest.TestCase):
    def wav(self, rate=16000, width=2, samples=321):
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setframerate(rate)
            writer.setsampwidth(width)
            writer.writeframes(bytes(width * samples))
        return output.getvalue()

    def test_exact_short_tail_and_eof_without_silence(self):
        engine = Engine()
        result = target.decode_samples(engine, list(range(321)))
        self.assertEqual([len(c) for c in engine.chunks], [160, 160, 1])
        self.assertEqual(sum(engine.chunks, []), list(range(321)))
        self.assertEqual(engine.finish_calls, 1)
        self.assertEqual(engine.resets, 4)
        self.assertEqual([e["available_samples"] for e in result["events"]], [160, 320, 321, 321])
        self.assertEqual([e["after_input_finished"] for e in result["events"]], [False, False, False, True])
        self.assertEqual(result["added_padding_samples"], 0)
        self.assertTrue(all(e["native_start_time_seconds"] is None for e in result["events"]))

    def test_budget_is_checked_before_decode(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            target.decode_samples(Engine(), [0.0] * 3, deadline=0)

    def test_not_ready_engine_cannot_hide_an_expired_budget(self):
        engine = Engine()
        engine.is_ready = lambda stream: False
        with self.assertRaisesRegex(ValueError, "budget"):
            target.decode_samples(engine, [0.0] * 321, deadline=0)
        self.assertEqual(engine.chunks, [])

    def test_pcm_contract_rejects_format_and_incomplete_file(self):
        self.assertEqual(target.pcm_wave(self.wav())[1], 321)
        for raw in (self.wav(24000), self.wav(width=1), self.wav(samples=0), self.wav()[:-1], self.wav()+b"x"):
            with self.assertRaises(ValueError):
                target.pcm_wave(raw)

    def test_inventory_binds_all_evidence_not_only_audio(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "audio").mkdir()
            (root / "audio/a.wav").write_bytes(self.wav())
            (root / "source.json").write_text("{}")
            inventory = {p: target.sha256((root / p).read_bytes()) for p in ("audio/a.wav", "source.json")}
            raw = json.dumps(inventory).encode()
            (root / "MANIFEST.json").write_bytes(raw)
            audio, count = target.verified_audio(root, target.sha256(raw))
            self.assertEqual((len(audio), count, audio[0]["samples"]), (1, 2, 321))
            (root / "source.json").write_text('{"changed":1}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                target.verified_audio(root, target.sha256(raw))

    def test_atomic_receipts_never_replace_an_existing_file(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "receipt.json"
            target.write_new_json(path, {"value": 1})
            raw = path.read_bytes()
            with self.assertRaises(FileExistsError):
                target.write_new_json(path, {"value": 2})
            self.assertEqual(path.read_bytes(), raw)

    def test_cached_native_and_claim_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "results").mkdir()
            (root / "claims").mkdir()
            item = {"samples": 321, "pcm_sha256": "pcm", "wav_sha256": "wav"}
            claim = root / "claims/a.json"
            target.write_new_json(claim, {"identity_sha256": "id", "input": item, "attempt": 1})
            value = {"schema": 1, "identity_sha256": "id", "input": item, "status": "completed",
                     "claim_sha256": target.sha256(claim.read_bytes()), "audio_samples": 321,
                     "input_finished_calls": 1, "added_padding_samples": 0, "events": []}
            value["result_sha256"] = target.sha256(target.canonical(value))
            result = root / "results/a.json"
            target.write_new_json(result, value)
            self.assertEqual(target.cached_result(result, "id", item), value)
            changed = dict(value, events=[{"keyword_id": 2}])
            result.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "result bytes drift"):
                target.cached_result(result, "id", item)
            result.write_text(json.dumps(value))
            claim.write_text(json.dumps({"identity_sha256": "another", "input": item, "attempt": 1}))
            with self.assertRaisesRegex(ValueError, "claim identity drift"):
                target.cached_result(result, "id", item)

    def test_complete_clip_receipts_rebuild_summary_without_model_construction(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            view, models, output = (root / x for x in ("view", "models", "output"))
            for p in (view, models, output):
                p.mkdir()
            for p in (output / "claims", output / "results"):
                p.mkdir()
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps({"keywords": [{"id": 1, "tokens": ["n", "ǐ"]}]}))
            item = {"path": "audio/a.wav", "samples": 321, "pcm_sha256": "pcm", "wav_sha256": "wav"}
            source_root = Path(target.__file__).resolve().parents[1]
            identity = {"schema": 1, "profile_sha256": target.sha256(profile_path.read_bytes()),
                        "view_manifest_sha256": "view", "runner_sha256": target.sha256(Path(target.__file__).read_bytes()),
                        "lock_helper_sha256": target.sha256((source_root / "tools/research_job_lock.py").read_bytes()),
                        "audio": [item], "python": sys.version.split()[0]}
            digest = target.sha256(target.canonical(identity))
            target.write_new_json(output / "identity.json", identity)
            claim = output / "claims/clip-000.json"
            target.write_new_json(claim, {"identity_sha256": digest, "input": item, "attempt": 1})
            result = {"schema": 1, "identity_sha256": digest, "input": item, "status": "completed",
                      "claim_sha256": target.sha256(claim.read_bytes()), "audio_samples": 321,
                      "input_finished_calls": 1, "added_padding_samples": 0, "events": [], "inference_seconds": 0.1}
            result["result_sha256"] = target.sha256(target.canonical(result))
            target.write_new_json(output / "results/clip-000.json", result)
            (output / "keywords.txt").write_text("n ǐ @kw1\n")
            args = SimpleNamespace(output=str(output), view=str(view), models=str(models), profile=str(profile_path),
                                   view_manifest_sha256="view", heavy_lock=str(root / "slot.lock"), max_seconds=60)
            with mock.patch.object(target, "verified_audio", return_value=([item], 2)), \
                 mock.patch.object(target, "check_profile"), \
                 mock.patch.object(target, "make_engine", side_effect=AssertionError("model must not load")) as factory:
                rebuilt = target.run(args)
                self.assertEqual(rebuilt["new_model_calls"], 0)
                self.assertEqual(rebuilt["model_load_seconds_this_attempt"], 0)
                self.assertEqual(rebuilt["cached_clips"], 1)
                self.assertTrue((output / "summary.json").is_file())
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
