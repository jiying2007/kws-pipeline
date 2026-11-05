#!/usr/bin/env python3
"""Stdlib-only native D1 boundaries; no model/runtime package is imported."""
from collections import Counter
import copy
import io
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import run_d1_zipformer_system as target
import test_research_d1 as fixture
import test_zipformer_system as native_fixture
from research_d1_recipe import project_rows, cohort_rows


def projected():
    raw = fixture.raw_rows()
    selected = [row for family in range(8) for row in raw[family * 32:family * 32 + 8]]
    return project_rows(selected, fixture.vocabulary())


def job():
    rows = projected()
    return {"schema": target.SCHEMA,
        "completed_preparation": {"queue_root": "/private/actual-training-queue", "job_id": "prepare",
            "receipt_sha256": "1" * 64, "manifest_sha256": "2" * 64},
        "plan_sha256": rows[0]["source_plan_sha256"], "source_ids": sorted(row["source_id"] for row in rows),
        "held_out_source_family_ids": ["synthetic-family-6", "synthetic-family-7"],
        "models": "/private/zipformer-assets", "profile_sha256": target.PROFILE_SHA256,
        "source_files": target.source_records(), "timeout_seconds": 120}


class D1NativeContracts(unittest.TestCase):
    def test_complete_predeclared_roster_sources_profile_and_120_second_cap(self):
        value = job()
        self.assertEqual(target.validate_job(value), value)
        for field, changed in (("source_ids", value["source_ids"][:-1]), ("source_ids", value["source_ids"][::-1]),
                ("held_out_source_family_ids", ["same", "same"]), ("timeout_seconds", 121),
                ("timeout_seconds", float("nan")), ("timeout_seconds", True),
                ("profile_sha256", "3" * 64), ("source_files", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                target.validate_job({**value, field: changed})
        bad = copy.deepcopy(value)
        bad["completed_preparation"]["receipt_sha256"] = "future"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            target.validate_job(bad)

    def test_global_roster_projection_and_role_replay_preserve_all_rows(self):
        rows = projected()
        record = {"observed_rows": 64, "projected_rows_by_split": dict(Counter(r["split"] for r in rows)),
                  "aggregate": {"plan_sha256": job()["plan_sha256"]}}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            def write(values):
                (path / "projected-rows.jsonl").write_bytes(b"".join(target.training.io_contract.canonical(r) for r in values))
            write(rows)
            with patch.object(target.development, "read_pcm", return_value=b"dummy PCM not used in this projection test"):
                result, digest = target.verified_rows(job(), path, record, fixture.vocabulary(), cohort_rows(rows))
                self.assertEqual(len(result), 64)
                self.assertEqual(digest, target.training.sha_file(path / "projected-rows.jsonl"))
                changed = copy.deepcopy(rows)
                changed[-1]["speaker_id"] = changed[0]["speaker_id"]
                write(changed)
                with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_LEAK"):
                    target.verified_rows(job(), path, record, fixture.vocabulary(), cohort_rows(rows))
                write(rows[:-1])
                with self.assertRaisesRegex(ValueError, "GLOBAL_ROSTER"):
                    target.verified_rows(job(), path, record, fixture.vocabulary(), cohort_rows(rows))

    def test_unknown_failure_and_quarantine_cannot_become_negatives(self):
        row = projected()[0]
        result = {"events": [{"keyword_id": 1}]}
        for changes, reason in (({"generation_outcome": "failed-native"}, "failed-native"),
                ({"label_status": "uncertain"}, "uncertain-label"), ({"quarantined": True}, "quarantined-label")):
            # Even a corrupted caller's positive masks cannot override exclusion.
            value = target.observed_row({**row, **changes}, result, "a" * 64)
            self.assertEqual([k["known"] for k in value["keyword_observations"]], [False, False])
            self.assertEqual([k["target"] for k in value["keyword_observations"]], [None, None])
            self.assertEqual([k["excluded_reason"] for k in value["keyword_observations"]], [reason, reason])
            summary = target.summarize([value])
            self.assertEqual((summary["raw_observed_clips"], summary["raw_clips_with_events"]), (1, 1))
            self.assertEqual(summary["keywords"]["1"]["known_machine_weak_clips"], 0)
            self.assertEqual(summary["formal_qualification_denominator"], 0)

    def test_keyword_denominators_are_per_known_mask_and_independent_of_ctc(self):
        row = {**projected()[0], "ctc_mask": False, "keyword_targets": [1, None],
               "keyword_mask": [True, False], "keyword_weight": [.5, 0]}
        observation = target.observed_row(row, {"events": [{"keyword_id": 2}]}, "a" * 64)
        result = target.summarize([observation])
        self.assertEqual(result["keywords"]["1"]["known_machine_weak_clips"], 1)
        self.assertEqual(result["keywords"]["1"]["no_trigger_positive_clips"], 1)
        self.assertEqual(result["keywords"]["2"]["known_machine_weak_clips"], 0)
        self.assertEqual(result["keywords"]["2"]["triggered_negative_clips"], 0)
        self.assertFalse(observation["ctc_eligibility_required"])

    def make_pcm_row(self, folder, row, samples):
        stream = io.BytesIO()
        pcm = struct.pack("<" + "h" * len(samples), *samples)
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        wav = stream.getvalue()
        name = row["source_id"] + ".wav"
        (folder / name).write_bytes(wav)
        return {**row, "audio": name, "audio_path": name, "samples": len(samples),
                "pcm_sha256": target.native.sha256(pcm), "file_sha256": target.native.sha256(wav)}

    def test_raw_events_every_row_exact_pcm_no_padding_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for part in ("claims", "native"):
                (root / part).mkdir()
            base = projected()[:3]
            base[1].update(generation_outcome="failed-native")
            base[2].update(label_status="uncertain")
            rows = [self.make_pcm_row(root, row, list(range(321))) for row in base]
            engine = native_fixture.Engine()
            convert = lambda pcm: [v / 32768 for v in struct.unpack("<" + "h" * (len(pcm) // 2), pcm)]
            values = target.run_calls(rows, root, root, "identity", engine, convert, time.monotonic() + 5)
            self.assertEqual(len(values), 3)
            self.assertEqual(len(list((root / "native").glob("*.json"))), 3)
            self.assertEqual(engine.finish_calls, 3)
            self.assertEqual([len(chunk) for chunk in engine.chunks], [160, 160, 1] * 3)
            self.assertEqual(sum(engine.chunks, []), convert(struct.pack("<" + "h" * 321, *range(321))) * 3)
            for row in rows:
                result = target.native.cached_result(root / "native" / (row["source_id"] + ".json"),
                                                      "identity", target.input_binding(row))
                self.assertEqual(result["added_padding_samples"], 0)
            replayed, seconds = target.readback_native(root, rows, "identity")
            self.assertEqual(replayed, values)
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual(target.summarize(values)["raw_observed_clips"], 3)
            self.assertEqual(target.summarize(values)["keywords"]["1"]["known_machine_weak_clips"], 1)
            with self.assertRaisesRegex(ValueError, "ALREADY_CONSUMED"):
                target.run_calls(rows, root, root, "identity", engine, convert, time.monotonic() + 5)
            self.assertEqual(engine.finish_calls, 3)
            native_path = root / "native" / (rows[0]["source_id"] + ".json")
            changed = target.training.read_json(native_path)
            changed["events"] = []
            native_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "result bytes drift"):
                target.readback_native(root, rows, "identity")

    def test_claim_without_native_is_consumed_and_pcm_change_prevents_claim(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for part in ("claims", "native"):
                (root / part).mkdir()
            row = self.make_pcm_row(root, projected()[0], [1, 2, 3])
            claim = root / "claims" / (row["source_id"] + ".json")
            claim.write_text("{}")
            with self.assertRaisesRegex(ValueError, "ALREADY_CONSUMED"):
                target.run_calls([row], root, root, "identity", native_fixture.Engine(), lambda _: [1, 2, 3], time.monotonic() + 5)
            claim.unlink()
            (root / row["audio"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "WAV_CHANGED"):
                target.run_calls([row], root, root, "identity", native_fixture.Engine(), lambda _: [1, 2, 3], time.monotonic() + 5)
            self.assertFalse(claim.exists())

    def test_real_external_queue_terminal_and_manifest_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            producer = root / "producer.py"
            producer.write_text("""import json,os
from pathlib import Path
out=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])
(out/'MANIFEST.json').write_text(json.dumps({'fixture':'not real training data'}))
""")
            recipe = root / "fixture.json"
            target.training.io_contract.write_json(recipe, {"fixture": "no model or optimizer"})
            request = root / "request.json"
            target.training.io_contract.write_json(request, {"schema": "kws-local-research-queue-v1",
                "queue_id": "external-completion-test", "global_budget_minutes": 1,
                "jobs": [{"job_id": "prepare", "recipe": str(recipe), "recipe_sha256": target.training.sha_file(recipe),
                    "argv": [sys.executable, "-I", "-S", "-B", str(producer)], "cwd": str(root),
                    "depends_on": [], "timeout_minutes": 1, "outputs": ["MANIFEST.json"]}]})
            queue = root / "queue"
            env = {key: value for key, value in os.environ.items() if not key.startswith("KWS_RESEARCH_")}
            cli = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_queue.py")]
            for command in (cli + ["init", "--root", str(queue), "--request", str(request), "--heavy-lock", str(root / "heavy.lock")],
                            cli + ["run", "--root", str(queue)]):
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            receipt = queue / "jobs/prepare/result.json"
            manifest = queue / "jobs/prepare/outputs/MANIFEST.json"
            binding = {"queue_root": str(queue), "job_id": "prepare", "receipt_sha256": target.training.sha_file(receipt),
                       "manifest_sha256": target.training.sha_file(manifest)}
            prepared, pin = target.completed_preparation(binding)
            self.assertEqual(prepared, manifest.parent)
            self.assertEqual(pin["receipt_sha256"], target.training.sha_file(receipt))
            with self.assertRaisesRegex(ValueError, "NOT_COMPLETE_OR_CHANGED"):
                target.completed_preparation({**binding, "receipt_sha256": "0" * 64})
            manifest.write_text("changed")
            with self.assertRaisesRegex(ValueError, "COMPLETED_OUTPUT_CHANGED"):
                target.completed_preparation(binding)

    def test_isolated_entry_and_import_have_no_ml_dependencies(self):
        command = [sys.executable, "-I", "-S", "-B"]
        result = subprocess.run(command + [str(ROOT / target.SOURCE_PATH), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(name in sys.modules for name in target.ML_MODULES))

    def test_real_successor_queue_uses_same_lease_and_exact_deadline_protocol(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            probe = root / "probe.py"
            probe.write_text("""import json,os,sys
from pathlib import Path
sys.path.insert(0, TOOLS)
import run_d1_zipformer_system as t
recipe=Path(os.environ['KWS_RESEARCH_RECIPE'])
output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])
job=t.validate_job(t.training.read_json(recipe))
with t.heavy_model_slot(os.environ['KWS_RESEARCH_HEAVY_LOCK_PATH']) as lease:
 binding=t.current_queue_binding(recipe,output,job,lease)
 if not (binding['inherited_heavy_lock'] and binding['supervised']): raise RuntimeError('not supervised')
 if any(name in sys.modules for name in t.ML_MODULES): raise RuntimeError('ML imported')
 t.native.write_new_json(output/'verified.json',binding)
""".replace("TOOLS", repr(str(ROOT / "tools"))))
            recipe = root / "recipe.json"
            target.training.io_contract.write_json(recipe, job())
            request = root / "request.json"
            target.training.io_contract.write_json(request, {"schema": "kws-local-research-queue-v1",
                "queue_id": "real-native-successor-contract", "global_budget_minutes": 2,
                "jobs": [{"job_id": "native", "recipe": str(recipe), "recipe_sha256": target.training.sha_file(recipe),
                    "argv": [sys.executable, "-I", "-S", "-B", str(probe)], "cwd": str(root),
                    "depends_on": [], "timeout_minutes": 2, "outputs": ["verified.json"]}]})
            queue = root / "queue"
            env = {key: value for key, value in os.environ.items() if not key.startswith("KWS_RESEARCH_")}
            cli = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_queue.py")]
            for command in (cli + ["init", "--root", str(queue), "--request", str(request), "--heavy-lock", str(root / "heavy.lock")],
                            cli + ["run", "--root", str(queue)]):
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            value = target.training.read_json(queue / "jobs/native/outputs/verified.json")
            self.assertEqual(value["allowed_runner_nanoseconds"], 120 * 10**9)
            command = target.training.read_json(queue / "jobs/native/command-started.json")
            self.assertEqual(value["pid"], command["pid"])
            self.assertEqual(value["process_group"], command["process_group"])

    def test_native_guard_precedes_first_ml_import_and_engine_load(self):
        source = inspect.getsource(target.run)
        self.assertLess(source.index("research_no_network.set_offline_environment()"), source.index("research_no_network.install()"))
        self.assertLess(source.index("research_no_network.install()"), source.index("import numpy"))
        self.assertLess(source.index("research_no_network.install()"), source.index("native.make_engine("))
        self.assertNotIn("subprocess", source)
        self.assertNotIn("start_new_session", source)


if __name__ == "__main__":
    unittest.main()
