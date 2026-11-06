"""Acoustic consumer counterexamples; default suite never imports Torch."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools"), str(ROOT / "tests")]
import research_acoustic_consume as consume
import research_acoustic_asr as observer
from research_d1_vocab import make_vocabulary
from test_research_d1_fit_diagnostic import base_recipe
from test_research_acoustic_asr import native_results

q = consume.training.io_contract
TOKENS = (ROOT / "tests/fixtures/d1-fulltext/fsmn-official-tokens.txt").read_bytes()
VOCAB = make_vocabulary(TOKENS)
SAVED = "--saved-tensors" in sys.argv
if SAVED:
    sys.argv.remove("--saved-tensors")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    q.write_json(path, value)


def raw(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    q.write_new(path, value)


def wave_bytes(value, samples=16000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(struct.pack("<" + "h" * samples, *([value, -value] * (samples // 2))))
    return target.getvalue()


def refresh(item, native=None):
    binding = item["binding"]
    binding["variant_row_sha256"] = q.sha(q.canonical(binding["variant_row"]))
    binding["parent_row_sha256"] = q.sha(q.canonical(binding["parent_row"]))
    item["observation"] = observer.observe_variant(binding, native or native_results(binding["parent_row"]["full_text"]))
    return item


def item_fixture(root, index=0, parent_index=0, kind="digital-level", split="train", *, samples=16000):
    ident = "aug-fixture-%03d" % index
    wav = root / (ident + ".wav")
    raw(wav, wave_bytes(index + 1, samples))
    pcm, floating, measured = observer.decode_pcm16(wav.read_bytes())
    role = "train-candidate" if split == "train" else "development-only"
    text = "现在你好小窝把灯打开" if parent_index % 2 == 0 else "现在小窝小窝把灯打开"
    parent = {"full_text": text, "ctc_mask": True, "ctc_weight": .5,
        "target_ids": consume.fulltext_targets(text, VOCAB)["target_ids"],
        "keyword_targets": [int("你好小窝" in text), int("小窝小窝" in text)],
        "keyword_mask": [True, True], "keyword_weight": [.5, .5],
        "generation_category": "full-positive", "source_role": role, "split": split,
        "source_family_id": "family-" + str(parent_index // 2), "speaker_id": "speaker-" + str(parent_index // 2),
        "global_source_key": ["a" * 64, "parent-%02d" % parent_index],
        "file_sha256": q.sha(("parent-wav-%02d" % parent_index).encode()),
        "pcm_sha256": q.sha(("parent-pcm-%02d" % parent_index).encode()),
        "reference_pcm_sha256": q.sha(("reference-pcm-%02d" % (parent_index // 2)).encode()),
        "reference_wav_sha256": q.sha(("reference-wav-%02d" % (parent_index // 2)).encode()),
        "parent_source_sha256": q.sha(("reference-source-%02d" % (parent_index // 2)).encode())}
    noise = kind == "recorded-noise-mix"
    asset = None
    recipe = {"digital_gain": .1}
    if kind in ("measured-rir-convolution", "recorded-noise-mix"):
        asset = {"split": split, "sha256": q.sha((kind + split).encode()), "source_group": kind + "-" + split}
        recipe = {"asset_sha256": asset["sha256"], "asset_split": split, "asset_group": asset["source_group"]}
    row = {"sample_id": ident, "audio": "audio/" + ident + ".wav", "file_sha256": q.sha(wav.read_bytes()),
        "pcm_sha256": q.sha(pcm), "samples": samples, "sample_rate_hz": 16000,
        "parent_key": list(parent["global_source_key"]), "parent_file_sha256": parent["file_sha256"],
        "kind": kind, "recipe": recipe, "source_family_id": parent["source_family_id"],
        "speaker_id": parent["speaker_id"], "source_role": role, "split": split,
        "vocabulary_sha256": consume.digest(VOCAB), "training_admitted": False,
        "new_independent_speaker": False, **observer.augment.inherited_labels(parent, noise)}
    binding = {"variant_row": row, "parent_row": parent, "variant_wav": observer.pin(wav),
        "original_pcm16_sha256": q.sha(pcm), "decoded_float32_sha256": q.sha(floating),
        "measurements": measured, "acoustic_asset": asset}
    return refresh({"binding": binding})


def population(root):
    items = []
    for parent in range(14):
        split = "train" if parent < 10 else "development"
        kinds = [kind for kind in ("onset-offset", "digital-level", "measured-rir-convolution") for _ in range(2)]
        if split == "train":
            kinds += ["recorded-noise-mix"] * 4
        for kind in kinds:
            items.append(item_fixture(root, len(items), parent, kind, split))
    return {"closure": {"variant_manifest": {"sha256": "e" * 64}}, "items": items}


def project(data, prior=()):
    return consume.project_rows(data, VOCAB, prior, TOKENS)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all124_are_retained_with_60_train_24_dev_and_40_masked_noise(self):
        data = population(self.root)
        original = copy.deepcopy(data)
        rows = project(data)
        train, dev = consume.selected_rows(rows)
        self.assertEqual((len(rows), len(train), len(dev)), (124, 60, 24))
        self.assertEqual(Counter(r["ctc_weight"] for r in rows), {.5: 84, 0.0: 40})
        self.assertTrue(all(r["training_admitted"] is False for r in dev))
        self.assertTrue(all(r["generation_category"] == "full-positive" and not r["human_gold"] for r in rows))
        self.assertEqual(data, original)
        recipe = consume.probe_recipe(train, base_recipe())
        self.assertEqual((recipe["optimizer_steps"], len(recipe["exposure_plan"]["draws"])), (25, 100))
        self.assertEqual({r["source_id"] for r in recipe["exposure_plan"]["draws"]}, {r["source_id"] for r in train})
        self.assertTrue(all(r["ctc_weight"] == .5 for r in recipe["exposure_plan"]["draws"]))
        self.assertFalse({r["source_id"] for r in dev} & {r["source_id"] for r in recipe["exposure_plan"]["draws"]})
        self.assertEqual(len({tuple(r["parent_global_source_key"]) for r in rows}), 14)

    def test_noise_consensus_does_not_restore_ctc_or_absent_keyword_labels(self):
        item = item_fixture(self.root, kind="recorded-noise-mix")
        row = consume.project_row(item["binding"], item["observation"], VOCAB, "e" * 64)
        self.assertEqual(row["asr_observation"]["conclusion"], "unanimous-parent-fulltext-match")
        self.assertFalse(row["ctc_mask"]); self.assertEqual(row["ctc_weight"], 0)
        self.assertIsNone(row["target_ids"]); self.assertIsNone(row["full_text"])
        self.assertEqual(row["keyword_mask"], [False, False])
        self.assertEqual(row["inherited_variant_labels"]["keyword_mask"], [True, False])

    def test_same_keyword_changed_context_is_quarantined_without_rewriting_parent(self):
        item = item_fixture(self.root)
        native = native_results(item["binding"]["parent_row"]["full_text"])
        native["whisper"]["text"] = "现在你好小窝把门打开"
        refresh(item, native)
        before = copy.deepcopy(item)
        row = consume.project_row(item["binding"], item["observation"], VOCAB, "e" * 64)
        self.assertFalse(row["ctc_mask"])
        self.assertIn("full-transcript-disagreement", row["ctc_reason"])
        self.assertEqual(item, before)
        self.assertEqual(row["parent_observed_row"]["full_text"], "现在你好小窝把灯打开")

    def test_unanimous_different_text_empty_and_quality_uncertainty_are_audit_only(self):
        item = item_fixture(self.root)
        for native in (native_results("现在你好小屋把灯打开"), native_results("")):
            refresh(item, native)
            row = consume.project_row(item["binding"], item["observation"], VOCAB, "e" * 64)
            self.assertFalse(row["ctc_mask"]); self.assertEqual(row["ctc_weight"], 0)
        for key, value in (("decoder_terminated", False), ("quality_flags", ["repeat_suspected"])):
            native = native_results(item["binding"]["parent_row"]["full_text"])
            native["qwen06"][key] = value
            refresh(item, native)
            self.assertFalse(consume.project_row(item["binding"], item["observation"], VOCAB, "e"*64)["ctc_mask"])

    def test_false_agreement_native_flags_and_omitted_engine_cannot_be_promoted(self):
        item = item_fixture(self.root)
        for mutate in (
            lambda x: x["observation"]["observations"]["qwen06"].update(quality_flags=["repeat_suspected"]),
            lambda x: x["observation"]["observations"]["whisper"].update(normalized_full_text="你好小窝"),
            lambda x: x["observation"]["observations"].pop("whisper")):
            altered = copy.deepcopy(item); mutate(altered)
            with self.assertRaisesRegex(ValueError, "FALSE_FULLTEXT_AGREEMENT|THREE_ASR_REQUIRED"):
                consume.project_row(altered["binding"], altered["observation"], VOCAB, "e"*64)

    def test_raw_label_hash_source_pcm_and_parent_target_tampering_rejected(self):
        item = item_fixture(self.root)
        altered = copy.deepcopy(item); altered["binding"]["variant_row"]["ctc_weight"] = 1
        with self.assertRaisesRegex(ValueError, "OBSERVATION_BINDING"):
            consume.project_row(altered["binding"], altered["observation"], VOCAB, "e"*64)
        altered = copy.deepcopy(item); altered["observation"]["original_pcm16_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "OBSERVATION_SOURCE_CHANGED"):
            consume.project_row(altered["binding"], altered["observation"], VOCAB, "e"*64)
        altered = copy.deepcopy(item)
        altered["binding"]["parent_row"]["target_ids"] = [1, 2, 3]
        altered["binding"]["variant_row"]["target_ids"] = [1, 2, 3]; refresh(altered)
        with self.assertRaisesRegex(ValueError, "PARENT_VOCABULARY_CHANGED"):
            consume.project_row(altered["binding"], altered["observation"], VOCAB, "e"*64)

    def test_insufficient_unpadded_ctc_geometry_stays_zero_weight(self):
        item = item_fixture(self.root, samples=400)
        row = consume.project_row(item["binding"], item["observation"], VOCAB, "e"*64)
        self.assertEqual(row["expected_c1_frames"], 1)
        self.assertFalse(row["ctc_mask"]); self.assertIn("geometry_infeasible", row["ctc_reason"])
        self.assertIsNone(row["target_ids"])

    def test_parent_family_reference_pcm_and_asset_cross_split_rejected_even_for_zero_rows(self):
        data = population(self.root)
        train = data["items"][0]["binding"]["parent_row"]
        for alias, previous in (
            ("family", {"source_family_id": train["source_family_id"]}),
            ("speaker", {"speaker_id": train["speaker_id"]}),
            ("reference PCM", {"pcm_sha256": train["reference_pcm_sha256"]}),
            ("original file", {"wav_sha256": train["file_sha256"]})):
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
                project(data, [{"role": "development-only", **previous}])
        noise = next(i for i in data["items"] if i["binding"]["variant_row"]["kind"] == "recorded-noise-mix")
        changed = copy.deepcopy(data)
        altered = next(i for i in changed["items"] if i["binding"]["variant_row"]["sample_id"] ==
                       noise["binding"]["variant_row"]["sample_id"])
        altered["binding"]["acoustic_asset"]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "ASSET_SPLIT"):
            project(changed)

    def test_mutated_parent_role_duplicate_pcm_and_incomplete_population_fail(self):
        data = population(self.root)
        changed = copy.deepcopy(data)
        changed["items"][0]["binding"]["parent_row"]["split"] = "development"; refresh(changed["items"][0])
        with self.assertRaisesRegex(ValueError, "PARENT_SPLIT_JOIN"):
            project(changed)
        changed = copy.deepcopy(data)
        changed["items"][1]["binding"]["variant_row"]["pcm_sha256"] = changed["items"][0]["binding"]["variant_row"]["pcm_sha256"]
        refresh(changed["items"][1])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_VARIANT"):
            project(changed)
        with self.assertRaisesRegex(ValueError, "FULL_ROSTER"):
            project({"closure": data["closure"], "items": data["items"][:-1]})

    def test_development_and_unknown_rows_cannot_enter_smoke(self):
        rows = project(population(self.root))
        train, dev = consume.selected_rows(rows)
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_OR_UNKNOWN_DRAW"):
            consume.probe_recipe(train + dev[:1], base_recipe())
        unknown = next(r for r in rows if not r["ctc_mask"])
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_OR_UNKNOWN_DRAW"):
            consume.probe_recipe(train + [unknown], base_recipe())
        altered_vocab = copy.deepcopy(VOCAB); altered_vocab["token_to_id"]["你"] += 1
        with self.assertRaisesRegex(ValueError, "DERIVED_VOCABULARY_CHANGED"):
            consume.project_rows(population(self.root / "alternate"), altered_vocab, [], TOKENS)


TOY_OBSERVER = r'''
# Explicit contract-only fixture: no native execution or real provider receipt.
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
JOBS=16
SCHEMA="fixture-observer-v1"
def read(path):
    return json.loads(Path(path).read_text())
def pin(path):
    path=Path(path);body=path.read_bytes()
    return {"path":str(path),"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}
def job_id(i):return "observe-%03d"%(i+1)
def load_stage(path):
    stage=read(path)
    return stage,read(Path(path).parent/"plan.json"),{},None
def inspect(root,reconcile=False):
    assert not reconcile
    return read(Path(root)/"report.json")
def verify_variants(root,sha):
    plan=read(Path(root)/"plan.json")
    return [{"sample_id":i} for i in plan["ordered_sample_ids"]],plan["bindings"]
def verify_prerequisites(stage):return pin(Path(stage["root"])/"prerequisites.json")
def aggregate(stage,plan):
    root=Path(stage["root"])/"validation"
    return read(root/"summary.json"),pin(root/"MANIFEST.json")
def phase_metadata(folder):
    return {n:pin(Path(folder)/n) for n in ("capsule.json","ack.json","provider-save.json","phase-marker.json")}
queue=SimpleNamespace(inspect=inspect,read_bytes=lambda p:Path(p).read_bytes(),loads=json.loads)
'''


class ClosedObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source_file = source / "tools/research_acoustic_asr.py"
        raw(source_file, TOY_OBSERVER.encode())
        self.stage_root = self.root / "stage"
        self.stage_path = self.stage_root / "stage.json"
        ids = ["observe-%03d" % i for i in range(1, 17)]
        requests = []
        for name in ids:
            native = self.stage_root / "native" / name
            save(native / "audit/final-summary.json", {"fixture": True})
            path = self.stage_root / "requests" / (name + ".json")
            save(path, {"output": str(native)})
            requests.append({"path": str(path)})
            save(self.stage_root / "chunks" / name / "validation.json", {"fixture": True})
        raw(self.stage_root / "driver.lock", b"")
        self.report = {"jobs": {name: {"status": "completed"} for name in ids}, "status": "completed",
            "actual_total_is_complete": True, "unresolved_jobs": [], "budget_overrun_nanoseconds": 0}
        save(self.stage_root / "queue/report.json", self.report)
        stage = {"root": str(self.stage_root), "asr_requests": requests, "heavy_lock": str(self.root / "heavy.lock"),
                 "source_hashes": {"tools/research_acoustic_asr.py": q.file_record(source_file)}}
        save(self.stage_path, stage)
        sample_ids = ["aug-%03d" % i for i in range(124)]
        save(self.stage_root / "variant-manifest.json", {"fixture": True})
        plan = {"ordered_sample_ids": sample_ids, "bindings": {i: {"sample_id": i} for i in sample_ids},
                "variant_manifest": observer.pin(self.stage_root / "variant-manifest.json")}
        save(self.stage_root / "plan.json", plan)
        save(self.stage_root / "prerequisites.json", {"fixture": True})
        aggregate = self.stage_root / "validation"
        save(aggregate / "MANIFEST.json", {"fixture": True})
        save(aggregate / "summary.json", {"observed_rows": 124, "native_asr_calls": 372,
            "noise_ctc_masked_rows_unchanged": 40, "frozen_variant_rows_changed": 0, "human_gold": False})
        raw(aggregate / "rows.jsonl", b"".join(q.canonical({"sample_id": i}) for i in sample_ids))
        final = self.stage_root / "checkpoints/final/campaign-complete"
        for name in ("ack.json", "provider-save.json", "phase-marker.json"):
            save(final / name, {"fixture": True})
        save(final / "capsule.json", {"context": {"schema": "fixture-observer-v1", "phase": "campaign-complete",
            "stage": observer.pin(self.stage_path), "aggregate": observer.pin(aggregate / "MANIFEST.json"),
            "maximum_asr_calls": 372, "source_rows_changed": 0, "new_tts_calls": 0,
            "new_optimizer_steps": 0, "launch_intent": False}})
        self.request = {"schema": consume.PREP, "stage": str(self.stage_path),
            "stage_sha256": q.file_record(self.stage_path)["sha256"], "observer": str(source_file),
            "observer_sha256": q.file_record(source_file)["sha256"], "anchor": str(self.root / "anchor"),
            "anchor_manifest_sha256": "a"*64}

    def inventory(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_fresh_process_closed_replay_is_read_only_and_zero_model(self):
        before = self.inventory()
        result = consume.closed_observer(self.request)
        self.assertEqual(len(result["items"]), 124)
        self.assertEqual(result["closure"]["new_model_calls"], 0)
        self.assertEqual(self.inventory(), before)
        self.assertNotIn("torch", sys.modules)

    def test_failed_or_unknown_is_not_a_completed124_campaign(self):
        for state in ("failed", "unknown"):
            report = copy.deepcopy(self.report); report["jobs"]["observe-001"]["status"] = state
            (self.stage_root / "queue/report.json").write_bytes(q.canonical(report))
            with self.assertRaisesRegex(ValueError, "COMPLETE_QUEUE_REQUIRED"):
                consume.closed_observer(self.request)

    def test_missing_native_audit_aggregate_or_final_save_does_not_create_them(self):
        paths = [self.stage_root / "native/observe-001/audit/final-summary.json",
                 self.stage_root / "validation/rows.jsonl",
                 self.stage_root / "checkpoints/final/campaign-complete/ack.json"]
        for path in paths:
            body = path.read_bytes(); path.unlink(); before = self.inventory()
            with self.assertRaisesRegex(ValueError, "OBSERVER_NOT_CLOSED"):
                consume.closed_observer(self.request)
            self.assertEqual(self.inventory(), before)
            path.write_bytes(body)

    def test_frozen_stage_hash_change_rejected_before_child_and_busy_driver_not_stolen(self):
        altered = dict(self.request, stage_sha256="b"*64)
        with mock.patch.object(consume.subprocess, "run", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(ValueError, "INPUT_CHANGED"):
                consume.closed_observer(altered)
        import fcntl
        with (self.stage_root / "driver.lock").open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "BlockingIOError"):
                consume.closed_observer(self.request)


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.observed = population(self.root / "original-variants")
        self.observed["closure"]["heavy_lock"] = str(self.root / "shared.lock")
        self.anchor_root = self.root / "original-anchor"
        raw(self.anchor_root / "official/tokens.txt", TOKENS)
        raw(self.anchor_root / "initial/c1-ctc2601.pt", b"explicit non-model checkpoint fixture")
        self.original = {"vocabulary": VOCAB, "prior_sources": [], "base_recipe": base_recipe(),
            "initial": {"file_sha256": q.file_record(self.anchor_root / "initial/c1-ctc2601.pt")["sha256"]}}
        self.request = {"schema": consume.PREP, "stage": str(self.root / "original-stage/stage.json"),
            "stage_sha256": "a"*64, "observer": str(self.root / "original-source/tools/research_acoustic_asr.py"),
            "observer_sha256": "b"*64, "anchor": str(self.anchor_root), "anchor_manifest_sha256": "c"*64}
        self.target = self.root / "prepared"
        self.anchor = self.enterContext(mock.patch.object(consume.historical, "anchor", return_value=self.original))
        self.implementation = self.enterContext(mock.patch.object(consume.core, "verify_original_implementation"))
        self.replay = self.enterContext(mock.patch.object(consume, "closed_observer", return_value=self.observed))
        self.enterContext(mock.patch.object(consume.training, "torch_runtime", side_effect=AssertionError("no model runtime")))

    def inventory(self, root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def prepare(self):
        return consume.prepare(self.request, self.target)

    def reseal(self):
        (self.target / "MANIFEST.json").unlink()
        consume.core.manifest(self.target)
        return q.file_record(self.target / "MANIFEST.json")["sha256"]

    def test_real_package_replay_preserves_originals_and_fixed_declaration(self):
        before = self.inventory(self.root / "original-variants"), self.inventory(self.anchor_root)
        result = self.prepare()
        record, _, rows, train, dev, _ = consume.verify_prepared(self.target, result["manifest_sha256"])
        self.assertEqual((len(rows), len(train), len(dev)), (124, 60, 24))
        self.assertEqual(record["source_rows_modified"], 0)
        self.assertEqual(before, (self.inventory(self.root / "original-variants"), self.inventory(self.anchor_root)))
        self.implementation.assert_called_with(self.original, str(self.anchor_root))
        sealed = self.inventory(self.target)
        declaration = consume.declare(self.target, result["manifest_sha256"], self.root / "recipe.json")
        self.assertEqual((declaration["optimizer_steps"], declaration["sample_exposures"]), (25, 100))
        self.assertEqual(self.inventory(self.target), sealed)
        self.assertNotIn("torch", sys.modules)

    def test_rehashed_projected_labels_and_audio_do_not_become_authoritative(self):
        self.prepare()
        row_path = self.target / "projected-rows.jsonl"
        body = row_path.read_bytes()
        rows = consume.historical.read_lines(row_path)
        rows[0]["ctc_weight"] = 1
        row_path.write_bytes(b"".join(q.canonical(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "PROJECTED_LABELS_CHANGED"):
            consume.verify_prepared(self.target, self.reseal())
        row_path.write_bytes(body)
        wav = self.target / rows[0]["audio"]
        wav.write_bytes(wave_bytes(999))
        with self.assertRaisesRegex(ValueError, "RETAINED_AUDIO_CHANGED"):
            consume.verify_prepared(self.target, self.reseal())

    def test_changed_observer_closure_initialization_and_execution_source_fail(self):
        self.prepare()
        original = copy.deepcopy(self.observed)
        self.observed["closure"]["variant_manifest"]["sha256"] = "d"*64
        with self.assertRaisesRegex(ValueError, "REPLAY_CHANGED"):
            consume.verify_prepared(self.target, self.reseal())
        self.observed.clear(); self.observed.update(original)
        initial = self.target / "initial.pt"; body = initial.read_bytes()
        initial.write_bytes(b"another invented checkpoint")
        with self.assertRaisesRegex(ValueError, "ANCHOR_CHANGED"):
            consume.verify_prepared(self.target, self.reseal())
        initial.write_bytes(body)
        saved = consume.source_records(); saved["training/research_acoustic_consume.py"] = "0"*64
        with mock.patch.object(consume, "source_records", return_value=saved):
            with self.assertRaisesRegex(ValueError, "PREPARED_SOURCE_CHANGED"):
                consume.verify_prepared(self.target, self.reseal())
        retained_source = self.target / "execution-source/training/research_acoustic_consume.py"
        retained_source.write_bytes(retained_source.read_bytes() + b"\n# changed retained source\n")
        with self.assertRaisesRegex(ValueError, "PREPARED_SOURCE_CHANGED"):
            consume.verify_prepared(self.target, self.reseal())

    def test_frozen_inputs_and_prepared_manifest_cannot_receive_new_outputs(self):
        for path in (self.anchor_root / "consumer", Path(self.request["stage"]).parent / "consumer",
                     Path(self.request["observer"]).parents[1] / "consumer"):
            with self.assertRaisesRegex(ValueError, "OUTPUT_INSIDE_FROZEN_INPUT"):
                consume.prepare(self.request, path)
            self.assertFalse(path.exists())
        result = self.prepare(); before = self.inventory(self.target)
        with self.assertRaisesRegex(ValueError, "DECLARATION_OUTSIDE_PREPARED"):
            consume.declare(self.target, result["manifest_sha256"], self.target / "recipe.json")
        self.assertEqual(self.inventory(self.target), before)

    def test_original_algorithm_change_blocks_before_output_creation(self):
        self.implementation.side_effect = ValueError("V3_INGESTION_ORIGINAL_ALGORITHM_SOURCE_CHANGED")
        with self.assertRaisesRegex(ValueError, "ORIGINAL_ALGORITHM_SOURCE_CHANGED"):
            self.prepare()
        self.assertFalse(self.target.exists())
        self.replay.assert_not_called()


@unittest.skipUnless(SAVED, "explicit tensor-only audit; no model execution")
class SavedNumericsTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, split in enumerate(("train", "development")):
            item = item_fixture(self.root, index, index * 2, split=split)
            row = consume.project_row(item["binding"], item["observation"], VOCAB, "e"*64)
            row["audio"] = row["audio_path"] = Path(row["audio_path"]).name
            self.rows.append(row)
        self.features, measured = consume.core.feature_rows(self.root, self.rows, torch)
        save(self.root / "feature-observations.json", measured)
        self.summary = {}
        for phase, selected in (("before", [0]), ("after", [0]), ("development", [1])):
            observed = []
            for index in selected:
                row, feature = self.rows[index], self.features[index]
                # Invented logits, never a model output or training result.
                logits = torch.zeros(len(feature), 1, 2601); logits[:, :, 0] = 1
                name = "numeric/" + phase + "/" + row["source_id"] + ".pt"
                observed.append({"source_id": row["source_id"], "pcm_sha256": row["pcm_sha256"],
                    "label_binding_sha256": consume.row_identity(row), "target_ids": row["target_ids"],
                    "ctc_mask": row["ctc_mask"], "ctc_weight": row["ctc_weight"],
                    **consume.diagnostic.score(logits, row),
                    "numeric_artifact": {"path": name, **consume.diagnostic.evaluation.save_numeric(
                        self.root / name, row["source_id"], feature, logits)}})
            save(self.root / (phase + ".json"), observed)
            self.summary[phase] = consume.core.observed_summary(observed)
        self.enterContext(mock.patch.object(consume.training, "build_fresh", side_effect=AssertionError("no construction")))
        self.enterContext(mock.patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("no forward")))
        self.enterContext(mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer")))

    def audit(self):
        return consume.audit_numerics(self.root, self.root, self.rows[:1], self.rows[1:], self.summary, self.torch)

    def test_actual_saved_tensor_scores_and_pcm_features_need_no_model(self):
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.summary["before"]["metrics"]["empty_outputs"], 1)

    def test_modified_logits_hash_and_rehashed_feature_or_score_are_rejected(self):
        phase = self.root / "before.json"
        observed = consume.training.read_json(phase); body = phase.read_bytes()
        numeric = self.root / observed[0]["numeric_artifact"]["path"]
        tensor_body = numeric.read_bytes()
        numeric.write_bytes(tensor_body + b"tampered")
        with self.assertRaisesRegex(ValueError, "NUMERIC_HASH"): self.audit()
        numeric.write_bytes(tensor_body)
        observed[0]["ctc_nll"] += 1
        phase.write_bytes(q.canonical(observed))
        with self.assertRaisesRegex(ValueError, "SAVED_NUMERIC_CHANGED"): self.audit()
        phase.write_bytes(body)
        observed = consume.training.read_json(phase); observed[0]["ctc_weight"] = 1
        phase.write_bytes(q.canonical(observed))
        with self.assertRaisesRegex(ValueError, "SAVED_NUMERIC_CHANGED"): self.audit()
        phase.write_bytes(body)
        measured = consume.training.read_json(self.root / "feature-observations.json")
        measured[0]["feature_sha256"] = "f"*64
        (self.root / "feature-observations.json").write_bytes(q.canonical(measured))
        with self.assertRaisesRegex(ValueError, "FEATURE_PCM_REPLAY_CHANGED"): self.audit()


class WorkerBoundaryTests(unittest.TestCase):
    def test_direct_worker_cannot_construct_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {"schema": consume.JOB, "prepared": str(root / "prepared"), "prepared_manifest_sha256": "a"*64,
                "candidate": consume.CANDIDATE, "optimizer_steps": 25, "timeout_seconds": 120,
                "train_binding_sha256": "b"*64, "development_binding_sha256": "c"*64,
                "source_hashes": consume.source_records(), "positive_only": True, "quality_acceptance_claim": False}
            path = root / "recipe.json"; q.write_json(path, job)
            with mock.patch.object(consume.training, "torch_runtime", side_effect=AssertionError("no model runtime")):
                with self.assertRaisesRegex(ValueError, "HASH_BOUND_DURABLE_QUEUE"):
                    consume.run(path, root / "output", root / "heavy.lock")
            self.assertFalse((root / "output").exists())

    def test_queue_request_uses_existing_worker_10min_budget_and_exact_shared_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {"schema": consume.JOB, "prepared": str(root / "prepared"), "prepared_manifest_sha256": "a"*64,
                "candidate": consume.CANDIDATE, "optimizer_steps": 25, "timeout_seconds": 120,
                "train_binding_sha256": "b"*64, "development_binding_sha256": "c"*64,
                "source_hashes": consume.source_records(), "positive_only": True, "quality_acceptance_claim": False}
            path = root / "recipe.json"; q.write_json(path, job)
            record = {"observer_closure": {"heavy_lock": str(root / "shared.lock")}}
            with mock.patch.object(consume, "verify_prepared", return_value=(record, None, None, None, None, None)):
                request = consume.queue_request(path, root / "shared.lock")
                self.assertEqual(request["global_budget_minutes"], 10)
                self.assertEqual(len(request["jobs"]), 1)
                self.assertEqual(request["jobs"][0]["job_id"], consume.JOB_ID)
                self.assertEqual(request["jobs"][0]["timeout_minutes"], 10)
                self.assertIn("run", request["jobs"][0]["argv"])
                with self.assertRaisesRegex(ValueError, "CANONICAL_HEAVY_LOCK"):
                    consume.queue_request(path, root / "other.lock")


if __name__ == "__main__":
    unittest.main()
