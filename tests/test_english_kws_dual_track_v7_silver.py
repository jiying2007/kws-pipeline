"""Pure-byte fixtures for V7 candidate-relative admission; no model calls."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "automation"))
import english_kws_dual_track_v7_silver as v7
import english_kws_dual_track_v7_admit as v7cli

PROFILE = json.loads((ROOT / "configs/research/english-pronunciation-profile-20261115.json").read_text())


def phone_tokens(row):
    if row["cell_id"].startswith("vosen"):
        return ["h", "eɪ", "v", "oʊ", "s", "ə", "n"]
    if row["cell_id"].startswith("mero"):
        return ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]
    if row["cell_id"] == "bosen-s":
        return ["h", "eɪ", "b", "oʊ", "s", "ə", "n"]
    if row["cell_id"] == "bosen-z":
        return ["h", "eɪ", "b", "oʊ", "z", "ə", "n"]
    if row["cell_id"] == "memo":
        return ["h", "eɪ", "m", "ɛ", "m", "oʊ"]
    if row["cell_id"] == "nero":
        return ["h", "eɪ", "n", "ɛ", "ɹ", "oʊ"]
    return ["h", "ə", "l", "oʊ"]


class V7AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.groups = []
        for index in range(12):
            role = "train" if index < 6 else "development" if index < 8 else "calibration" if index < 10 else "locked-test"
            cls.groups.append({
                "source_group_id": "piper-libritts-index-" + str(496 + index),
                "source_speaker_id": "p" + str(1000 + index),
                "source_speaker_index": 496 + index,
                "voice_family": "piper-libritts-p" + str(1000 + index),
                "role": role,
            })
        cls.plan = {
            "execution_stage": "G2_SOURCE_GENERATION_AND_BLIND_OBSERVATION_ONLY",
            "source_count": 144,
            "candidate_ids": ["hey-vosen-v7", "hey-mero-v7"],
            "observer_providers": ["qwen", "whisper", "parakeet", "phoneme"],
            "source_role_counts": {"train": 6, "development": 2, "calibration": 2, "locked-test": 2},
            "positive_min_samples": 6400,
            "positive_max_samples": 28672,
            "source_plan_sha256": "a" * 64,
            "recipe_sha256": "b" * 64,
            "training_allowed": False,
            "promotion_allowed": False,
            "new_kws_calls": 0,
            "optimizer_updates": 0,
        }
        cls.corpus, cls.evidence = cls.make_packet()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def make_packet(cls):
        rows = []
        evidence_rows = []
        for group in cls.groups:
            for cell_id, (text, requested_class, kind, raw_ipa, length_scale, labels) in v7.EXPECTED_CELLS.items():
                index = len(rows)
                pcm = struct.pack("<7200h", *([1000 + index] * 7200))
                pcm_path = "source/audio/item-" + str(index) + "/audio-16k.pcm"
                wav_path = "source/audio/item-" + str(index) + "/audio-16k.wav"
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as stream:
                    stream.setparams((1, 2, 16000, 7200, "NONE", "not compressed"))
                    stream.writeframes(pcm)
                wav = wav_buffer.getvalue()
                folder = cls.root / ("source/audio/item-" + str(index))
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "audio-16k.pcm").write_bytes(pcm)
                (folder / "audio-16k.wav").write_bytes(wav)
                row = {
                    **group,
                    "id": "item-" + str(index),
                    "request_sha256": "d" * 64,
                    "cell_id": cell_id,
                    "requested_text": text,
                    "requested_class": requested_class,
                    "pronunciation_hypothesis": ("vosen-us-s-v7" if requested_class == 1 else
                                                 "mero-us-eh-v7" if requested_class == 2 else
                                                 "contrast-negative-v7" if kind == "contrast-negative" else
                                                 "ordinary-negative-v7"),
                    "kind": kind,
                    "length_scale": length_scale,
                    "raw_phonemes": raw_ipa,
                    "candidate_labels": labels,
                    "generation_status": "success",
                    "pcm_path": pcm_path,
                    "wav_path": wav_path,
                    "pcm_sha256": v7.sha(pcm),
                    "wav_sha256": v7.sha(wav),
                    "pcm_samples": 7200,
                    "sample_rate": 16000,
                    "channels": 1,
                }
                rows.append(row)
                packet = {"id": row["id"], "observations": [], "phone_observation": None}
                for provider in ("qwen", "whisper", "parakeet", "phoneme"):
                    capture = {
                        **v7.observer_identity(provider),
                        "schema": v7.CAPTURE_SCHEMA,
                        "id": row["id"],
                        "source_pcm_sha256": row["pcm_sha256"],
                        "source_wav_sha256": row["wav_sha256"],
                        "source_samples": row["pcm_samples"],
                        "status": "success",
                        "target_conditioned": False,
                        "input_coverage_complete": True,
                        "decoder_output_complete": True,
                    }
                    if provider == "phoneme":
                        capture["phone_tokens"] = phone_tokens(row)
                    else:
                        capture["raw_text"] = row["requested_text"]
                    capture_path = "evidence/" + provider + "/" + row["id"] + "/CAPTURE.json"
                    capture_file = cls.root / capture_path
                    capture_file.parent.mkdir(parents=True, exist_ok=True)
                    raw = v7.canonical(capture)
                    capture_file.write_bytes(raw)
                    observation = capture | {
                        "capture_path": capture_path,
                        "capture_sha256": v7.sha(raw),
                    }
                    if provider == "phoneme":
                        packet["phone_observation"] = observation
                    else:
                        packet["observations"].append(observation)
                evidence_rows.append(packet)
        corpus = {
            "schema": v7.CORPUS_SCHEMA,
            "recipe_sha256": "b" * 64,
            "preflight_sha256": "c" * 64,
            "declared_count": 144,
            "valid_count": 144,
            "source_groups": cls.groups,
            "rows": rows,
            "human_gold": False,
        }
        evidence = {
            "schema": v7.EVIDENCE_SCHEMA,
            "rows": evidence_rows,
            "human_gold": False,
            "new_model_calls_during_join": 0,
            "optimizer_updates": 0,
        }
        return corpus, evidence

    def setUp(self):
        self.corpus = copy.deepcopy(self.corpus)
        self.evidence = copy.deepcopy(self.evidence)
        self.plan = copy.deepcopy(self.plan)

    def test_whisper_identity_matches_locked_g2_observer_capture(self):
        self.assertEqual(v7.observer_identity("whisper"), {
            "recognizer_id": "whisper",
            "model_family": "whisper",
            "model_sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
            "decode_config_sha256": "439b51d2e7e5f7ce4ade748a87c3d3e6d5b6b66d2acaec893bfc3bf673af04a7",
        })

    def test_exact_candidate_relative_manifests_and_split(self):
        result = v7.build_admission(self.corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        self.assertTrue(result["summary"]["data_admitted"])
        self.assertFalse(result["summary"]["training_authorized"])
        for candidate in v7.CANDIDATES:
            manifest = result["candidate_manifests"][candidate]
            self.assertTrue(manifest["training_snapshot_allowed"])
            self.assertEqual(manifest["balanced_groups_by_role"], self.plan["source_role_counts"])
            self.assertEqual(len(manifest["rows"]), 144)
            self.assertEqual(manifest["ctc_token_ids"], [1, 2, 3])
            self.assertFalse(manifest["optimizer_allowed"])
            self.assertTrue(manifest["locked_test_sealed_from_training_and_mining"])
            self.assertTrue(all(not row["optimizer_allowed"] for row in manifest["rows"]))

    def test_one_failed_source_is_unknown_without_retry_and_other_groups_remain_usable(self):
        self.corpus["rows"][0].update(generation_status="FAILED_NO_RETRY",
            pcm_path=None, wav_path=None, pcm_sha256=None, wav_sha256=None, pcm_samples=None)
        self.corpus["valid_count"] -= 1
        self.evidence["rows"][0]["observations"] = []
        self.evidence["rows"][0]["phone_observation"] = None
        result = v7.build_admission(self.corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        row = next(item for item in result["candidate_manifests"]["VOSEN"]["rows"]
                   if item["id"] == "item-0")
        self.assertEqual(row["tier"], "unknown")
        self.assertEqual(row["reason"], "SOURCE_NOT_VALID_NO_RETRY")
        self.assertTrue(result["candidate_manifests"]["VOSEN"]["training_snapshot_allowed"])

    def test_missing_observer_quarantines_only_that_row(self):
        self.evidence["rows"][0]["observations"].pop()
        result = v7.build_admission(self.corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        row = next(item for item in result["candidate_manifests"]["VOSEN"]["rows"]
                   if item["id"] == "item-0")
        self.assertEqual(row["tier"], "unknown")
        self.assertEqual(row["reason"], "MISSING_OR_INCOMPLETE_ASR")
        self.assertTrue(result["candidate_manifests"]["VOSEN"]["training_snapshot_allowed"])

    def test_other_brand_phrase_is_candidate_relative_cross_trigger_negative(self):
        result = v7.build_admission(self.corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        rows_vosen = {row["cell_id"]: row for row in result["candidate_manifests"]["VOSEN"]["rows"]}
        rows_mero = {row["cell_id"]: row for row in result["candidate_manifests"]["MERO"]["rows"]}
        self.assertEqual(rows_vosen["vosen-slow"]["research_class"], 1)
        self.assertEqual(rows_vosen["mero-slow"]["research_class"], 0)
        self.assertEqual(rows_mero["mero-slow"]["research_class"], 1)
        self.assertEqual(rows_mero["vosen-slow"]["research_class"], 0)
        self.assertEqual(rows_vosen["mero-slow"]["reason"], "OTHER_BRAND_CROSS_TRIGGER_NEGATIVE")

    def test_locked_calibration_and_development_never_enter_optimizer(self):
        result = v7.build_admission(self.corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        for candidate in v7.CANDIDATES:
            for row in result["candidate_manifests"][candidate]["rows"]:
                if row["role"] == "train":
                    self.assertTrue(row["training_eligible"])
                    self.assertFalse(row["optimizer_allowed"])
                else:
                    self.assertFalse(row["training_eligible"])
                    self.assertFalse(row["optimizer_allowed"])
                    self.assertIsNone(row["training_class"])
                    self.assertEqual(row["training_weight"], 0.0)

    def test_vosen_voiced_z_and_legacy_mero_are_unknown_not_negative(self):
        row = self.corpus["rows"][0]
        self.assertEqual(v7.decide_candidate(
            row, "VOSEN", ["Hey Vozen.", "Hey Vozen.", "Hey Vozen."],
            ["h", "eɪ", "v", "oʊ", "z", "ə", "n"])[0], "unknown")
        row = self.corpus["rows"][3]
        self.assertEqual(v7.decide_candidate(
            row, "MERO", ["Hey Meero.", "Hey Meero.", "Hey Meero."],
            ["h", "eɪ", "m", "iː", "r", "oʊ"])[0], "unknown")

    def test_two_asr_votes_are_weak_only_with_matching_independent_phone(self):
        row = self.corpus["rows"][0]
        tier, label, reason, observed = v7.decide_candidate(
            row, "VOSEN", ["Hey Vosen.", "Hey Vosen.", "something else"],
            ["h", "eɪ", "v", "oʊ", "s", "ə", "n"])
        self.assertEqual((tier, label), ("weak", 1))
        self.assertEqual(reason, "JOINT_ASR_AND_PHONE_TARGET_SUPPORT")

    def test_conflicting_other_keyword_vetoes_two_votes(self):
        row = self.corpus["rows"][0]
        tier, label, _, _ = v7.decide_candidate(
            row, "VOSEN", ["Hey Vosen.", "Hey Vosen.", "Hey Mero."],
            ["h", "eɪ", "v", "oʊ", "s", "ə", "n"])
        self.assertEqual((tier, label), ("unknown", None))

    def test_phone_candidate_is_not_confused_by_voicing(self):
        self.assertEqual(v7.phone_candidate(["h", "eɪ", "v", "oʊ", "s", "ə", "n"]), "VOSEN")
        self.assertIsNone(v7.phone_candidate(["h", "eɪ", "v", "oʊ", "z", "ə", "n"]))

    def test_invalid_positive_geometry_is_rejected(self):
        row = copy.deepcopy(self.corpus["rows"][0])
        pcm = struct.pack("<400h", *([1000] * 400))
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as stream:
            stream.setparams((1, 2, 16000, 400, "NONE", "not compressed"))
            stream.writeframes(pcm)
        wav = wav_buffer.getvalue()
        folder = self.root / "source/audio/short-positive"
        folder.mkdir(parents=True, exist_ok=True)
        pcm_path = folder / "audio-16k.pcm"
        wav_path = folder / "audio-16k.wav"
        pcm_path.write_bytes(pcm)
        wav_path.write_bytes(wav)
        row.update(pcm_path="source/audio/short-positive/audio-16k.pcm",
                   wav_path="source/audio/short-positive/audio-16k.wav",
                   pcm_samples=400, pcm_sha256=v7.sha(pcm), wav_sha256=v7.sha(wav))
        with self.assertRaisesRegex(ValueError, "MODEL_INPUT_GEOMETRY"):
            v7.verify_pcm(row, self.root)

    def test_admission_cli_readiness_builder_emits_candidate_hashes_and_tier_counts(self):
        corpus = copy.deepcopy(self.corpus)
        corpus["run"] = {
            "repository": "jiying2007/kws-data",
            "run_id": "46000882374",
            "sha": "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2",
            "event": "push",
            "ref": "refs/heads/main",
            "attempt": "1",
        }
        joined = v7.build_admission(corpus, self.evidence, self.root, profile=PROFILE, plan=self.plan)
        gate = v7cli.build_readiness(joined, {
            "files_manifest_sha256": "a" * 64,
            "file_count": 723,
            "verified_bytes": 123456,
        })
        self.assertTrue(gate["data_admitted"])
        self.assertFalse(gate["training_authorized"])
        self.assertEqual(gate["source_run"]["run_id"], "46000882374")
        for candidate in v7.CANDIDATES:
            item = gate["candidate_manifests"][candidate]
            self.assertTrue(item["training_snapshot_allowed"])
            self.assertEqual(item["balanced_groups_by_role"], self.plan["source_role_counts"])
            self.assertEqual(sum(item["tier_counts"].values()), 144)
            self.assertEqual(len(item["sha256"]), 64)

    def test_packet_inventory_authenticates_every_file_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            packet = Path(temp)
            files = []
            for name in ("CORPUS_MANIFEST.json", "MACHINE_EVIDENCE.json", "EXECUTION_PLAN.json",
                         "source/audio/a.wav"):
                target = packet / name
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = ("fixture:" + name).encode()
                target.write_bytes(raw)
                files.append({"path": name, "bytes": len(raw), "sha256": v7.sha(raw)})
            (packet / "FILES.json").write_bytes(v7.canonical({
                "schema": "english-kws-dual-track-v7-files-v1", "files": files
            }) + b"\n")
            result = v7cli.verify_packet_inventory(packet)
            self.assertEqual(result["file_count"], 4)
            (packet / "source/audio/a.wav").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "MEMBER_SHA256_MISMATCH"):
                v7cli.verify_packet_inventory(packet)

    def test_training_or_promotion_cannot_be_enabled_by_plan(self):
        self.plan["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "G2_CANNOT_AUTHORIZE_TRAINING_OR_PROMOTION"):
            v7.validate_plan(self.plan, PROFILE)


if __name__ == "__main__":
    unittest.main()
