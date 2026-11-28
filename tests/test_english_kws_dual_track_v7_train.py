"""Source-only frozen G3 training-plan contracts; no torch/model/audio execution."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import wave
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import english_kws_dual_track_v7_train as trainer


class V7TrainingPlanTests(unittest.TestCase):
    def test_two_candidates_have_independent_one_shot_budgets(self):
        report = trainer.source_check()
        self.assertEqual(report["status"], "MULTISOURCE_G3_PLAN_VALID_NO_MODEL_EXECUTION")
        self.assertTrue(report["training_enabled"])
        plan = trainer.load_plan()
        self.assertEqual(plan["status"], "READY_FOR_MULTISOURCE_G3")
        self.assertEqual(plan["source_roles"],
                         {"train": 8, "development": 4, "calibration": 3, "locked-test": 3})
        self.assertEqual(plan["source_bundle"]["selected_machine_reviewed_rows"], 63)
        self.assertEqual(plan["source_bundle"]["new_tts_calls"], 0)
        self.assertEqual(plan["source_bundle"]["new_asr_calls"], 0)
        self.assertEqual(plan["source_bundle"]["new_phone_calls"], 0)
        self.assertEqual(report["source_qualified_groups"]["train"], 8)
        self.assertEqual(report["source_qualified_groups"]["development"], 4)
        self.assertEqual(len(report["source_bundle_sha256"]), 64)
        self.assertEqual(report["candidates"], ["MERO", "VOSEN"])
        self.assertEqual(report["optimizer_updates_per_candidate"], 192)
        self.assertEqual(report["training_forwards_per_candidate"], 384)
        self.assertEqual(report["model_parameters"], 43876)
        self.assertEqual(report["pack_bytes"], 216960)
        self.assertFalse(report["shared_parameters"])
        self.assertFalse(report["shared_optimizer"])
        self.assertEqual(report["training_runs_per_candidate"], 1)
        self.assertEqual(report["retries"], 0)
        self.assertFalse(report["promotion_allowed"])

    def test_pull_request_and_rerun_are_not_training_authority(self):
        plan = trainer.load_plan()
        with self.assertRaisesRegex(ValueError, "PUBLIC_MAIN_FIRST_ATTEMPT_TRAINING_ONLY"):
            trainer.guard(plan, env={}, event={})

    def test_exact_six_artifact_bundle_and_first_main_guard(self):
        plan = trainer.load_plan()
        bundle = plan["source_bundle"]
        self.assertEqual(len(bundle["artifacts"]), 6)
        self.assertEqual(bundle["qualified_groups_by_role"],
                         {"train": 8, "development": 4, "calibration": 3, "locked-test": 3})
        self.assertEqual(bundle["source_policy_git_blob_sha1"],
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertFalse(bundle["audio_regenerated"])
        self.assertFalse(bundle["human_gold"])
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": trainer.REPO,
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_SHA": "f" * 40,
            "GITHUB_RUN_ID": "46010000001",
        }
        event = {
            "before": plan["next_main_before"],
            "after": env["GITHUB_SHA"],
            "ref": "refs/heads/main",
            "forced": False,
            "deleted": False,
            "repository": {"full_name": trainer.REPO, "private": False},
        }
        identity = trainer.guard(plan, env=env, event=event)
        self.assertEqual(identity["source_bundle_sha256"], plan["source_plan_sha256"])
        self.assertEqual(identity["parent"], plan["next_main_before"])

    def test_exact_brand_pronunciations_and_separate_seeds(self):
        plan = trainer.load_plan()
        self.assertEqual(plan["candidates"]["VOSEN"]["ipa"], "/heɪ ˈvoʊ.sən/")
        self.assertEqual(plan["candidates"]["MERO"]["ipa"], "/heɪ ˈmɛroʊ/")
        self.assertNotEqual(plan["candidates"]["VOSEN"]["seed"], plan["candidates"]["MERO"]["seed"])
        self.assertEqual(plan["candidates"]["VOSEN"]["keyword_id"], 1)
        self.assertEqual(plan["candidates"]["MERO"]["keyword_id"], 2)



    def make_snapshot(self, root):
        plan = trainer.load_plan()
        candidate = "VOSEN"
        candidate_plan = plan["candidates"][candidate]
        root.mkdir(parents=True)
        groups = []
        rows = []
        files = []
        total_bytes = 0
        pcm = b"\x00\x00" * 6400
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(pcm)
        wav = wav_buffer.getvalue()
        for role, count in (("train", 8), ("development", 4)):
            for index in range(count):
                group_id = f"{role}-{index}"
                groups.append({
                    "source_group_id": group_id,
                    "source_speaker_id": "speaker-" + group_id,
                    "role": role,
                })
                for label, amount in ((1, 1), (0, 2)):
                    for item_index in range(amount):
                        row_id = f"{group_id}-{'pos' if label else 'neg'}-{item_index}"
                        pcm_path = f"audio/{row_id}.pcm"
                        wav_path = f"audio/{row_id}.wav"
                        (root / pcm_path).parent.mkdir(parents=True, exist_ok=True)
                        (root / pcm_path).write_bytes(pcm)
                        (root / wav_path).write_bytes(wav)
                        for name, raw in ((pcm_path, pcm), (wav_path, wav)):
                            files.append({"path": name, "bytes": len(raw), "sha256": trainer.sha(raw)})
                            total_bytes += len(raw)
                        rows.append({
                            "id": row_id,
                            "role": role,
                            "source_group_id": group_id,
                            "candidate_id": candidate_plan["candidate_id"],
                            "research_class": label,
                            "tier": "silver",
                            "optimizer_allowed": False,
                            "locked_test_sealed": False,
                            "training_eligible": role == "train",
                            "training_class": label if role == "train" else None,
                            "training_weight": 1.0 if role == "train" else 0.0,
                            "pcm_path": pcm_path,
                            "wav_path": wav_path,
                            "pcm_sha256": trainer.sha(pcm),
                            "wav_sha256": trainer.sha(wav),
                            "pcm_samples": 6400,
                        })
        source_run = {
            "repository": trainer.REPO,
            "run_id": "46010000001",
            "commit": "f" * 40,
            "parent": plan["next_main_before"],
            "attempt": 1,
            "source_bundle_sha256": plan["source_plan_sha256"],
        }
        manifest = {
            "schema": "english-kws-dual-track-v7-training-snapshot-manifest-v1",
            "candidate": candidate,
            "candidate_id": candidate_plan["candidate_id"],
            "phrase": candidate_plan["phrase"],
            "locale": "en-US",
            "ipa": candidate_plan["ipa"],
            "ctc_tokens": candidate_plan["token_labels"],
            "ctc_token_ids": candidate_plan["token_ids"],
            "source_plan_sha256": plan["source_plan_sha256"],
            "source_run": source_run,
            "source_groups": groups,
            "role_counts": {"train": 8, "development": 4},
            "rows": rows,
            "training_rows": 24,
            "development_rows": 12,
            "training_snapshot_allowed": True,
            "calibration_rows_included": False,
            "locked_test_rows_included": False,
            "calibration_labels_included": False,
            "locked_test_labels_included": False,
            "calibration_audio_included": False,
            "locked_test_audio_included": False,
            "optimizer_updates": 0,
            "kws_model_forwards": 0,
            "product_promotion_allowed": False,
        }
        manifest_raw = trainer.canonical(manifest) + b"\n"
        (root / "MANIFEST.json").write_bytes(manifest_raw)
        files.append({"path": "MANIFEST.json", "bytes": len(manifest_raw),
                      "sha256": trainer.sha(manifest_raw)})
        total_bytes += len(manifest_raw)
        files.sort(key=lambda item: item["path"])
        inventory = {
            "schema": "english-kws-dual-track-v7-training-snapshot-files-v1",
            "candidate": candidate,
            "files": files,
        }
        inventory_raw = trainer.canonical(inventory) + b"\n"
        (root / "FILES.json").write_bytes(inventory_raw)
        snapshot = {
            "schema": "english-kws-dual-track-v7-training-snapshot-v1",
            "candidate": candidate,
            "candidate_id": candidate_plan["candidate_id"],
            "source_run": source_run,
            "source_plan_sha256": plan["source_plan_sha256"],
            "source_candidate_manifest_sha256": "a" * 64,
            "source_packet_files_manifest_sha256": "b" * 64,
            "manifest_sha256": trainer.sha(manifest_raw),
            "files_manifest_sha256": trainer.sha(inventory_raw),
            "file_count": len(files),
            "verified_bytes": total_bytes,
            "training_rows": 36,
            "development_rows": 12,
            "role_counts": {"train": 8, "development": 4},
            "calibration_rows_included": False,
            "locked_test_rows_included": False,
            "calibration_labels_included": False,
            "locked_test_labels_included": False,
            "calibration_audio_included": False,
            "locked_test_audio_included": False,
            "locked_test_access_in_training_job": False,
            "optimizer_updates": 0,
            "kws_model_forwards": 0,
            "product_promotion_allowed": False,
        }
        (root / "SNAPSHOT.json").write_bytes(trainer.canonical(snapshot) + b"\n")
        return manifest, inventory, snapshot

    def test_valid_candidate_snapshot_reads_only_train_and_development(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "VOSEN"
            manifest, inventory, snapshot = self.make_snapshot(root)
            verified, identity = trainer.verify_training_snapshot(root, "VOSEN", trainer.load_plan())
            self.assertEqual(len(verified["rows"]), 36)
            self.assertEqual({row["role"] for row in verified["rows"]}, {"train", "development"})
            self.assertEqual(identity["training_rows"], 24)
            self.assertEqual(identity["development_rows"], 12)
            self.assertEqual(trainer.read_pcm(verified["rows"][0], root), b"\x00\x00" * 6400)

    def test_locked_test_row_is_rejected_before_any_locked_audio_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "VOSEN"
            manifest, inventory, snapshot = self.make_snapshot(root)
            locked = dict(manifest["rows"][0])
            locked.update({
                "id": "locked-secret",
                "role": "locked-test",
                "source_group_id": "locked-0",
                "pcm_path": "audio/locked-secret.pcm",
                "wav_path": "audio/locked-secret.wav",
                "training_eligible": False,
                "training_class": None,
                "training_weight": 0.0,
            })
            manifest["rows"].append(locked)
            raw = trainer.canonical(manifest) + b"\n"
            (root / "MANIFEST.json").write_bytes(raw)
            snapshot["manifest_sha256"] = trainer.sha(raw)
            (root / "SNAPSHOT.json").write_bytes(trainer.canonical(snapshot) + b"\n")
            with self.assertRaisesRegex(ValueError, "NO_CALIBRATION_OR_LOCKED_ROWS_IN_TRAINING_SNAPSHOT"):
                trainer.verify_training_snapshot(root, "VOSEN", trainer.load_plan())
            self.assertFalse((root / "audio/locked-secret.pcm").exists())

    def test_extra_locked_audio_path_is_rejected_before_file_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "VOSEN"
            manifest, inventory, snapshot = self.make_snapshot(root)
            inventory["files"].append({
                "path": "audio/locked-secret.pcm",
                "bytes": 1,
                "sha256": trainer.sha(b"x"),
            })
            inventory["files"].sort(key=lambda item: item["path"])
            raw = trainer.canonical(inventory) + b"\n"
            (root / "FILES.json").write_bytes(raw)
            snapshot["files_manifest_sha256"] = trainer.sha(raw)
            snapshot["file_count"] = len(inventory["files"])
            (root / "SNAPSHOT.json").write_bytes(trainer.canonical(snapshot) + b"\n")
            with self.assertRaisesRegex(ValueError, "NO_EXTRA_OR_MISSING_TRAIN_DEV_FILES"):
                trainer.verify_training_snapshot(root, "VOSEN", trainer.load_plan())
            self.assertFalse((root / "audio/locked-secret.pcm").exists())


if __name__ == "__main__":
    unittest.main()
