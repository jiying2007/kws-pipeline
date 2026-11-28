"""Source-only frozen G3 training-plan contracts; no torch/model/audio execution."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import english_kws_dual_track_v7_train as trainer


class V7TrainingPlanTests(unittest.TestCase):
    def test_two_candidates_have_independent_one_shot_budgets(self):
        report = trainer.source_check()
        self.assertEqual(report["status"], "TRAINING_PLAN_VALID_NO_MODEL_EXECUTION")
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

    def test_exact_g2_source_run_is_distinct_from_training_run(self):
        plan = trainer.load_plan()
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
        self.assertEqual(identity["run_id"], 46010000001)
        self.assertEqual(identity["source_run_id"], 46000882374)
        self.assertEqual(identity["source_commit"], "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2")
        self.assertNotEqual(identity["run_id"], identity["source_run_id"])

    def test_exact_brand_pronunciations_and_separate_seeds(self):
        plan = trainer.load_plan()
        self.assertEqual(plan["candidates"]["VOSEN"]["ipa"], "/heɪ ˈvoʊ.sən/")
        self.assertEqual(plan["candidates"]["MERO"]["ipa"], "/heɪ ˈmɛroʊ/")
        self.assertNotEqual(plan["candidates"]["VOSEN"]["seed"], plan["candidates"]["MERO"]["seed"])
        self.assertEqual(plan["candidates"]["VOSEN"]["keyword_id"], 1)
        self.assertEqual(plan["candidates"]["MERO"]["keyword_id"], 2)


if __name__ == "__main__":
    unittest.main()
