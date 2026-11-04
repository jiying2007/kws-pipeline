#!/usr/bin/env python3
"""Research CI must not automatically publish commercial model or SDK releases.

Python 3.8+, no network, training, TTS/ASR, or audio side effects.
The historical publication commands remain available in Git history only.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class ResearchPublicationBoundaryTests(unittest.TestCase):
    def test_legacy_push_publication_workflows_are_retired(self):
        for relative in (
            "release.yml",
            "deployment-release.yml",
            "shipping-approval.yml",
            "target-cohort-promotion.yml",
            "model-promotion.yml",
            "real-human-qualification.yml",
            "target-dut-qualification.yml",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((WORKFLOWS / relative).exists())

    def test_orphaned_historical_release_tag_helper_stays_retired(self):
        self.assertFalse((ROOT / "tools/verify_release_tag.py").exists())
        docs = (ROOT / "docs/REPOSITORY_GOVERNANCE.md").read_text(encoding="utf-8")
        self.assertIn("Historical release records (not executable research)", docs)
        self.assertIn("now-unreferenced `tools/verify_release_tag.py`", docs)

    def test_historical_product_policy_directory_is_not_live(self):
        self.assertFalse((ROOT / "commercial").exists())
        for filename in ("real-human-phase-a.policy.json", "target-dut-phase-b.policy.json"):
            self.assertTrue((ROOT / "configs/research/history" / filename).is_file())
        for workflow in WORKFLOWS.glob("*.yml"):
            self.assertNotIn("configs/research/history/",
                             workflow.read_text(encoding="utf-8"),
                             "Historical product policy cannot be active research CI")

    def test_scientific_ci_remains_active(self):
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        for required in (
            "tests/test_research_publication_boundary.py",
            "tests/test_corpus_identity.py",
            "tests/test_training_admission.py",
            "tests/test_decoder_policy_replay.py",
            "tools/test_inventory.py --official-workflows",
            "tests/test_sbom.py",
            "tests/test_live_ruleset_contract.py",
            "tests/test_research_model_provenance.py",
        ):
            with self.subTest(required=required):
                self.assertIn(required, ci)
        self.assertNotIn("tests/test_release_workflow_contract.py", ci)

    def test_official_inventory_is_research_first(self):
        inventory = (ROOT / "tools" / "test_inventory.py").read_text(encoding="utf-8")
        self.assertIn("'.github/workflows/ci.yml'", inventory)
        self.assertNotIn("'.github/workflows/release.yml'", inventory)
        self.assertTrue((ROOT / "configs/research-baseline.xiaowo.json").is_file())

    def test_research_target_input_schema_is_not_a_commercial_entrypoint(self):
        policy = __import__("json").loads(
            (ROOT / "configs/research/history/target-dut-phase-b.policy.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            policy["resource_budget"]["schema"],
            "configs/research/physical-target-resource-budget.schema.json",
        )
        self.assertTrue((ROOT / "configs/research/physical-target-profile.schema.json").is_file())
        self.assertTrue((ROOT / "configs/research/physical-target-resource-budget.schema.json").is_file())
        self.assertFalse((ROOT / "commercial/target-bundle.schema.json").exists())
        profile = __import__("json").loads(
            (ROOT / "configs/research/physical-target-profile.schema.json").read_text(encoding="utf-8")
        )
        budget = __import__("json").loads(
            (ROOT / "configs/research/physical-target-resource-budget.schema.json").read_text(encoding="utf-8")
        )
        self.assertTrue(profile["$id"].endswith("/configs/research/physical-target-profile.schema.json"))
        self.assertTrue(budget["$id"].endswith("/configs/research/physical-target-resource-budget.schema.json"))
        self.assertEqual(budget["properties"]["measurement_contract_id"]["const"], "process-cpu-one-core-v1")
        self.assertEqual(set(budget["properties"]["limits"]["required"]),
                         {"max_cpu_percent", "max_rss_kib", "max_stack_high_water_bytes",
                          "max_temp_c", "max_average_power_mw"})

    def test_research_document_tracks_later_migration(self):
        doc = (ROOT / "docs/RESEARCH_ONLY_EXECUTION.md").read_text(encoding="utf-8")
        self.assertIn("Retired automatic publication surfaces", doc)
        self.assertIn("model-promotion.yml", doc)
        self.assertIn("shipping-approval.yml", doc)
        self.assertIn("real-human-qualification.yml", doc)
        self.assertIn("target-dut-qualification.yml", doc)


if __name__ == "__main__":
    unittest.main()
