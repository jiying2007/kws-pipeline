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
        ):
            with self.subTest(relative=relative):
                self.assertFalse((WORKFLOWS / relative).exists())

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
        ):
            with self.subTest(required=required):
                self.assertIn(required, ci)
        self.assertNotIn("tests/test_release_workflow_contract.py", ci)

    def test_official_inventory_is_research_first(self):
        inventory = (ROOT / "tools" / "test_inventory.py").read_text(encoding="utf-8")
        self.assertIn("'.github/workflows/ci.yml'", inventory)
        self.assertNotIn("'.github/workflows/release.yml'", inventory)
        self.assertTrue((ROOT / "configs/research-baseline.xiaowo.json").is_file())

    def test_research_document_tracks_later_migration(self):
        doc = (ROOT / "docs/RESEARCH_ONLY_EXECUTION.md").read_text(encoding="utf-8")
        self.assertIn("Retired automatic publication surfaces", doc)
        self.assertIn("model-promotion.yml", doc)
        self.assertIn("shipping-approval.yml", doc)


if __name__ == "__main__":
    unittest.main()
