#!/usr/bin/env python3
"""Research CI must not automatically publish commercial model or SDK releases.

Python 3.8+, no network, training, TTS/ASR, or audio side effects.
The historical publication commands remain available in Git history only.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"



def require_no_live_release_authority(source: str, path: str) -> None:
    """Active YAML must not publish or download Release as scientific authority.

    Keep this source-only: comments and historical docs are not executable.
    This does not disable any C/ASR/TTS/model quality regression gate.
    """
    header = source.split("\njobs:\n", 1)[0]
    if re.search(r"(?m)^  release:\s*$", header):
        raise AssertionError(path + ": release publication trigger is forbidden")
    if re.search(r"(?m)^  push:\s*$", header) and re.search(
            r"(?m)^    tags:\s*$", header):
        raise AssertionError(path + ": tag-push publication trigger is forbidden")
    if re.search(r"(?m)^\s*workflows:\s*\[\s*['\"]?Release\b", header):
        raise AssertionError(path + ": release-dependent workflow_run is forbidden")
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(
                r"^(?:-\s*run:\s*|run:\s*)?gh\s+release\s+"
                r"(?:create|upload|download|edit|delete|view|publish)\b",
                line):
            raise AssertionError(path + ": gh release command is forbidden")
        if re.match(
                r"^(?:-\s*run:\s*|run:\s*)?gh\s+api\s+.*?/releases(?:/|\b)",
                line):
            raise AssertionError(path + ": GitHub Release API use is forbidden")
        if re.match(
                r"^(?:-\s*run:\s*|run:\s*)?gh\s+api\s+.*?\breleases\b",
                line):
            raise AssertionError(path + ": GitHub Release API use is forbidden")
        if re.search(
                r"^\s*(?:-\s*)?uses:\s*(?:softprops/action-gh-release|"
                r"actions/create-release|actions/upload-release-asset)(?:@|\b)",
                line):
            raise AssertionError(path + ": publisher action is forbidden")


class ResearchPublicationBoundaryTests(unittest.TestCase):
    def test_active_workflow_tree_has_no_release_execution_authority(self):
        workflows = sorted(WORKFLOWS.glob("*.yml"))
        self.assertGreaterEqual(len(workflows), 20)
        for path in workflows:
            with self.subTest(path=path.name):
                require_no_live_release_authority(
                    path.read_text(encoding="utf-8"), str(path)
                )

    def test_release_authority_counterexamples_are_rejected(self):
        bad = (
            "name: old\non:\n  release:\n    types: [published]\njobs:\n",
            "name: old\non:\n  push:\n    tags:\n      - v*\njobs:\n",
            "name: old\non:\n  workflow_run:\n    workflows: [Release]\njobs:\n",
            "on:\n  workflow_dispatch:\njobs:\n  test:\n    steps:\n"
            "      - run: gh release create v1.0.0 build/model.kwm\n",
            "on:\n  workflow_dispatch:\njobs:\n  test:\n    steps:\n"
            "      - run: gh release download v1.0.0\n",
            "on:\n  workflow_dispatch:\njobs:\n  test:\n    steps:\n"
            "      - uses: softprops/action-gh-release@v2\n",
            "on:\n  workflow_dispatch:\njobs:\n  test:\n    steps:\n"
            "      - run: gh api repos/owner/repo/releases/tags/v1\n",
        )
        for source in bad:
            with self.subTest(source=source[:45]):
                with self.assertRaises(AssertionError):
                    require_no_live_release_authority(source, "fixture")
        # A historical mention in a comment is not a live shell command.
        require_no_live_release_authority(
            "name: review\non:\n  pull_request:\njobs:\n"
            "  source:\n    steps:\n      - run: echo research-only\n"
            "      # historical gh release create has been retired\n",
            "fixture-clean",
        )

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

    def test_shipping_comparator_and_legacy_asset_verifier_are_retired(self):
        self.assertFalse((ROOT / "configs/shipping.xiaowo.json").exists())
        self.assertFalse((ROOT / "tools/verify_model_release.py").exists())
        current = ROOT / "configs/research/two-keyword-synthetic-comparator.json"
        self.assertTrue(current.is_file())
        import json
        comparator = json.loads(current.read_text(encoding="utf-8"))
        self.assertEqual(comparator["research_scope"], "frozen-historical-synthetic-comparator")
        self.assertNotIn("shipping_approved", comparator)
        self.assertNotIn("shipping_wake_words", comparator)
        self.assertNotIn("shipping_evidence_boundary", comparator)
        self.assertFalse(comparator["independent_measurement_requirements"]["actual_ssc305_measured"])
        self.assertTrue((ROOT / "tools/verify_research_model_assets.py").is_file())

    def test_research_document_tracks_later_migration(self):
        doc = (ROOT / "docs/RESEARCH_ONLY_EXECUTION.md").read_text(encoding="utf-8")
        self.assertIn("Retired automatic publication surfaces", doc)
        self.assertIn("model-promotion.yml", doc)
        self.assertIn("shipping-approval.yml", doc)
        self.assertIn("real-human-qualification.yml", doc)
        self.assertIn("target-dut-qualification.yml", doc)


if __name__ == "__main__":
    unittest.main()
