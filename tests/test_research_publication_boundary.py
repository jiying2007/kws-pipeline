#!/usr/bin/env python3
"""Research CI must not automatically publish commercial model or SDK releases.

Python 3.8+, no network, training, TTS/ASR, or audio side effects.
The historical publication commands remain available in Git history only.
"""
from pathlib import Path
import hashlib
import json
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


# Research-only execution forbids indirect Release readers in every active job.
SHELL_CALL_RE = re.compile(
    r"\b(?:bash|sh)[ \t]+((?:training|tools)/[A-Za-z0-9_./-]+\.sh)\b"
)
RELEASE_READ_RE = re.compile(
    r"\bgh[ \t]+release[ \t]+(?:download|view)\b"
    r"|\bgh[ \t]+api[^\n]*?/releases(?:/|\b)"
)
EXPECTED_INDIRECT_RELEASE_READERS = frozenset()


def find_indirect_release_readers(workflows, scripts):
    """Identify Release-reading shell scripts actually referenced by live YAML.

    Keep source-only and Python 3.8 compatible. This first-pass detector covers
    direct bash/sh references in the training/tools paths, not arbitrary dynamic
    shell or Python imports; migration #733 must close the broader dependency.
    """
    invoked = set()
    for workflow in workflows.values():
        invoked.update(SHELL_CALL_RE.findall(workflow))
    return {
        path for path in invoked
        if path in scripts and RELEASE_READ_RE.search(scripts[path])
    }


def require_no_new_indirect_release_readers(workflows, scripts, known_debt):
    readers = find_indirect_release_readers(workflows, scripts)
    new = readers - set(known_debt)
    if new:
        raise AssertionError("new indirect Release readers: " + ", ".join(sorted(new)))
    return readers


class ResearchPublicationBoundaryTests(unittest.TestCase):

    def test_orphan_release_registry_importer_is_fail_closed(self):
        path = ROOT / "tools/materialize_model_registry.py"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8")
        self.assertLess(len(source), 500)
        self.assertIn('raise SystemExit("Retired Release importer:', source)
        self.assertNotIn("shipping_approved", source)
        self.assertNotIn("gh release", source)
        self.assertNotIn("import requests", source)
        self.assertTrue((ROOT / "tools/verify_model_registry.py").is_file())
        registry_ci = (WORKFLOWS / "model-registry-contract.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("python3 -m py_compile tools/materialize_model_registry.py", registry_ci)
        self.assertIn("tools/verify_model_registry.py", registry_ci)
        index = (ROOT / "tools/README.md").read_text(encoding="utf-8")
        self.assertIn("Inert fail-closed tombstone", index)
        self.assertIn("Historical-format offline replay only", index)


    def test_indirect_release_readers_are_zero(self):
        workflows = {
            p.name: p.read_text(encoding="utf-8")
            for p in WORKFLOWS.glob("*.yml")
        }
        scripts = {
            str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
            for directory in ("training", "tools")
            for p in (ROOT / directory).rglob("*.sh")
        }
        actual = require_no_new_indirect_release_readers(
            workflows, scripts, EXPECTED_INDIRECT_RELEASE_READERS
        )
        self.assertEqual(actual, set(), "no active workflow may indirectly read GitHub Releases")

    def test_exact_frozen_research_archive_and_non_admission_receipt(self):
        root = ROOT / "research" / "legacy-source-bases"
        contract = json.loads(
            (ROOT / "configs/training/product-speech-like-base-v1.json").read_text(
                encoding="utf-8"
            )
        )
        receipt = json.loads((root / "migration-receipt.json").read_text(encoding="utf-8"))
        archive = root / "speech-like-base-5204b798033f.tar.gz"
        digest = hashlib.sha256()
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), contract["release_archive_sha256"])
        self.assertEqual(digest.hexdigest(), receipt["archive_sha256"])
        self.assertEqual(archive.stat().st_size, receipt["archive_size_bytes"])
        self.assertEqual(receipt["source_release_id"], 392032606)
        self.assertEqual(receipt["source_asset_id"], 574584640)
        self.assertEqual(receipt["archive_wav_count"], 384)
        self.assertFalse(receipt["actual_speech_admitted"])
        self.assertFalse(receipt["product_publication_authority"])
        self.assertFalse(contract["protected_evidence_used"])

        materializer = (ROOT / "training/materialize_governed_product_base.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("research/legacy-source-bases/$archive", materializer)
        self.assertIn("sha256sum -c --status", materializer)
        self.assertNotRegex(materializer, RELEASE_READ_RE)
        self.assertNotIn("KWS_PRODUCT_RELEASE_ROOT", materializer)
        self.assertFalse((WORKFLOWS / "research-legacy-base-one-shot.yml").exists())

    def test_indirect_release_reader_negative_and_clean_fixtures(self):
        workflows = {"live.yml": "jobs:\n  train:\n    steps:\n"
                     "      - run: bash training/research-base.sh\n"}
        bad = {"training/research-base.sh":
               '#!/bin/sh\n gh release download legacy-tag\n'}
        with self.assertRaisesRegex(AssertionError, "new indirect Release readers"):
            require_no_new_indirect_release_readers(workflows, bad, set())
        bad["training/research-base.sh"] = (
            '#!/bin/sh\n gh api "repos/o/r/releases/tags/legacy"\n'
        )
        with self.assertRaisesRegex(AssertionError, "new indirect Release readers"):
            require_no_new_indirect_release_readers(workflows, bad, set())
        good = {"training/research-base.sh": "#!/bin/sh\nsha256sum -c sums\n"}
        self.assertEqual(
            require_no_new_indirect_release_readers(workflows, good, set()), set()
        )

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

    def test_legacy_product_qualification_cli_is_replay_only(self):
        source = (ROOT / "tools/qualification_gate.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--historical-replay-only"', source)
        self.assertIn("if not args.historical_replay_only:", source)
        self.assertIn("retired qualification CLI:", source)
        for path in WORKFLOWS.glob("*.yml"):
            self.assertNotIn("tools/qualification_gate.py",
                             path.read_text(encoding="utf-8"),
                             "old product-certified CLI must not be in active research Actions")

    def test_legacy_target_qualification_clis_are_replay_only(self):
        # Old deployment-format scorers may replay numerical records but are
        # never an active research acceptance or Shipping publisher path.
        for name, denied in (
            ("score_target_dut_qualification.py", "retired target-DUT qualification CLI:"),
            ("score_target_cohort.py", "retired target-cohort qualification CLI:"),
        ):
            with self.subTest(name=name):
                source = (ROOT / "tools" / name).read_text(encoding="utf-8")
                self.assertIn('parser.add_argument("--historical-replay-only"', source)
                self.assertIn("if not args.historical_replay_only:", source)
                self.assertIn(denied, source)
                for path in WORKFLOWS.glob("*.yml"):
                    self.assertNotIn(
                        "tools/" + name, path.read_text(encoding="utf-8"),
                        "historical target qualification CLI cannot be active Actions",
                    )

    def test_legacy_manifest_and_human_qualification_clis_are_replay_only(self):
        # Neither historical producer nor old deployment-tagged human scorer
        # may become an active admission/publication tool.
        for filename, denial in (
            ("qualification_manifest.py", "retired qualification manifest CLI:"),
            ("score_real_human_qualification.py", "retired real-human qualification CLI:"),
        ):
            with self.subTest(filename=filename):
                source = (ROOT / "tools" / filename).read_text(encoding="utf-8")
                self.assertIn('parser.add_argument("--historical-replay-only"', source)
                self.assertIn("if not args.historical_replay_only:", source)
                self.assertIn(denial, source)
                for workflow in WORKFLOWS.glob("*.yml"):
                    self.assertNotIn(
                        "tools/" + filename, workflow.read_text(encoding="utf-8"),
                        "retired qualification CLI must not be an active Actions path",
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

    def test_retired_commercial_bootstrap_helper_is_not_active(self):
        self.assertFalse((ROOT / "tools/cleanup_bootstrap_branch.sh").exists())
        index = (ROOT / "tools/README.md").read_text(encoding="utf-8")
        self.assertNotIn("](cleanup_bootstrap_branch.sh)", index)

    def test_current_readmes_do_not_reintroduce_release_download_as_nightly(self):
        """Docs must describe the active Git-registry research path, not old releases."""
        for relative, heading, terminator, required in (
            ("README.md", "### Nightly regression is intentionally separate",
             "\n## Final AFE integration contract",
             "does **not** download or consult a GitHub Release"),
            ("README.zh-CN.md", "### Nightly 与 formal qualification 完全分离",
             "\n## 最终 AFE 接入契约", "不下载 GitHub Release"),
        ):
            with self.subTest(relative=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("models/registry/model-749187ec1d66/", text)
                section = text.split(heading, 1)[1].split(terminator, 1)[0]
                self.assertIn("models/registry/model-749187ec1d66/", section)
                self.assertIn(required, section)
                self.assertNotIn("gh release download", section)
                self.assertNotIn("下载 exact immutable", section)
                self.assertNotIn("It downloads the exact immutable", section)

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
