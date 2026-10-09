from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def require_all(text: str, path: pathlib.Path, values: tuple[str, ...]) -> None:
    for value in values:
        assert value in text, f"{path}: missing {value}"


def load_keyword_rows(path: pathlib.Path) -> list[list[str]]:
    return [
        raw.split("\t")
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]


def navigation_targets(text: str) -> list[str]:
    return re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)\)", text)


def require_local_navigation(path: pathlib.Path) -> None:
    for target in navigation_targets(path.read_text(encoding="utf-8")):
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith(("#", "//")) or not parsed.path:
            continue
        candidate = path.parent / unquote(parsed.path)
        assert candidate.exists(), f"{path}: missing navigation target {target}"


def check_navigation_indexes() -> None:
    # Current navigation is checked separately from immutable historical notes,
    # whose original local-build references are explicitly explained by the guide.
    for relative in (
        "docs/README.md", "tools/README.md", "research/README.md",
        "research/consolidation/history/READING_NOTES.md",
    ):
        require_local_navigation(ROOT / relative)
    tools = ROOT / "tools"
    expected = {path.name for path in tools.iterdir() if path.suffix in {".py", ".c", ".h"}}
    indexed = navigation_targets((tools / "README.md").read_text(encoding="utf-8"))
    source_targets = [target for target in indexed if pathlib.PurePosixPath(target).suffix in {".py", ".c", ".h"}]
    assert len(source_targets) == len(set(source_targets)), "duplicate engineering-tool index entry"
    assert set(source_targets) == expected, "engineering-tool index is incomplete or stale"
    with tempfile.TemporaryDirectory(prefix="navigation-docs-") as tmp:
        root = pathlib.Path(tmp)
        page = root / "index.md"
        (root / "file name.md").write_text("fixture", encoding="utf-8")
        page.write_text("[ok](file%20name.md#section) [anchor](#here) [web](https://example.invalid/missing)", encoding="utf-8")
        require_local_navigation(page)
        page.write_text("[bad](missing.md)", encoding="utf-8")
        try:
            require_local_navigation(page)
        except AssertionError as error:
            assert "missing.md" in str(error)
        else:
            raise AssertionError("missing local navigation target was accepted")


def check_landing_status_contract() -> None:
    spec = importlib.util.spec_from_file_location("kws_landing_status", ROOT / "tools/kws_landing_status.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status = module.build_status(ROOT)
    module.verify(status)
    assert status["schema_version"] == 2
    assert status["policy"] == "kws-product-landing-status-v2"
    assert "evidence" not in status and "research" not in status
    assert status["assessment_scope"] == "repository-source-contract-only"
    historical = status["historical_release_qualification"]
    assert historical["scope"] == "frozen-model-release-only"
    assert historical["synthetic_qualification_passed"] is True
    assert historical["applies_to_current_source"] is False
    shipping_path = ROOT / "configs/shipping.xiaowo.json"
    shipping = json.loads(shipping_path.read_text(encoding="utf-8"))
    assert historical["qualification"] == shipping["model"]["qualification"]
    current = status["current_source"]
    assert current["recalibration_required"] is True
    assert current["recalibration_reason"] == shipping["threshold_calibration"]["recalibration_reason"]
    assert current["contract_path"] == "configs/shipping.xiaowo.json"
    assert current["contract_sha256"] == module.sha256(shipping_path)
    dated = status["dated_regression_and_research"]
    assert dated["scope"] == "retained-dated-observations-not-live-status"
    snapshot = ROOT / dated["snapshot_path"]
    assert dated["snapshot_sha256"] == module.sha256(snapshot)
    assert snapshot.name == "CURRENT_STATUS_2026-10-09.md"
    assert "2026-10-09T02:43:25Z" in snapshot.read_text(encoding="utf-8")
    assert dated["current_research_entry"] == "research/README.md"
    assert dated["current_admission_entry"].endswith("PLAN-SCHEMA.md#current-admission-checklist")
    assert status["historical_research_closure"]["scope"] == "retained-closed-research-line-not-current-authorization"
    assert status["control_plane"]["scope"] == "repository-file-presence-only"

    def expect_error(call, text: str) -> None:
        try:
            call()
        except ValueError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(f"landing status accepted invalid claim: {text}")

    # A source-contract pass must never imply current acoustic or live external
    # qualification, even when the historical release qualification passed.
    for path, value, message in (
        (("historical_release_qualification", "synthetic_qualification_passed"), False, "historical synthetic"),
        (("historical_release_qualification", "applies_to_current_source"), True, "current source"),
        (("current_source", "synthetic_qualification_checked"), True, "acoustic evaluation"),
        (("live_qualification_checked",), True, "live qualification"),
        (("dated_regression_and_research", "live_status_checked"), True, "live status"),
        (("external_qualification", "real_human_final_afe_passed"), True, "external qualification"),
        (("external_qualification", "physical_target_board_passed"), True, "external qualification"),
        (("product", "shipping_approved"), True, "shipping approval"),
    ):
        changed = copy.deepcopy(status)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        expect_error(lambda: module.verify(changed), message)

    with tempfile.TemporaryDirectory(prefix="landing-status-") as tmp:
        root = pathlib.Path(tmp)
        registry = pathlib.Path(status["model"]["git_registry_path"])
        shutil.copytree(ROOT / registry, root / registry)
        for relative in (
            "configs/shipping.xiaowo.json",
            "configs/training/kws-v2-efficient-encoder-closure-v1.json",
            ".github/workflows/dataset-driven-iteration.yml",
            ".github/workflows/real-human-qualification.yml",
            "commercial/real-human-qualification.policy.json",
            "commercial/target-qualification.policy.json",
            dated["snapshot_path"], dated["current_research_entry"],
            dated["current_admission_entry"].split("#", 1)[0],
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        fixture_shipping = root / "configs/shipping.xiaowo.json"

        def save_fixture(value: dict) -> None:
            fixture_shipping.write_text(json.dumps(value), encoding="utf-8")

        for invalid in (None, 0, "false"):
            changed = copy.deepcopy(shipping)
            changed["threshold_calibration"]["recalibration_required"] = invalid
            save_fixture(changed)
            expect_error(lambda: module.build_status(root), "explicitly declare recalibration_required")
        changed = copy.deepcopy(shipping)
        del changed["threshold_calibration"]["recalibration_required"]
        save_fixture(changed)
        expect_error(lambda: module.build_status(root), "explicitly declare recalibration_required")
        changed = copy.deepcopy(shipping)
        changed["shipping_approved"] = "false"
        save_fixture(changed)
        expect_error(lambda: module.build_status(root), "shipping_approved must be a boolean")
        changed = copy.deepcopy(shipping)
        changed["threshold_calibration"]["recalibration_required"] = False
        changed["threshold_calibration"]["recalibration_reason"] = "invented fixture only"
        save_fixture(changed)
        projected = module.build_status(root)
        module.verify(projected)
        assert projected["current_source"]["recalibration_required"] is False
        assert projected["current_source"]["recalibration_reason"] == "invented fixture only"
        assert projected["current_source"]["synthetic_qualification_checked"] is False
        assert projected["product"]["shipping_approved"] is False
        save_fixture(shipping)
        output = root / "output/status.json"
        subprocess.run([sys.executable, "-B", str(ROOT / "tools/kws_landing_status.py"),
                        "--root", str(root), "--verify", "--output", str(output)], check=True)
        assert json.loads(output.read_text(encoding="utf-8")) == module.build_status(root)
        (root / dated["snapshot_path"]).unlink()
        expect_error(lambda: module.build_status(root), "status evidence/navigation missing")
        shutil.copyfile(snapshot, root / dated["snapshot_path"])
        asset = root / registry / "xiaowo-model.kwm"
        asset.write_bytes(asset.read_bytes() + b"invented-corruption")
        expect_error(lambda: module.build_status(root), "pinned asset digest mismatch")


def check_current_status_docs() -> None:
    for relative in ("README.md", "README.zh-CN.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "models/registry/model-749187ec1d66/" in text
        assert "models/registry/**" in text
        assert "recalibration_required=true" in text
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "cannot fabricate Trie transitions" not in architecture
    for symbol in ("KWS_ROOT_START_LOGIT_MARGIN", "KWS_FUZZY_CHILD_RETENTION_COST_LOG",
                   "KWS_SILENCE_RETENTION_LOG", "KWS_MIN_PATH_RETENTION_LOG"):
        assert symbol in architecture
    dataset = (ROOT / "docs/DATASET_ITERATION.md").read_text(encoding="utf-8")
    assert "removes the solution space" not in dataset
    assert "finite, separable dataset" in dataset
    assert "confidence bounds" in dataset
    for name in ("KWS_LANDING_EXECUTION.md", "KWS_RESEARCH_PRODUCT_ROADMAP.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "历史范围" in text.split("\n## ", 1)[0]
        assert "../research/README.md" in text
        assert "PLAN-SCHEMA.md#current-admission-checklist" in text
    landing = ROOT / "docs/KWS_LANDING_STATUS.md"
    require_local_navigation(landing)
    require_all(landing.read_text(encoding="utf-8"), landing, (
        "historical_release_qualification", "current_source", "dated_regression_and_research",
        "external_qualification", "historical_research_closure", "EVAL_INCONCLUSIVE_LABEL_SUPPORT",
        "FAIL", "NOT_RUN", "37860130019", "not acoustic or",
    ))


def main() -> int:
    check_navigation_indexes()
    check_landing_status_contract()
    check_current_status_docs()
    docs = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs" / "TERMINAL_HARDENING.md",
        ROOT / "docs" / "CORPUS_IDENTITY.md",
        ROOT / "docs" / "TARGET_EVIDENCE.md",
        ROOT / "docs" / "RELEASE_QUALIFICATION.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
        ROOT / "docs" / "INTEGRATION.md",
        ROOT / "docs" / "CUSTOMIZATION.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), path

    boundary = (ROOT / "docs" / "TERMINAL_HARDENING.md").read_text(encoding="utf-8")
    assert "does not claim real Mandarin/device qualification" in boundary

    required_evidence_cli = (
        "--evidence-raw",
        "--attestation-verification",
        "--board-runner",
        "--model",
        "--keyword-pack",
        "--board-audio",
        "--sku",
        "--source-sha",
        "--builder-id",
        "--dut-id",
        "--collector-id",
    )
    for relative in ("README.md", "README.zh-CN.md", "docs/TARGET_EVIDENCE.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        require_all(path.read_text(encoding="utf-8"), path, required_evidence_cli)

    required_manifest_cli = ("--evidence-raw", "--attestation-verification", "--sku", "--corpus-id")
    for relative in ("README.md", "README.zh-CN.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        require_all(path.read_text(encoding="utf-8"), path, required_manifest_cli)

    workflow = (ROOT / ".github" / "workflows" / "training-integration.yml").read_text(encoding="utf-8")
    assert "vars.KWS_TRAINING_IMAGE" in workflow
    for relative in ("README.md", "README.zh-CN.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        assert "KWS_TRAINING_IMAGE" in path.read_text(encoding="utf-8"), f"{path}: training image variable drift"

    # Product-board evidence is generated from retained measurements and external
    # attestation; a hand-authored JSON example must not become an alternate
    # authority. Keep the terminal docs bound to the canonical collector instead.
    for relative in (
        "README.md",
        "README.zh-CN.md",
        "docs/TARGET_EVIDENCE.md",
        "docs/RELEASE_QUALIFICATION.md",
    ):
        path = ROOT / relative
        require_all(
            path.read_text(encoding="utf-8"),
            path,
            ("tools/collect_target_evidence.py", "product-board", "--attestation-verification"),
        )

    target_evidence = ROOT / "docs" / "TARGET_EVIDENCE.md"
    require_all(
        target_evidence.read_text(encoding="utf-8"),
        target_evidence,
        ("schema v3", "process-cpu-one-core-v1", "--runtime-soak", "--power-raw", "external attestation"),
    )

    shipping = json.loads((ROOT / "configs" / "shipping.xiaowo.json").read_text(encoding="utf-8"))
    formal_training = json.loads(
        (ROOT / "configs" / "training" / "xiaowo.torch-domain.json").read_text(encoding="utf-8")
    )
    assert shipping["schema_version"] == 1
    assert shipping["contract_id"] == "xiaowo-dual-wake-v1"
    assert shipping["product_scope"] == "dedicated-two-keyword-mandarin-kws"
    assert shipping["evidence_status"] == "synthetic-qualified"
    assert shipping["shipping_approved"] is False
    assert shipping["shipping_wake_words"] == [
        {
            "id": 1,
            "text": "你好小窝",
            "threshold": 0.55,
            "tokens": ["ni3", "hao3", "xiao3", "wo1"],
        },
        {
            "id": 2,
            "text": "小窝小窝",
            "threshold": 0.55,
            "tokens": ["xiao3", "wo1", "xiao3", "wo1"],
        },
    ]
    assert shipping["forbidden_shipping_wake_words"] == ["小窝"]
    assert all(len(row["text"]) == 4 for row in shipping["shipping_wake_words"])

    vocab = shipping["vocabulary"]
    assert vocab["kind"] == "dedicated-product-vocabulary"
    assert vocab["size"] == 5
    assert vocab["arbitrary_mandarin_l0_change_claimed"] is False
    assert (ROOT / vocab["tokens_path"]).read_text(encoding="utf-8").splitlines() == [
        "<blk> 0",
        "ni3 1",
        "hao3 2",
        "xiao3 3",
        "wo1 4",
    ]
    assert load_keyword_rows(ROOT / vocab["keyword_tsv_path"]) == [
        ["1", "你好小窝", "0.55", "ni3 hao3 xiao3 wo1"],
        ["2", "小窝小窝", "0.55", "xiao3 wo1 xiao3 wo1"],
    ]
    model = shipping["model"]
    assert model["release_tag"] == "model-749187ec1d66"
    assert model["training_run_id"] == 34134789576
    assert model["trained_head_sha"] == "749187ec1d6662658f06aa9c76d47fde835968db"
    assert model["model_sha256"] == "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402"
    assert model["keyword_pack_sha256"] == "370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723"
    assert model["qualification_seed"] == 271838
    assert model["qualification_seed_state"] == "consumed-and-frozen"
    assert model["qualification"] == {
        "expected_wakes": 256,
        "matched_wakes": 256,
        "false_rejects": 0,
        "false_accepts": 0,
    }
    assert model["robustness_qualified"] is True
    assert model["continuous_far_false_accepts"] == 0

    nightly_policy = shipping["nightly_policy"]
    assert nightly_policy["mode"] == "immutable-released-model-regression"
    assert nightly_policy["may_train"] is False
    assert nightly_policy["may_render_formal_qualification"] is False
    assert nightly_policy["may_consume_formal_qualification_seed"] is False
    assert nightly_policy["formal_qualification_seed"] == 271838
    assert nightly_policy["next_formal_candidate_seed_reserved"] == int(
        formal_training["qualification_holdout_seed"]
    )

    nightly_config = json.loads(
        (ROOT / "configs" / "nightly.xiaowo-frozen-model.json").read_text(encoding="utf-8")
    )
    assert nightly_config["regression_namespace"] == "nightly-frozen-model-v1"
    assert nightly_config["seed"] == 880137
    for key in (
        "qualification_holdout_seed",
        "retired_qualification_holdout_seeds",
        "far_holdout_round_namespace",
        "retired_far_holdout_round_namespaces",
    ):
        assert key not in nightly_config, f"nightly config leaked formal namespace: {key}"

    nightly_workflow = (ROOT / ".github" / "workflows" / "far-nightly.yml").read_text(encoding="utf-8")
    for value in (
        "MODEL_RELEASE_TAG: model-749187ec1d66",
        "EXPECTED_TRAINED_HEAD_SHA: 749187ec1d6662658f06aa9c76d47fde835968db",
        "EXPECTED_TRAINING_RUN_ID: \"34134789576\"",
        "EXPECTED_MODEL_SHA256: ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402",
        "EXPECTED_KEYWORD_PACK_SHA256: 370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723",
        "gh release download",
        "configs/nightly.xiaowo-frozen-model.json",
        "training/render_domains.py",
        "eval/long_far_stream.py",
        "eval/aggregate_far.py",
        "training_performed': False",
        "formal_qualification_seed_consumed': False",
    ):
        assert value in nightly_workflow, f"far-nightly missing frozen-model contract: {value}"
    for value in (
        "training/iterate_domain.py",
        "training/train_ctc.py",
        "training/render_qualification_holdout.py",
        "training/finalize_domain_candidate.py",
        "configs/training/xiaowo.torch-domain.json",
        ".venv/bin/python",
        "torch==",
    ):
        assert value not in nightly_workflow, f"far-nightly must not train/formally qualify: {value}"

    formal_workflow = (ROOT / ".github" / "workflows" / "model-training.yml").read_text(encoding="utf-8")
    trigger_block = formal_workflow.split("  workflow_dispatch:", 1)[0]
    require_all(
        formal_workflow,
        ROOT / ".github" / "workflows" / "model-training.yml",
        (
            "Refuse consumed formal qualification seed",
            "if active == frozen:",
            "if frozen not in retired:",
            "if active != reserved:",
            "qualification_seed_state",
            "next_formal_candidate_seed_reserved",
            "consumed/frozen",
        ),
    )
    for path_value in (
        "'configs/training/**'",
        "'include/**'",
        "'keywords/**'",
        "'src/**'",
        "'training/**'",
        "'tools/compile_keywords.py'",
        "'tools/kws_vocab.py'",
        "'tools/kws_wav.c'",
        "'tools/kws_raw_stream.c'",
    ):
        assert path_value in trigger_block, f"model-training trigger missing model-sensitive path: {path_value}"
    for non_model_path in (
        "'.github/workflows/model-training.yml'",
        "'CMakeLists.txt'",
        "'cmake/**'",
        "'tools/**'",
        "'tests/test_decoder_retention.c'",
    ):
        assert non_model_path not in trigger_block, f"model-training trigger still includes non-model path: {non_model_path}"

    shipping_boundary = shipping["shipping_evidence_boundary"]
    assert shipping_boundary["final_afe_required"] is True
    assert shipping_boundary["real_human_acoustic_required"] is True
    assert shipping_boundary["physical_target_board_required"] is True
    assert shipping_boundary["pending"] == [
        "real-human-final-afe-acoustic-qualification",
        "physical-target-board-performance-and-soak",
    ]

    afe_schema = json.loads((ROOT / "commercial" / "afe-evidence.schema.json").read_text(encoding="utf-8"))
    assert afe_schema["additionalProperties"] is False
    assert afe_schema["properties"]["backend"]["const"] == "command"
    assert afe_schema["properties"]["shipping_authority"]["const"] is True
    assert afe_schema["properties"]["sample_rate_hz"]["const"] == 16000
    assert afe_schema["properties"]["channels"]["const"] == 1
    assert afe_schema["properties"]["sample_format"]["const"] == "pcm_s16le"
    assert {
        "executable_sha256",
        "config_bundle_sha256",
        "input_pcm_sha256",
        "output_pcm_sha256",
        "result_sidecar_sha256",
        "latency_samples",
        "sku",
        "microphone_revision",
        "enclosure_revision",
    } <= set(afe_schema["required"])

    integration = (ROOT / "docs" / "INTEGRATION.md").read_text(encoding="utf-8")
    require_all(
        integration,
        ROOT / "docs" / "INTEGRATION.md",
        (
            "commercial/afe-evidence.schema.json",
            "configs/shipping.xiaowo.json",
            "shipping_authority=true",
            "commercial-candidate",
            "shipping_approved=false",
        ),
    )

    deployment_workflow = (ROOT / ".github" / "workflows" / "deployment-release.yml").read_text(
        encoding="utf-8"
    )
    for value in (
        "deployment/commercial-candidate",
        "Require exact protected main source",
        "test \"$protected\" = true",
        "configs/shipping.xiaowo.json",
        "configs/nightly.xiaowo-frozen-model.json",
        "commercial/afe-evidence.schema.json",
        "git diff --quiet \"$trained_head\" \"$GITHUB_SHA\" -- CMakeLists.txt cmake include src",
        "sha256sum -c MODEL_SHA256SUMS",
        "tools/check_reproducible_sdk.py",
        "deployment-manifest.json",
        "DEPLOYMENT_SHA256SUMS",
        "shipping_approved': False",
        "actions/attest@",
        '--repo \"$GITHUB_REPOSITORY\"',
        '--target \"$GITHUB_SHA\"',
    ):
        assert value in deployment_workflow, f"deployment release contract missing: {value}"
    for value in (
        "training/iterate_domain.py",
        "training/train_ctc.py",
        "training/render_qualification_holdout.py",
        "qualification_seed = 271839",
        "shipping_approved': True",
    ):
        assert value not in deployment_workflow, f"deployment workflow crosses evidence boundary: {value}"

    ruleset_path = ROOT / "governance" / "main-ruleset-target.json"
    ruleset = json.loads(ruleset_path.read_text(encoding="utf-8"))
    assert ruleset["name"] == "kws-main-terminal"
    assert ruleset["target"] == "branch"
    assert ruleset["enforcement"] == "active"
    assert ruleset["bypass_actors"] == []
    assert ruleset["conditions"] == {
        "ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}
    }
    rule_map = {row["type"]: row for row in ruleset["rules"]}
    assert set(rule_map) == {
        "deletion",
        "non_fast_forward",
        "pull_request",
        "required_status_checks",
    }
    assert set(rule_map["deletion"]) == {"type"}
    assert set(rule_map["non_fast_forward"]) == {"type"}
    pull_request = rule_map["pull_request"]["parameters"]
    assert pull_request == {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": False,
        "require_code_owner_review": False,
        "require_last_push_approval": False,
        "required_review_thread_resolution": True,
        "allowed_merge_methods": ["squash"],
    }
    required_checks = rule_map["required_status_checks"]["parameters"]
    assert required_checks["do_not_enforce_on_create"] is False
    assert required_checks["strict_required_status_checks_policy"] is True
    assert [row["context"] for row in required_checks["required_status_checks"]] == [
        "hosted (gcc)",
        "hosted (clang)",
        "coverage",
        "sanitizers",
        "fuzz",
        "armv7-cross",
    ]

    governance_doc = (ROOT / "docs" / "GOVERNANCE_TARGET.md").read_text(encoding="utf-8")
    require_all(
        governance_doc,
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
        (
            "governance/main-ruleset-target.json",
            "kws-main-terminal",
            "403 Resource not accessible by integration",
            "deployment/commercial-candidate",
        ),
    )
    cleanup_workflow = (ROOT / ".github" / "workflows" / "repository-cleanup.yml").read_text(
        encoding="utf-8"
    )
    assert "governance/" in cleanup_workflow

    print("test_terminal_docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
