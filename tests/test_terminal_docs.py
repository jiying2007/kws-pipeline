from __future__ import annotations

import copy
import json
import pathlib
import re
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


def check_research_comparator_status_contract() -> None:
    # Current status is read-only research identity, not release eligibility.
    import hashlib
    baseline_path = ROOT / "configs/research-baseline.xiaowo.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema"] == "kws-research-model-baseline-v1"
    assert baseline["policy"] == "frozen-historical-research-comparator"
    tag = baseline["model_tag"]
    assert re.fullmatch(r"model-[0-9a-f]{12}", tag)
    registry = ROOT / "models" / "registry" / tag
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools/verify_model_registry.py"),
         "--registry-dir", str(registry)],
        check=True, capture_output=True, text=True,
    )
    record = json.loads((registry / "registry.json").read_text(encoding="utf-8"))
    assert record["training"]["head_sha"] == baseline["training_source_sha"]
    for name, key in (("xiaowo-model.kwm", "model_sha256"),
                      ("xiaowo-keywords.kwk", "keyword_pack_sha256")):
        assert hashlib.sha256((registry / name).read_bytes()).hexdigest() == baseline[key]
    assert baseline["dataset_status"] == "exposed-historical-synthetic-diagnostic"
    for key in ("actual_human_wav_evaluated", "independent_longstream_far_measured",
                "ssc305_measured"):
        assert baseline[key] is False, key
    workflow = (ROOT / ".github/workflows/kws-landing-status.yml").read_text(encoding="utf-8")
    require_all(workflow, ROOT / ".github/workflows/kws-landing-status.yml", (
        "kws-research-comparator-status",
        "configs/research-baseline.xiaowo.json",
        "kws-research-model-baseline-v1",
        "frozen-historical-research-comparator",
        "tools/verify_model_registry.py",
        "research_model_accepted", "independent_acoustic_evidence",
    ))
    assert "configs/shipping.xiaowo.json" not in workflow
    assert "commercial/" not in workflow


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


def check_target_bundle_docs() -> None:
    path = ROOT / "docs/TARGET_AND_SHIPPING_PROMOTION.md"
    text = path.read_text(encoding="utf-8")
    policy = json.loads((ROOT / "commercial/target-qualification.policy.json").read_text(encoding="utf-8"))
    budget = policy["resource_budget"]
    schema = json.loads((ROOT / budget["schema"]).read_text(encoding="utf-8"))
    filename = budget["fixed_bundle_file"]
    bundle_block = re.search(r"Required fixed bundle files:\s*```text\n(.*?)\n```", text, re.S)
    assert bundle_block is not None, f"{path}: missing fixed bundle list"
    assert filename in bundle_block[1].splitlines(), f"{path}: fixed bundle list missing {filename}"
    budget_paragraph = re.search(rf"^`{re.escape(filename)}` is required[^\n]+", text, re.M)
    assert budget_paragraph is not None, f"{path}: missing resource budget requirements"
    require_all(budget_paragraph[0], path, (
        f"`fixed_bundle_file={filename}`",
        *(f"`{field}={schema['properties'][field]['const']}`" for field in (
            "schema_version", "status", "measurement_contract_id",
        )),
        "`target-profile.json.resource_budget_id` must match `resource-budget.json.budget_id`",
    ))
    # No archived release-coupled target Actions may be a research authority.
    assert not (ROOT / ".github/workflows/target-dut-qualification.yml").exists()
    scorer = ROOT / "tools/score_target_dut_qualification.py"
    source = scorer.read_text(encoding="utf-8")
    for name in ("validate_resource_budget(", '"--bundle"', '"--policy"',
                 '"--output"'):
        assert name in source, f"{scorer}: missing research evidence check {name}"


def main() -> int:
    check_target_bundle_docs()
    check_navigation_indexes()
    check_research_comparator_status_contract()
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
            "Refuse consumed research evaluation seed",
            "if active == frozen:",
            "if frozen not in retired:",
            "if active != reserved:",
            "frozen_seed_state",
            "next_research_seed",
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
            "Historical deployment identity (retired)",
            "independent measurements",
        ),
    )

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
