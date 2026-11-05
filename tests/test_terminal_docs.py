from __future__ import annotations

import copy
import json
import hashlib
from decimal import Decimal
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
    # Old product release procedures are preserved only as source records.
    release_doc = (ROOT / "docs/RELEASE_QUALIFICATION.md").read_text(encoding="utf-8")
    target_doc = (ROOT / "docs/TARGET_EVIDENCE.md").read_text(encoding="utf-8")
    roadmap_doc = (ROOT / "docs/KWS_RESEARCH_PRODUCT_ROADMAP.md").read_text(encoding="utf-8")
    assert "Historical record only; not an active KWS technical-research" in release_doc
    assert "not current runnable research instructions" in release_doc
    assert "Technical research only" in target_doc
    assert "not product Shipping or certification authority" in target_doc
    assert "旧配置已从活动主线退役" in roadmap_doc.split("\n## ", 1)[0]
    assert "产品机器权威仍为" not in roadmap_doc.split("\n## ", 1)[0]
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
    policy = json.loads((ROOT / "configs/research/history/target-dut-phase-b.policy.json").read_text(encoding="utf-8"))
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
    # No product Release publisher is a current research preflight. Preserve
    # resource-budget source/schema checks above, and the offline scorers.
    assert not (ROOT / ".github/workflows/target-dut-qualification.yml").exists()
    assert not (ROOT / ".github/workflows/real-human-qualification.yml").exists()
    assert (ROOT / "tools/score_target_dut_qualification.py").is_file()
    assert (ROOT / "tools/score_real_human_qualification.py").is_file()


def check_current_research_guidance() -> None:
    customization_path = ROOT / "docs/CUSTOMIZATION.md"
    customization = customization_path.read_text(encoding="utf-8")
    require_all(customization, customization_path, (
        "## Frozen synthetic research comparator",
        "models/registry/model-749187ec1d66/",
        "no GitHub Release downloads",
        "independent real-speech qualification",
    ))
    for stale in (
        "## Current shipping SKU",
        "Nightly is not another training/qualification path. It must download",
        "Production should provide explicit pinyin",
    ):
        assert stale not in customization, "stale current research guidance: " + stale

    index_path = ROOT / "docs/README.md"
    index = index_path.read_text(encoding="utf-8")
    assert "are being migrated under [#704]" not in index
    require_all(index, index_path, (
        "last live training Release-read migration",
        "pins the research archive locally by SHA256",
    ))

    training_path = ROOT / "docs/PRODUCT_MODEL_TRAINING.md"
    training = training_path.read_text(encoding="utf-8")
    require_all(training, training_path, (
        "## Current research materialization (2026-11-05)",
        "do **not** call `gh release download`",
        "retired workflow",
    ))

    landing_path = ROOT / "docs/KWS_LANDING_STATUS.md"
    landing = landing_path.read_text(encoding="utf-8")
    require_all(landing, landing_path, (
        "Historical engineering checkpoint",
        "**not** current research gates",
    ))
    assert "Shipping/recalibration authority stays" not in landing

    archive_path = ROOT / "docs/KWS_PRODUCTION_CLOSURE_2026-10-29.md"
    archive_header = archive_path.read_text(encoding="utf-8").split("\n## ", 1)[0]
    assert "历史快照，非当前研究执行说明" in archive_header
    assert "RESEARCH_ONLY_EXECUTION.md" in archive_header

    contributing_path = ROOT / "CONTRIBUTING.md"
    contributing = contributing_path.read_text(encoding="utf-8")
    require_all(contributing, contributing_path, (
        "This repository is KWS technical research only.",
        "## Reproducible research training",
        "## Independent physical-board research evidence",
        "## Archived qualification-format regression (not an active gate)",
        "Current research CI **does not publish commercial model/SDK GitHub Releases**",
        "source rights, speaker/session/source metadata",
    ))
    for retired_instruction in (
        "Shipping training should use an immutable OCI base",
        "## Product-board evidence invariant",
        "## Release qualification",
        "Formal release workflow must not publish",
        "matching `shipping_approved=true` SKU policy;",
    ):
        assert retired_instruction not in contributing, (
            "stale contributor Shipping/Release instruction: " + retired_instruction
        )


TARGET_RESEARCH_REQUIRED = (
    "Technical research only", "not product Shipping or certification authority",
    "Hosted fixtures and QEMU cannot substitute for physical measurements",
    "NOT_RUN", "BLOCKED", "tools/verify_model_registry.py",
    "tools/collect_runtime_soak.py", "--hours", "--sample-seconds", "--output", "--command",
    "process-cpu-one-core-v1", "process_cpu_seconds / wall_seconds * 100",
    "100% means one fully occupied core", "Descendants are outside the sampler's scope",
    "`audio_seconds` and `cpu_seconds_per_audio_second` remain null",
    "actual post-AFE WAV", "RSS", "stack_high_water_bytes",
    "actual instrument/CSV/calibration identity", "Recompute hashes independently at readback",
    "attestation-verification receipt", "source identity, rights",
)
TARGET_RETIRED_COMMANDS = (
    "python3 tools/qualification_manifest.py",
    "python3 tools/qualification_gate.py",
    "python3 tools/collect_target_evidence.py",
    "--command ./product-kws-soak",
)


def validate_target_research_guidance(text: str) -> None:
    # Explicit failures also exercise this boundary under -O and -OO.
    for value in TARGET_RESEARCH_REQUIRED:
        if value not in text:
            raise AssertionError("current target research guidance missing: " + value)
    for command in TARGET_RETIRED_COMMANDS:
        if command in text:
            raise AssertionError("retired target command in current guidance: " + command)


def check_target_research_guidance() -> None:
    text = (ROOT / "docs/TARGET_EVIDENCE.md").read_text(encoding="utf-8")
    validate_target_research_guidance(text)
    mutations = [text.replace(value, "") for value in TARGET_RESEARCH_REQUIRED]
    mutations += [text + "\n```bash\n" + command + "\n```\n"
                  for command in TARGET_RETIRED_COMMANDS]
    for index, mutated in enumerate(mutations):
        try:
            validate_target_research_guidance(mutated)
        except AssertionError:
            continue
        raise AssertionError(f"target guidance regression {index} was accepted")


def main() -> int:
    check_current_research_guidance()
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
    # Retired qualification formats are still tested in their historical
    # documentation; current landing pages must not be a shipping CLI guide.
    for relative in ("docs/RELEASE_QUALIFICATION.md",):
        path = ROOT / relative
        require_all(path.read_text(encoding="utf-8"), path, required_evidence_cli)

    required_manifest_cli = ("--evidence-raw", "--attestation-verification", "--sku", "--corpus-id")
    path = ROOT / "docs/RELEASE_QUALIFICATION.md"
    require_all(path.read_text(encoding="utf-8"), path, required_manifest_cli)

    for relative in ("README.md", "README.zh-CN.md"):
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        require_all(source, path, (
            "tools/collect_runtime_soak.py", "tools/collect_target_evidence.py",
            "--historical-replay-only", "NOT_RUN", "docs/TARGET_EVIDENCE.md",
        ))
        assert "python3 tools/qualification_manifest.py" not in source, path
        assert "python3 tools/qualification_gate.py" not in source, path

    workflow = (ROOT / ".github" / "workflows" / "training-integration.yml").read_text(encoding="utf-8")
    assert "vars.KWS_TRAINING_IMAGE" in workflow
    for relative in ("README.md", "README.zh-CN.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        assert "KWS_TRAINING_IMAGE" in path.read_text(encoding="utf-8"), f"{path}: training image variable drift"

    # The archived measurement protocol still requires exact raw external
    # attestation; current research guides describe the independent collector
    # without prescribing retired product-approval CLI arguments.
    for relative in ("docs/RELEASE_QUALIFICATION.md",):
        path = ROOT / relative
        require_all(
            path.read_text(encoding="utf-8"),
            path,
            ("tools/collect_target_evidence.py", "--attestation-verification"),
        )

    # Model source identity is research-only; product-board branding in an
    # old page is not a scientific validation prerequisite.
    assert not (ROOT / "configs/shipping.xiaowo.json").exists()
    assert (ROOT / "configs/research/two-keyword-synthetic-comparator.json").is_file()

    # Old schema/CLI identity belongs to the archive, not the current guide.
    archived_evidence = ROOT / "docs" / "RELEASE_QUALIFICATION.md"
    require_all(
        archived_evidence.read_text(encoding="utf-8"),
        archived_evidence,
        ("schema v3", "process-cpu-one-core-v1", "--runtime-soak", "--power-raw", "external attestation"),
    )
    check_target_research_guidance()

    baseline = json.loads(
        (ROOT / "configs/research-baseline.xiaowo.json").read_text(encoding="utf-8")
    )
    assert baseline["schema"] == "kws-research-model-baseline-v1"
    assert baseline["policy"] == "frozen-historical-research-comparator"
    assert baseline["dataset_status"] == "exposed-historical-synthetic-diagnostic"
    assert baseline["model_tag"] == "model-749187ec1d66"
    assert baseline["training_source_sha"] == "749187ec1d6662658f06aa9c76d47fde835968db"
    assert baseline["model_sha256"] == "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402"
    assert baseline["keyword_pack_sha256"] == "370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723"
    assert baseline["frozen_qualification_seed"] == 271838
    assert baseline["frozen_seed_state"] == "consumed-and-frozen"
    for field in ("actual_human_wav_evaluated", "independent_longstream_far_measured", "ssc305_measured"):
        assert baseline[field] is False
    formal_training = json.loads(
        (ROOT / "configs/training/xiaowo.torch-domain.json").read_text(encoding="utf-8")
    )
    assert baseline["next_research_seed"] == int(formal_training["qualification_holdout_seed"])

    registry = ROOT / "models" / "registry" / baseline["model_tag"]
    for filename, field in (
        ("xiaowo-model.kwm", "model_sha256"),
        ("xiaowo-keywords.kwk", "keyword_pack_sha256"),
    ):
        assert hashlib.sha256((registry / filename).read_bytes()).hexdigest() == baseline[field]
    assert (ROOT / "keywords/tokens.example.txt").read_text(encoding="utf-8").splitlines() == [
        "<blk> 0", "ni3 1", "hao3 2", "xiao3 3", "wo1 4",
    ]
    # Preserve the byte-exact archived TSV spelling; never rewrite the frozen
    # registry or weaken the independent numeric research-keyword validator.
    keyword_rows = load_keyword_rows(registry / "xiaowo-keywords.tsv")
    assert keyword_rows == [
        ["1", "你好小窝", "0.550000", "ni3 hao3 xiao3 wo1"],
        ["2", "小窝小窝", "0.550000", "xiao3 wo1 xiao3 wo1"],
    ]
    assert [Decimal(row[2]) for row in keyword_rows] == [Decimal("0.55")] * 2

    nightly_config = json.loads(
        (ROOT / "configs/nightly.xiaowo-frozen-model.json").read_text(encoding="utf-8")
    )
    assert nightly_config["regression_namespace"] == "nightly-frozen-model-v1"
    assert nightly_config["seed"] == 880137
    for key in (
        "qualification_holdout_seed", "retired_qualification_holdout_seeds",
        "far_holdout_round_namespace", "retired_far_holdout_round_namespaces",
    ):
        assert key not in nightly_config, "nightly cannot consume formal holdout namespace: " + key

    nightly_workflow = (ROOT / ".github/workflows/far-nightly.yml").read_text(encoding="utf-8")
    for value in (
        "FROZEN_MODEL_TAG: model-749187ec1d66",
        "EXPECTED_TRAINED_HEAD_SHA: 749187ec1d6662658f06aa9c76d47fde835968db",
        "EXPECTED_TRAINING_RUN_ID: \"34134789576\"",
        "EXPECTED_MODEL_SHA256: ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402",
        "EXPECTED_KEYWORD_PACK_SHA256: 370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723",
        "verify_model_registry.py", "configs/research-baseline.xiaowo.json",
        "validate_research_keyword_pair.py", "MODEL_SHA256SUMS",
        "training/render_domains.py", "eval/long_far_stream.py",
        "eval/aggregate_far.py", "far_regression_receipt.py produce",
        "training_performed': False", "formal_qualification_seed_consumed': False",
    ):
        assert value in nightly_workflow, "research nightly missing scientific contract: " + value
    for forbidden in (
        "gh release", "configs/shipping.xiaowo.json", "shipping_approved",
        "model-promotion.yml", "training/iterate_domain.py", "training/train_ctc.py",
        "training/render_qualification_holdout.py", "training/finalize_domain_candidate.py",
        "configs/training/xiaowo.torch-domain.json", ".venv/bin/python", "torch==",
    ):
        assert forbidden not in nightly_workflow, "research nightly has legacy authority: " + forbidden

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

    integration = (ROOT / "docs/INTEGRATION.md").read_text(encoding="utf-8")
    require_all(
        integration, ROOT / "docs/INTEGRATION.md",
        ("Historical deployment identity (retired)", "independent measurements"),
    )
    assert (ROOT / "docs/RESEARCH_ONLY_EXECUTION.md").is_file()

    governance_doc = (ROOT / "docs" / "GOVERNANCE_TARGET.md").read_text(encoding="utf-8")
    require_all(
        governance_doc,
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
        (
            "governance/main-ruleset-target.json",
            "kws-main-terminal",
            "403 Resource not accessible by integration",
            "former `deployment/commercial-candidate` branch",
            "not product publication",
            "NOT_RUN/FAIL",
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
