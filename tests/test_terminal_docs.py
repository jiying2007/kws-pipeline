from __future__ import annotations

import json
import pathlib

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


def main() -> int:
    docs = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs" / "TERMINAL_HARDENING.md",
        ROOT / "docs" / "CORPUS_IDENTITY.md",
        ROOT / "docs" / "TARGET_EVIDENCE.md",
        ROOT / "docs" / "RELEASE_QUALIFICATION.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
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

    evidence_example = json.loads(
        (ROOT / "configs" / "qualification.evidence.example.json").read_text(encoding="utf-8")
    )
    assert evidence_example["schema_version"] == 2
    assert evidence_example["evidence_class"] == "product-board"
    for key in (
        "sku",
        "source_sha",
        "builder_id",
        "dut_id",
        "collector_id",
        "raw_evidence_sha256",
        "attestation_verification_sha256",
        "board_runner_sha256",
        "model_sha256",
        "keyword_pack_sha256",
        "board_audio_sha256",
        "runtime_soak_raw",
    ):
        assert key in evidence_example, f"qualification.evidence.example.json: missing {key}"

    shipping = json.loads((ROOT / "configs" / "shipping.xiaowo.json").read_text(encoding="utf-8"))
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
    assert nightly_policy["next_formal_candidate_seed_reserved"] == 271839

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
        "'tools/build_vocab.py'",
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

    print("test_terminal_docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
