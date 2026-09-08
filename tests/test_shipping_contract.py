from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_keyword_rows(path: pathlib.Path) -> list[list[str]]:
    return [
        raw.split("\t")
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]


def main() -> int:
    contract = json.loads((ROOT / "configs" / "shipping.xiaowo.json").read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    assert contract["contract_id"] == "xiaowo-dual-wake-v1"
    assert contract["product_scope"] == "dedicated-two-keyword-mandarin-kws"
    assert contract["evidence_status"] == "synthetic-qualified"
    assert contract["shipping_approved"] is False

    words = contract["shipping_wake_words"]
    assert words == [
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
    assert contract["forbidden_shipping_wake_words"] == ["小窝"]
    assert all(len(row["text"]) == 4 for row in words)

    vocab = contract["vocabulary"]
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

    model = contract["model"]
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

    nightly_policy = contract["nightly_policy"]
    assert nightly_policy["mode"] == "immutable-released-model-regression"
    assert nightly_policy["may_train"] is False
    assert nightly_policy["may_render_formal_qualification"] is False
    assert nightly_policy["may_consume_formal_qualification_seed"] is False
    assert nightly_policy["formal_qualification_seed"] == 271838
    assert nightly_policy["next_formal_candidate_seed_reserved"] == 271839

    nightly_config_path = ROOT / "configs" / "nightly.xiaowo-frozen-model.json"
    nightly_config = json.loads(nightly_config_path.read_text(encoding="utf-8"))
    assert nightly_config["regression_namespace"] == "nightly-frozen-model-v1"
    assert nightly_config["seed"] == 880137
    for key in (
        "qualification_holdout_seed",
        "retired_qualification_holdout_seeds",
        "far_holdout_round_namespace",
        "retired_far_holdout_round_namespaces",
    ):
        assert key not in nightly_config, f"nightly config leaked formal namespace: {key}"

    workflow = (ROOT / ".github" / "workflows" / "far-nightly.yml").read_text(encoding="utf-8")
    required = (
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
    )
    for value in required:
        assert value in workflow, f"far-nightly missing frozen-model contract: {value}"

    forbidden_workflow = (
        "training/iterate_domain.py",
        "training/train_ctc.py",
        "training/render_qualification_holdout.py",
        "training/finalize_domain_candidate.py",
        "configs/training/xiaowo.torch-domain.json",
        ".venv/bin/python",
        "torch==",
    )
    for value in forbidden_workflow:
        assert value not in workflow, f"far-nightly must not train or consume formal qualification: {value}"

    boundary = contract["shipping_evidence_boundary"]
    assert boundary["final_afe_required"] is True
    assert boundary["real_human_acoustic_required"] is True
    assert boundary["physical_target_board_required"] is True
    assert contract["shipping_approved"] is False

    print("test_shipping_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
