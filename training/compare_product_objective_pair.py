#!/usr/bin/env python3
"""Causal comparator for same-runner product objective A/B experiments."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import tempfile

from compare_product_development_pair import (
    ARTIFACT_FILES,
    candidate_name,
    load_artifact,
    metric_summary,
    readback_map,
    record_key,
    records_by_round,
    round_best,
    semantic_corpus_rows,
    semantic_corpus_sha256,
    semantic_wake_balance,
)

VARIABLE_SEQUENCE_MARGIN = "train.sequence_margin_positive_policy"
VARIABLE_CTC_PRIMARY = "train.ctc_primary_policy"
OBJECTIVE_VARIABLES = {
    VARIABLE_SEQUENCE_MARGIN: "sequence_margin_positive_policy",
    VARIABLE_CTC_PRIMARY: "ctc_primary_policy",
}
DEFAULT_CONTROL = "sparse-chronological-v1"
DEFAULT_TREATMENT = "runtime-search-aligned-v1"
DEFAULT_CTC_CONTROL = "standard-v1"
DEFAULT_CTC_TREATMENT = "label-prior-v1"
NEGATIVE = "runtime-executable-v1"


def expected_values_from_pair_receipt(
    path: pathlib.Path,
) -> tuple[str, str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("evidence_class") != "product-development-same-runner-pair-receipt-v1":
        raise ValueError("pair receipt evidence_class mismatch")
    variable = value.get("variable")
    if not isinstance(variable, dict):
        raise ValueError("pair receipt objective variable is missing")
    variable_name = variable.get("name")
    if variable_name not in OBJECTIVE_VARIABLES:
        raise ValueError(
            "pair receipt objective variable must be one of "
            + ", ".join(sorted(OBJECTIVE_VARIABLES))
        )
    control = variable.get("control")
    treatment = variable.get("treatment")
    if (
        not isinstance(control, str)
        or not control
        or not isinstance(treatment, str)
        or not treatment
        or control == treatment
    ):
        raise ValueError("pair receipt objective values must be distinct strings")
    return variable_name, control, treatment


def delete_dotted(value: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cursor = value
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            return
        cursor = child
    cursor.pop(parts[-1], None)


def normalized_config(config: dict, variable_name: str) -> dict:
    value = copy.deepcopy(config)
    value.pop("development_experiment", None)
    delete_dotted(value, variable_name)
    return value


def normalized_overrides(receipt: dict, variable_name: str) -> dict:
    raw = receipt.get("config_overrides")
    if not isinstance(raw, dict):
        raise ValueError("config_overrides must be an object")
    value = copy.deepcopy(raw)
    value.pop(variable_name, None)
    return value


def no_replay(manifest: dict, label: str, failures: list[str]) -> None:
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        failures.append(f"{label} manifest has no records")
        return
    for row in rows:
        if not isinstance(row, dict):
            failures.append(f"{label} manifest has invalid record")
            continue
        tag = f"{label}:r{row.get('round')}:{row.get('frontend')}:{row.get('candidate',0)}"
        if int(row.get("base_failure_replay_examples", 0)) != 0:
            failures.append(f"{tag} replay examples are nonzero")
        if row.get("base_failure_replay_source_rounds") not in ([], None):
            failures.append(f"{tag} replay source rounds are nonempty")


def round0_inputs(
    control_root: pathlib.Path,
    treatment_root: pathlib.Path,
    control: dict,
    treatment: dict,
    *,
    variable_name: str,
    expected_control: str,
    expected_treatment: str,
) -> dict:
    c0 = {record_key(row): row for row in records_by_round(control["manifest"], 0)}
    t0 = {record_key(row): row for row in records_by_round(treatment["manifest"], 0)}
    failures: list[str] = []
    readback_field = OBJECTIVE_VARIABLES.get(variable_name)
    if readback_field is None:
        raise ValueError("unsupported objective variable")
    if set(c0) != set(t0):
        failures.append("round0 candidate set differs")
    crb = readback_map(control["readback"])
    trb = readback_map(treatment["readback"])
    corpus = {}
    for key in sorted(set(c0) & set(t0)):
        left, right = c0[key], t0[key]
        tag = f"round0[{key[1]}:{key[2]}]"
        for field in (
            "training_acoustic_seed_policy",
            "training_acoustic_seed_offset",
            "training_epochs",
            "training_learning_rate",
            "training_seed",
            "hard_negative_replay_examples",
            "warm_started",
            "warm_start_strategy",
        ):
            if left.get(field) != right.get(field):
                failures.append(f"{tag}.{field} differs")
        if semantic_wake_balance(left.get("wake_balance")) != semantic_wake_balance(
            right.get("wake_balance")
        ):
            failures.append(f"{tag}.wake_balance differs")
        lrows = semantic_corpus_rows(control_root, left)
        rrows = semantic_corpus_rows(treatment_root, right)
        corpus[f"{key[1]}:{key[2]}"] = {
            "control": semantic_corpus_sha256(lrows),
            "treatment": semantic_corpus_sha256(rrows),
        }
        if lrows != rrows:
            failures.append(f"{tag}.training corpus content differs")
        lread = crb.get(candidate_name(left))
        rread = trb.get(candidate_name(right))
        if not isinstance(lread, dict) or not isinstance(rread, dict):
            failures.append(f"{tag}.readback missing")
            continue
        if lread.get("sequence_margin_negative_policy") != NEGATIVE:
            failures.append(f"{tag}.control negative policy drifted")
        if rread.get("sequence_margin_negative_policy") != NEGATIVE:
            failures.append(f"{tag}.treatment negative policy drifted")
        if lread.get(readback_field) != expected_control:
            failures.append(
                f"{tag}.control {readback_field} drifted"
            )
        if rread.get(readback_field) != expected_treatment:
            failures.append(
                f"{tag}.treatment {readback_field} drifted"
            )
    return {
        "valid": not failures,
        "failures": failures,
        "candidate_count": len(set(c0) & set(t0)),
        "semantic_corpus_sha256": corpus,
    }


def metric_delta(control: dict, treatment: dict) -> dict:
    ids = sorted(set(control["per_keyword"]) | set(treatment["per_keyword"]), key=int)
    per_keyword = {}
    no_loss = True
    noncollapsed = True
    for key in ids:
        before = control["per_keyword"].get(key)
        after = treatment["per_keyword"].get(key)
        if before is None or after is None or before["expected"] != after["expected"]:
            per_keyword[key] = {"comparable": False}
            no_loss = False
            noncollapsed = False
            continue
        delta = int(after["matched"]) - int(before["matched"])
        no_loss &= delta >= 0
        noncollapsed &= int(after["matched"]) > 0
        per_keyword[key] = {
            "comparable": True,
            "expected": int(before["expected"]),
            "control_matched": int(before["matched"]),
            "treatment_matched": int(after["matched"]),
            "matched_delta": delta,
            "control_recall": before["recall"],
            "treatment_recall": after["recall"],
            "recall_delta": after["recall"] - before["recall"],
        }
    return {
        "matched_delta": treatment["matched"] - control["matched"],
        "false_accepts_delta": treatment["false_accepts"] - control["false_accepts"],
        "far_per_hour_delta": treatment["far_per_hour"] - control["far_per_hour"],
        "frr_delta": treatment["frr"] - control["frr"],
        "per_keyword": per_keyword,
        "recall_improved": treatment["matched"] > control["matched"],
        "no_keyword_match_loss": no_loss,
        "all_keywords_noncollapsed": noncollapsed,
        "false_accepts_nonincreasing": treatment["false_accepts"] <= control["false_accepts"],
        "far_nonincreasing": treatment["far_per_hour"] <= control["far_per_hour"],
    }


def compare_pair(
    control_root: pathlib.Path,
    treatment_root: pathlib.Path,
    *,
    variable_name: str,
    expected_control: str,
    expected_treatment: str,
) -> dict:
    if variable_name not in OBJECTIVE_VARIABLES:
        raise ValueError("unsupported objective variable")
    if (
        not isinstance(expected_control, str)
        or not expected_control
        or not isinstance(expected_treatment, str)
        or not expected_treatment
        or expected_control == expected_treatment
    ):
        raise ValueError(
            "expected objective pair values must be distinct non-empty strings"
        )
    control = load_artifact(control_root)
    treatment = load_artifact(treatment_root)
    framing: list[str] = []
    cr, tr = control["receipt"], treatment["receipt"]
    for label, receipt in (("control", cr), ("treatment", tr)):
        if receipt.get("source_policy") != "exact-pr-head":
            framing.append(f"{label} source policy is not exact-pr-head")
        if receipt.get("development_only") is not True:
            framing.append(f"{label} is not development-only")
        if receipt.get("protected_evidence_used") is not False:
            framing.append(f"{label} used protected evidence")
    if cr.get("pr_base_sha") != tr.get("pr_base_sha"):
        framing.append("PR base SHA differs")
    if cr.get("effective_config_sha256") != tr.get("effective_config_sha256"):
        framing.append("effective product base differs")
    if normalized_overrides(cr, variable_name) != normalized_overrides(
        tr, variable_name
    ):
        framing.append("overrides differ outside paired objective variable")
    if cr.get("config_overrides", {}).get(variable_name) != expected_control:
        framing.append("control objective value mismatch")
    if tr.get("config_overrides", {}).get(variable_name) != expected_treatment:
        framing.append("treatment objective value mismatch")
    if normalized_config(
        control["config"], variable_name
    ) != normalized_config(treatment["config"], variable_name):
        framing.append("materialized configs differ outside paired objective variable")
    for label, item in (("control", control), ("treatment", treatment)):
        train = item["config"].get("train")
        if not isinstance(train, dict) or train.get("sequence_margin_negative_policy") != NEGATIVE:
            framing.append(f"{label} negative policy drifted")

    replay: list[str] = []
    no_replay(control["manifest"], "control", replay)
    no_replay(treatment["manifest"], "treatment", replay)
    inputs = round0_inputs(
        control_root,
        treatment_root,
        control,
        treatment,
        variable_name=variable_name,
        expected_control=expected_control,
        expected_treatment=expected_treatment,
    )

    c0 = metric_summary(round_best(control["manifest"], 0))
    t0 = metric_summary(round_best(treatment["manifest"], 0))
    c1 = metric_summary(round_best(control["manifest"], 1))
    t1 = metric_summary(round_best(treatment["manifest"], 1))
    d0 = metric_delta(c0, t0)
    d1 = metric_delta(c1, t1)

    causal = not framing and not replay and inputs["valid"]
    positive = (
        causal
        and d1["recall_improved"]
        and d1["no_keyword_match_loss"]
        and d1["all_keywords_noncollapsed"]
        and d1["false_accepts_nonincreasing"]
        and d1["far_nonincreasing"]
    )
    if not causal:
        result_class, next_action = "causal-invalid", "do-not-interpret"
    elif positive:
        result_class = "positive-under-strict-precision-preserving-recall-gain"
        next_action = "candidate-follow-up-without-changing-this-pair"
    else:
        result_class, next_action = "negative", "close-no-sweep"

    return {
        "schema_version": 1,
        "evidence_class": "product-development-objective-paired-causal-comparison-v1",
        "development_only": True,
        "release_authority": False,
        "variable": variable_name,
        "control_value": expected_control,
        "treatment_value": expected_treatment,
        "common_base_sha": cr.get("pr_base_sha"),
        "framing": {"valid": not framing, "failures": framing},
        "replay_isolation": {"valid": not replay, "failures": replay},
        "round0_input_counterfactual": inputs,
        "causal_valid": causal,
        "round0": {"control": c0, "treatment": t0, "delta": d0},
        "round1": {"control": c1, "treatment": t1, "delta": d1},
        "predeclared_decision": {
            "primary_endpoint": "round1-test-v1",
            "requires_total_matched_improvement": True,
            "requires_each_keyword_nonloss": True,
            "requires_all_keywords_noncollapsed": True,
            "requires_false_accepts_nonincrease": True,
            "requires_far_nonincrease": True,
            "no_weight_or_policy_sweep_on_negative": True,
        },
        "result_class": result_class,
        "next_action": next_action,
        "limitations": [
            "Development-only evidence; no protected qualification or release authority.",
            "Round1 may contain intended downstream curriculum mediation from the single round0 objective-policy change.",
            "No tolerance is invented: precision must be non-worsening and each keyword recall non-losing.",
        ],
    }


def write_fixture(
    root: pathlib.Path,
    treatment: bool,
    positive: bool,
    *,
    variable_name: str = VARIABLE_SEQUENCE_MARGIN,
    control_value: str = DEFAULT_CONTROL,
    treatment_value: str = DEFAULT_TREATMENT,
) -> None:
    if variable_name not in OBJECTIVE_VARIABLES:
        raise ValueError("unsupported fixture objective variable")
    policy = treatment_value if treatment else control_value
    positive_policy = (
        policy
        if variable_name == VARIABLE_SEQUENCE_MARGIN
        else DEFAULT_CONTROL
    )
    primary_policy = (
        policy
        if variable_name == VARIABLE_CTC_PRIMARY
        else DEFAULT_CTC_CONTROL
    )
    config = {
        "train": {
            "ctc_vad_align": True,
            "ctc_primary_policy": primary_policy,
            "sequence_margin_negative_policy": NEGATIVE,
            "sequence_margin_positive_policy": positive_policy,
        },
        "domain_iteration": {"base_failure_replay_enabled": False},
        "development_experiment": {"arm": "t" if treatment else "c"},
    }
    receipt = {
        "source_policy": "exact-pr-head",
        "development_only": True,
        "protected_evidence_used": False,
        "pr_base_sha": "a" * 40,
        "pr_head_sha": ("2" if treatment else "1") * 40,
        "effective_config_sha256": "e" * 64,
        "config_overrides": {
            "train.ctc_vad_align": True,
            "train.sequence_margin_negative_policy": NEGATIVE,
            "domain_iteration.base_failure_replay_enabled": False,
            variable_name: policy,
        },
    }

    def metrics(round_index: int) -> dict:
        base_matched = 10 if round_index == 0 else 12
        matched = base_matched + (2 if treatment and positive else 0)
        base_fa = 20
        fa = base_fa - (2 if treatment and positive else 0)
        left = matched // 2
        right = matched - left
        per = {
            "1": {"expected": 32, "matched": left, "false_rejects": 32-left, "false_accepts": fa//2},
            "2": {"expected": 32, "matched": right, "false_rejects": 32-right, "false_accepts": fa-fa//2},
        }
        return {
            "expected": 64,
            "matched": matched,
            "false_rejects": 64-matched,
            "false_accepts": fa,
            "far_per_hour": float(fa) * 10.0,
            "frr": (64-matched)/64.0,
            "per_keyword": per,
        }

    records = []
    readback = []
    for round_index in (0, 1):
        name = f"r{round_index:02d}-logmel-00"
        record = {
            "round": round_index,
            "frontend": "logmel",
            "candidate": 0,
            "checkpoint": str(root/"build/product-development-experiment/candidates"/name/"model.pt"),
            "model": str(root/"build/product-development-experiment/candidates"/name/"model.kwm"),
            "model_sha256": ("4" if treatment else "3") * 64,
            "score": 10.0-round_index,
            "test": metrics(round_index),
            "calibration": metrics(round_index),
            "training_acoustic_seed_policy": "deterministic-v1",
            "training_acoustic_seed_offset": 0,
            "training_epochs": 12 if round_index == 0 else 6,
            "training_learning_rate": 0.001 if round_index == 0 else 0.00085,
            "training_seed": 1337+round_index,
            "hard_negative_replay_examples": 640,
            "base_failure_replay_examples": 0,
            "base_failure_replay_source_rounds": [],
            "warm_started": round_index == 1,
            "warm_start_strategy": "full" if round_index == 1 else "none",
            "wake_balance": {
                "policy": "fixture",
                "wake_keyword_weights": {"1": 2.0, "2": 2.0},
                "manifests": [{"path": str(root/"train.tsv"), "sha256": ("t" if treatment else "c")*64, "rows": 4}],
            },
        }
        records.append(record)
        readback.append({
            "candidate": name,
            "float_state_sha256": ("t" if treatment else "c")*64,
            "model_sha256": record["model_sha256"],
            "training_corpus_sha256": ("t" if treatment else "c")*64,
            "ctc_primary_policy": primary_policy,
            "sequence_margin_negative_policy": NEGATIVE,
            "sequence_margin_positive_policy": positive_policy,
        })
        prov = root/"build/product-development-experiment/candidates"/name/"model.kwm.provenance.json"
        prov.parent.mkdir(parents=True, exist_ok=True)
        prov.write_text(json.dumps({"training":{"corpus_identity":{"recordings":[{
            "recording":"manifest-0:1","manifest":"train.tsv","path":str(root/"relocated.wav"),
            "file_sha256":"1"*64,"pcm_sha256":"2"*64,"frames":16000
        }]}}}), encoding="utf-8")

    manifest = {"records": records, "candidate_selection":{"nondegeneracy":{"required_keyword_ids":["1","2"]}}}
    for rel, value in {
        ARTIFACT_FILES["config"]: config,
        ARTIFACT_FILES["receipt"]: receipt,
        ARTIFACT_FILES["manifest"]: manifest,
        ARTIFACT_FILES["readback"]: {"candidates": readback},
    }.items():
        path = root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="objective-pair-") as td:
        root = pathlib.Path(td)
        control, treatment = root/"control", root/"treatment"
        write_fixture(control, False, True)
        write_fixture(treatment, True, True)
        report = compare_pair(
            control,
            treatment,
            variable_name=VARIABLE_SEQUENCE_MARGIN,
            expected_control=DEFAULT_CONTROL,
            expected_treatment=DEFAULT_TREATMENT,
        )
        assert report["causal_valid"] is True
        assert report["round0_input_counterfactual"]["valid"] is True
        assert report["result_class"] == "positive-under-strict-precision-preserving-recall-gain"
        write_fixture(treatment, True, False)
        report = compare_pair(
            control,
            treatment,
            variable_name=VARIABLE_SEQUENCE_MARGIN,
            expected_control=DEFAULT_CONTROL,
            expected_treatment=DEFAULT_TREATMENT,
        )
        assert report["causal_valid"] is True
        assert report["result_class"] == "negative"
        assert report["next_action"] == "close-no-sweep"

        alternate = "ctc-keyword-competition-v1"
        pair_receipt = root / "pair-receipt.json"
        pair_receipt.write_text(
            json.dumps(
                {
                    "evidence_class": "product-development-same-runner-pair-receipt-v1",
                    "variable": {
                        "name": VARIABLE_SEQUENCE_MARGIN,
                        "control": DEFAULT_CONTROL,
                        "treatment": alternate,
                    },
                }
            ),
            encoding="utf-8",
        )
        variable_name, expected_control, expected_treatment = (
            expected_values_from_pair_receipt(pair_receipt)
        )
        assert variable_name == VARIABLE_SEQUENCE_MARGIN
        assert expected_control == DEFAULT_CONTROL
        assert expected_treatment == alternate
        write_fixture(
            treatment,
            True,
            True,
            treatment_value=alternate,
        )
        report = compare_pair(
            control,
            treatment,
            variable_name=variable_name,
            expected_control=expected_control,
            expected_treatment=expected_treatment,
        )
        assert report["causal_valid"] is True
        assert report["treatment_value"] == alternate
        assert report["result_class"] == "positive-under-strict-precision-preserving-recall-gain"

        pair_receipt.write_text(
            json.dumps(
                {
                    "evidence_class": "product-development-same-runner-pair-receipt-v1",
                    "variable": {
                        "name": VARIABLE_CTC_PRIMARY,
                        "control": DEFAULT_CTC_CONTROL,
                        "treatment": DEFAULT_CTC_TREATMENT,
                    },
                }
            ),
            encoding="utf-8",
        )
        variable_name, expected_control, expected_treatment = (
            expected_values_from_pair_receipt(pair_receipt)
        )
        assert variable_name == VARIABLE_CTC_PRIMARY
        write_fixture(
            control,
            False,
            True,
            variable_name=VARIABLE_CTC_PRIMARY,
            control_value=DEFAULT_CTC_CONTROL,
            treatment_value=DEFAULT_CTC_TREATMENT,
        )
        write_fixture(
            treatment,
            True,
            True,
            variable_name=VARIABLE_CTC_PRIMARY,
            control_value=DEFAULT_CTC_CONTROL,
            treatment_value=DEFAULT_CTC_TREATMENT,
        )
        report = compare_pair(
            control,
            treatment,
            variable_name=variable_name,
            expected_control=expected_control,
            expected_treatment=expected_treatment,
        )
        assert report["causal_valid"] is True
        assert report["variable"] == VARIABLE_CTC_PRIMARY
        assert report["control_value"] == DEFAULT_CTC_CONTROL
        assert report["treatment_value"] == DEFAULT_CTC_TREATMENT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=pathlib.Path)
    parser.add_argument("--treatment", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--pair-receipt", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("product objective paired causal comparison self-test: PASS")
        return 0
    if (
        args.control is None
        or args.treatment is None
        or args.output is None
        or args.pair_receipt is None
    ):
        parser.error(
            "--control, --treatment, --output and --pair-receipt are required "
            "unless --self-test is used"
        )
    try:
        variable_name, expected_control, expected_treatment = (
            expected_values_from_pair_receipt(args.pair_receipt)
        )
        report = compare_pair(
            args.control,
            args.treatment,
            variable_name=variable_name,
            expected_control=expected_control,
            expected_treatment=expected_treatment,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "evidence_class": "product-development-objective-paired-causal-comparison-v1",
            "infrastructure_complete": False,
            "causal_valid": False,
            "release_authority": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 2
    else:
        report["infrastructure_complete"] = True
        code = 0 if report["causal_valid"] else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("infrastructure_complete","causal_valid","result_class","next_action","error") if k in report}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
