#!/usr/bin/env python3
"""Compare paired product-development experiment artifacts without using qualification evidence."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
import tempfile

from development_signal import record_rank

VARIABLE = "domain_iteration.base_failure_replay_enabled"
ARTIFACT_FILES = {
    "config": pathlib.Path(".generated/xiaowo.product-experiment.json"),
    "receipt": pathlib.Path("build/product-development-experiment-receipt.json"),
    "manifest": pathlib.Path("build/product-development-experiment/domain-loop-manifest.json"),
    "readback": pathlib.Path("build/product-development-experiment/training-readback.json"),
}
METRIC_PATH_FIELDS = {"false_positives_path", "false_rejects_path"}


def load_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing paired experiment evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_artifact(root: pathlib.Path) -> dict:
    root = root.resolve()
    return {name: load_json(root / rel) for name, rel in ARTIFACT_FILES.items()}


def delete_dotted(value: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cursor = value
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            return
        cursor = child
    cursor.pop(parts[-1], None)


def normalized_effective_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("development_experiment", None)
    delete_dotted(value, VARIABLE)
    return value


def normalized_overrides(receipt: dict) -> dict:
    value = receipt.get("config_overrides")
    if not isinstance(value, dict):
        raise ValueError("experiment receipt config_overrides must be an object")
    result = copy.deepcopy(value)
    result.pop(VARIABLE, None)
    return result


def semantic_metrics(value):
    if isinstance(value, dict):
        return {
            key: semantic_metrics(item)
            for key, item in value.items()
            if key not in METRIC_PATH_FIELDS and not key.endswith("_path")
        }
    if isinstance(value, list):
        return [semantic_metrics(item) for item in value]
    return value


def record_key(record: dict) -> tuple[int, str, int]:
    return (
        int(record["round"]),
        str(record["frontend"]),
        int(record.get("candidate", 0)),
    )


def records_by_round(manifest: dict, round_index: int) -> list[dict]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("domain-loop manifest records must be a list")
    rows = [row for row in records if isinstance(row, dict) and int(row.get("round", -1)) == round_index]
    if not rows:
        raise ValueError(f"paired evidence is missing round {round_index}")
    return rows


def required_keyword_ids(manifest: dict, records: list[dict]) -> tuple[str, ...]:
    selection = manifest.get("candidate_selection")
    if isinstance(selection, dict):
        nondegeneracy = selection.get("nondegeneracy")
        if isinstance(nondegeneracy, dict):
            raw = nondegeneracy.get("required_keyword_ids")
            if isinstance(raw, list) and raw and all(isinstance(v, str) and v for v in raw):
                return tuple(raw)
    metrics = records[0].get("test")
    per_keyword = metrics.get("per_keyword") if isinstance(metrics, dict) else None
    if not isinstance(per_keyword, dict) or not per_keyword:
        raise ValueError("cannot resolve required keyword ids from paired evidence")
    return tuple(sorted((str(key) for key in per_keyword), key=lambda value: int(value)))


def round_best(manifest: dict, round_index: int) -> dict:
    rows = records_by_round(manifest, round_index)
    keyword_ids = required_keyword_ids(manifest, rows)
    return min(rows, key=lambda row: record_rank(row, keyword_ids))


def readback_map(readback: dict) -> dict[str, dict]:
    rows = readback.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("training readback candidates must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("training readback candidate must be an object")
        name = row.get("candidate")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("training readback candidate identity is invalid or duplicated")
        result[name] = row
    return result


def candidate_name(record: dict) -> str:
    value = record.get("checkpoint") or record.get("model")
    if not isinstance(value, str) or not value:
        raise ValueError("record candidate path is missing")
    return pathlib.PurePath(value).parent.name


def exact_fields(left: dict, right: dict, fields: tuple[str, ...], label: str, failures: list[str]) -> None:
    for field in fields:
        if left.get(field) != right.get(field):
            failures.append(f"{label}.{field} differs")


def metric_summary(record: dict) -> dict:
    metrics = record.get("test")
    if not isinstance(metrics, dict):
        raise ValueError("round record test metrics are missing")
    per_keyword = metrics.get("per_keyword")
    if not isinstance(per_keyword, dict):
        raise ValueError("round record per_keyword metrics are missing")
    keywords = {}
    for key, row in sorted(per_keyword.items(), key=lambda item: int(item[0])):
        if not isinstance(row, dict):
            raise ValueError(f"keyword {key} metrics must be an object")
        expected = int(row.get("expected", 0))
        matched = int(row.get("matched", 0))
        keywords[str(key)] = {
            "expected": expected,
            "matched": matched,
            "false_rejects": int(row.get("false_rejects", 0)),
            "false_accepts": int(row.get("false_accepts", 0)),
            "recall": (matched / expected) if expected else None,
        }
    return {
        "matched": int(metrics.get("matched", 0)),
        "expected": int(metrics.get("expected", 0)),
        "false_accepts": int(metrics.get("false_accepts", 0)),
        "far_per_hour": float(metrics.get("far_per_hour", math.nan)),
        "frr": float(metrics.get("frr", math.nan)),
        "per_keyword": keywords,
    }


def load_replay_evidence(root: pathlib.Path, round_index: int) -> dict:
    path = (
        root.resolve()
        / "build/product-development-experiment/base-failure-replay"
        / f"round-{round_index:02d}"
        / "development-failure-replay.json"
    )
    return load_json(path)


def compare_pair(control_root: pathlib.Path, treatment_root: pathlib.Path) -> dict:
    control = load_artifact(control_root)
    treatment = load_artifact(treatment_root)
    framing_failures: list[str] = []

    cr, tr = control["receipt"], treatment["receipt"]
    for label, receipt in (("control", cr), ("treatment", tr)):
        if receipt.get("source_policy") != "exact-pr-head":
            framing_failures.append(f"{label} source policy is not exact-pr-head")
        if receipt.get("development_only") is not True:
            framing_failures.append(f"{label} is not development-only")
        if receipt.get("protected_evidence_used") is not False:
            framing_failures.append(f"{label} used protected evidence")

    if cr.get("pr_base_sha") != tr.get("pr_base_sha"):
        framing_failures.append("PR base SHA differs")
    if cr.get("effective_config_sha256") != tr.get("effective_config_sha256"):
        framing_failures.append("governed effective product base differs")
    if normalized_overrides(cr) != normalized_overrides(tr):
        framing_failures.append("experiment overrides differ outside replay variable")
    if cr.get("config_overrides", {}).get(VARIABLE) is not False:
        framing_failures.append("control replay variable is not false")
    if tr.get("config_overrides", {}).get(VARIABLE) is not True:
        framing_failures.append("treatment replay variable is not true")
    if normalized_effective_config(control["config"]) != normalized_effective_config(treatment["config"]):
        framing_failures.append("materialized configs differ outside experiment metadata/replay variable")

    cm, tm = control["manifest"], treatment["manifest"]
    c0 = {record_key(row): row for row in records_by_round(cm, 0)}
    t0 = {record_key(row): row for row in records_by_round(tm, 0)}
    round0_failures: list[str] = []
    if set(c0) != set(t0):
        round0_failures.append("round0 candidate set differs")

    crb = readback_map(control["readback"])
    trb = readback_map(treatment["readback"])
    common_keys = sorted(set(c0) & set(t0))
    for key in common_keys:
        left, right = c0[key], t0[key]
        prefix = f"round0[{key[1]}:{key[2]}]"
        if int(left.get("base_failure_replay_examples", -1)) != 0:
            round0_failures.append(f"{prefix} control replay examples are nonzero")
        if int(right.get("base_failure_replay_examples", -1)) != 0:
            round0_failures.append(f"{prefix} treatment replay examples are nonzero")
        if left.get("base_failure_replay_source_rounds") not in ([], None):
            round0_failures.append(f"{prefix} control replay has source rounds")
        if right.get("base_failure_replay_source_rounds") not in ([], None):
            round0_failures.append(f"{prefix} treatment replay has source rounds")
        exact_fields(
            left,
            right,
            (
                "model_sha256",
                "training_acoustic_seed_policy",
                "training_acoustic_seed_offset",
                "training_epochs",
                "training_learning_rate",
                "training_seed",
                "hard_negative_replay_examples",
                "hard_negative_replay_manifest_sha256",
                "warm_started",
                "warm_start_strategy",
                "wake_balance",
            ),
            prefix,
            round0_failures,
        )
        for split in ("calibration", "calibration_domains", "test", "test_domains"):
            if semantic_metrics(left.get(split)) != semantic_metrics(right.get(split)):
                round0_failures.append(f"{prefix}.{split} C-runtime metrics differ")

        lname, rname = candidate_name(left), candidate_name(right)
        lread, rread = crb.get(lname), trb.get(rname)
        if not isinstance(lread, dict) or not isinstance(rread, dict):
            round0_failures.append(f"{prefix} training readback candidate is missing")
        else:
            exact_fields(
                lread,
                rread,
                ("float_state_sha256", "model_sha256", "training_corpus_sha256"),
                f"{prefix}.readback",
                round0_failures,
            )

    c1 = round_best(cm, 1)
    t1 = round_best(tm, 1)
    provenance_failures: list[str] = []
    if int(c1.get("base_failure_replay_examples", -1)) != 0:
        provenance_failures.append("control round1 replay examples are nonzero")
    if c1.get("base_failure_replay_source_rounds") not in ([], None):
        provenance_failures.append("control round1 replay source rounds are nonempty")
    treatment_examples = int(t1.get("base_failure_replay_examples", 0))
    if treatment_examples <= 0:
        provenance_failures.append("treatment round1 replay examples are not positive")
    if t1.get("base_failure_replay_source_rounds") != [0]:
        provenance_failures.append("treatment round1 replay source_rounds are not exactly [0]")

    try:
        replay = load_replay_evidence(treatment_root, 1)
    except ValueError as exc:
        provenance_failures.append(str(exc))
        replay = None
    if isinstance(replay, dict):
        if replay.get("enabled") is not True:
            provenance_failures.append("treatment replay evidence is not enabled")
        if replay.get("formal_qualification_used") is not False:
            provenance_failures.append("treatment replay used formal qualification")
        if replay.get("development_source_wav_bytes_copied") is not False:
            provenance_failures.append("treatment replay copied development WAV bytes")
        if replay.get("source_splits") != ["calibration", "test"]:
            provenance_failures.append("treatment replay source splits are not development calibration/test")
        selected = replay.get("selected")
        source_rounds = sorted(
            {
                int(value)
                for item in selected if isinstance(item, dict)
                for value in item.get("source_rounds", [])
            }
        ) if isinstance(selected, list) else []
        if source_rounds != [0]:
            provenance_failures.append("treatment replay evidence source rounds are not exactly [0]")
        if int(replay.get("examples", -1)) != treatment_examples:
            provenance_failures.append("treatment replay evidence example count differs from round record")
        if replay.get("manifest_sha256") != t1.get("base_failure_replay_manifest_sha256"):
            provenance_failures.append("treatment replay manifest SHA differs from round record")

    control_metrics = metric_summary(c1)
    treatment_metrics = metric_summary(t1)
    keyword_ids = sorted(
        set(control_metrics["per_keyword"]) | set(treatment_metrics["per_keyword"]),
        key=int,
    )
    per_keyword_delta = {}
    all_keywords_noncollapsed = True
    no_keyword_match_loss = True
    for key in keyword_ids:
        before = control_metrics["per_keyword"].get(key)
        after = treatment_metrics["per_keyword"].get(key)
        if before is None or after is None or before["expected"] != after["expected"]:
            per_keyword_delta[key] = {"comparable": False}
            all_keywords_noncollapsed = False
            no_keyword_match_loss = False
            continue
        all_keywords_noncollapsed &= after["expected"] > 0 and after["matched"] > 0
        no_keyword_match_loss &= after["matched"] >= before["matched"]
        per_keyword_delta[key] = {
            "comparable": True,
            "expected": before["expected"],
            "control_matched": before["matched"],
            "treatment_matched": after["matched"],
            "matched_delta": after["matched"] - before["matched"],
            "control_recall": before["recall"],
            "treatment_recall": after["recall"],
            "recall_delta": after["recall"] - before["recall"],
        }

    false_accepts_reduced = treatment_metrics["false_accepts"] < control_metrics["false_accepts"]
    far_reduced = treatment_metrics["far_per_hour"] < control_metrics["far_per_hour"]
    causal_valid = not framing_failures and not round0_failures and not provenance_failures
    if not causal_valid:
        result_class = "causal-invalid"
        next_action = "do-not-interpret"
    elif not false_accepts_reduced or not far_reduced or not all_keywords_noncollapsed:
        result_class = "negative"
        next_action = "close-no-sweep"
    elif no_keyword_match_loss:
        result_class = "positive-under-strict-nonloss"
        next_action = "candidate-follow-up-without-changing-this-pair"
    else:
        result_class = "directionally-positive-recall-materiality-review"
        next_action = "apply-predeclared-recall-materiality-rule-before-continuing"

    return {
        "schema_version": 1,
        "evidence_class": "product-development-paired-causal-comparison-v1",
        "development_only": True,
        "release_authority": False,
        "variable": VARIABLE,
        "common_base_sha": cr.get("pr_base_sha"),
        "control_head_sha": cr.get("pr_head_sha"),
        "treatment_head_sha": tr.get("pr_head_sha"),
        "framing": {"valid": not framing_failures, "failures": framing_failures},
        "round0_counterfactual": {
            "bit_identical": not round0_failures,
            "failures": round0_failures,
            "candidate_count": len(common_keys),
        },
        "round1_replay_provenance": {
            "valid": not provenance_failures,
            "failures": provenance_failures,
            "treatment_examples": treatment_examples,
            "treatment_source_rounds": t1.get("base_failure_replay_source_rounds"),
        },
        "causal_valid": causal_valid,
        "round1": {
            "control": control_metrics,
            "treatment": treatment_metrics,
            "delta": {
                "matched": treatment_metrics["matched"] - control_metrics["matched"],
                "false_accepts": treatment_metrics["false_accepts"] - control_metrics["false_accepts"],
                "far_per_hour": treatment_metrics["far_per_hour"] - control_metrics["far_per_hour"],
                "frr": treatment_metrics["frr"] - control_metrics["frr"],
                "per_keyword": per_keyword_delta,
            },
        },
        "predeclared_signals": {
            "false_accepts_reduced": false_accepts_reduced,
            "far_reduced": far_reduced,
            "all_keywords_noncollapsed": all_keywords_noncollapsed,
            "no_keyword_match_loss": no_keyword_match_loss,
        },
        "result_class": result_class,
        "next_action": next_action,
        "limitations": [
            "Development-only evidence; no protected qualification or release authority.",
            "Recall materiality is reported as exact per-keyword deltas unless strict nonloss holds; no undeclared tolerance is invented.",
        ],
    }


def write_fixture(root: pathlib.Path, *, treatment: bool, round0_model: str = "a" * 64) -> None:
    config = {
        "train": {
            "ctc_vad_align": True,
            "sequence_margin_negative_policy": "runtime-executable-v1",
        },
        "domain_iteration": {"base_failure_replay_enabled": treatment},
        "development_experiment": {
            "experiment_id": "treatment" if treatment else "control",
            "pr_head_sha": ("2" if treatment else "1") * 40,
        },
    }
    receipt = {
        "source_policy": "exact-pr-head",
        "development_only": True,
        "protected_evidence_used": False,
        "pr_base_sha": "0" * 40,
        "pr_head_sha": ("2" if treatment else "1") * 40,
        "effective_config_sha256": "e" * 64,
        "config_overrides": {
            "train.ctc_vad_align": True,
            "train.sequence_margin_negative_policy": "runtime-executable-v1",
            VARIABLE: treatment,
        },
    }

    def metrics(matched: int, false_accepts: int) -> dict:
        expected = 4
        return {
            "expected": expected,
            "matched": matched,
            "false_rejects": expected - matched,
            "false_accepts": false_accepts,
            "frr": (expected - matched) / expected,
            "far_per_hour": float(false_accepts * 10),
            "references_sha256": "r" * 64,
            "detections_sha256": ("d" if not treatment else "t") * 64,
            "per_keyword": {
                "1": {"expected": 2, "matched": 2, "false_rejects": 0, "false_accepts": false_accepts, "frr": 0.0},
                "2": {"expected": 2, "matched": 2, "false_rejects": 0, "false_accepts": 0, "frr": 0.0},
            },
        }

    common0 = {
        "round": 0,
        "frontend": "logmel",
        "candidate": 0,
        "score": 10.0,
        "model": "/tmp/candidates/round-00-logmel-00/model.kwm",
        "checkpoint": "/tmp/candidates/round-00-logmel-00/model.pt",
        "model_sha256": round0_model,
        "calibration": metrics(4, 2),
        "calibration_domains": {"domains": {}},
        "test": metrics(4, 2),
        "test_domains": {"domains": {}},
        "training_acoustic_seed_policy": "train-only-scene-seed-offset-v1",
        "training_acoustic_seed_offset": 0,
        "training_epochs": 12,
        "training_learning_rate": 0.001,
        "training_seed": 1337,
        "hard_negative_replay_examples": 8,
        "hard_negative_replay_manifest_sha256": "h" * 64,
        "base_failure_replay_enabled": treatment,
        "base_failure_replay_examples": 0,
        "base_failure_replay_manifest_sha256": None,
        "base_failure_replay_source_rounds": [],
        "warm_started": False,
        "warm_start_strategy": "cold-start",
        "wake_balance": {"policy": "fixture"},
    }
    # Round-0 runtime output must be exactly equal even though the experiment metadata differs.
    common0["calibration"]["detections_sha256"] = "0" * 64
    common0["test"]["detections_sha256"] = "0" * 64

    replay_examples = 4 if treatment else 0
    round1 = {
        **copy.deepcopy(common0),
        "round": 1,
        "score": 8.0 if treatment else 9.0,
        "model": "/tmp/candidates/round-01-logmel-00/model.kwm",
        "checkpoint": "/tmp/candidates/round-01-logmel-00/model.pt",
        "model_sha256": ("b" if treatment else "c") * 64,
        "test": metrics(4, 1 if treatment else 2),
        "calibration": metrics(4, 1 if treatment else 2),
        "base_failure_replay_examples": replay_examples,
        "base_failure_replay_manifest_sha256": ("f" * 64) if treatment else None,
        "base_failure_replay_source_rounds": [0] if treatment else [],
        "warm_started": True,
        "warm_start_strategy": "full",
        "training_epochs": 6,
    }
    manifest = {
        "records": [common0, round1],
        "candidate_selection": {
            "nondegeneracy": {"required_keyword_ids": ["1", "2"]}
        },
    }
    readback = {
        "candidates": [
            {
                "candidate": "round-00-logmel-00",
                "float_state_sha256": "s" * 64 if round0_model == "a" * 64 else "x" * 64,
                "model_sha256": round0_model,
                "training_corpus_sha256": "q" * 64,
            },
            {
                "candidate": "round-01-logmel-00",
                "float_state_sha256": ("u" if treatment else "v") * 64,
                "model_sha256": round1["model_sha256"],
                "training_corpus_sha256": ("j" if treatment else "k") * 64,
            },
        ]
    }
    payloads = {
        ARTIFACT_FILES["config"]: config,
        ARTIFACT_FILES["receipt"]: receipt,
        ARTIFACT_FILES["manifest"]: manifest,
        ARTIFACT_FILES["readback"]: readback,
    }
    for rel, value in payloads.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    if treatment:
        replay_path = root / "build/product-development-experiment/base-failure-replay/round-01/development-failure-replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "formal_qualification_used": False,
                    "development_source_wav_bytes_copied": False,
                    "source_splits": ["calibration", "test"],
                    "examples": 4,
                    "manifest_sha256": "f" * 64,
                    "selected": [{"source_rounds": [0]}],
                }
            ),
            encoding="utf-8",
        )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="product-pair-") as tmp:
        root = pathlib.Path(tmp)
        control, treatment = root / "control", root / "treatment"
        write_fixture(control, treatment=False)
        write_fixture(treatment, treatment=True)
        report = compare_pair(control, treatment)
        assert report["causal_valid"] is True
        assert report["round0_counterfactual"]["bit_identical"] is True
        assert report["result_class"] == "positive-under-strict-nonloss"
        write_fixture(treatment, treatment=True, round0_model="9" * 64)
        report = compare_pair(control, treatment)
        assert report["causal_valid"] is False
        assert report["round0_counterfactual"]["bit_identical"] is False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=pathlib.Path)
    parser.add_argument("--treatment", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("product development paired causal comparison self-test: PASS")
        return 0
    if args.control is None or args.treatment is None or args.output is None:
        parser.error("--control, --treatment and --output are required unless --self-test is used")
    try:
        report = compare_pair(args.control, args.treatment)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "evidence_class": "product-development-paired-causal-comparison-v1",
            "infrastructure_complete": False,
            "causal_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "release_authority": False,
        }
        code = 2
    else:
        report["infrastructure_complete"] = True
        code = 0 if report["causal_valid"] else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: report.get(key)
        for key in ("infrastructure_complete", "causal_valid", "result_class", "next_action", "error")
        if key in report
    }, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
