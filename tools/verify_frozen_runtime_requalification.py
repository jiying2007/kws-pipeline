#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile

POLICY = "frozen-promoted-model-runtime-paired-replay-v1"
EXPECTED_RECORDINGS = 640
EXPECTED_WAKES = 256
EXPECTED_PER_KEYWORD = {"1": 128, "2": 128}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def strict_summary(value: dict, label: str) -> dict:
    required = {
        "expected": EXPECTED_WAKES,
        "matched": EXPECTED_WAKES,
        "false_rejects": 0,
        "false_accepts": 0,
    }
    for key, expected in required.items():
        actual = int(value.get(key, -1))
        if actual != expected:
            raise ValueError(f"{label}.{key}={actual} expected {expected}")
    if float(value.get("frr", -1.0)) != 0.0 or float(value.get("far_per_hour", -1.0)) != 0.0:
        raise ValueError(f"{label} is not zero-FRR/zero-FAR")
    per_keyword = value.get("per_keyword")
    if not isinstance(per_keyword, dict):
        raise ValueError(f"{label}.per_keyword is missing")
    compact = {}
    for keyword_id, expected in EXPECTED_PER_KEYWORD.items():
        row = per_keyword.get(keyword_id)
        if not isinstance(row, dict):
            raise ValueError(f"{label}.per_keyword[{keyword_id}] is missing")
        if int(row.get("expected", -1)) != expected or int(row.get("matched", -1)) != expected:
            raise ValueError(f"{label}.per_keyword[{keyword_id}] is not {expected}/{expected}")
        if int(row.get("false_rejects", -1)) != 0 or int(row.get("false_accepts", 0)) != 0:
            raise ValueError(f"{label}.per_keyword[{keyword_id}] contains errors")
        compact[keyword_id] = {"expected": expected, "matched": expected}
    return compact


def reference_stats(path: pathlib.Path) -> tuple[int, int, dict[str, int]]:
    recordings = 0
    wakes = 0
    per_keyword = {key: 0 for key in EXPECTED_PER_KEYWORD}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or not isinstance(row.get("expected"), list):
            raise ValueError(f"references:{line_no}: malformed row")
        recordings += 1
        for event in row["expected"]:
            keyword_id = str(int(event["keyword_id"]))
            if keyword_id not in per_keyword:
                raise ValueError(f"references:{line_no}: unexpected keyword {keyword_id}")
            per_keyword[keyword_id] += 1
            wakes += 1
    return recordings, wakes, per_keyword


def verify(args: argparse.Namespace) -> dict:
    shipping = load(args.shipping_config, "shipping config")
    model_contract = shipping.get("model")
    if not isinstance(model_contract, dict):
        raise ValueError("shipping model contract is missing")
    cohort = load(args.cohort_receipt, "qualification cohort receipt")
    parameter_contract = load(args.parameter_contract, "parameter contract")
    retained = load(args.retained_summary, "retained qualification summary")
    baseline = load(args.baseline_summary, "baseline replay summary")
    candidate = load(args.candidate_summary, "candidate replay summary")

    model_sha = sha256(args.model)
    pack_sha = sha256(args.keyword_pack)
    if model_sha != str(model_contract.get("model_sha256") or ""):
        raise ValueError("frozen model SHA does not match shipping contract")
    if pack_sha != str(model_contract.get("keyword_pack_sha256") or ""):
        raise ValueError("keyword pack SHA does not match shipping contract")

    expected_refs = str(cohort.get("active_references_sha256") or cohort.get("qualification_references_sha256") or "")
    expected_index = str(cohort.get("domain_index_sha256") or "")
    actual_refs = sha256(args.regenerated_references)
    actual_index = sha256(args.regenerated_domain_index)
    if actual_refs != expected_refs:
        raise ValueError(f"regenerated references SHA mismatch: {actual_refs} != {expected_refs}")
    if actual_index != expected_index:
        raise ValueError(f"regenerated domain-index SHA mismatch: {actual_index} != {expected_index}")
    if int(cohort.get("qualification_seed", -1)) != int(model_contract.get("qualification_seed", -2)):
        raise ValueError("cohort/model qualification seed mismatch")
    if int(cohort.get("recordings", -1)) != EXPECTED_RECORDINGS or int(cohort.get("expected_wakes", -1)) != EXPECTED_WAKES:
        raise ValueError("retained cohort support drifted")

    recordings, wakes, ref_per_keyword = reference_stats(args.regenerated_references)
    if recordings != EXPECTED_RECORDINGS or wakes != EXPECTED_WAKES or ref_per_keyword != EXPECTED_PER_KEYWORD:
        raise ValueError(
            f"regenerated cohort support mismatch recordings={recordings} wakes={wakes} per_keyword={ref_per_keyword}"
        )

    retained_kw = strict_summary(retained, "retained")
    baseline_kw = strict_summary(baseline, "baseline")
    candidate_kw = strict_summary(candidate, "candidate")

    algorithm = parameter_contract.get("algorithm_constants", {})
    boundary = algorithm.get("KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES") if isinstance(algorithm, dict) else None
    if not isinstance(boundary, dict) or int(boundary.get("default", -1)) != 12:
        raise ValueError("current runtime does not declare reset12 boundary")
    if boundary.get("invalidates_thresholds") is not True:
        raise ValueError("reset12 boundary is not bound to threshold invalidation")

    if len(args.runtime_source_sha) != 40:
        raise ValueError("runtime source SHA must be canonical 40-hex")
    report = {
        "schema_version": 1,
        "policy": POLICY,
        "evidence_class": "frozen-model-runtime-only-diagnostic",
        "release_authority": False,
        "training_performed": False,
        "fresh_qualification_seed_consumed": False,
        "diagnostic_reuses_exposed_qualification_seed": True,
        "model_release_tag": str(model_contract.get("release_tag")),
        "model_sha256": model_sha,
        "keyword_pack_sha256": pack_sha,
        "qualification_seed": int(cohort["qualification_seed"]),
        "cohort": {
            "recordings": recordings,
            "expected_wakes": wakes,
            "references_sha256": actual_refs,
            "domain_index_sha256": actual_index,
            "per_keyword_expected": ref_per_keyword,
        },
        "runtime": {
            "source_sha": args.runtime_source_sha,
            "decoder_boundary_reset_inactive_frames": 12,
        },
        "retained_original": {"qualified": True, "per_keyword": retained_kw},
        "rebuilt_original_runtime": {"qualified": True, "per_keyword": baseline_kw},
        "current_reset12_runtime": {"qualified": True, "per_keyword": candidate_kw},
        "summary_sha256": {
            "retained": sha256(args.retained_summary),
            "baseline": sha256(args.baseline_summary),
            "candidate": sha256(args.candidate_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="runtime-requal-test-") as td:
        root = pathlib.Path(td)
        model = root / "m.kwm"; model.write_bytes(b"model")
        pack = root / "k.kwk"; pack.write_bytes(b"pack")
        refs = root / "refs.jsonl"
        rows=[]
        for i in range(EXPECTED_RECORDINGS):
            expected=[]
            if i < 128: expected=[{"keyword_id":1}]
            elif i < 256: expected=[{"keyword_id":2}]
            rows.append(json.dumps({"recording":str(i),"expected":expected}))
        refs.write_text("\n".join(rows)+"\n")
        index=root/"index.jsonl"; index.write_text("{}\n")
        summary={
            "expected":256,"matched":256,"false_rejects":0,"false_accepts":0,
            "frr":0.0,"far_per_hour":0.0,
            "per_keyword":{
                "1":{"expected":128,"matched":128,"false_rejects":0,"false_accepts":0},
                "2":{"expected":128,"matched":128,"false_rejects":0,"false_accepts":0},
            },
        }
        for name in ("retained","baseline","candidate"):
            (root/f"{name}.json").write_text(json.dumps(summary))
        shipping={"model":{"release_tag":"model-test","model_sha256":sha256(model),"keyword_pack_sha256":sha256(pack),"qualification_seed":271838}}
        (root/"shipping.json").write_text(json.dumps(shipping))
        cohort={"qualification_seed":271838,"recordings":640,"expected_wakes":256,"active_references_sha256":sha256(refs),"domain_index_sha256":sha256(index)}
        (root/"cohort.json").write_text(json.dumps(cohort))
        param={"algorithm_constants":{"KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES":{"default":12,"invalidates_thresholds":True}}}
        (root/"params.json").write_text(json.dumps(param))
        ns=argparse.Namespace(
            shipping_config=root/"shipping.json", cohort_receipt=root/"cohort.json",
            regenerated_references=refs, regenerated_domain_index=index,
            retained_summary=root/"retained.json", baseline_summary=root/"baseline.json",
            candidate_summary=root/"candidate.json", model=model, keyword_pack=pack,
            parameter_contract=root/"params.json", runtime_source_sha="a"*40,
            output=root/"receipt.json",
        )
        report=verify(ns)
        assert report["release_authority"] is False
        bad=json.loads((root/"candidate.json").read_text()); bad["matched"]=255
        (root/"candidate.json").write_text(json.dumps(bad))
        try:
            verify(ns)
        except ValueError as exc:
            assert "candidate.matched" in str(exc)
        else:
            raise AssertionError("non-strict candidate was accepted")


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--shipping-config", type=pathlib.Path)
    parser.add_argument("--cohort-receipt", type=pathlib.Path)
    parser.add_argument("--regenerated-references", type=pathlib.Path)
    parser.add_argument("--regenerated-domain-index", type=pathlib.Path)
    parser.add_argument("--retained-summary", type=pathlib.Path)
    parser.add_argument("--baseline-summary", type=pathlib.Path)
    parser.add_argument("--candidate-summary", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--keyword-pack", type=pathlib.Path)
    parser.add_argument("--parameter-contract", type=pathlib.Path)
    parser.add_argument("--runtime-source-sha")
    parser.add_argument("--output", type=pathlib.Path)
    args=parser.parse_args()
    if args.self_test:
        self_test(); print("frozen runtime requalification self-test: PASS"); return 0
    required=("shipping_config","cohort_receipt","regenerated_references","regenerated_domain_index","retained_summary","baseline_summary","candidate_summary","model","keyword_pack","parameter_contract","runtime_source_sha","output")
    missing=[name for name in required if getattr(args,name) is None]
    if missing: parser.error("missing required arguments: "+", ".join(missing))
    print(json.dumps(verify(args), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
