#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib


def load_json(path: pathlib.Path) -> dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: pathlib.Path,value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=pathlib.Path)
    parser.add_argument("--root",required=True,type=pathlib.Path)
    parser.add_argument("--output",required=True,type=pathlib.Path)
    args=parser.parse_args()

    config=load_json(args.config)
    records={}
    for identity_path in args.root.resolve().rglob("identity.json"):
        identity=load_json(identity_path)
        result=load_json(identity_path.parent/"classifier.json")
        candidate=str(identity["candidate"])
        seed=int(identity["model_seed"])
        replica=str(identity["replica"])
        key=(candidate,seed,replica)
        if key in records:
            raise ValueError(f"duplicate record: {key}")
        spec=config["candidates"].get(candidate)
        if spec is None:
            raise ValueError(f"unknown candidate: {candidate}")
        if str(result["encoder_architecture"])!=str(spec["encoder_architecture"]):
            raise ValueError(f"{key}: architecture drift")
        if int(result["hidden_dim"])!=int(spec["hidden_dim"]):
            raise ValueError(f"{key}: hidden drift")
        if int(result["model_seed"])!=seed:
            raise ValueError(f"{key}: model seed drift")
        if int(result["sampler_seed"])!=int(config["fixed"]["sampler_seed"]):
            raise ValueError(f"{key}: sampler seed drift")
        if result["optimizer_kernel_contract"]!=config["required_kernel_contract"]:
            raise ValueError(f"{key}: optimizer kernel contract drift")
        records[key]={
            "initial_model_state_sha256":result["initial_model_state_sha256"],
            "final_model_state_sha256":result["model_state_sha256"],
            "history":result["history"],
            "calibration":result["calibration"],
            "test":result["test"],
            "research_cpu_contract":result["research_cpu_contract"],
            "cpu_runtime":result["training_environment"]["cpu_runtime"],
            "torch_runtime":result["training_environment"]["torch_runtime"],
        }

    expected={
        (candidate,int(seed),replica)
        for candidate in config["candidates"]
        for seed in config["fixed"]["model_seeds"]
        for replica in config["fixed"]["replicas"]
    }
    if set(records)!=expected:
        raise ValueError(f"record mismatch: have={sorted(records)} expected={sorted(expected)}")

    groups={}
    passed=True
    for candidate in config["candidates"]:
        for raw_seed in config["fixed"]["model_seeds"]:
            seed=int(raw_seed)
            replicas=[records[(candidate,seed,replica)] for replica in config["fixed"]["replicas"]]
            checks={
                "initial_state_bit_exact":len({row["initial_model_state_sha256"] for row in replicas})==1,
                "final_state_bit_exact":len({row["final_model_state_sha256"] for row in replicas})==1,
                "history_bit_exact":all(row["history"]==replicas[0]["history"] for row in replicas[1:]),
                "calibration_bit_exact":all(row["calibration"]==replicas[0]["calibration"] for row in replicas[1:]),
                "test_bit_exact":all(row["test"]==replicas[0]["test"] for row in replicas[1:]),
                "cpu_contract_equal":all(row["research_cpu_contract"]==replicas[0]["research_cpu_contract"] for row in replicas[1:]),
            }
            group_pass=all(checks.values())
            passed=passed and group_pass
            groups[f"{candidate}:{seed}"]={
                "pass":group_pass,
                "checks":checks,
                "initial_model_state_sha256":replicas[0]["initial_model_state_sha256"],
                "final_model_state_sha256":replicas[0]["final_model_state_sha256"],
                "replica_cpu_runtime":[row["cpu_runtime"] for row in replicas],
                "replica_torch_runtime":[row["torch_runtime"] for row in replicas],
            }

    result={
        "schema_version":1,
        "evidence_class":"kws-v2-efficient-encoder-cross-runner-repro-summary-v1",
        "evidence_scope":"research-only",
        "diagnostic_only":True,
        "promotion_allowed":False,
        "runtime_format_change_allowed":False,
        "protected_evidence_used":False,
        "shipping_metric":False,
        "reproducibility_pass":passed,
        "groups":groups,
        "next_experiment":config["decision_rules"]["pass_next" if passed else "fail_next"],
    }
    write_json(args.output.resolve(),result)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
