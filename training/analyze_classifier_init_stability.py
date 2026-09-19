#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def load_json(path: pathlib.Path) -> dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: pathlib.Path,value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def select(result: dict, config: dict) -> dict:
    cal={float(row["threshold"]):row for row in result["calibration"]["operating_curve"]}
    test={float(row["threshold"]):row for row in result["test"]["operating_curve"]}
    out={}
    for spec in config["operating_points"]["constraints"]:
        limit=float(spec["max_negative_fp_rate"])
        eligible=[row for row in cal.values() if float(row["negative_false_positive_rate"])<=limit]
        if not eligible:
            out[str(spec["name"])]=None
            continue
        row=max(eligible,key=lambda x:(float(x["wake_exact_recall"]),-float(x["negative_false_positive_rate"]),-float(x["threshold"])))
        threshold=float(row["threshold"])
        out[str(spec["name"])]={
            "threshold":threshold,
            "test":{
                "wake_recall":float(test[threshold]["wake_exact_recall"]),
                "negative_fp_rate":float(test[threshold]["negative_false_positive_rate"]),
            },
        }
    return out


def delta(target: dict|None, reference: dict|None) -> dict|None:
    if target is None or reference is None:
        return None
    return {
        "wake_recall":float(target["test"]["wake_recall"])-float(reference["test"]["wake_recall"]),
        "negative_fp_rate":float(target["test"]["negative_fp_rate"])-float(reference["test"]["negative_fp_rate"]),
    }


def trial(config: dict, seed: int, root: pathlib.Path) -> dict:
    if seed not in [int(v) for v in config["fixed"]["model_seeds"]]:
        raise ValueError("unregistered model seed")
    rows={}
    envs=[]
    for name,spec in config["candidates"].items():
        result=load_json(root/name/"classifier.json")
        if int(result["model_seed"])!=seed:
            raise ValueError(f"{name}: model seed drift")
        if int(result["sampler_seed"])!=int(config["fixed"]["sampler_seed"]):
            raise ValueError(f"{name}: sampler seed drift")
        for key in ("frontend","init_mode"):
            if str(result[key])!=str(spec[key]):
                raise ValueError(f"{name}: {key} drift")
        if int(result["hidden_dim"])!=int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden drift")
        envs.append(result["training_environment"])
        history=result["history"]
        rows[name]={
            "frontend":result["frontend"],
            "hidden_dim":int(result["hidden_dim"]),
            "init_mode":result["init_mode"],
            "initial_model_state_sha256":result["initial_model_state_sha256"],
            "final_model_state_sha256":result["model_state_sha256"],
            "final_train_loss":float(history[-1]["loss"]),
            "min_train_loss":min(float(row["loss"]) for row in history),
            "operating_points":select(result,config),
        }
    if any(env!=envs[0] for env in envs[1:]):
        raise ValueError("paired init candidates did not share training environment")
    primary=str(config["decision_rules"]["primary_operating_point"])
    reference=rows["b0-logmel64-default"]["operating_points"].get(primary)
    default=rows["fc1-pcen128-default"]["operating_points"].get(primary)
    stable=rows["fc1-pcen128-stable-init"]["operating_points"].get(primary)
    return {
        "schema_version":1,
        "evidence_class":"kws-v2-classifier-init-stability-trial-v1",
        "evidence_scope":"research-only",
        "diagnostic_only":True,
        "promotion_allowed":False,
        "protected_evidence_used":False,
        "shipping_metric":False,
        "model_seed":seed,
        "sampler_seed":int(config["fixed"]["sampler_seed"]),
        "paired_same_runner":True,
        "candidates":rows,
        "default_target_vs_reference_primary":delta(default,reference),
        "stable_target_vs_reference_primary":delta(stable,reference),
    }


def summarize(values: list[dict|None]) -> dict:
    usable=[row for row in values if row is not None]
    recall=[float(row["wake_recall"]) for row in usable]
    fp=[float(row["negative_fp_rate"]) for row in usable]
    return {
        "usable":len(usable),
        "wake_recall_delta_mean":statistics.fmean(recall) if recall else None,
        "wake_recall_delta_stddev":statistics.pstdev(recall) if len(recall)>1 else 0.0 if recall else None,
        "negative_fp_delta_mean":statistics.fmean(fp) if fp else None,
        "negative_fp_delta_stddev":statistics.pstdev(fp) if len(fp)>1 else 0.0 if fp else None,
        "positive_wake_recall_delta_seeds":sum(value>0.0 for value in recall),
    }


def aggregate(config: dict, root: pathlib.Path) -> dict:
    rows={}
    for path in root.rglob("trial-evidence.json"):
        row=load_json(path)
        seed=int(row["model_seed"])
        if seed in rows:
            raise ValueError(f"duplicate seed {seed}")
        rows[seed]=row
    expected=[int(v) for v in config["fixed"]["model_seeds"]]
    if set(rows)!=set(expected):
        raise ValueError(f"seed mismatch: {sorted(rows)} vs {expected}")
    ordered=[rows[seed] for seed in expected]
    default=summarize([row["default_target_vs_reference_primary"] for row in ordered])
    stable=summarize([row["stable_target_vs_reference_primary"] for row in ordered])
    source=config["source_variance_evidence"]
    rules=config["decision_rules"]
    tolerance=float(rules["source_baseline_tolerance"])
    baseline_reproduced=(
        abs(float(default["wake_recall_delta_mean"])-float(source["model_init_primary_recall_delta_mean"]))<=tolerance
        and abs(float(default["wake_recall_delta_stddev"])-float(source["model_init_primary_recall_delta_stddev"]))<=tolerance
        and abs(float(default["negative_fp_delta_mean"])-float(source["model_init_primary_fp_delta_mean"]))<=tolerance
    )
    ratio=(
        None if default["wake_recall_delta_stddev"] in (None,0.0)
        else float(stable["wake_recall_delta_stddev"])/float(default["wake_recall_delta_stddev"])
    )
    stability_pass=(
        baseline_reproduced
        and ratio is not None
        and ratio<=float(rules["max_stddev_ratio_vs_default"])
        and float(stable["wake_recall_delta_stddev"])<=float(rules["max_primary_wake_recall_delta_stddev"])
        and float(stable["wake_recall_delta_mean"])>=float(rules["min_primary_wake_recall_delta_mean"])
        and float(stable["negative_fp_delta_mean"])<=float(rules["max_primary_negative_fp_delta_mean"])
        and int(stable["positive_wake_recall_delta_seeds"])>=int(rules["min_positive_primary_delta_seeds"])
    )
    if not baseline_reproduced:
        next_experiment=str(rules["baseline_drift_next"])
    elif stability_pass:
        next_experiment=str(rules["pass_next"])
    else:
        next_experiment=str(rules["fail_next"])
    return {
        "schema_version":1,
        "evidence_class":"kws-v2-classifier-init-stability-summary-v1",
        "evidence_scope":"research-only",
        "diagnostic_only":True,
        "promotion_allowed":False,
        "protected_evidence_used":False,
        "shipping_metric":False,
        "source_variance_evidence":source,
        "baseline_reproduced":baseline_reproduced,
        "default_init":default,
        "stable_init":stable,
        "stable_to_default_stddev_ratio":ratio,
        "stability_pass":stability_pass,
        "seed_results":{str(seed):rows[seed] for seed in expected},
        "next_experiment":next_experiment,
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("trial")
    p.add_argument("--config",required=True,type=pathlib.Path)
    p.add_argument("--model-seed",required=True,type=int)
    p.add_argument("--root",required=True,type=pathlib.Path)
    p.add_argument("--output",required=True,type=pathlib.Path)
    a=sub.add_parser("aggregate")
    a.add_argument("--config",required=True,type=pathlib.Path)
    a.add_argument("--root",required=True,type=pathlib.Path)
    a.add_argument("--output",required=True,type=pathlib.Path)
    args=parser.parse_args()
    config=load_json(args.config)
    result=trial(config,int(args.model_seed),args.root.resolve()) if args.mode=="trial" else aggregate(config,args.root.resolve())
    write_json(args.output.resolve(),result)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
