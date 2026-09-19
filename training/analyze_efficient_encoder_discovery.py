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


def select_operating_points(result: dict, config: dict) -> dict:
    cal={float(row["threshold"]):row for row in result["calibration"]["operating_curve"]}
    test={float(row["threshold"]):row for row in result["test"]["operating_curve"]}
    out={}
    for spec in config["operating_points"]["constraints"]:
        limit=float(spec["max_negative_fp_rate"])
        eligible=[row for row in cal.values() if float(row["negative_false_positive_rate"])<=limit]
        if not eligible:
            out[str(spec["name"])]=None
            continue
        chosen=max(
            eligible,
            key=lambda row:(
                float(row["wake_exact_recall"]),
                -float(row["negative_false_positive_rate"]),
                -float(row["threshold"]),
            ),
        )
        threshold=float(chosen["threshold"])
        out[str(spec["name"])]={
            "threshold":threshold,
            "calibration":{
                "wake_recall":float(chosen["wake_exact_recall"]),
                "negative_fp_rate":float(chosen["negative_false_positive_rate"]),
            },
            "test":{
                "wake_recall":float(test[threshold]["wake_exact_recall"]),
                "negative_fp_rate":float(test[threshold]["negative_false_positive_rate"]),
            },
        }
    return out


def calibration_delta(target: dict|None,reference: dict|None) -> dict|None:
    if target is None or reference is None:
        return None
    return {
        "wake_recall":float(target["calibration"]["wake_recall"])-float(reference["calibration"]["wake_recall"]),
        "negative_fp_rate":float(target["calibration"]["negative_fp_rate"])-float(reference["calibration"]["negative_fp_rate"]),
    }


def trial(config: dict, seed: int, root: pathlib.Path) -> dict:
    if seed not in [int(v) for v in config["fixed"]["model_seeds"]]:
        raise ValueError("unregistered model seed")
    rows={}
    runtime_identities=[]
    for name,spec in config["candidates"].items():
        result=load_json(root/name/"classifier.json")
        if int(result["model_seed"])!=seed:
            raise ValueError(f"{name}: model seed drift")
        if int(result["sampler_seed"])!=int(config["fixed"]["sampler_seed"]):
            raise ValueError(f"{name}: sampler seed drift")
        if str(result["frontend"])!=str(config["fixed"]["frontend"]):
            raise ValueError(f"{name}: frontend drift")
        if str(result["encoder_architecture"])!=str(spec["encoder_architecture"]):
            raise ValueError(f"{name}: encoder architecture drift")
        if int(result["hidden_dim"])!=int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden dimension drift")
        if int(result["encoder_layers"])!=int(spec["encoder_layers"]):
            raise ValueError(f"{name}: layer count drift")
        for key in ("context_frames","residual_context_adapter"):
            if key in spec and result.get(key)!=spec[key]:
                raise ValueError(f"{name}: {key} drift")
        runtime_identity={
            "research_cpu_contract":result["research_cpu_contract"],
            "optimizer_kernel_contract":result["optimizer_kernel_contract"],
            "cpu_runtime":result["training_environment"]["cpu_runtime"],
            "torch_runtime":result["training_environment"]["torch_runtime"],
        }
        runtime_identities.append(runtime_identity)
        rows[name]={
            "encoder_architecture":result["encoder_architecture"],
            "hidden_dim":int(result["hidden_dim"]),
            "encoder_layers":int(result["encoder_layers"]),
            "context_frames":int(result["context_frames"]),
            "residual_context_adapter":bool(result["residual_context_adapter"]),
            "initial_encoder_core_sha256":result.get("initial_encoder_core_sha256"),
            "trainable_parameters":int(result["trainable_parameters"]),
            "initial_model_state_sha256":result["initial_model_state_sha256"],
            "final_model_state_sha256":result["model_state_sha256"],
            "final_train_loss":float(result["history"][-1]["loss"]),
            "operating_points":select_operating_points(result,config),
            "resource":spec["resource"],
        }
    if any(value!=runtime_identities[0] for value in runtime_identities[1:]):
        raise ValueError("paired discovery candidates did not share identical runtime identity")
    reference_name=str(config["selection"]["reference_candidate"])
    reference=rows[reference_name]
    core_identity_checks={}
    for item in config.get("core_identity_checks",[]):
        candidate=str(item["candidate"])
        source=str(item["reference"])
        matched=(
            rows[candidate].get("initial_encoder_core_sha256") is not None
            and rows[candidate].get("initial_encoder_core_sha256")
            == rows[source].get("initial_encoder_core_sha256")
        )
        core_identity_checks[f"{candidate}:{source}"]={
            "matched":matched,
            "candidate_sha256":rows[candidate].get("initial_encoder_core_sha256"),
            "reference_sha256":rows[source].get("initial_encoder_core_sha256"),
        }
        if bool(item.get("required",False)) and not matched:
            raise ValueError(f"{candidate}: initial GRU core does not match {source}")
    primary=str(config["selection"]["primary_operating_point"])
    secondary=str(config["selection"]["secondary_operating_point"])
    comparisons={}
    for name in config["selection"]["candidate_order"]:
        target=rows[name]
        comparisons[name]={
            "primary_calibration_delta":calibration_delta(
                target["operating_points"].get(primary),
                reference["operating_points"].get(primary),
            ),
            "secondary_calibration_delta":calibration_delta(
                target["operating_points"].get(secondary),
                reference["operating_points"].get(secondary),
            ),
        }
    return {
        "schema_version":1,
        "evidence_class":"kws-v2-efficient-encoder-discovery-trial-v1",
        "evidence_scope":"research-only",
        "diagnostic_only":True,
        "promotion_allowed":False,
        "runtime_format_change_allowed":False,
        "protected_evidence_used":False,
        "shipping_metric":False,
        "model_seed":seed,
        "same_runner_paired":True,
        "runtime_identity":runtime_identities[0],
        "candidates":rows,
        "core_identity_checks":core_identity_checks,
        "calibration_comparisons":comparisons,
    }


def aggregate(config: dict, root: pathlib.Path) -> dict:
    trials={}
    for path in root.rglob("trial-evidence.json"):
        row=load_json(path)
        seed=int(row["model_seed"])
        if seed in trials:
            raise ValueError(f"duplicate seed {seed}")
        trials[seed]=row
    seeds=[int(v) for v in config["fixed"]["model_seeds"]]
    if set(trials)!=set(seeds):
        raise ValueError(f"seed mismatch: {sorted(trials)} vs {seeds}")
    runtime_identity=trials[seeds[0]]["runtime_identity"]
    same_runner_all_seeds=all(
        trial["runtime_identity"]==runtime_identity
        and trial.get("same_runner_paired") is True
        for trial in trials.values()
    )
    if config["numerical_contract"]["all_discovery_seeds_same_runner"] and not same_runner_all_seeds:
        raise ValueError("discovery seeds/candidates did not share one runtime identity")
    rules=config["decision_rules"]
    primary=str(config["selection"]["primary_operating_point"])
    secondary=str(config["selection"]["secondary_operating_point"])
    assessments={}
    for candidate in config["selection"]["candidate_order"]:
        primary_passes=0
        secondary_passes=0
        primary_recalls=[]
        primary_fps=[]
        for seed in seeds:
            trial_row=trials[seed]
            comparison=trial_row["calibration_comparisons"][candidate]
            pd=comparison["primary_calibration_delta"]
            sd=comparison["secondary_calibration_delta"]
            point=trial_row["candidates"][candidate]["operating_points"].get(primary)
            if point is not None:
                primary_recalls.append(float(point["calibration"]["wake_recall"]))
                primary_fps.append(float(point["calibration"]["negative_fp_rate"]))
            if (
                pd is not None
                and float(pd["wake_recall"])>=float(rules["primary_min_calibration_wake_recall_delta"])
                and float(pd["negative_fp_rate"])<=float(rules["max_calibration_negative_fp_delta_regression"])
            ):
                primary_passes+=1
            if (
                sd is not None
                and float(sd["wake_recall"])>=float(rules["secondary_min_calibration_wake_recall_delta"])
                and float(sd["negative_fp_rate"])<=float(rules["max_calibration_negative_fp_delta_regression"])
            ):
                secondary_passes+=1
        resource=config["candidates"][candidate]["resource"]
        resource_fit=(
            float(resource["dense_mmac_per_s"])<=float(rules["max_candidate_dense_mmac_per_s"])
            and int(resource["int8_weight_float_bias_payload_bytes"])<=int(rules["max_candidate_payload_bytes"])
            and int(resource["recurrent_state_bytes"])<=int(rules["max_candidate_recurrent_state_bytes"])
        )
        mean_recall=statistics.fmean(primary_recalls) if primary_recalls else None
        eligible=(
            resource_fit
            and primary_passes>=int(rules["min_primary_directional_passes"])
            and secondary_passes>=int(rules["min_secondary_directional_passes"])
            and mean_recall is not None
            and mean_recall>=float(rules["min_primary_calibration_wake_recall_mean"])
        )
        assessments[candidate]={
            "primary_directional_passes":primary_passes,
            "secondary_directional_passes":secondary_passes,
            "primary_calibration_wake_recall_mean":mean_recall,
            "primary_calibration_negative_fp_mean":statistics.fmean(primary_fps) if primary_fps else None,
            "resource_fit":resource_fit,
            "resource":resource,
            "eligible":eligible,
        }

    eligible=[name for name in config["selection"]["candidate_order"] if assessments[name]["eligible"]]
    selected=None
    if eligible:
        selected=min(
            eligible,
            key=lambda name:(
                float(assessments[name]["resource"]["dense_mmac_per_s"]),
                int(assessments[name]["resource"]["int8_weight_float_bias_payload_bytes"]),
                -float(assessments[name]["primary_calibration_wake_recall_mean"]),
                name,
            ),
        )

    # Test is reported only after the calibration/resource selection is frozen.
    selected_test=None
    if selected is not None:
        primary_test=[]
        secondary_test=[]
        for seed in seeds:
            row=trials[seed]["candidates"][selected]["operating_points"]
            p=row.get(primary)
            s=row.get(secondary)
            if p is not None:
                primary_test.append({
                    "seed":seed,
                    "wake_recall":float(p["test"]["wake_recall"]),
                    "negative_fp_rate":float(p["test"]["negative_fp_rate"]),
                })
            if s is not None:
                secondary_test.append({
                    "seed":seed,
                    "wake_recall":float(s["test"]["wake_recall"]),
                    "negative_fp_rate":float(s["test"]["negative_fp_rate"]),
                })
        selected_test={
            "primary":primary_test,
            "secondary":secondary_test,
        }

    passed=selected is not None
    result={
        "schema_version":1,
        "evidence_class":"kws-v2-efficient-encoder-discovery-summary-v1",
        "evidence_scope":"research-only",
        "diagnostic_only":True,
        "promotion_allowed":False,
        "runtime_format_change_allowed":False,
        "protected_evidence_used":False,
        "shipping_metric":False,
        "selection_authority":config["selection"]["authority"],
        "test_metrics_used_for_selection":False,
        "causal_authority":config["numerical_contract"]["causal_authority"],
        "same_runner_all_seeds":same_runner_all_seeds,
        "runtime_identity":runtime_identity,
        "candidate_assessments":assessments,
        "selected_candidate":selected,
        "selected_candidate_test_after_selection":selected_test,
        "discovery_pass":passed,
        "trials":{str(seed):trials[seed] for seed in seeds},
        "next_experiment":(
            f"{rules['pass_next']}:{selected}"
            if passed
            else str(rules["fail_next"])
        ),
    }
    return result


def main() -> int:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="mode",required=True)
    one=sub.add_parser("trial")
    one.add_argument("--config",required=True,type=pathlib.Path)
    one.add_argument("--model-seed",required=True,type=int)
    one.add_argument("--root",required=True,type=pathlib.Path)
    one.add_argument("--output",required=True,type=pathlib.Path)
    many=sub.add_parser("aggregate")
    many.add_argument("--config",required=True,type=pathlib.Path)
    many.add_argument("--root",required=True,type=pathlib.Path)
    many.add_argument("--output",required=True,type=pathlib.Path)
    args=parser.parse_args()
    config=load_json(args.config)
    result=(
        trial(config,int(args.model_seed),args.root.resolve())
        if args.mode=="trial"
        else aggregate(config,args.root.resolve())
    )
    write_json(args.output.resolve(),result)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
