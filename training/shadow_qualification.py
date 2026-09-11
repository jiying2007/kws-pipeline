#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

from iterate_domain import base_gate, domain_gate, evaluate, gate_values, sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from render_qualification_holdout import require_strict_development_candidate  # noqa: E402
from synthetic_audio import load_config, parse_keywords  # noqa: E402


def validate_shadow_policy(cfg: dict) -> dict:
    raw = cfg.get("shadow_qualification")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        raise ValueError("shadow qualification must be enabled before formal qualification")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not 8 <= len(seeds) <= 16:
        raise ValueError("shadow qualification requires 8..16 fixed development seeds")
    normalized = [int(value) for value in seeds]
    if len(set(normalized)) != len(normalized) or any(value < 0 for value in normalized):
        raise ValueError("shadow qualification seeds must be unique non-negative integers")
    training_seed = int(cfg.get("seed", 1337))
    formal_seed = int(cfg.get("qualification_holdout_seed", -1))
    retired = {int(value) for value in cfg.get("retired_qualification_holdout_seeds", [])}
    forbidden = retired | {training_seed, formal_seed}
    overlap = sorted(set(normalized) & forbidden)
    if overlap:
        raise ValueError(f"shadow qualification seed overlaps formal/development namespace: {overlap}")
    separation = float(raw.get("min_surrogate_separation", 0.06))
    if not 0.0 < separation < 1.0:
        raise ValueError("shadow surrogate separation must be in (0,1)")
    expected = int(raw.get("expected_wakes_per_seed", 256))
    if expected != 256:
        raise ValueError("shadow qualification must retain full 256-wake support per seed")
    return {
        "seeds": normalized,
        "min_surrogate_separation": separation,
        "expected_wakes_per_seed": expected,
    }


def _effective_config(cfg: dict, seed: int) -> dict:
    rendered = copy.deepcopy(cfg)
    rendered["seed"] = seed
    rendered.pop("qualification_holdout_seed", None)
    rendered.pop("retired_qualification_holdout_seeds", None)
    domains = rendered.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domains config is required")
    scenes = domains.get("scenes_per_example")
    if not isinstance(scenes, dict):
        raise ValueError("domains.scenes_per_example must be an object")
    for split in ("train", "calibration", "test"):
        scenes[split] = 1
    scenes["qualification"] = 8
    return rendered


def _selected_record(manifest: dict) -> dict:
    selection = manifest["candidate_selection"]
    selected_round = int(selection["selected_round"])
    selected_frontend = str(selection["selected_frontend"])
    rows = [
        row
        for row in manifest["records"]
        if int(row["round"]) == selected_round
        and str(row["frontend"]) == selected_frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("shadow qualification cannot resolve selected strict checkpoint")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def _surrogate_separation(
    *,
    checkpoint_path: pathlib.Path,
    index_path: pathlib.Path,
    keywords_path: pathlib.Path,
    tokens_path: pathlib.Path,
) -> dict:
    from surrogate_score import load_checkpoint_model, score_wav

    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    sequences = [list(item["token_ids"]) for item in keywords]
    by_id = {int(item["id"]): index for index, item in enumerate(keywords)}
    model, checkpoint = load_checkpoint_model(checkpoint_path)
    feature_dim = int(checkpoint["feature_dim"])
    frontend = str(checkpoint["frontend_name"])
    positives: dict[int, list[float]] = {int(item["id"]): [] for item in keywords}
    negatives: dict[int, list[float]] = {int(item["id"]): [] for item in keywords}

    for raw in index_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if str(row.get("split")) != "qualification":
            continue
        scores = score_wav(
            model,
            pathlib.Path(str(row["path"])),
            feature_dim=feature_dim,
            frontend=frontend,
            keyword_sequences=sequences,
        )
        if str(row.get("kind")) == "positive":
            keyword_id = int(row["keyword_id"])
            positives[keyword_id].append(float(scores[by_id[keyword_id]]))
        else:
            for keyword_id, index in by_id.items():
                negatives[keyword_id].append(float(scores[index]))

    result: dict[str, dict] = {}
    gaps: list[float] = []
    for item in keywords:
        keyword_id = int(item["id"])
        if not positives[keyword_id] or not negatives[keyword_id]:
            raise ValueError("shadow qualification lacks positive/negative surrogate support")
        minimum_positive = min(positives[keyword_id])
        maximum_negative = max(negatives[keyword_id])
        gap = minimum_positive - maximum_negative
        gaps.append(gap)
        result[str(keyword_id)] = {
            "minimum_positive_confidence": minimum_positive,
            "maximum_negative_confidence": maximum_negative,
            "separation": gap,
            "positive_samples": len(positives[keyword_id]),
            "negative_samples": len(negatives[keyword_id]),
        }
    return {"per_keyword": result, "minimum_separation": min(gaps)}


def _metric_compact(base: dict) -> dict:
    result = {
        key: base[key]
        for key in ("expected", "matched", "false_rejects", "false_accepts", "frr", "far_per_hour")
        if key in base
    }
    per_keyword = base.get("per_keyword")
    if isinstance(per_keyword, dict):
        result["per_keyword"] = {
            str(key): {
                field: row[field]
                for field in ("expected", "matched", "false_rejects", "false_accepts", "frr")
                if isinstance(row, dict) and field in row
            }
            for key, row in sorted(per_keyword.items(), key=lambda item: str(item[0]))
            if isinstance(row, dict)
        }
    return result


def _emit_failure_annotation(row: dict, required_separation: float) -> None:
    payload = {
        "seed": int(row["seed"]),
        "runtime_qualified": bool(row["runtime_qualified"]),
        "surrogate_separation_qualified": bool(row["surrogate_separation_qualified"]),
        "required_separation": float(required_separation),
        "qualification": _metric_compact(row["qualification"]),
        "surrogate": row["surrogate"],
    }
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title=Shadow qualification failed::{message}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    policy = validate_shadow_policy(cfg)
    work = args.work_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development = require_strict_development_candidate(work)
    selected = _selected_record(development)
    checkpoint = pathlib.Path(str(selected["checkpoint"])).resolve()
    model = work / "best" / "model.kwm"
    pack = work / "best" / "keywords.kwk"
    keywords = pathlib.Path(str(cfg["keywords"]))
    tokens = pathlib.Path(str(cfg["tokens"]))
    if not keywords.is_absolute():
        keywords = (ROOT / keywords).resolve()
    if not tokens.is_absolute():
        tokens = (ROOT / tokens).resolve()
    gates = gate_values(cfg.get("domain_gates", {}))

    results: list[dict] = []
    all_qualified = True
    for seed in policy["seeds"]:
        seed_root = output / f"seed-{seed}"
        effective = seed_root / "effective-config.json"
        effective.parent.mkdir(parents=True, exist_ok=True)
        effective.write_text(
            json.dumps(_effective_config(cfg, seed), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        dataset = seed_root / "dataset"
        render_domain_dataset(effective, dataset, curriculum_weights=None)
        base, domains = evaluate(
            runner=args.runner.resolve(),
            model=model,
            pack=pack,
            references=dataset / "qualification.references.jsonl",
            output=seed_root / "eval",
        )
        expected = int(base.get("expected_wakes", base.get("expected", 0)))
        if expected != policy["expected_wakes_per_seed"]:
            raise ValueError(f"shadow seed {seed} expected-wake support drifted: {expected}")
        separation = _surrogate_separation(
            checkpoint_path=checkpoint,
            index_path=dataset / "domain-index.jsonl",
            keywords_path=keywords,
            tokens_path=tokens,
        )
        runtime_qualified = base_gate(base, gates) and domain_gate(domains, gates)
        separation_qualified = (
            float(separation["minimum_separation"])
            >= float(policy["min_surrogate_separation"])
        )
        qualified = runtime_qualified and separation_qualified
        all_qualified = all_qualified and qualified
        row = {
            "seed": seed,
            "qualified": qualified,
            "runtime_qualified": runtime_qualified,
            "surrogate_separation_qualified": separation_qualified,
            "qualification": base,
            "qualification_domains": domains,
            "surrogate": separation,
            "domain_index_sha256": sha256_file(dataset / "domain-index.jsonl"),
            "references_sha256": sha256_file(dataset / "qualification.references.jsonl"),
        }
        results.append(row)
        if not qualified:
            _emit_failure_annotation(row, float(policy["min_surrogate_separation"]))

    selection = development["candidate_selection"]
    summary = {
        "schema_version": 1,
        "evidence_class": "development-only-shadow-qualification",
        "qualified": all_qualified,
        "seeds": policy["seeds"],
        "expected_wakes_per_seed": policy["expected_wakes_per_seed"],
        "min_surrogate_separation": policy["min_surrogate_separation"],
        "development_manifest_sha256": sha256_file(work / "domain-loop-manifest.json"),
        "development_selected_round": int(selection["selected_round"]),
        "development_selected_frontend": str(selection["selected_frontend"]),
        "formal_qualification_seed_consumed": False,
        "results": results,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    failures = [row for row in results if not bool(row["qualified"])]
    failure_summary = {
        "schema_version": 1,
        "evidence_class": "compact-shadow-failure-diagnostics",
        "qualified": all_qualified,
        "required_separation": float(policy["min_surrogate_separation"]),
        "failure_count": len(failures),
        "failures": [
            {
                "seed": int(row["seed"]),
                "runtime_qualified": bool(row["runtime_qualified"]),
                "surrogate_separation_qualified": bool(row["surrogate_separation_qualified"]),
                "qualification": _metric_compact(row["qualification"]),
                "surrogate": row["surrogate"],
            }
            for row in failures
        ],
    }
    (output / "failure-summary.json").write_text(
        json.dumps(failure_summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all_qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        message = str(exc).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title=Shadow qualification infrastructure failure::{message}")
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
