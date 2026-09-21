from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"

from adversarial_lexicon import mine_adversarial_lexicon
from development_failure_replay import render_development_failure_replay
from hard_negative_replay import render_hard_negative_replay
from feature_cached_trainer import feature_cache_max_items, rewrite_training_command
from iterate_domain import (
    base_gate,
    calibrate,
    domain_gate,
    evaluate,
    gate_values,
    objective,
    repo_path,
    run,
    sha256_file,
    train_acoustic_seed_offset,
)
from qualification_failure_replay import (
    POLICY as QUALIFICATION_REPAIR_POLICY,
    REPAIR_EPOCHS,
    REPAIR_LR_SCALE,
    render_qualification_failure_replay,
)
from render_domains import render_domain_dataset
from synthetic_audio import load_config

POLICY = "post-domain-adversarial-refinement-v1"
REFINEMENT_SOURCE_POLICY = "development-recall-first-refinement-source-v1"
WAKE_BALANCE_POLICY = "per-keyword-provenance-pressure-balance-v3"
PRESSURE_ASSIGNMENT_POLICY = "explicit-replay-focus-then-token-edit-distance-v2"
DEFAULT_POSITIVE_EXAMPLE_WEIGHT = 2.0
MAX_WAKE_EXAMPLE_WEIGHT = 12.0
REPAIR_VALIDATION_SEED_NAMESPACE = 171_000_003


def _selected_record(manifest: dict) -> dict:
    selection = manifest.get("candidate_selection", {})
    selected_round = int(selection.get("selected_round", -1))
    selected_frontend = str(selection.get("selected_frontend") or "")
    rows = [
        row
        for row in manifest.get("records", [])
        if isinstance(row, dict)
        and int(row.get("round", -1)) == selected_round
        and str(row.get("frontend") or "") == selected_frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("adversarial refinement cannot resolve selected strict checkpoint")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def _far_domain_frr(record: dict, split: str) -> float:
    domains = record.get(f"{split}_domains")
    if not isinstance(domains, dict):
        raise ValueError(f"refinement source {split} domains are missing")
    far = domains.get("domains", {}).get("distance:far")
    if not isinstance(far, dict):
        raise ValueError(f"refinement source {split} far-domain metrics are missing")
    return float(far["frr"])


def _per_keyword_frr(metrics: dict, split: str) -> tuple[float, ...]:
    raw = metrics.get("per_keyword")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"refinement source {split} per-keyword metrics are missing")
    values: list[float] = []
    for keyword_id in sorted(raw, key=str):
        row = raw[keyword_id]
        if not isinstance(row, dict):
            raise ValueError(f"refinement source {split} keyword {keyword_id} metrics are invalid")
        values.append(float(row["frr"]))
    return tuple(values)


def refinement_source_key(record: dict) -> tuple[float, float, float, float, float, int, str]:
    calibration = record.get("calibration")
    test = record.get("test")
    if not isinstance(calibration, dict) or not isinstance(test, dict):
        raise ValueError("refinement source calibration/test metrics are missing")
    recall_terms = (
        float(calibration["frr"]),
        float(test["frr"]),
        _far_domain_frr(record, "calibration"),
        _far_domain_frr(record, "test"),
        *_per_keyword_frr(calibration, "calibration"),
        *_per_keyword_frr(test, "test"),
    )
    far_terms = (
        float(calibration["far_per_hour"]),
        float(test["far_per_hour"]),
    )
    if any(not math.isfinite(value) for value in (*recall_terms, *far_terms)):
        raise ValueError("refinement source metrics must be finite")
    return (
        max(recall_terms),
        sum(recall_terms),
        max(far_terms),
        sum(far_terms),
        float(record["score"]),
        int(record["round"]),
        str(record["frontend"]),
    )


def select_refinement_source(manifest: dict) -> tuple[dict, str]:
    selection = manifest.get("candidate_selection")
    if not isinstance(selection, dict):
        raise ValueError("development candidate-selection evidence is missing")
    if selection.get("qualification_used_for_selection") is not False:
        raise ValueError("refinement source selection must not use qualification")

    if bool(manifest.get("development_qualified")):
        return _selected_record(manifest), "strict-development-candidate"

    if selection.get("objective_fallback_used") is not True:
        raise ValueError("unqualified development manifest lacks objective fallback evidence")
    if selection.get("selected_round") is not None or selection.get("selected_frontend") is not None:
        raise ValueError("unqualified development manifest unexpectedly claims a strict selection")

    candidates = [
        row
        for row in manifest.get("records", [])
        if isinstance(row, dict) and "checkpoint" in row
    ]
    if not candidates:
        raise ValueError("development manifest has no checkpoint eligible for refinement")
    return min(candidates, key=refinement_source_key), REFINEMENT_SOURCE_POLICY


def _manifest_targets(path: pathlib.Path) -> list[tuple[int, ...]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"refinement training manifest is missing/empty: {path}")
    rows: list[tuple[int, ...]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
        _, token_text = raw.split("\t", 1)
        try:
            targets = tuple(int(value) for value in token_text.split())
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid token id") from exc
        if any(value < 0 for value in targets):
            raise ValueError(f"{path}:{line_no}: token ids must be non-negative")
        rows.append(targets)
    if not rows:
        raise ValueError(f"refinement training manifest has no examples: {path}")
    return rows


def _keyword_target_sequences(
    tokens: pathlib.Path,
    keywords: pathlib.Path,
) -> dict[int, tuple[int, ...]]:
    token_map: dict[str, int] = {}
    for line_no, raw in enumerate(tokens.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        cols = value.split()
        if len(cols) == 1:
            token = cols[0]
            token_id = len(token_map)
        elif len(cols) == 2:
            token, raw_id = cols
            token_id = int(raw_id)
        else:
            raise ValueError(f"{tokens}:{line_no}: invalid token row")
        if token in token_map:
            raise ValueError(f"{tokens}:{line_no}: duplicate token")
        token_map[token] = token_id

    sequences: dict[int, tuple[int, ...]] = {}
    seen_sequences: set[tuple[int, ...]] = set()
    for line_no, raw in enumerate(keywords.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(f"{keywords}:{line_no}: expected four columns")
        keyword_id = int(cols[0])
        if keyword_id <= 0 or keyword_id in sequences:
            raise ValueError(f"{keywords}:{line_no}: keyword id must be unique and positive")
        names = cols[3].split()
        try:
            sequence = tuple(token_map[name] for name in names)
        except KeyError as exc:
            raise ValueError(f"{keywords}:{line_no}: unknown token {exc.args[0]}") from exc
        if not sequence:
            raise ValueError(f"{keywords}:{line_no}: empty wake sequence")
        if sequence in seen_sequences:
            raise ValueError(f"{keywords}:{line_no}: duplicate wake sequence")
        sequences[keyword_id] = sequence
        seen_sequences.add(sequence)
    if not sequences:
        raise ValueError("shipping keyword TSV contains no wake sequences")
    return sequences


def _sequence_edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            substitution = previous[right_index - 1] + int(left_value != right_value)
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def _nearest_keyword_ids(
    targets: tuple[int, ...],
    wake_sequences: dict[int, tuple[int, ...]],
) -> list[int]:
    keyword_ids = sorted(wake_sequences)
    if not targets:
        return keyword_ids
    distances = {
        keyword_id: _sequence_edit_distance(targets, wake_sequences[keyword_id])
        for keyword_id in keyword_ids
    }
    minimum = min(distances.values())
    return [
        keyword_id
        for keyword_id in keyword_ids
        if distances[keyword_id] == minimum
    ]


def _repeat_focus_rows(
    output: list[tuple[int, ...]],
    raw_focus: object,
    count: int,
    *,
    label: str,
) -> None:
    if count < 0:
        raise ValueError(f"{label} example count must be non-negative")
    if raw_focus is None:
        focus: tuple[int, ...] = ()
    elif isinstance(raw_focus, (list, tuple)):
        values = [int(value) for value in raw_focus if int(value) > 0]
        focus = tuple(sorted(set(values)))
    else:
        value = int(raw_focus)
        focus = (value,) if value > 0 else ()
    output.extend([focus] * count)


def build_refinement_focus_rows(
    *,
    static: dict,
    adversarial: dict,
    failure: dict,
) -> dict[pathlib.Path, list[tuple[int, ...]]]:
    """Recover row-aligned replay focus from renderer evidence.

    TSV manifests intentionally stay minimal for the trainer. The replay
    renderers already retain semantic focus in their sidecar evidence, so wake
    pressure attribution must consume that authority before falling back to
    token edit distance.
    """
    result: dict[pathlib.Path, list[tuple[int, ...]]] = {}

    static_rows: list[tuple[int, ...]] = []
    for index, item in enumerate(static.get("sequences", [])):
        if not isinstance(item, dict):
            raise ValueError("hard-negative sequence evidence must be objects")
        _repeat_focus_rows(
            static_rows,
            item.get("focus_keyword_id"),
            int(item.get("examples", 0)),
            label=f"hard-negative sequence {index}",
        )
    for index, item in enumerate(static.get("positive_stress", [])):
        if not isinstance(item, dict):
            raise ValueError("positive-stress evidence must be objects")
        _repeat_focus_rows(
            static_rows,
            item.get("keyword_id"),
            int(item.get("examples", 0)),
            label=f"positive-stress keyword {index}",
        )
    if len(static_rows) != int(static.get("examples", -1)):
        raise ValueError("hard-negative replay focus evidence row count drifted")
    static_manifest = pathlib.Path(str(static.get("manifest") or "")).resolve()
    result[static_manifest] = static_rows

    adversarial_rows: list[tuple[int, ...]] = []
    replay_per_sequence = int(adversarial.get("replay_examples_per_sequence", 0))
    for index, item in enumerate(adversarial.get("selected", [])):
        if not isinstance(item, dict):
            raise ValueError("adversarial selected evidence must be objects")
        _repeat_focus_rows(
            adversarial_rows,
            item.get("focus_keyword_id"),
            replay_per_sequence,
            label=f"adversarial sequence {index}",
        )
    if len(adversarial_rows) != int(adversarial.get("replay_examples", -1)):
        raise ValueError("adversarial replay focus evidence row count drifted")
    adversarial_manifest = pathlib.Path(str(adversarial.get("manifest") or "")).resolve()
    result[adversarial_manifest] = adversarial_rows

    failure_rows: list[tuple[int, ...]] = []
    base_repeat = int(failure.get("examples_per_failure", 0))
    repair_repeat = int(failure.get("qualification_repair_examples_per_failure", 0))
    for index, item in enumerate(failure.get("selected", [])):
        if not isinstance(item, dict):
            raise ValueError("failure replay selected evidence must be objects")
        source_splits = [str(value) for value in item.get("source_splits", [])]
        repeat = repair_repeat if "qualification" in source_splits else base_repeat
        focus = item.get("focus_keyword_ids", [])
        if not focus and item.get("source_keyword_id") is not None:
            focus = [int(item["source_keyword_id"])]
        _repeat_focus_rows(
            failure_rows,
            focus,
            repeat,
            label=f"failure replay selection {index}",
        )
    if len(failure_rows) != int(failure.get("examples", -1)):
        raise ValueError("failure replay focus evidence row count drifted")
    if int(failure.get("examples", 0)) > 0:
        failure_manifest = pathlib.Path(str(failure.get("manifest") or "")).resolve()
        if failure_manifest in result:
            raise ValueError("refinement replay focus manifests must be distinct")
        result[failure_manifest] = failure_rows

    return result


def derive_refinement_wake_balance(
    *,
    manifests: list[pathlib.Path],
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    positive_example_weight: float,
    focus_rows_by_manifest: dict[pathlib.Path, list[tuple[int, ...]]] | None = None,
) -> dict:
    if (
        not math.isfinite(positive_example_weight)
        or positive_example_weight <= 0.0
    ):
        raise ValueError("refinement positive example weight must be finite and > 0")
    wake_sequences = _keyword_target_sequences(tokens, keywords)
    sequence_to_keyword = {
        sequence: keyword_id for keyword_id, sequence in wake_sequences.items()
    }
    wake_rows_by_keyword = {keyword_id: 0 for keyword_id in wake_sequences}
    assigned_nonwake_mass = {keyword_id: 0.0 for keyword_id in wake_sequences}
    assigned_nonwake_rows = {keyword_id: 0.0 for keyword_id in wake_sequences}
    wake_rows = tokenized_nonwake_rows = empty_nonwake_rows = 0
    explicit_focus_nonwake_rows = fallback_edit_distance_nonwake_rows = 0
    per_manifest: list[dict] = []

    manifest_keys = {manifest.resolve() for manifest in manifests}
    normalized_focus = {
        pathlib.Path(path).resolve(): list(rows)
        for path, rows in (focus_rows_by_manifest or {}).items()
    }
    unknown_focus_manifests = sorted(
        str(path) for path in set(normalized_focus) - manifest_keys
    )
    if unknown_focus_manifests:
        raise ValueError(
            "refinement focus evidence contains unknown manifest(s): "
            + ", ".join(unknown_focus_manifests)
        )

    for manifest in manifests:
        targets = _manifest_targets(manifest)
        focus_rows = normalized_focus.get(manifest.resolve())
        if focus_rows is not None and len(focus_rows) != len(targets):
            raise ValueError(
                f"refinement focus evidence row count differs from manifest: {manifest}"
            )
        local_wake = {keyword_id: 0 for keyword_id in wake_sequences}
        local_nonwake_mass = {keyword_id: 0.0 for keyword_id in wake_sequences}
        local_nonwake_rows = {keyword_id: 0.0 for keyword_id in wake_sequences}
        local_tokenized = 0
        local_empty = 0
        local_explicit_focus = 0
        local_fallback_focus = 0
        for row_index, row in enumerate(targets):
            keyword_id = sequence_to_keyword.get(row)
            if keyword_id is not None:
                wake_rows += 1
                wake_rows_by_keyword[keyword_id] += 1
                local_wake[keyword_id] += 1
                continue

            mass = positive_example_weight if row else 1.0
            if row:
                tokenized_nonwake_rows += 1
                local_tokenized += 1
            else:
                empty_nonwake_rows += 1
                local_empty += 1
            raw_focus = focus_rows[row_index] if focus_rows is not None else ()
            if raw_focus:
                nearest = sorted({int(value) for value in raw_focus})
                unknown = [value for value in nearest if value not in wake_sequences]
                if unknown:
                    raise ValueError(
                        "refinement focus evidence references unknown keyword id(s): "
                        + ", ".join(str(value) for value in unknown)
                    )
                explicit_focus_nonwake_rows += 1
                local_explicit_focus += 1
            else:
                nearest = _nearest_keyword_ids(row, wake_sequences)
                fallback_edit_distance_nonwake_rows += 1
                local_fallback_focus += 1
            share = 1.0 / float(len(nearest))
            for nearest_id in nearest:
                assigned_nonwake_rows[nearest_id] += share
                assigned_nonwake_mass[nearest_id] += mass * share
                local_nonwake_rows[nearest_id] += share
                local_nonwake_mass[nearest_id] += mass * share

        per_manifest.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "rows": len(targets),
                "wake_rows": sum(local_wake.values()),
                "wake_rows_by_keyword": {
                    str(keyword_id): int(local_wake[keyword_id])
                    for keyword_id in sorted(local_wake)
                },
                "tokenized_nonwake_rows": local_tokenized,
                "empty_nonwake_rows": local_empty,
                "explicit_focus_nonwake_rows": local_explicit_focus,
                "fallback_edit_distance_nonwake_rows": local_fallback_focus,
                "assigned_nonwake_rows_by_keyword": {
                    str(keyword_id): local_nonwake_rows[keyword_id]
                    for keyword_id in sorted(local_nonwake_rows)
                },
                "assigned_nonwake_mass_by_keyword": {
                    str(keyword_id): local_nonwake_mass[keyword_id]
                    for keyword_id in sorted(local_nonwake_mass)
                },
            }
        )

    if wake_rows <= 0:
        raise ValueError("refinement manifests contain no exact configured wake examples")

    keyword_balance: dict[str, dict] = {}
    wake_keyword_weights: dict[str, float] = {}
    effective_wake_mass = 0.0
    any_bounded = False
    for keyword_id in sorted(wake_sequences):
        keyword_wake_rows = wake_rows_by_keyword[keyword_id]
        if keyword_wake_rows <= 0:
            raise ValueError(
                f"refinement manifests contain no exact wake examples for keyword {keyword_id}"
            )
        wake_base_mass = keyword_wake_rows * positive_example_weight
        target_mass = assigned_nonwake_mass[keyword_id]
        raw_weight = target_mass / wake_base_mass
        wake_weight = min(MAX_WAKE_EXAMPLE_WEIGHT, max(1.0, raw_weight))
        bounded = wake_weight != raw_weight
        effective_mass = wake_base_mass * wake_weight
        any_bounded = any_bounded or bounded
        effective_wake_mass += effective_mass
        wake_keyword_weights[str(keyword_id)] = wake_weight
        keyword_balance[str(keyword_id)] = {
            "wake_rows": keyword_wake_rows,
            "wake_base_mass": wake_base_mass,
            "assigned_nonwake_rows": assigned_nonwake_rows[keyword_id],
            "assigned_nonwake_mass": target_mass,
            "raw_wake_example_weight": raw_weight,
            "wake_example_weight": wake_weight,
            "effective_wake_mass": effective_mass,
            "bounded": bounded,
        }

    wake_base_mass = wake_rows * positive_example_weight
    nonwake_mass = (
        tokenized_nonwake_rows * positive_example_weight + empty_nonwake_rows
    )
    assigned_mass_total = sum(assigned_nonwake_mass.values())
    if not math.isclose(assigned_mass_total, nonwake_mass, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("per-keyword non-wake pressure does not conserve effective mass")

    return {
        "schema_version": 3,
        "policy": WAKE_BALANCE_POLICY,
        "positive_example_weight": positive_example_weight,
        "default_wake_example_weight": 1.0,
        "wake_keyword_weights": wake_keyword_weights,
        "max_wake_example_weight": MAX_WAKE_EXAMPLE_WEIGHT,
        "bounded": any_bounded,
        "wake_rows": wake_rows,
        "wake_rows_by_keyword": {
            str(keyword_id): wake_rows_by_keyword[keyword_id]
            for keyword_id in sorted(wake_rows_by_keyword)
        },
        "tokenized_nonwake_rows": tokenized_nonwake_rows,
        "empty_nonwake_rows": empty_nonwake_rows,
        "explicit_focus_nonwake_rows": explicit_focus_nonwake_rows,
        "fallback_edit_distance_nonwake_rows": fallback_edit_distance_nonwake_rows,
        "wake_base_mass": wake_base_mass,
        "nonwake_mass": nonwake_mass,
        "effective_wake_mass": effective_wake_mass,
        "keyword_balance": keyword_balance,
        "pressure_assignment": PRESSURE_ASSIGNMENT_POLICY,
        "manifests": per_manifest,
    }

def _refinement_policy(cfg: dict) -> dict:
    iteration = cfg.get("domain_iteration", {})
    if not isinstance(iteration, dict):
        raise ValueError("domain_iteration must be an object")
    raw = iteration.get("adversarial_lexicon")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        raise ValueError("adversarial lexicon must be enabled for refinement")
    epochs = int(raw.get("refinement_epochs", 12))
    lr_scale = float(raw.get("refinement_lr_scale", 0.5))
    if epochs <= 0 or not math.isfinite(lr_scale) or not 0.0 < lr_scale <= 1.0:
        raise ValueError("adversarial refinement epochs/lr scale are invalid")
    return {"epochs": epochs, "lr_scale": lr_scale}


def _audit(
    dataset: pathlib.Path,
    splits: tuple[str, ...] = ("train", "calibration", "test", "qualification"),
) -> None:
    argv = [sys.executable, str(TRAINING / "audit_dataset.py")]
    for split in splits:
        argv.extend(["--split", f"{split}={dataset / (split + '.tsv')}"])
    argv.extend(["--report", str(dataset / "audit.json"), "--fail-within-split"])
    run(argv)


def _write_effective_seed_config(cfg: dict, seed: int, path: pathlib.Path) -> pathlib.Path:
    rendered = copy.deepcopy(cfg)
    rendered["seed"] = seed
    rendered.pop("qualification_holdout_seed", None)
    rendered.pop("retired_qualification_holdout_seeds", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _train_refinement(
    *,
    cfg: dict,
    frontend: str,
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    dataset_manifest: pathlib.Path,
    static_manifest: pathlib.Path,
    adversarial_manifest: pathlib.Path,
    failure_manifest: pathlib.Path | None,
    focus_rows_by_manifest: dict[pathlib.Path, list[tuple[int, ...]]],
    warm_start: pathlib.Path,
    output: pathlib.Path,
    epochs: int,
    lr_scale: float,
    refinement_round: int,
    seed_offset: int = 0,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    train = cfg.get("train", {})
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    model = output / "model.kwm"
    provenance = pathlib.Path(str(model) + ".provenance.json")
    learning_rate = float(train.get("lr", 0.001)) * lr_scale
    seed = int(cfg.get("seed", 1337)) + 4_000_003 + refinement_round * 1009 + seed_offset
    training_manifests = [dataset_manifest, static_manifest, adversarial_manifest]
    if failure_manifest is not None:
        training_manifests.append(failure_manifest)
    positive_example_weight = float(
        train.get("positive_example_weight", DEFAULT_POSITIVE_EXAMPLE_WEIGHT)
    )
    wake_balance = derive_refinement_wake_balance(
        manifests=training_manifests,
        tokens=tokens,
        keywords=keywords,
        positive_example_weight=positive_example_weight,
        focus_rows_by_manifest=focus_rows_by_manifest,
    )
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(dataset_manifest),
        "--manifest",
        str(static_manifest),
        "--manifest",
        str(adversarial_manifest),
    ]
    if failure_manifest is not None:
        if not failure_manifest.is_file() or failure_manifest.stat().st_size == 0:
            raise ValueError("non-empty failure replay manifest was requested but is missing")
        command.extend(["--manifest", str(failure_manifest)])
    command.extend(
        [
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--frontend",
            frontend,
            "--feature-dim",
            str(int(cfg.get("model", {}).get("feature_dim", 32))),
            "--hidden-dim",
            str(int(cfg.get("model", {}).get("hidden_dim", 48))),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(int(train.get("batch_size", 16))),
            "--lr",
            str(learning_rate),
            "--seed",
            str(seed),
            "--positive-example-weight",
            str(wake_balance["positive_example_weight"]),
            "--wake-example-weight",
            str(wake_balance["default_wake_example_weight"]),
            "--wake-keyword-weights",
            json.dumps(wake_balance["wake_keyword_weights"], sort_keys=True),
            "--warm-start",
            str(warm_start),
            "--output",
            str(checkpoint),
        ]
    )
    command = rewrite_training_command(command, feature_cache_max_items(train))
    run(command)
    run(
        [
            sys.executable,
            str(TRAINING / "export_model.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(model),
        ]
    )
    return model, checkpoint, provenance, wake_balance


def _strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def _update_record_candidate(
    record: dict,
    *,
    model: pathlib.Path,
    checkpoint: pathlib.Path,
    provenance: pathlib.Path,
    calibrated: pathlib.Path,
    pack: pathlib.Path,
    cal_base: dict,
    cal_domains: dict,
    test_base: dict,
    test_domains: dict,
    failure: dict,
    failure_evidence: pathlib.Path,
    score: float,
    qualification_repair_used: bool,
    wake_balance: dict,
) -> None:
    gates = record["_gates"]
    record.update(
        {
            "score": score,
            "model": str(model),
            "model_sha256": sha256_file(model),
            "checkpoint": str(checkpoint),
            "provenance": str(provenance),
            "provenance_sha256": sha256_file(provenance),
            "keywords": str(calibrated),
            "pack": str(pack),
            "calibration": cal_base,
            "calibration_domains": cal_domains,
            "test": test_base,
            "test_domains": test_domains,
            "calibration_gate": _strict(cal_base, cal_domains, gates),
            "test_gate": _strict(test_base, test_domains, gates),
            "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
            "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
            "failure_replay_examples": int(failure["examples"]),
            "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
            "failure_replay_evidence_sha256": sha256_file(failure_evidence),
            "qualification_repair_used": qualification_repair_used,
            "wake_balance": wake_balance,
        }
    )


def _write_failed_summary(work: pathlib.Path, value: dict) -> None:
    out = work / "adversarial-refinement" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine development-only lexical adversaries and refine a development candidate."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    runner = args.runner.resolve()
    work = args.work_dir.resolve()
    cfg = load_config(config_path)
    if str(cfg.get("domain_iteration", {}).get("backend")) != "torch_ctc":
        raise ValueError("adversarial refinement requires torch_ctc backend")
    policy = _refinement_policy(cfg)
    manifest_path = work / "domain-loop-manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("development manifest is missing; refuse adversarial refinement")
    input_manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("development manifest must be an object")
    source, source_selection_policy = select_refinement_source(manifest)
    source_was_strict = bool(source.get("calibration_gate")) and bool(source.get("test_gate"))
    source_checkpoint = repo_path(str(source["checkpoint"]))
    source_round = int(source["round"])
    frontend = str(source["frontend"])
    refinement_round = max(int(row["round"]) for row in manifest["records"]) + 1
    if refinement_round <= source_round:
        raise ValueError("adversarial refinement round must follow development rounds")

    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    final_curriculum = manifest.get("final_curriculum")
    curriculum = final_curriculum if isinstance(final_curriculum, dict) else None

    dataset = work / "datasets" / f"round-{refinement_round:02d}"
    refinement_train_seed_offset = train_acoustic_seed_offset(
        cfg.get("domain_iteration", {}),
        refinement_round,
    )
    render_domain_dataset(
        config_path,
        dataset,
        curriculum_weights=curriculum,
        splits=("train", "calibration", "test"),
        train_seed_offset=refinement_train_seed_offset,
    )
    _audit(dataset, ("train", "calibration", "test"))

    static = render_hard_negative_replay(
        config_path,
        work / "hard-negative-replay" / f"round-{refinement_round:02d}",
        round_index=refinement_round,
        curriculum_weights=curriculum,
    )
    static_manifest = pathlib.Path(str(static["manifest"]))

    adversarial = mine_adversarial_lexicon(
        config_path,
        source_checkpoint,
        work / "adversarial-lexicon" / f"round-{refinement_round:02d}",
        round_index=refinement_round,
        frontend=frontend,
    )
    adversarial_manifest = pathlib.Path(str(adversarial["manifest"]))
    adversarial_evidence = pathlib.Path(str(adversarial["evidence"]))
    if bool(adversarial.get("formal_qualification_used", True)):
        raise ValueError("adversarial mining must not use formal qualification")
    adversarial_selection_policy = str(adversarial.get("selection_policy") or "")
    adversarial_data_policy = str(adversarial.get("data_augmentation_policy") or "")
    if not adversarial_selection_policy or not adversarial_data_policy:
        raise ValueError("adversarial evidence is missing data/selection policy provenance")

    failure = render_development_failure_replay(
        config_path,
        list(manifest.get("records", [])),
        work,
        work / "development-failure-replay" / f"round-{refinement_round:02d}",
    )
    failure_manifest_path = pathlib.Path(str(failure["manifest"]))
    failure_evidence = pathlib.Path(str(failure["evidence"]))
    if bool(failure.get("formal_qualification_used", True)):
        raise ValueError("development failure replay must not use formal qualification")
    if bool(failure.get("development_source_wav_bytes_copied", True)):
        raise ValueError("development failure replay copied evaluation WAV bytes")
    failure_manifest = failure_manifest_path if int(failure.get("examples", 0)) > 0 else None

    focus_rows_by_manifest = build_refinement_focus_rows(
        static=static,
        adversarial=adversarial,
        failure=failure,
    )
    candidate_dir = work / "candidates" / f"r{refinement_round:02d}-{frontend}-adversarial"
    model, checkpoint, provenance, wake_balance = _train_refinement(
        cfg=cfg,
        frontend=frontend,
        tokens=tokens,
        keywords=keywords,
        dataset_manifest=dataset / "train.tsv",
        static_manifest=static_manifest,
        adversarial_manifest=adversarial_manifest,
        failure_manifest=failure_manifest,
        focus_rows_by_manifest=focus_rows_by_manifest,
        warm_start=source_checkpoint,
        output=candidate_dir,
        epochs=int(policy["epochs"]),
        lr_scale=float(policy["lr_scale"]),
        refinement_round=refinement_round,
    )

    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    calibration_parallel_trials = int(
        cfg.get("calibration", {}).get("max_parallel_trials", 1)
    )
    if not 1 <= calibration_parallel_trials <= 4:
        raise ValueError("calibration.max_parallel_trials must be 1..4")
    gates = gate_values(cfg.get("domain_gates", {}))
    calibrated, pack, cal_base, cal_domains = calibrate(
        runner=runner,
        model=model,
        tokens=tokens,
        source_keywords=keywords,
        references=dataset / "calibration.references.jsonl",
        output=candidate_dir / "calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
        parallel_trials=calibration_parallel_trials,
    )
    test_base, test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=dataset / "test.references.jsonl",
        output=candidate_dir / "test",
    )
    cal_gate = _strict(cal_base, cal_domains, gates)
    test_gate = _strict(test_base, test_domains, gates)
    score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)

    record = {
        "round": refinement_round,
        "stage": POLICY,
        "frontend": frontend,
        "candidate": 0,
        "score": score,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "checkpoint": str(checkpoint),
        "provenance": str(provenance),
        "provenance_sha256": sha256_file(provenance),
        "keywords": str(calibrated),
        "pack": str(pack),
        "calibration": cal_base,
        "calibration_domains": cal_domains,
        "test": test_base,
        "test_domains": test_domains,
        "calibration_gate": cal_gate,
        "test_gate": test_gate,
        "warm_started": True,
        "warm_start_strategy": "full",
        "training_acoustic_seed_policy": "train-only-scene-seed-offset-v1",
        "training_acoustic_seed_offset": refinement_train_seed_offset,
        "source_round": source_round,
        "source_selection_policy": source_selection_policy,
        "source_was_strict": source_was_strict,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "wake_balance": wake_balance,
        "hard_negative_replay_examples": int(static.get("examples", 0)),
        "hard_negative_replay_manifest_sha256": str(static["manifest_sha256"]),
        "adversarial_policy": adversarial_selection_policy,
        "adversarial_data_augmentation_policy": adversarial_data_policy,
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_probes_per_sequence": int(adversarial["probes_per_sequence"]),
        "adversarial_replay_examples_per_sequence": int(adversarial["replay_examples_per_sequence"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "adversarial_min_per_keyword": int(adversarial["min_per_keyword"]),
        "adversarial_per_keyword_selected": dict(adversarial["per_keyword_selected"]),
        "adversarial_strict_prefix_anchors": list(adversarial["strict_prefix_anchors"]),
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "adversarial_evidence_sha256": sha256_file(adversarial_evidence),
        "adversarial_formal_qualification_used": False,
        "failure_replay_policy": str(failure["policy"]),
        "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
        "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
        "failure_replay_examples": int(failure["examples"]),
        "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
        "failure_replay_evidence_sha256": sha256_file(failure_evidence),
        "failure_replay_formal_qualification_used": False,
        "failure_replay_development_source_wav_bytes_copied": False,
        "qualification_repair_used": False,
        "_gates": gates,
    }
    if not cal_gate or not test_gate:
        _write_failed_summary(
            work,
            {
                "schema_version": 3,
                "policy": POLICY,
                "qualified": False,
                "input_development_manifest_sha256": input_manifest_sha,
                "source_round": source_round,
                "source_selection_policy": source_selection_policy,
                "source_was_strict": source_was_strict,
                "refinement_round": refinement_round,
                "adversarial_data_augmentation_policy": adversarial_data_policy,
                "adversarial_selection_policy": adversarial_selection_policy,
                "failure_replay_policy": str(failure["policy"]),
                "failure_replay_examples": int(failure["examples"]),
                "wake_balance": wake_balance,
                "qualification_repair_used": False,
                "record": {key: value for key, value in record.items() if key != "_gates"},
            },
        )
        raise ValueError("adversarial refinement did not retain calibration/test strict dual-pass")

    mining_qualification = work / "development-qualification-mining"
    render_domain_dataset(
        config_path,
        mining_qualification,
        curriculum_weights=None,
        splits=("qualification",),
    )
    mining_eval = candidate_dir / "development-qualification-mining"
    mining_qual_base, mining_qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=mining_qualification / "qualification.references.jsonl",
        output=mining_eval,
    )
    mining_qualification_qualified = _strict(mining_qual_base, mining_qual_domains, gates)
    qualification_repair = None
    validation_seed = int(cfg.get("seed", 1337))
    validation_config = config_path

    if not mining_qualification_qualified:
        failure = render_qualification_failure_replay(
            config_path,
            mining_qualification,
            mining_eval,
            failure_evidence,
            work / "development-failure-replay" / f"round-{refinement_round:02d}-qualification-repair",
        )
        failure_manifest_path = pathlib.Path(str(failure["manifest"]))
        failure_evidence = pathlib.Path(str(failure["evidence"]))
        repair_dir = candidate_dir / "qualification-repair"
        repair_focus_rows_by_manifest = build_refinement_focus_rows(
            static=static,
            adversarial=adversarial,
            failure=failure,
        )
        repaired_model, repaired_checkpoint, repaired_provenance, repaired_wake_balance = _train_refinement(
            cfg=cfg,
            frontend=frontend,
            tokens=tokens,
            keywords=keywords,
            dataset_manifest=dataset / "train.tsv",
            static_manifest=static_manifest,
            adversarial_manifest=adversarial_manifest,
            failure_manifest=failure_manifest_path,
            focus_rows_by_manifest=repair_focus_rows_by_manifest,
            warm_start=checkpoint,
            output=repair_dir,
            epochs=REPAIR_EPOCHS,
            lr_scale=REPAIR_LR_SCALE,
            refinement_round=refinement_round,
            seed_offset=700_001,
        )
        repaired_keywords, repaired_pack, repaired_cal_base, repaired_cal_domains = calibrate(
            runner=runner,
            model=repaired_model,
            tokens=tokens,
            source_keywords=keywords,
            references=dataset / "calibration.references.jsonl",
            output=repair_dir / "calibration",
            thresholds=thresholds,
            rounds=coordinate_rounds,
            gates=gates,
            parallel_trials=calibration_parallel_trials,
        )
        repaired_test_base, repaired_test_domains = evaluate(
            runner=runner,
            model=repaired_model,
            pack=repaired_pack,
            references=dataset / "test.references.jsonl",
            output=repair_dir / "test",
        )
        repaired_cal_gate = _strict(repaired_cal_base, repaired_cal_domains, gates)
        repaired_test_gate = _strict(repaired_test_base, repaired_test_domains, gates)
        repaired_score = objective(repaired_cal_base, repaired_cal_domains, gates) + objective(
            repaired_test_base, repaired_test_domains, gates
        )

        validation_seed = int(cfg.get("seed", 1337)) + REPAIR_VALIDATION_SEED_NAMESPACE
        validation_config = _write_effective_seed_config(
            cfg,
            validation_seed,
            work / "development-qualification-repair-validation-config.json",
        )
        development_qualification = work / "qualification-dataset"
        render_domain_dataset(
            validation_config,
            development_qualification,
            curriculum_weights=None,
            splits=("qualification",),
        )
        repaired_qual_base, repaired_qual_domains = evaluate(
            runner=runner,
            model=repaired_model,
            pack=repaired_pack,
            references=development_qualification / "qualification.references.jsonl",
            output=repair_dir / "development-qualification-validation",
        )
        repaired_qual_gate = _strict(repaired_qual_base, repaired_qual_domains, gates)
        qualification_repair = {
            "policy": QUALIFICATION_REPAIR_POLICY,
            "epochs": REPAIR_EPOCHS,
            "lr_scale": REPAIR_LR_SCALE,
            "mining_qualification": mining_qual_base,
            "mining_seed": int(cfg.get("seed", 1337)),
            "validation_seed": validation_seed,
            "validation_config_sha256": sha256_file(validation_config),
            "calibration_gate": repaired_cal_gate,
            "test_gate": repaired_test_gate,
            "qualification_gate": repaired_qual_gate,
            "failure_replay_examples": int(failure["examples"]),
            "wake_balance": repaired_wake_balance,
            "qualification_repair_examples": int(failure.get("qualification_repair_examples", 0)),
            "qualification_repair_selected_unique_failures": int(
                failure.get("qualification_repair_selected_unique_failures", 0)
            ),
            "mining_cohort_used_for_training": True,
            "validation_cohort_used_for_training": False,
            "formal_qualification_used": False,
        }
        record["qualification_repair_used"] = True
        record["qualification_repair"] = qualification_repair
        _update_record_candidate(
            record,
            model=repaired_model,
            checkpoint=repaired_checkpoint,
            provenance=repaired_provenance,
            calibrated=repaired_keywords,
            pack=repaired_pack,
            cal_base=repaired_cal_base,
            cal_domains=repaired_cal_domains,
            test_base=repaired_test_base,
            test_domains=repaired_test_domains,
            failure=failure,
            failure_evidence=failure_evidence,
            score=repaired_score,
            qualification_repair_used=True,
            wake_balance=repaired_wake_balance,
        )
        wake_balance = repaired_wake_balance
        if not repaired_cal_gate or not repaired_test_gate or not repaired_qual_gate:
            _write_failed_summary(
                work,
                {
                    "schema_version": 3,
                    "policy": POLICY,
                    "qualified": False,
                    "input_development_manifest_sha256": input_manifest_sha,
                    "source_round": source_round,
                    "refinement_round": refinement_round,
                    "adversarial_data_augmentation_policy": adversarial_data_policy,
                    "adversarial_selection_policy": adversarial_selection_policy,
                    "failure_replay_policy": str(failure["policy"]),
                    "failure_replay_examples": int(failure["examples"]),
                    "qualification_repair_used": True,
                    "qualification_repair": qualification_repair,
                    "record": {key: value for key, value in record.items() if key != "_gates"},
                    "development_qualification": repaired_qual_base,
                },
            )
            raise ValueError("development qualification repair did not reach strict triple-pass")

        model = repaired_model
        checkpoint = repaired_checkpoint
        provenance = repaired_provenance
        calibrated = repaired_keywords
        pack = repaired_pack
        cal_base = repaired_cal_base
        cal_domains = repaired_cal_domains
        test_base = repaired_test_base
        test_domains = repaired_test_domains
        cal_gate = repaired_cal_gate
        test_gate = repaired_test_gate
        score = repaired_score
    else:
        development_qualification = work / "qualification-dataset"
        render_domain_dataset(
            config_path,
            development_qualification,
            curriculum_weights=None,
            splits=("qualification",),
        )

    record.pop("_gates", None)
    manifest["records"].append(record)
    eligible_rounds = sorted(
        {
            int(row["round"])
            for row in manifest["records"]
            if bool(row.get("calibration_gate")) and bool(row.get("test_gate"))
        }
    )
    selection = manifest["candidate_selection"]
    selection.update(
        {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": eligible_rounds,
            "selected_round": refinement_round,
            "selected_frontend": frontend,
            "selected_score": score,
            "objective_fallback_used": False,
            "adversarial_refinement_used": True,
            "adversarial_refinement_policy": POLICY,
            "refinement_source_policy": source_selection_policy,
            "refinement_source_round": source_round,
            "refinement_source_frontend": frontend,
            "refinement_source_was_strict": source_was_strict,
            "adversarial_data_augmentation_policy": adversarial_data_policy,
            "adversarial_selection_policy": adversarial_selection_policy,
            "development_failure_replay_used": int(failure["examples"]) > 0,
            "development_failure_replay_policy": str(failure["policy"]),
            "development_qualification_repair_used": qualification_repair is not None,
            "development_qualification_repair_policy": (
                QUALIFICATION_REPAIR_POLICY if qualification_repair is not None else None
            ),
            "development_qualification_validation_seed": validation_seed,
            "development_qualification_validation_config_sha256": sha256_file(validation_config),
            "development_qualification_validation_used_for_training": False,
        }
    )
    manifest["best_round"] = refinement_round
    manifest["best_frontend"] = frontend
    manifest["best_score"] = score
    manifest["best_model_sha256"] = sha256_file(model)
    manifest["best_pack_sha256"] = sha256_file(pack)
    manifest["development_qualified"] = True

    best = work / "best"
    best.mkdir(parents=True, exist_ok=True)
    for source_path, name in (
        (model, "model.kwm"),
        (checkpoint, "model.pt"),
        (pack, "keywords.kwk"),
        (calibrated, "keywords.tsv"),
        (provenance, "model-provenance.json"),
        (adversarial_evidence, "adversarial-lexicon.json"),
        (failure_evidence, "development-failure-replay.json"),
    ):
        shutil.copy2(source_path, best / name)

    canonical_qual_base, canonical_qual_domains = evaluate(
        runner=runner,
        model=best / "model.kwm",
        pack=best / "keywords.kwk",
        references=development_qualification / "qualification.references.jsonl",
        output=best / "qualification",
    )
    qualification_qualified = _strict(canonical_qual_base, canonical_qual_domains, gates)
    manifest["qualification"] = canonical_qual_base
    manifest["qualification_domains"] = canonical_qual_domains
    manifest["qualification_qualified"] = qualification_qualified
    manifest["qualified"] = bool(qualification_qualified)
    manifest["evidence_class"] = (
        "synthetic-domain-qualified" if qualification_qualified else "synthetic-domain-unqualified"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 3,
        "policy": POLICY,
        "qualified": bool(qualification_qualified),
        "input_development_manifest_sha256": input_manifest_sha,
        "output_development_manifest_sha256": sha256_file(manifest_path),
        "source_round": source_round,
        "source_selection_policy": source_selection_policy,
        "source_was_strict": source_was_strict,
        "refinement_round": refinement_round,
        "frontend": frontend,
        "adversarial_data_augmentation_policy": adversarial_data_policy,
        "adversarial_selection_policy": adversarial_selection_policy,
        "adversarial_evidence_sha256": sha256_file(adversarial_evidence),
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_probes_per_sequence": int(adversarial["probes_per_sequence"]),
        "adversarial_replay_examples_per_sequence": int(adversarial["replay_examples_per_sequence"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "adversarial_min_per_keyword": int(adversarial["min_per_keyword"]),
        "adversarial_per_keyword_selected": dict(adversarial["per_keyword_selected"]),
        "strict_prefix_anchors": list(adversarial["strict_prefix_anchors"]),
        "failure_replay_policy": str(failure["policy"]),
        "failure_replay_evidence_sha256": sha256_file(failure_evidence),
        "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
        "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
        "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
        "failure_replay_examples": int(failure["examples"]),
        "wake_balance": wake_balance,
        "failure_replay_development_source_wav_bytes_copied": False,
        "qualification_repair_used": qualification_repair is not None,
        "qualification_repair": qualification_repair,
        "development_qualification_validation_seed": validation_seed,
        "development_qualification_validation_config_sha256": sha256_file(validation_config),
        "development_qualification_validation_used_for_training": False,
        "formal_qualification_used": False,
        "record": record,
        "development_qualification": canonical_qual_base,
    }
    out = work / "adversarial-refinement" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualification_qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
