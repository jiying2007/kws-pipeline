#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_hashes(path: pathlib.Path, allowed_splits: set[str]) -> set[str]:
    hashes: set[str] = set()
    if not path.is_file():
        return hashes
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        if str(row.get("split")) not in allowed_splits:
            continue
        digest = str(row.get("wav_sha256") or "")
        if len(digest) != 64:
            raise ValueError(f"{path}:{line_no}: missing WAV SHA256")
        hashes.add(digest)
    return hashes


def _manifest_hashes(path: pathlib.Path) -> set[str]:
    hashes: set[str] = set()
    if not path.is_file():
        return hashes
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        wav_text = raw.split("\t", 1)[0].strip()
        if not wav_text:
            raise ValueError(f"{path}:{line_no}: missing WAV path")
        wav = pathlib.Path(wav_text)
        if not wav.is_absolute():
            wav = (path.parent / wav).resolve()
        else:
            wav = wav.resolve()
        if not wav.is_file():
            raise ValueError(f"{path}:{line_no}: missing WAV: {wav}")
        hashes.add(sha256_file(wav))
    return hashes


def _reference_stats(path: pathlib.Path) -> tuple[int, int]:
    recordings = 0
    expected_wakes = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        expected = row.get("expected")
        if not isinstance(expected, list):
            raise ValueError(f"{path}:{line_no}: expected must be a list")
        recordings += 1
        expected_wakes += len(expected)
    return recordings, expected_wakes


def normalize_retired_qualification_seeds(
    raw: object,
    *,
    training_seed: int,
    qualification_seed: int,
) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("retired_qualification_holdout_seeds must be a list")
    seeds: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"retired_qualification_holdout_seeds[{index}] must be an integer"
            )
        if value < 0:
            raise ValueError("retired qualification seeds must be >= 0")
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError("retired qualification seeds must be unique")
    if training_seed in seeds:
        raise ValueError("training seed must not be listed as a retired qualification seed")
    if qualification_seed in seeds:
        raise ValueError("active qualification seed is also marked retired")
    return seeds


def _qualification_render_config(cfg: dict, seed: int) -> dict:
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
    # Qualification scene seeds depend on the global base-row index and the
    # qualification-local evaluation ordinal, not on how many domain scenes
    # were rendered for earlier splits. Render only one scene per non-holdout
    # base row to keep retired-seed SHA reconstruction bounded while preserving
    # byte-identical qualification WAV generation.
    for split in ("train", "calibration", "test"):
        if split not in scenes:
            raise ValueError(f"domains.scenes_per_example.{split} must be configured")
        scenes[split] = 1
    return rendered


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _retired_render_worker_count(cfg: dict, retired_seeds: list[int]) -> int:
    if not retired_seeds:
        return 0
    domains = cfg.get("domains", {})
    afe = domains.get("afe", {}) if isinstance(domains, dict) else {}
    backend = str(afe.get("backend", "proxy")) if isinstance(afe, dict) else "proxy"
    # Command AFE implementations may have hidden shared-device/process state.
    # Preserve the historical serial execution contract for those backends.
    if backend != "proxy":
        return 1
    return min(2, len(retired_seeds))


def _render_retired_seed(cfg: dict, scratch: pathlib.Path, retired_seed: int) -> dict:
    seed_root = scratch / f"seed-{retired_seed}"
    seed_config = seed_root / "effective-config.json"
    retired_config = _qualification_render_config(cfg, retired_seed)
    _write_json(seed_config, retired_config)
    retired_output = seed_root / "dataset"
    retired_summary = render_domain_dataset(
        seed_config, retired_output, curriculum_weights=None
    )
    exposed_index = retired_output / "domain-index.jsonl"
    exposed_references = retired_output / "qualification.references.jsonl"
    exposed_hashes = _index_hashes(exposed_index, {"qualification"})
    exposed_recordings, exposed_expected_wakes = _reference_stats(exposed_references)
    return {
        "seed": retired_seed,
        "wav_hashes": sorted(exposed_hashes),
        "recordings": exposed_recordings,
        "expected_wakes": exposed_expected_wakes,
        "effective_config_sha256": sha256_file(seed_config),
        "domain_index_sha256": str(retired_summary["domain_index_sha256"]),
        "references_sha256": sha256_file(exposed_references),
        "summary_references_sha256": str(
            retired_summary["splits"]["qualification"]["references_sha256"]
        ),
    }


def require_strict_development_candidate(work: pathlib.Path) -> dict:
    manifest_path = work / "domain-loop-manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("development manifest is missing; refuse formal qualification")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("development manifest must be an object")

    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("development manifest records are missing")
    eligible = [
        row
        for row in records
        if isinstance(row, dict)
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not eligible:
        raise ValueError(
            "no calibration/test strict development candidate; refuse formal qualification"
        )
    latest_round = max(int(row["round"]) for row in eligible)
    latest = [row for row in eligible if int(row["round"]) == latest_round]
    recomputed = min(
        latest,
        key=lambda row: (float(row["score"]), str(row["frontend"])),
    )
    recomputed_rounds = sorted({int(row["round"]) for row in eligible})

    if not bool(manifest.get("development_qualified")):
        raise ValueError("development_qualified disagrees with strict development records")
    selection = manifest.get("candidate_selection")
    if not isinstance(selection, dict):
        raise ValueError("development candidate-selection evidence is missing")
    if str(selection.get("policy")) != "latest-strict-gate-passing-round":
        raise ValueError("development candidate-selection policy is not strict/latest")
    eligible_rounds = selection.get("eligible_rounds")
    if not isinstance(eligible_rounds, list):
        raise ValueError("development manifest strict eligible rounds are missing")
    try:
        manifest_rounds = sorted(int(value) for value in eligible_rounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("development manifest strict eligible rounds are invalid") from exc
    if manifest_rounds != recomputed_rounds:
        raise ValueError("development strict eligible rounds disagree with records")

    selected_round = selection.get("selected_round")
    selected_frontend = selection.get("selected_frontend")
    if isinstance(selected_round, bool) or not isinstance(selected_round, int):
        raise ValueError("development selected round is missing")
    if not isinstance(selected_frontend, str) or not selected_frontend:
        raise ValueError("development selected frontend is missing")
    if selected_round != int(recomputed["round"]):
        raise ValueError("development selected round disagrees with strict records")
    if selected_frontend != str(recomputed["frontend"]):
        raise ValueError("development selected frontend disagrees with strict records")
    if "selected_score" in selection and float(selection["selected_score"]) != float(
        recomputed["score"]
    ):
        raise ValueError("development selected score disagrees with strict records")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retire exposed qualification cohorts and render a seed-disjoint replacement."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    work = args.work_dir.resolve()
    output = args.output.resolve()
    cfg = load_config(config_path)
    development_manifest = require_strict_development_candidate(work)
    development_manifest_path = work / "domain-loop-manifest.json"
    training_seed = int(cfg.get("seed", 1337))
    qualification_seed = int(cfg.get("qualification_holdout_seed", -1))
    if qualification_seed < 0:
        raise ValueError("qualification_holdout_seed must be configured")
    if qualification_seed == training_seed:
        raise ValueError("qualification_holdout_seed must differ from training seed")
    retired_exposed_seeds = normalize_retired_qualification_seeds(
        cfg.get("retired_qualification_holdout_seeds", []),
        training_seed=training_seed,
        qualification_seed=qualification_seed,
    )

    retired_index = output / "domain-index.jsonl"
    retired_references = output / "qualification.references.jsonl"
    if not retired_index.is_file() or not retired_references.is_file():
        raise ValueError("retired qualification cohort is missing; rotate only after training")
    retired_hashes = _index_hashes(retired_index, {"qualification"})
    if not retired_hashes:
        raise ValueError("retired qualification cohort has no qualification WAVs")
    retired_references_sha = sha256_file(retired_references)

    development_hashes: set[str] = set()
    for index_path in sorted((work / "datasets").glob("**/domain-index.jsonl")):
        development_hashes.update(
            _index_hashes(index_path, {"train", "calibration", "test"})
        )
    for manifest in sorted((work / "hard-negative-replay").glob("round-*/hard-negatives.tsv")):
        development_hashes.update(_manifest_hashes(manifest))
    if not development_hashes:
        raise ValueError("no development/training WAV evidence found")

    rotated = _qualification_render_config(cfg, qualification_seed)
    effective_config = work / "qualification-effective-config.json"
    _write_json(effective_config, rotated)

    retired_exposed: list[dict] = []
    scratch = work / "retired-qualification-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    worker_count = _retired_render_worker_count(cfg, retired_exposed_seeds)
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    futures: list[concurrent.futures.Future] = []
    try:
        if worker_count > 1:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
            futures = [
                executor.submit(_render_retired_seed, cfg, scratch, retired_seed)
                for retired_seed in retired_exposed_seeds
            ]

        # Active qualification rendering is independent of every retired seed.
        # Overlap it with proxy-AFE retired reconstruction so governance history
        # does not consume the fixed workflow wall-clock budget serially.
        shutil.rmtree(output)
        summary = render_domain_dataset(effective_config, output, curriculum_weights=None)
        active_index = output / "domain-index.jsonl"
        active_references = output / "qualification.references.jsonl"
        active_hashes = _index_hashes(active_index, {"qualification"})
        if not active_hashes:
            raise ValueError("active qualification cohort has no qualification WAVs")
        active_references_sha = sha256_file(active_references)
        retired_overlap = retired_hashes & active_hashes
        development_overlap = development_hashes & active_hashes
        if retired_overlap:
            raise ValueError(
                f"active qualification overlaps {len(retired_overlap)} retired qualification WAV SHA(s)"
            )
        if development_overlap:
            raise ValueError(
                f"active qualification overlaps {len(development_overlap)} development/training WAV SHA(s)"
            )
        if active_references_sha == retired_references_sha:
            raise ValueError("active qualification references are byte-identical to retired references")

        recordings, expected_wakes = _reference_stats(active_references)
        if futures:
            rendered_retired = [future.result() for future in futures]
        else:
            rendered_retired = [
                _render_retired_seed(cfg, scratch, retired_seed)
                for retired_seed in retired_exposed_seeds
            ]

        for rendered in rendered_retired:
            retired_seed = int(rendered["seed"])
            exposed_hashes = set(str(value) for value in rendered["wav_hashes"])
            if len(exposed_hashes) != len(active_hashes):
                raise ValueError(
                    "retired/active qualification WAV-count mismatch: "
                    f"seed={retired_seed} retired={len(exposed_hashes)} "
                    f"active={len(active_hashes)}"
                )
            overlap = exposed_hashes & active_hashes
            if overlap:
                raise ValueError(
                    f"active qualification overlaps {len(overlap)} WAV SHA(s) with "
                    f"retired qualification seed {retired_seed}"
                )
            exposed_references_sha = str(rendered["references_sha256"])
            if exposed_references_sha == active_references_sha:
                raise ValueError(
                    f"active qualification references are byte-identical to retired seed {retired_seed}"
                )
            exposed_recordings = int(rendered["recordings"])
            exposed_expected_wakes = int(rendered["expected_wakes"])
            if exposed_recordings != recordings or exposed_expected_wakes != expected_wakes:
                raise ValueError(
                    f"retired qualification seed {retired_seed} has different support: "
                    f"recordings={exposed_recordings}/{recordings} "
                    f"expected={exposed_expected_wakes}/{expected_wakes}"
                )
            if str(rendered["summary_references_sha256"]) != exposed_references_sha:
                raise ValueError(
                    f"retired qualification seed {retired_seed} reference SHA mismatch"
                )
            retired_exposed.append(
                {
                    "seed": retired_seed,
                    "wav_count": len(exposed_hashes),
                    "recordings": exposed_recordings,
                    "expected_wakes": exposed_expected_wakes,
                    "effective_config_sha256": str(rendered["effective_config_sha256"]),
                    "domain_index_sha256": str(rendered["domain_index_sha256"]),
                    "references_sha256": exposed_references_sha,
                    "overlapping_active_wav_sha256": 0,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)

    selection = development_manifest["candidate_selection"]
    evidence = {
        "schema_version": 2,
        "policy": "retire-exposed-qualification-and-rotate-seed-v2",
        "source_config_sha256": sha256_file(config_path),
        "effective_config_sha256": sha256_file(effective_config),
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "development_candidate_policy": str(selection["policy"]),
        "development_selected_round": int(selection["selected_round"]),
        "development_selected_frontend": str(selection["selected_frontend"]),
        "training_seed": training_seed,
        "qualification_seed": qualification_seed,
        "retired_exposed_qualification_seeds": retired_exposed_seeds,
        "retired_exposed_qualification": retired_exposed,
        "seed_disjoint": True,
        "generated_after_training": True,
        "strict_development_candidate_required": True,
        "retired_references_sha256": retired_references_sha,
        "active_references_sha256": active_references_sha,
        "retired_qualification_wav_count": len(retired_hashes),
        "active_qualification_wav_count": len(active_hashes),
        "development_training_wav_count": len(development_hashes),
        "overlapping_retired_active_wav_sha256": 0,
        "overlapping_exposed_active_wav_sha256": 0,
        "overlapping_development_active_wav_sha256": 0,
        "recordings": recordings,
        "expected_wakes": expected_wakes,
        "domain_index_sha256": str(summary["domain_index_sha256"]),
        "qualification_references_sha256": str(
            summary["splits"]["qualification"]["references_sha256"]
        ),
    }
    if evidence["qualification_references_sha256"] != active_references_sha:
        raise ValueError("domain summary qualification reference SHA does not match active cohort")
    evidence_path = output / "qualification-cohort.json"
    _write_json(evidence_path, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
