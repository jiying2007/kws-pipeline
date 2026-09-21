#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = "development-round-resume-v1"
SEGMENT_CONTINUE_EXIT_CODE = 3
CANDIDATE_PATH_FIELDS = ("model", "checkpoint", "provenance", "keywords", "pack")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_sha() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("repository HEAD is not a canonical git SHA")
    return value


def _rounds(records: list[dict]) -> None:
    observed = [int(row.get("round", -1)) for row in records]
    expected = list(range(len(records)))
    if observed != expected:
        raise ValueError(
            f"development resume rounds must be contiguous from zero: {observed}"
        )


def _under_root(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped development work root: {resolved}") from exc
    return resolved


def _optional_record_file_hashes(record: dict, work: pathlib.Path) -> dict:
    result: dict[str, str] = {}
    round_index = int(record["round"])
    domain_index = _under_root(
        work / "datasets" / f"round-{round_index:02d}" / "domain-index.jsonl",
        work,
        "domain-index",
    )
    if not domain_index.is_file() or domain_index.stat().st_size <= 0:
        raise ValueError(f"resume domain index missing: {domain_index}")
    result["domain_index"] = sha256_file(domain_index)
    for split in ("calibration", "test"):
        metrics = record.get(split)
        if not isinstance(metrics, dict):
            raise ValueError(f"resume record missing {split} metrics")
        for field in ("false_rejects_path", "false_positives_path"):
            raw = metrics.get(field)
            if raw is None or raw == "":
                continue
            if not isinstance(raw, str):
                raise ValueError(f"resume {split}.{field} must be a path string")
            item = _under_root(pathlib.Path(raw), work, f"{split}.{field}")
            if not item.is_file():
                raise ValueError(f"resume failure evidence missing: {item}")
            result[f"{split}.{field}"] = sha256_file(item)
    return dict(sorted(result.items()))


def _candidate_hashes(records: list[dict], work: pathlib.Path) -> list[dict]:
    result: list[dict] = []
    for record in records:
        row = {"round": int(record["round"]), "members": {}}
        for field in CANDIDATE_PATH_FIELDS:
            raw = record.get(field)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"resume record missing candidate path: {field}")
            path = _under_root(pathlib.Path(raw), work, field)
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"resume candidate member missing: {path}")
            row["members"][field] = sha256_file(path)
        row["replay_sources"] = _optional_record_file_hashes(record, work)
        result.append(row)
    return result


def _rebase(value, old_root: pathlib.Path, new_root: pathlib.Path):
    if isinstance(value, dict):
        return {key: _rebase(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        old = str(old_root)
        if value == old:
            return str(new_root)
        prefix = old + "/"
        if value.startswith(prefix):
            return str(new_root / value[len(prefix) :])
    return value


def write_state(
    path: pathlib.Path,
    *,
    work: pathlib.Path,
    model_family: str,
    architecture: str,
    source_policy: str,
    config_path: pathlib.Path,
    policy_path: pathlib.Path,
    records: list[dict],
    complete: bool,
) -> dict:
    work = work.resolve()
    _rounds(records)
    candidate_hashes = _candidate_hashes(records, work)
    curriculum_sha = None
    if records:
        curriculum = work / "curriculum" / f"round-{len(records) - 1:02d}.json"
        if not curriculum.is_file():
            raise ValueError("resume state is missing last curriculum")
        curriculum_sha = sha256_file(curriculum)
    value = {
        "schema_version": 1,
        "policy": POLICY,
        "repository_sha": repository_sha(),
        "work_root": str(work),
        "model_family": str(model_family),
        "architecture": str(architecture),
        "source_policy": str(source_policy),
        "config_sha256": sha256_file(config_path),
        "development_policy_sha256": sha256_file(policy_path),
        "next_round": len(records),
        "complete": bool(complete),
        "last_curriculum_sha256": curriculum_sha,
        "candidate_hashes": candidate_hashes,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return value


def load_state(
    path: pathlib.Path,
    *,
    work: pathlib.Path,
    model_family: str,
    architecture: str,
    source_policy: str,
    config_path: pathlib.Path,
    policy_path: pathlib.Path,
) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("development resume state schema mismatch")
    expected = {
        "policy": POLICY,
        "repository_sha": repository_sha(),
        "model_family": str(model_family),
        "architecture": str(architecture),
        "source_policy": str(source_policy),
        "config_sha256": sha256_file(config_path),
        "development_policy_sha256": sha256_file(policy_path),
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"development resume state {key} mismatch")
    old_root = pathlib.Path(str(value.get("work_root", ""))).resolve()
    work = work.resolve()
    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("development resume records must be a list")
    records = _rebase(raw_records, old_root, work)
    if not isinstance(records, list):
        raise ValueError("rebased development resume records are invalid")
    _rounds(records)
    if int(value.get("next_round", -1)) != len(records):
        raise ValueError("development resume next_round drifted")
    expected_hashes = value.get("candidate_hashes")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(records):
        raise ValueError("development resume candidate hashes are incomplete")
    actual_hashes = _candidate_hashes(records, work)
    if actual_hashes != expected_hashes:
        raise ValueError("development resume candidate bytes drifted")
    if records:
        curriculum = work / "curriculum" / f"round-{len(records) - 1:02d}.json"
        if not curriculum.is_file():
            raise ValueError("development resume curriculum is missing")
        if sha256_file(curriculum) != value.get("last_curriculum_sha256"):
            raise ValueError("development resume curriculum bytes drifted")
    # bool("false") is True. The state file is read from disk, so an
    # incomplete resume state must not be treated as a finished loop.
    if value.get("complete") is True:
        manifest = work / "development-loop-manifest.json"
        if not manifest.is_file():
            raise ValueError("complete development resume state lacks final manifest")
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest_value, dict):
            raise ValueError("complete development manifest must be an object")
        rebased_manifest = _rebase(manifest_value, old_root, work)
        manifest.write_text(
            json.dumps(
                rebased_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "records": records,
        "next_round": len(records),
        # bool("false") is True, and the caller branches on this.
        "complete": value.get("complete") is True,
    }


def rebuild_progress(
    records: list[dict],
    policy: dict,
    controller_initial: Callable[[dict], dict],
    controller_next: Callable[..., dict],
    strict_fn: Callable[[dict], bool],
) -> tuple[dict, float | None, int, int]:
    controller = controller_initial(policy)
    best: float | None = None
    stale = 0
    streak = 0
    for record in records:
        score = float(record.get("score"))
        if not math.isfinite(score):
            raise ValueError("development resume score must be finite")
        fr = int(record.get("false_rejects", 0))
        fa = int(record.get("false_accepts", 0))
        frr_values = [
            float(metrics["frr"])
            for split in ("calibration", "test")
            for metrics in [record.get(split)]
            if isinstance(metrics, dict) and isinstance(metrics.get("frr"), (int, float))
        ]
        far_values = [
            float(metrics["far_per_hour"])
            for split in ("calibration", "test")
            for metrics in [record.get(split)]
            if isinstance(metrics, dict)
            and isinstance(metrics.get("far_per_hour"), (int, float))
        ]
        controller = controller_next(
            policy,
            controller,
            fr,
            fa,
            frr=max(frr_values) if frr_values else None,
            far_per_hour=max(far_values) if far_values else None,
        )
        streak = streak + 1 if strict_fn(record) else 0
        if best is None or score < best - 1.0e-12:
            best = score
            stale = 0
        else:
            stale += 1
    return controller, best, stale, streak


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        old = root / "old-work"
        candidate = old / "candidates" / "round-00"
        candidate.mkdir(parents=True)
        members = {}
        for field in CANDIDATE_PATH_FIELDS:
            suffix = {"model": ".kwm", "checkpoint": ".pt", "provenance": ".json", "keywords": ".tsv", "pack": ".kwk"}[field]
            member = candidate / f"{field}{suffix}"
            member.write_bytes((field + "\n").encode())
            members[field] = str(member)
        curriculum = old / "curriculum" / "round-00.json"
        curriculum.parent.mkdir(parents=True)
        curriculum.write_text('{"weight":1}\n', encoding="utf-8")
        domain_index = old / "datasets" / "round-00" / "domain-index.jsonl"
        domain_index.parent.mkdir(parents=True)
        domain_index.write_text('{"split":"test"}\n', encoding="utf-8")
        failures = candidate / "calibration"
        failures.mkdir(parents=True)
        false_rejects = failures / "false-rejects.jsonl"
        false_positives = failures / "false-positives.jsonl"
        false_rejects.write_text('', encoding="utf-8")
        false_positives.write_text('', encoding="utf-8")
        config = root / "config.json"
        policy = root / "policy.json"
        config.write_text('{"x":1}\n', encoding="utf-8")
        policy.write_text('{"y":1}\n', encoding="utf-8")
        record = {
            "round": 0,
            "score": 1.0,
            "false_rejects": 1,
            "false_accepts": 0,
            "calibration_gate": True,
            "test_gate": True,
            "calibration": {
                "false_rejects_path": str(false_rejects),
                "false_positives_path": str(false_positives),
            },
            "test": {},
            **members,
        }
        state = old / "development-resume-state.json"
        write_state(
            state,
            work=old,
            model_family="test",
            architecture="test-v1",
            source_policy="test-policy",
            config_path=config,
            policy_path=policy,
            records=[record],
            complete=False,
        )
        new = root / "new-work"
        shutil.copytree(old, new)
        loaded = load_state(
            new / "development-resume-state.json",
            work=new,
            model_family="test",
            architecture="test-v1",
            source_policy="test-policy",
            config_path=config,
            policy_path=policy,
        )
        assert loaded["next_round"] == 1
        assert str(loaded["records"][0]["checkpoint"]).startswith(str(new))
        pathlib.Path(loaded["records"][0]["model"]).write_bytes(b"tampered")
        try:
            load_state(
                new / "development-resume-state.json",
                work=new,
                model_family="test",
                architecture="test-v1",
                source_policy="test-policy",
                config_path=config,
                policy_path=policy,
            )
        except ValueError as exc:
            assert "candidate bytes drifted" in str(exc)
        else:
            raise AssertionError("tampered resume candidate was accepted")
        shutil.copy2(state, new / "development-resume-state.json")
        # Restore the candidate byte and tamper a replay input instead.
        pathlib.Path(loaded["records"][0]["model"]).write_bytes(b"model\n")
        rebased_index = new / "datasets" / "round-00" / "domain-index.jsonl"
        rebased_index.write_text('{"split":"tampered"}\n', encoding="utf-8")
        try:
            load_state(
                new / "development-resume-state.json",
                work=new,
                model_family="test",
                architecture="test-v1",
                source_policy="test-policy",
                config_path=config,
                policy_path=policy,
            )
        except ValueError as exc:
            assert "candidate bytes drifted" in str(exc)
        else:
            raise AssertionError("tampered resume replay source was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("development round resume self-test: PASS")
        return 0
    parser.error("--self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
