#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTION_POLICY = "best-strict-development-objective-round"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def compact_base(value: dict) -> dict:
    excluded = {"false_positives_path", "false_rejects_path"}
    return {key: item for key, item in value.items() if key not in excluded}


def manifest_wav_hashes(path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    if not path.is_file():
        return result
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
        result.add(sha256_file(wav))
    return result


def development_index_hashes(path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    if not path.is_file():
        return result
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        if str(row.get("split")) not in {"train", "calibration", "test"}:
            continue
        digest = str(row.get("wav_sha256") or "")
        if len(digest) != 64:
            raise ValueError(f"{path}:{line_no}: missing development WAV SHA")
        result.add(digest)
    return result


def select_record(manifest: dict) -> dict:
    if manifest.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("development selection policy mismatch")
    selected_round = manifest.get("selected_round")
    if isinstance(selected_round, bool) or not isinstance(selected_round, int):
        raise ValueError("development manifest does not contain a selected round")
    candidates = [
        row
        for row in manifest.get("records", [])
        if isinstance(row, dict)
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
    ]
    if not candidates:
        raise ValueError("cannot resolve selected strict development record")
    selected = min(
        candidates,
        key=lambda row: (
            float(row["score"]),
            -int(row["round"]),
            str(row.get("frontend", "")),
        ),
    )
    if int(selected["round"]) != selected_round:
        raise ValueError("selected round is not the best strict development objective")
    if float(selected["score"]) != float(manifest.get("selected_score")):
        raise ValueError("selected score does not match the best strict development objective")
    return selected


def finalize(work: pathlib.Path, config: pathlib.Path, policy: pathlib.Path) -> dict:
    work = work.resolve()
    config = config.resolve()
    policy = policy.resolve()
    frozen = work / "frozen-candidate"
    freeze_path = frozen / "freeze-manifest.json"
    development_path = work / "development-loop-manifest.json"
    if not freeze_path.is_file() or not development_path.is_file():
        raise ValueError("development loop did not produce frozen candidate evidence")
    freeze = load_object(freeze_path)
    development = load_object(development_path)
    source_policy = load_object(policy)
    if freeze.get("policy") != "gru-frozen-candidate-v1":
        raise ValueError("unexpected freeze policy")
    if freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("unexpected freeze selection policy")
    if development.get("policy") != "gru-development-curriculum-loop-v1":
        raise ValueError("unexpected development loop policy")
    candidate_freeze = source_policy.get("candidate_freeze")
    if not isinstance(candidate_freeze, dict) or candidate_freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("source development selection policy mismatch")
    if not bool(development.get("development_qualified")):
        raise ValueError("cannot finalize an unqualified development loop")
    if sha256_file(config) != str(freeze.get("config_sha256", "")):
        raise ValueError("source config SHA drifted before freeze finalization")
    if sha256_file(policy) != str(freeze.get("development_policy_sha256", "")):
        raise ValueError("source development policy SHA drifted before freeze finalization")

    selected = select_record(development)
    evidence = {
        "schema_version": 1,
        "evidence_class": "gru-frozen-development-selection",
        "source_policy": "gru-development-curriculum-loop-v1",
        "selection_policy": SELECTION_POLICY,
        "selected_round": int(selected["round"]),
        "selected_frontend": str(selected.get("frontend", "logmel")),
        "selected_score": float(selected["score"]),
        "model_sha256": str(selected["model_sha256"]),
        "calibration_gate": bool(selected["calibration_gate"]),
        "test_gate": bool(selected["test_gate"]),
        "calibration": compact_base(dict(selected["calibration"])),
        "calibration_domains": selected["calibration_domains"],
        "test": compact_base(dict(selected["test"])),
        "test_domains": selected["test_domains"],
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    evidence_path = frozen / "selection-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(config, frozen / "source-config.json")
    shutil.copy2(policy, frozen / "source-development-policy.json")

    hashes: set[str] = set()
    for index in sorted((work / "datasets").glob("round-*/domain-index.jsonl")):
        hashes.update(development_index_hashes(index))
    for root_name in ("fixed-replay", "failure-replay"):
        for manifest_path in sorted((work / root_name).glob("**/*.tsv")):
            hashes.update(manifest_wav_hashes(manifest_path))
    if not hashes:
        raise ValueError("frozen candidate has no development/training WAV identity evidence")
    corpus = {
        "schema_version": 1,
        "evidence_class": "gru-frozen-development-wav-identities",
        "development_only": True,
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "wav_sha256_count": len(hashes),
        "wav_sha256": sorted(hashes),
    }
    corpus_path = frozen / "development-wav-sha256.json"
    corpus_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    freeze["selection_evidence_sha256"] = sha256_file(evidence_path)
    freeze["development_wav_identities_sha256"] = sha256_file(corpus_path)
    freeze["source_config_snapshot_sha256"] = sha256_file(frozen / "source-config.json")
    freeze["source_development_policy_snapshot_sha256"] = sha256_file(
        frozen / "source-development-policy.json"
    )
    freeze["selected_model_matches_selection_evidence"] = (
        str(freeze["model_sha256"]) == str(evidence["model_sha256"])
    )
    if not freeze["selected_model_matches_selection_evidence"]:
        raise ValueError("frozen model does not match selected development record")
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = finalize(args.work_dir, args.config, args.policy)
    print(
        json.dumps(
            {
                "finalized": True,
                "policy": result["policy"],
                "selection_policy": result["selection_policy"],
                "selected_round": result["selected_round"],
                "model_sha256": result["model_sha256"],
                "selection_evidence_sha256": result["selection_evidence_sha256"],
                "development_wav_identities_sha256": result[
                    "development_wav_identities_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
