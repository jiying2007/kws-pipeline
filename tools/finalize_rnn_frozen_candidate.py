#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

SELECTION_POLICY = "best-strict-development-objective-round"
FREEZE_POLICY = "rnn-frozen-candidate-v1"
SOURCE_POLICY = "rnn-development-curriculum-loop-v1"
STABILITY_EVIDENCE_POLICY = "rnn-development-stability-evidence-v1"
ARCHITECTURE = "tiny-streaming-rnn-v1"


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


def gate_bool(record: dict, key: str) -> bool:
    """Read a gate that is already a boolean.

    bool("false") is True, so coercing here would count a round that did not
    pass: it inflates the strict-pass streak, selects a failing round as the
    stable candidate, and writes a pass into the frozen manifest. The producers
    write real booleans, so require them.
    """
    value = record.get(key)
    if value is None:
        # A round with no recorded gate did not pass. That is a fact, not
        # an error, and it must not be coerced into a pass either.
        return False
    if not isinstance(value, bool):
        raise ValueError("RNN development record " + key + " must be a boolean")
    return value


def terminal_strict_streak(records: object) -> int:
    if not isinstance(records, list) or not records:
        raise ValueError("RNN development records must be a non-empty list")
    streak = 0
    for expected_round, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("RNN development record must be an object")
        round_value = row.get("round")
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("RNN development record round must be an integer")
        if round_value != expected_round:
            raise ValueError("RNN development records must be contiguous starting at zero")
        if gate_bool(row, "calibration_gate") and gate_bool(row, "test_gate"):
            streak += 1
        else:
            streak = 0
    return streak


def round_gate_evidence(records: object) -> list[dict]:
    if not isinstance(records, list) or not records:
        raise ValueError("RNN development records must be a non-empty list")
    result: list[dict] = []
    for expected_round, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("RNN development record must be an object")
        round_value = row.get("round")
        # True == 1 and 1.0 == 1, so a bare != accepts both as a round number
        # and the record is silently rewritten with the index instead of being
        # rejected. The GRU twin and the stability verifier both check the type.
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("RNN development record round must be an integer")
        if round_value != expected_round:
            raise ValueError("RNN development round evidence is malformed")
        calibration = row.get("calibration_gate")
        test = row.get("test_gate")
        # bool("false") is True, so a coercion here turns a recorded failure
        # into a pass. The producers write real booleans, so require them.
        if not isinstance(calibration, bool) or not isinstance(test, bool):
            raise ValueError("RNN development gate values must be booleans")
        result.append({
            "round": round_value,
            "calibration_gate": calibration,
            "test_gate": test,
        })
    return result


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
        wav = (path.parent / wav).resolve() if not wav.is_absolute() else wav.resolve()
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
        raise ValueError("RNN development selection policy mismatch")
    selected_round = manifest.get("selected_round")
    if isinstance(selected_round, bool) or not isinstance(selected_round, int):
        raise ValueError("RNN development manifest does not contain a selected round")
    candidates = [
        row for row in manifest.get("records", [])
        if isinstance(row, dict) and gate_bool(row, "calibration_gate") and gate_bool(row, "test_gate")
    ]
    if not candidates:
        raise ValueError("cannot resolve selected strict RNN development record")
    selected = min(candidates, key=lambda row: (float(row["score"]), -int(row["round"]), str(row.get("frontend", ""))))
    if int(selected["round"]) != selected_round:
        raise ValueError("selected RNN round is not best strict development objective")
    if float(selected["score"]) != float(manifest.get("selected_score")):
        raise ValueError("selected RNN score mismatch")
    return selected


def finalize(work: pathlib.Path, config: pathlib.Path, policy: pathlib.Path) -> dict:
    work = work.resolve(); config = config.resolve(); policy = policy.resolve()
    frozen = work / "frozen-candidate"
    freeze_path = frozen / "freeze-manifest.json"
    development_path = work / "development-loop-manifest.json"
    if not freeze_path.is_file() or not development_path.is_file():
        raise ValueError("RNN development loop did not produce frozen candidate evidence")
    freeze = load_object(freeze_path)
    development = load_object(development_path)
    source_policy = load_object(policy)
    if freeze.get("policy") != FREEZE_POLICY or freeze.get("source_policy") != SOURCE_POLICY:
        raise ValueError("unexpected RNN freeze policy")
    if freeze.get("model_family") != "rnn" or freeze.get("architecture") != ARCHITECTURE:
        raise ValueError("unexpected RNN frozen model identity")
    if development.get("policy") != SOURCE_POLICY or development.get("model_family") != "rnn":
        raise ValueError("unexpected RNN development identity")
    if freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("unexpected RNN selection policy")
    candidate_freeze = source_policy.get("candidate_freeze")
    if not isinstance(candidate_freeze, dict) or candidate_freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("RNN source development selection policy mismatch")
    if sha256_file(config) != str(freeze.get("config_sha256", "")):
        raise ValueError("RNN source config SHA drifted before finalization")
    if sha256_file(policy) != str(freeze.get("development_policy_sha256", "")):
        raise ValueError("RNN source policy SHA drifted before finalization")

    required = source_policy.get("stable_strict_pass_rounds")
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise ValueError("stable_strict_pass_rounds must be positive")
    observed = terminal_strict_streak(development.get("records"))
    if observed < required:
        development["development_qualified"] = False
        development["selected_round"] = None
        development["selected_score"] = None
        development_path.write_text(json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        shutil.rmtree(frozen, ignore_errors=True)
        raise ValueError(f"RNN candidate lacks stable strict-pass streak: observed={observed} required={required}")
    development["stable_strict_pass_rounds_required"] = required
    development["stable_strict_pass_rounds_observed"] = observed
    development["development_qualified"] = True
    development_path.write_text(json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    selected = select_record(development)
    evidence = {
        "schema_version": 1,
        "evidence_class": "rnn-frozen-development-selection",
        "model_family": "rnn",
        "architecture": ARCHITECTURE,
        "source_policy": SOURCE_POLICY,
        "selection_policy": SELECTION_POLICY,
        "selected_round": int(selected["round"]),
        "selected_frontend": str(selected.get("frontend", "logmel")),
        "selected_score": float(selected["score"]),
        "model_sha256": str(selected["model_sha256"]),
        "calibration_gate": gate_bool(selected, "calibration_gate"),
        "test_gate": gate_bool(selected, "test_gate"),
        "calibration": compact_base(dict(selected["calibration"])),
        "calibration_domains": selected["calibration_domains"],
        "test": compact_base(dict(selected["test"])),
        "test_domains": selected["test_domains"],
        "stable_strict_pass_rounds_required": required,
        "stable_strict_pass_rounds_observed": observed,
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    evidence_path = frozen / "selection-evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    stability = {
        "schema_version": 1,
        "policy": STABILITY_EVIDENCE_POLICY,
        "source_policy": SOURCE_POLICY,
        "model_family": "rnn",
        "development_only": True,
        "stable_strict_pass_rounds_required": required,
        "stable_strict_pass_rounds_observed": observed,
        "round_gates": round_gate_evidence(development.get("records")),
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    stability_path = frozen / "stability-evidence.json"
    stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    shutil.copy2(config, frozen / "source-config.json")
    shutil.copy2(policy, frozen / "source-development-policy.json")
    hashes: set[str] = set()
    for index in sorted((work / "datasets").glob("round-*/domain-index.jsonl")):
        hashes.update(development_index_hashes(index))
    for root_name in ("fixed-replay", "failure-replay"):
        for manifest_path in sorted((work / root_name).glob("**/*.tsv")):
            hashes.update(manifest_wav_hashes(manifest_path))
    if not hashes:
        raise ValueError("frozen RNN candidate has no development/training WAV identity evidence")
    corpus = {
        "schema_version": 1,
        "evidence_class": "rnn-frozen-development-wav-identities",
        "model_family": "rnn",
        "development_only": True,
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "wav_sha256_count": len(hashes),
        "wav_sha256": sorted(hashes),
    }
    corpus_path = frozen / "development-wav-sha256.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    freeze["selection_evidence_sha256"] = sha256_file(evidence_path)
    freeze["stability_evidence_sha256"] = sha256_file(stability_path)
    freeze["development_wav_identities_sha256"] = sha256_file(corpus_path)
    freeze["source_config_snapshot_sha256"] = sha256_file(frozen / "source-config.json")
    freeze["source_development_policy_snapshot_sha256"] = sha256_file(frozen / "source-development-policy.json")
    freeze["stable_strict_pass_rounds_required"] = required
    freeze["stable_strict_pass_rounds_observed"] = observed
    freeze["selected_model_matches_selection_evidence"] = str(freeze["model_sha256"]) == str(evidence["model_sha256"])
    if not freeze["selected_model_matches_selection_evidence"]:
        raise ValueError("frozen RNN model does not match selected development record")
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    args = parser.parse_args()
    value = finalize(args.work_dir, args.config, args.policy)
    print(json.dumps({
        "finalized": True,
        "policy": value["policy"],
        "model_family": "rnn",
        "selected_round": value["selected_round"],
        "model_sha256": value["model_sha256"],
        "selection_evidence_sha256": value["selection_evidence_sha256"],
        "stability_evidence_sha256": value["stability_evidence_sha256"],
        "development_wav_identities_sha256": value["development_wav_identities_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
