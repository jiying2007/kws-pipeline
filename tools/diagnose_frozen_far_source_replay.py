#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from synthetic_audio import clamp16  # noqa: E402

SCHEMA_VERSION = 1
EVIDENCE_CLASS = "frozen-far-source-replay-request-v1"
RESULT_CLASS = "frozen-far-source-replay-evidence-v1"
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,111}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def normalize_spec(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen FAR replay spec must be an object")
    required = {
        "schema_version",
        "evidence_class",
        "experiment_id",
        "development_only",
        "selection_feedback_allowed",
        "protected_evidence_used",
        "historical_run_id",
        "historical_artifact_id",
        "historical_head_sha",
        "expected_model_sha256",
        "expected_keyword_pack_sha256",
        "cases",
    }
    if set(value) != required:
        raise ValueError(
            "frozen FAR replay spec fields mismatch: "
            f"missing={sorted(required-set(value))} extra={sorted(set(value)-required)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("frozen FAR replay schema_version must be 1")
    if value["evidence_class"] != EVIDENCE_CLASS:
        raise ValueError("frozen FAR replay evidence_class mismatch")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or ID_RE.fullmatch(experiment_id) is None:
        raise ValueError("frozen FAR replay experiment_id is invalid")
    if value["development_only"] is not True:
        raise ValueError("frozen FAR replay must be development_only")
    if value["selection_feedback_allowed"] is not False:
        raise ValueError("frozen FAR replay must not feed selection")
    if value["protected_evidence_used"] is not False:
        raise ValueError("frozen FAR replay must not use protected evidence")
    for key in ("historical_run_id", "historical_artifact_id"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    historical_head = value["historical_head_sha"]
    if (
        not isinstance(historical_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", historical_head) is None
    ):
        raise ValueError("historical_head_sha must be a lowercase git SHA")
    for key in ("expected_model_sha256", "expected_keyword_pack_sha256"):
        if not isinstance(value[key], str) or SHA256_RE.fullmatch(value[key]) is None:
            raise ValueError(f"{key} must be lowercase SHA256")
    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("frozen FAR replay cases must be non-empty")
    normalized_cases: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(cases):
        required_case = {
            "case_id",
            "seed",
            "detection_time_s",
            "historical_confidence",
            "detected_keyword_id",
            "rendered_wav_sha256",
            "historical_gain",
            "expected_family_id",
            "expected_tokens",
        }
        if not isinstance(row, dict) or set(row) != required_case:
            raise ValueError(f"case[{index}] fields mismatch")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or ID_RE.fullmatch(case_id) is None:
            raise ValueError(f"case[{index}].case_id is invalid")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        seed = row["seed"]
        keyword_id = row["detected_keyword_id"]
        family_id = row["expected_family_id"]
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in (seed, keyword_id, family_id)):
            raise ValueError(f"case[{index}] integer identity is invalid")
        detection_time = float(row["detection_time_s"])
        confidence = float(row["historical_confidence"])
        gain = float(row["historical_gain"])
        if not math.isfinite(detection_time) or detection_time < 0.0:
            raise ValueError(f"case[{index}].detection_time_s is invalid")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"case[{index}].historical_confidence is invalid")
        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError(f"case[{index}].historical_gain is invalid")
        rendered_sha = row["rendered_wav_sha256"]
        if not isinstance(rendered_sha, str) or SHA256_RE.fullmatch(rendered_sha) is None:
            raise ValueError(f"case[{index}].rendered_wav_sha256 is invalid")
        tokens = row["expected_tokens"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) or not token for token in tokens)
        ):
            raise ValueError(f"case[{index}].expected_tokens is invalid")
        normalized_cases.append(
            {
                "case_id": case_id,
                "seed": seed,
                "detection_time_s": detection_time,
                "historical_confidence": confidence,
                "detected_keyword_id": keyword_id,
                "rendered_wav_sha256": rendered_sha,
                "historical_gain": gain,
                "expected_family_id": family_id,
                "expected_tokens": list(tokens),
            }
        )
    result = dict(value)
    result["cases"] = normalized_cases
    return result


def scale_wav(source: pathlib.Path, gain: float, output: pathlib.Path) -> dict:
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"source WAV must be mono 16-kHz PCM16: {source}")
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if len(raw) != frames * 2:
        raise ValueError("source WAV is truncated")
    import struct

    values = struct.unpack("<" + "h" * frames, raw)
    scaled = [clamp16(value * gain) for value in values]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<" + "h" * len(scaled), *scaled))
    return {
        "frames": frames,
        "seconds": frames / 16000.0,
        "source_sha256": sha256_file(source),
        "scaled_sha256": sha256_file(output),
        "gain": gain,
    }


def run_json_lines(command: list[str]) -> list[dict]:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-4000:]
        )
    rows: list[dict] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("diagnostic command emitted non-object JSON")
        rows.append(value)
    return rows


def detection_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if isinstance(row.get("keyword_id"), int)
        and isinstance(row.get("time_s"), (int, float))
        and isinstance(row.get("confidence"), (int, float))
    ]


def canonical_detections(rows: list[dict]) -> list[tuple[int, float, float]]:
    return [
        (
            int(row["keyword_id"]),
            round(float(row["time_s"]), 6),
            round(float(row["confidence"]), 6),
        )
        for row in detection_rows(rows)
    ]


def find_domain_row(rows: list[dict], case: dict) -> dict:
    matches = [
        row for row in rows
        if row.get("wav_sha256") == case["rendered_wav_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{case['case_id']}: expected exactly one rendered domain row, got {len(matches)}"
        )
    row = matches[0]
    if int(row.get("family_id", -1)) != case["expected_family_id"]:
        raise ValueError(f"{case['case_id']}: domain family_id drifted")
    if row.get("tokens") != case["expected_tokens"]:
        raise ValueError(f"{case['case_id']}: domain tokens drifted")
    path = pathlib.Path(str(row.get("path", ""))).resolve()
    if not path.is_file():
        raise ValueError(f"{case['case_id']}: rendered WAV is missing: {path}")
    if sha256_file(path) != case["rendered_wav_sha256"]:
        raise ValueError(f"{case['case_id']}: rendered WAV SHA drifted")
    result = dict(row)
    result["path"] = str(path)
    return result


def target_path_summary(rows: list[dict], keyword_id: int) -> dict:
    if len(rows) != 1:
        raise ValueError("decoder path replay must emit exactly one JSON object")
    value = rows[0]
    keywords = value.get("keywords")
    if not isinstance(keywords, list):
        raise ValueError("decoder path replay lacks keyword rows")
    matches = [
        row for row in keywords
        if isinstance(row, dict) and int(row.get("keyword_id", -1)) == keyword_id
    ]
    if len(matches) != 1:
        raise ValueError("decoder path replay target keyword is missing or duplicated")
    return matches[0]


def run_replay(
    *,
    spec: dict,
    model: pathlib.Path,
    pack: pathlib.Path,
    domain_index: pathlib.Path,
    runner: pathlib.Path,
    posterior_dump: pathlib.Path,
    decoder_replay: pathlib.Path,
    decoder_path_replay: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    if sha256_file(model) != spec["expected_model_sha256"]:
        raise ValueError("frozen replay model SHA mismatch")
    if sha256_file(pack) != spec["expected_keyword_pack_sha256"]:
        raise ValueError("frozen replay keyword-pack SHA mismatch")
    domain_rows = load_jsonl(domain_index)
    results: list[dict] = []
    for case in spec["cases"]:
        domain = find_domain_row(domain_rows, case)
        case_root = output_dir / case["case_id"]
        scaled_wav = case_root / "historical-gain-source.wav"
        scale = scale_wav(
            pathlib.Path(domain["path"]),
            case["historical_gain"],
            scaled_wav,
        )
        recording = case["case_id"]
        direct_rows = run_json_lines(
            [str(runner), str(model), str(pack), str(scaled_wav), recording]
        )
        trace = case_root / "historical-gain-source.kwtr"
        posterior_rows = run_json_lines(
            [str(posterior_dump), str(model), str(scaled_wav), str(trace)]
        )
        replay_rows = run_json_lines(
            [
                str(decoder_replay),
                str(model),
                str(pack),
                str(trace),
                recording,
            ]
        )
        path_rows = run_json_lines(
            [
                str(decoder_path_replay),
                str(model),
                str(pack),
                str(trace),
                recording,
            ]
        )
        direct = canonical_detections(direct_rows)
        replay = canonical_detections(replay_rows)
        parity = direct == replay
        if not parity:
            raise ValueError(
                f"{case['case_id']}: direct runtime and fixed-posterior replay differ: "
                f"{direct} != {replay}"
            )
        keyword_id = case["detected_keyword_id"]
        target_hits = [row for row in direct if row[0] == keyword_id]
        classification = (
            "standalone-source-sufficient-under-historical-gain"
            if target_hits
            else "standalone-source-not-sufficient-mixture-or-stream-context-required"
        )
        result = {
            "case": case,
            "domain": {
                key: domain[key]
                for key in (
                    "kind",
                    "family_id",
                    "keyword_id",
                    "tokens",
                    "target_ids",
                    "wav_sha256",
                    "source_wav_sha256",
                    "scene_seed",
                    "scene",
                    "domain_id",
                    "split",
                )
                if key in domain
            },
            "scale": scale,
            "direct_detections": direct_rows,
            "posterior_dump_receipts": posterior_rows,
            "replay_detections": replay_rows,
            "direct_replay_exact_parity": True,
            "target_keyword_standalone_hits": len(target_hits),
            "target_keyword_path": target_path_summary(path_rows, keyword_id),
            "classification": classification,
        }
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    output = {
        "schema_version": 1,
        "evidence_class": RESULT_CLASS,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "training_changed": False,
        "decoder_math_changed": False,
        "thresholds_changed": False,
        "model_sha256": sha256_file(model),
        "keyword_pack_sha256": sha256_file(pack),
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="frozen-far-source-replay-") as td:
        root = pathlib.Path(td)
        wav = root / "source.wav"
        import struct

        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(
                struct.pack("<hhhh", 1000, -1000, 32767, -32768)
            )
        scaled = root / "scaled.wav"
        receipt = scale_wav(wav, 0.5, scaled)
        assert receipt["frames"] == 4
        with wave.open(str(scaled), "rb") as reader:
            values = struct.unpack("<hhhh", reader.readframes(4))
        assert values == (500, -500, 16384, -16384)

        spec_path = root / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_class": EVIDENCE_CLASS,
                    "experiment_id": "fixture-frozen-far-source-replay",
                    "development_only": True,
                    "selection_feedback_allowed": False,
                    "protected_evidence_used": False,
                    "historical_run_id": 1,
                    "historical_artifact_id": 2,
                    "historical_head_sha": "d" * 40,
                    "expected_model_sha256": "a" * 64,
                    "expected_keyword_pack_sha256": "b" * 64,
                    "cases": [
                        {
                            "case_id": "seed-1103-fixture",
                            "seed": 1103,
                            "detection_time_s": 1.25,
                            "historical_confidence": 0.5,
                            "detected_keyword_id": 2,
                            "rendered_wav_sha256": "c" * 64,
                            "historical_gain": 0.9,
                            "expected_family_id": 3,
                            "expected_tokens": ["xiao3", "wo1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        normalized = normalize_spec(spec_path)
        assert normalized["cases"][0]["historical_gain"] == 0.9
        assert canonical_detections(
            [{"keyword_id": 2, "time_s": 1.2345678, "confidence": 0.8765432}]
        ) == [(2, 1.234568, 0.876543)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True, type=pathlib.Path)

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True, type=pathlib.Path)
    run.add_argument("--model", required=True, type=pathlib.Path)
    run.add_argument("--pack", required=True, type=pathlib.Path)
    run.add_argument("--domain-index", required=True, type=pathlib.Path)
    run.add_argument("--runner", required=True, type=pathlib.Path)
    run.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    run.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    run.add_argument("--decoder-path-replay", required=True, type=pathlib.Path)
    run.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("frozen FAR source replay self-test: PASS")
        return 0
    if args.command == "verify":
        value = normalize_spec(args.spec.resolve())
        print(json.dumps(value, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        spec = normalize_spec(args.spec.resolve())
        value = run_replay(
            spec=spec,
            model=args.model.resolve(),
            pack=args.pack.resolve(),
            domain_index=args.domain_index.resolve(),
            runner=args.runner.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            decoder_path_replay=args.decoder_path_replay.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": row["case"]["case_id"],
                            "classification": row["classification"],
                            "target_keyword_standalone_hits": row[
                                "target_keyword_standalone_hits"
                            ],
                        }
                        for row in value["cases"]
                    ]
                },
                sort_keys=True,
            )
        )
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
