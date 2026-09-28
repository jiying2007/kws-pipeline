#!/usr/bin/env python3
"""Evaluate external audio-pipeline VAD boundary feasibility on KWS train pairs."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import wave
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from diagnose_acoustic_boundary_segmentation import (  # noqa: E402
    accuracy,
    cross_validate,
    fit_centroid,
)
from diagnose_boundary_separability import (  # noqa: E402
    build as build_gap_separability,
    write_json,
)
from build_decoder_boundary_references import safe_name  # noqa: E402

POLICY = "external-afe-vad-boundary-feasibility-v1"
EVIDENCE_CLASS = "external-afe-vad-boundary-development-v1"
DEFAULT_GAPS_MS = (160, 240, 320, 400, 480)
PROFILES = ("vad-isolated", "ns-isolated")

PROBABILITY_FEATURES = (
    "pre_probability_mean_30ms",
    "pre_probability_last",
    "gap_probability_mean",
    "gap_probability_max",
    "gap_probability_min",
    "post_probability_mean_30ms",
    "post_probability_first",
    "release_delta",
    "attack_delta",
)
ACTIVE_FEATURES = (
    "gap_active_ratio",
    "gap_active_longest_run_frames",
    "gap_active_prefix_frames",
    "gap_active_suffix_frames",
)
FEATURE_SETS = {
    "probability_only": PROBABILITY_FEATURES,
    "probability_plus_active": PROBABILITY_FEATURES + ACTIVE_FEATURES,
}
AP_FRAME_SAMPLES = 160


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_pin(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "evidence_class": "external-afe-vad-source-pin-v1",
        "repository": "jiying2007/audio-pipeline",
        "frame_ms": 10,
        "sample_rate_hz": 16000,
        "development_only": True,
        "selection_feedback_allowed": True,
        "protected_evidence_used": False,
        "release_authority": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"external VAD source pin mismatch: {key}")
    sha = str(value.get("head_sha", ""))
    if len(sha) != 40:
        raise ValueError("external VAD source SHA is invalid")
    if tuple(value.get("profiles", ())) != PROFILES:
        raise ValueError("external VAD profile set mismatch")
    return value


def wav_to_raw(path: pathlib.Path, output: pathlib.Path) -> int:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: expected mono PCM16 16-kHz WAV")
        raw = reader.readframes(reader.getnframes())
    samples = len(raw) // 2
    pad_samples = (-samples) % AP_FRAME_SAMPLES
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw + b"\x00\x00" * pad_samples)
    return pad_samples


def run_processor(
    processor: pathlib.Path,
    wav_path: pathlib.Path,
    *,
    profile: str,
    work_dir: pathlib.Path,
) -> list[dict]:
    raw = work_dir / f"{wav_path.stem}.pcm"
    out = work_dir / f"{wav_path.stem}.{profile}.out.pcm"
    metrics = work_dir / f"{wav_path.stem}.{profile}.metrics.jsonl"
    pad_samples = wav_to_raw(wav_path, raw)
    subprocess.run(
        [
            str(processor),
            "--sample-rate", "16000",
            "--mic-channels", "1",
            "--capture-only",
            "--capture-profile", profile,
            "--metrics-jsonl", str(metrics),
            str(raw),
            str(out),
        ],
        check=True,
    )
    rows: list[dict] = []
    for line_no, raw_line in enumerate(metrics.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        frame = int(row.get("frame", -1))
        probability = float(row.get("vad_probability", math.nan))
        active = int(row.get("vad_active", -1))
        if frame != len(rows) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{metrics}:{line_no}: invalid VAD probability trace")
        if active not in (0, 1):
            raise ValueError(f"{metrics}:{line_no}: invalid VAD active trace")
        rows.append(
            {
                "frame": frame,
                "start_sample": frame * AP_FRAME_SAMPLES,
                "end_sample": (frame + 1) * AP_FRAME_SAMPLES,
                "vad_probability": probability,
                "vad_active": active,
            }
        )
    if not rows:
        raise ValueError(f"empty VAD metrics: {metrics}")
    return rows


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return statistics.fmean(values)


def longest_run(values: list[int], target: int = 1) -> int:
    best = current = 0
    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def prefix_run(values: list[int], target: int = 1) -> int:
    count = 0
    for value in values:
        if value != target:
            break
        count += 1
    return count


def extract_features(rows: list[dict], *, gap_start: int, gap_end: int) -> dict[str, float]:
    pre = [row for row in rows if int(row["end_sample"]) <= gap_start]
    gap = [
        row for row in rows
        if int(row["start_sample"]) >= gap_start and int(row["end_sample"]) <= gap_end
    ]
    post = [row for row in rows if int(row["start_sample"]) >= gap_end]
    if len(pre) < 3 or len(gap) < 3 or len(post) < 3:
        raise ValueError("external VAD trace lacks full pre/gap/post support")
    pre3 = pre[-3:]
    post3 = post[:3]
    gp = [float(row["vad_probability"]) for row in gap]
    ga = [int(row["vad_active"]) for row in gap]
    pre_last = float(pre[-1]["vad_probability"])
    post_first = float(post[0]["vad_probability"])
    return {
        "pre_probability_mean_30ms": mean([float(row["vad_probability"]) for row in pre3]),
        "pre_probability_last": pre_last,
        "gap_probability_mean": mean(gp),
        "gap_probability_max": max(gp),
        "gap_probability_min": min(gp),
        "post_probability_mean_30ms": mean([float(row["vad_probability"]) for row in post3]),
        "post_probability_first": post_first,
        "release_delta": pre_last - gp[0],
        "attack_delta": post_first - gp[-1],
        "gap_active_ratio": sum(ga) / len(ga),
        "gap_active_longest_run_frames": float(longest_run(ga)),
        "gap_active_prefix_frames": float(prefix_run(ga)),
        "gap_active_suffix_frames": float(prefix_run(list(reversed(ga)))),
    }


def sample_rows(profile_records: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for record in profile_records:
        common = {
            "voice_id": record["voice_id"],
            "keyword_id": record["keyword_id"],
            "gap_ms": record["gap_ms"],
        }
        samples.append({**common, "label": True, "features": record["positive_features"]})
        samples.append({**common, "label": False, "features": record["negative_features"]})
    return samples


def evaluate_feature_set(samples: list[dict], features: tuple[str, ...]) -> dict:
    rule = fit_centroid(samples, features)
    train_correct, train_total = accuracy(rule, samples)
    voice = cross_validate(samples, features=features, group_key="voice_id")
    gap = cross_validate(samples, features=features, group_key="gap_ms")
    return {
        "features": list(features),
        "classifier": "zscore-nearest-centroid-linear-v1",
        "train": {
            "correct": train_correct,
            "total": train_total,
            "accuracy": train_correct / train_total,
        },
        "leave_one_voice_out": voice,
        "leave_one_gap_out": gap,
        "stable": (
            train_correct == train_total
            and voice["correct"] == voice["total"]
            and gap["correct"] == gap["total"]
        ),
    }


def build(args: argparse.Namespace) -> dict:
    source_pin = read_source_pin(args.source_pin.resolve())
    processor = args.processor.resolve()
    if not processor.is_file():
        raise ValueError(f"external VAD processor missing: {processor}")
    gaps = tuple(sorted(set(int(value) for value in args.gap_ms)))
    if len(gaps) < 3 or any(value <= 0 or value % 20 for value in gaps):
        raise ValueError("gap set must contain >=3 positive 20-ms multiples")

    output = args.output_dir.resolve()
    profile_records: dict[str, list[dict]] = {profile: [] for profile in PROFILES}
    gap_reports: dict[str, dict] = {}
    for gap_ms in gaps:
        gap_root = output / f"gap-{gap_ms}ms"
        result = build_gap_separability(
            SimpleNamespace(
                dataset_index=args.dataset_index,
                model=args.model,
                posterior_dump=args.posterior_dump,
                posterior_cache=args.posterior_cache,
                output_dir=gap_root,
                gap_ms=gap_ms,
                lead_ms=args.lead_ms,
                tail_ms=args.tail_ms,
            )
        )
        if (
            result.get("source_split") != "train"
            or result.get("selection_feedback_allowed") is not True
            or result.get("protected_evidence_used") is not False
        ):
            raise ValueError("KWS boundary generator authority mismatch")

        for record in result["records"]:
            voice = str(record["voice_id"])
            keyword_id = int(record["keyword_id"])
            stem = f"{safe_name(voice)}-kw{keyword_id}"
            positive_wav = gap_root / "audio" / f"{stem}-within-word-pause.wav"
            negative_wav = gap_root / "audio" / f"{stem}-cross-boundary.wav"
            for profile in PROFILES:
                profile_work = gap_root / "external-vad" / profile
                positive_trace = run_processor(
                    processor, positive_wav, profile=profile, work_dir=profile_work
                )
                negative_trace = run_processor(
                    processor, negative_wav, profile=profile, work_dir=profile_work
                )
                profile_records[profile].append(
                    {
                        "voice_id": voice,
                        "keyword_id": keyword_id,
                        "gap_ms": gap_ms,
                        "positive_audio_sha256": sha256_file(positive_wav),
                        "negative_audio_sha256": sha256_file(negative_wav),
                        "positive_features": extract_features(
                            positive_trace,
                            gap_start=int(record["positive"]["gap_start_sample"]),
                            gap_end=int(record["positive"]["gap_end_sample"]),
                        ),
                        "negative_features": extract_features(
                            negative_trace,
                            gap_start=int(record["negative"]["gap_start_sample"]),
                            gap_end=int(record["negative"]["gap_end_sample"]),
                        ),
                    }
                )
        gap_reports[str(gap_ms)] = {
            "pairs": int(result["pairs"]),
            "voices": int(result["voices"]),
        }

    profiles: dict[str, dict] = {}
    stable_profiles: list[str] = []
    for profile, records in profile_records.items():
        samples = sample_rows(records)
        feature_sets = {
            name: evaluate_feature_set(samples, features)
            for name, features in FEATURE_SETS.items()
        }
        stable_sets = [
            name for name, report in feature_sets.items() if report["stable"] is True
        ]
        profiles[profile] = {
            "samples": len(samples),
            "records": records,
            "feature_sets": feature_sets,
            "stable_feature_sets": stable_sets,
            "external_vad_signal_observed": bool(stable_sets),
        }
        if stable_sets:
            stable_profiles.append(profile)

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": True,
        "protected_evidence_used": False,
        "release_authority": False,
        "source_split": "train",
        "external_source": source_pin,
        "processor_sha256": sha256_file(processor),
        "gap_ms": list(gaps),
        "voices": len({r["voice_id"] for records in profile_records.values() for r in records}),
        "keywords": len({r["keyword_id"] for records in profile_records.values() for r in records}),
        "gap_reports": gap_reports,
        "profiles": profiles,
        "stable_profiles": stable_profiles,
        "external_vad_signal_observed": bool(stable_profiles),
    }
    write_json(output / "external-vad-feasibility.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pinned audio-pipeline VAD on train-only KWS boundaries."
    )
    parser.add_argument("--source-pin", required=True, type=pathlib.Path)
    parser.add_argument("--processor", required=True, type=pathlib.Path)
    parser.add_argument("--dataset-index", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gap-ms", nargs="+", type=int, default=list(DEFAULT_GAPS_MS))
    parser.add_argument("--lead-ms", type=int, default=1000)
    parser.add_argument("--tail-ms", type=int, default=1000)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({
        "gap_ms": result["gap_ms"],
        "stable_profiles": result["stable_profiles"],
        "external_vad_signal_observed": result["external_vad_signal_observed"],
        "selection_feedback_allowed": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
