#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import struct
import wave

from acoustic_scene import render_scene, sha256_file
from frontend_spec import SAMPLE_RATE_HZ
from synthetic_audio import SPLITS, generate_dataset, load_config, write_wav

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DISTANCE_ORDER = ("far", "mid", "near")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def read_wav(path: pathlib.Path) -> list[int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getframerate() != SAMPLE_RATE_HZ or reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def range_pair(value, label: str, minimum: float | None = None) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be [min,max]")
    low = finite(value[0], f"{label}[0]")
    high = finite(value[1], f"{label}[1]")
    if low > high or (minimum is not None and low < minimum):
        raise ValueError(f"{label} range is invalid")
    return low, high


def validate_domains(config: dict) -> dict:
    domains = config.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domains config is required")
    bands = domains.get("distance_bands")
    if not isinstance(bands, dict) or set(bands) != {"near", "mid", "far"}:
        raise ValueError("domains.distance_bands must define near/mid/far")
    normalized_bands: dict[str, dict] = {}
    for name in ("near", "mid", "far"):
        item = bands[name]
        if not isinstance(item, dict):
            raise ValueError(f"domains.distance_bands.{name} must be an object")
        low, high = range_pair(item.get("distance_m"), f"distance_bands.{name}.distance_m", 0.05)
        weight = finite(item.get("weight", 1.0), f"distance_bands.{name}.weight")
        if weight <= 0.0:
            raise ValueError("distance band weights must be > 0")
        normalized_bands[name] = {"distance_m": [low, high], "weight": weight}
    azimuths = domains.get("azimuth_deg")
    if not isinstance(azimuths, list) or not azimuths:
        raise ValueError("domains.azimuth_deg must be non-empty")
    normalized_azimuths = [finite(value, "domains.azimuth_deg") for value in azimuths]
    rt60 = range_pair(domains.get("rt60_s"), "domains.rt60_s", 0.05)
    snr = range_pair(domains.get("snr_db"), "domains.snr_db")
    playback = domains.get("playback", {})
    if not isinstance(playback, dict):
        raise ValueError("domains.playback must be an object")
    probability = finite(playback.get("probability", 0.0), "domains.playback.probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("playback probability must be in [0,1]")
    sir = range_pair(playback.get("sir_db", [-5.0, 20.0]), "domains.playback.sir_db")
    noise_profiles = domains.get("noise_profiles", ["white", "fan", "motor", "media"])
    if not isinstance(noise_profiles, list) or not noise_profiles:
        raise ValueError("domains.noise_profiles must be non-empty")
    scenes_per_split = domains.get("scenes_per_example", {})
    if not isinstance(scenes_per_split, dict):
        raise ValueError("domains.scenes_per_example must be an object")
    counts: dict[str, int] = {}
    for split in SPLITS:
        count = int(scenes_per_split.get(split, 1))
        if count <= 0 or count > 64:
            raise ValueError(f"domains.scenes_per_example.{split} must be 1..64")
        counts[split] = count
    afe = domains.get("afe", {"backend": "proxy"})
    if not isinstance(afe, dict):
        raise ValueError("domains.afe must be an object")
    if str(afe.get("backend", "proxy")) not in {"proxy", "command"}:
        raise ValueError("domains.afe.backend must be proxy or command")
    return {
        "distance_bands": normalized_bands,
        "azimuth_deg": normalized_azimuths,
        "rt60_s": rt60,
        "snr_db": snr,
        "playback_probability": probability,
        "playback_sir_db": sir,
        "noise_profiles": [str(value) for value in noise_profiles],
        "scenes_per_example": counts,
        "mic_spacing_m": finite(domains.get("mic_spacing_m", 0.06), "domains.mic_spacing_m"),
        "afe": afe,
    }


def _dimension_weights(curriculum: dict | None, dimension: str) -> dict[str, float]:
    if not isinstance(curriculum, dict):
        return {}
    dimensions = curriculum.get("dimension_weights")
    if not isinstance(dimensions, dict):
        return {}
    raw = dimensions.get(dimension, {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        weight = finite(value, f"curriculum.{dimension}.{key}")
        if weight <= 0.0:
            raise ValueError("curriculum weights must be > 0")
        result[str(key)] = weight
    return result


def _weighted_choice(rng: random.Random, values: list, weights: list[float]):
    if not values or len(values) != len(weights):
        raise ValueError("weighted choice requires aligned non-empty values/weights")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weighted choice contains invalid weight")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weighted choice total weight must be > 0")
    pick = rng.random() * total
    accumulated = 0.0
    for value, weight in zip(values, weights):
        accumulated += weight
        if pick <= accumulated:
            return value
    return values[-1]


def _azimuth_band(value: float) -> str:
    if abs(value) <= 30.0:
        return "front"
    if abs(value) <= 90.0:
        return "side"
    return "rear"


def _rt60_band(value: float) -> str:
    if value < 0.30:
        return "dry"
    if value < 0.55:
        return "medium"
    return "reverb"


def _composite_value(scene: dict) -> str:
    playback = "playback" if scene["playback_sir_db"] is not None else "no-playback"
    return (
        f"distance={scene['distance_band']}|az={_azimuth_band(float(scene['azimuth_deg']))}|"
        f"rt60={_rt60_band(float(scene['rt60_s']))}|noise={scene['noise_profile']}|{playback}"
    )


def choose_band(rng: random.Random, bands: dict[str, dict], curriculum: dict | None = None) -> str:
    names = ["near", "mid", "far"]
    adaptive = _dimension_weights(curriculum, "distance")
    values = [bands[name]["weight"] * adaptive.get(name, 1.0) for name in names]
    return str(_weighted_choice(rng, names, values))


def _sample_rt60(domains: dict, rng: random.Random, curriculum: dict | None) -> float:
    low, high = domains["rt60_s"]
    adaptive = _dimension_weights(curriculum, "rt60")
    intervals = {
        "dry": (low, min(high, 0.30)),
        "medium": (max(low, 0.30), min(high, 0.55)),
        "reverb": (max(low, 0.55), high),
    }
    available = [name for name, (a, b) in intervals.items() if b > a]
    if not available:
        return rng.uniform(low, high)
    band = str(_weighted_choice(rng, available, [adaptive.get(name, 1.0) for name in available]))
    a, b = intervals[band]
    return rng.uniform(a, b)


def _sample_candidate(
    domains: dict,
    rng: random.Random,
    *,
    curriculum: dict | None,
    forced_band: str | None,
) -> dict:
    if forced_band is not None and forced_band not in domains["distance_bands"]:
        raise ValueError(f"unsupported forced distance band: {forced_band}")
    band = forced_band or choose_band(rng, domains["distance_bands"], curriculum)
    low, high = domains["distance_bands"][band]["distance_m"]
    snr_low, snr_high = domains["snr_db"]

    az_weights = _dimension_weights(curriculum, "azimuth")
    azimuth = float(
        _weighted_choice(
            rng,
            domains["azimuth_deg"],
            [az_weights.get(_azimuth_band(float(value)), 1.0) for value in domains["azimuth_deg"]],
        )
    )
    noise_weights = _dimension_weights(curriculum, "noise")
    noise = str(
        _weighted_choice(
            rng,
            domains["noise_profiles"],
            [noise_weights.get(str(value), 1.0) for value in domains["noise_profiles"]],
        )
    )

    playback_weights = _dimension_weights(curriculum, "playback")
    base_on = domains["playback_probability"]
    playback_on = bool(
        _weighted_choice(
            rng,
            [False, True],
            [
                (1.0 - base_on) * playback_weights.get("no-playback", 1.0),
                base_on * playback_weights.get("playback", 1.0),
            ],
        )
    )
    playback = None
    if playback_on:
        sir_low, sir_high = domains["playback_sir_db"]
        playback = rng.uniform(sir_low, sir_high)

    return {
        "distance_m": rng.uniform(low, high),
        "azimuth_deg": azimuth,
        "rt60_s": _sample_rt60(domains, rng, curriculum),
        "snr_db": rng.uniform(snr_low, snr_high),
        "noise_profile": noise,
        "playback_sir_db": playback,
        "mic_spacing_m": domains["mic_spacing_m"],
        "room_id": f"sim-{band}",
        "rir_id": f"sim-{band}",
        "distance_band": band,
    }


def sample_scene(
    domains: dict,
    rng: random.Random,
    *,
    curriculum_weights: dict | None = None,
    forced_band: str | None = None,
) -> dict:
    # Independent dimensions are sampled with their own adaptive weights. A final
    # bounded candidate choice applies composite-domain hardness without turning
    # scene generation into an unbounded rejection loop.
    candidates = [
        _sample_candidate(
            domains,
            rng,
            curriculum=curriculum_weights,
            forced_band=forced_band,
        )
        for _ in range(4)
    ]
    composite = _dimension_weights(curriculum_weights, "composite")
    if not composite:
        return candidates[0]
    return _weighted_choice(
        rng,
        candidates,
        [composite.get(_composite_value(scene), 1.0) for scene in candidates],
    )


def render_domain_dataset(
    config_path: pathlib.Path,
    output: pathlib.Path,
    *,
    curriculum_weights: dict | None = None,
) -> dict:
    config = load_config(config_path)
    domains = validate_domains(config)
    base_dir = output / "base"
    base_summary = generate_dataset(config_path, base_dir)
    base_rows = load_jsonl(base_dir / "dataset-index.jsonl")
    seed = int(config.get("seed", 1337))

    output.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    domain_rows: list[dict] = []
    positive_scene_ordinals = {split: 0 for split in SPLITS}
    for base_index, row in enumerate(base_rows):
        split = str(row["split"])
        if split not in rows_by_split:
            raise ValueError(f"unknown base split: {split}")
        source = pathlib.Path(row["path"])
        clean = read_wav(source)
        for scene_index in range(domains["scenes_per_example"][split]):
            scene_seed = seed + base_index * 1_000_003 + scene_index * 65_537 + SPLITS.index(split) * 9_000_001
            rng = random.Random(scene_seed)
            forced_band = None
            if split != "train" and row["kind"] == "positive":
                ordinal = positive_scene_ordinals[split]
                forced_band = EVAL_DISTANCE_ORDER[ordinal % len(EVAL_DISTANCE_ORDER)]
                positive_scene_ordinals[split] = ordinal + 1
            scene = sample_scene(
                domains,
                rng,
                curriculum_weights=curriculum_weights if split == "train" else None,
                forced_band=forced_band,
            )
            mono, scene_meta = render_scene(clean, scene, seed=scene_seed, afe=domains["afe"])
            target = output / "clips" / split / f"d{base_index:05d}-s{scene_index:02d}.wav"
            write_wav(target, mono)
            meta = {
                **row,
                "path": str(target.resolve()),
                "source_path": str(source.resolve()),
                "source_wav_sha256": str(row["wav_sha256"]),
                "wav_sha256": sha256_file(target),
                "scene_seed": scene_seed,
                "scene": scene_meta,
                "domain_id": (
                    f"{scene_meta['distance_band']}|az={int(round(scene_meta['azimuth_deg']))}|"
                    f"rt60={scene_meta['rt60_s']:.2f}|noise={scene_meta['noise_profile']}|"
                    f"playback={'on' if scene_meta['playback_sir_db'] is not None else 'off'}"
                ),
            }
            rows_by_split[split].append(meta)
            domain_rows.append(meta)

    manifest_paths: dict[str, pathlib.Path] = {}
    reference_paths: dict[str, pathlib.Path] = {}
    for split in SPLITS:
        manifest = output / f"{split}.tsv"
        manifest.write_text(
            "".join(
                f"{row['path']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
                for row in rows_by_split[split]
            ),
            encoding="utf-8",
        )
        manifest_paths[split] = manifest
        if split == "train":
            continue
        references = output / f"{split}.references.jsonl"
        lines: list[str] = []
        for index, row in enumerate(rows_by_split[split]):
            samples = read_wav(pathlib.Path(row["path"]))
            expected: list[dict] = []
            if row["kind"] == "positive":
                start = int(row["event_start_frame"]) + int(row["scene"]["direct_delay_samples"]) + int(row["scene"]["afe_latency_samples"])
                end = int(row["event_end_frame"]) + int(row["scene"]["direct_delay_samples"]) + int(row["scene"]["afe_latency_samples"])
                start = max(0, min(start, len(samples) - 1))
                end = max(start + 1, min(end, len(samples)))
                expected.append(
                    {
                        "keyword_id": int(row["keyword_id"]),
                        "start_s": start / SAMPLE_RATE_HZ,
                        "end_s": end / SAMPLE_RATE_HZ,
                    }
                )
            lines.append(
                json.dumps(
                    {
                        "recording": f"domain-{split}-{index:06d}",
                        "path": pathlib.Path(row["path"]).name,
                        "audio_path": row["path"],
                        "duration_s": len(samples) / SAMPLE_RATE_HZ,
                        "expected": expected,
                        "domain": row["scene"],
                        "domain_id": row["domain_id"],
                        "kind": row["kind"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        references.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reference_paths[split] = references

    index_path = output / "domain-index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in domain_rows) + "\n",
        encoding="utf-8",
    )
    histogram: dict[str, int] = {}
    histogram_by_split: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for row in domain_rows:
        key = str(row["scene"]["distance_band"])
        split = str(row["split"])
        histogram[key] = histogram.get(key, 0) + 1
        split_histogram = histogram_by_split[split]
        split_histogram[key] = split_histogram.get(key, 0) + 1
    summary = {
        "schema_version": 2,
        "evidence_class": "synthetic-domain",
        "config_sha256": sha256_file(config_path),
        "base_dataset_summary_sha256": sha256_file(base_dir / "dataset-summary.json"),
        "domain_index_sha256": sha256_file(index_path),
        "distance_histogram": histogram,
        "distance_histogram_by_split": histogram_by_split,
        "evaluation_positive_distance_order": list(EVAL_DISTANCE_ORDER),
        "curriculum": curriculum_weights or {},
        "afe": domains["afe"],
        "splits": {
            split: {
                "examples": len(rows_by_split[split]),
                "manifest": str(manifest_paths[split]),
                "manifest_sha256": sha256_file(manifest_paths[split]),
                **(
                    {
                        "references": str(reference_paths[split]),
                        "references_sha256": sha256_file(reference_paths[split]),
                    }
                    if split in reference_paths
                    else {}
                ),
            }
            for split in SPLITS
        },
    }
    summary_path = output / "domain-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--curriculum", type=pathlib.Path)
    args = parser.parse_args()
    curriculum = None
    if args.curriculum:
        value = json.loads(args.curriculum.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 2:
            raise ValueError("curriculum must be schema_version 2")
        if not isinstance(value.get("dimension_weights"), dict):
            raise ValueError("curriculum must contain dimension_weights")
        curriculum = value
    result = render_domain_dataset(
        args.config.resolve(),
        args.output.resolve(),
        curriculum_weights=curriculum,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, wave.Error) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
