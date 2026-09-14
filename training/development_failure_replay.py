from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from render_domains import validate_domains  # noqa: E402
from synthetic_audio import (  # noqa: E402
    augment,
    generate_background,
    load_config,
    parse_keywords,
    render_command_tts,
    render_tone_tokens,
    token_carriers,
    validate_augment_config,
    validate_tone_config,
    write_wav,
)

POLICY = "development-failure-resynthesis-v1"
RECORDING_RE = re.compile(r"^domain-(calibration|test)-(\d{6})$")
SEED_NAMESPACE = 151_000_003


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def _policy(cfg: dict) -> dict:
    raw = cfg.get("data_augmentation_v3", {})
    if not isinstance(raw, dict):
        raise ValueError("data_augmentation_v3 must be an object")
    enabled = bool(raw.get("failure_replay_enabled", False))
    if enabled:
        if str(raw.get("policy")) != "train-only-balanced-mining-v1":
            raise ValueError("failure replay requires train-only data v3 policy")
        if raw.get("formal_qualification_used") is not False:
            raise ValueError("failure replay must not use formal qualification")
        if raw.get("expand_evaluation_splits") is not False:
            raise ValueError("failure replay must not mutate evaluation splits")
    examples = int(raw.get("failure_replay_examples_per_failure", 4))
    max_unique = int(raw.get("failure_replay_max_unique_failures", 64))
    max_per_keyword = int(raw.get("failure_replay_max_per_keyword", 32))
    if examples <= 0 or examples > 32:
        raise ValueError("failure replay examples_per_failure must be 1..32")
    if max_unique <= 0 or max_unique > 256:
        raise ValueError("failure replay max_unique_failures must be 1..256")
    if max_per_keyword <= 0 or max_per_keyword > max_unique:
        raise ValueError("failure replay max_per_keyword is invalid")
    return {
        "enabled": enabled,
        "examples_per_failure": examples,
        "max_unique_failures": max_unique,
        "max_per_keyword": max_per_keyword,
    }


def _round_split_rows(work: pathlib.Path, round_index: int, split: str) -> list[dict]:
    path = work / "datasets" / f"round-{round_index:02d}" / "domain-index.jsonl"
    rows = [row for row in _load_jsonl(path) if str(row.get("split")) == split]
    if not rows:
        raise ValueError(f"failure replay cannot resolve {split} domain rows for round {round_index}")
    return rows


def _failure_rows(metrics: dict, kind: str) -> list[dict]:
    field = "false_positives_path" if kind == "false-accept" else "false_rejects_path"
    value = metrics.get(field)
    if not isinstance(value, str) or not value:
        return []
    return _load_jsonl(pathlib.Path(value))


def _scene_signature(scene: dict) -> tuple:
    playback = scene.get("playback_sir_db")
    return (
        round(float(scene.get("distance_m", 0.0)), 2),
        round(float(scene.get("azimuth_deg", 0.0)), 1),
        round(float(scene.get("snr_db", 0.0)), 1),
        round(float(scene.get("rt60_s", 0.0)), 2),
        str(scene.get("noise_profile", "")),
        playback is not None,
    )


def collect_failure_specs(records: list[dict], work: pathlib.Path) -> list[dict]:
    aggregated: dict[str, dict] = {}
    row_cache: dict[tuple[int, str], list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        round_index = int(record.get("round", -1))
        if round_index < 0:
            continue
        for split in ("calibration", "test"):
            metrics = record.get(split)
            if not isinstance(metrics, dict):
                continue
            cache_key = (round_index, split)
            if cache_key not in row_cache:
                row_cache[cache_key] = _round_split_rows(work, round_index, split)
            domain_rows = row_cache[cache_key]
            for failure_kind in ("false-accept", "false-reject"):
                for failure in _failure_rows(metrics, failure_kind):
                    match = RECORDING_RE.match(str(failure.get("recording", "")))
                    if match is None or match.group(1) != split:
                        raise ValueError("failure replay saw an unexpected development recording id")
                    index = int(match.group(2))
                    if index >= len(domain_rows):
                        raise ValueError("failure replay recording index exceeds domain-index split")
                    source = domain_rows[index]
                    source_sha = str(source.get("wav_sha256") or "")
                    if len(source_sha) != 64:
                        raise ValueError("failure replay source is missing development WAV SHA")
                    scene = source.get("scene")
                    if not isinstance(scene, dict):
                        raise ValueError("failure replay source is missing acoustic scene metadata")
                    tokens = [str(value) for value in source.get("tokens", [])]
                    target_ids = [int(value) for value in source.get("target_ids", [])]
                    source_keyword = source.get("keyword_id")
                    focus_keyword = int(
                        failure.get("keyword_id")
                        if failure.get("keyword_id") is not None
                        else source_keyword or 0
                    )
                    key = source_sha
                    current = aggregated.get(key)
                    if current is None:
                        current = {
                            "source_wav_sha256": source_sha,
                            "source_base_wav_sha256": str(source.get("source_wav_sha256") or ""),
                            "source_kind": str(source.get("kind") or ""),
                            "tokens": tokens,
                            "target_ids": target_ids,
                            "source_keyword_id": (
                                int(source_keyword) if source_keyword is not None else None
                            ),
                            "focus_keyword_ids": set(),
                            "failure_kinds": set(),
                            "failure_hits": 0,
                            "max_false_accept_confidence": 0.0,
                            "scene": dict(scene),
                            "scene_signature": _scene_signature(scene),
                            "source_splits": set(),
                            "source_rounds": set(),
                        }
                        aggregated[key] = current
                    if current["tokens"] != tokens or current["target_ids"] != target_ids:
                        raise ValueError("same development WAV SHA maps to inconsistent targets")
                    current["failure_hits"] += 1
                    current["failure_kinds"].add(failure_kind)
                    current["source_splits"].add(split)
                    current["source_rounds"].add(round_index)
                    if focus_keyword > 0:
                        current["focus_keyword_ids"].add(focus_keyword)
                    if failure_kind == "false-accept":
                        current["max_false_accept_confidence"] = max(
                            float(current["max_false_accept_confidence"]),
                            float(failure.get("confidence", 0.0)),
                        )

    normalized: list[dict] = []
    for item in aggregated.values():
        normalized.append(
            {
                **item,
                "focus_keyword_ids": sorted(item["focus_keyword_ids"]),
                "failure_kinds": sorted(item["failure_kinds"]),
                "source_splits": sorted(item["source_splits"]),
                "source_rounds": sorted(item["source_rounds"]),
            }
        )
    normalized.sort(
        key=lambda item: (
            -int(item["failure_hits"]),
            -float(item["max_false_accept_confidence"]),
            str(item["source_wav_sha256"]),
        )
    )
    return normalized


def select_failure_specs(specs: list[dict], *, max_unique: int, max_per_keyword: int) -> list[dict]:
    selected: list[dict] = []
    counts: dict[int, int] = {}
    for item in specs:
        focus = [int(value) for value in item.get("focus_keyword_ids", []) if int(value) > 0]
        primary = focus[0] if focus else int(item.get("source_keyword_id") or 0)
        if primary > 0 and counts.get(primary, 0) >= max_per_keyword:
            continue
        selected.append(item)
        if primary > 0:
            counts[primary] = counts.get(primary, 0) + 1
        if len(selected) >= max_unique:
            break
    return selected


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _jitter_scene(base: dict, domains: dict, rng: random.Random, example_index: int) -> dict:
    distance_low = min(float(item["distance_m"][0]) for item in domains["distance_bands"].values())
    distance_high = max(float(item["distance_m"][1]) for item in domains["distance_bands"].values())
    rt60_low, rt60_high = [float(value) for value in domains["rt60_s"]]
    snr_low, snr_high = [float(value) for value in domains["snr_db"]]
    sir_low, sir_high = [float(value) for value in domains["playback_sir_db"]]
    distance = float(base["distance_m"])
    azimuth = float(base["azimuth_deg"])
    rt60 = float(base["rt60_s"])
    snr = float(base["snr_db"])
    playback = base.get("playback_sir_db")
    noise = str(base.get("noise_profile", "white"))
    if example_index > 0:
        distance = _clamp(distance * rng.uniform(0.92, 1.08), distance_low, distance_high)
        azimuth = ((azimuth + rng.uniform(-15.0, 15.0) + 180.0) % 360.0) - 180.0
        rt60 = _clamp(rt60 + rng.uniform(-0.08, 0.08), rt60_low, rt60_high)
        snr = _clamp(snr + rng.uniform(-2.5, 2.5), snr_low, snr_high)
        if playback is not None:
            playback = _clamp(float(playback) + rng.uniform(-2.0, 2.0), sir_low, sir_high)
        if example_index >= 2 and domains["noise_profiles"]:
            noise = str(rng.choice(domains["noise_profiles"]))
    return {
        "distance_m": distance,
        "azimuth_deg": azimuth,
        "rt60_s": rt60,
        "snr_db": snr,
        "noise_profile": noise,
        "playback_sir_db": playback,
        "mic_spacing_m": float(domains["mic_spacing_m"]),
        "room_id": "development-failure-resynthesis",
        "rir_id": "development-failure-resynthesis",
    }


def render_development_failure_replay(
    config_path: pathlib.Path,
    records: list[dict],
    work: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    cfg = load_config(config_path)
    policy = _policy(cfg)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "development-failure-replay.tsv"
    evidence_path = output / "development-failure-replay.json"
    if not policy["enabled"]:
        manifest.write_text("", encoding="utf-8")
        evidence = {
            "schema_version": 1,
            "evidence_class": "development-only-failure-resynthesis",
            "policy": POLICY,
            "enabled": False,
            "examples": 0,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "formal_qualification_used": False,
            "development_source_wav_bytes_copied": False,
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**evidence, "evidence": str(evidence_path)}

    specs = collect_failure_specs(records, work)
    selected = select_failure_specs(
        specs,
        max_unique=int(policy["max_unique_failures"]),
        max_per_keyword=int(policy["max_per_keyword"]),
    )
    token_path = pathlib.Path(str(cfg["tokens"]))
    keyword_path = pathlib.Path(str(cfg["keywords"]))
    if not token_path.is_absolute():
        token_path = (ROOT / token_path).resolve()
    if not keyword_path.is_absolute():
        keyword_path = (ROOT / keyword_path).resolve()
    token_map = load_tokens(token_path)
    keywords = parse_keywords(keyword_path, token_map)
    keyword_text = {int(item["id"]): str(item["text"]) for item in keywords}
    carriers = token_carriers(keywords, int(cfg.get("model", {}).get("feature_dim", 32)))
    generator = cfg.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    tts = generator.get("tts", {"backend": "tone"})
    augment_config = generator.get("augment", {})
    if not isinstance(tts, dict) or not isinstance(augment_config, dict):
        raise ValueError("failure replay generator config is invalid")
    validate_tone_config(tts)
    validate_augment_config(augment_config)
    domains = validate_domains(cfg)
    seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE

    rows: list[dict] = []
    for spec_index, spec in enumerate(selected):
        for example_index in range(int(policy["examples_per_failure"])):
            example_seed = (
                seed
                + spec_index * 65_537
                + example_index * 4099
                + int(str(spec["source_wav_sha256"])[:8], 16)
            ) & 0x7FFFFFFF
            rng = random.Random(example_seed)
            tokens = [str(value) for value in spec["tokens"]]
            path = output / "wav" / f"f{spec_index:03d}-e{example_index:02d}.wav"
            clean_path = output / "clean" / f"f{spec_index:03d}-e{example_index:02d}.wav"
            if tokens:
                if str(tts.get("backend", "tone")) == "tone":
                    clean = render_tone_tokens(tokens, carriers, rng, tts)
                else:
                    source_keyword = spec.get("source_keyword_id")
                    text = keyword_text.get(int(source_keyword), " ".join(tokens)) if source_keyword else " ".join(tokens)
                    clean = render_command_tts(text, tokens, str(spec["source_kind"]), clean_path, tts)
            else:
                clean = generate_background(
                    str(spec["scene"].get("noise_profile", "white")),
                    rng.uniform(1.2, 2.0),
                    rng,
                )
            augmented = augment(clean, rng, augment_config)
            scene = _jitter_scene(spec["scene"], domains, rng, example_index)
            rendered, scene_meta = render_scene(
                augmented,
                scene,
                seed=example_seed + 31_337,
                afe=domains["afe"],
            )
            write_wav(path, rendered)
            replay_sha = sha256_file(path)
            if replay_sha == str(spec["source_wav_sha256"]):
                raise ValueError("failure replay accidentally copied a development evaluation WAV")
            rows.append(
                {
                    "path": str(path.resolve()),
                    "target_ids": [int(value) for value in spec["target_ids"]],
                    "tokens": tokens,
                    "source_kind": str(spec["source_kind"]),
                    "source_wav_sha256": str(spec["source_wav_sha256"]),
                    "failure_kinds": list(spec["failure_kinds"]),
                    "failure_hits": int(spec["failure_hits"]),
                    "focus_keyword_ids": list(spec["focus_keyword_ids"]),
                    "example_seed": example_seed,
                    "wav_sha256": replay_sha,
                    "scene": scene_meta,
                }
            )

    manifest.write_text(
        "".join(
            f"{row['path']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": "development-only-failure-resynthesis",
        "policy": POLICY,
        "enabled": True,
        "source_splits": ["calibration", "test"],
        "observed_unique_failures": len(specs),
        "selected_unique_failures": len(selected),
        "examples_per_failure": int(policy["examples_per_failure"]),
        "max_unique_failures": int(policy["max_unique_failures"]),
        "max_per_keyword": int(policy["max_per_keyword"]),
        "examples": len(rows),
        "selected": selected,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "formal_qualification_used": False,
        "development_source_wav_bytes_copied": False,
        "seed_namespace": SEED_NAMESPACE,
    }
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**evidence, "evidence": str(evidence_path)}
