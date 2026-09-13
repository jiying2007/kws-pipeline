#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "tools"))

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
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

POLICY = "gru-generalization-resynthesis-v1"
SEED_NAMESPACE = 261_000_007
EXPECTED_OPEN_SEEDS = [941102, 941105, 941107]


def sha256_json(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def scene_variant(base: dict, domains: dict, rng: random.Random, ordinal: int) -> dict:
    distance_low = min(float(v["distance_m"][0]) for v in domains["distance_bands"].values())
    distance_high = max(float(v["distance_m"][1]) for v in domains["distance_bands"].values())
    rt60_low, rt60_high = [float(v) for v in domains["rt60_s"]]
    snr_low, snr_high = [float(v) for v in domains["snr_db"]]
    sir_low, sir_high = [float(v) for v in domains["playback_sir_db"]]

    distance = clamp(float(base["distance_m"]) * rng.uniform(0.88, 1.12), distance_low, distance_high)
    azimuth = ((float(base["azimuth_deg"]) + rng.uniform(-20.0, 20.0) + 180.0) % 360.0) - 180.0
    rt60 = clamp(float(base["rt60_s"]) + rng.uniform(-0.12, 0.12), rt60_low, rt60_high)
    snr = clamp(float(base["snr_db"]) + rng.uniform(-3.5, 3.5), snr_low, snr_high)
    playback = base.get("playback_sir_db")
    if playback is not None:
        playback = clamp(float(playback) + rng.uniform(-3.0, 3.0), sir_low, sir_high)
    noise = str(base.get("noise_profile", "white"))
    if ordinal >= 4 and domains["noise_profiles"]:
        candidates = [str(v) for v in domains["noise_profiles"] if str(v) != noise]
        if candidates:
            noise = str(rng.choice(candidates))

    return {
        "distance_m": distance,
        "azimuth_deg": azimuth,
        "rt60_s": rt60,
        "snr_db": snr,
        "noise_profile": noise,
        "playback_sir_db": playback,
        "mic_spacing_m": float(domains["mic_spacing_m"]),
        "room_id": "gru-generalization-resynthesis",
        "rir_id": "gru-generalization-resynthesis",
    }


def forbidden_source_material(value) -> None:
    forbidden_keys = {"path", "audio_path", "source_path", "wav_sha256", "source_wav_sha256"}
    if isinstance(value, dict):
        overlap = forbidden_keys & set(value)
        if overlap:
            raise ValueError(f"diagnostic input unexpectedly contains source material references: {sorted(overlap)}")
        for child in value.values():
            forbidden_source_material(child)
    elif isinstance(value, list):
        for child in value:
            forbidden_source_material(child)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--diagnostics", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--examples-per-event", type=int, default=8)
    args = parser.parse_args()

    if not 4 <= args.examples_per_event <= 16:
        raise ValueError("examples-per-event must be 4..16")
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("evidence_class") != "development-only-open-shadow-failure-diagnostics":
        raise ValueError("unexpected GRU diagnostics evidence class")
    if diagnostics.get("formal_qualification_used") is not False or diagnostics.get("formal_seed_consumed") is not False:
        raise ValueError("GRU replay cannot consume formal qualification evidence")
    if [int(v) for v in diagnostics.get("open_development_seeds", [])] != EXPECTED_OPEN_SEEDS:
        raise ValueError("GRU replay diagnostics seed set drifted")
    if int(diagnostics.get("failure_event_count", -1)) != 4:
        raise ValueError("GRU replay requires exactly four diagnosed failure events")
    forbidden_source_material(diagnostics)

    token_path = pathlib.Path(str(cfg["tokens"]))
    keyword_path = pathlib.Path(str(cfg["keywords"]))
    if not token_path.is_absolute():
        token_path = (ROOT / token_path).resolve()
    if not keyword_path.is_absolute():
        keyword_path = (ROOT / keyword_path).resolve()
    token_map = load_tokens(token_path)
    keywords = parse_keywords(keyword_path, token_map)
    keyword_text = {int(v["id"]): str(v["text"]) for v in keywords}
    carriers = token_carriers(keywords, int(cfg.get("model", {}).get("feature_dim", 32)))

    generator = cfg.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    tts = generator.get("tts", {"backend": "tone"})
    augment_cfg = generator.get("augment", {})
    if not isinstance(tts, dict) or not isinstance(augment_cfg, dict):
        raise ValueError("invalid generator config")
    validate_tone_config(tts)
    validate_augment_config(augment_cfg)
    domains = validate_domains(cfg)

    events: list[dict] = []
    for seed_row in diagnostics["seeds"]:
        for event in seed_row.get("events", []):
            if int(event["seed"]) not in EXPECTED_OPEN_SEEDS:
                raise ValueError("event references unexpected opened shadow seed")
            source = event.get("source")
            if not isinstance(source, dict):
                raise ValueError("diagnostic event lacks source metadata")
            events.append(event)
    if len(events) != 4:
        raise ValueError("diagnostic event cardinality drifted")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wav_root = output / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    base_seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE
    manifest_rows: list[tuple[pathlib.Path, list[int]]] = []
    evidence_rows: list[dict] = []

    for event_index, event in enumerate(events):
        source = event["source"]
        tokens = [str(v) for v in source.get("tokens", [])]
        target_ids = [int(v) for v in source.get("target_ids", [])]
        source_kind = str(source.get("kind", ""))
        source_keyword = source.get("source_keyword_id")
        event_fingerprint = sha256_json(
            {
                "seed": int(event["seed"]),
                "event_type": str(event["event_type"]),
                "recording": str(event["recording"]),
                "source": source,
            }
        )
        generated: list[dict] = []
        for ordinal in range(args.examples_per_event):
            seed = (
                base_seed
                + event_index * 1_000_003
                + ordinal * 65_537
                + int(event_fingerprint[:8], 16)
            ) & 0x7FFFFFFF
            rng = random.Random(seed)
            clean_path = output / "clean" / f"e{event_index:02d}-r{ordinal:02d}.wav"
            if tokens:
                if str(tts.get("backend", "tone")) == "tone":
                    clean = render_tone_tokens(tokens, carriers, rng, tts)
                else:
                    text = (
                        keyword_text.get(int(source_keyword), " ".join(tokens))
                        if source_keyword is not None and source_kind == "positive"
                        else " ".join(tokens)
                    )
                    clean = render_command_tts(text, tokens, source_kind, clean_path, tts)
            else:
                clean = generate_background(
                    str(source["scene"].get("noise_profile", "white")),
                    rng.uniform(1.2, 2.0),
                    rng,
                )
            augmented = augment(clean, rng, augment_cfg)
            scene = scene_variant(source["scene"], domains, rng, ordinal)
            rendered, scene_meta = render_scene(
                augmented,
                scene,
                seed=seed + 31_337,
                afe=domains["afe"],
            )
            path = wav_root / f"e{event_index:02d}-r{ordinal:02d}.wav"
            write_wav(path, rendered)
            digest = sha256_file(path)
            manifest_rows.append((path, target_ids))
            generated.append(
                {
                    "ordinal": ordinal,
                    "seed": seed,
                    "wav_sha256": digest,
                    "scene": scene_meta,
                }
            )
        evidence_rows.append(
            {
                "event_index": event_index,
                "source_seed": int(event["seed"]),
                "event_type": str(event["event_type"]),
                "source_kind": source_kind,
                "source_keyword_id": source_keyword,
                "detected_keyword_id": event.get("detected_keyword_id"),
                "expected_keyword_id": event.get("expected_keyword_id"),
                "tokens": tokens,
                "target_ids": target_ids,
                "source_scene": source["scene"],
                "source_fingerprint_sha256": event_fingerprint,
                "generated": generated,
            }
        )

    manifest = output / "gru-generalization-replay.tsv"
    manifest.write_text(
        "".join(
            f"{path.resolve()}\t{' '.join(str(v) for v in targets)}\n"
            for path, targets in manifest_rows
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": "training-only-gru-generalization-resynthesis",
        "policy": POLICY,
        "formal_qualification_used": False,
        "shadow_wav_bytes_copied": False,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "config_sha256": sha256_file(config_path),
        "seed_namespace": SEED_NAMESPACE,
        "examples_per_event": args.examples_per_event,
        "source_event_count": len(events),
        "examples": len(manifest_rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "events": evidence_rows,
    }
    evidence_path = output / "gru-generalization-replay.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
