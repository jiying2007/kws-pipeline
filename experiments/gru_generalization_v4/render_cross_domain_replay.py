#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

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

POLICY = "gru-generalization-cross-domain-resynthesis-v1"
EVIDENCE_CLASS = "training-only-gru-generalization-cross-domain-resynthesis"
SEED_NAMESPACE = 361_000_013
EXPECTED_OPEN_SEEDS = [941102, 941105, 941107]
EXPECTED_FAILURE_EVENT_COUNT = 4
EXPECTED_SOURCE_MODEL_SHA256 = "83112b829edbc5969cb2f5eb6f50b01b4a35064f56dce812440126110b10236f"
EXPECTED_DIAGNOSTIC_POLICY = "gru-open-shadow-failure-lexical-diagnostics-v1"
EXPECTED_DISTANCE_BANDS = ["near", "mid", "far"]
EXAMPLES_PER_EVENT = 12
SNR_FRACTIONS = (
    0.0,
    1.0 / 18.0,
    2.0 / 18.0,
    3.0 / 18.0,
    4.0 / 18.0,
    6.0 / 18.0,
    8.0 / 18.0,
    10.0 / 18.0,
    12.0 / 18.0,
    14.0 / 18.0,
    16.0 / 18.0,
    1.0,
)


def sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def forbidden_source_material(value: object) -> None:
    forbidden_keys = {"path", "audio_path", "source_path", "wav_sha256", "source_wav_sha256"}
    if isinstance(value, dict):
        overlap = forbidden_keys & set(value)
        if overlap:
            raise ValueError(
                "diagnostic input unexpectedly contains source material references: "
                f"{sorted(overlap)}"
            )
        for child in value.values():
            forbidden_source_material(child)
    elif isinstance(value, list):
        for child in value:
            forbidden_source_material(child)


def event_identity(event: dict) -> dict:
    source = event["source"]
    return {
        "seed": int(event["seed"]),
        "event_type": str(event["event_type"]),
        "recording": str(event["recording"]),
        "source_kind": str(source.get("kind", "")),
        "source_keyword_id": source.get("source_keyword_id"),
        "expected_keyword_id": event.get("expected_keyword_id"),
        "detected_keyword_id": event.get("detected_keyword_id"),
        "tokens": [str(v) for v in source.get("tokens", [])],
        "target_ids": [int(v) for v in source.get("target_ids", [])],
    }


def lattice_scene(domains: dict, event_index: int, ordinal: int, rng: random.Random) -> tuple[dict, str, str]:
    distance_bands = domains["distance_bands"]
    if list(distance_bands) != EXPECTED_DISTANCE_BANDS:
        raise ValueError("distance-band contract drifted")
    azimuths = [float(v) for v in domains["azimuth_deg"]]
    if len(azimuths) != EXAMPLES_PER_EVENT or len(set(azimuths)) != EXAMPLES_PER_EVENT:
        raise ValueError("cross-domain replay requires exactly twelve unique configured azimuths")
    noise_profiles = [str(v) for v in domains["noise_profiles"]]
    if len(noise_profiles) != 4 or len(set(noise_profiles)) != 4:
        raise ValueError("cross-domain replay requires four unique configured noise profiles")

    band_name = EXPECTED_DISTANCE_BANDS[(ordinal + event_index) % len(EXPECTED_DISTANCE_BANDS)]
    band = distance_bands[band_name]
    distance_low, distance_high = [float(v) for v in band["distance_m"]]
    distance_m = rng.uniform(distance_low, distance_high)

    azimuth_deg = azimuths[(ordinal + event_index * 5) % len(azimuths)]
    noise_profile = noise_profiles[(ordinal + event_index) % len(noise_profiles)]

    rt60_low, rt60_high = [float(v) for v in domains["rt60_s"]]
    rt60_fraction = ((ordinal * 7 + event_index * 3) % EXAMPLES_PER_EVENT) / (EXAMPLES_PER_EVENT - 1)
    rt60_s = rt60_low + (rt60_high - rt60_low) * rt60_fraction

    snr_low, snr_high = [float(v) for v in domains["snr_db"]]
    snr_fraction = SNR_FRACTIONS[(ordinal + event_index * 3) % len(SNR_FRACTIONS)]
    snr_db = snr_low + (snr_high - snr_low) * snr_fraction

    playback_state = "playback" if (ordinal + event_index) % 2 == 0 else "no-playback"
    playback_sir_db = None
    if playback_state == "playback":
        sir_low, sir_high = [float(v) for v in domains["playback_sir_db"]]
        sir_fraction = ((ordinal * 5 + event_index) % EXAMPLES_PER_EVENT) / (EXAMPLES_PER_EVENT - 1)
        playback_sir_db = sir_low + (sir_high - sir_low) * sir_fraction

    scene = {
        "distance_m": distance_m,
        "azimuth_deg": azimuth_deg,
        "rt60_s": rt60_s,
        "snr_db": snr_db,
        "noise_profile": noise_profile,
        "playback_sir_db": playback_sir_db,
        "mic_spacing_m": float(domains["mic_spacing_m"]),
        "room_id": "gru-generalization-cross-domain-v4",
        "rir_id": "gru-generalization-cross-domain-v4",
    }
    return scene, band_name, playback_state


def require_event_contract(events: list[dict]) -> None:
    if len(events) != EXPECTED_FAILURE_EVENT_COUNT:
        raise ValueError("diagnostic event cardinality drifted")
    false_rejects = []
    false_accepts = []
    for event in events:
        source = event["source"]
        if int(event["seed"]) not in EXPECTED_OPEN_SEEDS:
            raise ValueError("event references unexpected open-development seed")
        if event["event_type"] == "false_reject":
            false_rejects.append(event)
            if str(source.get("kind")) != "positive" or int(event.get("expected_keyword_id", -1)) != 2:
                raise ValueError("GRU cross-domain FR contract drifted")
        elif event["event_type"] == "false_accept":
            false_accepts.append(event)
            if str(source.get("kind")) != "negative" or int(event.get("detected_keyword_id", -1)) != 2:
                raise ValueError("GRU cross-domain FA contract drifted")
        else:
            raise ValueError("unexpected diagnostic event type")
    if len(false_rejects) != 2 or len(false_accepts) != 2:
        raise ValueError("expected two FR and two FA open-development events")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--diagnostics", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("evidence_class") != "development-only-open-shadow-failure-diagnostics":
        raise ValueError("unexpected GRU diagnostics evidence class")
    if diagnostics.get("policy") != EXPECTED_DIAGNOSTIC_POLICY:
        raise ValueError("unexpected GRU diagnostics policy")
    if (
        diagnostics.get("formal_qualification_used") is not False
        or diagnostics.get("formal_seed_consumed") is not False
    ):
        raise ValueError("cross-domain replay cannot consume formal qualification evidence")
    if [int(v) for v in diagnostics.get("open_development_seeds", [])] != EXPECTED_OPEN_SEEDS:
        raise ValueError("GRU diagnostics seed set drifted")
    if int(diagnostics.get("failure_event_count", -1)) != EXPECTED_FAILURE_EVENT_COUNT:
        raise ValueError("GRU diagnostics failure count drifted")
    if str(diagnostics.get("model_sha256")) != EXPECTED_SOURCE_MODEL_SHA256:
        raise ValueError("GRU diagnostics source model drifted")
    forbidden_source_material(diagnostics)

    cfg = load_config(config_path)
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
            if not isinstance(event.get("source"), dict):
                raise ValueError("diagnostic event lacks source metadata")
            events.append(event)
    require_event_contract(events)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wav_root = output / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    clean_root = output / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE
    manifest_rows: list[tuple[pathlib.Path, list[int]]] = []
    evidence_rows: list[dict] = []
    total_coverage = {
        "distance_bands": Counter(),
        "azimuth_deg": Counter(),
        "noise_profiles": Counter(),
        "playback_states": Counter(),
    }

    for event_index, event in enumerate(events):
        identity = event_identity(event)
        tokens = identity["tokens"]
        target_ids = identity["target_ids"]
        source_kind = identity["source_kind"]
        source_keyword = identity["source_keyword_id"]
        fingerprint = sha256_json(identity)
        generated: list[dict] = []
        event_coverage = {
            "distance_bands": Counter(),
            "azimuth_deg": Counter(),
            "noise_profiles": Counter(),
            "playback_states": Counter(),
        }

        for ordinal in range(EXAMPLES_PER_EVENT):
            seed = (
                base_seed
                + event_index * 1_000_003
                + ordinal * 65_537
                + int(fingerprint[:8], 16)
            ) & 0x7FFFFFFF
            rng = random.Random(seed)
            clean_path = clean_root / f"e{event_index:02d}-r{ordinal:02d}.wav"
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
                clean = generate_background("white", rng.uniform(1.2, 2.0), rng)
            augmented = augment(clean, rng, augment_cfg)
            scene, band_name, playback_state = lattice_scene(domains, event_index, ordinal, rng)
            rendered, scene_meta = render_scene(
                augmented,
                scene,
                seed=seed + 31_337,
                afe=domains["afe"],
            )
            path = wav_root / f"e{event_index:02d}-r{ordinal:02d}.wav"
            write_wav(path, rendered)
            manifest_rows.append((path, target_ids))

            azimuth_value = float(scene["azimuth_deg"])
            azimuth_key = str(int(azimuth_value) if azimuth_value.is_integer() else azimuth_value)
            event_coverage["distance_bands"][band_name] += 1
            event_coverage["azimuth_deg"][azimuth_key] += 1
            event_coverage["noise_profiles"][str(scene["noise_profile"])] += 1
            event_coverage["playback_states"][playback_state] += 1
            generated.append(
                {
                    "ordinal": ordinal,
                    "seed": seed,
                    "wav_sha256": sha256_file(path),
                    "distance_band": band_name,
                    "playback_state": playback_state,
                    "scene": scene_meta,
                }
            )

        expected_azimuths = {str(int(v) if float(v).is_integer() else v) for v in domains["azimuth_deg"]}
        if set(event_coverage["distance_bands"]) != set(EXPECTED_DISTANCE_BANDS):
            raise ValueError("per-event distance coverage incomplete")
        if set(event_coverage["azimuth_deg"]) != expected_azimuths:
            raise ValueError("per-event azimuth coverage incomplete")
        if set(event_coverage["noise_profiles"]) != {str(v) for v in domains["noise_profiles"]}:
            raise ValueError("per-event noise coverage incomplete")
        if set(event_coverage["playback_states"]) != {"playback", "no-playback"}:
            raise ValueError("per-event playback coverage incomplete")
        for key in total_coverage:
            total_coverage[key].update(event_coverage[key])

        evidence_rows.append(
            {
                "event_index": event_index,
                "source_identity": identity,
                "source_fingerprint_sha256": fingerprint,
                "source_scene_used": False,
                "coverage": {key: dict(sorted(counter.items())) for key, counter in event_coverage.items()},
                "generated": generated,
            }
        )

    manifest = output / "gru-generalization-cross-domain-replay.tsv"
    manifest.write_text(
        "".join(
            f"{path.resolve()}\t{' '.join(str(v) for v in targets)}\n"
            for path, targets in manifest_rows
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "formal_qualification_used": False,
        "calibration_evidence_used": False,
        "test_evidence_used": False,
        "shadow_seed_consumed": False,
        "shadow_wav_bytes_copied": False,
        "source_scene_used": False,
        "scene_generation_basis": "config-domain-lattice-only",
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "config_sha256": sha256_file(config_path),
        "seed_namespace": SEED_NAMESPACE,
        "examples_per_event": EXAMPLES_PER_EVENT,
        "source_event_count": len(events),
        "examples": len(manifest_rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "coverage": {key: dict(sorted(counter.items())) for key, counter in total_coverage.items()},
        "events": evidence_rows,
    }
    evidence_path = output / "gru-generalization-cross-domain-replay.json"
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
