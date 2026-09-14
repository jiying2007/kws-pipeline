#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
V2 = ROOT / "experiments" / "gru_generalization_v2"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(V2))

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from render_domains import validate_domains  # noqa: E402
from render_replay import scene_variant  # noqa: E402
from synthetic_audio import (  # noqa: E402
    augment,
    load_config,
    parse_keywords,
    render_command_tts,
    render_tone_tokens,
    token_carriers,
    validate_augment_config,
    validate_tone_config,
    write_wav,
)

POLICY = "gru-v4-development-confusable-resynthesis-v1"
EVIDENCE_CLASS = "training-only-gru-v4-development-confusable-resynthesis"
SOURCE_EVIDENCE_CLASS = "development-only-gru-v3-failure-source-metadata"
SEED_NAMESPACE = 271_000_011
EXPECTED_TOKENS = ["ni3", "hao3", "xiao3", "xiao3"]
EXPECTED_TARGET_IDS = [1, 2, 3, 3]
EXPECTED_KEYWORD_ID = 1
EXPECTED_EXAMPLES = 16


def sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--source-metadata", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--examples", type=int, default=EXPECTED_EXAMPLES)
    args = parser.parse_args()
    if args.examples != EXPECTED_EXAMPLES:
        raise ValueError(f"GRU V4 replay is frozen at exactly {EXPECTED_EXAMPLES} examples")

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    source_path = args.source_metadata.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("evidence_class") != SOURCE_EVIDENCE_CLASS:
        raise ValueError("unexpected V3 failure source evidence class")
    if source.get("policy") != "metadata-only-no-wav-copy-v1":
        raise ValueError("unexpected V3 failure source metadata policy")
    if source.get("formal_qualification_used") is not False:
        raise ValueError("V4 replay cannot consume formal qualification evidence")
    if source.get("shadow_used") is not False:
        raise ValueError("V4 replay cannot consume reserved shadow evidence")
    if source.get("wav_bytes_copied") is not False:
        raise ValueError("V4 replay cannot consume copied evaluation WAV bytes")

    index = source.get("domain_index")
    reference = source.get("reference")
    false_positive = source.get("false_positive")
    if not isinstance(index, dict) or not isinstance(reference, dict) or not isinstance(false_positive, dict):
        raise ValueError("V3 failure source metadata is incomplete")
    tokens = [str(v) for v in index.get("tokens", [])]
    target_ids = [int(v) for v in index.get("target_ids", [])]
    if tokens != EXPECTED_TOKENS or target_ids != EXPECTED_TARGET_IDS:
        raise ValueError("V3 failure confusable identity drifted")
    if str(index.get("kind")) != "confusable" or str(reference.get("kind")) != "confusable":
        raise ValueError("V4 replay source must be a confusable negative")
    if reference.get("expected") != []:
        raise ValueError("V4 replay source must have no expected wake")
    if int(index.get("keyword_id", -1)) != EXPECTED_KEYWORD_ID:
        raise ValueError("V4 replay source keyword identity drifted")
    if int(false_positive.get("keyword_id", -1)) != EXPECTED_KEYWORD_ID:
        raise ValueError("V4 false-positive keyword identity drifted")
    if float(false_positive.get("confidence", 0.0)) < 0.8:
        raise ValueError("V4 source is no longer the diagnosed high-confidence false accept")
    scene = index.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("V4 replay source scene is missing")

    token_path = pathlib.Path(str(cfg["tokens"]))
    keyword_path = pathlib.Path(str(cfg["keywords"]))
    if not token_path.is_absolute():
        token_path = (ROOT / token_path).resolve()
    if not keyword_path.is_absolute():
        keyword_path = (ROOT / keyword_path).resolve()
    token_map = load_tokens(token_path)
    keywords = parse_keywords(keyword_path, token_map)
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

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wav_root = output / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    source_identity = {
        "recording": str(source.get("recording", "")),
        "wav_name": str(source.get("wav_name", "")),
        "tokens": tokens,
        "target_ids": target_ids,
        "scene": scene,
        "false_positive": false_positive,
    }
    fingerprint = sha256_json(source_identity)
    base_seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE + int(fingerprint[:8], 16)
    generated: list[dict] = []
    manifest_rows: list[tuple[pathlib.Path, list[int]]] = []
    source_wav_sha = str(index.get("wav_sha256", ""))

    for ordinal in range(args.examples):
        seed = (base_seed + ordinal * 65_537) & 0x7FFFFFFF
        rng = random.Random(seed)
        clean_path = output / "clean" / f"r{ordinal:02d}.wav"
        if str(tts.get("backend", "tone")) == "tone":
            clean = render_tone_tokens(tokens, carriers, rng, tts)
        else:
            clean = render_command_tts(" ".join(tokens), tokens, "confusable", clean_path, tts)
        augmented = augment(clean, rng, augment_cfg)
        variant = scene_variant(scene, domains, rng, ordinal)
        rendered, scene_meta = render_scene(
            augmented,
            variant,
            seed=seed + 31_337,
            afe=domains["afe"],
        )
        path = wav_root / f"r{ordinal:02d}.wav"
        write_wav(path, rendered)
        digest = sha256_file(path)
        if source_wav_sha and digest == source_wav_sha:
            raise ValueError("V4 replay unexpectedly reproduced source evaluation WAV bytes")
        manifest_rows.append((path, target_ids))
        generated.append(
            {
                "ordinal": ordinal,
                "seed": seed,
                "wav_sha256": digest,
                "scene": scene_meta,
            }
        )

    manifest = output / "gru-v4-development-confusable-replay.tsv"
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
        "shadow_used": False,
        "source_wav_bytes_copied": False,
        "development_qualification_feedback_used": True,
        "source_metadata_sha256": sha256_file(source_path),
        "source_fingerprint_sha256": fingerprint,
        "config_sha256": sha256_file(config_path),
        "seed_namespace": SEED_NAMESPACE,
        "examples": len(manifest_rows),
        "effective_exposure_with_repeat_3": len(manifest_rows) * 3,
        "tokens": tokens,
        "target_ids": target_ids,
        "focus_keyword_id": EXPECTED_KEYWORD_ID,
        "source_scene": scene,
        "generated": generated,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
    }
    evidence_path = output / "gru-v4-development-confusable-replay.json"
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
