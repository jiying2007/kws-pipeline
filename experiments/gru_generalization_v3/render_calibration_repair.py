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
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from development_failure_replay import _jitter_scene  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from render_domains import validate_domains  # noqa: E402
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

POLICY = "gru-v3-calibration-repair-resynthesis-v1"
SEED_NAMESPACE = 181_500_019
EXAMPLES_PER_FAILURE = 8
EXPECTED_CANDIDATE_RUN_ID = 34790831573
EXPECTED_CANDIDATE_MODEL_SHA256 = "8dc7d85505147fb6c9b402b2b07f6f0b047335207ca954800ac5e311c46a3c7d"


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
    parser.add_argument("--diagnostics", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    diagnostics_path = args.diagnostics.resolve()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("evidence_class") != "development-only-gru-v2-calibration-failure-diagnostics":
        raise ValueError("unexpected GRU V2 calibration diagnostic evidence class")
    if diagnostics.get("policy") != "gru-v2-calibration-failure-diagnostics-v1":
        raise ValueError("unexpected GRU V2 calibration diagnostic policy")
    if diagnostics.get("formal_qualification_used") is not False or diagnostics.get("formal_seed_consumed") is not False:
        raise ValueError("calibration repair cannot consume formal evidence")
    if diagnostics.get("shadow_used") is not False or diagnostics.get("shadow_seed_consumed") is not False:
        raise ValueError("calibration repair cannot consume shadow evidence")
    if int(diagnostics.get("candidate_run_id", -1)) != EXPECTED_CANDIDATE_RUN_ID:
        raise ValueError("calibration diagnostic candidate run drifted")
    if str(diagnostics.get("candidate_model_sha256")) != EXPECTED_CANDIDATE_MODEL_SHA256:
        raise ValueError("calibration diagnostic model identity drifted")
    if diagnostics.get("reproduced") != {"false_accepts": 0, "false_rejects": 1}:
        raise ValueError("calibration diagnostic failure count drifted")

    event = diagnostics.get("event")
    if not isinstance(event, dict) or event.get("event_type") != "false_reject":
        raise ValueError("calibration repair requires exactly one false reject")
    if int(event.get("expected_keyword_id", -1)) != 2:
        raise ValueError("calibration repair is frozen to keyword 2")
    if str(event.get("recording")) != "domain-calibration-000095":
        raise ValueError("calibration failure recording drifted")
    source = event.get("source")
    if not isinstance(source, dict):
        raise ValueError("calibration diagnostic lacks source metadata")
    if int(source.get("source_keyword_id", -1)) != 2:
        raise ValueError("calibration source keyword drifted")
    if str(source.get("kind")) != "positive":
        raise ValueError("calibration repair source must be positive")
    forbidden = {"path", "audio_path", "source_path", "wav_sha256", "source_wav_sha256"}
    if forbidden & set(source):
        raise ValueError("calibration diagnostic unexpectedly carries source WAV material")

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

    tokens = [str(v) for v in source.get("tokens", [])]
    target_ids = [int(v) for v in source.get("target_ids", [])]
    if tokens != ["xiao3", "wo1", "xiao3", "wo1"] or target_ids != [3, 4, 3, 4]:
        raise ValueError("calibration repair lexical target drifted")
    scene = source.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("calibration repair source lacks scene metadata")

    output = args.output.resolve()
    wav_root = output / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_json({"event": event, "diagnostics_sha256": sha256_file(diagnostics_path)})
    base_seed = int(cfg.get("seed", 1337)) + SEED_NAMESPACE + int(fingerprint[:8], 16)
    rows: list[dict] = []
    manifest_rows: list[tuple[pathlib.Path, list[int]]] = []
    for ordinal in range(EXAMPLES_PER_FAILURE):
        seed = (base_seed + ordinal * 65_537) & 0x7FFFFFFF
        rng = random.Random(seed)
        clean_path = output / "clean" / f"repair-{ordinal:02d}.wav"
        if str(tts.get("backend", "tone")) == "tone":
            clean = render_tone_tokens(tokens, carriers, rng, tts)
        else:
            clean = render_command_tts(
                keyword_text.get(2, "小窝小窝"),
                tokens,
                "positive",
                clean_path,
                tts,
            )
        augmented = augment(clean, rng, augment_cfg)
        rendered_scene = _jitter_scene(scene, domains, rng, ordinal)
        rendered, scene_meta = render_scene(
            augmented,
            rendered_scene,
            seed=seed + 31_337,
            afe=domains["afe"],
        )
        path = wav_root / f"repair-{ordinal:02d}.wav"
        write_wav(path, rendered)
        manifest_rows.append((path, target_ids))
        rows.append(
            {
                "ordinal": ordinal,
                "seed": seed,
                "wav_sha256": sha256_file(path),
                "scene": scene_meta,
            }
        )

    manifest = output / "gru-v3-calibration-repair.tsv"
    manifest.write_text(
        "".join(
            f"{path.resolve()}\t{' '.join(str(v) for v in targets)}\n"
            for path, targets in manifest_rows
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": "training-only-gru-v3-calibration-repair-resynthesis",
        "policy": POLICY,
        "formal_qualification_used": False,
        "shadow_used": False,
        "development_source_wav_bytes_copied": False,
        "source_split": "calibration",
        "source_recording": str(event["recording"]),
        "source_event_fingerprint_sha256": fingerprint,
        "source_candidate_run_id": EXPECTED_CANDIDATE_RUN_ID,
        "source_candidate_model_sha256": EXPECTED_CANDIDATE_MODEL_SHA256,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "config_sha256": sha256_file(config_path),
        "seed_namespace": SEED_NAMESPACE,
        "examples_per_failure": EXAMPLES_PER_FAILURE,
        "examples": len(rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "source": source,
        "generated": rows,
    }
    evidence_path = output / "gru-v3-calibration-repair.json"
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
