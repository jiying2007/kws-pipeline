from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from render_domains import sample_scene, validate_domains  # noqa: E402
from synthetic_audio import (  # noqa: E402
    augment,
    load_config,
    parse_keywords,
    render_command_tts,
    render_tone_tokens,
    safe_negative,
    token_carriers,
    validate_augment_config,
    validate_tone_config,
    write_wav,
)

MAX_REPLAY_EXAMPLES_PER_SEQUENCE = 256


def _repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def normalize_hard_negative_replay(
    raw: object,
    *,
    active_tokens: list[str],
    forbidden: list[list[str]],
    token_map: dict[str, int],
) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("domain_iteration.hard_negative_replay must be a list")
    normalized: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for index, item in enumerate(raw):
        label = f"domain_iteration.hard_negative_replay[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        token_value = item.get("tokens")
        if not isinstance(token_value, list) or not token_value:
            raise ValueError(f"{label}.tokens must be a non-empty list")
        tokens = [str(value) for value in token_value]
        if any(not token for token in tokens):
            raise ValueError(f"{label}.tokens must contain non-empty strings")
        missing = [token for token in tokens if token not in active_tokens]
        if missing:
            raise ValueError(
                f"{label}.tokens contain tokens without synthetic carriers: {', '.join(missing)}"
            )
        if not safe_negative(tokens, forbidden):
            raise ValueError(f"{label}.tokens contain a configured wake path")
        key = tuple(tokens)
        if key in seen:
            raise ValueError(f"{label}.tokens duplicate an earlier replay sequence")
        examples = int(item.get("examples", 1))
        if not 1 <= examples <= MAX_REPLAY_EXAMPLES_PER_SEQUENCE:
            raise ValueError(
                f"{label}.examples must be 1..{MAX_REPLAY_EXAMPLES_PER_SEQUENCE}"
            )
        normalized.append(
            {
                "tokens": tokens,
                "target_ids": [int(token_map[token]) for token in tokens],
                "examples": examples,
            }
        )
        seen.add(key)
    return normalized


def render_hard_negative_replay(
    config_path: pathlib.Path,
    output: pathlib.Path,
    *,
    round_index: int,
    curriculum_weights: dict | None,
) -> dict:
    if round_index < 0:
        raise ValueError("hard-negative replay round index must be >= 0")
    config = load_config(config_path)
    iteration = config.get("domain_iteration", {})
    if not isinstance(iteration, dict):
        raise ValueError("domain_iteration must be an object")

    tokens_path = _repo_path(str(config["tokens"]))
    keywords_path = _repo_path(str(config["keywords"]))
    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    feature_dim = int(config.get("model", {}).get("feature_dim", 32))
    carriers = token_carriers(keywords, feature_dim)
    active_tokens = list(carriers)
    forbidden = [list(keyword["tokens"]) for keyword in keywords]
    sequences = normalize_hard_negative_replay(
        iteration.get("hard_negative_replay", []),
        active_tokens=active_tokens,
        forbidden=forbidden,
        token_map=token_map,
    )

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "hard-negatives.tsv"
    evidence_path = output / "hard-negatives.json"
    if not sequences:
        manifest.write_text("", encoding="utf-8")
        evidence = {
            "schema_version": 1,
            "round": round_index,
            "examples": 0,
            "sequences": [],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {**evidence, "evidence": str(evidence_path)}

    generator = config.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    tts = generator.get("tts", {"backend": "tone"})
    if not isinstance(tts, dict):
        raise ValueError("generator.tts must be an object")
    validate_tone_config(tts)
    backend = str(tts.get("backend", "tone"))
    if backend not in {"tone", "command"}:
        raise ValueError(f"unsupported hard-negative TTS backend: {backend}")
    if backend == "command":
        command = tts.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("command TTS backend requires generator.tts.command argv list")
    augment_config = generator.get("augment", {})
    validate_augment_config(augment_config)
    domains = validate_domains(config)

    seed = int(config.get("seed", 1337))
    rows: list[dict] = []
    for sequence_index, sequence in enumerate(sequences):
        token_names = list(sequence["tokens"])
        target_ids = list(sequence["target_ids"])
        for example_index in range(int(sequence["examples"])):
            example_seed = (
                seed
                + 70_000_019
                + round_index * 1_000_003
                + sequence_index * 65_537
                + example_index * 4099
            )
            rng = random.Random(example_seed)
            clean_path = output / "clean" / f"h{sequence_index:02d}-e{example_index:03d}.wav"
            if backend == "tone":
                clean = render_tone_tokens(token_names, carriers, rng, tts)
            else:
                clean = render_command_tts(
                    " ".join(token_names),
                    token_names,
                    "hard-negative",
                    clean_path,
                    tts,
                )
            augmented = augment(clean, rng, augment_config)
            scene_seed = example_seed + 31_337
            scene_rng = random.Random(scene_seed)
            scene = sample_scene(
                domains,
                scene_rng,
                curriculum_weights=curriculum_weights,
                forced_band=None,
            )
            mono, scene_meta = render_scene(
                augmented,
                scene,
                seed=scene_seed,
                afe=domains["afe"],
            )
            wav_path = output / "wav" / f"h{sequence_index:02d}-e{example_index:03d}.wav"
            write_wav(wav_path, mono)
            rows.append(
                {
                    "path": str(wav_path.resolve()),
                    "tokens": token_names,
                    "target_ids": target_ids,
                    "example_seed": example_seed,
                    "scene_seed": scene_seed,
                    "scene": scene_meta,
                    "wav_sha256": sha256_file(wav_path),
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
        "round": round_index,
        "examples": len(rows),
        "sequences": [
            {"tokens": item["tokens"], "examples": int(item["examples"])}
            for item in sequences
        ],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**evidence, "evidence": str(evidence_path)}
