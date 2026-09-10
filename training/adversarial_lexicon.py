from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

from acoustic_scene import render_scene, sha256_file  # noqa: E402
from hard_negative_replay import apply_focus, hard_negative_stress_focus  # noqa: E402
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


def enumerate_safe_sequences(
    active_tokens: list[str], forbidden: list[list[str]], *, max_length: int = 5
) -> list[tuple[str, ...]]:
    if not active_tokens:
        raise ValueError("active token set is empty")
    if max_length <= 0 or max_length > 8:
        raise ValueError("adversarial max_length must be 1..8")
    rows: list[tuple[str, ...]] = []
    for length in range(1, max_length + 1):
        for sequence in itertools.product(active_tokens, repeat=length):
            if safe_negative(list(sequence), forbidden):
                rows.append(tuple(sequence))
    return rows


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _render_sequence(
    *,
    token_names: list[str],
    sequence_index: int,
    example_index: int,
    round_index: int,
    seed: int,
    carriers: dict,
    tts: dict,
    augment_config: dict,
    domains: dict,
    output_path: pathlib.Path | None,
) -> tuple[list[int], dict]:
    example_seed = (
        seed
        + 91_000_003
        + round_index * 1_000_003
        + sequence_index * 65_537
        + example_index * 4099
        + _stable_seed(" ".join(token_names))
    ) & 0x7FFFFFFF
    rng = random.Random(example_seed)
    backend = str(tts.get("backend", "tone"))
    if backend == "tone":
        clean = render_tone_tokens(token_names, carriers, rng, tts)
    elif backend == "command":
        if output_path is None:
            raise ValueError("command TTS adversarial probe requires an output path")
        clean = render_command_tts(
            " ".join(token_names), token_names, "adversarial-negative", output_path, tts
        )
    else:
        raise ValueError(f"unsupported adversarial TTS backend: {backend}")
    augmented = augment(clean, rng, augment_config)
    scene_seed = example_seed + 31_337
    scene_rng = random.Random(scene_seed)
    scene = sample_scene(domains, scene_rng, curriculum_weights=None, forced_band=None)
    focus = hard_negative_stress_focus(
        domains,
        round_index=round_index,
        item_index=sequence_index,
        example_index=example_index,
    )
    scene = apply_focus(scene, focus, domains)
    mono, scene_meta = render_scene(augmented, scene, seed=scene_seed, afe=domains["afe"])
    return mono, {
        "example_seed": example_seed,
        "scene_seed": scene_seed,
        "focus": focus,
        "scene": scene_meta,
    }


def mine_adversarial_lexicon(
    config_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    output: pathlib.Path,
    *,
    round_index: int,
    frontend: str,
) -> dict:
    cfg = load_config(config_path)
    iteration = cfg.get("domain_iteration", {})
    policy = iteration.get("adversarial_lexicon", {}) if isinstance(iteration, dict) else {}
    if not isinstance(policy, dict) or not bool(policy.get("enabled", False)):
        raise ValueError("adversarial lexicon policy is not enabled")
    max_length = int(policy.get("max_length", 5))
    top_k = int(policy.get("top_k", 24))
    probes_per_sequence = int(policy.get("probes_per_sequence", 2))
    replay_examples = int(policy.get("replay_examples_per_sequence", 4))
    if top_k <= 0 or probes_per_sequence <= 0 or replay_examples <= 0:
        raise ValueError("adversarial lexicon counts must be positive")

    tokens_path = pathlib.Path(str(cfg["tokens"]))
    keywords_path = pathlib.Path(str(cfg["keywords"]))
    if not tokens_path.is_absolute():
        tokens_path = (ROOT / tokens_path).resolve()
    if not keywords_path.is_absolute():
        keywords_path = (ROOT / keywords_path).resolve()
    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    carriers = token_carriers(keywords, int(cfg.get("model", {}).get("feature_dim", 32)))
    active_tokens = list(carriers)
    forbidden = [list(keyword["tokens"]) for keyword in keywords]
    candidates = enumerate_safe_sequences(active_tokens, forbidden, max_length=max_length)
    if not candidates:
        raise ValueError("adversarial lexicon enumeration produced no safe negatives")

    generator = cfg.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    tts = generator.get("tts", {"backend": "tone"})
    augment_config = generator.get("augment", {})
    if not isinstance(tts, dict) or not isinstance(augment_config, dict):
        raise ValueError("generator TTS/augment config is invalid")
    validate_tone_config(tts)
    validate_augment_config(augment_config)
    domains = validate_domains(cfg)

    from surrogate_score import keyword_confidences, load_checkpoint_model

    model, checkpoint = load_checkpoint_model(checkpoint_path)
    if str(checkpoint.get("frontend_name")) != frontend:
        raise ValueError("adversarial miner frontend differs from checkpoint")
    keyword_sequences = [list(keyword["token_ids"]) for keyword in keywords]
    feature_dim = int(checkpoint["feature_dim"])
    output.mkdir(parents=True, exist_ok=True)

    ranked: list[dict] = []
    base_seed = int(cfg.get("seed", 1337))
    for sequence_index, sequence in enumerate(candidates):
        probe_rows: list[dict] = []
        maximum = -1.0
        maximum_keyword = -1
        for probe_index in range(probes_per_sequence):
            scratch = output / "probe-clean" / f"q{sequence_index:04d}-p{probe_index:02d}.wav"
            samples, meta = _render_sequence(
                token_names=list(sequence),
                sequence_index=sequence_index,
                example_index=probe_index,
                round_index=round_index,
                seed=base_seed,
                carriers=carriers,
                tts=tts,
                augment_config=augment_config,
                domains=domains,
                output_path=scratch,
            )
            confidences = keyword_confidences(
                model,
                samples,
                feature_dim=feature_dim,
                frontend=frontend,
                keyword_sequences=keyword_sequences,
            )
            local_keyword = max(range(len(confidences)), key=lambda index: confidences[index])
            local_confidence = float(confidences[local_keyword])
            if local_confidence > maximum:
                maximum = local_confidence
                maximum_keyword = local_keyword
            probe_rows.append(
                {
                    "probe": probe_index,
                    "max_confidence": local_confidence,
                    "keyword_id": int(keywords[local_keyword]["id"]),
                    **meta,
                }
            )
        ranked.append(
            {
                "tokens": list(sequence),
                "target_ids": [int(token_map[token]) for token in sequence],
                "max_confidence": maximum,
                "focus_keyword_id": int(keywords[maximum_keyword]["id"]),
                "probes": probe_rows,
            }
        )

    ranked.sort(key=lambda item: (-float(item["max_confidence"]), tuple(item["tokens"])))
    selected = ranked[: min(top_k, len(ranked))]
    replay_rows: list[dict] = []
    for selected_index, item in enumerate(selected):
        for example_index in range(replay_examples):
            path = output / "wav" / f"a{selected_index:03d}-e{example_index:02d}.wav"
            samples, meta = _render_sequence(
                token_names=list(item["tokens"]),
                sequence_index=selected_index + 10_000,
                example_index=example_index + probes_per_sequence,
                round_index=round_index + 1,
                seed=base_seed,
                carriers=carriers,
                tts=tts,
                augment_config=augment_config,
                domains=domains,
                output_path=path,
            )
            write_wav(path, samples)
            replay_rows.append(
                {
                    "path": str(path.resolve()),
                    "tokens": list(item["tokens"]),
                    "target_ids": list(item["target_ids"]),
                    "focus_keyword_id": int(item["focus_keyword_id"]),
                    "wav_sha256": sha256_file(path),
                    **meta,
                }
            )

    manifest = output / "adversarial-hard-negatives.tsv"
    manifest.write_text(
        "".join(
            f"{row['path']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
            for row in replay_rows
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": 1,
        "evidence_class": "development-only-adversarial-lexicon",
        "round": round_index,
        "frontend": frontend,
        "max_length": max_length,
        "enumerated_sequences": len(candidates),
        "top_k": len(selected),
        "probes_per_sequence": probes_per_sequence,
        "replay_examples_per_sequence": replay_examples,
        "selected": selected,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "replay_examples": len(replay_rows),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "formal_qualification_used": False,
    }
    evidence_path = output / "adversarial-lexicon.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    evidence["evidence"] = str(evidence_path)
    return evidence
