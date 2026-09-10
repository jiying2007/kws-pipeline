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

SELECTION_POLICY = "balanced-strict-prefix-anchor-topk-v2"


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


def strict_prefix_anchors(keywords: list[dict]) -> set[tuple[str, ...]]:
    anchors: set[tuple[str, ...]] = set()
    for keyword in keywords:
        tokens = [str(token) for token in keyword.get("tokens", [])]
        for length in range(1, len(tokens)):
            anchors.add(tuple(tokens[:length]))
    return anchors


def select_adversarial_candidates(
    ranked: list[dict],
    keywords: list[dict],
    *,
    top_k: int,
    min_per_keyword: int = 0,
    include_strict_prefix_anchors: bool = False,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("adversarial top_k must be positive")
    if min_per_keyword < 0:
        raise ValueError("adversarial min_per_keyword must be >= 0")
    keyword_ids = [int(keyword["id"]) for keyword in keywords]
    if len(set(keyword_ids)) != len(keyword_ids) or not keyword_ids:
        raise ValueError("adversarial keyword ids must be non-empty and unique")
    if min_per_keyword * len(keyword_ids) > top_k:
        raise ValueError("adversarial per-keyword quota exceeds top_k capacity")

    limit = min(top_k, len(ranked))
    if limit <= 0:
        return []
    anchors = strict_prefix_anchors(keywords) if include_strict_prefix_anchors else set()
    selected: list[dict] = []
    seen: set[tuple[str, ...]] = set()

    def add(item: dict, reason: str) -> bool:
        tokens = tuple(str(token) for token in item.get("tokens", []))
        if not tokens or tokens in seen or len(selected) >= limit:
            return False
        row = dict(item)
        reasons = list(row.get("selection_reasons", []))
        if reason not in reasons:
            reasons.append(reason)
        row["selection_reasons"] = reasons
        selected.append(row)
        seen.add(tokens)
        return True

    # These anchors encode terminal-completion failure families generically from
    # configured wake paths. No formal clip, formal seed or formal scene is used.
    if anchors:
        for item in ranked:
            if tuple(str(token) for token in item.get("tokens", [])) in anchors:
                add(item, "strict-prefix-anchor")
        missing = anchors - seen
        if missing:
            raise ValueError(
                "adversarial ranking omitted configured strict-prefix anchors: "
                + ", ".join(" ".join(tokens) for tokens in sorted(missing))
            )

    # Reserve capacity for both shipping keywords before the global hardest fill.
    for keyword_id in keyword_ids:
        have = sum(int(item.get("focus_keyword_id", -1)) == keyword_id for item in selected)
        if have >= min_per_keyword:
            continue
        for item in ranked:
            if int(item.get("focus_keyword_id", -1)) != keyword_id:
                continue
            if add(item, f"keyword-{keyword_id}-quota"):
                have += 1
            if have >= min_per_keyword:
                break
        if have < min_per_keyword:
            raise ValueError(
                f"adversarial ranking cannot satisfy keyword {keyword_id} quota "
                f"{min_per_keyword}; selected={have}"
            )

    for item in ranked:
        if len(selected) >= limit:
            break
        add(item, "global-hardest-fill")

    if len(selected) != limit:
        raise ValueError(
            f"adversarial selection produced {len(selected)} entries, expected {limit}"
        )
    return selected


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


def _effective_policy(cfg: dict, legacy: dict) -> dict:
    data_v3 = cfg.get("data_augmentation_v3", {})
    if data_v3 is None:
        data_v3 = {}
    if not isinstance(data_v3, dict):
        raise ValueError("data_augmentation_v3 must be an object")
    if data_v3:
        if str(data_v3.get("policy")) != "train-only-balanced-mining-v1":
            raise ValueError("unsupported data_augmentation_v3 policy")
        if data_v3.get("formal_qualification_used") is not False:
            raise ValueError("data v3 must not use formal qualification")
        if data_v3.get("expand_evaluation_splits") is not False:
            raise ValueError("data v3 must not expand calibration/test/qualification")
    return {
        "data_policy": str(data_v3.get("policy") or "legacy"),
        "top_k": int(data_v3.get("adversarial_top_k", legacy.get("top_k", 24))),
        "probes_per_sequence": int(
            data_v3.get("adversarial_probes_per_sequence", legacy.get("probes_per_sequence", 2))
        ),
        "replay_examples_per_sequence": int(
            data_v3.get(
                "adversarial_replay_examples_per_sequence",
                legacy.get("replay_examples_per_sequence", 4),
            )
        ),
        "min_per_keyword": int(data_v3.get("adversarial_min_per_keyword", 0)),
        "include_strict_prefix_anchors": bool(
            data_v3.get("adversarial_include_strict_prefix_anchors", False)
        ),
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
    legacy = iteration.get("adversarial_lexicon", {}) if isinstance(iteration, dict) else {}
    if not isinstance(legacy, dict) or not bool(legacy.get("enabled", False)):
        raise ValueError("adversarial lexicon policy is not enabled")
    max_length = int(legacy.get("max_length", 5))
    effective = _effective_policy(cfg, legacy)
    top_k = int(effective["top_k"])
    probes_per_sequence = int(effective["probes_per_sequence"])
    replay_examples = int(effective["replay_examples_per_sequence"])
    min_per_keyword = int(effective["min_per_keyword"])
    include_prefix_anchors = bool(effective["include_strict_prefix_anchors"])
    if top_k <= 0 or probes_per_sequence <= 0 or replay_examples <= 0:
        raise ValueError("adversarial lexicon counts must be positive")
    if min_per_keyword < 0:
        raise ValueError("adversarial min_per_keyword must be >= 0")

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
    selected = select_adversarial_candidates(
        ranked,
        keywords,
        top_k=top_k,
        min_per_keyword=min_per_keyword,
        include_strict_prefix_anchors=include_prefix_anchors,
    )
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
    per_keyword_selected = {
        str(keyword["id"]): sum(
            int(item["focus_keyword_id"]) == int(keyword["id"]) for item in selected
        )
        for keyword in keywords
    }
    anchors = strict_prefix_anchors(keywords) if include_prefix_anchors else set()
    evidence = {
        "schema_version": 2,
        "evidence_class": "development-only-adversarial-lexicon",
        "data_augmentation_policy": str(effective["data_policy"]),
        "selection_policy": SELECTION_POLICY,
        "round": round_index,
        "frontend": frontend,
        "max_length": max_length,
        "enumerated_sequences": len(candidates),
        "top_k": len(selected),
        "probes_per_sequence": probes_per_sequence,
        "replay_examples_per_sequence": replay_examples,
        "min_per_keyword": min_per_keyword,
        "include_strict_prefix_anchors": include_prefix_anchors,
        "strict_prefix_anchors": [list(tokens) for tokens in sorted(anchors)],
        "per_keyword_selected": per_keyword_selected,
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
