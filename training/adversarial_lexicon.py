from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import os
import pathlib
import random
import struct
import sys
import wave

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
MULTI_KEYWORD_SELECTION_POLICY = "per-keyword-confidence-quota-topk-v1"
MAX_ADVERSARIAL_SEQUENCE_LENGTH = 16
COMMAND_TTS_EXECUTION_POLICY = "bounded-parallel-clean-prerender-v1"
MAX_COMMAND_TTS_WORKERS = 4
CANDIDATE_POLICY = "exhaustive-when-bounded-else-hybrid-v1"
HYBRID_CANDIDATE_POLICY = "prefix-edit-deterministic-sample-v1"


def enumerate_safe_sequences(
    active_tokens: list[str],
    forbidden: list[list[str]],
    *,
    max_length: int = 5,
    max_sequences: int | None = None,
) -> list[tuple[str, ...]]:
    if not active_tokens:
        raise ValueError("active token set is empty")
    if max_length <= 0 or max_length > MAX_ADVERSARIAL_SEQUENCE_LENGTH:
        raise ValueError(
            f"adversarial max_length must be 1..{MAX_ADVERSARIAL_SEQUENCE_LENGTH}"
        )
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("adversarial max_sequences must be positive")
    rows: list[tuple[str, ...]] = []
    for length in range(1, max_length + 1):
        for sequence in itertools.product(active_tokens, repeat=length):
            if safe_negative(list(sequence), forbidden):
                rows.append(tuple(sequence))
                if max_sequences is not None and len(rows) > max_sequences:
                    raise ValueError(
                        "adversarial safe-sequence search exceeds configured budget: "
                        f"{len(rows)} > {max_sequences}"
                    )
    return rows


def strict_prefix_anchors(keywords: list[dict]) -> set[tuple[str, ...]]:
    wake_paths = {
        tuple(str(token) for token in keyword.get("tokens", []))
        for keyword in keywords
    }
    anchors: set[tuple[str, ...]] = set()
    for keyword in keywords:
        tokens = [str(token) for token in keyword.get("tokens", [])]
        for length in range(1, len(tokens)):
            anchor = tuple(tokens[:length])
            # A configured wake path must never become a hard negative merely
            # because it is a strict prefix of another configured wake path.
            if anchor not in wake_paths:
                anchors.add(anchor)
    return anchors


def effective_adversarial_max_length(
    configured_max_length: int,
    keywords: list[dict],
) -> int:
    if configured_max_length <= 0:
        raise ValueError("adversarial configured max_length must be positive")
    longest_prefix = max(
        (max(1, len(keyword.get("tokens", [])) - 1) for keyword in keywords),
        default=1,
    )
    result = max(configured_max_length, longest_prefix)
    if result > MAX_ADVERSARIAL_SEQUENCE_LENGTH:
        raise ValueError(
            "configured wake path requires adversarial prefix length "
            f"{result}, exceeding {MAX_ADVERSARIAL_SEQUENCE_LENGTH}"
        )
    return result


def _raw_cartesian_count(active_token_count: int, max_length: int) -> int:
    if active_token_count <= 0:
        return 0
    return sum(active_token_count**length for length in range(1, max_length + 1))


def _keyword_edit_candidates(
    tokens: tuple[str, ...],
    active_tokens: list[str],
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(value: tuple[str, ...]) -> None:
        if value and value != tokens and value not in seen:
            seen.add(value)
            rows.append(value)

    for index in range(len(tokens)):
        add(tokens[:index] + tokens[index + 1 :])
    for index, original in enumerate(tokens):
        for replacement in active_tokens:
            if replacement == original:
                continue
            value = list(tokens)
            value[index] = replacement
            add(tuple(value))
    for index in range(len(tokens) - 1):
        value = list(tokens)
        value[index], value[index + 1] = value[index + 1], value[index]
        add(tuple(value))
    return rows


def build_adversarial_candidate_plan(
    *,
    active_tokens: list[str],
    keywords: list[dict],
    max_length: int,
    max_sequences: int,
    hybrid_candidate_budget: int,
    seed: int,
) -> dict:
    if not active_tokens:
        raise ValueError("active token set is empty")
    if max_length <= 0 or max_length > MAX_ADVERSARIAL_SEQUENCE_LENGTH:
        raise ValueError(
            f"adversarial max_length must be 1..{MAX_ADVERSARIAL_SEQUENCE_LENGTH}"
        )
    if max_sequences <= 0 or hybrid_candidate_budget <= 0:
        raise ValueError("adversarial candidate budgets must be positive")
    forbidden = [list(keyword.get("tokens", [])) for keyword in keywords]
    raw_count = _raw_cartesian_count(len(active_tokens), max_length)
    if raw_count <= max_sequences:
        candidates = enumerate_safe_sequences(
            active_tokens,
            forbidden,
            max_length=max_length,
            max_sequences=max_sequences,
        )
        return {
            "policy": CANDIDATE_POLICY,
            "mode": "exhaustive",
            "raw_cartesian_sequences": raw_count,
            "candidates": candidates,
            "strict_prefix_anchors": sorted(strict_prefix_anchors(keywords)),
            "sample_attempts": 0,
        }

    candidate_budget = min(max_sequences, hybrid_candidate_budget)
    anchors = sorted(
        strict_prefix_anchors(keywords),
        key=lambda value: (len(value), value),
    )
    if len(anchors) > candidate_budget:
        raise ValueError(
            "adversarial strict-prefix anchors exceed hybrid candidate budget: "
            f"{len(anchors)} > {candidate_budget}"
        )

    candidates: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(value: tuple[str, ...]) -> bool:
        if (
            not value
            or value in seen
            or not safe_negative(list(value), forbidden)
            or len(candidates) >= candidate_budget
        ):
            return False
        seen.add(value)
        candidates.append(value)
        return True

    for anchor in anchors:
        add(anchor)

    per_keyword_edits = [
        [
            value
            for value in _keyword_edit_candidates(
                tuple(str(token) for token in keyword.get("tokens", [])),
                active_tokens,
            )
            if safe_negative(list(value), forbidden)
        ]
        for keyword in keywords
    ]
    edit_index = 0
    while len(candidates) < candidate_budget:
        progressed = False
        for rows in per_keyword_edits:
            if edit_index < len(rows):
                progressed = add(rows[edit_index]) or progressed
        edit_index += 1
        if not progressed and all(edit_index >= len(rows) for rows in per_keyword_edits):
            break

    seed_material = json.dumps(
        {
            "seed": seed,
            "active_tokens": active_tokens,
            "keywords": [list(keyword.get("tokens", [])) for keyword in keywords],
            "max_length": max_length,
            "max_sequences": max_sequences,
            "hybrid_candidate_budget": hybrid_candidate_budget,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rng = random.Random(_stable_seed(seed_material))
    attempts = 0
    max_attempts = max(10_000, candidate_budget * 128)
    while len(candidates) < candidate_budget and attempts < max_attempts:
        attempts += 1
        length = rng.randint(1, max_length)
        value = tuple(rng.choice(active_tokens) for _ in range(length))
        add(value)

    if not candidates:
        raise ValueError("bounded adversarial candidate generation produced no safe negatives")
    return {
        "policy": CANDIDATE_POLICY,
        "mode": HYBRID_CANDIDATE_POLICY,
        "raw_cartesian_sequences": raw_count,
        "hybrid_candidate_budget": candidate_budget,
        "candidates": candidates,
        "strict_prefix_anchors": anchors,
        "sample_attempts": attempts,
    }


def effective_adversarial_top_k(
    *,
    keyword_count: int,
    configured_top_k: int,
    min_per_keyword: int,
    global_hardest_fill: int,
    strict_prefix_anchor_count: int,
    max_selected_sequences: int,
) -> int:
    if keyword_count <= 0:
        raise ValueError("adversarial keyword count must be positive")
    if configured_top_k <= 0 or min_per_keyword < 0 or global_hardest_fill < 0:
        raise ValueError("adversarial selection budgets are invalid")
    if max_selected_sequences <= 0:
        raise ValueError("adversarial max selected sequences must be positive")
    required = max(
        configured_top_k,
        min_per_keyword * keyword_count + global_hardest_fill,
        strict_prefix_anchor_count,
    )
    if required > max_selected_sequences:
        raise ValueError(
            "adversarial selection budget exceeds configured maximum: "
            f"{required} > {max_selected_sequences}"
        )
    return required


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

    multi_keyword_mode = len(keyword_ids) > 2

    def add(
        item: dict,
        reason: str,
        *,
        focus_keyword_id: int | None = None,
    ) -> bool:
        tokens = tuple(str(token) for token in item.get("tokens", []))
        if not tokens or tokens in seen or len(selected) >= limit:
            return False
        row = dict(item)
        if focus_keyword_id is not None:
            row["focus_keyword_id"] = focus_keyword_id
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

    # Reserve capacity for every configured keyword before the global hardest fill.
    # Preserve the exact historical 2-keyword behavior. For 3+ keywords, rank
    # quota candidates by that keyword's own surrogate confidence so a weak
    # keyword cannot disappear merely because another keyword wins the global max.
    for keyword_id in keyword_ids:
        have = sum(int(item.get("focus_keyword_id", -1)) == keyword_id for item in selected)
        if have >= min_per_keyword:
            continue
        if multi_keyword_mode:
            quota_ranked = sorted(
                ranked,
                key=lambda item: (
                    -float(
                        item.get("per_keyword_max_confidence", {}).get(
                            str(keyword_id), -1.0
                        )
                    ),
                    tuple(str(token) for token in item.get("tokens", [])),
                ),
            )
        else:
            quota_ranked = ranked
        for item in quota_ranked:
            if not multi_keyword_mode and int(item.get("focus_keyword_id", -1)) != keyword_id:
                continue
            per_keyword = item.get("per_keyword_max_confidence", {})
            if multi_keyword_mode and (
                not isinstance(per_keyword, dict)
                or str(keyword_id) not in per_keyword
            ):
                raise ValueError(
                    f"adversarial ranking is missing keyword {keyword_id} confidence"
                )
            if add(
                item,
                f"keyword-{keyword_id}-quota",
                focus_keyword_id=keyword_id if multi_keyword_mode else None,
            ):
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



def _command_tts_worker_count(task_count: int, *, cpu_count: int | None = None) -> int:
    if task_count <= 0:
        return 0
    available = os.cpu_count() if cpu_count is None else cpu_count
    if available is None or available <= 0:
        available = 1
    return min(MAX_COMMAND_TTS_WORKERS, int(available), task_count)


def _read_command_tts_output(path: pathlib.Path) -> list[int]:
    if not path.is_file():
        raise ValueError(f"pre-rendered command TTS WAV is missing: {path}")
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError("pre-rendered command TTS must be mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError("pre-rendered command TTS emitted an empty WAV")
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def _pre_render_command_tts(
    tasks: list[tuple[list[str], pathlib.Path]],
    tts: dict,
    *,
    workers: int | None = None,
) -> int:
    if str(tts.get("backend", "tone")) != "command" or not tasks:
        return 0
    worker_count = (
        _command_tts_worker_count(len(tasks))
        if workers is None
        else min(max(1, int(workers)), len(tasks))
    )

    def render(task: tuple[list[str], pathlib.Path]) -> None:
        token_names, output_path = task
        render_command_tts(
            " ".join(token_names),
            token_names,
            "adversarial-negative",
            output_path,
            tts,
        )

    if worker_count == 1:
        for task in tasks:
            render(task)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(render, tasks))
    return worker_count


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
    command_tts_pre_rendered: bool = False,
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
        if command_tts_pre_rendered:
            clean = _read_command_tts_output(output_path)
        else:
            clean = render_command_tts(
                " ".join(token_names),
                token_names,
                "adversarial-negative",
                output_path,
                tts,
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
        "global_hardest_fill": int(data_v3.get("adversarial_global_hardest_fill", 16)),
        "hybrid_candidate_budget": int(
            data_v3.get("adversarial_hybrid_candidate_budget", 2048)
        ),
        "max_selected_sequences": int(
            data_v3.get("adversarial_max_selected_sequences", 256)
        ),
        "candidate_policy": str(
            data_v3.get("adversarial_candidate_policy") or CANDIDATE_POLICY
        ),
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
        "max_enumerated_sequences": int(
            data_v3.get(
                "adversarial_max_enumerated_sequences",
                legacy.get("max_enumerated_sequences", 4096),
            )
        ),
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
    configured_max_length = int(legacy.get("max_length", 5))
    effective = _effective_policy(cfg, legacy)
    configured_top_k = int(effective["top_k"])
    global_hardest_fill = int(effective["global_hardest_fill"])
    hybrid_candidate_budget = int(effective["hybrid_candidate_budget"])
    max_selected_sequences = int(effective["max_selected_sequences"])
    candidate_policy = str(effective["candidate_policy"])
    probes_per_sequence = int(effective["probes_per_sequence"])
    replay_examples = int(effective["replay_examples_per_sequence"])
    min_per_keyword = int(effective["min_per_keyword"])
    max_enumerated_sequences = int(effective["max_enumerated_sequences"])
    include_prefix_anchors = bool(effective["include_strict_prefix_anchors"])
    if configured_top_k <= 0 or probes_per_sequence <= 0 or replay_examples <= 0:
        raise ValueError("adversarial lexicon counts must be positive")
    if candidate_policy != CANDIDATE_POLICY:
        raise ValueError(f"unsupported adversarial candidate policy: {candidate_policy}")
    if (
        global_hardest_fill < 0
        or hybrid_candidate_budget <= 0
        or max_selected_sequences <= 0
    ):
        raise ValueError("adversarial selection/candidate budgets are invalid")
    if min_per_keyword < 0:
        raise ValueError("adversarial min_per_keyword must be >= 0")
    if max_enumerated_sequences <= 0:
        raise ValueError("adversarial max_enumerated_sequences must be positive")

    tokens_path = pathlib.Path(str(cfg["tokens"]))
    keywords_path = pathlib.Path(str(cfg["keywords"]))
    if not tokens_path.is_absolute():
        tokens_path = (ROOT / tokens_path).resolve()
    if not keywords_path.is_absolute():
        keywords_path = (ROOT / keywords_path).resolve()
    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    max_length = effective_adversarial_max_length(configured_max_length, keywords)
    carriers = token_carriers(keywords, int(cfg.get("model", {}).get("feature_dim", 32)))
    active_tokens = list(carriers)
    candidate_plan = build_adversarial_candidate_plan(
        active_tokens=active_tokens,
        keywords=keywords,
        max_length=max_length,
        max_sequences=max_enumerated_sequences,
        hybrid_candidate_budget=hybrid_candidate_budget,
        seed=int(cfg.get("seed", 1337)),
    )
    candidates = list(candidate_plan["candidates"])
    anchors = strict_prefix_anchors(keywords) if include_prefix_anchors else set()
    top_k = effective_adversarial_top_k(
        keyword_count=len(keywords),
        configured_top_k=configured_top_k,
        min_per_keyword=min_per_keyword,
        global_hardest_fill=global_hardest_fill,
        strict_prefix_anchor_count=len(anchors),
        max_selected_sequences=max_selected_sequences,
    )
    if len(candidates) < top_k:
        raise ValueError(
            "adversarial candidate pool is smaller than the selection budget: "
            f"{len(candidates)} < {top_k}"
        )

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
    command_backend = str(tts.get("backend", "tone")) == "command"
    probe_tasks = [
        (
            list(sequence),
            output / "probe-clean" / f"q{sequence_index:04d}-p{probe_index:02d}.wav",
        )
        for sequence_index, sequence in enumerate(candidates)
        for probe_index in range(probes_per_sequence)
    ]
    probe_tts_workers = _pre_render_command_tts(probe_tasks, tts)

    for sequence_index, sequence in enumerate(candidates):
        probe_rows: list[dict] = []
        maximum = -1.0
        maximum_keyword = -1
        per_keyword_maximum = {
            str(keyword["id"]): -1.0 for keyword in keywords
        }
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
                command_tts_pre_rendered=command_backend,
            )
            confidences = keyword_confidences(
                model,
                samples,
                feature_dim=feature_dim,
                frontend=frontend,
                keyword_sequences=keyword_sequences,
            )
            for keyword_index, confidence in enumerate(confidences):
                keyword_id = str(keywords[keyword_index]["id"])
                per_keyword_maximum[keyword_id] = max(
                    per_keyword_maximum[keyword_id],
                    float(confidence),
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
                "per_keyword_max_confidence": per_keyword_maximum,
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
    replay_tasks = [
        (
            list(item["tokens"]),
            output / "wav" / f"a{selected_index:03d}-e{example_index:02d}.wav",
        )
        for selected_index, item in enumerate(selected)
        for example_index in range(replay_examples)
    ]
    replay_tts_workers = _pre_render_command_tts(replay_tasks, tts)

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
                command_tts_pre_rendered=command_backend,
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
    evidence = {
        "schema_version": 2,
        "evidence_class": "development-only-adversarial-lexicon",
        "data_augmentation_policy": str(effective["data_policy"]),
        "selection_policy": (
            MULTI_KEYWORD_SELECTION_POLICY if len(keywords) > 2 else SELECTION_POLICY
        ),
        "candidate_policy": str(candidate_plan["policy"]),
        "candidate_mode": str(candidate_plan["mode"]),
        "candidate_raw_cartesian_sequences": int(
            candidate_plan["raw_cartesian_sequences"]
        ),
        "candidate_hybrid_budget": int(
            candidate_plan.get("hybrid_candidate_budget", max_enumerated_sequences)
        ),
        "candidate_sample_attempts": int(candidate_plan["sample_attempts"]),
        "command_tts_execution_policy": COMMAND_TTS_EXECUTION_POLICY,
        "command_tts_parallel_workers": max(probe_tts_workers, replay_tts_workers),
        "command_tts_probe_pre_rendered": len(probe_tasks) if command_backend else 0,
        "command_tts_replay_pre_rendered": len(replay_tasks) if command_backend else 0,
        "round": round_index,
        "frontend": frontend,
        "configured_max_length": configured_max_length,
        "max_length": max_length,
        "max_enumerated_sequences": max_enumerated_sequences,
        "enumerated_sequences": len(candidates),
        "configured_top_k": configured_top_k,
        "global_hardest_fill": global_hardest_fill,
        "max_selected_sequences": max_selected_sequences,
        "keyword_count": len(keywords),
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
