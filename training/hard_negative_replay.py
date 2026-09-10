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
MAX_POSITIVE_STRESS_EXAMPLES_PER_KEYWORD = 256
DISTANCE_POINTS_M = {"0.5m": 0.5, "1m": 1.0, "2m": 2.0, "3m": 3.0, "5m": 5.0}
HARD_NEGATIVE_AZIMUTH_BANDS = ("front", "side", "rear")
HARD_NEGATIVE_SNR_BANDS = ("critical", "low", "mid", "high")


def _repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def normalize_hard_negative_replay(
    raw: object,
    *,
    active_tokens: list[str],
    forbidden: list[list[str]],
    token_map: dict[str, int],
    keyword_ids: set[int] | None = None,
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
        focus_keyword_id = item.get("focus_keyword_id")
        if focus_keyword_id is not None:
            focus_keyword_id = int(focus_keyword_id)
            if focus_keyword_id <= 0:
                raise ValueError(f"{label}.focus_keyword_id must be positive")
            if keyword_ids is not None and focus_keyword_id not in keyword_ids:
                raise ValueError(f"{label}.focus_keyword_id is not a configured keyword")
        normalized.append(
            {
                "tokens": tokens,
                "target_ids": [int(token_map[token]) for token in tokens],
                "examples": examples,
                "focus_keyword_id": focus_keyword_id,
            }
        )
        seen.add(key)
    return normalized


def normalize_positive_stress_replay(
    raw: object,
    *,
    keywords: list[dict],
) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("domain_iteration.positive_stress_replay must be a list")
    by_id = {int(keyword["id"]): keyword for keyword in keywords}
    normalized: list[dict] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        label = f"domain_iteration.positive_stress_replay[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        keyword_id = int(item.get("keyword_id", 0))
        if keyword_id not in by_id:
            raise ValueError(f"{label}.keyword_id is not configured")
        if keyword_id in seen:
            raise ValueError(f"{label}.keyword_id duplicates an earlier entry")
        examples = int(item.get("examples", 1))
        if not 1 <= examples <= MAX_POSITIVE_STRESS_EXAMPLES_PER_KEYWORD:
            raise ValueError(
                f"{label}.examples must be 1..{MAX_POSITIVE_STRESS_EXAMPLES_PER_KEYWORD}"
            )
        mode = str(item.get("focus", "adaptive"))
        if mode not in {"adaptive", "fallback"}:
            raise ValueError(f"{label}.focus must be adaptive or fallback")
        fallback = item.get("fallback", {})
        if not isinstance(fallback, dict):
            raise ValueError(f"{label}.fallback must be an object")
        keyword = by_id[keyword_id]
        normalized.append(
            {
                "keyword_id": keyword_id,
                "text": str(keyword["text"]),
                "tokens": list(keyword["tokens"]),
                "target_ids": [int(value) for value in keyword["token_ids"]],
                "examples": examples,
                "focus": mode,
                "fallback": {str(key): value for key, value in fallback.items()},
            }
        )
        seen.add(keyword_id)
    return normalized


def _keyword_curriculum(curriculum: dict | None, keyword_id: int | None) -> dict | None:
    if not isinstance(curriculum, dict) or keyword_id is None:
        return curriculum
    global_dimensions = curriculum.get("dimension_weights", {})
    keyword_dimensions = curriculum.get("keyword_dimension_weights", {})
    if not isinstance(global_dimensions, dict) or not isinstance(keyword_dimensions, dict):
        return curriculum
    specific = keyword_dimensions.get(str(keyword_id), {})
    if not isinstance(specific, dict):
        return curriculum
    merged: dict[str, dict[str, float]] = {}
    for dimension in sorted(set(global_dimensions) | set(specific)):
        base = global_dimensions.get(dimension, {})
        local = specific.get(dimension, {})
        if not isinstance(base, dict) or not isinstance(local, dict):
            continue
        values = {str(key): float(value) for key, value in base.items()}
        for key, value in local.items():
            values[str(key)] = max(values.get(str(key), 1.0), float(value))
        merged[str(dimension)] = values
    return {**curriculum, "dimension_weights": merged}


def _parse_focus_domain(value: str) -> dict[str, str]:
    if ":" not in value:
        return {}
    _, payload = value.split(":", 1)
    result: dict[str, str] = {}
    for field in payload.split("|"):
        if "=" not in field:
            continue
        key, item = field.split("=", 1)
        if key and item:
            result[key] = item
    return result


def adaptive_focus(
    curriculum: dict | None,
    keyword_id: int,
    fallback: dict,
) -> dict:
    result = {str(key): value for key, value in fallback.items()}
    if not isinstance(curriculum, dict):
        return result
    raw = curriculum.get("keyword_worst_domains", {})
    if not isinstance(raw, dict):
        return result
    ranked = raw.get(str(keyword_id), [])
    if not isinstance(ranked, list):
        return result
    candidates = [
        item
        for item in ranked
        if isinstance(item, dict) and isinstance(item.get("domain"), str)
    ]
    candidates.sort(
        key=lambda item: (
            0 if str(item["domain"]).startswith("distance_azimuth_snr:") else 1,
            -float(item.get("hardness", 0.0)),
            str(item["domain"]),
        )
    )
    if candidates:
        result.update(_parse_focus_domain(str(candidates[0]["domain"])))
    return result


def _band_for_distance(domains: dict, distance_m: float) -> str:
    candidates: list[tuple[float, str]] = []
    for name, item in domains["distance_bands"].items():
        low, high = item["distance_m"]
        if float(low) - 1.0e-9 <= distance_m <= float(high) + 1.0e-9:
            candidates.append((abs((float(low) + float(high)) * 0.5 - distance_m), str(name)))
    if not candidates:
        raise ValueError("focused replay distance is outside configured distance bands")
    return min(candidates)[1]


def _snr_point(domains: dict, name: str) -> float:
    low, high = [float(value) for value in domains["snr_db"]]
    ranges = {
        "critical": (low, min(high, 6.0)),
        "low": (max(low, 6.001), min(high, 12.0)),
        "mid": (max(low, 12.001), min(high, 20.0)),
        "high": (max(low, 20.001), high),
    }
    if name not in ranges:
        return float(name)
    start, end = ranges[name]
    if start > end:
        raise ValueError(f"focused replay SNR band is outside configured range: {name}")
    return (start + end) * 0.5


def _azimuth_band(value: float) -> str:
    if abs(value) <= 30.0:
        return "front"
    if abs(value) <= 90.0:
        return "side"
    return "rear"


def hard_negative_stress_focus(
    domains: dict,
    *,
    round_index: int,
    item_index: int,
    example_index: int,
) -> dict:
    """Guarantee bounded hard-negative coverage without increasing replay count.

    Synthetic geometry uses a deterministic azimuth-band x SNR-band x playback
    cube. With the product's 24-example replay entries, every one of the 3x4x2
    primary stress combinations is present once per sequence and round. Exact
    azimuth values rotate across rounds, while distance/noise/RT60 remain sampled
    by the existing curriculum so replay keeps acoustic diversity. Measured-RIR
    evidence never has its geometry overridden; only SNR/playback are stratified.
    """
    snr_bands: list[str] = []
    for name in HARD_NEGATIVE_SNR_BANDS:
        try:
            _snr_point(domains, name)
        except ValueError:
            continue
        snr_bands.append(name)
    if not snr_bands:
        return {}

    playback_probability = float(domains.get("playback_probability", 0.0))
    if playback_probability <= 0.0:
        playback_states = [False]
    elif playback_probability >= 1.0:
        playback_states = [True]
    else:
        playback_states = [False, True]

    measured_rir = isinstance(domains.get("rir_manifest"), dict)
    azimuth_values: dict[str, list[float]] = {}
    if not measured_rir:
        raw_azimuths = [float(value) for value in domains.get("azimuth_deg", [])]
        for band in HARD_NEGATIVE_AZIMUTH_BANDS:
            values = [value for value in raw_azimuths if _azimuth_band(value) == band]
            if values:
                azimuth_values[band] = values

    if measured_rir or not azimuth_values:
        combinations = [
            (None, snr, playback)
            for playback in playback_states
            for snr in snr_bands
        ]
    else:
        combinations = [
            (band, snr, playback)
            for playback in playback_states
            for snr in snr_bands
            for band in HARD_NEGATIVE_AZIMUTH_BANDS
            if band in azimuth_values
        ]
    if not combinations:
        return {}

    rotation = (round_index + item_index) * 7
    band, snr, playback = combinations[(example_index + rotation) % len(combinations)]
    focus: dict[str, object] = {"snr": snr, "playback": playback}
    if band is not None:
        values = azimuth_values[band]
        cycle = example_index // len(combinations)
        focus["azimuth"] = values[(round_index + item_index + cycle) % len(values)]
    return focus


def apply_focus(scene: dict, focus: dict, domains: dict) -> dict:
    if not focus:
        return scene
    result = dict(scene)
    geometry_fields = {"distance_bin", "distance_m", "azimuth"} & set(focus)
    if geometry_fields and isinstance(domains.get("rir_manifest"), dict):
        raise ValueError("focused synthetic geometry cannot override measured RIR evidence")

    if "distance_bin" in focus:
        key = str(focus["distance_bin"])
        if key not in DISTANCE_POINTS_M:
            raise ValueError(f"unsupported focused distance bin: {key}")
        distance_m = DISTANCE_POINTS_M[key]
        result["distance_m"] = distance_m
        result["distance_band"] = _band_for_distance(domains, distance_m)
    elif "distance_m" in focus:
        distance_m = float(focus["distance_m"])
        result["distance_m"] = distance_m
        result["distance_band"] = _band_for_distance(domains, distance_m)

    if "azimuth" in focus:
        value = str(focus["azimuth"])
        azimuth = {"front": 0.0, "side": 90.0, "rear": 180.0}.get(value)
        result["azimuth_deg"] = float(value) if azimuth is None else azimuth
    if "snr" in focus:
        result["snr_db"] = _snr_point(domains, str(focus["snr"]))
    if "noise" in focus:
        noise = str(focus["noise"])
        if noise not in domains["noise_profiles"]:
            raise ValueError(f"unsupported focused noise profile: {noise}")
        result["noise_profile"] = noise
    if "playback" in focus:
        raw = focus["playback"]
        enabled = raw if isinstance(raw, bool) else str(raw).lower() in {"1", "true", "on", "playback"}
        if enabled:
            low, high = domains["playback_sir_db"]
            result["playback_sir_db"] = (float(low) + float(high)) * 0.5
        else:
            result["playback_sir_db"] = None
    return result


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
    keyword_ids = {int(keyword["id"]) for keyword in keywords}
    feature_dim = int(config.get("model", {}).get("feature_dim", 32))
    carriers = token_carriers(keywords, feature_dim)
    active_tokens = list(carriers)
    forbidden = [list(keyword["tokens"]) for keyword in keywords]
    sequences = normalize_hard_negative_replay(
        iteration.get("hard_negative_replay", []),
        active_tokens=active_tokens,
        forbidden=forbidden,
        token_map=token_map,
        keyword_ids=keyword_ids,
    )
    positive_replay = normalize_positive_stress_replay(
        iteration.get("positive_stress_replay", []),
        keywords=keywords,
    )

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "hard-negatives.tsv"
    evidence_path = output / "hard-negatives.json"
    if not sequences and not positive_replay:
        manifest.write_text("", encoding="utf-8")
        evidence = {
            "schema_version": 1,
            "round": round_index,
            "examples": 0,
            "hard_negative_examples": 0,
            "positive_stress_examples": 0,
            "sequences": [],
            "positive_stress": [],
            "hard_negative_stress_policy": "azimuth-snr-playback-cube-v1",
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
        raise ValueError(f"unsupported replay TTS backend: {backend}")
    if backend == "command":
        command = tts.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("command TTS backend requires generator.tts.command argv list")
    augment_config = generator.get("augment", {})
    validate_augment_config(augment_config)
    domains = validate_domains(config)
    seed = int(config.get("seed", 1337))
    rows: list[dict] = []

    def render_item(
        *,
        kind: str,
        item_index: int,
        example_index: int,
        token_names: list[str],
        target_ids: list[int],
        focus_keyword_id: int | None,
        focus: dict | None,
    ) -> None:
        kind_offset = 0 if kind == "hard-negative" else 400_000_003
        example_seed = (
            seed
            + 70_000_019
            + kind_offset
            + round_index * 1_000_003
            + item_index * 65_537
            + example_index * 4099
        )
        rng = random.Random(example_seed)
        stem = ("h" if kind == "hard-negative" else "p") + f"{item_index:02d}-e{example_index:03d}"
        clean_path = output / "clean" / f"{stem}.wav"
        if backend == "tone":
            clean = render_tone_tokens(token_names, carriers, rng, tts)
        else:
            clean = render_command_tts(
                " ".join(token_names),
                token_names,
                kind,
                clean_path,
                tts,
            )
        augmented = augment(clean, rng, augment_config)
        scene_seed = example_seed + 31_337
        scene_rng = random.Random(scene_seed)
        scene = sample_scene(
            domains,
            scene_rng,
            curriculum_weights=_keyword_curriculum(
                curriculum_weights, focus_keyword_id
            ),
            forced_band=None,
        )
        if focus:
            scene = apply_focus(scene, focus, domains)
        mono, scene_meta = render_scene(
            augmented,
            scene,
            seed=scene_seed,
            afe=domains["afe"],
        )
        wav_path = output / "wav" / f"{stem}.wav"
        write_wav(wav_path, mono)
        rows.append(
            {
                "kind": kind,
                "path": str(wav_path.resolve()),
                "tokens": token_names,
                "target_ids": target_ids,
                "focus_keyword_id": focus_keyword_id,
                "focus": focus or {},
                "example_seed": example_seed,
                "scene_seed": scene_seed,
                "scene": scene_meta,
                "wav_sha256": sha256_file(wav_path),
            }
        )

    for sequence_index, sequence in enumerate(sequences):
        for example_index in range(int(sequence["examples"])):
            render_item(
                kind="hard-negative",
                item_index=sequence_index,
                example_index=example_index,
                token_names=list(sequence["tokens"]),
                target_ids=list(sequence["target_ids"]),
                focus_keyword_id=sequence.get("focus_keyword_id"),
                focus=hard_negative_stress_focus(
                    domains,
                    round_index=round_index,
                    item_index=sequence_index,
                    example_index=example_index,
                ),
            )

    for positive_index, item in enumerate(positive_replay):
        focus = (
            adaptive_focus(curriculum_weights, int(item["keyword_id"]), item["fallback"])
            if item["focus"] == "adaptive"
            else dict(item["fallback"])
        )
        for example_index in range(int(item["examples"])):
            render_item(
                kind="positive-stress",
                item_index=positive_index,
                example_index=example_index,
                token_names=list(item["tokens"]),
                target_ids=list(item["target_ids"]),
                focus_keyword_id=int(item["keyword_id"]),
                focus=focus,
            )

    manifest.write_text(
        "".join(
            f"{row['path']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    hard_negative_examples = sum(
        int(item["examples"]) for item in sequences
    )
    positive_stress_examples = sum(
        int(item["examples"]) for item in positive_replay
    )
    evidence = {
        "schema_version": 1,
        "round": round_index,
        "examples": len(rows),
        "hard_negative_examples": hard_negative_examples,
        "positive_stress_examples": positive_stress_examples,
        "hard_negative_stress_policy": "azimuth-snr-playback-cube-v1",
        "sequences": [
            {
                "tokens": item["tokens"],
                "examples": int(item["examples"]),
                "focus_keyword_id": item.get("focus_keyword_id"),
            }
            for item in sequences
        ],
        "positive_stress": [
            {
                "keyword_id": int(item["keyword_id"]),
                "text": item["text"],
                "examples": int(item["examples"]),
                "focus": item["focus"],
                "fallback": item["fallback"],
            }
            for item in positive_replay
        ],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**evidence, "evidence": str(evidence_path)}
