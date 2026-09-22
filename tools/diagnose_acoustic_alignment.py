#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "eval"))

from adversarial_refinement import select_refinement_source  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from score_events import (  # noqa: E402
    DEFAULT_POST_TOLERANCE_MS,
    DEFAULT_PRE_TOLERANCE_MS,
)

EVIDENCE_CLASS = "kws-acoustic-alignment-diagnostic-v2"
MODEL_HEADER = struct.Struct("<4sHHHHHHIIIfffQIIIIII")
MODEL_MAGIC = b"KWSP"
MODEL_VERSION = 2
FRONTENDS = {0: "logmel", 1: "pcen-lite"}
NEG_INF = float("-inf")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def resolve_repo_path(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_keyword_thresholds(path: pathlib.Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_no}: expected four TSV columns")
        keyword_id = int(parts[0])
        threshold = float(parts[2])
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: keyword threshold is invalid")
        if keyword_id in result:
            raise ValueError(f"{path}:{line_no}: duplicate keyword id {keyword_id}")
        result[keyword_id] = threshold
    if not result:
        raise ValueError("keyword threshold set is empty")
    return result


def load_keywords(path: pathlib.Path, token_map: dict[str, int]) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_no}: expected four TSV columns")
        keyword_id = int(parts[0])
        token_names = parts[3].strip().split()
        if not token_names:
            raise ValueError(f"{path}:{line_no}: keyword token sequence is empty")
        try:
            token_ids = tuple(token_map[name] for name in token_names)
        except KeyError as exc:
            raise ValueError(f"{path}:{line_no}: unknown token {exc.args[0]}") from exc
        if keyword_id in result:
            raise ValueError(f"{path}:{line_no}: duplicate keyword id {keyword_id}")
        result[keyword_id] = token_ids
    if not result:
        raise ValueError("keyword set is empty")
    return result


def load_model(path: pathlib.Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if len(blob) < MODEL_HEADER.size:
        raise ValueError("model is smaller than KWSP header")
    values = MODEL_HEADER.unpack_from(blob)
    (
        magic,
        version,
        header_bytes,
        feature_dim,
        hidden_dim,
        vocab_size,
        frontend_kind,
        sample_rate_hz,
        frame_length_samples,
        frame_hop_samples,
        wx_scale,
        wh_scale,
        wo_scale,
        vocab_fingerprint,
        wx_off,
        wh_off,
        bh_off,
        wo_off,
        bo_off,
        total_bytes,
    ) = values
    if (
        magic != MODEL_MAGIC
        or version != MODEL_VERSION
        or header_bytes != MODEL_HEADER.size
        or total_bytes != len(blob)
        or frontend_kind not in FRONTENDS
        or feature_dim <= 0
        or hidden_dim <= 0
        or vocab_size < 2
    ):
        raise ValueError("unsupported or malformed KWSP model")

    wx_count = hidden_dim * feature_dim
    wh_count = hidden_dim * hidden_dim
    wo_count = vocab_size * hidden_dim
    wx = struct.unpack_from(f"<{wx_count}b", blob, wx_off)
    wh = struct.unpack_from(f"<{wh_count}b", blob, wh_off)
    bh = struct.unpack_from(f"<{hidden_dim}f", blob, bh_off)
    wo = struct.unpack_from(f"<{wo_count}b", blob, wo_off)
    bo = struct.unpack_from(f"<{vocab_size}f", blob, bo_off)
    scalars = (wx_scale, wh_scale, wo_scale, *bh, *bo)
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("model contains non-finite scales or biases")
    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "frontend_kind": frontend_kind,
        "frontend_name": FRONTENDS[frontend_kind],
        "sample_rate_hz": sample_rate_hz,
        "frame_length_samples": frame_length_samples,
        "frame_hop_samples": frame_hop_samples,
        "wx_scale": float(wx_scale),
        "wh_scale": float(wh_scale),
        "wo_scale": float(wo_scale),
        "vocab_fingerprint": int(vocab_fingerprint),
        "wx": wx,
        "wh": wh,
        "bh": bh,
        "wo": wo,
        "bo": bo,
    }


def infer_logits(model: dict[str, Any], features: list[list[float]]) -> list[list[float]]:
    fdim = int(model["feature_dim"])
    hdim = int(model["hidden_dim"])
    vocab = int(model["vocab_size"])
    wx = model["wx"]
    wh = model["wh"]
    wo = model["wo"]
    bh = model["bh"]
    bo = model["bo"]
    sx = float(model["wx_scale"])
    sh = float(model["wh_scale"])
    so = float(model["wo_scale"])
    hidden = [0.0] * hdim
    result: list[list[float]] = []

    for frame_index, frame in enumerate(features):
        if len(frame) != fdim:
            raise ValueError(
                f"feature frame {frame_index} has dim {len(frame)}, expected {fdim}"
            )
        next_hidden = [0.0] * hdim
        for h in range(hdim):
            wx_base = h * fdim
            wh_base = h * hdim
            in_sum = sum(float(wx[wx_base + i]) * float(frame[i]) for i in range(fdim))
            rec_sum = sum(
                float(wh[wh_base + i]) * float(hidden[i]) for i in range(hdim)
            )
            next_hidden[h] = math.tanh(float(bh[h]) + sx * in_sum + sh * rec_sum)
        hidden = next_hidden
        logits: list[float] = []
        for token in range(vocab):
            base = token * hdim
            value = float(bo[token]) + so * sum(
                float(wo[base + h]) * hidden[h] for h in range(hdim)
            )
            logits.append(value)
        result.append(logits)
    return result


def log_softmax(row: list[float]) -> list[float]:
    maximum = max(row)
    norm = maximum + math.log(sum(math.exp(value - maximum) for value in row))
    return [value - norm for value in row]


def decoder_surrogate_log_confidence(
    logits: list[list[float]], sequence: tuple[int, ...]
) -> float:
    """Match the differentiable training surrogate in sequence_margin.py."""
    if not logits or not sequence:
        raise ValueError("decoder surrogate requires logits and a keyword sequence")
    log_probs = [log_softmax(row) for row in logits]
    steps = len(log_probs)
    score = [row[sequence[0]] for row in log_probs]
    previous = sequence[0]
    for token in sequence[1:]:
        gap = 2 if token == previous else 1
        if steps <= gap:
            return NEG_INF
        prefix_best: list[float] = []
        running = NEG_INF
        for value in score:
            running = max(running, value)
            prefix_best.append(running)
        shifted = [NEG_INF] * steps
        for index in range(gap, steps):
            shifted[index] = prefix_best[index - gap]
        score = [
            shifted[index] + log_probs[index][token]
            if shifted[index] != NEG_INF
            else NEG_INF
            for index in range(steps)
        ]
        previous = token
    return max(score) / float(len(sequence))


def parse_runtime_detections(stdout: str, recording_id: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(stdout.splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"runtime output line {line_no} is not an object")
        if str(value.get("recording")) != recording_id:
            raise ValueError(f"runtime output line {line_no} recording id drifted")
        keyword_id = int(value["keyword_id"])
        confidence = float(value["confidence"])
        time_s = float(value["time_s"])
        if (
            keyword_id < 0
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or not math.isfinite(time_s)
            or time_s < 0.0
        ):
            raise ValueError(f"runtime output line {line_no} is invalid")
        rows.append(
            {
                "keyword_id": keyword_id,
                "confidence": confidence,
                "time_s": time_s,
            }
        )
    return rows


def run_runtime(
    runner: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    wav: pathlib.Path,
    *,
    recording_id: str,
) -> list[dict]:
    completed = subprocess.run(
        [str(runner), str(model), str(pack), str(wav), recording_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runtime evaluation failed for {wav}: {completed.stderr.strip()}"
        )
    return parse_runtime_detections(completed.stdout, recording_id)


def logsumexp(values: list[float]) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_log_probability(logits: list[list[float]], target: tuple[int, ...]) -> float:
    if not logits or not target:
        raise ValueError("CTC diagnostic requires non-empty logits and target")
    log_probs = [log_softmax(row) for row in logits]
    extended: list[int] = [0]
    for token in target:
        if token <= 0 or token >= len(logits[0]):
            raise ValueError(f"target token id out of range: {token}")
        extended.extend((token, 0))

    previous = [NEG_INF] * len(extended)
    previous[0] = log_probs[0][0]
    if len(extended) > 1:
        previous[1] = log_probs[0][extended[1]]
    for frame in log_probs[1:]:
        current = [NEG_INF] * len(extended)
        for index, label in enumerate(extended):
            candidates = [previous[index]]
            if index > 0:
                candidates.append(previous[index - 1])
            if (
                index > 1
                and label != 0
                and label != extended[index - 2]
            ):
                candidates.append(previous[index - 2])
            current[index] = logsumexp(candidates) + frame[label]
        previous = current
    return logsumexp(previous[-2:] if len(previous) > 1 else previous)


def greedy_collapse(logits: list[list[float]]) -> tuple[int, ...]:
    output: list[int] = []
    previous: int | None = None
    for row in logits:
        token = max(range(len(row)), key=row.__getitem__)
        if token != previous and token != 0:
            output.append(token)
        previous = token
    return tuple(output)


def longest_prefix_subsequence(target: tuple[int, ...], sequence: tuple[int, ...]) -> int:
    depth = 0
    for token in sequence:
        if depth < len(target) and token == target[depth]:
            depth += 1
    return depth


def token_best_ranks(logits: list[list[float]], target: tuple[int, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for token in sorted(set(target)):
        best = len(logits[0])
        for row in logits:
            rank = 1 + sum(1 for value in row if value > row[token])
            best = min(best, rank)
        result[str(token)] = best
    return result


def run_feature_dump(
    executable: pathlib.Path,
    wav: pathlib.Path,
    *,
    feature_dim: int,
    frontend_name: str,
) -> list[list[float]]:
    completed = subprocess.run(
        [str(executable), str(wav), str(feature_dim), frontend_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"feature dump failed for {wav}: {completed.stderr.strip()}"
        )
    result: list[list[float]] = []
    for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        values = row.get("features")
        if not isinstance(values, list):
            raise ValueError(f"feature dump line {line_no} lacks features")
        result.append([float(value) for value in values])
    if not result:
        raise ValueError(f"feature dump produced no frames for {wav}")
    return result


def stable_sample(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("source_wav_sha256") or ""),
            int(row.get("scene_seed", 0)),
            str(row.get("wav_sha256") or ""),
            str(row.get("path") or ""),
        ),
    )
    selected: list[dict] = []
    used_sources: set[str] = set()
    for row in ordered:
        source = str(row.get("source_wav_sha256") or row.get("source_path") or "")
        if source in used_sources:
            continue
        used_sources.add(source)
        selected.append(row)
        if len(selected) >= limit:
            return selected
    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def analyze_recording(
    *,
    row: dict,
    target: tuple[int, ...],
    keyword_id: int,
    threshold: float,
    model: dict[str, Any],
    model_path: pathlib.Path,
    pack_path: pathlib.Path,
    feature_dump: pathlib.Path,
    runner: pathlib.Path,
    root_tokens: set[int],
    root_margin: float,
) -> dict:
    wav = pathlib.Path(str(row["path"]))
    if not wav.is_file():
        raise ValueError(f"diagnostic WAV is missing: {wav}")
    features = run_feature_dump(
        feature_dump,
        wav,
        feature_dim=int(model["feature_dim"]),
        frontend_name=str(model["frontend_name"]),
    )
    logits = infer_logits(model, features)
    greedy = greedy_collapse(logits)
    root = target[0]
    root_top1_frames = 0
    root_within_margin_frames = 0
    blank_root_admissible_frames = 0
    decoder_root_admissible_frames = 0
    minimum_root_gap = math.inf
    blank_top1_frames = 0
    for frame in logits:
        top = max(range(len(frame)), key=frame.__getitem__)
        if top == 0:
            blank_top1_frames += 1
        gap = frame[top] - frame[root]
        minimum_root_gap = min(minimum_root_gap, gap)
        if top == root:
            root_top1_frames += 1
        if gap <= root_margin:
            root_within_margin_frames += 1
            if top == 0:
                blank_root_admissible_frames += 1
            if top == root or top == 0 or top in root_tokens:
                decoder_root_admissible_frames += 1

    ctc_logp = ctc_log_probability(logits, target)
    surrogate_log_confidence = decoder_surrogate_log_confidence(logits, target)
    surrogate_confidence = (
        math.exp(surrogate_log_confidence)
        if surrogate_log_confidence != NEG_INF
        else 0.0
    )
    recording_id = str(row.get("wav_sha256") or sha256_file(wav))
    runtime_detections = run_runtime(
        runner,
        model_path,
        pack_path,
        wav,
        recording_id=recording_id,
    )
    expected_runtime = [
        item for item in runtime_detections if int(item["keyword_id"]) == keyword_id
    ]
    wrong_runtime = [
        item for item in runtime_detections if int(item["keyword_id"]) != keyword_id
    ]
    event_start_frame = int(row.get("event_start_frame", -1))
    event_end_frame = int(row.get("event_end_frame", -1))
    if (
        event_start_frame < 0
        or event_end_frame < event_start_frame
        or event_end_frame > int(row.get("frames", -1))
    ):
        raise ValueError("diagnostic positive row has invalid event frame bounds")
    sample_rate_hz = int(model["sample_rate_hz"])
    event_start_s = event_start_frame / float(sample_rate_hz)
    event_end_s = event_end_frame / float(sample_rate_hz)
    match_lower_s = event_start_s - DEFAULT_PRE_TOLERANCE_MS / 1000.0
    match_upper_s = event_end_s + DEFAULT_POST_TOLERANCE_MS / 1000.0
    expected_runtime_matched = [
        item
        for item in expected_runtime
        if match_lower_s <= float(item["time_s"]) <= match_upper_s
    ]
    wrong_runtime_in_window = [
        item
        for item in wrong_runtime
        if match_lower_s <= float(item["time_s"]) <= match_upper_s
    ]
    depth = longest_prefix_subsequence(target, greedy)
    best_ranks = token_best_ranks(logits, target)
    return {
        "split": str(row["split"]),
        "keyword_id": int(row["keyword_id"]),
        "wav_sha256": str(row.get("wav_sha256") or sha256_file(wav)),
        "source_wav_sha256": str(row.get("source_wav_sha256") or ""),
        "frames": len(logits),
        "target": list(target),
        "greedy_collapsed": list(greedy),
        "greedy_exact": greedy == target,
        "greedy_contains_target_as_subsequence": depth == len(target),
        "longest_target_prefix_depth": depth,
        "longest_target_prefix_ratio": depth / len(target),
        "ctc_log_probability": ctc_logp,
        "ctc_nll_per_token": -ctc_logp / len(target),
        "surrogate_log_confidence": surrogate_log_confidence,
        "surrogate_confidence": surrogate_confidence,
        "runtime_threshold": threshold,
        "surrogate_above_runtime_threshold": surrogate_confidence >= threshold,
        "runtime_detection_count": len(runtime_detections),
        "runtime_expected_detection_count": len(expected_runtime),
        "runtime_expected_matched_count": len(expected_runtime_matched),
        "runtime_out_of_window_expected_detection_count": (
            len(expected_runtime) - len(expected_runtime_matched)
        ),
        "runtime_wrong_keyword_detection_count": len(wrong_runtime),
        "runtime_wrong_keyword_in_window_count": len(wrong_runtime_in_window),
        "runtime_detected_expected": bool(expected_runtime),
        "runtime_matched_expected": bool(expected_runtime_matched),
        "runtime_detected_keyword_ids": sorted(
            {int(item["keyword_id"]) for item in runtime_detections}
        ),
        "runtime_max_expected_confidence": (
            max(float(item["confidence"]) for item in expected_runtime)
            if expected_runtime
            else None
        ),
        "runtime_max_matched_expected_confidence": (
            max(float(item["confidence"]) for item in expected_runtime_matched)
            if expected_runtime_matched
            else None
        ),
        "event_start_s": event_start_s,
        "event_end_s": event_end_s,
        "match_pre_tolerance_ms": DEFAULT_PRE_TOLERANCE_MS,
        "match_post_tolerance_ms": DEFAULT_POST_TOLERANCE_MS,
        "blank_top1_fraction": blank_top1_frames / len(logits),
        "root_top1_frames": root_top1_frames,
        "root_within_margin_frames": root_within_margin_frames,
        "blank_root_admissible_frames": blank_root_admissible_frames,
        "decoder_root_admissible_frames": decoder_root_admissible_frames,
        "minimum_root_logit_gap": minimum_root_gap,
        "target_token_best_ranks": best_ranks,
        "all_target_tokens_reach_top3": all(rank <= 3 for rank in best_ranks.values()),
    }


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"recordings": 0}
    nll = [float(row["ctc_nll_per_token"]) for row in records]
    prefix = [float(row["longest_target_prefix_ratio"]) for row in records]
    return {
        "recordings": len(records),
        "ctc_nll_per_token_mean": sum(nll) / len(nll),
        "ctc_nll_per_token_p50": percentile(nll, 0.50),
        "ctc_nll_per_token_p90": percentile(nll, 0.90),
        "greedy_exact_recordings": sum(bool(row["greedy_exact"]) for row in records),
        "greedy_subsequence_recordings": sum(
            bool(row["greedy_contains_target_as_subsequence"]) for row in records
        ),
        "surrogate_above_threshold_recordings": sum(
            bool(row["surrogate_above_runtime_threshold"]) for row in records
        ),
        "runtime_expected_detected_recordings": sum(
            bool(row["runtime_detected_expected"]) for row in records
        ),
        "runtime_expected_matched_recordings": sum(
            bool(row["runtime_matched_expected"]) for row in records
        ),
        "runtime_wrong_keyword_recordings": sum(
            int(row["runtime_wrong_keyword_detection_count"]) > 0 for row in records
        ),
        "runtime_wrong_keyword_in_window_recordings": sum(
            int(row["runtime_wrong_keyword_in_window_count"]) > 0 for row in records
        ),
        "surrogate_above_threshold_runtime_miss_recordings": sum(
            bool(row["surrogate_above_runtime_threshold"])
            and not bool(row["runtime_matched_expected"])
            for row in records
        ),
        "greedy_subsequence_runtime_miss_recordings": sum(
            bool(row["greedy_contains_target_as_subsequence"])
            and not bool(row["runtime_matched_expected"])
            for row in records
        ),
        "mean_longest_target_prefix_ratio": sum(prefix) / len(prefix),
        "root_top1_recordings": sum(int(row["root_top1_frames"]) > 0 for row in records),
        "root_within_margin_recordings": sum(
            int(row["root_within_margin_frames"]) > 0 for row in records
        ),
        "blank_root_admissible_recordings": sum(
            int(row["blank_root_admissible_frames"]) > 0 for row in records
        ),
        "decoder_root_admissible_recordings": sum(
            int(row["decoder_root_admissible_frames"]) > 0 for row in records
        ),
        "all_target_tokens_reach_top3_recordings": sum(
            bool(row["all_target_tokens_reach_top3"]) for row in records
        ),
    }


def build(
    *,
    config_path: pathlib.Path,
    manifest_path: pathlib.Path,
    feature_dump: pathlib.Path,
    runner: pathlib.Path,
    max_per_keyword_split: int,
) -> dict:
    config = load_object(config_path, "preflight config")
    manifest = load_object(manifest_path, "development manifest")
    source, source_policy = select_refinement_source(manifest)
    model_path = pathlib.Path(str(source["model"]))
    pack_path = pathlib.Path(str(source["pack"]))
    keywords_path = pathlib.Path(str(source["keywords"]))
    tokens_path = resolve_repo_path(str(config["tokens"]))
    if not runner.is_file():
        raise ValueError(f"runtime runner is missing: {runner}")
    if not pack_path.is_file():
        raise ValueError(f"keyword pack is missing: {pack_path}")
    model = load_model(model_path)
    token_map = load_tokens(tokens_path)
    if len(token_map) != int(model["vocab_size"]):
        raise ValueError("token vocabulary size differs from model")
    keywords = load_keywords(keywords_path, token_map)
    thresholds = load_keyword_thresholds(keywords_path)
    if set(thresholds) != set(keywords):
        raise ValueError("keyword thresholds differ from keyword sequences")
    roots = {tokens[0] for tokens in keywords.values()}

    contract_path = ROOT / "configs/parameter-contract.json"
    contract = load_object(contract_path, "parameter contract")
    root_margin = float(
        contract["algorithm_constants"]["KWS_ROOT_START_LOGIT_MARGIN"]["default"]
    )
    if not math.isfinite(root_margin) or root_margin < 0.0:
        raise ValueError("root-start logit margin is invalid")

    round_index = int(source["round"])
    index_path = (
        manifest_path.parent
        / "datasets"
        / f"round-{round_index:02d}"
        / "domain-index.jsonl"
    )
    index_rows = load_jsonl(index_path)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in index_rows:
        split = str(row.get("split") or "")
        keyword_id = row.get("keyword_id")
        if split not in {"calibration", "test"} or keyword_id is None:
            continue
        keyword_id = int(keyword_id)
        if keyword_id not in keywords or str(row.get("kind")) != "positive":
            continue
        target_ids = tuple(int(value) for value in row.get("target_ids", []))
        if target_ids != keywords[keyword_id]:
            raise ValueError(
                f"{split} keyword {keyword_id} target ids differ from calibrated keyword"
            )
        groups[(split, keyword_id)].append(row)

    records: list[dict] = []
    for split in ("calibration", "test"):
        for keyword_id in sorted(keywords):
            group = groups.get((split, keyword_id), [])
            if not group:
                raise ValueError(f"{split} keyword {keyword_id} has no positive rows")
            for row in stable_sample(group, max_per_keyword_split):
                records.append(
                    analyze_recording(
                        row=row,
                        target=keywords[keyword_id],
                        keyword_id=keyword_id,
                        threshold=thresholds[keyword_id],
                        model=model,
                        model_path=model_path,
                        pack_path=pack_path,
                        feature_dump=feature_dump,
                        runner=runner,
                        root_tokens=roots,
                        root_margin=root_margin,
                    )
                )

    grouped_output: dict[str, dict[str, dict]] = {"calibration": {}, "test": {}}
    for split in ("calibration", "test"):
        for keyword_id in sorted(keywords):
            subset = [
                row
                for row in records
                if row["split"] == split and int(row["keyword_id"]) == keyword_id
            ]
            grouped_output[split][str(keyword_id)] = aggregate(subset)

    return {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "development_only": True,
        "source_round": round_index,
        "source_frontend": str(source["frontend"]),
        "source_selection_policy": source_policy,
        "source_was_strict": (
            source.get("calibration_gate") is True and source.get("test_gate") is True
        ),
        "model_sha256": sha256_file(model_path),
        "tokens_sha256": sha256_file(tokens_path),
        "keywords_sha256": sha256_file(keywords_path),
        "domain_index_sha256": sha256_file(index_path),
        "feature_dump_sha256": sha256_file(feature_dump),
        "runner_sha256": sha256_file(runner),
        "keyword_pack_sha256": sha256_file(pack_path),
        "parameter_contract_sha256": sha256_file(contract_path),
        "root_start_logit_margin": root_margin,
        "event_match_pre_tolerance_ms": DEFAULT_PRE_TOLERANCE_MS,
        "event_match_post_tolerance_ms": DEFAULT_POST_TOLERANCE_MS,
        "max_recordings_per_keyword_split": max_per_keyword_split,
        "model": {
            key: model[key]
            for key in (
                "feature_dim",
                "hidden_dim",
                "vocab_size",
                "frontend_name",
                "sample_rate_hz",
                "frame_length_samples",
                "frame_hop_samples",
                "vocab_fingerprint",
            )
        },
        "aggregates": grouped_output,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only acoustic alignment diagnostic using the exact C "
            "frontend and exported quantized KWSP model, without the decoder."
        )
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dump", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--max-per-keyword-split", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_per_keyword_split <= 32:
        raise ValueError("max-per-keyword-split must be 1..32")
    result = build(
        config_path=args.config.resolve(),
        manifest_path=args.manifest.resolve(),
        feature_dump=args.feature_dump.resolve(),
        runner=args.runner.resolve(),
        max_per_keyword_split=args.max_per_keyword_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_round": result["source_round"],
                "source_frontend": result["source_frontend"],
                "recordings": len(result["records"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        struct.error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
