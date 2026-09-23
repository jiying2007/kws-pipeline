#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import subprocess
from typing import Iterable

EVIDENCE_CLASS = "sequence-margin-runtime-gap-development-v1"
SURROGATE_POLICY = "sparse-chronological-acoustic-maxplus-v1"
TRACE_MAGIC = b"KWTRACE1"
TRACE_HEADER_BYTES = 120


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokens(path: pathlib.Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid token row at {path}:{line_no}")
        token, raw_id = parts
        token_id = int(raw_id)
        if token in result or token_id < 0:
            raise ValueError("token vocabulary contains duplicate/invalid ids")
        result[token] = token_id
    if not result:
        raise ValueError("token vocabulary is empty")
    return result


def load_keywords(
    path: pathlib.Path,
    token_map: dict[str, int],
) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"invalid keyword row at {path}:{line_no}")
        keyword_id = int(parts[0])
        threshold = float(parts[2])
        names = parts[3].split()
        sequence = tuple(token_map[name] for name in names)
        if (
            keyword_id in result
            or not sequence
            or not math.isfinite(threshold)
            or not 0.0 < threshold < 1.0
        ):
            raise ValueError("keyword table contains invalid/duplicate rows")
        result[keyword_id] = {
            "threshold": threshold,
            "tokens": sequence,
        }
    if not result:
        raise ValueError("keyword table is empty")
    return result


def read_trace_logits(path: pathlib.Path) -> list[list[float]]:
    data = path.read_bytes()
    if len(data) < TRACE_HEADER_BYTES or data[:8] != TRACE_MAGIC:
        raise ValueError(f"invalid posterior trace header: {path}")
    vocab_size = struct.unpack_from("<H", data, 12)[0]
    frame_count = struct.unpack_from("<Q", data, 40)[0]
    if vocab_size < 2 or frame_count <= 0:
        raise ValueError(f"invalid posterior trace dimensions: {path}")
    record_bytes = 16 + 4 * vocab_size
    expected = TRACE_HEADER_BYTES + frame_count * record_bytes
    if len(data) != expected:
        raise ValueError(f"posterior trace size mismatch: {path}")
    rows: list[list[float]] = []
    offset = TRACE_HEADER_BYTES
    last_end = 0
    for _ in range(frame_count):
        end_sample = struct.unpack_from("<Q", data, offset)[0]
        if end_sample <= last_end:
            raise ValueError(f"posterior trace frame order is invalid: {path}")
        flags = data[offset + 8 : offset + 16]
        if flags[0] not in (0, 1) or any(flags[1:]):
            raise ValueError(f"posterior trace flags are invalid: {path}")
        row = list(
            struct.unpack_from(
                "<" + "f" * vocab_size,
                data,
                offset + 16,
            )
        )
        if any(not math.isfinite(value) for value in row):
            raise ValueError(f"posterior trace contains non-finite logits: {path}")
        rows.append(row)
        last_end = end_sample
        offset += record_bytes
    return rows


def log_softmax(rows: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for row in rows:
        maximum = max(row)
        norm = maximum + math.log(sum(math.exp(value - maximum) for value in row))
        result.append([value - norm for value in row])
    return result


def decoder_sequence_log_confidence(
    sample_log_probs: list[list[float]],
    sequence: tuple[int, ...],
) -> float:
    if not sample_log_probs or not sequence:
        raise ValueError("surrogate score requires non-empty trace and keyword")
    steps = len(sample_log_probs)
    score = [row[sequence[0]] for row in sample_log_probs]
    previous = sequence[0]
    for token in sequence[1:]:
        gap = 2 if token == previous else 1
        if steps <= gap:
            return float("-inf")
        prefix_best: list[float] = []
        running = float("-inf")
        for value in score:
            running = max(running, value)
            prefix_best.append(running)
        shifted = [float("-inf")] * steps
        for index in range(gap, steps):
            shifted[index] = prefix_best[index - gap]
        score = [
            shifted[index] + sample_log_probs[index][token]
            for index in range(steps)
        ]
        previous = token
    return max(score) / float(len(sequence))


def runtime_hits(
    *,
    decoder_replay: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
) -> set[int]:
    completed = subprocess.run(
        [
            str(decoder_replay),
            str(model),
            str(pack),
            str(trace),
            recording,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    result: set[int] = set()
    for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("recording") != recording:
            raise ValueError(
                f"decoder replay emitted invalid event at {trace}:{line_no}"
            )
        result.add(int(value["keyword_id"]))
    return result


def trace_paths(cache: pathlib.Path, model_sha256: str) -> list[pathlib.Path]:
    root = cache / model_sha256
    if not root.is_dir():
        raise ValueError("posterior cache has no traces for selected model")
    paths = sorted(root.rglob("*.kwtr"))
    if not paths:
        raise ValueError("posterior cache selected-model trace set is empty")
    return paths


def summarize(
    *,
    traces: Iterable[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    keywords: dict[int, dict],
    decoder_replay: pathlib.Path,
) -> dict[str, dict]:
    aggregate = {
        keyword_id: {
            "traces": 0,
            "surrogate_above_threshold": 0,
            "runtime_hit": 0,
            "surrogate_above_threshold_runtime_miss": 0,
            "runtime_hit_surrogate_below_threshold": 0,
        }
        for keyword_id in keywords
    }
    total_traces = 0
    for trace_index, trace in enumerate(traces):
        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        recording = f"posterior-trace-{trace_index:06d}"
        hits = runtime_hits(
            decoder_replay=decoder_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
        )
        total_traces += 1
        for keyword_id, item in keywords.items():
            score = decoder_sequence_log_confidence(
                log_probs,
                tuple(item["tokens"]),
            )
            confidence = 0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
            surrogate = confidence >= float(item["threshold"])
            runtime = keyword_id in hits
            row = aggregate[keyword_id]
            row["traces"] += 1
            row["surrogate_above_threshold"] += int(surrogate)
            row["runtime_hit"] += int(runtime)
            row["surrogate_above_threshold_runtime_miss"] += int(
                surrogate and not runtime
            )
            row["runtime_hit_surrogate_below_threshold"] += int(
                runtime and not surrogate
            )
    if total_traces <= 0:
        raise ValueError("no posterior traces were evaluated")
    return {str(key): value for key, value in sorted(aggregate.items())}


def self_test() -> None:
    rows = log_softmax(
        [
            [0.0, 4.0, -4.0],
            [4.0, -4.0, -4.0],
            [0.0, -4.0, 4.0],
        ]
    )
    score = decoder_sequence_log_confidence(rows, (1, 2))
    confidence = math.exp(score)
    if not 0.9 < confidence <= 1.0:
        raise AssertionError("chronological surrogate self-test failed")
    repeated = decoder_sequence_log_confidence(rows, (1, 1))
    if math.isfinite(repeated):
        raise AssertionError("repeated-token separator self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("sequence-margin/runtime-gap self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "pack": args.pack,
        "decoder-replay": args.decoder_replay,
        "posterior-cache": args.posterior_cache,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    model = args.model.resolve()
    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    pack = args.pack.resolve()
    decoder_replay = args.decoder_replay.resolve()
    posterior_cache = args.posterior_cache.resolve()
    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (decoder_replay, "decoder replay"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError(f"posterior cache is missing: {posterior_cache}")

    model_sha256 = sha256_file(model)
    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    traces = trace_paths(posterior_cache, model_sha256)
    per_keyword = summarize(
        traces=traces,
        model=model,
        pack=pack,
        keywords=keywords,
        decoder_replay=decoder_replay,
    )
    totals = {
        key: sum(int(row[key]) for row in per_keyword.values())
        for key in (
            "surrogate_above_threshold",
            "runtime_hit",
            "surrogate_above_threshold_runtime_miss",
            "runtime_hit_surrogate_below_threshold",
        )
    }
    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "surrogate_policy": SURROGATE_POLICY,
        "trace_count": len(traces),
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "keyword_pack_sha256": sha256_file(pack),
        "decoder_replay_sha256": sha256_file(decoder_replay),
        "per_keyword": per_keyword,
        "totals": totals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "traces": len(traces),
                **totals,
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
