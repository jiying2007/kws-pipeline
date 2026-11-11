#!/usr/bin/env python3
"""Fail-closed owner-local comparison of real-board R1 offline CPU receipts.

Consumes existing private receipts only; executes no model, TTS, ASR or training.
A passing CPU screen is NOT SSC305 deployment/acoustic qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics

SCHEMA = "r1-cpu-board-receipt-comparison-v1"
PLAN_SCHEMA = "r1-cpu-board-rotated-plan-v1"
BENCH_SCHEMA = "r1-strong-fsmn-offline-board-resource-v1"
MODEL_SHA = "63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4"
FRONTEND_SHA = "49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49"
VARIANTS = ("par4", "block2", "block4")
ORDERS = (VARIANTS, ("block2", "block4", "par4"), ("block4", "par4", "block2"))
MAX_JSON_BYTES = 65536


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key: " + key)
        out[key] = value
    return out


def no_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def guarded_file(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "invalid receipt path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and "." not in relative.parts and str(relative) == name,
            "noncanonical receipt path")
    path = root.joinpath(*relative.parts)
    for depth in range(1, len(relative.parts) + 1):
        require(not root.joinpath(*relative.parts[:depth]).is_symlink(),
                "symlink in evidence path")
    require(path.is_file() and not path.is_symlink(), "missing evidence file: " + name)
    require(path.stat().st_size <= MAX_JSON_BYTES, "oversized JSON receipt")
    return path


def read_json(path):
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_JSON_BYTES, "invalid evidence JSON length")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                       parse_constant=no_constant)
    require(type(value) is dict, "JSON root must be object")
    return value, hashlib.sha256(raw).hexdigest()


def number(value, name, strictly_positive=True):
    require(type(value) in (int, float) and math.isfinite(value),
            "invalid numeric value: " + name)
    require(value > 0 if strictly_positive else value >= 0,
            "invalid numeric range: " + name)
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer: " + name)
    return value


def validate_receipt(row, plan, expected_machine):
    require(row.get("schema") == BENCH_SCHEMA, "unknown bench receipt schema")
    for name in ("device_qualified", "live_afe", "independent_far_frr"):
        require(row.get(name) is False, "benchmark qualification boundary changed: " + name)
    for name in ("power_mw", "stack_high_water_bytes"):
        require(name in row and row[name] is None, "unsupported board metric asserted: " + name)
    require(row.get("model_kind") == "source-hash-checked-by-sdk", "model source identity marker missing")
    require(row.get("arch") == expected_machine and
            isinstance(row.get("kernel"), str) and row["kernel"],
            "machine/kernel identity mismatch")
    require(integer(row.get("pcm_bytes"), "pcm_bytes", 2) == plan["pcm_bytes"],
            "PCM byte exposure differs")
    audio_s = number(row.get("audio_seconds"), "audio_seconds")
    require(abs(audio_s - row["pcm_bytes"] / 32000.0) <= 1e-6,
            "PCM exposure duration inconsistent with 16k PCM16")
    require(integer(row.get("feed_chunk_bytes"), "feed_chunk_bytes", 640) == 640,
            "unexpected feed chunk size")
    require(integer(row.get("feed_calls"), "feed_calls", 1) ==
            (row["pcm_bytes"] + 639) // 640, "wrong feed call count")
    frames = integer(row.get("sdk_model_frames"), "sdk_model_frames", 1)
    events = integer(row.get("keyword_events"), "keyword_events")
    e1 = integer(row.get("keyword1_events"), "keyword1_events")
    e2 = integer(row.get("keyword2_events"), "keyword2_events")
    require(e1 + e2 == events, "keyword-event sum mismatch")
    require(integer(row.get("feed_p95_histogram_upper_us"),
                    "feed_p95_histogram_upper_us", 1) > 0,
            "missing feed latency upper")
    integer(row.get("feed_max_us"), "feed_max_us", 1)
    integer(row.get("process_peak_rss_kib"), "process_peak_rss_kib", 1)
    number(row.get("model_init_wall_ms"), "model_init_wall_ms")
    wall_s = number(row.get("offline_feed_wall_seconds"), "offline_feed_wall_seconds")
    cpu_s = number(row.get("offline_feed_cpu_seconds"), "offline_feed_cpu_seconds")
    cpu_ratio = number(row.get("cpu_seconds_per_audio_second"),
                       "cpu_seconds_per_audio_second")
    rtf = number(row.get("offline_rtf"), "offline_rtf")
    require(abs(cpu_ratio - cpu_s / audio_s) < 2e-6, "inconsistent CPU seconds per audio second")
    require(abs(rtf - wall_s / audio_s) < 2e-6, "inconsistent realtime factor")
    return (frames, events, e1, e2), cpu_ratio


def examine(root):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), "evidence root must be a real directory")
    plan_path = guarded_file(root, "PLAN.json")
    plan, plan_sha = read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "unexpected CPU plan schema")
    require(plan.get("model_sha256") == MODEL_SHA and
            plan.get("frontend_sha256") == FRONTEND_SHA, "unexpected pretrained R1 identity")
    require(is_hash(plan.get("pcm_sha256")), "PCM SHA256 required")
    integer(plan.get("pcm_bytes"), "plan pcm_bytes", 2)
    require(plan["pcm_bytes"] % 2 == 0 and plan["pcm_bytes"] <= 536870912,
            "PCM size unsupported")
    require(plan.get("input_mode") == "offline_pcm16k", "only file-feed measurements are supported")
    board = plan.get("board")
    require(type(board) is dict, "board metadata missing")
    require(isinstance(board.get("id"), str) and 0 < len(board["id"]) <= 160,
            "board revision/id missing")
    require(is_hash(board.get("firmware_sha256")), "firmware SHA256 required")
    require(board.get("machine") in ("armv7l", "armv8l", "arm"),
            "this study requires native ARM userspace metadata")
    require(isinstance(board.get("governor"), str) and board["governor"] not in ("", "unknown"),
            "CPU governor missing")
    integer(board.get("frequency_khz"), "CPU frequency", 1)
    require(isinstance(board.get("affinity"), str) and board["affinity"],
            "CPU affinity missing")
    bins = plan.get("binary_sha256")
    require(type(bins) is dict and set(bins) == set(VARIANTS)
            and all(is_hash(bins[v]) for v in VARIANTS),
            "three distinct target binary hashes required")
    require(len(set(bins.values())) == 3, "variant binaries must be distinct")
    rounds = plan.get("rounds")
    require(type(rounds) is list and 9 <= len(rounds) <= 30 and len(rounds) % 3 == 0,
            "rotated repetitions must be 9-30, in complete triples")
    expected_kernel = None
    expected_counts = None
    seen = set()
    cpu = {v: [] for v in VARIANTS}
    upper = {v: [] for v in VARIANTS}
    identities = []
    for index, batch in enumerate(rounds):
        require(type(batch) is dict and batch.get("round") == index,
                "round counter mismatch")
        require(batch.get("order") == list(ORDERS[index % 3]),
                "non-counterbalanced run order")
        captures = batch.get("receipts")
        require(type(captures) is dict and set(captures) == set(VARIANTS),
                "missing or additional receipt variant")
        counts_this_round = None
        for variant in batch["order"]:
            entry = captures[variant]
            require(type(entry) is dict and set(entry) == {"path", "sha256"},
                    "invalid receipt locator")
            require(is_hash(entry.get("sha256")), "receipt hash missing")
            path = entry.get("path")
            require(isinstance(path, str) and path not in seen, "duplicate receipt file")
            seen.add(path)
            row, actual_sha = read_json(guarded_file(root, path))
            require(actual_sha == entry["sha256"], "receipt SHA256 mismatch")
            counts, cpu_ratio = validate_receipt(row, plan, board["machine"])
            pair = (row["arch"], row["kernel"])
            if expected_kernel is None:
                expected_kernel = pair
            require(pair == expected_kernel, "kernel/arch drift between rotated runs")
            if expected_counts is None:
                expected_counts = counts
            require(counts == expected_counts, "frame/event disagreement across variants")
            if counts_this_round is None:
                counts_this_round = counts
            require(counts == counts_this_round, "within-round frame/event mismatch")
            cpu[variant].append(cpu_ratio)
            upper[variant].append(row["feed_p95_histogram_upper_us"])
            identities.append({"round": index, "variant": variant,
                               "sha256": actual_sha, "path": path})
    baseline = cpu["par4"]
    baseline_p95 = upper["par4"]
    candidates = {}
    for variant in ("block2", "block4"):
        savings = [100.0 * (1.0 - x / b) for x, b in zip(cpu[variant], baseline)]
        wins = sum(x > 0.0 for x in savings)
        p95_nonregressions = sum(x <= b for x, b in zip(upper[variant], baseline_p95))
        median_savings = statistics.median(savings)
        screen = (median_savings >= 5.0 and wins >= len(rounds) - 2 and
                  p95_nonregressions >= len(rounds) - 2)
        candidates[variant] = {
            "median_cpu_savings_pct": round(median_savings, 6),
            "paired_cpu_wins": wins,
            "p95_upper_nonregressions": p95_nonregressions,
            "rotated_repetitions": len(rounds),
            "median_cpu_seconds_per_audio_second":
                round(statistics.median(cpu[variant]), 9),
            "screen": "RESOURCE_SCREEN_PASS_NOT_QUALIFIED" if screen else "NO_CREDIBLE_CPU_GAIN",
        }
    return {
        "schema": SCHEMA,
        "plan_sha256": plan_sha,
        "receipt_identities": identities,
        "rounds": len(rounds),
        "model_sha256": MODEL_SHA,
        "frontend_sha256": FRONTEND_SHA,
        "pcm_sha256": plan["pcm_sha256"],
        "pcm_bytes": plan["pcm_bytes"],
        "board_operator_metadata": board,
        "reference": {"variant": "par4",
                      "median_cpu_seconds_per_audio_second":
                          round(statistics.median(baseline), 9),
                      "median_feed_p95_upper_us": statistics.median(baseline_p95)},
        "candidates": candidates,
        "analyzer_model_forwards": 0,
        "real_model_output_parity": "NOT_CHECKED_BY_THIS_TOOL",
        "physical_device_identity_independently_verified": False,
        "device_qualified": False,
        "live_afe": False,
        "independent_far_frr": False,
        "model_promotion_authorized": False,
        "note": "offline operator-reported CPU screen, not independent SSC305/acoustic qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    result = examine(args.root)
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps({"status": "ANALYZED_NOT_QUALIFIED",
                      "rounds": result["rounds"],
                      "candidates": result["candidates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
