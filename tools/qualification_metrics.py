from __future__ import annotations

import hashlib
import json

from corpus_identity import corpus_digest
from qualification_common import (
    FRAME_HOP_SAMPLES,
    close_enough,
    finite,
    json_int,
    required_text,
    sha256_value,
)

CPU_PERCENT_SEMANTICS = "process_cpu_time / elapsed / online_cpu_capacity * 100"


def validate_eval(
    summary: dict,
    provenance: dict,
    actual_hashes: dict[str, str],
    actual_corpus: dict,
) -> dict:
    if json_int(provenance.get("schema_version"), "evaluation provenance schema_version") != 2:
        raise ValueError("evaluation provenance schema_version must be 2")
    measured_hashes = {
        key: sha256_value(provenance.get(key), f"evaluation provenance {key}")
        for key in actual_hashes
    }
    for key, expected in actual_hashes.items():
        if measured_hashes[key] != expected:
            raise ValueError(f"evaluation provenance {key} does not match selected file")
    recordings = json_int(provenance.get("recordings"), "evaluation provenance recordings", 1)
    detections = json_int(provenance.get("detections"), "evaluation provenance detections", 0)
    if (
        summary.get("references_sha256") != actual_hashes["references_sha256"]
        or summary.get("detections_sha256") != actual_hashes["detections_sha256"]
    ):
        raise ValueError("evaluation summary does not match selected files")

    audio_files = provenance.get("audio_files")
    if not isinstance(audio_files, list) or not audio_files:
        raise ValueError("evaluation provenance audio_files must be non-empty")
    measured_corpus = sha256_value(
        provenance.get("audio_corpus_sha256"), "evaluation provenance audio_corpus_sha256"
    )
    if measured_corpus != corpus_digest(audio_files):
        raise ValueError("evaluation provenance audio corpus digest is internally inconsistent")
    if measured_corpus != actual_corpus.get("corpus_sha256") or audio_files != actual_corpus.get("recordings"):
        raise ValueError("evaluation provenance does not match selected qualification WAV bytes")

    result = {
        "recordings": json_int(summary["recordings"], "evaluation.recordings", 1),
        "audio_hours": finite(summary["audio_hours"], "evaluation.audio_hours", 0.0),
        "expected": json_int(summary["expected"], "evaluation.expected", 0),
        "matched": json_int(summary["matched"], "evaluation.matched", 0),
        "false_rejects": json_int(summary["false_rejects"], "evaluation.false_rejects", 0),
        "false_accepts": json_int(summary["false_accepts"], "evaluation.false_accepts", 0),
        "frr": finite(summary["frr"], "evaluation.frr", 0.0),
        "far_per_hour": finite(summary["far_per_hour"], "evaluation.far_per_hour", 0.0),
        "p50_post_end_latency_ms": finite(
            summary["p50_post_end_latency_ms"], "evaluation.p50_post_end_latency_ms", 0.0
        ),
        "p95_post_end_latency_ms": finite(
            summary["p95_post_end_latency_ms"], "evaluation.p95_post_end_latency_ms", 0.0
        ),
        "audio_corpus_sha256": measured_corpus,
        "audio_files": audio_files,
        **actual_hashes,
    }
    if result["audio_hours"] <= 0.0 or result["recordings"] != recordings or result["recordings"] != len(audio_files):
        raise ValueError("evaluation duration/recording count is invalid")
    if result["matched"] + result["false_rejects"] != result["expected"]:
        raise ValueError("evaluation expected/matched/false-reject counts disagree")
    if result["matched"] + result["false_accepts"] != detections:
        raise ValueError("evaluation matched/false-accept counts disagree with detections")
    expected_frr = result["false_rejects"] / result["expected"] if result["expected"] else 0.0
    expected_far = result["false_accepts"] / result["audio_hours"]
    close_enough(result["frr"], expected_frr, "evaluation.frr", 1e-9, 1e-12)
    close_enough(result["far_per_hour"], expected_far, "evaluation.far_per_hour", 1e-9, 1e-12)
    if result["frr"] > 1.0 or result["p50_post_end_latency_ms"] > result["p95_post_end_latency_ms"]:
        raise ValueError("evaluation summary contains impossible values")
    return result


def validate_board(summary: dict, model_bytes: int, pack_bytes: int, actual_hashes: dict[str, str]) -> dict:
    if json_int(summary.get("schema_version"), "board.schema_version") != 1:
        raise ValueError("board benchmark schema_version must be 1")
    for key, expected in actual_hashes.items():
        measured = sha256_value(summary.get(key), f"board.{key}")
        if measured != expected:
            raise ValueError(f"board benchmark {key} does not match selected file")
    if json_int(summary.get("block_samples"), "board.block_samples") != FRAME_HOP_SAMPLES:
        raise ValueError("board benchmark must use one KWS hop per block")
    if (
        json_int(summary.get("model_bytes"), "board.model_bytes", 1) != model_bytes
        or json_int(summary.get("keyword_pack_bytes"), "board.keyword_pack_bytes", 1) != pack_bytes
    ):
        raise ValueError("board benchmark artifact sizes do not match selected artifacts")
    result = {
        **actual_hashes,
        "audio_seconds": finite(summary["audio_seconds"], "board.audio_seconds", 0.0),
        "repeats": json_int(summary["repeats"], "board.repeats", 1),
        "blocks": json_int(summary["blocks"], "board.blocks", 1),
        "arena_bytes": json_int(summary["arena_bytes"], "board.arena_bytes", 1),
        "block_deadline_us": finite(summary["block_deadline_us"], "board.block_deadline_us", 0.0),
        "total_process_us": finite(summary["total_process_us"], "board.total_process_us", 0.0),
        "mean_process_us": finite(summary["mean_process_us"], "board.mean_process_us", 0.0),
        "p50_process_us": finite(summary["p50_process_us"], "board.p50_process_us", 0.0),
        "p95_process_us": finite(summary["p95_process_us"], "board.p95_process_us", 0.0),
        "p99_process_us": finite(summary["p99_process_us"], "board.p99_process_us", 0.0),
        "max_process_us": finite(summary["max_process_us"], "board.max_process_us", 0.0),
        "rtf": finite(summary["rtf"], "board.rtf", 0.0),
        "p99_headroom": finite(summary["p99_headroom"], "board.p99_headroom", 0.0),
    }
    if result["audio_seconds"] <= 0.0 or result["block_deadline_us"] != 20000.0:
        raise ValueError("board benchmark audio/deadline is invalid")
    if not result["p50_process_us"] <= result["p95_process_us"] <= result["p99_process_us"] <= result["max_process_us"]:
        raise ValueError("board benchmark percentiles are not monotonic")
    close_enough(
        result["mean_process_us"],
        result["total_process_us"] / result["blocks"],
        "board.mean_process_us",
        1e-6,
        0.01,
    )
    close_enough(
        result["rtf"],
        result["total_process_us"] / (result["audio_seconds"] * result["repeats"] * 1_000_000.0),
        "board.rtf",
        2e-5,
        1e-8,
    )
    expected_headroom = result["block_deadline_us"] / result["p99_process_us"] if result["p99_process_us"] > 0.0 else 0.0
    close_enough(result["p99_headroom"], expected_headroom, "board.p99_headroom", 2e-5, 1e-4)
    return result


def _runtime_soak_metrics(evidence: dict, runtime_soak_sha256: str) -> dict[str, float]:
    raw_text = evidence.get("runtime_soak_raw")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("evidence.runtime_soak_raw must be non-empty text")
    if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != runtime_soak_sha256:
        raise ValueError("embedded runtime soak bytes do not match retained raw evidence identity")
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("embedded runtime soak is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("embedded runtime soak must be a JSON object")
    if json_int(value.get("schema_version"), "runtime soak schema_version") != 2:
        raise ValueError("runtime soak schema_version must be 2")
    if value.get("completed_requested_duration") is not True:
        raise ValueError("runtime soak did not complete the requested duration")
    if value.get("cpu_percent_semantics") != CPU_PERCENT_SEMANTICS:
        raise ValueError("runtime soak CPU percentage semantics are unsupported")

    capacity = json_int(value.get("cpu_capacity_count"), "runtime soak cpu_capacity_count", 1)
    requested_hours = finite(value.get("requested_hours"), "runtime soak requested_hours", 0.0)
    elapsed_seconds = finite(value.get("elapsed_seconds"), "runtime soak elapsed_seconds", 0.0)
    elapsed_hours = finite(value.get("elapsed_hours"), "runtime soak elapsed_hours", 0.0)
    if requested_hours <= 0.0 or elapsed_seconds <= 0.0:
        raise ValueError("runtime soak requested/elapsed duration must be > 0")
    close_enough(elapsed_hours, elapsed_seconds / 3600.0, "runtime soak elapsed_hours", 1e-9, 1e-12)
    if elapsed_hours + 1e-6 < requested_hours:
        raise ValueError("runtime soak elapsed_hours is shorter than requested_hours")

    initial_cpu = finite(value.get("initial_cpu_seconds"), "runtime soak initial_cpu_seconds", 0.0)
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("runtime soak samples must be non-empty")
    rss_values: list[float] = []
    cpu_values: list[float] = []
    temp_values: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"runtime soak samples[{index}] must be an object")
        finite(sample.get("elapsed_s"), f"runtime soak samples[{index}].elapsed_s", 0.0)
        if sample.get("rss_kib") is not None:
            rss_values.append(finite(sample["rss_kib"], f"runtime soak samples[{index}].rss_kib", 0.0))
        if sample.get("cpu_seconds") is not None:
            cpu_values.append(finite(sample["cpu_seconds"], f"runtime soak samples[{index}].cpu_seconds", 0.0))
        if sample.get("temp_c") is not None:
            temp_values.append(finite(sample["temp_c"], f"runtime soak samples[{index}].temp_c"))
    if not rss_values or not cpu_values or not temp_values:
        raise ValueError("runtime soak must retain RSS, CPU and thermal samples")
    if cpu_values[-1] < initial_cpu:
        raise ValueError("runtime soak CPU time regressed")

    expected_cpu = min(100.0, max(0.0, (cpu_values[-1] - initial_cpu) / elapsed_seconds) / capacity * 100.0)
    expected_rss = max(rss_values)
    expected_temp = max(temp_values)
    close_enough(
        finite(value.get("average_cpu_percent"), "runtime soak average_cpu_percent", 0.0),
        expected_cpu,
        "runtime soak average_cpu_percent",
        1e-9,
        1e-9,
    )
    close_enough(
        finite(value.get("max_rss_kib"), "runtime soak max_rss_kib", 0.0),
        expected_rss,
        "runtime soak max_rss_kib",
        1e-9,
        1e-9,
    )
    close_enough(
        finite(value.get("max_temp_c"), "runtime soak max_temp_c"),
        expected_temp,
        "runtime soak max_temp_c",
        1e-9,
        1e-9,
    )
    return {
        "soak_hours": elapsed_hours,
        "cpu_percent": expected_cpu,
        "rss_kib": expected_rss,
        "max_temp_c": expected_temp,
    }


def validate_evidence(evidence: dict, collector_sha256: str) -> dict:
    if json_int(evidence.get("schema_version"), "evidence.schema_version") != 2:
        raise ValueError("target evidence schema_version must be 2")
    if required_text(evidence, "collector", "evidence") != "collect_target_evidence.py":
        raise ValueError("target evidence must be produced by collect_target_evidence.py")
    if sha256_value(evidence.get("collector_sha256"), "evidence.collector_sha256") != collector_sha256:
        raise ValueError("target evidence collector hash does not match retained collector")

    raw = evidence.get("raw_evidence")
    if not isinstance(raw, list) or not raw:
        raise ValueError("target evidence must bind at least one raw evidence artifact")
    normalized_raw = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"evidence.raw_evidence[{index}] must be an object")
        name = required_text(item, "name", f"evidence.raw_evidence[{index}]")
        if name in seen_names:
            raise ValueError("target evidence raw artifact names must be unique")
        seen_names.add(name)
        normalized_raw.append(
            {
                "name": name,
                "sha256": sha256_value(item.get("sha256"), f"evidence.raw_evidence[{index}].sha256"),
                "bytes": json_int(item.get("bytes"), f"evidence.raw_evidence[{index}].bytes", 1),
            }
        )

    runtime_soak_name = required_text(evidence, "runtime_soak_name", "evidence")
    runtime_soak_sha256 = sha256_value(evidence.get("runtime_soak_sha256"), "evidence.runtime_soak_sha256")
    power_raw_name = required_text(evidence, "power_raw_name", "evidence")
    power_raw_sha256 = sha256_value(evidence.get("power_raw_sha256"), "evidence.power_raw_sha256")
    raw_by_name = {item["name"]: item for item in normalized_raw}
    if runtime_soak_name not in raw_by_name or raw_by_name[runtime_soak_name]["sha256"] != runtime_soak_sha256:
        raise ValueError("target evidence runtime soak identity is missing from raw evidence")
    if power_raw_name not in raw_by_name or raw_by_name[power_raw_name]["sha256"] != power_raw_sha256:
        raise ValueError("target evidence power raw identity is missing from raw evidence")

    raw_metrics = _runtime_soak_metrics(evidence, runtime_soak_sha256)
    result = {
        "collector": "collect_target_evidence.py",
        "collector_sha256": collector_sha256,
        "target": required_text(evidence, "target", "evidence"),
        "board_revision": required_text(evidence, "board_revision", "evidence"),
        "soc": required_text(evidence, "soc", "evidence"),
        "toolchain": required_text(evidence, "toolchain", "evidence"),
        "compiler_flags": required_text(evidence, "compiler_flags", "evidence"),
        "governor": required_text(evidence, "governor", "evidence"),
        "audio_frontend": required_text(evidence, "audio_frontend", "evidence"),
        "kernel": required_text(evidence, "kernel", "evidence"),
        "machine": required_text(evidence, "machine", "evidence"),
        "cpu_online": required_text(evidence, "cpu_online", "evidence"),
        "uptime_s": finite(evidence["uptime_s"], "evidence.uptime_s", 0.0),
        "soak_hours": finite(evidence["soak_hours"], "evidence.soak_hours", 0.0),
        "cpu_percent": finite(evidence["cpu_percent"], "evidence.cpu_percent", 0.0),
        "rss_kib": finite(evidence["rss_kib"], "evidence.rss_kib", 0.0),
        "stack_high_water_bytes": finite(evidence["stack_high_water_bytes"], "evidence.stack_high_water_bytes", 0.0),
        "max_temp_c": finite(evidence["max_temp_c"], "evidence.max_temp_c"),
        "average_power_mw": finite(evidence["average_power_mw"], "evidence.average_power_mw", 0.0),
        "runtime_soak_name": runtime_soak_name,
        "runtime_soak_sha256": runtime_soak_sha256,
        "power_raw_name": power_raw_name,
        "power_raw_sha256": power_raw_sha256,
        "instrument_id": required_text(evidence, "instrument_id", "evidence"),
        "calibration_id": required_text(evidence, "calibration_id", "evidence"),
        "raw_evidence": normalized_raw,
    }
    if result["cpu_percent"] > 100.0:
        raise ValueError("evidence.cpu_percent must be <= 100")
    for key, expected in raw_metrics.items():
        close_enough(result[key], expected, f"evidence.{key}", 1e-9, 1e-9)
    return result
