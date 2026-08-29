from __future__ import annotations

import re

from qualification_common import (
    FRAME_HOP_SAMPLES,
    close_enough,
    finite,
    json_int,
    required_text,
    sha256_value,
)


def validate_eval(summary: dict, provenance: dict, actual_hashes: dict[str, str]) -> dict:
    if json_int(provenance.get("schema_version"), "evaluation provenance schema_version") != 1:
        raise ValueError("evaluation provenance schema_version must be 1")
    measured_hashes = {
        key: sha256_value(provenance.get(key), f"evaluation provenance {key}")
        for key in actual_hashes
    }
    for key, expected in actual_hashes.items():
        if measured_hashes[key] != expected:
            raise ValueError(f"evaluation provenance {key} does not match selected file")
    recordings = json_int(provenance.get("recordings"), "evaluation provenance recordings", 1)
    detections = json_int(provenance.get("detections"), "evaluation provenance detections", 0)
    if summary.get("references_sha256") != actual_hashes["references_sha256"] or summary.get("detections_sha256") != actual_hashes["detections_sha256"]:
        raise ValueError("evaluation summary does not match selected files")
    result = {
        "recordings": json_int(summary["recordings"], "evaluation.recordings", 1),
        "audio_hours": finite(summary["audio_hours"], "evaluation.audio_hours", 0.0),
        "expected": json_int(summary["expected"], "evaluation.expected", 0),
        "matched": json_int(summary["matched"], "evaluation.matched", 0),
        "false_rejects": json_int(summary["false_rejects"], "evaluation.false_rejects", 0),
        "false_accepts": json_int(summary["false_accepts"], "evaluation.false_accepts", 0),
        "frr": finite(summary["frr"], "evaluation.frr", 0.0),
        "far_per_hour": finite(summary["far_per_hour"], "evaluation.far_per_hour", 0.0),
        "p50_post_end_latency_ms": finite(summary["p50_post_end_latency_ms"], "evaluation.p50_post_end_latency_ms", 0.0),
        "p95_post_end_latency_ms": finite(summary["p95_post_end_latency_ms"], "evaluation.p95_post_end_latency_ms", 0.0),
        **actual_hashes,
    }
    if result["audio_hours"] <= 0.0 or result["recordings"] != recordings:
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


def validate_board(
    summary: dict,
    model_bytes: int,
    pack_bytes: int,
    source_sha: str,
    actual_hashes: dict[str, str],
) -> dict:
    if json_int(summary.get("schema_version"), "board.schema_version") != 1:
        raise ValueError("board benchmark schema_version must be 1")
    for key, expected in actual_hashes.items():
        measured = sha256_value(summary.get(key), f"board.{key}")
        if measured != expected:
            raise ValueError(f"board benchmark {key} does not match selected file")
    if json_int(summary.get("block_samples"), "board.block_samples") != FRAME_HOP_SAMPLES:
        raise ValueError("board benchmark must use one KWS hop per block")
    if json_int(summary.get("model_bytes"), "board.model_bytes", 1) != model_bytes or json_int(summary.get("keyword_pack_bytes"), "board.keyword_pack_bytes", 1) != pack_bytes:
        raise ValueError("board benchmark artifact sizes do not match selected artifacts")
    result = {
        **actual_hashes,
        "runtime_version": required_text(summary, "runtime_version", "board"),
        "runtime_source_revision": required_text(
            summary, "runtime_source_revision", "board"
        ),
        "runtime_config_digest": sha256_value(
            summary.get("runtime_config_digest"), "board.runtime_config_digest"
        ),
        "runtime_target": required_text(summary, "runtime_target", "board"),
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
    if result["runtime_source_revision"] != source_sha:
        raise ValueError("board benchmark runtime source does not match qualification source")
    if not result["p50_process_us"] <= result["p95_process_us"] <= result["p99_process_us"] <= result["max_process_us"]:
        raise ValueError("board benchmark percentiles are not monotonic")
    close_enough(result["mean_process_us"], result["total_process_us"] / result["blocks"], "board.mean_process_us", 1e-6, 0.01)
    close_enough(result["rtf"], result["total_process_us"] / (result["audio_seconds"] * result["repeats"] * 1_000_000.0), "board.rtf", 2e-5, 1e-8)
    expected_headroom = result["block_deadline_us"] / result["p99_process_us"] if result["p99_process_us"] > 0.0 else 0.0
    close_enough(result["p99_headroom"], expected_headroom, "board.p99_headroom", 2e-5, 1e-4)
    return result


UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def validate_evidence(
    evidence: dict,
    *,
    sku: str,
    source_sha: str,
    actual_hashes: dict[str, str],
) -> dict:
    if json_int(evidence.get("schema_version"), "evidence.schema_version") != 2:
        raise ValueError("evidence schema_version must be 2")
    if evidence.get("evidence_class") != "product-board":
        raise ValueError("evidence_class must be product-board")
    if required_text(evidence, "sku", "evidence") != sku:
        raise ValueError("evidence SKU does not match qualification SKU")
    if required_text(evidence, "source_sha", "evidence") != source_sha:
        raise ValueError("evidence source_sha does not match qualification source")
    collected_at = required_text(evidence, "collected_at_utc", "evidence")
    if UTC_RE.fullmatch(collected_at) is None:
        raise ValueError("evidence.collected_at_utc must be UTC YYYY-MM-DDTHH:MM:SSZ")
    measured_hashes = {
        key: sha256_value(evidence.get(key), f"evidence.{key}")
        for key in actual_hashes
    }
    for key, expected in actual_hashes.items():
        if measured_hashes[key] != expected:
            raise ValueError(f"evidence {key} does not match selected artifact")
    result = {
        "schema_version": 2,
        "evidence_class": "product-board",
        "sku": sku,
        "source_sha": source_sha,
        "collected_at_utc": collected_at,
        "builder_id": required_text(evidence, "builder_id", "evidence"),
        "dut_id": required_text(evidence, "dut_id", "evidence"),
        "collector_id": required_text(evidence, "collector_id", "evidence"),
        "target": required_text(evidence, "target", "evidence"),
        "board_revision": required_text(evidence, "board_revision", "evidence"),
        "soc": required_text(evidence, "soc", "evidence"),
        "toolchain": required_text(evidence, "toolchain", "evidence"),
        "compiler_flags": required_text(evidence, "compiler_flags", "evidence"),
        "governor": required_text(evidence, "governor", "evidence"),
        "audio_frontend": required_text(evidence, "audio_frontend", "evidence"),
        "soak_hours": finite(evidence["soak_hours"], "evidence.soak_hours", 0.0),
        "cpu_percent": finite(evidence["cpu_percent"], "evidence.cpu_percent", 0.0),
        "rss_kib": finite(evidence["rss_kib"], "evidence.rss_kib", 0.0),
        "stack_high_water_bytes": finite(evidence["stack_high_water_bytes"], "evidence.stack_high_water_bytes", 0.0),
        "max_temp_c": finite(evidence["max_temp_c"], "evidence.max_temp_c"),
        "average_power_mw": finite(evidence["average_power_mw"], "evidence.average_power_mw", 0.0),
        **measured_hashes,
    }
    if result["builder_id"] == result["dut_id"]:
        raise ValueError("evidence builder_id and dut_id must be distinct")
    if result["cpu_percent"] > 100.0:
        raise ValueError("evidence.cpu_percent must be <= 100")
    return result
