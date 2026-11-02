"""Distinguish sample-coordinate diagnostics from measured product callback latency.

No measured callback collector is supported by this version. Legacy phase
qualification remains readable, but it cannot authorize product shipping.
"""
from __future__ import annotations

import math

CONTRACT = "raw-capture-to-application-callback-v1"
LIMIT_MS = 500


def unmeasured_end_to_end() -> dict:
    return {
        "contract": CONTRACT,
        "status": "NOT_RUN",
        "qualified": False,
        "limit_ms": LIMIT_MS,
        "reason": "measured-callback-evidence-not-supported",
        "metrics": None,
    }


def normalize_end_to_end(value: object) -> dict:
    expected = unmeasured_end_to_end()
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("unsupported end-to-end latency evidence")
    if any(type(value[key]) is not type(want) or value[key] != want
           for key, want in expected.items()):
        raise ValueError("unsupported end-to-end latency evidence")
    return expected


def require_receipt_projection(summary: dict, receipt: dict,
                               summary_sha256: str) -> dict:
    result = normalize_end_to_end(summary.get("end_to_end_latency", unmeasured_end_to_end()))
    projected = normalize_end_to_end(receipt.get("end_to_end_latency", unmeasured_end_to_end()))
    if result != projected:
        raise ValueError("end-to-end receipt projection differs")
    # Historical phase artifacts may omit both fields. New projections must
    # carry the same field and bind the exact summary bytes.
    if ("end_to_end_latency" in summary) != ("end_to_end_latency" in receipt):
        raise ValueError("end-to-end receipt projection missing")
    if "end_to_end_latency" in summary and receipt.get("qualification_summary_sha256") != summary_sha256:
        raise ValueError("end-to-end receipt summary hash differs")
    return result


def require_measured_end_to_end(*values: object) -> None:
    if not values:
        raise ValueError("measured end-to-end callback evidence required")
    for value in values:
        normalize_end_to_end(value)
    # Deliberately no PASS adapter: sample offsets, status strings and an
    # unsigned receipt cannot invent a measured clock/tuple/event binding.
    raise ValueError("measured end-to-end callback evidence required; NOT_RUN")


def raw_sample_latency_diagnostic(score: dict) -> dict:
    timing = score.get("timing_coordinates", {})
    total = timing.get("total_recordings")
    covered = timing.get("afe_contract_recordings")
    matched = timing.get("matched_events")
    complete = (
        type(total) is int and total > 0 and type(covered) is int
        and covered == total and timing.get("raw_metrics_available") is True
        and type(matched) is int and matched > 0
    )
    raw = score.get("p95_raw_capture_signed_end_offset_ms")
    post = score.get("p95_post_afe_signed_end_offset_ms")
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    available = complete and finite(raw) and finite(post)
    return {
        "scope": "sample-coordinate-raw-capture-end-to-decision",
        "status": ("PASS" if raw <= LIMIT_MS else "FAIL") if available else "NOT_EVALUABLE",
        "limit_ms": LIMIT_MS,
        "matched_events": matched if type(matched) is int and matched >= 0 else None,
        "complete_timing_coverage": complete,
        "p95_raw_signed_end_offset_ms": raw if available else None,
        "p95_post_afe_signed_end_offset_ms": post if available else None,
        "product_qualification": False,
        "measurement_boundary": "not-application-callback-or-wall-clock; p95-not-per-event-maximum",
    }
