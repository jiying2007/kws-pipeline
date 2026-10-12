"""Disabled K1 ASR18 metadata prototype. Python standard library only.

No filesystem/network/runtime/model access, claim writer, publisher, workflow or
execution authority. Shape/hash syntax checks cannot establish actual byte
custody, history coverage, source authority, rights or native execution. Inputs
to tests are invented synthetic metadata, never new actual audio identities.
"""
import hashlib
import json
import re

EXPERIMENT = "k1-punctuation6-v1"
FAMILIES = ("qwen06", "sensevoice", "whisper")
CELLS = tuple(f"k1p6-{index:03d}" for index in range(1, 7))
QUALITY_FLAGS = ("silent", "source_peak_ge_one", "normalized_peak_ge_one",
                 "derived_pcm_silent")
DISPOSITIONS = ("eligible", "not_run_duplicate", "not_run_quality")
SHA = re.compile(r"[0-9a-f]{64}\Z")
HEAD = re.compile(r"[0-9a-f]{40}\Z")
MAX_ARCHIVE = 12 * 1024 ** 2
MAX_MANIFEST = 16 * 1024
MAX_CHUNK = 256 * 1024
MAX_CHUNKS = 48
MAX_RESPONSE_BODIES = 16 * 1024 ** 2
ROW_FIELDS = frozenset(("cell_id", "status", "attempts", "wav_sha256", "pcm_sha256",
    "raw_wav_sha256", "raw_float_sha256", "source_frames", "frames", "sample_rate_hz",
    "channels", "sample_width_bytes", "eos_verified", "generation_iterations", "quality_flags"))
PROJECTION_FIELDS = frozenset(("schema", "experiment", "status", "execution_enabled", "cells"))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _exact(value, fields, reason):
    require(type(value) is dict and set(value) == set(fields), reason)


def _hash(value, pattern=SHA):
    require(type(value) is str and pattern.fullmatch(value), "hash syntax; actual custody unverified")


def _integer(value, low, high, reason):
    require(type(value) is int and low <= value <= high, reason)


def slots():
    return [{"slot_id": cell + "-" + family, "cell_id": cell, "family": family,
             "attempts_max": 1} for family in FAMILIES for cell in CELLS]


def proposed_preparation():
    """Actual identities remain null. This is not an input or execution freeze."""
    return {"schema": "k1-asr18-disabled-preparation-v1", "experiment": EXPERIMENT,
        "status": "disabled_metadata_preparation_only", "execution_enabled": False,
        "actual_tts_data_commit": None, "actual_tts_source_commit": None,
        "actual_tts_run_id": None, "actual_tts_job_id": None,
        "actual_tts_source_freeze_sha256": None, "actual_tts_archive_sha256": None,
        "actual_input_freeze_sha256": None, "actual_new_audio_identities": None,
        "slots": slots(), "max_new_asr_calls": 18, "old_calls_allowed": 0,
        "retries": 0, "model_loads": 0, "audio_forward_calls": 0,
        "production_claim_writer_implemented": False,
        "historical_review_authority_implemented": False,
        "reader_implemented": False, "publisher_implemented": False}


def validate_disabled_preparation(value):
    require(canonical(value) == canonical(proposed_preparation()), "disabled proposal drift")
    return value


def authorize_execution(*_arguments, **_keywords):
    raise ValueError("ASR execution unavailable: this metadata prototype has no release authority")


def _validate_rows(rows):
    require(type(rows) is list and len(rows) == 6, "complete fixed six-cell metadata required")
    for cell, row in zip(CELLS, rows):
        _exact(row, ROW_FIELDS, "only new-cell identity metadata fields allowed")
        require(row["cell_id"] == cell, "fixed cell order and identity")
        require(row["status"] == "complete", "all six TTS cells complete before ASR")
        _integer(row["attempts"], 1, 1, "single TTS attempt")
        for key in ("wav_sha256", "pcm_sha256", "raw_wav_sha256", "raw_float_sha256"):
            _hash(row[key])
        _integer(row["source_frames"], 1, 12 * 24000 - 1, "strict source duration bound")
        _integer(row["frames"], 1, 12 * 16000 - 1, "strict derived duration bound")
        require(row["frames"] == (2 * row["source_frames"] + 2) // 3, "fixed 2:3 frame geometry")
        for key, expected in (("sample_rate_hz", 16000), ("channels", 1), ("sample_width_bytes", 2)):
            _integer(row[key], expected, expected, "16k mono PCM16 geometry")
        require(row["eos_verified"] is True, "EOS metadata required; actual verification external")
        _integer(row["generation_iterations"], 2, 119, "nonempty codec output and natural EOS before token cap")
        require(row["source_frames"] == (row["generation_iterations"] - 1) * 1920,
                "exact frozen TTS EOS/codec source-frame geometry")
        flags = row["quality_flags"]
        require(type(flags) is list and all(type(flag) is str for flag in flags), "quality flag list")
        require(flags == [flag for flag in QUALITY_FLAGS if flag in flags], "ordered unique known quality flags")


def screen_metadata(rows, private_historical_pairs):
    """In-memory comparison only. Never returns historical IDs/hashes or matches.

    This function cannot decide whether the supplied private history is complete
    or approved. Its output remains disabled even when all cells look eligible.
    The production parent must separately approve coverage and remeasure bytes.
    """
    _validate_rows(rows)
    require(type(private_historical_pairs) is list and bool(private_historical_pairs),
            "nonempty private historical comparison required; coverage unverified")
    for item in private_historical_pairs:
        _exact(item, ("wav_sha256", "pcm_sha256"), "private comparison hash pair fields")
        _hash(item["wav_sha256"])
        _hash(item["pcm_sha256"])
    projected = []
    for row in rows:
        collision = any(other["cell_id"] != row["cell_id"] and
            (other["wav_sha256"] == row["wav_sha256"] or other["pcm_sha256"] == row["pcm_sha256"])
            for other in rows)
        historical = any(item["wav_sha256"] == row["wav_sha256"] or
                         item["pcm_sha256"] == row["pcm_sha256"] for item in private_historical_pairs)
        disposition = ("not_run_duplicate" if collision or historical else
                       "not_run_quality" if row["quality_flags"] else "eligible")
        projected.append({**row, "quality_flags": list(row["quality_flags"]), "disposition": disposition})
    return {"schema": "k1-asr18-untrusted-metadata-projection-v1", "experiment": EXPERIMENT,
            "status": "metadata_only_external_custody_and_review_required",
            "execution_enabled": False, "cells": projected}


def _validate_projection(projection):
    _exact(projection, PROJECTION_FIELDS, "public projection fields only")
    require(projection["schema"] == "k1-asr18-untrusted-metadata-projection-v1" and
            projection["experiment"] == EXPERIMENT and projection["execution_enabled"] is False and
            projection["status"] == "metadata_only_external_custody_and_review_required", "disabled projection only")
    rows = projection["cells"]
    require(type(rows) is list and len(rows) == 6, "all six projection rows")
    for row in rows:
        _exact(row, ROW_FIELDS | {"disposition"}, "no private historical projection fields")
        require(row["disposition"] in DISPOSITIONS, "fixed pre-ASR disposition")
    _validate_rows([{key: row[key] for key in ROW_FIELDS} for row in rows])
    for row in rows:
        collisions = any(other["cell_id"] != row["cell_id"] and
            (row["wav_sha256"] == other["wav_sha256"] or row["pcm_sha256"] == other["pcm_sha256"])
            for other in rows)
        require(not collisions or row["disposition"] == "not_run_duplicate", "every collision member quarantined")
        require(not row["quality_flags"] or row["disposition"] in ("not_run_duplicate", "not_run_quality"),
                "quality cells cannot run")
        require(row["disposition"] != "not_run_quality" or bool(row["quality_flags"]), "quality reason required")
    return rows


def blind_metadata(projection):
    """Host-only plan draft; only job and execution_plan belong in a future worker.

    No files are written. Hash-order IDs satisfy retained bind_inputs, while the
    separate execution order preserves cell001..006 within each ASR family.
    New worker/host wrappers must enforce this plan; old main/run cannot do so.
    """
    rows = _validate_projection(projection)
    eligible = [row for row in rows if row["disposition"] == "eligible"]
    by_hash = sorted(eligible, key=lambda row: row["wav_sha256"])
    clips = [{"audio_id": f"clip-{index:06d}", "audio_path": "audio/" + row["wav_sha256"] + ".wav",
              "wav_sha256": row["wav_sha256"]} for index, row in enumerate(by_hash, 1)]
    job = {"schema": "blind-source-screen32-job-v1", "clips": clips}
    lookup = {row["wav_sha256"]: row["audio_id"] for row in clips}
    cell_map = [{"cell_id": row["cell_id"], "audio_id": lookup[row["wav_sha256"]]} for row in eligible]
    plan = {"schema": "k1-asr18-anonymous-order-metadata-proposal-v1", "execution_enabled": False,
            "job_sha256": digest(job), "family_order": list(FAMILIES),
            "audio_order": [row["audio_id"] for row in cell_map]}
    return {"execution_enabled": False, "job": job, "execution_plan": plan,
            "host_only_cell_map": cell_map, "eligible_cells": len(eligible),
            "max_new_asr_calls": 3 * len(eligible), "release_available": False}


def validate_blind_metadata(projection, candidate):
    require(canonical(candidate) == canonical(blind_metadata(projection)),
            "exact new eligible-only blind job, private host map and fixed cell-order plan required")
    return candidate


def initial_ledger(projection):
    rows = _validate_projection(projection)
    dispositions = {row["cell_id"]: row["disposition"] for row in rows}
    return [{"id": slot["slot_id"], "status": "not_run" if dispositions[slot["cell_id"]] == "eligible"
             else dispositions[slot["cell_id"]], "attempts": 0} for slot in slots()]


def validate_ledger(projection, ledger):
    """Validate a metadata assertion, not proof of actual calls or durable claims."""
    expected = initial_ledger(projection)
    require(type(ledger) is list and len(ledger) == 18, "retain all eighteen denominator slots")
    stopped = False
    for initial, actual in zip(expected, ledger):
        _exact(actual, ("id", "status", "attempts"), "exact ledger metadata")
        require(actual["id"] == initial["id"], "family-major fixed cell order")
        _integer(actual["attempts"], 0, 1, "integer once-only count; booleans prohibited")
        if initial["status"] in ("not_run_duplicate", "not_run_quality"):
            require(actual == initial, "excluded slot retained with zero attempts")
            continue
        status = actual["status"]
        require(status in ("not_run", "started", "complete", "failed_no_retry"), "eligible slot state")
        require(actual["attempts"] == (0 if status == "not_run" else 1), "started attempt stays consumed")
        require(not stopped or status == "not_run", "stop after first unstarted, incomplete or failed eligible slot")
        stopped = stopped or status != "complete"
    return {"denominator": 18, "consumed": sum(row["attempts"] for row in ledger),
            "complete": sum(row["status"] == "complete" for row in ledger), "execution_enabled": False}


def validate_source_authority_shape(value):
    """Future authority shape only; never substitutes for independent verification."""
    _exact(value, ("repository", "data_commit", "source_commit", "run_id", "run_attempt", "job_id", "job_key",
        "source_freeze_sha256", "archive_sha256", "archive_bytes", "outer_manifest_sha256", "member_manifest_sha256"),
        "exact bounded public TTS authority fields")
    require(value["repository"] == "jiying2007/kws-data" and value["job_key"] == "k1-tts", "actual TTS repository/job only")
    for key in ("data_commit", "source_commit"):
        _hash(value[key], HEAD)
    for key in ("run_id", "job_id"):
        _integer(value[key], 1, 10 ** 20 - 1, "actual positive run/job identity required")
    _integer(value["run_attempt"], 1, 1, "TTS first attempt only")
    for key in ("source_freeze_sha256", "archive_sha256", "outer_manifest_sha256", "member_manifest_sha256"):
        _hash(value[key])
    _integer(value["archive_bytes"], 1, MAX_ARCHIVE, "complete archive cap")
    return {"status": "syntax_only_not_verified_authority", "execution_enabled": False}


def reader_budget_shape(archive_bytes, manifest_bytes, chunk_lengths):
    """Validate a proposed complete response-body roster; does not make GETs."""
    _integer(archive_bytes, 1, MAX_ARCHIVE, "archive cap")
    _integer(manifest_bytes, 1, MAX_MANIFEST, "manifest included and bounded")
    require(type(chunk_lengths) is list and 1 <= len(chunk_lengths) <= MAX_CHUNKS, "one through 48 chunks")
    expected = [min(MAX_CHUNK, archive_bytes - offset) for offset in range(0, archive_bytes, MAX_CHUNK)]
    require(all(type(length) is int for length in chunk_lengths) and chunk_lengths == expected,
            "complete exact ordered chunk lengths")
    total = manifest_bytes + sum(chunk_lengths)
    require(total <= MAX_RESPONSE_BODIES, "entire package body reservation")
    return {"gets": len(chunk_lengths) + 1, "response_body_bytes": total,
            "response_body_reservation_bytes": MAX_RESPONSE_BODIES, "retries": 0,
            "redirects": 0, "extra_source_gets": 0, "execution_enabled": False}
