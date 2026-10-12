"""Disabled host-only input, private screening and anonymous worker bindings.

A receipt maker is not a review authority. Only an independently supplied
expected receipt digest can authorize the receipt as an input to later review.
The receipt never exports old identities, history digests or coverage details.
All returned values remain disabled. This module never claims an ASR slot.
"""
import io
import math
import os
from pathlib import Path
import stat
import struct
import wave

import asr18_metadata as metadata
import raw_data_reader as reader

require = metadata.require
canonical = metadata.canonical
SCREENING_SCHEMA = "k1-asr18-public-screening-receipt-v1"
EXECUTION_ENABLED = False


def _wave(raw, *, native):
    """Strict retained-format WAV geometry; hashes cover original bytes."""
    cap = 12 * (24000 * 4 if native else 16000 * 2) + 256
    require(type(raw) is bytes and 44 <= len(raw) <= cap, "bounded waveform")
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and
            struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw), "complete RIFF length")
    offset, chunks = 12, {}
    tags = {b"fmt ", b"data", b"fact", b"PEAK"} if native else {b"fmt ", b"data"}
    while offset < len(raw):
        require(offset + 8 <= len(raw), "complete chunk header")
        tag, size = raw[offset:offset + 4], struct.unpack_from("<I", raw, offset + 4)[0]
        offset += 8
        require(tag in tags and tag not in chunks and offset + size <= len(raw), "unique allowed complete chunk")
        chunks[tag] = raw[offset:offset + size]
        offset += size + (size & 1)
    require(offset == len(raw) and {b"fmt ", b"data"} <= chunks.keys(), "complete waveform chunks")
    fmt = chunks[b"fmt "]
    expected = (3, 1, 24000, 96000, 4, 32) if native else (1, 1, 16000, 32000, 2, 16)
    require(len(fmt) in (16, 18) and struct.unpack_from("<HHIIHH", fmt) == expected,
            "native float24k mono or derived PCM16k mono")
    require(len(fmt) != 18 or fmt[16:] == b"\0\0", "empty format extension")
    samples = chunks[b"data"]
    width, rate = (4, 24000) if native else (2, 16000)
    require(0 < len(samples) < 12 * rate * width and len(samples) % width == 0, "nonempty strict duration and sample geometry")
    frames = len(samples) // width
    if native:
        values = [value[0] for value in struct.iter_unpack("<f", samples)]
        require(all(math.isfinite(value) for value in values), "finite float samples")
        peak = max(abs(value) for value in values)
        if b"fact" in chunks:
            require(len(chunks[b"fact"]) == 4 and struct.unpack("<I", chunks[b"fact"])[0] == frames, "native fact frames")
        if b"PEAK" in chunks:
            require(len(chunks[b"PEAK"]) == 16, "native peak record")
            version, timestamp, observed, peak_frame = struct.unpack("<IIfI", chunks[b"PEAK"])
            require(version == 1 and observed == peak and peak_frame < frames and
                    abs(values[peak_frame]) == peak, "native peak byte binding")
        return {"raw_wav_sha256": reader.sha(raw), "raw_float_sha256": reader.sha(samples),
                "source_frames": frames, "source_peak": peak}
    with wave.open(io.BytesIO(raw), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (16000, 1, 2, "NONE") and stream.getnframes() == frames and stream.readframes(frames) == samples,
                "independent PCM WAV decoder agrees")
    return {"wav_sha256": reader.sha(raw), "pcm_sha256": reader.sha(samples),
            "frames": frames, "derived_pcm_silent": not any(samples)}


def measure_complete_rows(members):
    """Remeasure all six new WAV pairs after full retained semantic verification.

    The normalized float peak is a source-attested TTS observation; original
    float/PCM identities, source peak and derived silence are byte-recomputed.
    This helper alone is deliberately not a semantic/source admission gate.
    """
    require(type(members) is dict, "complete member mapping")
    outcomes = reader.decode(members["outcomes.json"], maximum=64 * 1024)
    require(outcomes.get("status") == "completed_six_once" and type(outcomes.get("consumed_attempts")) is int and
            outcomes["consumed_attempts"] == 6 and type(outcomes.get("cells")) is list and len(outcomes["cells"]) == 6,
            "all six natural complete TTS cells before ASR")
    measured = []
    for cell_id, cell in zip(metadata.CELLS, outcomes["cells"]):
        require(cell.get("cell_id") == cell_id and cell.get("status") == "complete" and
                type(cell.get("attempts")) is int and cell["attempts"] == 1, "fixed one-attempt completed cell")
        raw = _wave(members[f"raw/{cell_id}.wav"], native=True)
        pcm = _wave(members[f"derived/{cell_id}.wav"], native=False)
        observation, eos = cell["observation"], cell["eos"]
        require(type(eos) is dict and eos.get("capture_calls") == 1 and type(eos.get("capture_calls")) is int and
                type(eos.get("iterations")) is int and 2 <= eos["iterations"] <= 119 and
                eos.get("sequence_length") == eos["iterations"] and type(eos.get("sequence_length")) is int and
                eos.get("last_token_ids") == [2150] and type(eos["last_token_ids"][0]) is int and
                eos.get("eos") == 2150 and type(eos.get("eos")) is int and eos.get("ended_with_eos") is True and
                eos.get("earlier_eos_seen") is False and eos.get("hit_token_cap_without_eos") is False,
                "natural EOS singleton, no cap or earlier EOS")
        require(raw["source_frames"] == (eos["iterations"] - 1) * 1920 and
                pcm["frames"] == (2 * raw["source_frames"] + 2) // 3, "exact native codec and derivative frame geometry")
        peak = observation.get("derivative_peak")
        require(type(peak) in (int, float) and math.isfinite(peak) and peak >= 0, "source-attested normalized float peak")
        flags = (["silent"] if raw["source_peak"] == 0 else []) + (["source_peak_ge_one"] if raw["source_peak"] >= 1 else [])
        flags += (["normalized_peak_ge_one"] if peak >= 1 else []) + (["derived_pcm_silent"] if pcm["derived_pcm_silent"] else [])
        row = {"cell_id": cell_id, "status": "complete", "attempts": 1,
            **{key: raw[key] for key in ("raw_wav_sha256", "raw_float_sha256", "source_frames")},
            **{key: pcm[key] for key in ("wav_sha256", "pcm_sha256", "frames")},
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
            "eos_verified": True, "generation_iterations": eos["iterations"], "quality_flags": flags}
        for key in ("raw_wav_sha256", "raw_float_sha256", "source_frames", "wav_sha256", "pcm_sha256", "frames",
                    "sample_rate_hz", "channels", "sample_width_bytes", "quality_flags"):
            require(canonical(observation.get(key)) == canonical(row[key]), "measured new cell identity/quality mismatch")
        measured.append(row)
    metadata._validate_rows(measured)
    return measured


def make_screening_receipt(measured_rows, private_historical_pairs, actual_authority, *,
                           expected_authority_sha256, screening_policy_sha256):
    """Host-only draft for root review; never chooses/sets its expected digest.

    Private history is the root-accepted comparison input. Its sufficiency is
    separately reviewed, and cannot be proven by this pure hash-pair function.
    """
    reader.validate_authority(actual_authority, expected_authority_sha256=expected_authority_sha256)
    metadata._hash(screening_policy_sha256)
    projection = metadata.screen_metadata(measured_rows, private_historical_pairs)
    return canonical({"schema": SCREENING_SCHEMA, "execution_enabled": False,
        "tts_authority": actual_authority["input"], "tts_authority_sha256": expected_authority_sha256,
        "screening_policy_sha256": screening_policy_sha256, "cells": projection["cells"]})


def verify_screening_receipt(receipt_raw, *, expected_receipt_sha256, actual_authority,
                             measured_rows, expected_policy_sha256, expected_authority_sha256):
    """Match immutable reviewed public receipt, all six bytes and dispositions.

    The expected hashes must come from independently reviewed ASR release/input
    authority, not from this receipt or a freshly generated projection. This
    function cannot itself approve private history or actual execution.
    """
    reader.validate_authority(actual_authority, expected_authority_sha256=expected_authority_sha256)
    metadata._hash(expected_receipt_sha256)
    metadata._hash(expected_policy_sha256)
    require(type(receipt_raw) is bytes and reader.sha(receipt_raw) == expected_receipt_sha256, "independent immutable screening receipt hash")
    receipt = reader.decode(receipt_raw, maximum=32 * 1024)
    metadata._exact(receipt, ("schema", "execution_enabled", "tts_authority", "tts_authority_sha256",
                             "screening_policy_sha256", "cells"), "public-safe receipt exact fields")
    require(receipt["schema"] == SCREENING_SCHEMA and receipt["execution_enabled"] is False,
            "receipt is a disabled input binding, not execution permission")
    require(canonical(receipt["tts_authority"]) == canonical(actual_authority["input"]) and
            receipt["tts_authority_sha256"] == expected_authority_sha256 and
            receipt["screening_policy_sha256"] == expected_policy_sha256, "actual independent TTS/source/policy binding")
    metadata._validate_rows(measured_rows)
    projection = {"schema": "k1-asr18-untrusted-metadata-projection-v1", "experiment": metadata.EXPERIMENT,
        "status": "metadata_only_external_custody_and_review_required", "execution_enabled": False,
        "cells": receipt["cells"]}
    metadata._validate_projection(projection)  # Includes symmetric same-batch collisions and quality precedence.
    for expected, observed in zip(receipt["cells"], measured_rows):
        require(canonical({key: expected[key] for key in metadata.ROW_FIELDS}) == canonical(observed),
                "exact six new WAV/PCM/raw/frame/quality identities")
    blind = metadata.blind_metadata(projection)
    require(blind["eligible_cells"] > 0, "zero eligible cells blocks all setup/model loads")
    by_cell = {row["cell_id"]: row for row in measured_rows}
    audio = []
    for mapping in blind["host_only_cell_map"]:
        row = by_cell[mapping["cell_id"]]
        audio.append({"audio_id": mapping["audio_id"], "wav_sha256": row["wav_sha256"],
                      "pcm_sha256": row["pcm_sha256"], "audio_frames": row["frames"]})
    worker_binding = {"schema": "k1-asr18-worker-input-binding-v1", "job_sha256": metadata.digest(blind["job"]),
        "execution_plan_sha256": metadata.digest(blind["execution_plan"]),
        "screening_receipt_sha256": expected_receipt_sha256, "tts_authority_sha256": expected_authority_sha256,
        "audio": audio}
    return {"execution_enabled": False, "projection": projection, "blind": blind,
            "worker_input_binding": worker_binding, "screening_receipt_sha256": expected_receipt_sha256,
            "tts_authority_sha256": expected_authority_sha256}


def bind_complete_input_package(payload, actual_authority, receipt_raw, *, expected_authority_sha256,
                                expected_source_freeze_sha256, expected_receipt_sha256, expected_policy_sha256):
    """Host entry: complete source/ZIP/semantics first, reviewed screening second."""
    members = reader.verify_complete_package(payload, actual_authority,
        expected_authority_sha256=expected_authority_sha256,
        expected_source_freeze_sha256=expected_source_freeze_sha256)
    measured = measure_complete_rows(members)
    verified = verify_screening_receipt(receipt_raw, expected_receipt_sha256=expected_receipt_sha256,
        actual_authority=actual_authority, measured_rows=measured, expected_policy_sha256=expected_policy_sha256,
        expected_authority_sha256=expected_authority_sha256)
    audio = {row["wav_sha256"]: members[f'derived/{row["cell_id"]}.wav'] for row in verified["projection"]["cells"]
             if row["disposition"] == "eligible"}
    return {"verification": verified, "eligible_audio": audio}


def worker_input_files(verification, eligible_audio):
    """Only job.json and eligible audio/... bytes, plus separate opaque metadata."""
    metadata.validate_blind_metadata(verification["projection"], verification["blind"])
    blind = verification["blind"]
    metadata._exact(verification, ("execution_enabled", "projection", "blind", "worker_input_binding",
                                  "screening_receipt_sha256", "tts_authority_sha256"), "verified input result fields only")
    for key in ("screening_receipt_sha256", "tts_authority_sha256"):
        metadata._hash(verification[key])
    by_cell = {row["cell_id"]: row for row in verification["projection"]["cells"]}
    expected_worker_binding = {"schema": "k1-asr18-worker-input-binding-v1", "job_sha256": metadata.digest(blind["job"]),
        "execution_plan_sha256": metadata.digest(blind["execution_plan"]),
        "screening_receipt_sha256": verification["screening_receipt_sha256"],
        "tts_authority_sha256": verification["tts_authority_sha256"], "audio": [
            {"audio_id": mapping["audio_id"], "wav_sha256": by_cell[mapping["cell_id"]]["wav_sha256"],
             "pcm_sha256": by_cell[mapping["cell_id"]]["pcm_sha256"], "audio_frames": by_cell[mapping["cell_id"]]["frames"]}
            for mapping in blind["host_only_cell_map"]]}
    require(canonical(verification["worker_input_binding"]) == canonical(expected_worker_binding), "exact public-safe opaque worker binding")
    require(blind["eligible_cells"] > 0 and verification["execution_enabled"] is False, "disabled eligible binding required")
    require(type(eligible_audio) is dict and set(eligible_audio) == {row["wav_sha256"] for row in blind["job"]["clips"]},
            "exact eligible-only audio roster")
    rows = {row["wav_sha256"]: row for row in verification["projection"]["cells"] if row["disposition"] == "eligible"}
    files = {"job.json": canonical(blind["job"])}
    for clip in blind["job"]["clips"]:
        raw = eligible_audio[clip["wav_sha256"]]
        observed = _wave(raw, native=False)
        row = rows[clip["wav_sha256"]]
        require(all(observed[key] == row[key] for key in ("wav_sha256", "pcm_sha256", "frames")) and
                observed["derived_pcm_silent"] is False, "worker audio is exact non-silent accepted bytes")
        files[clip["audio_path"]] = raw
    return {"input_files": files, "opaque_execution_plan": canonical(blind["execution_plan"]),
            "opaque_input_binding": canonical(verification["worker_input_binding"])}


def stage_worker_inputs(input_root, verification, eligible_audio):
    """Exclusive local input stage. Opaque order/binding are returned, never here.

    The host must mount input_root read-only and mount returned opaque bytes in
    another read-only directory; no receipt, map or TTS source enters /input.
    """
    material = worker_input_files(verification, eligible_audio)
    root = Path(input_root).absolute()
    require(not any(path.is_symlink() for path in (root, *root.parents)), "no input aliases")
    parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(root.name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
        descriptor = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        os.mkdir("audio", mode=0o700, dir_fd=descriptor)
        audio_fd = os.open("audio", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            for name, raw in material["input_files"].items():
                directory = audio_fd if name.startswith("audio/") else descriptor
                filename = name.split("/")[-1]
                fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400, dir_fd=directory)
                with os.fdopen(fd, "wb") as stream:
                    require(stream.write(raw) == len(raw), "complete staged audio write")
                    stream.flush()
                    os.fsync(stream.fileno())
            os.fsync(audio_fd)
        finally:
            os.close(audio_fd)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"input_root": str(root), "opaque_execution_plan": material["opaque_execution_plan"],
            "opaque_input_binding": material["opaque_input_binding"], "execution_enabled": False}
