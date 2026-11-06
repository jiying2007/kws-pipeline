#!/usr/bin/env python3
"""Prepare declared natural-audio windows for the existing audio-only ASR tools.

Standard library only. No acquisition, model calls, queue execution, labels or
negative-duration admission. Parent recordings are read and hashed in full;
only predeclared integer sample slices are copied into the new private bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import re
import struct
import sys
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import active_asr_selective as selective
import research_queue as q
import research_runtime
from research_acoustic_asr import decode_pcm16

DECLARATION = "kws-natural-asr-window-declaration-v1"
BINDINGS = "kws-natural-asr-window-bindings-v1"
SUMMARY = "kws-natural-asr-window-summary-v1"
RATE, MAX_WINDOWS, MAX_PARENTS = 16000, 64, 8
MAX_WINDOW_SAMPLES = RATE * 30
MAX_PARENT_BYTES = 256 * 1024 ** 2
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
BOUNDARY = {
    "evidence_class": "unlabelled-natural-audio-windows",
    "target_presence": "unknown", "human_gold": False,
    "training_admitted": False, "natural_far_available": False,
    "source_independence": "not_established", "new_model_calls": 0,
    "duration_is_negative_exposure": False,
}
DERIVATION = {
    "source": "declared sample slices of mono PCM16 16000Hz parent WAVs",
    "operation": "signed-int16 / 32768 exactly represented as little-endian float32",
    "window_rule": "integer start-inclusive end-exclusive sample ranges",
    "window_wav": "new PCM16 container; exact parent sample bytes retained",
    "resampling": False, "clipping": False, "normalization": False,
    "padding": False, "acoustic_trimming": False,
    "rms_used_as_admission_gate": False,
    "output": "mono16000Hz little-endian float32 PCM",
}


def require(ok, reason):
    q.require(ok, "NATURAL_WINDOW_" + reason)


def relative(value):
    require(type(value) is str and 0 < len(value) <= 512 and "\\" not in value,
            "RELATIVE_PATH_REQUIRED")
    path = Path(value)
    require(not path.is_absolute() and path.as_posix() == value and
            all(part not in ("", ".", "..") for part in value.split("/")),
            "RELATIVE_PATH_REQUIRED")
    return path


def is_id(value):
    return type(value) is str and ID.fullmatch(value) is not None


def is_sha(value):
    return type(value) is str and q.SHA.fullmatch(value) is not None


def validate_declaration(value):
    require(type(value) is dict and set(value) == {
        "schema", "status", "selection_reason", "parents", "windows"} and
        value["schema"] == DECLARATION and value["status"] == "declared_not_materialized",
        "DECLARATION_FIELDS")
    require(type(value["selection_reason"]) is str and
            1 <= len(value["selection_reason"]) <= 1024, "SELECTION_REASON_REQUIRED")
    parents, windows = value["parents"], value["windows"]
    require(type(parents) is list and 1 <= len(parents) <= MAX_PARENTS and
            type(windows) is list and 1 <= len(windows) <= MAX_WINDOWS,
            "BOUNDED_PARENT_AND_WINDOW_ROSTER")
    by_id, paths, pcms, sessions, families = {}, set(), set(), set(), {}
    for parent in parents:
        require(type(parent) is dict and set(parent) == {
            "parent_id", "audio_path", "wav_sha256", "pcm_sha256", "sample_rate_hz",
            "sample_count", "dataset_id", "source_partition", "session_id",
            "source_family_id", "split", "provenance"}, "PARENT_FIELDS")
        for key in ("parent_id", "dataset_id", "source_partition", "session_id", "source_family_id"):
            require(is_id(parent[key]), "PARENT_IDENTITY")
        relative(parent["audio_path"])
        require(is_sha(parent["wav_sha256"]) and is_sha(parent["pcm_sha256"]), "PARENT_HASH")
        require(type(parent["sample_rate_hz"]) is int and parent["sample_rate_hz"] == RATE and
                type(parent["sample_count"]) is int and
                1 <= parent["sample_count"] <= (MAX_PARENT_BYTES - 44) // 2, "PARENT_PCM_GEOMETRY")
        require(parent["split"] in ("train", "development") and
                parent["source_partition"].casefold() not in ("test", "heldout", "holdout"),
                "NO_TEST_OR_HELDOUT_INPUTS")
        session = (parent["dataset_id"], parent["session_id"])
        # One selected mono parent per session prevents alternate channels or
        # parent aliases from inflating the aggregate coverage denominator.
        require(parent["parent_id"] not in by_id and parent["audio_path"] not in paths and
                parent["pcm_sha256"] not in pcms and session not in sessions,
                "DUPLICATE_PARENT_PCM_PATH_OR_SESSION")
        family = parent["source_family_id"]
        require(families.setdefault(family, parent["split"]) == parent["split"],
                "SOURCE_FAMILY_SPLIT_OVERLAP")
        provenance = parent["provenance"]
        require(type(provenance) is dict and set(provenance) == {
            "source_archive_sha256", "source_member", "source_wav_sha256",
            "channel_index", "source_audit_sha256", "annotation_sha256"}, "PROVENANCE_FIELDS")
        for key in ("source_archive_sha256", "source_wav_sha256", "source_audit_sha256", "annotation_sha256"):
            require(provenance[key] is None or is_sha(provenance[key]), "PROVENANCE_HASH")
        if provenance["source_member"] is not None:
            relative(provenance["source_member"])
        require(provenance["channel_index"] is None or
                (type(provenance["channel_index"]) is int and 0 <= provenance["channel_index"] < 64),
                "PROVENANCE_CHANNEL")
        by_id[parent["parent_id"]] = parent
        paths.add(parent["audio_path"]); pcms.add(parent["pcm_sha256"]); sessions.add(session)
    ids, ranges, used = set(), set(), set()
    for window in windows:
        require(type(window) is dict and set(window) == {
            "window_id", "parent_id", "start_sample", "end_sample"}, "WINDOW_FIELDS")
        require(is_id(window["window_id"]) and window["window_id"] not in ids and
                is_id(window["parent_id"]) and window["parent_id"] in by_id, "WINDOW_IDENTITY")
        start, end = window["start_sample"], window["end_sample"]
        require(type(start) is int and type(end) is int and
                0 <= start < end <= by_id[window["parent_id"]]["sample_count"] and
                end - start <= MAX_WINDOW_SAMPLES, "INTEGER_HALF_OPEN_BOUNDS")
        key = (window["parent_id"], start, end)
        require(key not in ranges, "DUPLICATE_WINDOW_RANGE")
        ids.add(window["window_id"]); ranges.add(key); used.add(window["parent_id"])
    require(used == set(by_id), "UNUSED_PARENT_DECLARATION")
    return by_id


def read_declaration(path, expected_sha=None):
    raw = q.read_bytes(path, maximum=1024 ** 2)
    if expected_sha is not None:
        require(is_sha(expected_sha) and q.sha(raw) == expected_sha, "DECLARATION_HASH_CHANGED")
    value = q.loads(raw)
    validate_declaration(value)
    return raw, value


def merged_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def coverage(value):
    rows = []
    for parent in value["parents"]:
        spans = [(w["start_sample"], w["end_sample"]) for w in value["windows"]
                 if w["parent_id"] == parent["parent_id"]]
        union = merged_ranges(spans)
        processed = sum(end - start for start, end in spans)
        unique = sum(end - start for start, end in union)
        rows.append({key: parent[key] for key in
                     ("parent_id", "dataset_id", "session_id", "source_family_id", "split")} | {
            "union_sample_ranges": union, "window_input_samples": processed,
            "unique_parent_samples": unique, "overlap_samples": processed - unique})
    processed = sum(row["window_input_samples"] for row in rows)
    unique = sum(row["unique_parent_samples"] for row in rows)
    return {"sample_rate_hz": RATE, "window_input_samples": processed,
            "unique_parent_samples": unique, "overlap_samples": processed - unique,
            "window_input_seconds": processed / RATE, "unique_parent_seconds": unique / RATE,
            "overlap_seconds": (processed - unique) / RATE, "by_parent": rows,
            "semantics": "union within each parent PCM; not independent or verified negative exposure"}


def parent_pcm(body):
    """Strict complete classic PCM16 RIFF; retain all samples, including silence."""
    require(len(body) >= 44 and body[:4] == b"RIFF" and body[8:12] == b"WAVE" and
            struct.unpack_from("<I", body, 4)[0] + 8 == len(body), "PARENT_RIFF_EXTENT")
    offset, fmt, pcm = 12, None, None
    while offset < len(body):
        require(offset + 8 <= len(body), "PARENT_CHUNK_EXTENT")
        kind, count = struct.unpack_from("<4sI", body, offset)
        start, end = offset + 8, offset + 8 + count
        require(end + (count & 1) <= len(body), "PARENT_CHUNK_EXTENT")
        if kind == b"fmt ":
            require(fmt is None and count in (16, 18), "PARENT_FMT")
            fmt = body[start:end]
            require(struct.unpack_from("<HHIIHH", fmt) == (1, 1, RATE, RATE * 2, 2, 16) and
                    (count == 16 or fmt[16:] == b"\0\0"), "MONO_PCM16_16K_REQUIRED")
        elif kind == b"data":
            require(fmt is not None and pcm is None and count > 0 and count % 2 == 0,
                    "PARENT_DATA_CHUNK")
            pcm = body[start:end]
        offset = end + (count & 1)
    require(fmt is not None and pcm is not None and offset == len(body), "PARENT_PCM_REQUIRED")
    return pcm


def wav_bytes(pcm):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(RATE)
        writer.writeframes(pcm)
    return stream.getvalue()


def producer_sources():
    names = ("tools/prepare_natural_asr_windows.py", "tools/research_acoustic_asr.py",
             "tools/research_queue.py", "tools/active_asr_batch.py", "tools/active_asr_selective.py",
             "tools/summarize_active_asr_batch.py", "tools/active_asr_worker.py",
             "tools/setup_asr_runtime.py", "tools/research_runtime.py", "configs/runtime/research-runtime.json")
    return {name: q.file_record(ROOT / name) for name in names}


def build_files(raw, value, audio_root, created_at):
    """Reconstruct exact expected output bytes from the pinned full parents."""
    parents = validate_declaration(value)
    audio_root = q.safe_path(audio_root)
    require(audio_root.is_dir(), "PARENT_AUDIO_ROOT_REQUIRED")
    slices = {}
    for parent in value["parents"]:
        body = q.read_bytes(audio_root / relative(parent["audio_path"]), maximum=MAX_PARENT_BYTES)
        require(q.sha(body) == parent["wav_sha256"], "PARENT_WAV_HASH_CHANGED")
        pcm = parent_pcm(body)
        require(q.sha(pcm) == parent["pcm_sha256"] and len(pcm) == 2 * parent["sample_count"],
                "PARENT_PCM_HASH_OR_SAMPLE_COUNT_CHANGED")
        for window in value["windows"]:
            if window["parent_id"] == parent["parent_id"]:
                slices[window["window_id"]] = pcm[2 * window["start_sample"]:2 * window["end_sample"]]
    source_files = producer_sources()
    producer_sha = source_files["tools/prepare_natural_asr_windows.py"]["sha256"]
    files, clips, mapping, bindings = {}, [], [], []
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, window in enumerate(value["windows"]):
            parent = parents[window["parent_id"]]
            clip_id = f"clip-{index:03d}"
            pcm = slices[window["window_id"]]
            wav = wav_bytes(pcm)
            observed_pcm, floating, _measurements = decode_pcm16(wav)
            require(observed_pcm == pcm, "PCM_CONVERSION_CHANGED_SLICE")
            frames = window["end_sample"] - window["start_sample"]
            original, derived = f"original/{clip_id}.wav", f"pcm16k/{clip_id}.f32"
            files[original], files[derived] = wav, floating
            archive.writestr(zipfile.ZipInfo(original), wav)
            # Source identity survives a new opaque clip position or alias name.
            source_id = "natural-window-" + q.sha(q.canonical({
                "parent_pcm_sha256": parent["pcm_sha256"], "sample_rate_hz": RATE,
                "start_sample": window["start_sample"], "end_sample": window["end_sample"]}))
            clips.append({"id": clip_id, "original_member": original,
                "original_wav": {"path": original, "sha256": q.sha(wav), "bytes": len(wav)},
                "original_pcm": {"sha256": q.sha(pcm), "sample_rate": RATE, "channels": 1,
                                 "frames": frames, "dtype": "<i2"},
                "derived_pcm": {"path": derived, "sha256": q.sha(floating), "sample_rate": RATE,
                                "channels": 1, "frames": frames, "dtype": "<f4"}})
            mapping.append({"clip_id": clip_id, "source_id": source_id,
                "original_wav_sha256": q.sha(wav), "original_pcm_sha256": q.sha(pcm),
                "derived_pcm_sha256": q.sha(floating), "quality_uncertain": True,
                "quality_reasons": ["unreviewed-natural-audio-window"]})
            bindings.append({"clip_id": clip_id, "source_id": source_id, **window,
                **{key: parent[key] for key in ("dataset_id", "session_id", "source_family_id", "split")},
                "parent_wav_sha256": parent["wav_sha256"], "parent_pcm_sha256": parent["pcm_sha256"],
                "window_wav_sha256": q.sha(wav), "window_pcm_sha256": q.sha(pcm),
                "derived_pcm_sha256": q.sha(floating)})
    archive_raw = archive_stream.getvalue()
    files["window-audio.zip"] = archive_raw
    manifest = {"schema": "active-asr-inputs-v1", "source_archive_sha256": q.sha(archive_raw),
                "created_at": created_at, "preparation_source_sha256": producer_sha,
                "derivation": DERIVATION, "clips": clips}
    files["inputs.json"] = q.canonical(manifest)
    source_map = {"schema": "active-asr-source-map-v2", "input_manifest_sha256": q.sha(files["inputs.json"]),
                  "source_archive_sha256": q.sha(archive_raw), "plan_sha256": q.sha(raw), "rows": mapping}
    files["source-map.json"] = q.canonical(source_map)
    files["declaration.json"] = raw
    files["preparation-source.py"] = q.read_bytes(Path(__file__))
    counts = coverage(value)
    binding = {"schema": BINDINGS, "declaration_sha256": q.sha(raw),
        "inputs_manifest_sha256": q.sha(files["inputs.json"]),
        "source_map_sha256": q.sha(files["source-map.json"]),
        "window_archive_sha256": q.sha(archive_raw), "preparation_sources": source_files,
        "not_decoder_input": True, "target_text_included": False,
        "parent_identity_scope": "full parent WAV and decoded PCM checked against declaration",
        "upstream_provenance_scope": "retained declaration only; upstream archives and annotations not read",
        "parents": value["parents"], "rows": bindings, "coverage": counts, "boundary": BOUNDARY}
    files["natural-window-bindings.json"] = q.canonical(binding)
    files["summary.json"] = q.canonical({"schema": SUMMARY, "status": "materialized_unlabelled_inputs",
        "declaration_sha256": q.sha(raw), "parents": len(parents), "windows": len(clips),
        "coverage": counts, "bindings_sha256": q.sha(files["natural-window-bindings.json"]),
        "boundary": BOUNDARY})
    return files


def prepare(declaration, expected_sha, audio_root, output):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    output = q.safe_path(output)
    require(not output.exists() and output.parent.is_dir(), "CREATE_ONLY_OUTPUT_REQUIRED")
    require(output != ROOT and ROOT not in output.parents, "PRIVATE_OUTPUT_OUTSIDE_REPOSITORY_REQUIRED")
    raw, value = read_declaration(declaration, expected_sha)
    # Validate every parent before creating any materialized output.
    files = build_files(raw, value, audio_root, q.utc())
    output.mkdir(mode=0o700)
    for name, body in files.items():
        path = output / relative(name)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        q.write_new(path, body)
    del files
    return verify(output, audio_root, expected_sha)


def verify(bundle, audio_root, expected_sha):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    bundle = q.safe_path(bundle)
    raw, value = read_declaration(bundle / "declaration.json", expected_sha)
    manifest = q.loads(q.read_bytes(bundle / "inputs.json"))
    created_at = manifest.get("created_at")
    require(type(created_at) is str and 1 <= len(created_at) <= 64, "CREATION_TIME")
    instant = datetime.fromisoformat(created_at)
    require(instant.tzinfo is not None and instant.utcoffset() == timezone.utc.utcoffset(instant), "CREATION_TIME")
    files = build_files(raw, value, audio_root, created_at)
    actual = set()
    for path in bundle.rglob("*"):
        q.safe_path(path)
        require(path.is_file() or path.is_dir(), "REGULAR_BUNDLE_FILES_REQUIRED")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    require(actual == set(files), "BUNDLE_FILE_ROSTER_CHANGED")
    for name, expected in files.items():
        require(q.file_record(bundle / relative(name)) == {"sha256": q.sha(expected), "bytes": len(expected)},
                "BUNDLE_FILE_CHANGED:" + name)
    selective.validate_map(bundle / "inputs.json", bundle / "source-map.json")
    return q.loads(files["summary.json"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate declaration only; no audio reads or writes")
    plan.add_argument("--declaration", type=Path, required=True)
    prep = commands.add_parser("prepare", help="Create a new private audio-only input bundle")
    prep.add_argument("--declaration", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="Reconstruct from the full parents; no model calls or writes")
    check.add_argument("--bundle", type=Path, required=True)
    for command in (prep, check):
        command.add_argument("--declaration-sha256", required=True)
        command.add_argument("--audio-root", type=Path, required=True)
    args = parser.parse_args(argv)
    research_runtime.check_interpreter(research_runtime.load_manifest())
    if args.command == "plan":
        raw, value = read_declaration(args.declaration)
        result = {"schema": SUMMARY, "status": "declared_not_materialized",
                  "declaration_sha256": q.sha(raw), "parents": len(value["parents"]),
                  "windows": len(value["windows"]), "coverage": coverage(value), "boundary": BOUNDARY}
    elif args.command == "prepare":
        result = prepare(args.declaration, args.declaration_sha256, args.audio_root, args.output)
    else:
        result = verify(args.bundle, args.audio_root, args.declaration_sha256)
    print(q.canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
