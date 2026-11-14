"""Exact public synthetic input and a separately declared full-wave derivative.

The archive verifier uses only the standard library. The derive command runs
inside the already hash-verified ASR runtime, before any recognizer is loaded.
Target text stays in the retained original ZIP and never enters a blind job.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
import wave
import zipfile

CAMPAIGN = "public-zh64-asr-20261114-v1"
SOURCE_CAMPAIGN = "qwen-nonclone64-20261104-v1"
SOURCE_SHA = "0704e28608f636fa14e8bfa30ebc5ed5b870b9576ec8c80fbe15da9e05033ac5"
INDEX_SHA = "addfe05bd0a33c0edc74391c514f0d260560238f6fb2d8c2fa6cf414b146b836"
SOURCE_HEAD = "f0a26c5f39910ae6a8cb674a31cf08e57fcfc2f2"
RETAINED_COMMIT = "22505bf68b6c57747a8da35fd83e6ea4982d096c"
PRESETS = ("Vivian", "Serena", "Uncle_Fu", "Dylan")
SPLITS = {"Vivian": "train", "Dylan": "train", "Serena": "dev", "Uncle_Fu": "dev"}
TEXTS = ("你好小窝", "小窝小窝", "你好小好", "你好小你", "小窝你好", "小好小窝", "你好你好", "小小窝窝")
ORDER = tuple(16 * lane + offset for offsets in ((0, 2, 4, 6), (1, 3, 5, 7),
                                                (8, 9, 10, 11), (12, 13, 14, 15))
              for lane in range(4) for offset in offsets)
MIB = 1024 ** 2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(body, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read(path, maximum):
    path = Path(path)
    need(path.resolve() == path.absolute() and path.is_file() and not path.is_symlink(),
         "unaliased regular input")
    before = path.stat()
    need(before.st_nlink == 1 and 0 <= before.st_size <= maximum, "bounded unique input")
    with path.open("rb") as stream:
        body = stream.read(maximum + 1)
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(stamp(before) == stamp(after) and len(body) == before.st_size, "input changed while reading")
    return body


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def request_value():
    return {
        "schema": "public-zh64-first-three-asr-request-v1", "campaign": CAMPAIGN,
        "source_repository": "jiying2007/kws-pipeline", "source_run_id": 41814320901,
        "source_head_sha": SOURCE_HEAD, "source_artifact_id": 13314781496,
        "source_artifact_name": "qwen-nonclone64-20261104-v1-public",
        "source_artifact_sha256": SOURCE_SHA, "source_artifact_bytes": 11560534,
        "source_index_sha256": INDEX_SHA, "source_native_frames": 3123840,
        "source_native_sample_rate": 24000, "source_audio_count": 64,
        "retained_source_commit": RETAINED_COMMIT,
        "preset_split": SPLITS, "split_unit": "entire_public_preset_including_both_seeds",
        "research_train_rows": 32, "research_dev_rows": 32,
        "source_original_intended_split": "train_only", "source_policy_preserved": True,
        "positive_intent_rows": 16, "nearphone_quarantined_rows": 48,
        "ordinary_natural_negative_rows": 0, "optimizer_loss_authorized": False,
        "providers": ["qwen06", "sensevoice", "whisper"], "max_asr_calls": 192,
        "max_asr_calls_per_provider": 64, "attempts_per_cell_provider": 1,
        "retry": False, "new_tts_calls": 0, "new_training_runs": 0,
        "input_transform": {"numpy": "1.26.4", "scipy": "1.17.1",
            "resample": "resample_poly(up=2,down=3,window=('kaiser',5.0),padtype='constant')",
            "pcm16": "floor(float64(resampled_float32)*32768); reject clipping; no gain or trim"},
        "inner_wall_seconds": 9000, "stage_wall_seconds": 1500,
        "source_kind": "synthetic-public-preset", "reference_voice_inputs": 0,
        "public_synthetic_observations_allowed": True,
        "training_admitted": False, "human_gold": False,
        "natural_negative_qualified": False, "evaluation_or_production_use": False,
        "target_text_received_by_recognizer": False, "peer_transcript_received": False,
    }


def frozen_request(path):
    body = read(path, MIB)
    value = parse(body)
    need(value == request_value(), "frozen request changed")
    return value, sha(body)


def float_wav(body):
    need(44 <= len(body) <= MIB and body[:4] == b"RIFF" and body[8:12] == b"WAVE"
         and struct.unpack("<I", body[4:8])[0] + 8 == len(body), "native RIFF framing")
    chunks = {}
    position = 12
    while position < len(body):
        need(position + 8 <= len(body), "native chunk header")
        key = body[position:position + 4]
        size = struct.unpack("<I", body[position + 4:position + 8])[0]
        need(key not in chunks and position + 8 + size <= len(body), "native chunk boundary")
        chunks[key] = body[position + 8:position + 8 + size]
        position += 8 + size + size % 2
    need(position == len(body) and b"fmt " in chunks and b"data" in chunks, "complete native chunks")
    fmt = chunks[b"fmt "]
    need(len(fmt) >= 16 and struct.unpack("<HHIIHH", fmt[:16]) == (3, 1, 24000, 96000, 4, 32),
         "exact native mono float32 geometry")
    pcm = chunks[b"data"]
    need(len(pcm) % 4 == 0 and 0 < len(pcm) // 4 <= 192000, "bounded native frame count")
    values = array.array("f")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    need(all(math.isfinite(x) and abs(x) < 1 for x in values), "native finite unclipped signal")
    return pcm, len(values)


def verify_archive(body):
    need(len(body) == 11560534 and sha(body) == SOURCE_SHA, "exact original source ZIP")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        need(len(names) == len(set(names)) == 98 and
             sum(x.file_size for x in infos) <= 20 * MIB, "closed bounded source ZIP")
        for item in infos:
            path = PurePosixPath(item.filename)
            need(not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
                 and not item.is_dir() and not item.flag_bits & 1
                 and ((item.external_attr >> 16) & 0o170000) in (0, 0o100000), "regular safe ZIP member")
        index_raw = archive.read("artifact-index.json")
        need(sha(index_raw) == INDEX_SHA, "original artifact index identity")
        index = parse(index_raw)
        need(index["campaign"] == SOURCE_CAMPAIGN and index["source_kind"] == "synthetic-public-preset"
             and index["training_admitted"] is False and index["human_gold"] is False
             and set(names) == set(index["files"]) | {"artifact-index.json"}, "source index closure")
        for name, digest in index["files"].items():
            need(sha(archive.read(name)) == digest, "original indexed member changed")
        host = parse(archive.read("host-receipt.json"))
        need(host["head_sha"] == SOURCE_HEAD and host["run_id"] == "41814320901"
             and host["new_asr_calls"] == host["new_training_runs"] == host["reference_voice_inputs"] == 0
             and host["max_model_calls"] == 64 and host["retry"] is False, "original generation provenance")
        rows = []
        for lane, preset in enumerate(PRESETS):
            generation = parse(archive.read(f"lane-{lane}/output/generation.json"))
            need(generation["campaign"] == SOURCE_CAMPAIGN and generation["lane"] == lane
                 and len(generation["planned"]) == len(generation["outcomes"]) == 16, "exact original preset lane")
            for offset, (planned, actual) in enumerate(zip(generation["planned"], generation["outcomes"])):
                ordinal = lane * 16 + offset
                cell = f"{SOURCE_CAMPAIGN}-{ordinal:02d}"
                need(planned["cell_id"] == cell and planned["global_ordinal"] == ordinal
                     and planned["speaker"] == preset and planned["text"] == TEXTS[offset // 2]
                     and planned["repeat"] == offset % 2 and planned["seed"] == 2026110400 + ordinal
                     and planned["attempts"] == 1 and planned["intended_split"] == "train_only"
                     and planned["label_authority"] == "requested_text_only"
                     and planned["intended_kind"] == ("positive-intent" if offset < 4 else "near-miss-intent")
                     and all(actual[k] == v for k, v in planned.items()), "exact original generation cell")
                name = f"lane-{lane}/output/{cell}.raw-float.wav"
                raw = archive.read(name)
                pcm, frames = float_wav(raw)
                need(actual["status"] == "generated" and actual["raw_wav_sha256"] == sha(raw)
                     and actual["native"] == {"sample_rate_hz": 24000, "channels": 1,
                         "frames": frames, "subtype": "FLOAT", "source_values_sha256": sha(pcm)}
                     and actual["metrics"]["technically_valid"] is True
                     and actual["training_admitted"] is False, "native WAV/PCM/source join")
                rows.append({"producer_ordinal": ordinal, "source_cell_id": cell,
                    "original_member": name, "native_wav_sha256": sha(raw),
                    "native_pcm_sha256": sha(pcm), "native_frames": frames,
                    "preset": preset, "research_split": SPLITS[preset],
                    "original_intended_split": "train_only", "training_admitted": False,
                    "original_intended_kind": planned["intended_kind"],
                    "nearphone_quarantined": offset >= 4, "optimizer_loss_authorized": False})
        need(len(rows) == 64 and sum(r["native_frames"] for r in rows) == 3123840
             and len({r["native_wav_sha256"] for r in rows}) == 64
             and len({r["native_pcm_sha256"] for r in rows}) == 64, "complete unique original denominator")
        return rows


def derive(source, output):
    raw = read(source, 20 * MIB)
    rows = verify_archive(raw)  # Entire source is checked before scientific imports.
    need(Path("/runtime").is_dir() and Path("/code/research/source-screen32-v1").is_dir(),
         "fixed offline verified derivative runtime")
    sys.path.insert(0, "/code/research/source-screen32-v1")
    from setup_adapter import verify_runtime
    verify_runtime(Path("/runtime"), "asr")
    import importlib.metadata
    need(importlib.metadata.version("numpy") == "1.26.4" and
         importlib.metadata.version("scipy") == "1.17.1", "locked transform packages")
    import numpy as np
    from scipy.signal import resample_poly
    output = Path(output)
    need(output.is_dir() and not list(output.iterdir()), "new empty derivative directory")
    write(output / "progress.json", {"stage": "load", "index": 0, "started_monotonic": time.monotonic()})
    (output / "audio").mkdir()
    by_ordinal = {r["producer_ordinal"]: r for r in rows}
    clips = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for index, ordinal in enumerate(ORDER, 1):
            row = by_ordinal[ordinal]
            pcm, frames = float_wav(archive.read(row["original_member"]))
            samples = np.frombuffer(pcm, dtype="<f4")
            resampled = resample_poly(samples, 2, 3, window=("kaiser", 5.0), padtype="constant")
            need(resampled.dtype == np.float32 and len(resampled) == (frames * 2 + 2) // 3
                 and np.isfinite(resampled).all(), "whole-wave derivative geometry")
            scaled = np.floor(resampled.astype(np.float64) * 32768.0)
            need(scaled.min() >= -32768 and scaled.max() <= 32767, "derived clipping prohibited")
            pcm16 = scaled.astype("<i2").tobytes()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(pcm16)
            wav = buffer.getvalue(); digest = sha(wav)
            with (output / "audio" / (digest + ".wav")).open("xb") as stream:
                stream.write(wav)
            opaque = f"clip-{index:06d}"
            clips.append({"audio_id": opaque, "path": "audio/" + digest + ".wav",
                "wav_sha256": digest, "pcm_sha256": sha(pcm16), "frames": len(pcm16) // 2,
                "source_id": "public64-" + row["native_pcm_sha256"],
                "request_id": "zh64-" + digest, "producer_ordinal": ordinal})
            row.update(audio_id=opaque, derived_wav_sha256=digest,
                       derived_pcm_sha256=sha(pcm16), derived_frames=len(pcm16) // 2)
    need(len({r["pcm_sha256"] for r in clips}) == 64, "distinct derived input PCM")
    write(output / "input-lock.json", {"schema": "exact-generated-asr64-input-v1", "clips": clips})
    write(output / "source-map.json", {"schema": "public-zh64-original-to-derived-v1",
        "source_artifact_sha256": SOURCE_SHA, "transform": request_value()["input_transform"],
        "rows": rows, "training_admitted": False, "human_gold": False,
        "source_policy_preserved": True, "natural_negative_qualified": False})


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", type=Path)
    cli.add_argument("--derive", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args(argv)
    need(bool(args.verify) != bool(args.derive), "exactly one input operation")
    if args.verify:
        rows = verify_archive(read(args.verify, 20 * MIB))
        print(json.dumps({"native_audio": len(rows), "native_seconds": sum(r["native_frames"] for r in rows) / 24000,
                          "source_sha256": SOURCE_SHA, "new_model_calls": 0}))
    else:
        need(args.output is not None, "derivative destination required")
        derive(args.derive, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
