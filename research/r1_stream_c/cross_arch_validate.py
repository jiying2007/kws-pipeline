#!/usr/bin/env python3
"""Bounded, streaming x86-64/ARMv7-A replay for the fixed R1 Binary64 reference.

The default validator uses only the Python standard library and keeps memory
bounded to one captured frame per architecture. NumPy is imported lazily only
when optional saved-NPZ comparisons are requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import struct

MODEL_BYTES = 3028860
TABLE_BYTES = 83840
MODEL_SHA256 = "e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb"
TABLE_SHA256 = "49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49"
CHECKPOINT_SHA256 = "99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571"
MAGIC = b"R1DREF1\0"
CLOCK_STRUCT = struct.Struct("<QQQ")
FEATURE_STRUCT = struct.Struct("<400d")
LOGIT_STRUCT = struct.Struct("<2601d")
FRAME_BYTES = CLOCK_STRUCT.size + FEATURE_STRUCT.size + LOGIT_STRUCT.size

FEATURE_ATOL = 1.0e-8
FEATURE_RTOL = 1.0e-8
FEATURE_MAX_ABS = 1.0e-8
LOGIT_ATOL = 1.0e-8
LOGIT_RTOL = 1.0e-8
LOGIT_MAX_ABS = 1.0e-6
POSTERIOR_ATOL = 1.0e-8
POSTERIOR_RTOL = 1.0e-8
POSTERIOR_MAX_ABS = 1.0e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_input_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("schema") == "kws-r1-arm-input-recovery-v1",
            "unsupported R1 input manifest")
    records = data.get("records")
    require(isinstance(records, list) and 1 <= len(records) <= 256,
            "input record count outside fixed bound")
    ids = set()
    indices = set()
    for record in records:
        index = record.get("index")
        source_id = record.get("source_id")
        require(type(index) is int and 0 <= index < 256, "invalid record index")
        require(isinstance(source_id, str) and source_id and source_id not in ids,
                "invalid or duplicate source identity")
        require(index not in indices, "duplicate record index")
        require(record.get("split") in ("train", "development"),
                "unexpected source role")
        require(type(record.get("samples")) is int and record["samples"] > 0,
                "invalid PCM sample count")
        require(type(record.get("pcm_bytes")) is int and
                record["pcm_bytes"] == record["samples"] * 2,
                "PCM byte/sample geometry mismatch")
        require(isinstance(record.get("pcm_sha256"), str) and
                len(record["pcm_sha256"]) == 64, "invalid PCM hash")
        shape = record.get("feature_shape")
        require(isinstance(shape, list) and len(shape) == 2 and
                type(shape[0]) is int and shape[0] > 0 and shape[1] == 400,
                "invalid saved feature geometry")
        ids.add(source_id)
        indices.add(index)
    records.sort(key=lambda row: row["index"])
    require([row["index"] for row in records] == list(range(len(records))),
            "input indices must be contiguous and ordered")
    return data


def stable_softmax(logits: Tuple[float, ...]) -> Tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    denominator = math.fsum(exponentials)
    require(math.isfinite(denominator) and denominator > 0.0,
            "stable-softmax denominator is invalid")
    return tuple(value / denominator for value in exponentials)


def append_greedy(sequence: List[int], previous: Optional[int],
                  logits: Tuple[float, ...]) -> Optional[int]:
    token = max(range(len(logits)), key=logits.__getitem__)
    if token != 0 and token != previous:
        sequence.append(token)
    return token


def open_capture(path: Path, expected_frames: int):
    expected_bytes = len(MAGIC) + expected_frames * FRAME_BYTES
    require(path.is_file() and not path.is_symlink(),
            "capture file missing or symlinked: " + str(path))
    require(path.stat().st_size == expected_bytes,
            "capture byte count mismatch: " + str(path))
    stream = path.open("rb")
    if stream.read(len(MAGIC)) != MAGIC:
        stream.close()
        raise ValueError("capture magic mismatch: " + str(path))
    return stream


def read_frame(stream, label: str, index: int):
    raw = stream.read(FRAME_BYTES)
    require(len(raw) == FRAME_BYTES,
            "truncated capture frame %d (%s)" % (index, label))
    frame, source_end, delivery_end = CLOCK_STRUCT.unpack_from(raw, 0)
    feature_offset = CLOCK_STRUCT.size
    logit_offset = feature_offset + FEATURE_STRUCT.size
    features = FEATURE_STRUCT.unpack_from(raw, feature_offset)
    logits = LOGIT_STRUCT.unpack_from(raw, logit_offset)
    return raw, frame, source_end, delivery_end, features, logits


def compare_record(x86_path: Path, arm_path: Path, samples: int,
                   expected_frames: int) -> Dict[str, Any]:
    require(expected_frames > 0, "expected frame count must be positive")
    x86 = open_capture(x86_path, expected_frames)
    arm = open_capture(arm_path, expected_frames)
    feature_max = 0.0
    logit_max = 0.0
    posterior_max = 0.0
    bitwise_identical = True
    greedy: List[int] = []
    previous: Optional[int] = None
    try:
        for index in range(expected_frames):
            xraw, xf, xs, xd, xfeatures, xlogits = read_frame(x86, "x86", index)
            araw, af, ass, ad, afeatures, alogits = read_frame(arm, "arm", index)
            require((xf, xs, xd) == (af, ass, ad),
                    "x86/ARM source clock mismatch")
            require(xf == index, "non-contiguous model frame sequence")
            require(xs == 720 + 480 * index,
                    "unexpected source dependency clock")
            require(xd >= xs and xd <= samples and xd % 320 == 0,
                    "invalid delivery clock")
            bitwise_identical = bitwise_identical and xraw == araw

            for left, right in zip(xfeatures, afeatures):
                require(math.isfinite(left) and math.isfinite(right),
                        "non-finite cross-architecture feature")
                difference = abs(left - right)
                feature_max = max(feature_max, difference)
                require(difference <= FEATURE_ATOL + FEATURE_RTOL * abs(left) and
                        difference <= FEATURE_MAX_ABS,
                        "cross-architecture feature numerical contract failed")

            for left, right in zip(xlogits, alogits):
                require(math.isfinite(left) and math.isfinite(right),
                        "non-finite cross-architecture logit")
                difference = abs(left - right)
                logit_max = max(logit_max, difference)
                require(difference <= LOGIT_ATOL + LOGIT_RTOL * abs(left) and
                        difference <= LOGIT_MAX_ABS,
                        "cross-architecture logit numerical contract failed")

            xposterior = stable_softmax(xlogits)
            aposterior = stable_softmax(alogits)
            for left, right in zip(xposterior, aposterior):
                difference = abs(left - right)
                posterior_max = max(posterior_max, difference)
                require(difference <= POSTERIOR_ATOL + POSTERIOR_RTOL * abs(left) and
                        difference <= POSTERIOR_MAX_ABS,
                        "cross-architecture posterior numerical contract failed")

            x_top = max(range(len(xlogits)), key=xlogits.__getitem__)
            a_top = max(range(len(alogits)), key=alogits.__getitem__)
            require(x_top == a_top, "cross-architecture frame argmax mismatch")
            previous = append_greedy(greedy, previous, xlogits)

        require(x86.read(1) == b"" and arm.read(1) == b"",
                "capture contains unaccounted trailing bytes")
    finally:
        x86.close()
        arm.close()

    with x86_path.open("rb") as stream:
        stream.seek(len(MAGIC) + (expected_frames - 1) * FRAME_BYTES)
        last_raw = stream.read(FRAME_BYTES)
    _, _, _, final_delivery = CLOCK_STRUCT.unpack_from(last_raw, 0)
    require(0 <= samples - final_delivery < 480,
            "EOF residual sample accounting mismatch")

    return {
        "frames": expected_frames,
        "bitwise_identical": bitwise_identical,
        "max_feature_abs_difference": feature_max,
        "max_logit_abs_difference": logit_max,
        "max_posterior_abs_difference": posterior_max,
        "source_clock_equal": True,
        "argmax_equal": True,
        "greedy_sequence": greedy,
    }


def check_status(path: Path, expected_frames: int,
                 expected_pcm_bytes: int) -> Dict[str, Any]:
    status = json.loads(path.read_text(encoding="utf-8"))
    require(status.get("status") == "complete", "capture did not complete")
    require(status.get("frames") == expected_frames, "capture frame count mismatch")
    require(status.get("pcm_bytes") == expected_pcm_bytes, "capture PCM byte count mismatch")
    require(status.get("synthetic_eof") is False, "synthetic EOF output is prohibited")
    return status


def compare_saved_numeric(result: Dict[str, Any], record: Dict[str, Any],
                          x86_path: Path, frames: int, numeric_root: Path,
                          final_index: Dict[str, Any]) -> None:
    import numpy as np
    archive_path = numeric_root / record["numeric_archive"]
    member = record["numeric_member"]
    require(archive_path.is_file(), "saved numeric archive missing: " + str(archive_path))
    import zipfile
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        with archive.open(member, "r") as stream:
            with np.load(stream) as saved:
                expected_clock = np.asarray(saved["clock"], dtype=np.uint64)
                saved_logits = np.asarray(saved["logits"], dtype=np.float32)
    require(expected_clock.shape == (frames, 3),
            "saved FP32 clock shape mismatch")
    require(saved_logits.shape == (frames, 2601),
            "saved FP32 logit shape mismatch")
    actual_clock = []
    saved_argmax_equal = True
    greedy: List[int] = []
    previous: Optional[int] = None
    with x86_path.open("rb") as stream:
        require(stream.read(len(MAGIC)) == MAGIC, "x86 capture magic mismatch")
        for index in range(frames):
            raw = stream.read(FRAME_BYTES)
            require(len(raw) == FRAME_BYTES, "x86 capture truncated during saved comparison")
            frame, source_end, delivery_end = CLOCK_STRUCT.unpack_from(raw, 0)
            logits = LOGIT_STRUCT.unpack_from(raw, CLOCK_STRUCT.size + FEATURE_STRUCT.size)
            actual_clock.append((frame, source_end, delivery_end))
            expected_top = int(np.argmax(saved_logits[index]))
            actual_top = max(range(len(logits)), key=logits.__getitem__)
            saved_argmax_equal = saved_argmax_equal and expected_top == actual_top
            previous = append_greedy(greedy, previous, logits)
    require(np.array_equal(np.asarray(actual_clock, dtype=np.uint64), expected_clock),
            "saved FP32 source clock mismatch")
    require(saved_argmax_equal, "saved FP32 argmax mismatch")
    expected = final_index[record["source_id"]]["greedy_token_ids"]
    require(greedy == expected, "saved greedy sequence mismatch")
    result["saved_fp32_clock_equal"] = True
    result["saved_fp32_argmax_equal"] = True
    result["saved_greedy_sequence_equal"] = True


def run_one(binary: Path, qemu: Optional[Path], model: Path, tables: Path,
            pcm: Path, output: Path, timeout_seconds: int) -> Tuple[Path, Dict[str, Any]]:
    command: List[str] = []
    if qemu is not None:
        command.extend([str(qemu), "-cpu", "cortex-a15"])
    command.extend([str(binary), str(model), str(tables)])
    with pcm.open("rb") as source, output.open("xb") as target:
        completed = subprocess.run(command, stdin=source, stdout=target,
                                   stderr=subprocess.PIPE, timeout=timeout_seconds)
    status_path = output.with_suffix(".stderr.json")
    status_path.write_bytes(completed.stderr)
    require(completed.returncode == 0,
            "capture process failed; inspect retained output: " + str(output))
    return output, json.loads(completed.stderr.decode("utf-8"))


def verify_build_manifest(path: Path, host_binary: Path, arm_binary: Path,
                          qemu: Path) -> Dict[str, Any]:
    build = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "host/r1d_capture_x86_64": host_binary,
        "arm/r1d_capture_armhf_static": arm_binary,
        "executor/qemu-arm-static": qemu,
    }
    for name, actual in expected.items():
        require(name in build.get("files", {}), "build manifest missing " + name)
        pin = build["files"][name]
        require(actual.is_file() and not actual.is_symlink() and
                actual.stat().st_size == pin["bytes"] and
                sha256_file(actual) == pin["sha256"],
                "build artifact identity mismatch: " + name)
    return build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--pcm-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--host-binary", type=Path, required=True)
    parser.add_argument("--arm-binary", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path)
    parser.add_argument("--final-index", type=Path)
    args = parser.parse_args()

    try:
        manifest = load_input_manifest(args.input_manifest)
        require(args.output.is_absolute() and not args.output.exists() and
                not args.output.is_symlink(), "output must be a fresh absolute directory")
        require(args.model.is_file() and not args.model.is_symlink() and
                args.model.stat().st_size == MODEL_BYTES and
                sha256_file(args.model) == MODEL_SHA256, "model payload identity mismatch")
        require(args.tables.is_file() and not args.tables.is_symlink() and
                args.tables.stat().st_size == TABLE_BYTES and
                sha256_file(args.tables) == TABLE_SHA256, "frontend table identity mismatch")
        require(args.host_binary.is_file() and args.arm_binary.is_file() and
                args.qemu.is_file(), "one or more capture executables are missing")
        build = None
        if args.build_manifest is not None:
            build = verify_build_manifest(args.build_manifest, args.host_binary,
                                          args.arm_binary, args.qemu)
        if args.numeric_root is not None:
            require(args.final_index is not None, "--final-index is required with --numeric-root")
            final_index = json.loads(args.final_index.read_text(encoding="utf-8"))
        else:
            require(args.final_index is None, "--final-index requires --numeric-root")
            final_index = {}

        args.output.mkdir(parents=True)
        host_dir = args.output / "x86"
        arm_dir = args.output / "arm"
        host_dir.mkdir()
        arm_dir.mkdir()
        records_out: List[Dict[str, Any]] = []
        start = time.monotonic()
        total_frames = 0
        total_pcm_bytes = 0
        for record in manifest["records"]:
            index = record["index"]
            pcm = args.pcm_dir / ("%03d.pcm" % index)
            require(pcm.is_file() and not pcm.is_symlink() and
                    pcm.stat().st_size == record["pcm_bytes"] and
                    sha256_file(pcm) == record["pcm_sha256"],
                    "PCM identity mismatch for record %d" % index)
            x86_path = host_dir / ("%03d.bin" % index)
            arm_path = arm_dir / ("%03d.bin" % index)
            _, x86_status = run_one(args.host_binary, None, args.model, args.tables,
                                    pcm, x86_path, 60)
            _, arm_status = run_one(args.arm_binary, args.qemu, args.model, args.tables,
                                    pcm, arm_path, 120)
            frames = int(record["feature_shape"][0])
            check_status(x86_path.with_suffix(".stderr.json"), frames, record["pcm_bytes"])
            check_status(arm_path.with_suffix(".stderr.json"), frames, record["pcm_bytes"])
            result = compare_record(x86_path, arm_path, record["samples"], frames)
            if args.numeric_root is not None:
                compare_saved_numeric(result, record, x86_path, frames,
                                      args.numeric_root, final_index)
            result.update({
                "index": index,
                "source_id": record["source_id"],
                "split": record["split"],
                "pcm_sha256": record["pcm_sha256"],
                "x86_output_sha256": sha256_file(x86_path),
                "arm_output_sha256": sha256_file(arm_path),
                "x86_state_bytes": x86_status.get("state_bytes"),
                "arm_state_bytes": arm_status.get("state_bytes"),
            })
            records_out.append(result)
            total_frames += frames
            total_pcm_bytes += record["pcm_bytes"]

        summary = {
            "schema": "kws-r1-reference64-cross-arch-replay-v1",
            "status": "PASS_CROSS_ARCH_NUMERIC_CONTRACT",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_payload_sha256": sha256_file(args.model),
            "frontend_table_sha256": sha256_file(args.tables),
            "input_manifest_sha256": sha256_file(args.input_manifest),
            "input_capsule_sha256": manifest.get("input_capsule_sha256"),
            "build_manifest_sha256": sha256_file(args.build_manifest) if args.build_manifest else None,
            "build_source_commit": build.get("source_commit") if build else None,
            "record_count": len(records_out),
            "total_frames_per_arch": total_frames,
            "total_pcm_bytes": total_pcm_bytes,
            "audio_seconds": total_pcm_bytes / 32000.0,
            "accepted_inference_streams": 2 * len(records_out),
            "accepted_model_frames": 2 * total_frames,
            "bitwise_identical_records": sum(1 for row in records_out if row["bitwise_identical"]),
            "max_feature_abs_difference": max(row["max_feature_abs_difference"] for row in records_out),
            "max_logit_abs_difference": max(row["max_logit_abs_difference"] for row in records_out),
            "max_posterior_abs_difference": max(row["max_posterior_abs_difference"] for row in records_out),
            "new_training_updates": 0,
            "new_training_forwards": 0,
            "new_tts": 0,
            "new_asr": 0,
            "physical_ssc305": False,
            "int8_qualification": False,
            "acoustic_qualification": False,
            "records": records_out,
            "elapsed_seconds": time.monotonic() - start,
        }
        summary_path = args.output / "CROSS_ARCH_REPLAY.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items()
                          if key != "records"}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, ImportError) as exc:
        failure = {"schema": "kws-r1-cross-arch-failure-v1",
                   "status": "FAILED_OR_INCOMPLETE", "reason": str(exc),
                   "model_training": 0, "tts": 0, "asr": 0}
        if args.output.exists() and args.output.is_dir():
            path = args.output / "FAILURE.json"
            if not path.exists():
                path.write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n",
                                encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
