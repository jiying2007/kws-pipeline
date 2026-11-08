#!/usr/bin/env python3
"""Run and compare the fixed R1 Binary64 reference on x86-64 and ARMv7-A.

This is a research numerical-replay tool. It does not train, synthesize audio,
call ASR, calibrate wake events, or qualify a physical SSC305.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL_BYTES = 3028860
TABLE_BYTES = 83840
MODEL_SHA256 = "e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb"
TABLE_SHA256 = "49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49"
CHECKPOINT_SHA256 = "99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571"
MAGIC = b"R1DREF1\0"
FRAME_DTYPE = np.dtype([
    ("frame", "<u8"),
    ("source_end", "<u8"),
    ("delivery_end", "<u8"),
    ("features", "<f8", (400,)),
    ("logits", "<f8", (2601,)),
])
FRAME_BYTES = 8 * 3 + 8 * 400 + 8 * 2601

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
                "unexpected retained source role")
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


def read_capture(path: Path, expected_frames: int) -> np.ndarray:
    raw = path.read_bytes()
    expected_bytes = 8 + expected_frames * FRAME_BYTES
    require(len(raw) == expected_bytes, "capture byte count mismatch: " + str(path))
    require(raw[:8] == MAGIC, "capture magic mismatch: " + str(path))
    return np.frombuffer(raw, dtype=FRAME_DTYPE, offset=8, count=expected_frames)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits - maximum)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def greedy_sequence(logits: np.ndarray) -> List[int]:
    sequence: List[int] = []
    previous: Optional[int] = None
    for value in np.argmax(logits, axis=1):
        token = int(value)
        if token != 0 and token != previous:
            sequence.append(token)
        previous = token
    return sequence


def check_status(path: Path, expected_frames: int, expected_pcm_bytes: int) -> Dict[str, Any]:
    status = json.loads(path.read_text(encoding="utf-8"))
    require(status.get("status") == "complete", "capture did not complete")
    require(status.get("frames") == expected_frames, "capture frame count mismatch")
    require(status.get("pcm_bytes") == expected_pcm_bytes, "capture PCM byte count mismatch")
    require(status.get("synthetic_eof") is False, "synthetic EOF output is prohibited")
    return status


def compare_record(x86: np.ndarray, arm: np.ndarray, samples: int) -> Dict[str, Any]:
    require(x86.shape == arm.shape, "x86/ARM frame count mismatch")
    x_clock = np.column_stack((x86["frame"], x86["source_end"], x86["delivery_end"]))
    a_clock = np.column_stack((arm["frame"], arm["source_end"], arm["delivery_end"]))
    require(np.array_equal(x_clock, a_clock), "x86/ARM source clock mismatch")
    frame_ids = x_clock[:, 0]
    require(np.array_equal(frame_ids, np.arange(len(x86), dtype=np.uint64)),
            "non-contiguous model frame sequence")
    require(np.array_equal(x_clock[:, 1], 720 + 480 * frame_ids),
            "unexpected source dependency clock")
    require(np.all(x_clock[:, 2] >= x_clock[:, 1]) and
            np.all(x_clock[:, 2] <= samples) and
            np.all(x_clock[:, 2] % 320 == 0),
            "invalid delivery clock")
    require(0 <= samples - int(x_clock[-1, 2]) < 480,
            "EOF residual sample accounting mismatch")

    feature_diff = np.abs(x86["features"] - arm["features"])
    logit_diff = np.abs(x86["logits"] - arm["logits"])
    require(np.all(feature_diff <= FEATURE_ATOL + FEATURE_RTOL * np.abs(x86["features"])) and
            float(np.max(feature_diff)) <= FEATURE_MAX_ABS,
            "cross-architecture feature numerical contract failed")
    require(np.all(logit_diff <= LOGIT_ATOL + LOGIT_RTOL * np.abs(x86["logits"])) and
            float(np.max(logit_diff)) <= LOGIT_MAX_ABS,
            "cross-architecture logit numerical contract failed")

    x_posterior = stable_softmax(x86["logits"])
    a_posterior = stable_softmax(arm["logits"])
    posterior_diff = np.abs(x_posterior - a_posterior)
    require(np.all(posterior_diff <= POSTERIOR_ATOL + POSTERIOR_RTOL * np.abs(x_posterior)) and
            float(np.max(posterior_diff)) <= POSTERIOR_MAX_ABS,
            "cross-architecture posterior numerical contract failed")
    x_argmax = np.argmax(x86["logits"], axis=1)
    a_argmax = np.argmax(arm["logits"], axis=1)
    require(np.array_equal(x_argmax, a_argmax), "cross-architecture frame argmax mismatch")
    return {
        "frames": int(len(x86)),
        "bitwise_identical": bool(np.array_equal(x86.view(np.uint8), arm.view(np.uint8))),
        "max_feature_abs_difference": float(np.max(feature_diff)),
        "max_logit_abs_difference": float(np.max(logit_diff)),
        "max_posterior_abs_difference": float(np.max(posterior_diff)),
        "source_clock_equal": True,
        "argmax_equal": True,
        "greedy_sequence": greedy_sequence(x86["logits"]),
    }


def compare_saved_numeric(result: Dict[str, Any], record: Dict[str, Any],
                          x86: np.ndarray, numeric_root: Path,
                          final_index: Dict[str, Any]) -> None:
    archive_path = numeric_root / record["numeric_archive"]
    member = record["numeric_member"]
    require(archive_path.is_file(), "saved numeric archive missing: " + str(archive_path))
    import zipfile
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        with archive.open(member, "r") as stream:
            with np.load(stream) as saved:
                expected_clock = np.asarray(saved["clock"], dtype=np.uint64)
                actual_clock = np.column_stack((
                    x86["frame"], x86["source_end"], x86["delivery_end"])).astype(np.uint64)
                require(np.array_equal(actual_clock, expected_clock),
                        "saved FP32 source clock mismatch")
                saved_logits = np.asarray(saved["logits"], dtype=np.float32)
                require(np.array_equal(np.argmax(x86["logits"], axis=1),
                                       np.argmax(saved_logits, axis=1)),
                        "saved FP32 argmax mismatch")
    expected = final_index[record["source_id"]]["greedy_token_ids"]
    require(result["greedy_sequence"] == expected,
            "saved greedy sequence mismatch")
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
        require(actual.is_file() and actual.stat().st_size == pin["bytes"] and
                sha256_file(actual) == pin["sha256"], "build artifact identity mismatch: " + name)
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
            x86 = read_capture(x86_path, frames)
            arm = read_capture(arm_path, frames)
            result = compare_record(x86, arm, record["samples"])
            if args.numeric_root is not None:
                compare_saved_numeric(result, record, x86, args.numeric_root, final_index)
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

        feature_max = max(row["max_feature_abs_difference"] for row in records_out)
        logit_max = max(row["max_logit_abs_difference"] for row in records_out)
        posterior_max = max(row["max_posterior_abs_difference"] for row in records_out)
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
            "max_feature_abs_difference": feature_max,
            "max_logit_abs_difference": logit_max,
            "max_posterior_abs_difference": posterior_max,
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
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
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
