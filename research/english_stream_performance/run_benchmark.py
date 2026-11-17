#!/usr/bin/env python3
"""Build a new consumer of an authenticated installed SDK; replay each PCM once.

No download, synthesis, ASR, training, source modification, or automatic retry.
The caller authenticates the upstream SDK artifact before invoking this driver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def save_process_logs(output, case_id, stdout, stderr):
    for suffix, value in ((".stdout.json", stdout), (".stderr", stderr)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        (output / (case_id + suffix)).write_text(value or "", encoding="utf-8")


def load_inputs(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "english-stream-performance-inputs-v1":
        raise ValueError("unexpected input manifest schema")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError("expected 1..16 input cases")
    seen = set()
    total = 0
    inputs = []
    for entry in cases:
        case = dict(entry)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", case_id) or case_id in seen:
            raise ValueError("invalid or duplicate case id")
        seen.add(case_id)
        sample_count = case.get("samples")
        if type(sample_count) is not int or not 51200 <= sample_count <= 4800000:
            raise ValueError("each input must contain 3.2..300 seconds at 16 kHz")
        if case.get("sample_rate_hz") != 16000 or case.get("encoding") != "s16le_mono":
            raise ValueError("inputs must be explicitly declared 16 kHz mono s16le")
        if case.get("speech_kind") not in {"natural_speech", "synthetic_speech", "mixed_speech", "environment_recording"}:
            raise ValueError("declare actual input type; artificial verification signals are not speech")
        if case.get("continuity") not in {"single_recording", "concatenated_recordings"}:
            raise ValueError("declare whether original recording boundaries were concatenated")
        if not isinstance(case.get("source"), dict) or not case["source"]:
            raise ValueError("each case requires upstream source provenance")
        expected = case.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid PCM sha256")
        raw_path = case.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("missing PCM path")
        pcm = Path(raw_path)
        if not pcm.is_absolute():
            pcm = path.parent / pcm
        if not pcm.is_file() or pcm.is_symlink() or pcm.stat().st_size != sample_count * 2 or digest(pcm) != expected:
            raise ValueError("PCM length/hash/regular-file check failed: " + case_id)
        total += sample_count
        inputs.append((case, pcm.resolve()))
    if total > 4800000:
        raise ValueError("bounded initial experiment allows at most 300 seconds total input")
    return manifest, inputs


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "english-stream-performance-run-v1", "passed": False,
              "measurement_completed": False, "performance_budget_evaluated": False,
              "started_unix_seconds": time.time(), "cases": [],
              "input_repetitions": 1, "performance_only": True,
              "far_frr_measured": False, "word_end_latency_measured": False,
              "ssc305_qualification": False, "model_promotion_allowed": False}
    write_json(output / "RESULT.json", report)
    try:
        manifest_path = args.manifest.resolve()
        manifest, inputs = load_inputs(manifest_path)
        report["input_manifest"] = {"sha256": digest(manifest_path), "content": manifest}
        sdk = args.sdk_root.resolve()
        header = sdk / "include/ekws_research_stream.h"
        library = sdk / "lib/libekws_research_stream.a"
        head = sdk / "models/causal-head.bin"
        source = Path(__file__).with_name("ekws_stream_benchmark.c")
        for file in (header, library, head, source):
            if not file.is_file() or file.is_symlink():
                raise ValueError("missing regular SDK/source file: " + str(file))
        if head.stat().st_size != 197908:
            raise ValueError("unexpected SDK head size")
        report["sdk_files"] = {str(file.relative_to(sdk)): identity(file) for file in (header, library, head)}
        report["benchmark_source"] = identity(source)
        report["benchmark_driver"] = identity(Path(__file__))
        report["host"] = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                          "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
                          "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
        version = subprocess.run([args.cc, "--version"], text=True, capture_output=True, timeout=30, check=True)
        report["compiler_version"] = version.stdout
        executable = output / "ekws_stream_benchmark"
        command = [args.cc, "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                   "-I" + str(header.parent), str(source), str(library), "-lm", "-lrt", "-o", str(executable)]
        report["build_command"] = command
        built = subprocess.run(command, text=True, capture_output=True, timeout=180)
        (output / "build.stdout").write_text(built.stdout, encoding="utf-8")
        (output / "build.stderr").write_text(built.stderr, encoding="utf-8")
        report["build_exit_code"] = built.returncode
        if built.returncode:
            raise RuntimeError("benchmark consumer build failed")
        report["executable"] = identity(executable)
        write_json(output / "RESULT.json", report)
        for case, pcm in inputs:
            case_result = {"input": case, "status": "started", "started_unix_seconds": time.time()}
            report["cases"].append(case_result)
            write_json(output / "RESULT.json", report)
            try:
                completed = subprocess.run([str(executable), str(head), str(pcm)], text=True, capture_output=True, timeout=args.case_timeout)
            except subprocess.TimeoutExpired as error:
                save_process_logs(output, case["id"], error.stdout, error.stderr)
                case_result["timed_out"] = True
                case_result["termination"] = "driver_timeout_killed_process"
                raise
            save_process_logs(output, case["id"], completed.stdout, completed.stderr)
            case_result["exit_code"] = completed.returncode
            case_result["finished_unix_seconds"] = time.time()
            case_result["pcm_sha256_after"] = digest(pcm)
            if completed.returncode or case_result["pcm_sha256_after"] != case["sha256"]:
                case_result["status"] = "failed_no_retry"
                raise RuntimeError("benchmark case failed: " + case["id"])
            measurements = json.loads(completed.stdout)
            if (measurements.get("schema") != "english-stream-performance-v1" or measurements.get("passed") is not True
                    or measurements.get("input_samples") != case["samples"]
                    or measurements.get("score_observer_enabled") is not False
                    or measurements.get("input_repetitions") != 1 or measurements.get("eof_padding_samples") != 0
                    or measurements.get("resets") != 0):
                raise RuntimeError("benchmark result contract mismatch")
            case_result["measurements"] = measurements
            case_result["status"] = "completed"
            write_json(output / "RESULT.json", report)
        if digest(head) != report["sdk_files"]["models/causal-head.bin"]["sha256"]:
            raise RuntimeError("head changed during measurement")
        report["passed"] = True
        report["measurement_completed"] = True
        report["actual_input_seconds"] = sum(c["samples"] for c, _ in inputs) / 16000
        report["independent_far_test_hours"] = 0
        report["limitations"] = ["Unpaced host replay of saved input; not SSC305 or real-time microphone evidence.",
            "Warm-up and steady measurements preserve the same stream state; every input is replayed once.",
            "Per-call time is service time, not acoustic word-end latency; average throughput does not guarantee real-time deadlines.",
            "Process memory includes the benchmark input and instrumentation; sampled PSS is not an isolated KWS peak.",
            "Input provenance and speech kind are caller-declared; this experiment does not grant label or model admission."]
    except Exception as error:
        report["failure"] = {"class": type(error).__name__, "message": str(error), "retry_performed": False}
        if report["cases"] and report["cases"][-1]["status"] == "started":
            report["cases"][-1]["status"] = "failed_no_retry"
            report["cases"][-1]["finished_unix_seconds"] = time.time()
    finally:
        report["finished_unix_seconds"] = time.time()
        write_json(output / "RESULT.json", report)
    return 0 if report["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--case-timeout", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.case_timeout <= 1800:
        parser.error("case timeout must be in 1..1800 seconds")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
