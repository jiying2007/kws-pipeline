#!/usr/bin/env python3
"""Compile or run fictional-token C contracts. No model weights or audio."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEPLOY = HERE / "candidate_sdk"


def checked_process(command, *, env, timeout, phase):
    """Retain the actual bounded compiler/fixture diagnostics on failure."""
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        print(json.dumps({"status": "TIMED_OUT", "phase": phase,
                          "timeout_seconds": timeout, "stdout": decoded(exc.stdout),
                          "stderr": decoded(exc.stderr)}, sort_keys=True), file=sys.stderr, flush=True)
        raise
    if result.returncode:
        print(json.dumps({"status": "FAILED", "phase": phase,
                          "returncode": result.returncode, "stdout": result.stdout,
                          "stderr": result.stderr}, sort_keys=True), file=sys.stderr, flush=True)
        result.check_returncode()
    return result


def run(cc="gcc", sanitize=False, compile_only=False):
    spec = importlib.util.spec_from_file_location("original_sdk_mock_header", DEPLOY / "run_contracts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    with tempfile.TemporaryDirectory(prefix="r1-gap-fixtures-") as directory:
        root = Path(directory)
        (root / "r1d_stream.h").write_text(module.HEADER)
        command = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                   "-fno-fast-math", "-ffp-contract=off", "-I" + str(root), "-I" + str(DEPLOY),
                   str(HERE / "test_gap.c"), str(DEPLOY / "r1_sdk.c"),
                   "-Wl,--wrap=malloc,--wrap=calloc,--wrap=free", "-lm", "-o", str(root / "test")]
        if sanitize:
            command[2:2] = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        checked_process(command, env=env, timeout=45, phase="compile:" + cc)
        if compile_only:
            return {"status": "COMPILED_NOT_EXECUTED", "compiler": cc, "model_forwards": 0,
                    "program_invocations": 0}
        result = checked_process([str(root / "test")], env=env, timeout=30,
                                 phase="mock-fixture:" + cc)
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        if (len(lines) != 2 or lines[0].get("groups") != 21 or
                lines[1].get("new_groups") != 15 or lines[1].get("legacy_groups") != 21 or
                any(r.get("status") != "PASS" or r.get("model_forwards") != 0 for r in lines)):
            raise RuntimeError("incomplete original/new fixture run")
        return {"status": "PASS", "compiler": cc, "sanitize": sanitize,
                "model_forwards": 0, "program_invocations": 1, "reports": lines}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.cc, args.sanitize, args.compile_only), sort_keys=True))
