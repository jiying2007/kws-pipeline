#!/usr/bin/env python3
"""Opt-in bounded build, eight legacy and two capture mock groups.

No payloads, real audio, Torch, downloads or numerical qualification. Production
is inspected as ELF symbols only, never loaded. All generated files are removed.
Linux /proc and resource limits are required; there is no weaker fallback.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

MODULE = "research/fsmn2599-stream-c"
DECODER = "research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018"
FFT = "research/donor_fbank/fft_twiddles.h"
LOCAL_FILES = (
    "b_stream_adapter.c", "b_stream_adapter.h", "canonical_pcm.c", "canonical_pcm.h",
    "canonical_fbank.c", "canonical_fbank.h", "canonical_tables.h", "fsmn2599.c",
    "fsmn2599.h", "sha256.c", "sha256.h", "softmax2599.c", "softmax2599.h",
    "test_b_stream_adapter.c", "example_pcm.c", "bsa_capture.h",
    "test_capture_contract.c", "test_cpp_link.cpp",
    "b_profile.c", "b_profile.h", "test_profile_contract.c", "test_profile_unit.c",
    "test_timer_overhead.c", "test_affine_block2.c", "affine_reference.inc",
    "check_profile.py", "check_arm_synthetic.py", "test_arm_synthetic.py", "qemu-i386-tool.json", "build.py",
    "check_arm_dependencies.py", "test_arm_dependencies.py", "arm_environment_probe.c",
)
INPUTS = frozenset([f"{MODULE}/{name}" for name in LOCAL_FILES] +
                   [f"{DECODER}/kws3.c", f"{DECODER}/kws3.h", FFT])
ADAPTER = {"bsa_" + name for name in (
    "model", "abi_size", "storage_bytes", "create", "begin", "feed", "finish",
    "reset", "get", "destroy")}
DECODER_EXPORTS = {"kws3_" + name for name in (
    "create", "create_with_threshold", "get_threshold_status", "destroy", "storage_bytes",
    "reset", "step", "step_f32", "process", "finish", "get_summary", "status_string", "keyword")}
CORE_EXPORTS = {
    "kcf_pcm_init", "kcf_pcm_feed", "kcf_pcm_finish", "kcf_fbank_init",
    "kcf_fbank_frame", "kcf_fbank_callback", "fsmn2599_load", "fsmn2599_reset",
    "fsmn2599_copy_cache", "fsmn2599_step", "fsmn2599_environment_ok",
    "kws_sha256_memory_hex", "ksx_softmax2599",
}
FLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-fno-fast-math", "-ffp-contract=off", "-fexcess-precision=standard"]
AS_LIMIT = 512 * 1024 * 1024
DISK_LIMIT = 16 * 1024 * 1024
OUTPUT_LIMIT = 1024 * 1024
FIXTURE_MESSAGE = "eight synthetic glue groups completed; no acoustic qualification"
COMPILE_BUDGET_SECONDS = 60


def validate_sources(root):
    """An exact fixed set, regular nonsymlink files, sizes and SHA-256 identities."""
    binding = root / MODULE / "source-binding.json"
    if binding.is_symlink() or binding.stat().st_size > OUTPUT_LIMIT:
        raise RuntimeError("invalid source binding")
    data = json.loads(binding.read_text(encoding="utf-8"))
    required = {"schema", "files", "qualification_allowed"}
    if (not isinstance(data, dict) or not required <= set(data) or
            set(data) - required - {"dependency_repo_commit", "note"} or
            type(data["schema"]) is not int or data["schema"] != 1 or
            data["qualification_allowed"] is not False or not isinstance(data["files"], list)):
        raise RuntimeError("invalid binding schema")
    if "dependency_repo_commit" in data:
        commit = data["dependency_repo_commit"]
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise RuntimeError("invalid dependency metadata")
    if "note" in data and not isinstance(data["note"], str):
        raise RuntimeError("invalid binding note")
    seen = set()
    for record in data["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise RuntimeError("invalid binding record")
        name = record["path"]
        if not isinstance(name, str) or name not in INPUTS or name in seen:
            raise RuntimeError("unexpected or repeated source path")
        seen.add(name)
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise RuntimeError("symlink source path")
        if not path.is_file() or path.resolve() != root / name:
            raise RuntimeError("nonregular source path")
        size = record["bytes"]
        digest = record["sha256"]
        if type(size) is not int or not 0 < size <= DISK_LIMIT:
            raise RuntimeError("invalid source size")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError("invalid source digest")
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source binding mismatch: " + name)
    if seen != INPUTS:
        raise RuntimeError("incomplete source binding")
    return hashlib.sha256(binding.read_bytes()).hexdigest()


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def group_cpu(pgid):
    # Include each process's waited-for descendants, so gcc's exited cc1/as/ld
    # remain charged. Transient double counting is conservative, never a rebate.
    ticks = os.sysconf("SC_CLK_TCK")
    total = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                total += sum(int(fields[i]) for i in (11, 12, 13, 14))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total / ticks


def disk_bytes(directory):
    total = 0
    for path in directory.rglob("*"):
        try:
            if not path.is_symlink() and path.is_file():
                total += path.stat().st_size
        except (FileNotFoundError, ProcessLookupError):
            continue  # Compiler-owned temporaries may disappear between reads.
    return total


def run_bounded(argv, directory, wall, cpu, output_limit=OUTPUT_LIMIT):
    if wall <= 0 or cpu <= 0 or not Path("/proc/self/stat").is_file():
        raise RuntimeError("resource budget unavailable")
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
        seconds = max(1, math.ceil(cpu))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (DISK_LIMIT, DISK_LIMIT))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "TMPDIR": str(directory)}
    start = time.monotonic()
    before = child_cpu()
    output = bytearray()
    proc = None
    selector = None
    try:
        proc = subprocess.Popen(argv, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=limits)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        while selector.get_map() or proc.poll() is None:
            if time.monotonic() - start > wall or group_cpu(proc.pid) > cpu:
                raise RuntimeError("child wall/aggregate CPU budget exhausted")
            if disk_bytes(directory) > DISK_LIMIT:
                raise RuntimeError("aggregate temporary output exceeds 16 MiB")
            for key, _ in selector.select(0.02):
                block = os.read(key.fd, 65536)
                if not block:
                    selector.unregister(key.fileobj)
                else:
                    output.extend(block)
                    if len(output) > output_limit:
                        raise RuntimeError("child output limit exceeded")
        proc.wait()
        elapsed_cpu = child_cpu() - before
        if elapsed_cpu > cpu or time.monotonic() - start > wall:
            raise RuntimeError("child budget exhausted")
        if disk_bytes(directory) > DISK_LIMIT:
            raise RuntimeError("aggregate temporary output exceeds 16 MiB")
        if proc.returncode:
            raise RuntimeError("child failed (%s): %s" %
                               (proc.returncode, output.decode("utf-8", "replace")))
        return output.decode("utf-8", "strict"), elapsed_cpu
    finally:
        # Also kill orphaned group members after a successful direct-child exit.
        try:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            try:
                if proc is not None:
                    proc.wait()
            finally:
                try:
                    if selector is not None:
                        selector.close()
                finally:
                    if proc is not None and proc.stdout is not None:
                        proc.stdout.close()


def execute():
    root = Path(__file__).resolve().parents[2]
    if "LD_PRELOAD" in os.environ or "LD_LIBRARY_PATH" in os.environ:
        raise RuntimeError("loader overrides are forbidden")
    binding_sha = validate_sources(root)
    gcc = shutil.which("gcc", path="/usr/bin:/bin")
    cxx = shutil.which("g++", path="/usr/bin:/bin")
    nm = shutil.which("nm", path="/usr/bin:/bin")
    if not gcc or not cxx or not nm:
        raise RuntimeError("gcc, g++ and nm are required")
    module = root / MODULE
    decoder = root / DECODER
    with tempfile.TemporaryDirectory(prefix="fsmn-stream-build-") as temporary:
        temp = Path(temporary)
        vendor = temp / "include" / "vendor"
        vendor.mkdir(parents=True)
        (vendor / "fft_twiddles.h").symlink_to(root / FFT)
        include = ["-I" + str(module), "-I" + str(decoder), "-I" + str(temp / "include")]
        production = temp / "libb_stream_adapter.so"
        fixture = temp / "test_b_stream_adapter"
        example = temp / "example_pcm"
        capture = temp / "test_capture_contract"
        cpp = temp / "test_cpp_link"
        shared_sources = [module / name for name in (
            "b_stream_adapter.c", "canonical_pcm.c", "canonical_fbank.c", "fsmn2599.c",
            "sha256.c", "softmax2599.c")] + [decoder / "kws3.c"]
        compile_start = time.monotonic()
        compile_cpu = 0.0
        commands = [
            [gcc, *FLAGS, "-fPIC", "-shared", "-Wl,-z,defs", *include,
             *map(str, shared_sources), "-lm", "-o", str(production)],
            [gcc, *FLAGS, *include, str(module / "test_b_stream_adapter.c"),
             str(module / "canonical_pcm.c"), "-lm", "-o", str(fixture)],
            # Compile/link the public usage example, but never execute it.
            [gcc, *FLAGS, *include, str(module / "example_pcm.c"),
             *map(str, shared_sources), "-lm", "-o", str(example)],
            [gcc, *FLAGS, *include, str(module / "test_capture_contract.c"),
             str(module / "canonical_pcm.c"), "-lm", "-o", str(capture)],
            [cxx, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic", *include,
             str(module / "test_cpp_link.cpp"), str(production), "-lm", "-o", str(cpp)],
        ]
        for command in commands:
            _, spent = run_bounded(
                command, temp, COMPILE_BUDGET_SECONDS - (time.monotonic() - compile_start),
                COMPILE_BUDGET_SECONDS - compile_cpu)
            compile_cpu += spent
        compile_wall = time.monotonic() - compile_start
        if validate_sources(root) != binding_sha:
            raise RuntimeError("source binding changed during build")
        symbols, _ = run_bounded([nm, "-D", "--defined-only", str(production)], temp, 5, 5)
        exported = {line.split()[-1] for line in symbols.splitlines() if line.strip()}
        expected = ADAPTER | DECODER_EXPORTS | CORE_EXPORTS
        if exported != expected:
            raise RuntimeError("production export mismatch: " + repr({
                "missing": sorted(expected - exported), "unexpected": sorted(exported - expected)}))
        # Legacy mock executable: exactly eight groups once.
        result, fixture_cpu = run_bounded([str(fixture)], temp, 30, 30)
        if result.strip() != FIXTURE_MESSAGE:
            raise RuntimeError("unexpected fixture completion output")
        if validate_sources(root) != binding_sha:
            raise RuntimeError("source binding changed during fixtures")
        capture_result, capture_cpu = run_bounded([str(capture)], temp, 30, 30)
        if capture_result.strip() != "capture contract groups=2; gap reset and chunk invariance; mock only":
            raise RuntimeError("unexpected capture fixture output")
        if validate_sources(root) != binding_sha:
            raise RuntimeError("source binding changed during capture fixtures")
        report = {"schema": 1, "capture_fixture_groups": 2, "capture_fixture_runs": 1,
                  "capture_fixture_cpu_seconds": capture_cpu, "cpp_link_pass": True,
                  "cpp_executed": False, "source_binding_sha256": binding_sha,
                  "production_loaded": False, "fixture_groups": 8, "fixture_runs": 1,
                  "compiler_invocations": 5, "example_executed": False,
                  "compile_budget_seconds": COMPILE_BUDGET_SECONDS,
                  "qualification_allowed": False, "compile_wall_seconds": compile_wall,
                  "compile_cpu_seconds": compile_cpu, "fixture_cpu_seconds": fixture_cpu,
                  "production_exports": sorted(exported), "fixture_message": result.strip(),
                  "production_sha256": hashlib.sha256(production.read_bytes()).hexdigest(),
                  "example_sha256": hashlib.sha256(example.read_bytes()).hexdigest(),
                  "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="build and run mocked glue fixtures once")
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    print(json.dumps(execute(), sort_keys=True))


if __name__ == "__main__":
    main()
