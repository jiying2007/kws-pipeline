#!/usr/bin/env python3
"""Focused source/ownership/stdio regressions; no actual model or PCM inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMPILERS = ["gcc"]
SANITIZE = False
LEAK_CHECK = False
CORE = ["core/r1d_stream.c", "core/r1d_pcm.c", "core/reference64_fbank.c",
        "core/r1dfsmn.c", "core/sha256.c", "src/fsmn_sdk.c", "src/fsmn_detector.c"]
FLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-fno-fast-math", "-ffp-contract=off", "-fexcess-precision=standard"]
WRAPS = ["malloc", "calloc", "free", "r1dfsmn_load", "r1dfsmn_load_view",
         "r1d_sha256_memory_hex", "r1d_fbank_fbank_init", "r1d_pcm_init", "r1dfsmn_step"]


def checked(command, **kwargs):
    result = subprocess.run([str(x) for x in command], capture_output=True,
                            timeout=90, **kwargs)
    if result.returncode:
        raise AssertionError(f"command failed ({result.returncode}): {command}\n"
                             + result.stdout.decode(errors="replace")
                             + result.stderr.decode(errors="replace"))
    return result


class BorrowedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True
        cls.temp = tempfile.TemporaryDirectory(prefix="r1-borrowed-contracts-")
        cls.root = Path(cls.temp.name)
        spec = importlib.util.spec_from_file_location("r1_borrowed_materializer", HERE / "materialize.py")
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)
        cls.sdk = cls.root / "view"
        cls.report = cls.m.materialize(REPO, cls.sdk)
        qspec = importlib.util.spec_from_file_location("r1_qhead_base_for_test", cls.m.QROOT / "materialize.py")
        q = importlib.util.module_from_spec(qspec)
        qspec.loader.exec_module(q)
        cls.base = cls.root / "base"
        q.materialize_from_repository(REPO, cls.base, kernel="par4")
        cls.builds = {}
        for cc in COMPILERS:
            if not shutil.which(cc):
                raise AssertionError("required compiler unavailable: " + cc)
            build = cls.root / Path(cc).name
            build.mkdir()
            inc = ["-I" + str(cls.sdk / "include"), "-I" + str(cls.sdk / "core")]
            checked([cc, *FLAGS, *inc, "-c", *(cls.sdk / n for n in CORE)], cwd=build)
            objects = [build / (Path(n).stem + ".o") for n in CORE]
            cls.builds[cc] = (build, inc, objects)
        cls.model = cls.root / "invented-model.bin"
        cls.model.write_bytes(b"\x37" * 1960140)
        cls.tables = cls.root / "invented-frontend.bin"
        cls.tables.write_bytes(b"\0" * 83840)
        cls.pcm = cls.root / "invented-pcm.bin"
        cls.pcm.write_bytes(b"\0" * 640)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_old_constructor_and_all_inference_code_are_identical(self):
        for path, start, old_end, new_end in (
            ("core/r1dfsmn.c", "r1dfsmn_status r1dfsmn_load(",
             "r1dfsmn_status r1dfsmn_reset(", "r1dfsmn_status r1dfsmn_load_view("),
            ("core/r1d_stream.c", "r1d_stream *r1d_create(", "int r1d_feed(", "r1d_stream *r1d_create_view("),
            ("src/fsmn_sdk.c", "fsmn_sdk *fsmn_sdk_create(", "int fsmn_sdk_feed(", "fsmn_sdk *fsmn_sdk_create_view("),
        ):
            old = (self.base / path).read_text()
            new = (self.sdk / path).read_text()
            self.assertEqual(old[:old.index(start)], new[:new.index(start)])
            self.assertEqual(self.m.section(old, start, old_end), self.m.section(new, start, new_end))
            self.assertEqual(old[old.index(old_end):], new[new.index(old_end):])
        for path in ("src/fsmn_detector.c", "src/fsmn_run.c", "core/reference64_fbank.c",
                     "core/r1d_pcm.c", "core/model_identity.h"):
            self.assertEqual((self.base / path).read_bytes(), (self.sdk / path).read_bytes())
        self.assertEqual(self.report["model_forwards"], 0)
        self.assertEqual(self.report["frame_event_clock_parity"], "NOT_RUN_FOR_THIS_BUILD")

    def test_source_and_output_guards_fail_without_partial_delivery(self):
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.m.materialize(REPO, self.sdk)
        link = self.root / "symlink"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.m.materialize(REPO, link / "forbidden")
        destination = self.root / "incomplete-output"
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.m.materialize(self.root / "missing-repository", destination)
        self.assertFalse(destination.exists())
        expected = self.m.DEPENDENCIES["materialize.py"]
        try:
            self.m.DEPENDENCIES["materialize.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "dependency drift"):
                self.m.materialize(REPO, destination)
        finally:
            self.m.DEPENDENCIES["materialize.py"] = expected
        self.assertFalse(destination.exists())
        for path, digest in self.m.DEPENDENCIES.items():
            self.assertEqual(hashlib.sha256((self.m.QROOT / path).read_bytes()).hexdigest(), digest)

    def test_real_loader_checks_and_immutability(self):
        for cc, (build, inc, _) in self.builds.items():
            with self.subTest(compiler=cc):
                exe = build / "loader-contract"
                checked([cc, *FLAGS, *inc, HERE / "test_loader.c", build / "r1dfsmn.o",
                         build / "sha256.o", "-lm", "-o", exe])
                result = json.loads(checked([exe]).stdout)
                self.assertEqual(result["model_forwards"], 0)
                self.assertEqual(result["fixture"], "invented_structural_payload")
                if SANITIZE:
                    sanitized = build / "loader-sanitized"
                    checked([cc, *FLAGS, "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                             *inc, HERE / "test_loader.c", self.sdk / "core/r1dfsmn.c",
                             self.sdk / "core/sha256.c", "-lm", "-o", sanitized])
                    env = dict(os.environ, ASAN_OPTIONS=f"detect_leaks={int(LEAK_CHECK)}:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
                    checked([sanitized], env=env)

    def test_mocked_constructor_ownership_and_failures(self):
        for cc, (build, inc, objects) in self.builds.items():
            with self.subTest(compiler=cc):
                exe = build / "ownership-contract"
                wrap = ["-Wl,--wrap=" + name for name in WRAPS]
                checked([cc, *FLAGS, *inc, HERE / "test_ownership.c", *objects,
                         *wrap, "-lm", "-o", exe])
                result = json.loads(checked([exe]).stdout)
                self.assertEqual(result["copy_allocations"], result["copy_frees"])
                self.assertEqual(result["caller_frees_by_sdk"], 0)
                self.assertEqual(result["caller_table_frees_by_sdk"], 0)
                self.assertEqual(result["state_allocations"], result["state_frees"])
                self.assertEqual(result["model_forwards"], 0)
                if SANITIZE:
                    sanitized = build / "ownership-sanitized"
                    checked([cc, *FLAGS, "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                             *inc, HERE / "test_ownership.c", *(self.sdk / p for p in CORE),
                             *wrap, "-lm", "-o", sanitized])
                    env = dict(os.environ, ASAN_OPTIONS=f"detect_leaks={int(LEAK_CHECK)}:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
                    checked([sanitized], env=env)

    def test_real_binaries_compile_and_reject_invented_model(self):
        for cc, (build, inc, objects) in self.builds.items():
            for name in ("fsmn_run", "fsmn_run_view", "fsmn_trace_view", "fsmn_init_bench"):
                with self.subTest(compiler=cc, runner=name):
                    exe = build / name
                    checked([cc, *FLAGS, *inc, self.sdk / "src" / (name + ".c"),
                             *objects, "-lm", "-o", exe])
                    if name == "fsmn_init_bench":
                        args = ["view", self.model, self.tables]
                    elif name == "fsmn_trace_view":
                        args = [self.model, self.tables, self.pcm]
                    else:
                        args = ["--research", self.model, self.tables, "--pcm", self.pcm]
                    result = subprocess.run([str(x) for x in (exe, *args)], capture_output=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)

    def test_new_runner_receipts_and_io_failures_with_explicit_stubs(self):
        for cc, (build, inc, _) in self.builds.items():
            for name in ("fsmn_run_view", "fsmn_trace_view", "fsmn_init_bench"):
                exe = build / (name + "-stdio-stub")
                checked([cc, *FLAGS, *inc, self.sdk / "src" / (name + ".c"),
                         HERE / "test_runner_stubs.c", "-lm", "-o", exe])
                if name == "fsmn_init_bench":
                    args = ["view", self.model, self.tables]
                elif name == "fsmn_trace_view":
                    args = [self.model, self.tables, self.pcm]
                else:
                    args = ["--research", self.model, self.tables, "--pcm", self.pcm]
                command = [str(x) for x in (exe, *args)]
                result = checked(command)
                receipt = json.loads(result.stdout if name == "fsmn_init_bench" else result.stderr)
                if name == "fsmn_init_bench":
                    self.assertEqual(receipt["model_forwards"], 0)
                    self.assertEqual(receipt["pcm_bytes"], 0)
                    copy = json.loads(checked([exe, "copy", self.model, self.tables]).stdout)
                    mapped = json.loads(checked([exe, "mmap", self.model, self.tables]).stdout)
                    self.assertEqual(copy["model_forwards"], 0)
                    self.assertEqual(mapped["model_forwards"], 0)
                    self.assertEqual(mapped["pcm_bytes"], 0)
                    self.assertEqual(mapped["model_storage"], "read_only_private_file_mapping")
                    self.assertEqual(mapped["copied_model_bytes"], 0)
                    self.assertEqual(copy["copied_model_bytes"], 1960140)
                    for value in (receipt, copy, mapped):
                        self.assertGreaterEqual(value["model_load_and_init_cpu_ms"], 0)
                        self.assertGreaterEqual(value["model_load_and_init_wall_ms"],
                                                value["model_init_wall_ms"])
                else:
                    self.assertEqual(receipt["ok"] if name == "fsmn_trace_view" else receipt["status"],
                                     True if name == "fsmn_trace_view" else "complete")
                failure = subprocess.run(command, capture_output=True, timeout=10,
                                         env=dict(os.environ, R1_TEST_FAIL_DESTROY="1"))
                self.assertNotEqual(failure.returncode, 0)
                if name != "fsmn_init_bench":
                    failed_receipt = json.loads(failure.stderr)
                    self.assertEqual(failed_receipt["ok"] if name == "fsmn_trace_view" else failed_receipt["status"],
                                     False if name == "fsmn_trace_view" else "failed")
                self.assertTrue(Path("/dev/full").is_char_device(), "Linux /dev/full required")
                # The init tool writes its receipt to stdout; others write both channels.
                for channel in (["stdout"] if name == "fsmn_init_bench" else ["stdout", "stderr"]):
                    with open("/dev/full", "wb") as full:
                        result = subprocess.run(command, timeout=10,
                                                stdout=full if channel == "stdout" else subprocess.PIPE,
                                                stderr=full if channel == "stderr" else subprocess.PIPE)
                    self.assertNotEqual(result.returncode, 0, (cc, name, channel))


    def test_mmap_fails_closed_for_symlink_bad_size_and_nonregular_files(self):
        """The stub SDK checks input bytes at destroy, proving mmap lifetime."""
        source = (HERE / "init_bench.c").read_text()
        for required in ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK", "PROT_READ,MAP_PRIVATE",
                         "munmap(model,FSMN_MODEL_BYTES)", "S_ISREG(st.st_mode)"):
            self.assertIn(required, source)
        for cc, (build, inc, _) in self.builds.items():
            with self.subTest(compiler=cc):
                exe = build / "mmap-model-stub"
                checked([cc, *FLAGS, *inc, HERE / "init_bench.c",
                         HERE / "test_runner_stubs.c", "-lm", "-o", exe])
                valid = json.loads(checked([exe, "mmap", self.model, self.tables]).stdout)
                self.assertEqual(valid["initializations"], 1)
                self.assertEqual(valid["model_forwards"], 0)
                self.assertEqual(valid["mode"], "mmap")
                self.assertEqual(valid["model_storage"], "read_only_private_file_mapping")
                self.assertEqual(valid["copied_model_bytes"], 0)
                self.assertFalse(valid["physical_ssc305"])
                self.assertGreaterEqual(valid["model_load_and_init_cpu_ms"], 0)
                fake = build / "bad-size-model"
                fake.write_bytes(b"\x37" * 640)
                link = build / "linked-model"
                link.symlink_to(self.model)
                directory = build / "a-directory"
                directory.mkdir(exist_ok=True)
                fifo = build / "model-fifo-no-writer"
                os.mkfifo(fifo)
                for bad_model in (fake, link, directory, Path("/dev/null"), fifo):
                    result = subprocess.run([str(x) for x in
                        (exe, "mmap", bad_model, self.tables)],
                        capture_output=True, timeout=2)
                    self.assertNotEqual(result.returncode, 0, str(bad_model))
                    self.assertFalse(result.stdout)
                not_table = build / "invalid-frontend"
                not_table.write_bytes(b"\x00" * 640)
                result = subprocess.run([str(x) for x in
                    (exe, "mmap", self.model, not_table)],
                    capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout)
                failed_destroy = subprocess.run([str(x) for x in
                    (exe, "mmap", self.model, self.tables)],
                    env=dict(os.environ, R1_TEST_FAIL_DESTROY="1"),
                    capture_output=True, timeout=10)
                self.assertNotEqual(failed_destroy.returncode, 0)
                self.assertFalse(failed_destroy.stdout)
                with open("/dev/full", "wb") as full:
                    bad_output = subprocess.run([str(x) for x in
                        (exe, "mmap", self.model, self.tables)],
                        stdout=full, stderr=subprocess.PIPE, timeout=10)
                self.assertNotEqual(bad_output.returncode, 0)


    def test_offline_stream_mmap_preserves_sdk_lifetime_and_receipt(self):
        """Real stream resource source, explicit no-model stub only."""
        source = REPO / "research/r1_stream_c/quant_head8/target_resource_bench.c"
        for cc, (build, inc, _) in self.builds.items():
            with self.subTest(compiler=cc):
                exe = build / "mmap-offline-stream-stub"
                checked([cc, *FLAGS, *inc, source,
                         HERE / "test_runner_stubs.c", "-lm", "-o", exe])
                arguments = [self.model, self.tables, self.pcm]
                copied = json.loads(checked([exe, *arguments]).stdout)
                mapped = json.loads(checked([exe, *arguments, "--mmap"]).stdout)
                self.assertEqual(copied["schema"], "r1-strong-fsmn-offline-board-resource-v1")
                self.assertEqual(mapped["schema"], copied["schema"])
                self.assertEqual(copied["model_storage"], "sdk_owned_copy")
                self.assertEqual(mapped["model_storage"], "read_only_mmap")
                self.assertEqual(copied["model_caller_heap_bytes"], 1960140)
                self.assertEqual(mapped["model_caller_heap_bytes"], 0)
                for item in (copied, mapped):
                    self.assertEqual(item["pcm_bytes"], 640)
                    self.assertEqual(item["feed_calls"], 1)
                    self.assertEqual(item["sdk_model_frames"], 1)
                    self.assertEqual(item["keyword_events"], 1)
                    self.assertEqual(item["keyword1_events"], 1)
                    self.assertFalse(item["device_qualified"])
                    self.assertFalse(item["live_afe"])
                    self.assertFalse(item["independent_far_frr"])
                    self.assertGreaterEqual(item["offline_feed_cpu_seconds"], 0)
                    self.assertGreater(item["process_peak_rss_kib"], 0)
                    self.assertIn("proc_self_pss_kib", item)
                    self.assertIn("proc_self_private_dirty_kib", item)
                self.assertEqual(mapped["sdk_model_frames"], copied["sdk_model_frames"])
                self.assertEqual(mapped["keyword_events"], copied["keyword_events"])
                missing = build / "wrong-size-mmap"
                missing.write_bytes(b"\x37" * 640)
                link = build / "symlink-mmap-model"
                link.symlink_to(self.model)
                directory = build / "directory-not-model"
                directory.mkdir(exist_ok=True)
                fifo = build / "stream-model-fifo-no-writer"
                os.mkfifo(fifo)
                for bad in (missing, link, directory, Path("/dev/null"), fifo):
                    rejected = subprocess.run([str(x) for x in
                        (exe, bad, self.tables, self.pcm, "--mmap")],
                        capture_output=True, timeout=2)
                    self.assertNotEqual(rejected.returncode, 0, str(bad))
                    self.assertFalse(rejected.stdout)
                failed_destroy = subprocess.run([str(x) for x in
                    (exe, self.model, self.tables, self.pcm, "--mmap")],
                    env=dict(os.environ, R1_TEST_FAIL_DESTROY="1"),
                    capture_output=True, timeout=10)
                self.assertNotEqual(failed_destroy.returncode, 0)
                self.assertFalse(failed_destroy.stdout)
                self.assertTrue(Path("/dev/full").is_char_device())
                with open("/dev/full", "wb") as full:
                    output_error = subprocess.run([str(x) for x in
                        (exe, self.model, self.tables, self.pcm, "--mmap")],
                        stdout=full, stderr=subprocess.PIPE, timeout=10)
                self.assertNotEqual(output_error.returncode, 0)
                self.assertIn(b"unable to write complete resource receipt",
                              output_error.stderr)

    def test_offline_mmap_real_sdk_rejects_invented_model(self):
        source = REPO / "research/r1_stream_c/quant_head8/target_resource_bench.c"
        for cc, (build, inc, objects) in self.builds.items():
            with self.subTest(compiler=cc):
                exe = build / "mmap-offline-real-sdk"
                checked([cc, *FLAGS, *inc, source, *objects, "-lm", "-o", exe])
                for flags in ([], ["--mmap"]):
                    rejected = subprocess.run([str(x) for x in
                        (exe, self.model, self.tables, self.pcm, *flags)],
                        capture_output=True, timeout=15)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertFalse(rejected.stdout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", action="append")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--leak-check", action="store_true",
                        help="enable LeakSanitizer (unsupported under ptrace); requires --sanitize")
    args, remaining = parser.parse_known_args()
    COMPILERS = args.cc or ["gcc"]
    SANITIZE = args.sanitize
    LEAK_CHECK = args.leak_check
    if LEAK_CHECK and not SANITIZE:
        parser.error("--leak-check requires --sanitize")
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
