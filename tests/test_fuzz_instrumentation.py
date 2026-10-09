#!/usr/bin/env python3
"""Build real parser objects and verify fuzz coverage and installed SDK isolation.

Requires Clang/libFuzzer, CMake, a C compiler, and nm. This runs in both official
fuzz jobs; it deliberately fails rather than skips when the toolchain is absent.
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("CMAKE_COMMAND", "cmake")
CLANG = os.environ.get("KWS_FUZZ_CC", "clang")
NM = os.environ.get("NM", "nm")


def run(*command: str | pathlib.Path) -> str:
    result = subprocess.run([str(argument) for argument in command], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


class FuzzInstrumentationTests(unittest.TestCase):
    def test_core_instrumentation_and_shipping_isolation(self):
        with tempfile.TemporaryDirectory(prefix="kws-fuzz-contract-") as temporary:
            directory = pathlib.Path(temporary)
            for enabled in (True, False):
                with self.subTest(fuzz=enabled):
                    build = directory / ("fuzz" if enabled else "plain")
                    # Keep all ordinary tests/tools enabled in the fuzz build to
                    # catch coverage-runtime and duplicate-main link pollution.
                    run(CMAKE, "-S", ROOT, "-B", build, "-G", "Unix Makefiles",
                        f"-DCMAKE_C_COMPILER={CLANG}", "-DCMAKE_BUILD_TYPE=Debug",
                        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DKWS_STRICT=ON",
                        f"-DKWS_BUILD_FUZZ={'ON' if enabled else 'OFF'}")
                    run(CMAKE, "--build", build, "--parallel", "2")
                    ctest = str(pathlib.Path(shutil.which(CMAKE) or CMAKE).with_name("ctest"))
                    run(ctest, "--test-dir", build, "--output-on-failure")
                    commands = json.loads((build / "compile_commands.json").read_text())
                    for source in ("model.c", "keyword_pack.c"):
                        entries = [entry for entry in commands
                                   if pathlib.Path(entry["file"]).name == source]
                        self.assertEqual(len(entries), 2 if enabled else 1)
                        seen = set()
                        for entry in entries:
                            arguments = entry.get("arguments") or shlex.split(entry["command"])
                            output = entry.get("output") or arguments[arguments.index("-o") + 1]
                            obj = pathlib.Path(entry["directory"]) / output
                            fuzz = "kws_fuzz_core.dir" in obj.parts
                            seen.add(fuzz)
                            self.assertEqual("-fsanitize=fuzzer-no-link" in arguments, fuzz)
                            self.assertNotIn("-fsanitize=fuzzer", arguments)
                            # Object symbols, not CMake text or filenames, prove
                            # the real parser has libFuzzer coverage feedback.
                            symbols = run(NM, "-u", obj)
                            self.assertEqual("__sanitizer_cov_" in symbols, fuzz, str(obj))
                        self.assertEqual(seen, {False, True} if enabled else {False})

                    sdk = build / "sdk"
                    run(CMAKE, "--install", build, "--prefix", sdk)
                    archives = list(sdk.rglob("*.a"))
                    self.assertEqual([archive.name for archive in archives], ["libkws_pipeline.a"])
                    symbols = run(NM, "-u", archives[0])
                    self.assertNotIn("__sanitizer_cov_", symbols)
                    self.assertNotIn("__sancov", symbols)
                    for config in list(sdk.rglob("*.cmake")) + list(sdk.rglob("*.pc")):
                        self.assertNotIn("fuzzer", config.read_text())
                    consumer = build / "consumer.c"
                    consumer.write_text('#include <kws_pipeline/kws.h>\n'
                                        'int main(void) { kws_model_t m = {0}; kws_keyword_pack_t p;\n'
                                        ' (void)kws_model_open(0, 0, &m);\n'
                                        ' (void)kws_keyword_pack_open(0, 0, &m, &p); return 0; }\n')
                    # No fuzzer/runtime flags: an ordinary SDK consumer must link.
                    run(os.environ.get("KWS_CONSUMER_CC", "cc"), "-I", sdk / "include",
                        consumer, archives[0], "-lm", "-o", build / "consumer")
                    run(build / "consumer")
                    if enabled:
                        corpus = build / "corpus"
                        run(sys.executable, ROOT / "tests/generate_fuzz_seeds.py", "--output", corpus)
                        for parser in ("model", "keyword_pack"):
                            output = run(build / f"kws_fuzz_{parser}", "-runs=100", "-seed=1",
                                         "-timeout=5", corpus / parser)
                            self.assertIn("cov:", output, "libFuzzer did not report coverage")
                    else:
                        self.assertFalse((build / "libkws_fuzz_core.a").exists())


if __name__ == "__main__":
    unittest.main()
