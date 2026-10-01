#!/usr/bin/env python3
# New kws-pipeline test glue. Apache-2.0; see LICENSE.
"""Pure fake launcher tests; no compiler, shared library or model.

All fake payloads are literal text. This must never point to a real bin/kws.
Run directly with the CLI source root as the sole argument.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


PAYLOADS = (
    "bin/kws", "lib/libsherpa-onnx-kws-c-api.so", "lib/libonnxruntime.so",
    "models/encoder.int8.onnx", "models/decoder.onnx", "models/joiner.onnx",
    "models/tokens.txt", "config/keywords.txt", "config/profile.json",
)
EXPECTED_ENV = {
    "PATH": "/usr/bin:/bin", "LC_ALL": "C", "ORT_DISABLE_TELEMETRY": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
}


def need(value, message):
    if not value:
        raise AssertionError(message)


def write_manifest(kit):
    text = "".join(hashlib.sha256((kit / path).read_bytes()).hexdigest() + "  " + path + "\n" for path in PAYLOADS)
    (kit / "MANIFEST.sha256").write_text(text)


def main(source):
    # The script contains an absolute interpreter path from this authorized test's
    # own Python executable, not from a guessed installation or a downloaded tool.
    interpreter = pathlib.Path(sys.executable).resolve()
    need(" " not in str(interpreter) and "\n" not in str(interpreter), "unsupported fake interpreter path")
    fake = f"#!{interpreter}\n" + (
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], 'env': dict(os.environ)}, sort_keys=True))\n"
    )
    with tempfile.TemporaryDirectory(prefix="kws launcher fake ") as temp:
        base = pathlib.Path(temp)
        caller = base / "caller with spaces"; caller.mkdir()
        kit = base / "kit original"; kit.mkdir()
        for path in PAYLOADS:
            target = kit / path; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fake if path == "bin/kws" else "FAKE TEST PAYLOAD: never load this file\n")
        (kit / "bin/kws").chmod(0o700)
        shutil.copyfile(source / "run-kws", kit / "run-kws")
        (kit / "run-kws").chmod(0o700)
        write_manifest(kit)
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": "99", "OPENBLAS_NUM_THREADS": "99",
                            "ORT_DISABLE_TELEMETRY": "0", "LD_LIBRARY_PATH": str(base / "nonexistent"),
                            "LD_PRELOAD": "", "LD_AUDIT": "", "PYTHONPATH": str(base / "nonexistent"),
                            "KWS_TEST_SENTINEL": "must not reach exec"})

        def invoke(root, args):
            return subprocess.run([str(root / "run-kws"), *args], cwd=caller, env=environment,
                                  text=True, capture_output=True, timeout=10, check=False)

        cases = [["relative.wav"], ["sub dir/audio file.wav"], ["--wav", "--", "-starts-with-dash.wav"],
                 ["--pcm-s16le-16000-mono", "--", "a b.pcm"], ["--", "--"], ["--", ""]]
        for moved in (False, True):
            if moved:
                relocated = base / "relocated kit with spaces"
                kit.rename(relocated); kit = relocated
            for args in cases:
                result = invoke(kit, args)
                need(result.returncode == 0, "launcher failed with fake executable: " + result.stderr)
                row = json.loads(result.stdout)
                need(row["cwd"] == str(caller), "launcher changed the caller CWD")
                need(row["args"] == args, "launcher altered positional arguments")
                # Python may set LC_CTYPE=C.UTF-8 itself (locale coercion), so only
                # require the exact launcher variables plus that documented entry.
                actual = row["env"]
                need(all(actual.get(k) == v for k, v in EXPECTED_ENV.items()), "fixed environment mismatch")
                need(set(actual) <= set(EXPECTED_ENV) | {"LC_CTYPE"}, "inherited environment survived exec")
                need(result.stderr == "", "successful hash check polluted stderr")
        symlink = base / "run via symlink"
        symlink.symlink_to(kit / "run-kws")
        linked = subprocess.run([str(symlink), "--", "-relative.wav"], cwd=caller, env=environment,
                                text=True, capture_output=True, timeout=10, check=False)
        need(linked.returncode == 0 and json.loads(linked.stdout)["cwd"] == str(caller), "launcher symlink relocation")
        # Each mandatory payload's changed bytes must stop before fake execution.
        for path in PAYLOADS:
            target = kit / path; before = target.read_bytes(); target.write_bytes(before + b"changed\n")
            result = invoke(kit, ["relative.wav"])
            target.write_bytes(before)
            need(result.returncode != 0 and result.stdout == "", "hash mismatch reached executable: " + path)
        (kit / "MANIFEST.sha256").unlink()
        result = invoke(kit, ["relative.wav"])
        need(result.returncode != 0 and result.stdout == "", "missing manifest reached executable")
    print(json.dumps({"fake_launcher_checks": "passed", "compiler_called": False,
                      "model_or_shared_library_loaded": False}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: launcher_fake_tests.py CLI_SOURCE_ROOT")
    main(pathlib.Path(sys.argv[1]).resolve())
