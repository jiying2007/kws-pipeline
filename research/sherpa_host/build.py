#!/usr/bin/env python3
"""Explicit offline, single-job build; no model execution or runtime staging.

Apache-2.0; see the repository LICENSE.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from materialize import ROOT, checked_bytes, digest, load_json, verify_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source, build, ort = args.source.absolute(), args.build.absolute(), args.ort.absolute()
    verify_tree(source, load_json(ROOT / "sources.lock.json"))
    lock = load_json(ROOT / "dependencies.lock.json")
    checked_bytes(ort.parent, ort.name, lock["ort_runtime"]["sha256"], lock["ort_runtime"]["bytes"])
    if (build.exists() or build.is_symlink() or build.parent.resolve(strict=True) != build.parent
            or build.is_relative_to(ROOT.parents[1]) or build.is_relative_to(source)):
        raise ValueError("use a new build directory outside the repository and source tree")
    identities = {}
    for name, executable in (("cmake", args.cmake), ("gcc", args.cc), ("g++", args.cxx), ("make", "make"), ("ld", "ld")):
        path = Path(shutil.which(executable) or executable).resolve(strict=True)
        identities[name] = {"sha256": digest(path.read_bytes()),
                            "version": subprocess.check_output([str(path), "--version"], text=True).splitlines()[0]}
    command = [args.cmake, "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
               "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + args.cc,
               "-DCMAKE_CXX_COMPILER=" + args.cxx, "-DKWS_ORT_LIBRARY:FILEPATH=" + str(ort)]
    subprocess.run(command, check=True)
    subprocess.run([args.cmake, "--build", str(build), "--target", "kws", "--parallel", "1"], check=True)
    artifacts = {name: digest((build / name).read_bytes()) for name in ("bin/kws", "lib/libsherpa-onnx-kws-c-api.so", "compile_commands.json")}
    receipt = {"schema": 1, "tools": identities, "artifacts": artifacts,
               "source_manifest_sha256": digest((ROOT / "sources.lock.json").read_bytes()),
               "model_loaded": False, "qualification": "A successful build is not ABI, quality, latency or board acceptance"}
    (build / "BUILD-RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
