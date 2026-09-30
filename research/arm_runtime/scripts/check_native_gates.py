#!/usr/bin/env python3
"""Necessary static release gates for a future unstripped BSP build, not qualification."""
import argparse
import json
import os
import pathlib
import re
import subprocess
from audit_arm_runtime import parse_elf, sha256


def version(text):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", text):
        raise ValueError("Expected numeric dotted version: " + text)
    return tuple(int(x) for x in text.split("."))


def check_versions(required, ceilings):
    failures = []
    for label in required:
        family, number = label.rsplit("_", 1)
        if family in ceilings and version(number) > version(ceilings[family]):
            failures.append(label + " exceeds supplied " + family + " ceiling " + ceilings[family])
    return failures


def defined_espeak_symbols(nm_output):
    return [line for line in nm_output.splitlines()
            if re.search(r"(?:espeak|phonemize)", line, re.I)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--max-glibc", required=True)
    parser.add_argument("--max-glibcxx", required=True)
    parser.add_argument("--max-cxxabi", required=True)
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--nm", default="nm")
    args = parser.parse_args()
    ceilings = {"GLIBC": args.max_glibc, "GLIBCXX": args.max_glibcxx, "CXXABI": args.max_cxxabi}
    try:
        for value in ceilings.values():
            version(value)
        path = args.library
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected a regular unstripped build-output library")
        env = {**os.environ, "LC_ALL": "C"}
        text = subprocess.check_output([args.readelf, "-h", "-A", "-d", "-V", "-S", "--wide", str(path)], text=True, env=env)
        if not re.search(r"\s\.symtab\s+SYMTAB\s", text):
            raise ValueError("Unstripped .symtab required; stripped symbol absence cannot pass the no-eSpeak gate")
        elf = parse_elf(text)
        failures = check_versions(elf["required_version_labels"], ceilings)
        if elf["class"] != "ELF32" or elf["machine"] != "ARM" or "hard-float ABI" not in (elf["flags"] or ""):
            failures.append("Expected ELF32 ARM hard-float build")
        if not elf["required_version_labels"]:
            failures.append("Missing symbol-version requirements; manual ABI review required")
        symbols = subprocess.check_output([args.nm, "--defined-only", str(path)], text=True, env=env)
        matches = defined_espeak_symbols(symbols)
        if matches:
            failures.append("Defined eSpeak/phonemize symbols remain")
        result = {"static_gates_passed": not failures, "sha256": sha256(path),
                  "supplied_version_ceilings": ceilings, "elf": elf,
                  "defined_espeak_or_phonemize_symbols": matches, "failures": failures,
                  "ssc305_compatible": None, "legal_review_completed": False,
                  "note": "Necessary gates only; absence of symbols does not prove absence of all third-party code. Review build inputs/link map/notices and all actual SDK symbol providers, then link/load/execute tests."}
        print(json.dumps(result, indent=2))
        return 0 if not failures else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
