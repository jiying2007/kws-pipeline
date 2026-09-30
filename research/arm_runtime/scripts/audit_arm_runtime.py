#!/usr/bin/env python3
"""Hash and statically inspect the exact reviewed ARM runtime; never load it."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PINNED = {
    "libsherpa-onnx-c-api.so": (5101612, "db57b8ab6136016c75f2becebb81159a6dcbdad875e16d8aef6d5aa5a5209969"),
    "libonnxruntime.so": (20617061, "4636872e9b50f985b7ecaa55a6829dee81bdb96f77010b5bc0a4059e51d872dc"),
}


def validate_lock(lock):
    if not isinstance(lock, dict) or type(lock.get("schema_version")) is not int or lock["schema_version"] != 1:
        raise ValueError("Invalid lock schema")
    if lock.get("sherpa_source_commit") != "11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf":
        raise ValueError("Unexpected source commit")
    entries = lock.get("libraries")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("Lock must name exactly two reviewed libraries")
    names = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
            raise ValueError("Invalid library entry")
        name = entry["filename"]
        if name not in PINNED or name in names:
            raise ValueError("Expected unique reviewed library basenames")
        names.append(name)
        size, digest = PINNED[name]
        if type(entry.get("bytes")) is not int or entry["bytes"] != size or entry.get("sha256") != digest:
            raise ValueError("Lock does not match reviewed bytes/hash")
        for key in ["needed", "required_version_labels"]:
            value = entry.get(key)
            if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value) or len(value) != len(set(value)):
                raise ValueError("Invalid lock field: " + key)
        if entry.get("soname") != name or entry.get("rpath") != "$ORIGIN":
            raise ValueError("Unexpected lock SONAME/RPATH")
    limits = lock.get("limitations")
    if not isinstance(limits, list) or not limits or not all(isinstance(x, str) for x in limits):
        raise ValueError("Missing lock limitations")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_elf(text):
    """Parse readelf output under the fixed C locale, not strings(1) guesses."""
    def field(label):
        match = re.search(r"^\s*" + re.escape(label) + r":\s*(.*?)\s*$", text, re.M)
        return match.group(1) if match else None

    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", text)
    needs_lines = []
    in_needs = False
    for line in text.splitlines():
        if line.startswith("Version needs section"):
            in_needs = True
        elif line.startswith(("Version ", "Attribute Section:")):
            in_needs = False
        elif in_needs:
            needs_lines.append(line)
    versions = sorted(set(re.findall(
        r"Name: ((?:GLIBCXX|GLIBC|CXXABI_ARM|CXXABI|GCC)_[0-9.]+)", "\n".join(needs_lines))))
    return {
        "class": field("Class"), "machine": field("Machine"),
        "data": field("Data"), "flags": field("Flags"),
        "cpu_arch": field("Tag_CPU_arch"), "fp_arch": field("Tag_FP_arch"),
        "simd_arch": field("Tag_Advanced_SIMD_arch"),
        "vfp_args": field("Tag_ABI_VFP_args"),
        "needed": needed, "required_version_labels": versions,
        "soname": re.findall(r"\(SONAME\).*\[(.*?)\]", text),
        "rpath": re.findall(r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]", text),
    }


def verify(directory, lock, readelf="readelf"):
    validate_lock(lock)
    records = []
    for entry in lock["libraries"]:
        path = directory / entry["filename"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected reviewed regular file: " + str(path))
        digest = sha256(path)
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Runtime bytes/hash mismatch: " + str(path))
        # No ldd, dlopen, subprocess execution of a target binary, or model read.
        text = subprocess.check_output(
            [readelf, "-h", "-A", "-d", "-V", str(path)],
            text=True, env={**os.environ, "LC_ALL": "C"})
        info = parse_elf(text)
        required = {"class": "ELF32", "machine": "ARM", "cpu_arch": "v7",
                    "fp_arch": "VFPv3", "simd_arch": "NEONv1",
                    "vfp_args": "VFP registers"}
        for key, value in required.items():
            if info[key] != value:
                raise ValueError("ELF attribute mismatch: " + key)
        if not all(x in (info["flags"] or "") for x in ["Version5 EABI", "hard-float ABI"]):
            raise ValueError("ARM EABI/float mismatch")
        if "little endian" not in (info["data"] or ""):
            raise ValueError("Endianness mismatch")
        if sorted(info["needed"]) != sorted(entry["needed"]):
            raise ValueError("Dependency closure mismatch")
        if info["required_version_labels"] != entry["required_version_labels"]:
            raise ValueError("Symbol-version requirement mismatch")
        if info["soname"] != [entry["soname"]] or info["rpath"] != [entry["rpath"]]:
            raise ValueError("SONAME/RPATH mismatch")
        records.append({"filename": path.name, "bytes": path.stat().st_size,
                        "sha256": digest, "elf": info})
    return {"static_audit_passed": True, "target_execution_tested": False,
            "adapter_cross_build_tested": False, "ssc305_compatible": None,
            "libraries": records, "runtime_selected_bytes": sum(x["bytes"] for x in records),
            "limitations": lock["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=pathlib.Path, required=True)
    parser.add_argument("--lock", type=pathlib.Path, default=ROOT / "report/runtime-arm32.lock.json")
    parser.add_argument("--readelf", default="readelf")
    args = parser.parse_args()
    try:
        result = verify(args.library_dir, json.loads(args.lock.read_text()), args.readelf)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
