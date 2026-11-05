#!/usr/bin/env python3
"""Select and inspect the active research interpreter without importing an ML runtime.

One manifest owns the active Python version and the three dependency roles.
Historical source locks are read-only references, not proof of a new model run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/runtime/research-runtime.json"
ROLES = ("core", "tts", "asr")
CPU_PACKAGES = frozenset(("torch", "torchaudio"))


def load_manifest(path: Path = MANIFEST) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1):
        raise ValueError("unsupported research runtime manifest")
    python = value.get("python", {})
    if python.get("implementation") != "CPython" or python.get("minor") != "3.12":
        raise ValueError("active host tooling requires CPython 3.12")
    if re.fullmatch(r"3\.12\.[0-9]+", str(python.get("version", ""))) is None:
        raise ValueError("manifest must pin an exact Python 3.12 patch")
    if set(value.get("roles", {})) != set(ROLES):
        raise ValueError("manifest requires the core, tts and asr roles")
    for role, spec in value["roles"].items():
        packages = spec.get("packages")
        if not isinstance(packages, dict) or not packages:
            raise ValueError(f"{role}: package pins are missing")
        for name, version in packages.items():
            if (re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is None
                    or re.fullmatch(r"[0-9][a-zA-Z0-9.+-]*", str(version)) is None):
                raise ValueError(f"{role}: invalid package pin")
        if not isinstance(spec.get("reference_locks"), list):
            raise ValueError(f"{role}: reference_locks must be a list")
    return value


def check_interpreter(manifest: dict, observed: tuple[str, str] | None = None) -> dict:
    implementation, version = observed or (platform.python_implementation(), platform.python_version())
    expected = manifest["python"]
    if (implementation, version) != (expected["implementation"], expected["version"]):
        raise ValueError(
            f"active research Python must be {expected['implementation']} {expected['version']}; "
            f"observed {implementation} {version}. Use the shared setup action or this exact interpreter. "
            "Completed historical campaigns retain their original runtime locks."
        )
    return {"implementation": implementation, "version": version}


def check_packages(manifest: dict, role: str) -> dict:
    observed = {}
    for name, expected in sorted(manifest["roles"][role]["packages"].items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f"{role}: required package {name}=={expected} is not installed") from None
        if actual != expected:
            raise ValueError(f"{role}: expected {name}=={expected}; observed {actual}")
        observed[name] = actual
    return observed


def verify_reference_locks(manifest: dict, role: str, root: Path = ROOT) -> dict:
    """Verify historical bytes and reject version conflicts before combining a role.

    This does not install packages or claim that model preparation/inference ran.
    """
    packages: dict[str, str] = {}
    for reference in manifest["roles"][role]["reference_locks"]:
        relative = Path(reference["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("runtime lock path must stay inside the repository")
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError(f"historical runtime lock changed: {relative}")
        value = json.loads(raw)
        for item in value["files"]:
            name, version = item["name"], item["version"]
            previous = packages.setdefault(name, version)
            if previous != version:
                raise ValueError(f"{role}: conflicting {name} pins: {previous} versus {version}")
    for name, version in manifest["roles"][role]["packages"].items():
        if packages and packages.get(name) != version:
            raise ValueError(f"{role}: {name} pin differs from the referenced source locks")
    return packages


def requirement_lines(manifest: dict, role: str, index: str) -> list[str]:
    return [f"{name}=={version}"
            for name, version in sorted(manifest["roles"][role]["packages"].items())
            if (name in CPU_PACKAGES) == (index == "pytorch_cpu")]


def training_lock(manifest: dict) -> str:
    # Existing image receipts bind this small assertion file; OCI digests bind
    # the complete installed filesystem. Keep it derived from the one manifest.
    return (
        "# Generated from configs/runtime/research-runtime.json.\n"
        "# The immutable OCI image digest binds the full dependency tree.\n"
        f"python=={manifest['python']['version']}\n"
        f"torch=={manifest['roles']['core']['packages']['torch']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the configured interpreter patch")
    check = sub.add_parser("check", help="check interpreter; optionally inspect installed role packages")
    check.add_argument("--role", choices=ROLES, default="core")
    check.add_argument("--packages", action="store_true")
    locks = sub.add_parser("check-locks", help="inspect retained dependency locks without executing models")
    locks.add_argument("--role", choices=ROLES)
    requirements = sub.add_parser("requirements", help="emit role package pins; not a transitive wheel lock")
    requirements.add_argument("--role", choices=ROLES, default="core")
    requirements.add_argument("--index", choices=("pypi", "pytorch_cpu"), required=True)
    legacy = sub.add_parser("training-lock", help="render the derived training image assertion file")
    legacy.add_argument("--check", type=Path)
    run = sub.add_parser("run", help="run a Python script with the checked active interpreter")
    run.add_argument("--role", choices=ROLES, default="core")
    run.add_argument("--packages", action="store_true")
    run.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "version":
            print(manifest["python"]["version"])
        elif args.command == "requirements":
            print("\n".join(requirement_lines(manifest, args.role, args.index)))
        elif args.command == "training-lock":
            expected = training_lock(manifest)
            if args.check:
                if args.check.read_text(encoding="utf-8") != expected:
                    raise ValueError("training/requirements.lock drifted from the runtime manifest")
                print("training image assertions match the runtime manifest")
            else:
                print(expected, end="")
        elif args.command == "check-locks":
            roles = [args.role] if args.role else ROLES
            result = {role: len(verify_reference_locks(manifest, role)) for role in roles}
            print(json.dumps({"dependency_inventory_counts": result, "model_execution_verified": False}))
        else:
            observed = check_interpreter(manifest)
            if args.packages:
                observed["packages"] = check_packages(manifest, args.role)
            if args.command == "run":
                script_args = args.script_args
                if script_args[:1] == ["--"]:
                    script_args = script_args[1:]
                if not script_args:
                    raise ValueError("run requires a Python script after --")
                return subprocess.call([sys.executable, *script_args])
            observed.update(role=args.role, model_smoke=manifest["validation"]["model_smoke"])
            print(json.dumps(observed, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"research runtime: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
