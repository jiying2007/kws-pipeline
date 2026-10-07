#!/usr/bin/env python3
"""Build or verify an offline keyword identity; never authorizes deployment.

All input paths are canonical, root-relative paths. Output is JSON on stdout.
See docs/KEYWORD_SET_IDENTITY.md for the v2 identity and authority boundaries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile

import compile_keywords as compiler
import gen_parameter_limits as parameters
from kws_vocab import MAX_VOCAB_SIZE, load_tokens

SCHEMA_VERSION = 2
POLICY = "kws-keyword-set-identity-v2"
SOURCES = ("tokens", "keywords", "parameter_contract")
ENTRY_FIELDS = {
    "layer", "type", "default", "min", "max", "min_exclusive", "max_exclusive",
    "unit", "effective", "invalidates_thresholds", "summary",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def exact_keys(value: object, required: set[str], label: str,
               optional: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ValueError(f"{label}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def load_json(raw: bytes) -> dict:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value

    def constant(text):
        raise ValueError(f"invalid JSON constant: {text}")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_float=number, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def relative_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("input path must be a canonical root-relative string")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
        raise ValueError(f"noncanonical root-relative input path: {name}")
    return path.parts


def open_directory(name: str, *, parent: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, dir_fd=parent)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError("root and input parents must be directories")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def open_root(root: Path):
    # Fail closed on platforms without descriptor-relative no-follow opening.
    if (os.open not in os.supports_dir_fd or
            any(not hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"))):
        raise RuntimeError("keyword identity requires POSIX no-follow descriptor-relative file access")
    # Traverse the explicit root itself with pinned descriptors too. Resolving
    # symlinks then opening by pathname would reintroduce a root/ancestor race.
    path = Path(os.path.abspath(root))
    fd = open_directory(path.anchor)
    try:
        for part in path.parts[1:]:
            child = open_directory(part, parent=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_input(root_fd: int, name: str) -> bytes:
    parts = relative_parts(name)
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = open_directory(part, parent=parent)
            os.close(parent)
            parent = child
        # O_NONBLOCK avoids hanging if an attacker swaps in a FIFO; fstat
        # rejects every nonregular inode before reading from the pinned handle.
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        fd = os.open(parts[-1], flags, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"input must be a regular file inside root: {name}")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def finite_number(value: object, label: str, ctype: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number (not boolean/string)")
    if ctype in parameters.INTEGER_TYPES:
        if type(value) is not int or not 0 <= value <= parameters.INTEGER_MAX[ctype]:
            raise ValueError(f"{label}: expected {ctype} integer")
    else:
        try:
            packed = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"{label}: number exceeds float32") from exc
        if not math.isfinite(packed):
            raise ValueError(f"{label}: number exceeds float32")


def validate_parameter_contract(contract: dict) -> None:
    """Strict input boundary around the existing parameter/header validator."""
    exact_keys(contract, {"schema_version", "contract_id", "layers", "runtime",
                         "keyword_pack", "policy_defaults", "algorithm_constants"},
               "parameter contract", {"note"})
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise ValueError("unsupported parameter contract schema")
    if contract["contract_id"] != "kws-parameter-contract-v1":
        raise ValueError("unsupported parameter contract identity")
    if "note" in contract and not isinstance(contract["note"], str):
        raise ValueError("parameter contract note must be text")
    exact_keys(contract["layers"], {"L0", "L1", "L2", "L3"}, "layers")
    if any(not isinstance(v, str) or not v for v in contract["layers"].values()):
        raise ValueError("layers must contain nonempty text")
    tables = (("runtime", parameters.REQUIRED_RUNTIME, "L2"),
              ("keyword_pack", parameters.REQUIRED_KEYWORD_PACK, "L3"),
              ("algorithm_constants", parameters.REQUIRED_ALGORITHM_CONSTANTS, "L1"))
    for table, names, layer in tables:
        exact_keys(contract[table], set(names), table)
        for name in names:
            entry = contract[table][name]
            label = f"{table}.{name}"
            required = {"layer", "type", "default", "unit", "summary"}
            if table != "algorithm_constants":
                required |= {"min", "max", "min_exclusive", "max_exclusive", "effective"}
            exact_keys(entry, required, label, ENTRY_FIELDS - required)
            ctype = ("uint32" if name == "refractory_ms" else "uint8"
                     if name in ("min_trailing_blanks", "priority", "grace_frames",
                                 "KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES") else "float")
            if entry["type"] != ctype or entry["layer"] != layer:
                raise ValueError(f"{label}: unexpected type/layer")
            for key in ("unit", "summary"):
                if not isinstance(entry[key], str):
                    raise ValueError(f"{label}.{key}: expected text")
            for key in ("min_exclusive", "max_exclusive", "effective", "invalidates_thresholds"):
                if key in entry and type(entry[key]) is not bool:
                    raise ValueError(f"{label}.{key}: expected boolean")
            for key in ("default", "min", "max"):
                if entry.get(key) is not None:
                    finite_number(entry[key], f"{label}.{key}", ctype)
            if entry["default"] is None and name != "threshold":
                raise ValueError(f"{label}: default is required")
            if table == "keyword_pack":
                if entry["min"] is None or entry["max"] is None or not entry["effective"]:
                    raise ValueError(f"{label}: effective bounded field required")
                # bounded_int deliberately has inclusive comparisons. Do not claim
                # to support an exclusive integer policy the canonical parser ignores.
                if ctype != "float" and (entry["min_exclusive"] or entry["max_exclusive"]):
                    raise ValueError(f"{label}: exclusive integer bounds unsupported")
            parameters.inspect_entry(label, entry)
    exact_keys(contract["policy_defaults"], {"longest", "grace"}, "policy_defaults")
    for policy, field in (("longest", "min_trailing_blanks"), ("grace", "grace_frames")):
        entry = contract["policy_defaults"][policy]
        exact_keys(entry, {field}, f"policy_defaults.{policy}")
        finite_number(entry[field], f"policy_defaults.{policy}.{field}", "uint8")
        compiler.bounded_int(str(entry[field]), field, contract["keyword_pack"][field])
        if entry[field] == 0:
            raise ValueError(f"policy_defaults.{policy}.{field}: must be positive")
    # Reuse the existing authority for defaults, wire limits and L1 invariants.
    parameters.render_header(contract, "0" * 64)


def keyword_policy(contract: dict) -> dict:
    result = {}
    for name in parameters.REQUIRED_KEYWORD_PACK:
        entry = contract["keyword_pack"][name]
        row = {key: entry[key] for key in ("type", "default", "min", "max",
                                          "min_exclusive", "max_exclusive")}
        if row["type"] == "float":
            for key in ("default", "min", "max"):
                if row[key] is not None:
                    row[key] = float(row[key]) if row[key] != 0 else 0.0
        result[name] = row
    return {"keyword_pack": result, "policy_defaults": contract["policy_defaults"]}


def build_keyword_set_identity(*, root: Path, tokens_path: str, keywords_path: str,
                               parameter_contract_path: str) -> dict:
    """Snapshot explicit inputs and bind all current parsed keyword semantics."""
    names = dict(zip(SOURCES, (tokens_path, keywords_path, parameter_contract_path)))
    with open_root(root) as root_fd:
        return _build_from_root_fd(root_fd, names)


def _build_from_root_fd(root_fd: int, names: dict[str, str]) -> dict:
    raw = {key: read_input(root_fd, name) for key, name in names.items()}
    contract = load_json(raw["parameter_contract"])
    validate_parameter_contract(contract)
    # Only this additional production-safety restriction sits ahead of the
    # canonical parser. It cannot call text_to_pinyin or import pypinyin.
    for number, line in enumerate(raw["keywords"].decode("utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if not 4 <= len(cols) <= 8 or not cols[3].strip():
            raise ValueError(f"keywords:{number}: require 4..8 columns and explicit tokens")
        if not cols[1].strip():
            raise ValueError(f"keywords:{number}: keyword text is required")
    # Parse immutable snapshots, so the raw digests describe the exact bytes the
    # canonical parsers consumed, even if the original files are later changed.
    with tempfile.TemporaryDirectory(prefix="kws-keyword-identity-") as temporary:
        paths = {key: Path(temporary) / key for key in SOURCES}
        for key, path in paths.items():
            path.write_bytes(raw[key])
        token_map = load_tokens(paths["tokens"])
        items = compiler.parse_keywords(paths["keywords"], token_map,
                                        compiler.load_contract(paths["parameter_contract"]))
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "tokens": [{"id": idx, "token": token}
                   for token, idx in sorted(token_map.items(), key=lambda row: row[1])],
        # Preserve compiler row order: it can affect compiled packs/tie handling.
        "keywords": items,
        "parameter_policy": keyword_policy(contract),
        "compiler_limits": {"max_keywords": compiler.MAX_KEYWORDS,
                            "max_tokens_per_keyword": compiler.MAX_TOKENS_PER_KEYWORD,
                            "max_vocab_size": MAX_VOCAB_SIZE,
                            "pack_version": compiler.PACK_VERSION,
                            "prefix_policies": compiler.PREFIX_POLICIES},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "semantic_sha256": hashlib.sha256(canonical_bytes(semantic)).hexdigest(),
        "semantics": semantic,
        "sources": {key: {"path": names[key], "bytes": len(raw[key]),
                          "sha256": hashlib.sha256(raw[key]).hexdigest()} for key in SOURCES},
    }


def verify_keyword_set_contract(contract_path: str, *, root: Path) -> dict:
    """Require exact raw input identities AND recomputed v2 semantic identity."""
    with open_root(root) as root_fd:
        contract = load_json(read_input(root_fd, contract_path))
        exact_keys(contract, {"schema_version", "policy", "sources", "semantics", "semantic_sha256"},
                   "keyword identity contract")
        if type(contract["schema_version"]) is not int or contract["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported keyword identity schema")
        if contract["policy"] != POLICY:
            raise ValueError("unsupported keyword identity policy")
        if not isinstance(contract["semantic_sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", contract["semantic_sha256"]):
            raise ValueError("invalid semantic_sha256")
        exact_keys(contract["sources"], set(SOURCES), "sources")
        for key in SOURCES:
            source = contract["sources"][key]
            exact_keys(source, {"path", "bytes", "sha256"}, f"sources.{key}")
            if type(source["bytes"]) is not int or source["bytes"] < 0:
                raise ValueError(f"sources.{key}.bytes: expected nonnegative integer")
            if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
                raise ValueError(f"sources.{key}.sha256: invalid digest")
        actual = _build_from_root_fd(root_fd, {key: contract["sources"][key]["path"] for key in SOURCES})
        # Canonical bytes, rather than Python equality: True must never equal 1,
        # unknown fields must never be ignored, and no unverified metadata survives.
        if canonical_bytes(contract) != canonical_bytes(actual):
            raise ValueError("keyword identity contract mismatch (raw bytes or semantic content)")
        return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="emit a v2 identity contract to stdout")
    verify = commands.add_parser("verify", help="verify exact bytes and all semantic fields")
    for command in (build, verify):
        command.add_argument("--root", required=True, type=Path)
    build.add_argument("--tokens", required=True)
    build.add_argument("--keywords", required=True)
    build.add_argument("--parameter-contract", required=True)
    verify.add_argument("--contract", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_keyword_set_identity(root=args.root, tokens_path=args.tokens,
                                               keywords_path=args.keywords,
                                               parameter_contract_path=args.parameter_contract)
        else:
            result = verify_keyword_set_contract(args.contract, root=args.root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, OverflowError) as exc:
        print(f"keyword identity: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
