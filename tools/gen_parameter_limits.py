#!/usr/bin/env python3
"""Generate the C parameter-limit header from configs/parameter-contract.json.

configs/parameter-contract.json is the single source of truth for every tunable
parameter, its validation range and the L1 algorithm constants.  This generator
turns that contract into a private C header so src/*.c cannot drift from the
documented ranges; the python tools read the same JSON directly.  ``--check``
fails when the generated header is stale, which is what CI asserts.

Examples:
  python3 tools/gen_parameter_limits.py configs/parameter-contract.json build/generated
  python3 tools/gen_parameter_limits.py configs/parameter-contract.json build/generated --check
  python3 tools/gen_parameter_limits.py configs/parameter-contract.json --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HEADER_NAME = "kws_parameter_limits.h"
RUNTIME_PREFIX = "KWS_PARAM_"
INTEGER_TYPES = ("uint8", "uint32")
FLOAT_TYPES = ("float",)
INTEGER_MAX = {"uint8": 255, "uint32": 4294967295}
COMMENT_WIDTH = 78
CONTINUATION_COLUMN = 78

REQUIRED_RUNTIME = (
    "min_speech_dbfs",
    "token_boost",
    "state_retention",
    "refractory_ms",
    "external_vad_threshold",
)

REQUIRED_KEYWORD_PACK = (
    "threshold",
    "min_trailing_blanks",
    "priority",
    "grace_frames",
)

REQUIRED_ALGORITHM_CONSTANTS = (
    "KWS_SILENCE_RETENTION_LOG",
    "KWS_MIN_PATH_RETENTION_LOG",
    "KWS_ROOT_START_LOGIT_MARGIN",
    "KWS_FUZZY_CHILD_RETENTION_COST_LOG",
    "KWS_PCEN_SMOOTHING",
    "KWS_PCEN_ALPHA",
    "KWS_PCEN_DELTA",
    "KWS_PCEN_EPSILON",
    "KWS_LOGMEL_COMPRESSION",
    "KWS_FEATURE_NORMALIZATION",
    "KWS_MEL_LOW_HZ",
    "KWS_MEL_HIGH_HZ",
)

# Domain invariants that must hold for any accepted contract.  C cannot evaluate
# floating-point constant expressions in a preprocessor condition or inside
# _Static_assert, so these are enforced here -- at configure time, before any
# object file exists -- rather than in the generated header.  Keyed by a
# human-readable expression, mapped to (reason, predicate).
STRUCTURAL_INVARIANTS = {
    "KWS_SILENCE_RETENTION_LOG < 0": (
        "silence retention must decay a live prefix",
        lambda v: v["KWS_SILENCE_RETENTION_LOG"] < 0.0,
    ),
    "KWS_MIN_PATH_RETENTION_LOG < KWS_SILENCE_RETENTION_LOG": (
        "the abandonment budget must exceed a single silent frame",
        lambda v: v["KWS_MIN_PATH_RETENTION_LOG"]
        < v["KWS_SILENCE_RETENTION_LOG"],
    ),
    "2 * KWS_FUZZY_CHILD_RETENTION_COST_LOG <= KWS_MIN_PATH_RETENTION_LOG": (
        "two fuzzy child advances must exhaust the abandonment budget",
        lambda v: 2.0 * v["KWS_FUZZY_CHILD_RETENTION_COST_LOG"]
        <= v["KWS_MIN_PATH_RETENTION_LOG"],
    ),
    "KWS_ROOT_START_LOGIT_MARGIN > 0": (
        "the root start margin must be positive",
        lambda v: v["KWS_ROOT_START_LOGIT_MARGIN"] > 0.0,
    ),
    "0 < KWS_PCEN_SMOOTHING < 1": (
        "pcen smoothing must be a ratio",
        lambda v: 0.0 < v["KWS_PCEN_SMOOTHING"] < 1.0,
    ),
    "KWS_PCEN_ALPHA > 0": (
        "pcen alpha must be positive",
        lambda v: v["KWS_PCEN_ALPHA"] > 0.0,
    ),
    "KWS_PCEN_DELTA > 0": (
        "pcen delta must be positive",
        lambda v: v["KWS_PCEN_DELTA"] > 0.0,
    ),
    "KWS_PCEN_EPSILON > 0": (
        "pcen epsilon must be positive",
        lambda v: v["KWS_PCEN_EPSILON"] > 0.0,
    ),
    "KWS_LOGMEL_COMPRESSION > 0": (
        "logmel compression gain must be positive",
        lambda v: v["KWS_LOGMEL_COMPRESSION"] > 0.0,
    ),
    "KWS_FEATURE_NORMALIZATION > 0": (
        "feature normalization gain must be positive",
        lambda v: v["KWS_FEATURE_NORMALIZATION"] > 0.0,
    ),
    "KWS_MEL_LOW_HZ >= 0": (
        "the mel low edge must be non-negative",
        lambda v: v["KWS_MEL_LOW_HZ"] >= 0.0,
    ),
    "KWS_MEL_LOW_HZ < KWS_MEL_HIGH_HZ": (
        "the mel filterbank edges must be ordered",
        lambda v: v["KWS_MEL_LOW_HZ"] < v["KWS_MEL_HIGH_HZ"],
    ),
    "KWS_MEL_HIGH_HZ < 8000": (
        "the mel high edge must stay below the 16 kHz Nyquist limit",
        lambda v: v["KWS_MEL_HIGH_HZ"] < 8000.0,
    ),
}


class ContractError(RuntimeError):
    """The parameter contract is missing, malformed or self-inconsistent."""


def load_contract(path: pathlib.Path) -> tuple[dict, str]:
    if not path.is_file():
        raise ContractError(f"parameter contract not found: {path}")
    raw = path.read_bytes()
    try:
        contract = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ContractError(f"{path}: top level must be a JSON object")
    return contract, hashlib.sha256(raw).hexdigest()


def require_table(contract: dict, key: str, required: tuple[str, ...]) -> dict:
    table = contract.get(key)
    if not isinstance(table, dict):
        raise ContractError(f"contract is missing the '{key}' table")
    missing = [name for name in required if name not in table]
    if missing:
        raise ContractError(
            f"contract '{key}' table is missing required entries: "
            + ", ".join(missing)
        )
    return table


def exclusive_bound(entry: dict, key: str, label: str) -> bool:
    """Read an exclusive-bound flag that is already a boolean.

    bool("false") is True, so coercing here turns an inclusive bound into
    an exclusive one in the generated C header. The contract is a file.
    """
    value = entry.get(key, False)
    if not isinstance(value, bool):
        raise ContractError(f"{label}: {key} must be a boolean")
    return value


def inspect_entry(label: str, entry: object) -> dict:
    """Validate one parameter entry and return its resolved bounds."""
    if not isinstance(entry, dict):
        raise ContractError(f"{label}: entry must be a JSON object")
    ctype = entry.get("type")
    if ctype not in FLOAT_TYPES + INTEGER_TYPES:
        raise ContractError(f"{label}: unsupported type {ctype!r}")
    low = entry.get("min")
    high = entry.get("max")
    low_exclusive = exclusive_bound(entry, "min_exclusive", label)
    high_exclusive = exclusive_bound(entry, "max_exclusive", label)
    if low is not None and high is not None and float(low) >= float(high):
        raise ContractError(
            f"{label}: min {low} must be strictly below max {high}"
        )
    default = entry.get("default")
    if default is not None:
        if ctype in INTEGER_TYPES and int(default) != float(default):
            raise ContractError(f"{label}: default {default!r} is not an integer")
        value = float(default)
        if low is not None and (
            value < float(low) or (low_exclusive and value == float(low))
        ):
            raise ContractError(
                f"{label}: default {default} violates its own lower bound {low}"
            )
        if high is not None and (
            value > float(high) or (high_exclusive and value == float(high))
        ):
            raise ContractError(
                f"{label}: default {default} violates its own upper bound {high}"
            )
    return {
        "type": ctype,
        "min": None if low is None else float(low),
        "max": None if high is None else float(high),
        "min_exclusive": low_exclusive,
        "max_exclusive": high_exclusive,
        "default": None if default is None else float(default),
    }


def c_literal(value: float, ctype: str) -> str:
    if ctype in INTEGER_TYPES:
        return f"{int(value)}u"
    return f"{float(value)!r}f"


def wrap_words(text: str, width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def banner(title: str) -> list[str]:
    rule = "-" * (COMMENT_WIDTH - 6)
    return [f"/* {rule}", f" * {title}", f" * {rule} */"]


def render_entry(
    name: str, entry: object, macro: str, with_bounds: bool
) -> list[str]:
    info = inspect_entry(name, entry)
    ctype = info["type"]
    low = info["min"]
    high = info["max"]
    low_exclusive = info["min_exclusive"]
    high_exclusive = info["max_exclusive"]
    default = info["default"]

    flags = [str(entry.get("layer", "?"))]
    if entry.get("unit"):
        flags.append(str(entry["unit"]))
    if entry.get("effective") is not None:
        flags.append("effective=" + ("yes" if entry["effective"] else "NO"))
    if entry.get("invalidates_thresholds") is not None:
        flags.append(
            "invalidates_thresholds="
            + ("yes" if entry["invalidates_thresholds"] else "no")
        )

    lines = ["/* " + " | ".join(flags)]
    for text_line in wrap_words(entry.get("summary", ""), COMMENT_WIDTH - 3):
        lines.append(f" * {text_line}")
    lines.append(" */")

    if not with_bounds:
        if default is None:
            raise ContractError(f"{name}: an algorithm constant needs a default")
        lines.append(f"#define {macro} ({c_literal(default, ctype)})")
        lines.append("")
        return lines

    if default is not None:
        lines.append(f"#define {macro}_DEFAULT ({c_literal(default, ctype)})")
    if low is not None:
        lines.append(f"#define {macro}_MIN ({c_literal(low, ctype)})")
    if high is not None:
        lines.append(f"#define {macro}_MAX ({c_literal(high, ctype)})")
    lines.append(f"#define {macro}_MIN_EXCLUSIVE {1 if low_exclusive else 0}")
    lines.append(f"#define {macro}_MAX_EXCLUSIVE {1 if high_exclusive else 0}")

    conditions = []
    if ctype in FLOAT_TYPES:
        conditions.append("isfinite(v)")
    # Unsigned operands are widened explicitly so a uint8_t field does not trip
    # -Wsign-compare after integer promotion, and so a negative input can never
    # slip past an upper bound.
    operand = "(uint32_t)(v)" if ctype in INTEGER_TYPES else "(v)"
    # Skip a bound that every value of the declared type already satisfies;
    # emitting it would make each caller fail -Wtype-limits under -Werror.
    floor_ = 0 if ctype in INTEGER_TYPES else None
    ceiling = INTEGER_MAX.get(ctype)
    if low is not None and not (
        floor_ is not None and not low_exclusive and float(low) == floor_
    ):
        operator = ">" if low_exclusive else ">="
        conditions.append(f"{operand} {operator} ({c_literal(low, ctype)})")
    if high is not None and not (
        ceiling is not None and not high_exclusive and float(high) == ceiling
    ):
        operator = "<" if high_exclusive else "<="
        conditions.append(f"{operand} {operator} ({c_literal(high, ctype)})")
    signature = f"#define {macro}_VALID(v)"
    if conditions:
        lines.append(signature.ljust(CONTINUATION_COLUMN) + " \\")
        lines.append("  (" + " && ".join(conditions) + ")")
    else:
        lines.append("/* every value of the declared type is in range */")
        lines.append(f"#define {macro}_VALID(v) ((void)(v), 1)")
    lines.append("")
    return lines


def check_invariants(constants: dict) -> None:
    """Evaluate the domain invariants against the contract's L1 constants."""
    values = {}
    for name in REQUIRED_ALGORITHM_CONSTANTS:
        info = inspect_entry(name, constants[name])
        if info["default"] is None:
            raise ContractError(f"{name}: an algorithm constant needs a default")
        values[name] = info["default"]
    for expression, (reason, predicate) in STRUCTURAL_INVARIANTS.items():
        if not predicate(values):
            raise ContractError(f"invariant violated: {expression} ({reason})")


def render_header(contract: dict, digest: str) -> str:
    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ContractError("contract is missing 'contract_id'")
    schema_version = contract.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ContractError("contract is missing a positive integer 'schema_version'")

    runtime = require_table(contract, "runtime", REQUIRED_RUNTIME)
    keyword_pack = require_table(contract, "keyword_pack", REQUIRED_KEYWORD_PACK)
    constants = require_table(
        contract, "algorithm_constants", REQUIRED_ALGORITHM_CONSTANTS
    )

    lines = [
        "/* Generated by tools/gen_parameter_limits.py from configs/parameter-contract.json.",
        " * Do not edit by hand: edit the contract and re-run the generator.",
        f" * contract_id: {contract_id}",
        f" * contract_sha256: {digest}",
        " */",
        "#ifndef KWS_PARAMETER_LIMITS_H",
        "#define KWS_PARAMETER_LIMITS_H",
        "",
        "#include <math.h>",
        "#include <stdint.h>",
        "",
        f"#define KWS_PARAMETER_CONTRACT_ID {json.dumps(contract_id)}",
        f"#define KWS_PARAMETER_CONTRACT_SCHEMA_VERSION {schema_version}u",
        f"#define KWS_PARAMETER_CONTRACT_SHA256 {json.dumps(digest)}",
        "",
    ]

    lines += banner("L1 firmware constants -> algorithm_constants")
    lines.append("")
    for name in REQUIRED_ALGORITHM_CONSTANTS:
        # Algorithm constants already carry the KWS_ prefix in the contract.
        lines += render_entry(name, constants[name], name, with_bounds=False)

    lines += banner("L2 product configuration -> runtime")
    lines.append("")
    for name in REQUIRED_RUNTIME:
        lines += render_entry(
            name, runtime[name], f"{RUNTIME_PREFIX}{name.upper()}", with_bounds=True
        )

    lines += banner("L3 field policy -> keyword_pack")
    lines.append("")
    for name in REQUIRED_KEYWORD_PACK:
        lines += render_entry(
            name, keyword_pack[name], f"{RUNTIME_PREFIX}{name.upper()}", with_bounds=True
        )

    check_invariants(constants)

    lines += banner("Invariants")
    lines.append("")
    lines.append(
        "/* The floating-point invariants below are enforced by "
        "tools/gen_parameter_limits.py"
    )
    lines.append(
        " * at configure time, because C cannot evaluate them in a preprocessor"
    )
    lines.append(" * condition or inside _Static_assert:")
    for expression, (reason, _predicate) in STRUCTURAL_INVARIANTS.items():
        lines.append(f" *   {expression}  -- {reason}")
    lines.append(" */")
    lines.append("")

    for table, names in (
        (runtime, REQUIRED_RUNTIME),
        (keyword_pack, REQUIRED_KEYWORD_PACK),
    ):
        for name in names:
            macro = f"{RUNTIME_PREFIX}{name.upper()}"
            info = inspect_entry(name, table[name])
            if info["type"] not in INTEGER_TYPES:
                continue
            if info["min"] is None or info["max"] is None:
                continue
            lines.append(
                f'_Static_assert({macro}_MIN < {macro}_MAX, "{name}: bounds must be ordered");'
            )
    lines.append("")
    lines.append("#endif /* KWS_PARAMETER_LIMITS_H */")
    lines.append("")
    return "\n".join(lines)


def render_json(contract: dict, digest: str) -> str:
    runtime = require_table(contract, "runtime", REQUIRED_RUNTIME)
    keyword_pack = require_table(contract, "keyword_pack", REQUIRED_KEYWORD_PACK)
    constants = require_table(
        contract, "algorithm_constants", REQUIRED_ALGORITHM_CONSTANTS
    )
    payload = {
        "contract_id": contract.get("contract_id"),
        "contract_sha256": digest,
        "schema_version": contract.get("schema_version"),
        "runtime": {name: inspect_entry(name, runtime[name]) for name in REQUIRED_RUNTIME},
        "keyword_pack": {
            name: inspect_entry(name, keyword_pack[name])
            for name in REQUIRED_KEYWORD_PACK
        },
        "algorithm_constants": {
            name: inspect_entry(name, constants[name])
            for name in REQUIRED_ALGORITHM_CONSTANTS
        },
        "policy_defaults": contract.get("policy_defaults", {}),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate kws_parameter_limits.h from the parameter contract."
    )
    parser.add_argument("contract", type=pathlib.Path, help="configs/parameter-contract.json")
    parser.add_argument(
        "outdir",
        nargs="?",
        type=pathlib.Path,
        help="directory that receives " + HEADER_NAME,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated header on disk is stale",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the resolved table instead of C"
    )
    args = parser.parse_args(argv)

    contract, digest = load_contract(args.contract)

    if args.json:
        sys.stdout.write(render_json(contract, digest))
        return 0

    if args.outdir is None:
        parser.error("outdir is required unless --json is used")

    rendered = render_header(contract, digest)
    target = args.outdir / HEADER_NAME

    if args.check:
        if not target.is_file():
            print(f"error: {target} does not exist", file=sys.stderr)
            return 1
        existing = target.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                f"error: {target} is stale; re-run {sys.argv[0]}",
                file=sys.stderr,
            )
            return 1
        print(f"{HEADER_NAME} matches {args.contract} (sha256 {digest[:16]})")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {target} (contract sha256 {digest[:16]})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
