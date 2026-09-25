#!/usr/bin/env python3
"""Assert the parameter contract, the generated C header and the tools agree.

configs/parameter-contract.json is the single source of truth for every tunable
parameter.  This test re-derives every bound from that file and checks that:

  1. the header renders deterministically and --check accepts it,
  2. every declared parameter exposes its whole DEFAULT/MIN/MAX/VALID set,
  3. the bounds parsed back out of the header equal the contract values,
  4. tools/compile_keywords.py rejects values outside the contract ranges,
  5. an inconsistent contract fails generation instead of shipping.

It writes only into a temporary directory, so the tracked worktree stays clean.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "parameter-contract.json"
TOOLS = ROOT / "tools"

RUNTIME = (
    "min_speech_dbfs",
    "token_boost",
    "state_retention",
    "refractory_ms",
    "external_vad_threshold",
)
KEYWORD_PACK = ("threshold", "min_trailing_blanks", "priority", "grace_frames")
TABLES = (("runtime", RUNTIME), ("keyword_pack", KEYWORD_PACK))


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def c_float(text: str) -> float:
    return float(text.rstrip("f"))


def c_int(text: str) -> int:
    return int(text.rstrip("u"))


def check_macro_surface(contract: dict, rendered: str, prefix: str) -> None:
    for table, names in TABLES:
        for name in names:
            entry = contract[table][name]
            macro = prefix + name.upper()
            assert re.search(rf"^#define {macro}_MIN_EXCLUSIVE [01]$", rendered, re.M), macro
            assert re.search(rf"^#define {macro}_MAX_EXCLUSIVE [01]$", rendered, re.M), macro
            assert re.search(rf"^#define {macro}_VALID\(", rendered, re.M), macro
            if entry.get("default") is not None:
                assert re.search(rf"^#define {macro}_DEFAULT \(", rendered, re.M), macro
            if entry.get("min") is not None:
                assert re.search(rf"^#define {macro}_MIN \(", rendered, re.M), macro
            if entry.get("max") is not None:
                assert re.search(rf"^#define {macro}_MAX \(", rendered, re.M), macro


def check_bounds_round_trip(contract: dict, rendered: str, prefix: str) -> None:
    for table, names in TABLES:
        for name in names:
            entry = contract[table][name]
            macro = prefix + name.upper()
            cast = c_float if entry["type"] == "float" else c_int
            if entry.get("default") is not None:
                found = re.search(rf"^#define {macro}_DEFAULT \(([^)]*)\)$", rendered, re.M)
                assert found, macro
                assert cast(found.group(1)) == entry["default"], macro
            if entry.get("min") is not None:
                found = re.search(rf"^#define {macro}_MIN \(([^)]*)\)$", rendered, re.M)
                assert found, macro
                assert cast(found.group(1)) == entry["min"], macro
            if entry.get("max") is not None:
                found = re.search(rf"^#define {macro}_MAX \(([^)]*)\)$", rendered, re.M)
                assert found, macro
                assert cast(found.group(1)) == entry["max"], macro
            found = re.search(rf"^#define {macro}_MIN_EXCLUSIVE ([01])$", rendered, re.M)
            assert found, macro
            assert (found.group(1) == "1") == bool(entry.get("min_exclusive", False)), macro
            found = re.search(rf"^#define {macro}_MAX_EXCLUSIVE ([01])$", rendered, re.M)
            assert found, macro
            assert (found.group(1) == "1") == bool(entry.get("max_exclusive", False)), macro


def check_rejects(compiler, contract: dict, token_map: dict, scratch: pathlib.Path) -> None:
    pack = contract["keyword_pack"]
    over_priority = int(pack["priority"]["max"]) + 1
    over_blanks = int(pack["min_trailing_blanks"]["max"]) + 1
    over_grace = int(pack["grace_frames"]["max"]) + 1
    cases = {
        "priority": f"1\t小窝\t0.55\txiao3 wo1\t0\t{over_priority}\timmediate\t0\n",
        "min_trailing_blanks": (
            f"1\t小窝\t0.55\txiao3 wo1\t{over_blanks}\t0\timmediate\t0\n"
        ),
        "grace_frames": f"1\t小窝\t0.55\txiao3 wo1\t0\t0\tgrace\t{over_grace}\n",
    }
    for label, row in cases.items():
        path = scratch / f"over-{label}.tsv"
        path.write_text(row, encoding="utf-8")
        try:
            compiler.parse_keywords(path, token_map, contract)
        except ValueError as exc:
            assert "must be <=" in str(exc), (label, exc)
        else:
            raise AssertionError(f"out-of-range {label} must be rejected")

    # A record sitting exactly on the contract ceilings must still compile, so the
    # guard is a range check and not a blanket rejection.
    ok = scratch / "ceiling.tsv"
    ok.write_text(
        f"1\t小窝\t0.55\txiao3 wo1\t{pack['min_trailing_blanks']['max']}\t"
        f"{pack['priority']['max']}\tgrace\t{pack['grace_frames']['max']}\n",
        encoding="utf-8",
    )
    compiled = compiler.parse_keywords(ok, token_map, contract)
    assert compiled[0]["priority"] == pack["priority"]["max"]
    assert compiled[0]["grace_frames"] == pack["grace_frames"]["max"]


def check_broken_contracts(generator, contract: dict, scratch: pathlib.Path) -> None:
    drifted_default = json.loads(json.dumps(contract))
    drifted_default["runtime"]["state_retention"]["default"] = 1.5
    path = scratch / "drifted-default.json"
    path.write_text(json.dumps(drifted_default), encoding="utf-8")
    try:
        generator.render_header(*generator.load_contract(path))
    except generator.ContractError as exc:
        assert "violates its own upper bound" in str(exc), exc
    else:
        raise AssertionError("a default outside its own range must be rejected")

    broken_invariant = json.loads(json.dumps(contract))
    broken_invariant["algorithm_constants"]["KWS_MEL_HIGH_HZ"]["default"] = 9000.0
    path = scratch / "broken-invariant.json"
    path.write_text(json.dumps(broken_invariant), encoding="utf-8")
    try:
        generator.render_header(*generator.load_contract(path))
    except generator.ContractError as exc:
        assert "invariant violated" in str(exc), exc
    else:
        raise AssertionError("a broken L1 invariant must be rejected")


def within(value: float, entry: dict) -> bool:
    low = entry.get("min")
    high = entry.get("max")
    if low is not None and (
        value < low or (entry.get("min_exclusive") and value == low)
    ):
        return False
    if high is not None and (
        value > high or (entry.get("max_exclusive") and value == high)
    ):
        return False
    return True


def check_shipping_contract(contract: dict, digest: str) -> None:
    """The shipping contract must pin the contract and re-derive the same list."""
    shipping = json.loads(
        (ROOT / "configs" / "shipping.xiaowo.json").read_text(encoding="utf-8")
    )
    pinned = shipping["parameter_contract"]
    assert pinned["contract_id"] == contract["contract_id"]
    assert pinned["path"] == "configs/parameter-contract.json"
    assert pinned["sha256"] == digest, "shipping contract pins a stale parameter contract"

    calibration = shipping["threshold_calibration"]
    assert calibration["parameter_contract_sha256"] == digest

    # The recalibration list must equal the contract's own invalidation flags,
    # plus the L0 model release tag which the contract cannot see.
    flagged = sorted(
        f"runtime.{name}"
        for name in RUNTIME
        if contract["runtime"][name].get("invalidates_thresholds")
    )
    assert sorted(calibration["invalidated_by"]) == sorted(flagged + ["model.release_tag"])

    # Every calibrated threshold must still match the shipping wake word list.
    words = {word["text"]: word["threshold"] for word in shipping["shipping_wake_words"]}
    assert calibration["thresholds"], "calibration must record the thresholds"
    for entry in calibration["thresholds"]:
        assert entry["text"] in words, entry
        assert entry["value"] == words[entry["text"]], entry

    # Every L2 field must be present and inside the contract range.
    assert set(shipping["runtime"]) == set(RUNTIME), sorted(shipping["runtime"])
    for name in RUNTIME:
        assert within(shipping["runtime"][name], contract["runtime"][name]), name



def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    generator = load_module(
        "gen_parameter_limits", TOOLS / "gen_parameter_limits.py"
    )
    sys.path.insert(0, str(TOOLS))
    compiler = load_module("compile_keywords", TOOLS / "compile_keywords.py")

    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td)
        outdir = scratch / "generated"
        assert generator.main([str(CONTRACT_PATH), str(outdir)]) == 0
        rendered = (outdir / generator.HEADER_NAME).read_text(encoding="utf-8")

        # Regeneration must be byte-identical, otherwise --check is meaningless.
        with contextlib.redirect_stdout(io.StringIO()):
            assert generator.main([str(CONTRACT_PATH), str(outdir), "--check"]) == 0
        assert generator.render_header(*generator.load_contract(CONTRACT_PATH)) == rendered

        digest = generator.load_contract(CONTRACT_PATH)[1]
        assert f'#define KWS_PARAMETER_CONTRACT_SHA256 "{digest}"' in rendered

        prefix = generator.RUNTIME_PREFIX
        check_macro_surface(contract, rendered, prefix)
        check_bounds_round_trip(contract, rendered, prefix)

        for name in generator.REQUIRED_ALGORITHM_CONSTANTS:
            found = re.search(rf"^#define {name} \(([^)]*)\)$", rendered, re.M)
            assert found, name
            entry = contract["algorithm_constants"][name]
            if entry["type"] in generator.INTEGER_TYPES:
                actual = int(found.group(1).rstrip("u"))
            else:
                actual = c_float(found.group(1))
            assert actual == entry["default"], name

        stale = scratch / "stale"
        stale.mkdir()
        (stale / generator.HEADER_NAME).write_text("stale\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            assert generator.main([str(CONTRACT_PATH), str(stale), "--check"]) == 1

        token_map = compiler.load_tokens(ROOT / "keywords" / "tokens.example.txt")
        check_rejects(compiler, contract, token_map, scratch)
        check_broken_contracts(generator, contract, scratch)
        check_shipping_contract(contract, digest)

    print("test_parameter_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
