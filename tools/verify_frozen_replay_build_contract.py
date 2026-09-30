#!/usr/bin/env python3
"""Allow only the reviewed trace-slice target addition to historical CMake.

This is deliberately a byte-exact contract, not a permissive CMake parser.
Every pre-existing byte, including flags, library links and conditions, must
remain unchanged. New observation targets require a separately reviewed rule.
"""
from __future__ import annotations

import argparse
import subprocess


# Exact insertion sites and additions from the observation-only history tool.
# Do not strip arbitrary lines containing a target name: those can also change
# global compiler options or dependencies of the frozen inference targets.
INSERTIONS = (
    (
        b"  target_link_libraries(kws_decoder_path_replay PRIVATE KwsPipeline::core kws_tool_io kws_trace_io)\n\n",
        b"  add_executable(kws_trace_slice tools/kws_trace_slice.c)\n"
        b"  target_link_libraries(kws_trace_slice PRIVATE kws_trace_io)\n\n",
    ),
    (
        b"    target_compile_options(kws_decoder_path_replay PRIVATE ${KWS_STRICT_OPTIONS})\n",
        b"    target_compile_options(kws_trace_slice PRIVATE ${KWS_STRICT_OPTIONS})\n",
    ),
    (
        b"    target_compile_options(kws_trace_slice PRIVATE ${KWS_STRICT_OPTIONS})\n  endif()\n",
        b"\n  if(KWS_BUILD_TESTS)\n"
        b"    add_test(NAME kws_trace_slice_self_test COMMAND kws_trace_slice --self-test)\n"
        b"  endif()\n",
    ),
)


def verify_contract(historical: bytes, current: bytes) -> str:
    if historical == current:
        return "unchanged"
    if b"kws_trace_slice" in historical:
        raise ValueError("historical CMake already contains trace-slice; no additional drift allowed")
    expected = historical
    for anchor, addition in INSERTIONS:
        if expected.count(anchor) != 1:
            raise ValueError("historical CMake insertion anchor is missing or ambiguous")
        expected = expected.replace(anchor, anchor + addition, 1)
    if current != expected:
        raise ValueError("historical CMake drift exceeds exact observation-only trace-slice additions")
    return "exact-trace-slice-target-addition"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-head", required=True)
    parser.add_argument("--current-head", default="HEAD")
    args = parser.parse_args()
    def read(ref: str) -> bytes:
        # Resolve a commit first so user input cannot turn into a git option.
        sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        ).decode("ascii").strip()
        return subprocess.check_output(["git", "show", sha + ":CMakeLists.txt"])
    try:
        result = verify_contract(read(args.historical_head), read(args.current_head))
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"frozen replay build contract failed: {exc}\n")
    print(f"frozen replay build contract: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
