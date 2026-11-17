#!/usr/bin/env python3
"""Create a pinned source copy with one experimental 3x1 dispatch; never build/run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FRONTEND_SHA256 = "5a476954887b59032ea5d419e97d6fb913695e3ad99f5d09164513a4cbd87a89"
MANIFEST_SHA256 = "b94549aec824a8eddb014e19f01629d8733c35f809d763927b712df845b5677b"
GENERATED_C_SHA256 = "9bf740a692370dfed02f034f87f6f0dfcaac42f3f83152c79fd27088ccc71941"
SPATIAL_ANCHOR = "static int spatial_run(const ef_tensor *x, const ef_tensor *w, const ef_tensor *y,\n"
DISPATCH_ANCHOR = "    /* END FIXED MEL CONV1D DISPATCH */\n"
DISPATCH = """    /* BEGIN EXPERIMENTAL ORDERED 3X1 DISPATCH */
    if (ekws_3x1_matches(x, w, y, n, a, weights, bias, out)) {
        ekws_3x1_ordered(x, y, a, weights, bias, out);
        return EF_OK;
    }
    /* END EXPERIMENTAL ORDERED 3X1 DISPATCH */
"""


def identity(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def specialize(source: bytes, manifest: bytes, helper: bytes) -> tuple[bytes, dict]:
    if identity(source)["sha256"] != FRONTEND_SHA256:
        raise ValueError("frozen frontend SHA256 mismatch")
    if identity(manifest)["sha256"] != MANIFEST_SHA256:
        raise ValueError("frozen export manifest SHA256 mismatch")
    description = json.loads(manifest)
    if description["generated_c_sha256"] != GENERATED_C_SHA256:
        raise ValueError("generated C identity mismatch")
    original = source.decode("utf-8")
    helpers = helper.decode("utf-8")
    if original.count(SPATIAL_ANCHOR) != 1 or original.count(DISPATCH_ANCHOR) != 1:
        raise ValueError("source-copy insertion anchors must occur exactly once")
    begin = "/* BEGIN EXPERIMENTAL ORDERED 3X1 HELPERS */\n"
    end = "/* END EXPERIMENTAL ORDERED 3X1 HELPERS */\n"
    block = begin + helpers.rstrip("\n") + "\n" + end + "\n"
    candidate = original.replace(SPATIAL_ANCHOR, block + SPATIAL_ANCHOR, 1)
    candidate = candidate.replace(DISPATCH_ANCHOR, DISPATCH_ANCHOR + DISPATCH, 1)
    # Removing only the two additions must reproduce the complete frozen file.
    restored = candidate.replace(block, "", 1).replace(DISPATCH, "", 1)
    if restored != original:
        raise ValueError("candidate unexpectedly changed frozen source text")
    output = candidate.encode("utf-8")
    receipt = {
        "schema": "english-ordered-3x1-source-copy-v1",
        "experimental": True,
        "source_only": True,
        "kernel_executed": False,
        "model_executed": False,
        "baseline_unchanged": True,
        "original_source": identity(source),
        "export_manifest": identity(manifest),
        "generated_c_sha256": GENERATED_C_SHA256,
        "helper": identity(helper),
        "specializer": identity(Path(__file__).read_bytes()),
        "candidate_source": identity(output),
        "eligible_embedding_nodes": [6, 13, 19, 26, 32, 39, 45, 52, 58],
        "node62_uses_fallback": True,
        "required_compile_flags": ["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off"],
        "required_fp_evaluation_method_for_dispatch": 0,
        "independent_output_position_block": 4,
        "dispatch_requires_output_plane_multiple_of_four": True,
        "dispatch_requires_output_disjoint_from_all_reads": True,
        "nonfinite_validation_and_generic_fallback_unchanged": True,
        "ssc305_qualification": False,
        "model_promotion_allowed": False,
    }
    return output, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    helper_path = Path(__file__).with_name("ordered_3x1.h")
    sources = [args.source, args.manifest, helper_path, Path(__file__)]
    for path in sources:
        if path.is_symlink() or not path.is_file():
            parser.error(f"required regular source: {path}")
    outputs = [args.output, args.receipt]
    if len({path.resolve() for path in sources + outputs}) != len(sources + outputs):
        parser.error("source and output paths must all be distinct")
    for path in outputs:
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace an existing path: {path}")
    try:
        candidate, receipt = specialize(args.source.read_bytes(), args.manifest.read_bytes(), helper_path.read_bytes())
    except (ValueError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(candidate)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"source_copy_created": True, "candidate": receipt["candidate_source"], "model_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
