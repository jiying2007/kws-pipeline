#!/usr/bin/env python3
"""Validate retained tensor bytes and report the first OBSERVED divergence.

Exact-byte comparison only: numeric deltas explain mismatches, never relax the
gate. A matching same-vendor pair does not resolve the retained AMD/Intel failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct

POLICY = "actual-micro-trainer-numeric-trace-v1"
DTYPES = {"torch.float32": ("f", 4), "torch.float64": ("d", 8), "torch.int64": ("q", 8),
          "torch.int32": ("i", 4), "torch.uint8": ("B", 1), "torch.bool": ("?", 1)}
MAX_BYTES = 32 * 1024 * 1024


def digest(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trace(identity_path: pathlib.Path) -> tuple[dict, dict, pathlib.Path]:
    identity = json.loads(identity_path.read_text())
    root = identity_path.parent / identity_path.stem / "numeric-trace"
    trace_path = root / "trace.json"
    if digest(trace_path) != identity.get("numeric_trace_sha256"):
        raise ValueError("numeric trace receipt hash mismatch")
    trace = json.loads(trace_path.read_text())
    if trace.get("policy") != POLICY or trace.get("schema_version") != 1 or trace.get("completed") is not True:
        raise ValueError("numeric trace is incomplete or has unsupported policy")
    if type(trace.get("steps")) is not int or trace["steps"] != 9:
        raise ValueError("micro trace must contain exactly nine optimizer steps")
    events = trace.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 512:
        raise ValueError("numeric trace event set is missing or oversized")
    total = 0
    for ordinal, event in enumerate(events):
        if type(event.get("ordinal")) is not int or event["ordinal"] != ordinal:
            raise ValueError("numeric trace event order mismatch")
        if type(event.get("step")) is not int or not 0 <= event["step"] <= 9:
            raise ValueError("numeric trace step is invalid")
        names = set()
        for row in event["tensors"]:
            name = row["file"]
            if not isinstance(name, str) or pathlib.Path(name).name != name or not name.endswith(".bin"):
                raise ValueError("numeric trace tensor path must be confined")
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("numeric trace tensor is missing or a symlink")
            if row["name"] in names:
                raise ValueError("numeric trace tensor names must be unique")
            names.add(row["name"])
            if row["dtype"] not in DTYPES or any(type(n) is not int or n < 0 for n in row["shape"]):
                raise ValueError("numeric trace tensor dtype or shape is invalid")
            expected = math.prod(row["shape"]) * DTYPES[row["dtype"]][1]
            if row["bytes"] != expected or path.stat().st_size != expected:
                raise ValueError("numeric trace tensor byte count mismatch")
            total += expected
            if total > MAX_BYTES or digest(path) != row["sha256"]:
                raise ValueError("numeric trace tensor hash mismatch or oversized trace")
    if total != trace.get("tensor_bytes"):
        raise ValueError("numeric trace byte accounting mismatch")
    for phase in ("batch-input", "forward-logits", "raw-ctc", "total-loss", "gradient-before-clip",
                  "gradient-after-clip", "optimizer-step"):
        if [e["step"] for e in events if e["phase"] == phase] != list(range(1, 10)):
            raise ValueError(f"numeric trace missing ordered phase: {phase}")
    if events[0]["phase"] != "corpus" or events[-1]["phase"] != "final-state":
        raise ValueError("numeric trace missing endpoint evidence")
    if events[-1].get("state_sha256") != identity.get("float_state_sha256"):
        raise ValueError("trace final-state identity differs from exported training state")
    return identity, trace, root


def tensor_delta(left: pathlib.Path, right: pathlib.Path, row: dict) -> dict:
    fmt, width = DTYPES[row["dtype"]]
    a, b = left.read_bytes(), right.read_bytes()
    changed = 0
    maximum = 0.0
    first = None
    for index, ((av,), (bv,)) in enumerate(zip(struct.iter_unpack("<" + fmt, a), struct.iter_unpack("<" + fmt, b))):
        if a[index * width:(index + 1) * width] != b[index * width:(index + 1) * width]:
            if first is None:
                first = index
            changed += 1
            if not math.isfinite(av) or not math.isfinite(bv):
                raise ValueError("nonfinite tensor in numeric trace")
            maximum = max(maximum, abs(float(av) - float(bv)))
    return {"different_elements": changed, "first_different_flat_index": first, "max_abs_delta": maximum}


def cpu_vendor(identity: dict) -> str:
    model = str(identity.get("cpu_runtime", {}).get("model", "")).upper()
    if "AMD" in model:
        return "AMD"
    if "INTEL" in model:
        return "Intel"
    return "unknown"


def compare(left: pathlib.Path, right: pathlib.Path) -> dict:
    li, lt, lr = load_trace(left)
    ri, rt, rr = load_trace(right)
    required = ("torch_version", "fixture_manifest_sha256", "training_code_sha256", "micro_driver_sha256",
                "torch_num_threads", "torch_num_interop_threads", "pythonhashseed")
    for key in required:
        if key not in li or key not in ri or li[key] != ri[key]:
            raise ValueError(f"paired numeric trace input mismatch: {key}")
    if li["torch_num_threads"] != 1 or li["torch_num_interop_threads"] != 1 or li["pythonhashseed"] != "0":
        raise ValueError("paired numeric trace topology is not the pinned micro topology")
    mismatches = []
    for ordinal in range(max(len(lt["events"]), len(rt["events"]))):
        if ordinal >= min(len(lt["events"]), len(rt["events"])):
            mismatches.append({"ordinal": ordinal, "kind": "event-count"})
            break
        a, b = lt["events"][ordinal], rt["events"][ordinal]
        keys = ("phase", "step", "metadata")
        if any(a.get(k) != b.get(k) for k in keys):
            mismatches.append({"ordinal": ordinal, "phase": a["phase"], "step": a["step"], "kind": "event-metadata"})
            continue
        ar = {x["name"]: x for x in a["tensors"]}
        br = {x["name"]: x for x in b["tensors"]}
        if ar.keys() != br.keys():
            mismatches.append({"ordinal": ordinal, "phase": a["phase"], "step": a["step"], "kind": "tensor-set"})
            continue
        for name in ar:
            x, y = ar[name], br[name]
            if any(x[k] != y[k] for k in ("dtype", "shape", "sha256")):
                row = {"ordinal": ordinal, "phase": a["phase"], "step": a["step"], "tensor": name,
                       "left_sha256": x["sha256"], "right_sha256": y["sha256"]}
                if x["dtype"] == y["dtype"] and x["shape"] == y["shape"]:
                    row.update(tensor_delta(lr / x["file"], rr / y["file"], x))
                else:
                    row["kind"] = "tensor-geometry"
                mismatches.append(row)
    vendors = [cpu_vendor(li), cpu_vendor(ri)]
    state_equal = li["float_state_sha256"] == ri["float_state_sha256"]
    model_equal = li["model_sha256"] == ri["model_sha256"]
    return {"schema_version": 1, "evidence_class": "micro-numeric-divergence-report-v1",
            "infrastructure_complete": True, "exact_match": not mismatches and state_equal and model_equal,
            "float_state_equal": state_equal, "deployment_model_equal": model_equal,
            "first_observed_divergence": mismatches[0] if mismatches else None,
            "mismatches": mismatches, "cpu_vendors": vendors,
            "cross_vendor_pair_observed": set(vendors) == {"AMD", "Intel"},
            "left_environment": lt["environment"], "right_environment": rt["environment"],
            "historical_failure_resolved": False, "release_authority": False,
            "limitations": ["First recorded stage is not necessarily the first differing operator.",
                            "One matching pair is not a cross-platform reproducibility guarantee."]}


def observer_parity(plain_root: pathlib.Path, traced_root: pathlib.Path) -> None:
    import torch
    from training_state import state_identity
    a = torch.load(plain_root / "model.pt", weights_only=True, map_location="cpu")
    b = torch.load(traced_root / "model.pt", weights_only=True, map_location="cpu")
    if state_identity(a["state_dict"]) != state_identity(b["state_dict"]):
        raise ValueError("observer changed floating training state")
    if a["epoch_history"] != b["epoch_history"]:
        raise ValueError("observer changed epoch history")
    if digest(plain_root / "model.kwm") != digest(traced_root / "model.kwm"):
        raise ValueError("observer changed deployment weights")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=pathlib.Path)
    parser.add_argument("--right", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--observer-parity", action="store_true")
    args = parser.parse_args()
    try:
        if args.observer_parity:
            observer_parity(args.left, args.right)
            result = {"schema_version": 1, "observer_parity": True, "exact_match": True}
        else:
            result = compare(args.left, args.right)
        code = 0 if result["exact_match"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result = {"schema_version": 1, "infrastructure_complete": False, "exact_match": False,
                  "error": f"{type(exc).__name__}: {exc}", "release_authority": False}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("exact_match", "first_observed_divergence", "error") if k in result}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
