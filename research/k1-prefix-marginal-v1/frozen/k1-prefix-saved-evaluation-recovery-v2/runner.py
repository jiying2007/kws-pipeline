#!/usr/bin/env python3
"""Disarmed, one-shot, stdlib-only evaluation of two exactly pinned raw files."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import types

HERE = Path(__file__).resolve().parent
OUTPUT_CAP = 1024 * 1024
CODE_FILES = ("manifest.json", "protocol.md", "runner.py", "test_runner.py", "causal-note.md")
PREVIOUS_ATTEMPT_SHA256 = "cf9c92a0e8db14cd187d8bbbd606a4dc0739fc62781e0f5dae69404c0313dbc2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bounded(path, limit):
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "input exceeds declared byte bound")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_constant(_):
    raise ValueError("nonfinite JSON constant")


def json_read(data):
    return json.loads(data, parse_constant=no_constant)


def softmax6(row, counts=None):
    """Recover collector FP32s; ordinary six-way softmax in binary64."""
    require(isinstance(row, list) and len(row) == 6, "expected six logits")
    recovered = []
    for value in row:
        require(type(value) in (int, float) and math.isfinite(value),
                "logit is not a finite number")
        value = struct.unpack("!f", struct.pack("!f", value))[0]
        require(math.isfinite(value), "FP32 logit overflow")
        recovered.append(value)
    high = max(recovered)
    terms = [math.exp(value - high) for value in recovered]
    for value, term in zip(recovered, terms):
        if term == 0:
            require(Fraction(value) - Fraction(high) <= -512,
                    "unexpected softmax exponential zero outside underflow guard")
            if counts is not None:
                counts["exponential_zeros"] += 1
    total = math.fsum(terms)
    weights = [value / total for value in terms]
    for value, term, weight in zip(recovered, terms, weights):
        if weight == 0:
            require(Fraction(value) - Fraction(high) <= -512,
                    "unexpected softmax division zero outside underflow guard")
            if counts is not None and term > 0:
                counts["division_only_zeros"] += 1
    if counts is not None:
        counts["rows_converted"] += 1
        counts["rows_with_zero_weight"] += int(any(value == 0 for value in weights))
    return weights


def softmax_callback(callback, observation, callback_index, counts=None):
    for row_index, row in enumerate(callback):
        try:
            yield softmax6(row, counts)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"observation={observation} callback_index={callback_index} "
                             f"row_index={row_index}: {exc}") from exc


def extract_source(raw, source, manifest):
    """Validate every callback and return complete observations in file order."""
    require(len(raw) == source["bytes"] and sha(raw) == source["sha256"],
            "saved source byte count or SHA256 mismatch")
    entries = [json_read(line) for line in raw.splitlines()]
    require(entries and entries[0]["kind"] == "run_start" and
            entries[-1]["kind"] == "run_end", "incomplete source run")
    header = entries[0]
    require(header["schema"] == source["schema"], "source schema mismatch")
    for key, value in manifest["common_raw_header"].items():
        require(header[key] == value, "raw header provenance mismatch")
    specs = {x["id"]: x for x in manifest["observations"]}
    collected, starts, active = {}, [], None
    for entry in entries:
        kind = entry["kind"]
        if kind == "clip_start":
            name = entry["recording"]
            require(active is None and name not in starts and
                    name in source["observations"], "unexpected observation boundary")
            require(entry["frames"] == specs[name]["eof_samples"], "EOF mismatch")
            active = name
            starts.append(name)
            collected[name] = []
        elif kind == "callback":
            name = entry["recording"]
            require(name == active, "callback outside its observation")
            spec = specs[name]
            index = len(collected[name])
            require(index < spec["callbacks"] and entry["call_index"] == index,
                    "callback index mismatch")
            rows = entry["logits"]
            require(isinstance(rows, list) and len(rows) == spec["rows_per_callback"][index]
                    and entry["selected_rows"] == len(rows)
                    and len(entry["centers"]) == len(rows)
                    and all(isinstance(row, list) and len(row) == 6 for row in rows),
                    "callback geometry mismatch")
            require(entry["phase"] == spec["phases"][index]
                    and entry["available_samples"] == min(4800 * (index + 1), spec["eof_samples"]),
                    "callback phase or availability mismatch")
            collected[name].append(rows)
        elif kind == "clip_end":
            name = entry["recording"]
            require(name == active and entry["complete"] is True,
                    "incomplete observation")
            spec = specs[name]
            require(entry["frames"] == spec["eof_samples"] and
                    entry["model_rows"] == spec["rows"] and
                    entry["callbacks"] == spec["callbacks"] and
                    len(collected[name]) == spec["callbacks"] and
                    sum(map(len, collected[name])) == spec["rows"], "observation count mismatch")
            active = None
        else:
            require(kind in {"run_start", "model_loaded", "feed", "finish", "run_end"},
                    "unknown source record kind")
    require(active is None and starts == source["observations"], "observation order mismatch")
    require(entries[-1]["complete"] is True and entries[-1]["clips"] == len(starts),
            "incomplete source footer")
    return collected


class Outputs:
    """Exclusive attempt, bounded output, and durable progress/failure evidence."""
    def __init__(self, directory):
        self.directory = directory
        self.used = 0
        self.fd = os.open(directory / "attempt.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    def encode(self, obj):
        data = (json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()
        require(len(data) <= 65536 and self.used + len(data) <= OUTPUT_CAP - 4096,
                "total output budget exceeded")
        self.used += len(data)
        return data

    def event(self, kind, **fields):
        data = self.encode({"kind": kind, **fields})
        with os.fdopen(os.dup(self.fd), "ab", buffering=0) as stream:
            stream.write(data)
            os.fsync(stream.fileno())

    def result(self, obj):
        data = self.encode(obj)
        with open(self.directory / "results.json", "xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def check_release(args):
    raw = read_bounded(args.release, 16384)
    require(sha(raw) == args.release_sha256, "release SHA256 mismatch")
    release = json_read(raw)
    require(release["schema"] == "k1-first-prefix-saved-release-v2" and
            release["state"] == "RELEASED" and release["candidate"] == "k1-first-prefix-ctc-eof-v1",
            "release is not armed for this candidate")
    require(release["cumulative_attempt"] == 2 and
            release["previous_attempt_sha256"] == PREVIOUS_ATTEMPT_SHA256,
            "release does not bind the preserved failed first attempt")
    require(isinstance(release["review_ref"], str) and release["review_ref"].strip()
            and isinstance(release["authorization_ref"], str) and release["authorization_ref"].strip(),
            "review and authorization references are required")
    for name in CODE_FILES:
        require(sha(read_bounded(HERE / name, 131072)) == release["sha256"][name],
                "reviewed source hash mismatch: " + name)
    module_bytes = read_bounded(args.module, 131072)
    require(sha(module_bytes) == release["sha256"]["k1_prefix_marginal.py"],
            "reviewed mathematical module hash mismatch")
    return json_read(read_bounded(HERE / "manifest.json", 131072)), release, module_bytes


def load_module(path, source):
    sys.dont_write_bytecode = True
    module = types.ModuleType("reviewed_k1_prefix")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def finite_log(value):
    return "-inf" if value == -math.inf else value


def evaluate(module, manifest, observations, outputs):
    results = []
    for spec in manifest["observations"]:
        verifier = module.K1PrefixMarginal(token_order=manifest["token_order"])
        counts = {key: 0 for key in ("exponential_zeros", "division_only_zeros",
                                    "rows_converted", "rows_with_zero_weight")}
        try:
            for callback_index, callback in enumerate(observations[spec["id"]]):
                verifier.update_chunk(softmax_callback(callback, spec["id"], callback_index, counts))
        except Exception as exc:
            outputs.event("observation_failed", id=spec["id"], softmax_underflow=counts,
                          committed_rows=verifier.snapshot().frames,
                          error_type=type(exc).__name__, reason=str(exc)[:1000])
            raise
        final = verifier.finalize()
        require(final.emitted and final.snapshot.frames == spec["rows"], "EOF lifecycle mismatch")
        status = final.decision_status
        expected = "CERTIFIED_ACCEPT" if spec["label"] == "A" else "CERTIFIED_REJECT"
        result = {"id": spec["id"], "human_text": spec["human_text"], "label": spec["label"],
                  "rows": spec["rows"], "callbacks": spec["callbacks"],
                  "masses": dict(zip(("A", "B", "R"), final.masses)),
                  "log_masses": dict(zip(("A", "B", "R"), map(finite_log, final.log_masses))),
                  "mass_bounds": {bucket: {side: {"numerator": bound.numerator,
                                  "denominator": bound.denominator}
                                  for side, bound in zip(("lower", "upper"), bounds)}
                                  for bucket, bounds in zip(("A", "B", "R"), final.snapshot.mass_bounds)},
                  "certificate_status": status, "decision": final.decision,
                  "softmax_underflow": counts,
                  "gate_pass": status == expected,
                  "eof_availability_samples": spec["eof_samples"],
                  "eof_availability_ms": spec["eof_samples"] * 1000 / manifest["sample_rate_hz"]}
        results.append(result)
        outputs.event("observation_eof", **result)
    return {"schema": manifest["schema"], "candidate": manifest["candidate"],
            "cumulative_attempt": 2, "previous_attempt_sha256": PREVIOUS_ATTEMPT_SHA256,
            "softmax_underflow_totals": {key: sum(x["softmax_underflow"][key] for x in results)
                                         for key in results[0]["softmax_underflow"]},
            "results": results, "gate_pass": all(x["gate_pass"] for x in results),
            "primary_gate_pass": all(x["gate_pass"] for x in results if x["id"] in ("M3", "N1")),
            "failure_action": "STOP_CANDIDATE_NO_TUNING", "totals": manifest["totals"],
            "interpretation": "Existing exposed single stock voice; EOF availability is not word latency."}


def exceeded(signum, frame):
    raise TimeoutError("resource deadline exceeded")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for key in ("release", "release-sha256", "module", "melo5-raw", "n1-raw"):
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    if not args.execute:
        print("DISARMED: no saved values read; explicit reviewed release required")
        return 77
    outputs = None
    started_wall, started_cpu = time.monotonic(), time.process_time()
    try:
        require(all(getattr(args, key) for key in ("release", "release_sha256", "module", "melo5_raw", "n1_raw")),
                "all explicit release and source arguments are required")
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_CAP, OUTPUT_CAP))
        signal.signal(signal.SIGALRM, exceeded)
        signal.signal(signal.SIGXCPU, exceeded)
        signal.alarm(15)
        manifest, release, module_bytes = check_release(args)
        outputs = Outputs(HERE)
        outputs.event("attempt_started", release_sha256=args.release_sha256, source_hashes=release["sha256"],
                      cumulative_attempt=2, previous_attempt_sha256=PREVIOUS_ATTEMPT_SHA256)
        observations = {}
        for key, path in (("melo5", args.melo5_raw), ("n1", args.n1_raw)):
            source = manifest["sources"][key]
            observations.update(extract_source(read_bounded(path, source["bytes"]), source, manifest))
        module = load_module(args.module, module_bytes)
        require(tuple(manifest["token_order"]) == module.TOKENS, "token mapping mismatch")
        result = evaluate(module, manifest, observations, outputs)
        result.update(release_sha256=args.release_sha256,
                      wall_seconds=time.monotonic() - started_wall,
                      cpu_seconds=time.process_time() - started_cpu)
        outputs.result(result)
        outputs.event("attempt_complete", gate_pass=result["gate_pass"])
        print("Completed one saved-logit attempt; gate " + ("PASS" if result["gate_pass"] else "FAIL"))
        return 0 if result["gate_pass"] else 1
    except Exception as exc:
        if outputs is not None:
            outputs.event("attempt_failed", error_type=type(exc).__name__, reason=str(exc)[:1000])
        print("STOP: " + type(exc).__name__ + ": " + str(exc)[:1000], file=sys.stderr)
        return 2
    finally:
        signal.alarm(0)
        if outputs is not None:
            os.close(outputs.fd)


if __name__ == "__main__":
    raise SystemExit(main())
