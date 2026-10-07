#!/usr/bin/env python3
"""One-shot source qualification, acquisition and bounded CosyVoice3 generation.

No command installs dependencies. Network transfers occur only in acquire-source
and acquire-model, on the externally admitted GitHub runner. Static tests do not
import the runtime. All gates fail closed; no automatic retry is implemented.
"""
from __future__ import annotations

import argparse
import array
import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_common import *
from bounded_log import BoundedLog


def layout(root):
    root = Path(root).resolve()
    return {"root": root, "generation": root / "generation", "source": root / "generation/source", "model": root / "generation/model", "references": root / "generation/references", "scratch": root / "generation/scratch", "receipts": root / "receipts/generation", "output": root / "outputs"}


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump({"run_id": os.environ.get("GITHUB_RUN_ID"), "claimed_at": time.time(), "retry_permitted": False}, f)


def source_gate(paths, args):
    runtime = runtime_gate(args.runtime_receipt, args.runtime_lock)
    sources = verify_source(paths["source"])
    reference = verify_references(paths["references"], load_config())
    return {"runtime": runtime, "source": sources, "reference": reference}


def qualified_source_gate(paths, args):
    inputs = source_gate(paths, args)
    q = read_json(paths["receipts"] / "source-qualification.json")
    require(q.get("status") == "qualified", "source qualification did not pass")
    require(q.get("inputs") == inputs, "source/runtime/reference changed after qualification")
    require(q.get("adapter_sha256") == sha256(HERE / "soundfile_adapter.py"), "adapter changed")
    return inputs


def acquire_source(paths, args):
    claim(paths["receipts"] / "source-acquisition.claim.json")
    manifest = read_json(HERE / "source-allowlist.json")
    paths["source"].mkdir(parents=True, exist_ok=False)
    for row in manifest["files"]:
        job_gate(paths["root"], RESERVE + row["bytes"])
        download_one(paths["source"], row["path"], row, row["url"])
    paths["references"].mkdir(exist_ok=False)
    for r in load_config()["references"]:
        url = "https://raw.githubusercontent.com/" + r["public_repository"] + "/" + r["commit"] + "/" + r["repository_path"]
        job_gate(paths["root"], RESERVE + r["bytes"])
        download_one(paths["references"], r["reference_file"], {"bytes": r["bytes"], "sha256": r["sha256"], "git_blob_sha1": r["blob_sha1"]}, url)
    write_json(paths["receipts"] / "source-acquisition.json", {"status": "acquired", "source": verify_source(paths["source"]), "reference": verify_references(paths["references"], load_config())})


def process_tree_rss(pid):
    """Linux process descendants; threads share RSS and are not double counted."""
    rows = {}
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = p.read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            rows[int(p.parent.name)] = (int(fields[1]), max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {pid}
    while True:
        enlarged = descendants | {p for p, (parent, _) in rows.items() if parent in descendants}
        if enlarged == descendants:
            break
        descendants = enlarged
    return sum(rows[p][1] for p in descendants if p in rows)


def terminate_tree(proc):
    # The outer controller owns the process group and cleans every descendant
    # on phase exit. Do not detach a new process group from that ownership.
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def supervise(paths, mode):
    """900-second startup and 300-second per-call monotonic deadlines."""
    claim(paths["receipts"] / (mode + ".claim.json"))
    state = paths["receipts"] / (mode + "-state.json")
    result = paths["receipts"] / (mode + "-worker.json")
    log = paths["receipts"] / (mode + ".log")
    scratch = paths["scratch"] / mode
    scratch.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, "-I", "-B", str(HERE / "pilot_worker.py"), mode, "--source", str(paths["source"]), "--scratch", str(scratch), "--state", str(state), "--result", str(result)]
    if mode == "generate":
        cmd += ["--model", str(paths["model"]), "--reference", str(paths["references"]), "--output", str(paths["output"])]
    started = time.monotonic()
    peak = 0
    failure = None
    events = []
    proc = None
    capture = None
    try:
        with open(log, "xb") as stream:
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=execution_env())
                capture = BoundedLog(proc.stdout, stream, LOG_LIMIT)
                while proc.poll() is None:
                    capture.pump(timeout=0.2)
                    require(not capture.truncated, "worker log exceeded byte limit")
                    now = time.monotonic()
                    current = read_json(state) if state.exists() else {"phase": "startup", "clip_id": None, "monotonic": started}
                    phase = current["phase"]
                    require(phase in ("startup", "clip", "complete"), "unknown worker state")
                    require(started - 1 <= current["monotonic"] <= now + 1, "invalid worker timestamp")
                    if not events or (events[-1]["phase"], events[-1]["clip_id"]) != (phase, current["clip_id"]):
                        events.append(current)
                    limit = 300 if phase == "clip" else 900
                    require(now - current["monotonic"] <= limit, f"{phase} exceeded {limit}s")
                    require(now - started <= 3300, "generation phase wall limit exceeded")
                    peak = max(peak, process_tree_rss(proc.pid))
                    require(peak <= RSS_LIMIT, "worker process tree RSS exceeded 12 GiB")
                    memory_gate()
                    job_gate(paths["root"], scan=False)
                    output_bytes = tree_bytes(paths["output"]) if paths["output"].exists() else 0
                    require(output_bytes + tree_bytes(paths["receipts"]) <= OUTPUT_LIMIT, "outputs+receipts exceeded 128 MiB")
                    if capture.eof:
                        time.sleep(0.2)
            finally:
                try:
                    if proc is not None and proc.poll() is None:
                        terminate_tree(proc)
                finally:
                    if capture is not None:
                        capture.finish()
            # Fast-exiting children can leave their entire output buffered in the
            # pipe, so enforce the same cap again after final draining.
            require(not capture.truncated, "worker log exceeded byte limit")
            require(capture.eof, "worker log pipe remained open after exit")
        require(proc.returncode == 0, f"worker exited {proc.returncode}")
        worker = read_json(result)
        require(worker.get("status") == ("qualified" if mode == "qualify" else "complete"), "worker result not successful")
    except Exception as e:
        failure = str(e)
    receipt = {"status": "passed" if failure is None else "failed", "failure": failure, "returncode": proc.returncode if proc is not None else None, "peak_process_tree_rss_bytes": peak, "elapsed_seconds": time.monotonic() - started, "events": events, "no_retry": True, "log": log.name, "log_bytes": capture.written if capture is not None else 0, "log_limit_bytes": LOG_LIMIT, "log_truncated": capture.truncated if capture is not None else False, "log_observed_bytes": capture.observed_bytes if capture is not None else 0, "log_eof": capture.eof if capture is not None else False, "log_drain_timed_out": capture.drain_timed_out if capture is not None else False}
    write_json(paths["receipts"] / (mode + "-supervisor.json"), receipt)
    require(failure is None, failure)
    return read_json(result)


def qualify_source(paths, args):
    inputs = source_gate(paths, args)
    job_gate(paths["root"])
    memory_gate()
    result = supervise(paths, "qualify")
    require(result.get("model_constructed") is False and result.get("network_attempts") == [], "source gate violated")
    write_json(paths["receipts"] / "source-qualification.json", {"status": "qualified", "inputs": inputs, "adapter_sha256": sha256(HERE / "soundfile_adapter.py"), "qualification": result})


def acquire_model(paths, args):
    inputs = qualified_source_gate(paths, args)
    admission = job_gate(paths["root"], MODEL_FREE_MIN)
    memory_gate()
    claim(paths["receipts"] / "model-acquisition.claim.json")
    paths["model"].mkdir(parents=True, exist_ok=False)
    files = []
    for row in load_config()["weights"]["files"]:
        job_gate(paths["root"], row["size"] + RESERVE + OUTPUT_LIMIT)
        files.append(download_one(paths["model"], row["rfilename"], row, row["source_url"]))
    write_json(paths["receipts"] / "model-acquisition.json", {"status": "acquired", "inputs": inputs, "admission": admission, "files": files})


def run_generation(paths, args):
    qualified_source_gate(paths, args)
    require(read_json(paths["receipts"] / "model-acquisition.json").get("status") == "acquired", "model acquisition receipt missing")
    model_files = verify_models(paths["model"], load_config())
    require(model_files == read_json(paths["receipts"] / "model-acquisition.json")["files"], "model identities changed")
    job_gate(paths["root"], RESERVE + OUTPUT_LIMIT)
    memory_gate()
    paths["output"].mkdir(parents=True, exist_ok=False)
    write_json(paths["output"] / "input-identities.json", {"config_sha256": CONFIG_SHA256, "source": verify_source(paths["source"]), "models": model_files, "reference": verify_references(paths["references"], load_config())})
    supervise(paths, "generate")
    # In-memory prompt caching must not have created a model-side spk2info file.
    verify_models(paths["model"], load_config())
    verify_source(paths["source"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("acquire-source", "qualify-source", "acquire-model", "run"))
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--runtime-lock")
    parser.add_argument("--runtime-receipt")
    args = parser.parse_args()
    paths = layout(args.job_root)
    load_config()
    require_runner()
    paths["receipts"].mkdir(parents=True, exist_ok=True)
    try:
        if args.command in ("qualify-source", "acquire-model", "run"):
            require(args.runtime_lock and args.runtime_receipt, "runtime gate inputs required")
        job_gate(paths["root"])
        memory_gate()
        {"acquire-source": acquire_source, "qualify-source": qualify_source, "acquire-model": acquire_model, "run": run_generation}[args.command](paths, args)
    except Exception as e:
        write_json(paths["receipts"] / (args.command + "-failure.json"), {"status": "failed", "error_type": type(e).__name__, "error": str(e)[:2000], "retry_permitted": False})
        raise


if __name__ == "__main__":
    main()
