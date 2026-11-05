#!/usr/bin/env python3
"""Read original evidence in its exact stdlib source environment.

The caller uses a fresh isolated Python process, the same process group and a
bounded timeout. This helper never executes an engine, queue worker or model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "D1_OLD_HELPER_BOUNDED_REQUEST")
    request = json.loads(raw)
    source = Path(request["source_root"]).absolute()
    require(not any(path.is_symlink() for path in (source, *source.parents)), "D1_OLD_HELPER_SOURCE_ALIAS")
    files = request["source_files"]
    require(type(files) is dict and len(files) == 25, "D1_OLD_HELPER_25_FILES")
    for name, expected in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name,
            "D1_OLD_HELPER_UNSAFE_SOURCE")
        path = source / relative
        require(not any(part.is_symlink() for part in (path, *path.parents))
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected, "D1_OLD_HELPER_SOURCE_CHANGED")
    sys.path[:0] = [str(source / "tools"), str(source / "training"), str(source)]
    import run_d1_zipformer_system as original
    require(Path(original.__file__).resolve() == source / "tools/run_d1_zipformer_system.py"
        and original.source_records() == files, "D1_OLD_HELPER_SOURCE_ENVIRONMENT")
    io = original.training.io_contract
    if request["action"] == "queue":
        root, _, definition, queue_sha = io.load_queue(request["queue_root"])
        value = {"root": str(root), "request": definition, "queue_sha256": queue_sha,
            "jobs": {job["job_id"]: io.result_for(root, job, queue_sha, check_active=False)
                     for job in definition["jobs"]}}
    else:
        require(request["action"] == "native", "D1_OLD_HELPER_ACTION")
        binding = request["previous_native"]
        root, _, definitions, queue_sha = io.load_queue(binding["queue_root"])
        jobs = [job for job in definitions["jobs"] if job["job_id"] == binding["job_id"]]
        require(len(jobs) == 1, "D1_OLD_HELPER_JOB_MISSING")
        terminal = io.result_for(root, jobs[0], queue_sha, check_active=False)
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0
            and terminal["actual_duration_known"] is True
            and terminal["receipt_sha256"] == binding["receipt_sha256"]
            and terminal["outputs"]["MANIFEST.json"]["sha256"] == binding["manifest_sha256"],
            "D1_OLD_HELPER_TERMINAL_CHANGED")
        recipe_path = root / "recipes" / (binding["job_id"] + ".json")
        recipe = original.validate_job(original.training.read_json(recipe_path))
        output = root / "jobs" / binding["job_id"] / "outputs"
        original.audit_finished(recipe_path, output)
        _, _, rows, _ = original.load_bound_rows(recipe)
        identity = original.training.read_json(output / "identity.json")
        observations, seconds = original.readback_native(output, rows, original.digest(identity))
        require(len(rows) == len(observations) == 64, "D1_OLD_HELPER_ORIGINAL64")
        value = {"root": str(root), "output": str(output), "recipe_path": str(recipe_path), "recipe": recipe,
            "terminal": terminal, "rows": rows, "observations": observations, "identity": identity,
            "identity_sha256": original.digest(identity), "native_inference_seconds": seconds}
    require(not any(name in sys.modules for name in original.ML_MODULES), "D1_OLD_HELPER_ML_IMPORT")
    value.update(helper_pid=os.getpid(), helper_pgid=os.getpgrp(), ml_imports=0, new_model_calls=0,
        verified_source_files=files)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
