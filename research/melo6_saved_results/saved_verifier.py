"""Verbatim pure saved-verifier functions from Qwen6 compare_saved.py.
Configured by join_saved.py with the final Melo source's pure validators/recovery.
No original private panel/comparison logic is imported.
"""
from pathlib import Path, PurePosixPath
import json

def _safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input/output path")
    return path

def _read(path, limit=8 * 1024**2):
    path = _safe(path)
    require(path.is_file() and path.stat().st_size <= limit, "Missing/oversized regular input")
    raw = path.read_bytes()
    require(len(raw) <= limit, "Input grew beyond byte cap")
    return raw

def _json(raw):
    value = json.loads(raw, object_pairs_hook=_audio.no_duplicate_keys,
                       parse_float=_audio.finite_json_float,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    _audio.validate_json_values(value)
    return value

def _tree(root, total_cap):
    root = _safe(root)
    require(root.is_dir(), "Missing artifact directory")
    files, directories, size = {}, set(), 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Artifact symlink")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(name)
        else:
            raw = _read(path, min(total_cap, 8 * 1024**2))
            size += len(raw)
            require(size <= total_cap, "Artifact aggregate byte cap")
            files[name] = raw
    expected_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents
                     if str(parent) != "."}
    require(directories == expected_dirs, "Unexpected empty artifact directory")
    return files

def _verify_freeze(raw, payloads, schema, false_field):
    freeze = _json(raw)
    keys = {"files", false_field} | ({"schema"} if schema else set())
    require(type(freeze) is dict and set(freeze) == keys, "Freeze fields")
    require(freeze[false_field] is False and (not schema or freeze["schema"] == schema), "Freeze scope")
    require(type(freeze["files"]) is dict and set(freeze["files"]) == set(payloads), "Frozen membership differs")
    for name, expected in freeze["files"].items():
        _audio.valid_hash(expected)
        require(digest(payloads[name]) == expected, "Frozen file hash mismatch: " + name)
    return freeze

def _verify_asr(root, tts_sha, asr_sha):
    files = _tree(root, 20 * 1024**2)
    mandatory = {"artifact-freeze.json", "raw-freeze.json", "blind-job.json", "blind-input-freeze.json",
                 "final-status.json", "resources.json"}
    top = {name for name in files if "/" not in name}
    require(mandatory <= top <= mandatory | {"setup-summary.json"}, "ASR artifact top-level membership")
    raw = {name: data for name, data in files.items() if name.startswith("raw/")}
    require(set(files) == top | set(raw), "Unexpected artifact directory")
    _verify_freeze(files["artifact-freeze.json"], {k: v for k, v in files.items() if k != "artifact-freeze.json"},
                   "asr6-artifact-freeze-v1", "private_labels_joined")
    _verify_freeze(files["raw-freeze.json"], raw, "asr6-raw-freeze-v1", "labels_joined")
    common = {"model-raw-freeze.json", "not-run.json", "plan.json", "contract.json", "decoder-inputs.json",
              "environment.json", "outcomes.initial.json", "model-load-started.json", "model-load.json",
              "load-failure.json", "outcomes.json", "summary.json"}
    allowed = common | {f"outcomes.{i:04d}.json" for i in range(1, 7)} | {
        f"{audio_id}.{suffix}.json" for audio_id in _helpers["BATCH_IDS"]["all6"]
        for suffix in ("input-failure", "started", "decoder", "receipt")}
    for name in MODEL_DIRS:
        prefix = "raw/" + name + "/"
        members = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        require("model-raw-freeze.json" in members and set(members) <= allowed, "Model raw membership")
        _verify_freeze(members["model-raw-freeze.json"],
                       {k: v for k, v in members.items() if k != "model-raw-freeze.json"}, None, "labels_joined")
        if "not-run.json" in members:
            not_run = _json(members["not-run.json"])
            require(set(members) == {"not-run.json", "model-raw-freeze.json"}
                    and not_run == {"schema": "asr6-supervisor-not-run-v1", "status": "not_run",
                                    "model_decode_attempts": 0, "record_origin": "supervisor_no_model_load_receipt",
                                    "requested_ids": _helpers["BATCH_IDS"]["all6"]}, "Invalid zero-attempt not-run receipt")
    require(all(k.split("/")[1] in MODEL_DIRS for k in raw), "Unexpected raw model directory")
    resources = _json(files["resources.json"])
    require(resources["candidate_sha256"] == asr_sha and resources["tts_candidate_sha256"] == tts_sha,
            "ASR resources do not match reviewed paired candidates")
    _audio.valid_hash(resources["preregistered_plan_sha256"])
    job_sha = resources["blind_job_sha256"]
    job = _helpers["validate_job"](files["blind-job.json"], job_sha)
    blind = _helpers["validate_input_freeze"](files["blind-input-freeze.json"], resources["blind_input_freeze_sha256"])
    require(blind["job_sha256"] == job_sha, "Blind job/freeze mismatch")
    bound = {row["path"]: row for row in blind["files"]}
    require(set(bound) == {"job.json"} | {row["audio_path"] for row in job["clips"]}
            and bound["job.json"]["bytes"] == len(files["blind-job.json"]), "Blind freeze/job membership mismatch")
    results = {}
    for name, model_id in zip(MODEL_DIRS, MODELS):
        recovered = _helpers["recover_rows"](Path(root) / "raw" / name, job, resources.get(name, {}))
        results[model_id] = _helpers["adapt"](recovered, {"model_id": model_id}, job_sha)["clips"]
    # Recovery reads saved files. Detect source changes before exposing the plan.
    require(_tree(root, 20 * 1024**2) == files, "ASR artifact changed during recovery")
    return files, resources, job, bound, results
