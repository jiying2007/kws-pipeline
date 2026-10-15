#!/usr/bin/env python3
"""New blind PCM recognizer worker. No producer code, target context, or retries.

The only scientific execution paths are the byte-pinned retained loaders and
inference functions. This module owns a new durable campaign-local attempt
ledger, strict PCM transport, and kill-safe recovery; old runners are unused.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import platform
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from blind_pcm import (MAX_FILES, MAX_INPUT_BYTES, MAX_JOB_BYTES, RETAINED_PINS,
                       atomic, bind_input, canonical, canonical_wav, decode,
                       fsync_directory, identity_map, read_regular, reconstruct,
                       require, sha, unaliased, validate_job, write_once)

RECOGNIZERS = ("qwen_asr", "sensevoice", "whisper_small")
LOAD_SECONDS = 1200
CALL_SECONDS = 300
SOURCE_ROOT = HERE.parents[1]
SCRATCH = Path("/scratch")
OUTCOME_FIELDS = {"opaque_id", "recognizer", "pcm_sha256", "wav_sha256", "status", "attempts",
                  "raw_text", "completeness", "quality_flags", "error_type"}


def verify_sources(root=SOURCE_ROOT):
    root = unaliased(root)
    expected = set(RETAINED_PINS) | {"research/kw1-preset-supplement-v1/blind_asr_worker.py",
                                    "research/kw1-preset-supplement-v1/blind_pcm.py"}
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "source aliases and special files forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == expected, "isolated source inventory differs")
    for name in expected - set(RETAINED_PINS):
        read_regular(root / name, 2 * 1024**2)
    for name, expected_sha in RETAINED_PINS.items():
        require(sha(read_regular(root / name, 2 * 1024**2)) == expected_sha, "retained scientific source drift")
    return dict(RETAINED_PINS)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module source unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_scope():
    path = SOURCE_ROOT / "research/source-screen32-v1/runtime_scope.py"
    require(sha(read_regular(path, MAX_JOB_BYTES)) == RETAINED_PINS["research/source-screen32-v1/runtime_scope.py"], "scope verifier source drift")
    return module_from_file("runtime_scope", path).verify_runtime_scope()


def prepare_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", ORT_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="",
                      TOKENIZERS_PARALLELISM="false")
    for name in ("HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
                 "NUMBA_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR"):
        path = SCRATCH / name.lower(); path.mkdir(exist_ok=True)
        require(unaliased(path).is_dir(), "private writable cache")
        os.environ[name] = str(path)


def verify_runtime(recognizer, runtime):
    """Read-only runtime proof before loading any third-party inference module."""
    verify_scope()
    runtime = unaliased(runtime)
    require(runtime == Path("/runtime"), "fixed runtime mount required")
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.12.14"
            and platform.machine() == "x86_64" and sys.platform == "linux" and sys.byteorder == "little", "exact runtime host required")
    require(Path(sys.executable).absolute() == runtime / "venv/bin/python", "locked interpreter required")
    if recognizer != "whisper_small":
        adapter = module_from_file("kw1_setup_adapter", SOURCE_ROOT / "research/source-screen32-v1/setup_adapter.py")
        return adapter.verify_runtime(runtime, "asr")
    common = module_from_file("common", SOURCE_ROOT / "research/source-screen32-whisper-v1/common.py")
    lock = common.read_json(SOURCE_ROOT / "research/source-screen32-whisper-v1/source-lock.json")
    common.verify_inventory(runtime / "source", {r["path"]: {"sha256": r["sha256"], "bytes": r["size_bytes"]} for r in lock["files"]})
    inventory = common.read_json(runtime / "installed-inventory.json")
    require(set(inventory) == {"schema", "files"} and inventory["schema"] == "whisper-screen-installed-runtime-v1", "installed runtime inventory")
    common.verify_inventory(runtime / "venv/lib/python3.12/site-packages", inventory["files"])
    asset = common.read_json(runtime / "asset-lock.json")
    require(asset["files"]["small.pt"]["sha256"] == common.MODEL_SHA, "locked small checkpoint")
    for name, expected in asset["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "asset path")
        path = runtime / "model" / name if name == "small.pt" else runtime / "source" / name
        require(common.identity(path) == expected, "runtime asset identity")
    return {"installed_inventory_sha256": sha(read_regular(runtime / "installed-inventory.json", MAX_INPUT_BYTES)),
            "asset_lock_sha256": sha(read_regular(runtime / "asset-lock.json", MAX_JOB_BYTES))}


class RetainedBackend:
    """Tiny bridge into the original source. No patched scientific functions."""
    def __init__(self, recognizer, runtime, wav_root, output):
        self.recognizer, self.runtime, self.wav_root, self.output = recognizer, Path(runtime), Path(wav_root), Path(output)
        primary = SOURCE_ROOT / "research/source-screen32-v1"
        if str(primary) not in sys.path:
            sys.path.insert(0, str(primary))
        self.worker = module_from_file("kw1_retained_asr_worker", primary / "asr_worker.py")
        self.worker.verify_sources()
        self.helpers, _ = self.worker.import_helpers()
        _, _, self.decoder_raw = self.worker.bind_inputs(wav_root)
        self.inputs = self.worker.validate_decoder(self.decoder_raw, sha(self.decoder_raw))
        write_once(output / "decoder-inputs.json", decode(self.decoder_raw, 1024**2))
        if recognizer == "whisper_small":
            whisper_dir = SOURCE_ROOT / "research/source-screen32-whisper-v1"
            sys.path.insert(0, str(whisper_dir))
            module_from_file("common", whisper_dir / "common.py")
            adapter = module_from_file("kw1_retained_whisper_adapter", whisper_dir / "adapter.py")
            self.model = adapter.FixedWhisper(runtime, output)
            self.receipt = {"native_config_sha256": sha(read_regular(output / "native-config.json", MAX_INPUT_BYTES)),
                            "load_receipt_sha256": sha(read_regular(output / "load-receipt.json", MAX_INPUT_BYTES)),
                            "decoder_manifest_sha256": sha(self.decoder_raw)}
        else:
            self.model_key = "qwen06" if recognizer == "qwen_asr" else "sensevoice"
            lock = decode(read_regular(self.worker.RETAINED / "model-locks.json", MAX_JOB_BYTES))[self.model_key]
            declaration = self.worker.execution_contract(self.model_key, lock, self.decoder_raw)
            contract_sha = write_once(output / "execution-contract.json", declaration)
            self.model_root = self.runtime / "models" / self.model_key
            loader = self.worker.load_qwen_after_approval if recognizer == "qwen_asr" else self.worker.load_sense_after_approval
            self.model = loader(str(self.model_root), lock["asset_lock"], lock["asset_lock_sha256"],
                                lock["source_lock"], lock["source_lock_sha256"], declaration, contract_sha, self.decoder_raw)
            self.sense = module_from_file("kw1_retained_sense_adapter", primary / "sense_adapter.py") if recognizer == "sensevoice" else None
            self.receipt = {"scientific_load": self.model.receipt, "execution_contract_sha256": contract_sha,
                            "decoder_manifest_sha256": sha(self.decoder_raw)}

    def decode(self, identity, pcm, cell):
        oid = identity["retained_opaque_id"]
        bound = self.worker.bind_wave(str((self.wav_root / "audio").absolute()),
            self.worker.WaveExpectation(oid, identity["wav_sha256"] + ".wav", identity["wav_sha256"],
                                       identity["wav_bytes"], identity["frames"]))
        require(bound.pcm16_le == pcm and bound.descriptor["raw_pcm16"]["sha256"] == identity["pcm_sha256"], "PCM-to-WAV identity")
        require(bound.descriptor == self.inputs[oid]["descriptor"] and
                bound.descriptor_sha256 == self.inputs[oid]["binding_sha256"], "retained input binding changed")
        if self.recognizer == "whisper_small":
            raw = self.model.decode(bound.pcm16_le, identity["frames"], cell)
            require(type(raw) is dict and type(raw.get("decoder_completion_validated")) is bool and
                    raw.get("full_input_processed") is True, "native Whisper result contract")
            row = {"raw_text": raw["raw_text"],
                   "completeness": "complete" if raw["decoder_completion_validated"] else "incomplete", "quality_flags": []}
            return row, raw
        if self.recognizer == "qwen_asr":
            return self.helpers.infer_qwen_once_after_approval(self.model, bound, sha(self.decoder_raw))
        state = {}
        return self.sense.infer_sense_once_after_approval(self.model, bound, sha(self.decoder_raw), self.model_root, state)


@contextmanager
def deadline(seconds):
    def alarm(_signal, _frame):
        raise TimeoutError("single model attempt deadline")
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def progress(output, stage, index):
    require(stage in ("load", "cell") and type(index) is int and
            ((stage == "load" and index == 0) or (stage == "cell" and 1 <= index <= MAX_FILES)), "progress state")
    atomic(Path(output) / "progress.json", {"stage": stage, "index": index, "started_monotonic": time.monotonic()})


def initial_outcomes(mapping, recognizer):
    return [{"opaque_id": r["opaque_id"], "recognizer": recognizer, "pcm_sha256": r["pcm_sha256"],
             "wav_sha256": r["wav_sha256"], "status": "NOT_RUN", "attempts": 0,
             "raw_text": None, "completeness": "unknown", "quality_flags": [], "error_type": None} for r in mapping["clips"]]


def validate_result(row):
    require(type(row) is dict and set(row) == {"raw_text", "completeness", "quality_flags"}, "recognizer result fields")
    require(type(row["raw_text"]) is str and len(row["raw_text"]) <= 32768 and
            row["completeness"] in ("complete", "incomplete", "unknown") and
            type(row["quality_flags"]) is list and len(row["quality_flags"]) <= 64 and
            all(type(flag) is str and len(flag) <= 256 for flag in row["quality_flags"]), "recognizer result values")
    return row["completeness"] == "complete" and row["quality_flags"] == []


def error_kind(error):
    return "timeout" if isinstance(error, TimeoutError) else "exception"


def run(recognizer, input_root, output_root, runtime_root):
    require(recognizer in RECOGNIZERS, "recognizer choice")
    job, job_raw, pcm = bind_input(input_root)
    mapping = identity_map(job, job_raw, pcm)
    output = unaliased(output_root)
    require(output.is_dir() and not any(output.iterdir()), "fresh empty output directory required; claims cannot be reused")
    claim = {"schema": "kw1-blind-asr-stage-claim-v1", "recognizer": recognizer,
             "job_sha256": sha(job_raw), "identity_map_sha256": sha(canonical(mapping)), "attempt": 1}
    write_once(output / "stage-claim.json", claim)
    write_once(output / "identity-map.json", mapping)
    outcomes = initial_outcomes(mapping, recognizer)
    write_once(output / "outcomes.initial.json", outcomes)
    cells = output / "cells"; cells.mkdir(mode=0o700); fsync_directory(output)
    failure = None
    load_started = False
    load_completed = False
    try:
        verify_sources()
        scope = verify_scope()
        write_once(output / "scope.json", scope)
        prepare_environment()
        wav_root, observed = reconstruct(job, job_raw, pcm, SCRATCH)
        require(observed == mapping, "reconstruction identity")
        progress(output, "load", 0)
        write_once(output / "model-load-started.json", {**claim, "load_seconds": LOAD_SECONDS})
        load_started = True
        with deadline(LOAD_SECONDS):
            runtime_identity = verify_runtime(recognizer, runtime_root)
            verify_scope()
            model = RetainedBackend(recognizer, runtime_root, wav_root, output)
            load_sha = write_once(output / "model-load-completed.json", {**claim, "receipt": {"schema":"kw1-public-model-load-identity-v1", "recognizer":recognizer, "scientific_load_sha256":sha(canonical(model.receipt)), "runtime_identity_sha256":sha(canonical(runtime_identity)), "source_pins_sha256":sha(canonical(RETAINED_PINS))}})
            load_completed = True
        for index, (identity, outcome) in enumerate(zip(mapping["clips"], outcomes), 1):
            # Progress covers the entire per-cell scope/binding/inference operation.
            progress(output, "cell", index)
            cell = cells / identity["opaque_id"]; cell.mkdir(mode=0o700); fsync_directory(cells)
            started_sha = None
            try:
                with deadline(CALL_SECONDS):
                    verify_scope()
                    current_job, current_raw, current_pcm = bind_input(input_root)
                    require(current_raw == job_raw and identity_map(current_job, current_raw, current_pcm) == mapping, "per-call PCM identity changed")
                    started = {"schema": "kw1-blind-asr-cell-started-v1", "recognizer": recognizer,
                               "opaque_id": identity["opaque_id"], "pcm_sha256": identity["pcm_sha256"],
                               "wav_sha256": identity["wav_sha256"], "job_sha256": sha(job_raw),
                               "identity_map_sha256": sha(canonical(mapping)), "load_completed_sha256": load_sha, "attempt": 1}
                    started_sha = write_once(cell / "started.json", started)
                    outcome.update(status="FAILED", attempts=1, error_type="interrupted")
                    result, evidence = model.decode(identity, current_pcm[identity["opaque_id"]], cell)
                    accepted = validate_result(result)
                    outcome.update(result)
                    evidence_sha = write_once(cell / "decoder-evidence.json", {"result": result, "evidence": {"schema": "kw1-public-decoder-evidence-v1", "recognizer": recognizer, "scientific_result_sha256": sha(canonical(evidence))}})
                    if not accepted:
                        outcome["error_type"] = "decoder_quality"
                        write_once(cell / "failed.json", {"schema": "kw1-blind-asr-cell-failed-v1", "reason": "decoder_quality",
                            "started_sha256": started_sha, "decoder_evidence_sha256": evidence_sha, "outcome": outcome})
                        failure = "decoder_quality"
                        break
                    outcome.update(status="SUCCESS", error_type=None)
                    write_once(cell / "completed.json", {"schema": "kw1-blind-asr-cell-completed-v1", "started_sha256": started_sha,
                        "decoder_evidence_sha256": evidence_sha, "outcome": outcome})
            except BaseException as error:
                failure = error_kind(error)
                if started_sha is not None:
                    outcome.update(status="FAILED", attempts=1, error_type=failure)
                    write_once(cell / "failed.json", {"schema": "kw1-blind-asr-cell-failed-v1", "reason": failure,
                        "started_sha256": started_sha, "decoder_evidence_sha256": None, "outcome": outcome})
                break
            atomic(output / "outcomes.progress.json", outcomes)
    except BaseException as error:
        failure = error_kind(error)
        if load_started and not load_completed:
            write_once(output / "model-load-failed.json", {**claim, "reason": failure})
    status = "complete" if failure is None and all(row["status"] == "SUCCESS" for row in outcomes) else "stopped"
    if failure is not None:
        write_once(output / "stage-failed.json", {"reason": failure, "retries": 0})
    write_once(output / "outcomes.json", outcomes)
    summary = {"schema": "kw1-blind-asr-summary-v1", "recognizer": recognizer, "status": status,
               "requested": len(outcomes), "attempted": sum(r["attempts"] for r in outcomes),
               "success": sum(r["status"] == "SUCCESS" for r in outcomes), "not_run": sum(r["status"] == "NOT_RUN" for r in outcomes),
               "retries": 0, "job_sha256": sha(job_raw), "identity_map_sha256": sha(canonical(mapping))}
    write_once(output / "summary.json", summary)
    return summary


def _read(path):
    return decode(read_regular(path, MAX_INPUT_BYTES), MAX_INPUT_BYTES)


def _validate_outcome(outcome, initial, status):
    require(type(outcome) is dict and set(outcome) == OUTCOME_FIELDS, "terminal outcome fields")
    require(all(outcome[key] == initial[key] for key in ("opaque_id", "recognizer", "pcm_sha256", "wav_sha256")), "terminal PCM/WAV identity")
    require(outcome["status"] == status and type(outcome["attempts"]) is int and outcome["attempts"] == 1, "terminal attempt identity")
    require(outcome["error_type"] is None if status == "SUCCESS" else outcome["error_type"] in
            ("exception", "timeout", "decoder_quality", "interrupted"), "terminal error classification")
    if outcome["raw_text"] is None:
        require(status == "FAILED" and outcome["completeness"] == "unknown" and outcome["quality_flags"] == [], "terminal absent result")
    else:
        accepted = validate_result({key: outcome[key] for key in ("raw_text", "completeness", "quality_flags")})
        require(status != "SUCCESS" or accepted, "terminal unsuccessful decoder")
    return outcome


def recover_outcomes(input_root, output_root, recognizer):
    """Read-only kill-safe accounting. Model/runtime modules are never imported.

    Immutable receipts, rather than mutable snapshots, establish each completed
    call. A start without a valid completion consumes its attempt and is FAILED.
    Unstarted cells remain NOT_RUN; no recovery operation retries model work.
    """
    require(recognizer in RECOGNIZERS, "recognizer choice")
    job, job_raw, pcm = bind_input(input_root)
    mapping = identity_map(job, job_raw, pcm)
    outcomes = initial_outcomes(mapping, recognizer)
    output = unaliased(output_root)
    if not output.exists() or not any(output.iterdir()):
        return outcomes
    claim = {"schema": "kw1-blind-asr-stage-claim-v1", "recognizer": recognizer,
             "job_sha256": sha(job_raw), "identity_map_sha256": sha(canonical(mapping)), "attempt": 1}
    require(_read(output / "stage-claim.json") == claim, "terminal stage identity")
    if not (output / "identity-map.json").exists():
        require(not (output / "cells").exists(), "missing terminal identity map")
        return outcomes
    require(_read(output / "identity-map.json") == mapping, "terminal identity map changed")
    cells = output / "cells"
    if not cells.exists():
        return outcomes
    require(unaliased(cells).is_dir(), "terminal cell directory")
    require({p.name for p in cells.iterdir()} <= {r["opaque_id"] for r in outcomes}, "unexpected terminal cell")
    for outcome in outcomes:
        cell = cells / outcome["opaque_id"]
        if not cell.exists():
            continue
        require(unaliased(cell).is_dir(), "terminal cell path")
        started_path = cell / "started.json"
        if not os.path.lexists(started_path):
            require(not (cell / "completed.json").exists() and not (cell / "failed.json").exists(), "receipt without start")
            continue
        initial = dict(outcome)
        outcome.update(status="FAILED", attempts=1, error_type="interrupted")
        started_raw = read_regular(started_path, MAX_JOB_BYTES)
        try:
            started = decode(started_raw)
        except (ValueError, UnicodeError):
            # Interrupted exclusive write can never become a free retry.
            continue
        completed_load_raw = read_regular(output / "model-load-completed.json", MAX_INPUT_BYTES)
        completed_load = decode(completed_load_raw, MAX_INPUT_BYTES)
        require(type(completed_load) is dict and set(completed_load) == set(claim) | {"receipt"} and
                all(completed_load[key] == value for key, value in claim.items()), "terminal load binding")
        expected = {"schema": "kw1-blind-asr-cell-started-v1", "recognizer": recognizer,
                    "opaque_id": initial["opaque_id"], "pcm_sha256": initial["pcm_sha256"], "wav_sha256": initial["wav_sha256"],
                    "job_sha256": sha(job_raw), "identity_map_sha256": sha(canonical(mapping)),
                    "load_completed_sha256": sha(completed_load_raw), "attempt": 1}
        require(started == expected, "terminal call binding")
        completed_path, failed_path = cell / "completed.json", cell / "failed.json"
        require(not (completed_path.exists() and failed_path.exists()), "conflicting terminal receipts")
        terminal_path = completed_path if completed_path.exists() else failed_path
        if not terminal_path.exists():
            continue
        try:
            terminal = _read(terminal_path)
        except (ValueError, UnicodeError):
            continue
        is_success = terminal_path == completed_path
        expected_fields = {"schema", "started_sha256", "decoder_evidence_sha256", "outcome"} | (set() if is_success else {"reason"})
        require(type(terminal) is dict and set(terminal) == expected_fields and terminal["schema"] ==
                ("kw1-blind-asr-cell-completed-v1" if is_success else "kw1-blind-asr-cell-failed-v1"), "terminal receipt schema")
        require(terminal["started_sha256"] == sha(started_raw), "terminal start hash")
        if terminal["decoder_evidence_sha256"] is not None:
            evidence_raw = read_regular(cell / "decoder-evidence.json", MAX_INPUT_BYTES)
            require(terminal["decoder_evidence_sha256"] == sha(evidence_raw), "terminal decoder evidence hash")
            evidence = decode(evidence_raw, MAX_INPUT_BYTES)
            require(type(evidence) is dict and set(evidence) == {"result", "evidence"} and
                    evidence["result"] == {key: terminal["outcome"][key] for key in ("raw_text", "completeness", "quality_flags")},
                    "terminal decoder result binding")
        else:
            require(not is_success, "missing completed decoder evidence")
        outcome.update(_validate_outcome(terminal["outcome"], initial, "SUCCESS" if is_success else "FAILED"))
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognizer", choices=RECOGNIZERS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.input == Path("/input") and args.output == Path("/output") and args.runtime == Path("/runtime"), "fixed mounted paths required")
    summary = run(args.recognizer, args.input, args.output, args.runtime)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(canonical({"status": "blocked", "reason": error_kind(error)}).decode(), file=sys.stderr)
        raise SystemExit(2)
