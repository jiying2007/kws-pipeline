"""Fixed, blind K1 ASR18 worker; the checked-in execution release is disabled.

This is a new six-cell wrapper, not an invocation of the consumed campaign.
Only the separately mounted opaque order and audio-only job enter the worker.
Retained loaders, native adapters, model pins and decode controls are unchanged.
No ML import occurs at module import. The public entrypoint has no plugin,
callback, arbitrary module path, input-root or release-path option. Tests mock
this module's fixed dependency symbols; they do not replace sys.modules.
"""
from pathlib import Path
import importlib
import importlib.util
import json
import os
import re
import signal
import stat
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAMILIES = ("qwen06", "sensevoice", "whisper")
MODES = ("prepare", "decode")
NEW = "research/k1-punctuation6-asr18-v1/"
FIXED_MODULES = {
    "retained_worker": ("k1_asr18_retained_worker", "research/source-screen32-v1/asr_worker.py"),
    "native_adapter": ("k1_asr18_native_adapter", "research/source-screen32-v1/native_v2_adapter.py"),
    "whisper_adapter": ("k1_asr18_whisper_adapter", "research/source-screen32-whisper-v1/adapter.py"),
}
FIXED_LOCAL = frozenset(("campaign_contract", "worker_receipts", "cpu_budget", "native_locks",
                         "setup_adapter", "runtime_scope", "whisper_binding", "optional_tools"))
MAX_FILE = 8 * 1024 ** 2
_FROZEN_REQUIRED = ("native-source-lock.json", "native-asset-lock.json", "native-config.json")
_FROZEN_OPTIONAL = ("native-resolved-generation-config.json", "native-frontend-config.json")
_DIAGNOSTIC_CODES = {}
_DIAGNOSTIC_PROBE_CODE = None
_FAILURE_CONTEXT = ("startup", 0)
_FAILURE_FAMILY = "unknown"
_FAILURE_MODE = "unknown"
_NATIVE_CONSTRUCTED = False
_PUBLIC_ENTRY_USED = False
_STAGES = frozenset(("startup", "pre_model_import", "release", "scope", "guard", "inputs",
                    "load_model", "prepare_receipt", "compare_frozen", "bind_predecode",
                    "bind_audio", "start_attempt", "infer", "complete_attempt", "worker_complete"))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path, cap=MAX_FILE):
    path = Path(path)
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "aliased worker path")
    before = path.stat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= cap,
             "bounded unaliased worker file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        _require(_stamp(opened) == _stamp(before), "worker file replaced at open")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        _require(len(raw) == before.st_size and _stamp(os.fstat(fd)) == _stamp(before), "worker file changed")
    finally:
        os.close(fd)
    _require(_stamp(path.stat()) == _stamp(before), "worker file path replaced")
    return raw


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _json(path, cap=MAX_FILE):
    raw = _read(path, cap)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate worker JSON field")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique)
    _require(type(value) is dict and _canonical(value) == raw, "canonical worker JSON object")
    return value


def _local(name):
    """A fixed allowlist, with origin checks before and after ordinary import."""
    _require(name in FIXED_LOCAL, "fixed local dependency only")
    expected = HERE / (name + ".py")
    origins = {item: HERE / (item + ".py") for item in FIXED_LOCAL}
    origins["asr18_metadata"] = HERE / "asr18_metadata.py"
    core = ROOT / "research/qwen6_asr/core"
    origins.update({"asr_stage." + item: core / "asr_stage" / (item + ".py") for item in
                    ("adapters", "architecture", "assets", "decoding", "execution", "runtime_support")})
    origins.update({"asr_stage": core / "asr_stage/__init__.py",
        "pcm.pcm_binding": core / "pcm/pcm_binding.py", "asr6_contract": core / "asr6_contract.py",
        "runtime_version_gate": core / "runtime_version_gate.py", "sense_binding": core / "sense_binding.py",
        "sense_parser": ROOT / "research/qwen6_asr/sense_parser.py",
        "vendor": core / "vendor/__init__.py", "vendor.qwen_loading_gate_v2": core / "vendor/qwen_loading_gate_v2.py",
        "vendor.checkpoint_guard": core / "vendor/checkpoint_guard/__init__.py",
        "vendor.checkpoint_guard.exact_apply": core / "vendor/checkpoint_guard/exact_apply.py",
        "vendor.checkpoint_guard.checkpoint_contract": core / "vendor/checkpoint_guard/checkpoint_contract.py",
        "common": ROOT / "research/source-screen32-whisper-v1/common.py",
        "setup_runtime": ROOT / "research/source-screen32-whisper-v1/setup_runtime.py"})
    origins.update({item: ROOT / "tools" / (item + ".py") for item in
                    ("asr_native_contracts", "speech_asr_consensus", "speech_label_admission")})
    for module_name, origin in origins.items():
        cached = sys.modules.get(module_name)
        _require(cached is None or (type(cached) is types.ModuleType and
                 vars(cached).get("__file__") == str(origin)), "foreign fixed-source module cache")
    loaded = sys.modules.get(name)
    _require(loaded is None or (type(loaded) is types.ModuleType and
             vars(loaded).get("__file__") == str(expected)), "foreign local-module cache")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    loaded = importlib.import_module(name)
    _require(vars(loaded).get("__file__") == str(expected), "local dependency origin drift")
    _register_module(loaded, NEW + name + ".py")
    return loaded


def _production_gate():
    """No model, runtime, native helper or audio work before the fixed release."""
    contract = _local("campaign_contract")
    release = _json(HERE / "execution-release.json", 64 * 1024)
    # The host supplies this independently approved identity in its fixed,
    # allowlisted worker environment; hashing a candidate here is not authority.
    freeze = _json(Path("/authority/execution-freeze.json")) if release.get("enabled") is True else None
    admitted = contract.require_execution_release(release, freeze,
        expected_freeze_sha256=os.environ.get("KWS_SOURCE_FREEZE_SHA256"))
    _verify_closed_sources(freeze)
    _require({p.name for p in Path("/authority").iterdir()} ==
             {"execution-freeze.json", "source-claim.json"}, "fixed public worker authority roster")
    claim_raw = contract.read_bytes(Path("/authority/source-claim.json"), cap=65536, readonly=True)
    contract.validate_source_claim(claim_raw,
        expected_sha256=os.environ.get("KWS_SOURCE_CLAIM_SHA256"), release=admitted,
        source_freeze_sha256=os.environ.get("KWS_SOURCE_FREEZE_SHA256"),
        run_id=os.environ.get("KWS_RUN_ID", ""), executed_head_sha=os.environ.get("KWS_EXECUTED_HEAD_SHA", ""))
    return admitted


def _verify_closed_sources(freeze):
    """The mounted /code is exactly the independently frozen inference roster."""
    _require(ROOT == Path("/code") and sys.flags.isolated == 1 and sys.dont_write_bytecode,
             "read-only isolated -I -B /code worker required")
    contract = _local("campaign_contract")
    expected = freeze.get("inference_code")
    _require(type(expected) is dict and bool(expected), "frozen inference source roster")
    observed = set()
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        _require(not any((Path(base) / name).is_symlink() for name in dirs + files), "source closure alias")
        for name in files:
            path = Path(base) / name
            observed.add(path.relative_to(ROOT).as_posix())
    # The freeze is carried separately, not recursively included in its own tree.
    _require(observed == set(expected), "extra or missing inference source member")
    for name, identity in expected.items():
        _require(type(name) is str and not name.startswith("/") and
                 all(p not in ("", ".", "..") for p in name.split("/")), "source relative path")
        _require(bool(os.statvfs(ROOT / name).f_flag & os.ST_RDONLY), "read-only source mount required")
        body = _read(ROOT / name)
        _require(identity == {"bytes": len(body), "sha256": contract.digest(body)}, "frozen source identity drift")
    _require(not any(name.startswith("research/qwen12-native-admission-v1/") for name in observed),
             "old campaign closure is never mounted")


def _fixed_module(key):
    _production_gate()  # Direct calls cannot bypass the checked-in disabled gate.
    _require(key in FIXED_MODULES, "fixed native dependency only")
    name, relative = FIXED_MODULES[key]
    path = ROOT / relative
    _require(name not in sys.modules, "native wrapper must be a fresh module")
    if key == "whisper_adapter":
        common = sys.modules.get("common")
        _require(common is None or vars(common).get("__file__") == str(path.parent / "common.py"),
                 "foreign Whisper common module")
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "fixed source loader")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    _require(vars(loaded).get("__file__") == str(path), "native dependency origin drift")
    _register_module(loaded, relative)
    return loaded


def _register_module(module, relative):
    pending = [value.__code__ for value in vars(module).values() if type(value) is types.FunctionType]
    for _ in range(2048):
        if not pending or len(_DIAGNOSTIC_CODES) >= 2048:
            break
        code = pending.pop()
        if code.co_filename != str(ROOT / relative) or id(code) in _DIAGNOSTIC_CODES:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*|<(?:lambda|listcomp|dictcomp|setcomp|genexpr)>", code.co_name):
            continue
        _DIAGNOSTIC_CODES[id(code)] = (code, relative, code.co_name)
        pending.extend(value for value in code.co_consts if type(value) is types.CodeType)


def _register_diagnostic_codes():
    # Compatibility for the unchanged setup helper: observations, no imports.
    current = sys.modules.get(__name__)
    if type(current) is types.ModuleType:
        _register_module(current, NEW + "native_worker.py")
    for name in FIXED_LOCAL:
        module = sys.modules.get(name)
        if type(module) is types.ModuleType and vars(module).get("__file__") == str(HERE / (name + ".py")):
            _register_module(module, NEW + name + ".py")


def register_optional_probe_code(code):
    global _DIAGNOSTIC_PROBE_CODE
    if type(code) is types.CodeType and code.co_name == "is_ffmpeg_installed":
        _DIAGNOSTIC_PROBE_CODE = code


def _diagnostic_stage(stage, index=0):
    global _FAILURE_CONTEXT
    _FAILURE_CONTEXT = (stage, index)


def _safe_error_type(error):
    known = (ValueError, TypeError, OSError, PermissionError, FileNotFoundError, RuntimeError,
             TimeoutError, ImportError, ModuleNotFoundError, MemoryError, KeyError, IndexError,
             AttributeError, AssertionError, RecursionError, KeyboardInterrupt, SystemExit)
    return next((kind.__name__ for kind in known if type(error) is kind), "OtherException")


def _failure_diagnostic(error):
    stage, index = _FAILURE_CONTEXT
    result = {"schema": "k1-asr18-bounded-failure-v1", "stage": stage if stage in _STAGES else "unknown",
              "index": index if type(index) is int and 0 <= index <= 6 else None,
              "family": _FAILURE_FAMILY if _FAILURE_FAMILY in FAMILIES else "unknown",
              "mode": _FAILURE_MODE if _FAILURE_MODE in MODES else "unknown", "frames": [],
              "frames_truncated": False, "chain_followed": False, "message_disclosed": False,
              "source_lines_disclosed": False, "locals_disclosed": False}
    trace = BaseException.__traceback__.__get__(error)
    for _ in range(64):
        if trace is None:
            break
        code = trace.tb_frame.f_code
        known = _DIAGNOSTIC_CODES.get(id(code))
        frame = {"file": "<redacted>", "function": "<redacted>", "line": None}
        if known is not None and known[0] is code:
            frame = {"file": known[1], "function": known[2], "line": trace.tb_lineno}
        elif _DIAGNOSTIC_PROBE_CODE is not None and code == _DIAGNOSTIC_PROBE_CODE:
            frame = {"file": "runtime/site-packages/funasr/utils/load_utils.py", "function": "is_ffmpeg_installed", "line": trace.tb_lineno}
        result["frames"].append(frame)
        if len(result["frames"]) > 16:
            del result["frames"][0]
            result["frames_truncated"] = True
        trace = trace.tb_next
    result["frames_truncated"] |= trace is not None
    _require(len(_canonical(result)) <= 8192, "bounded diagnostic")
    return result


def _alarm(_signal, _frame):
    raise TimeoutError("clip bound")


def _native_call(session, bound, directory):
    """Exactly one fixed adapter call; timer cleanup failure is terminal too."""
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        return session.infer(bound, directory)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _compare_frozen(output, frozen):
    for name in _FROZEN_REQUIRED:
        _require(_read(output / name) == _read(frozen / name), "fresh native lock/config drift")
    for name in _FROZEN_OPTIONAL:
        _require((output / name).exists() == (frozen / name).exists(), "native optional config roster drift")
        if (output / name).exists():
            _require(_read(output / name) == _read(frozen / name), "fresh native dependency config drift")


class _QwenSenseSession:
    def __init__(self, worker, adapter, inputs, decoder_raw, contract):
        self.worker, self.adapter, self.inputs = worker, adapter, inputs
        self.decoder_raw, self.contract = decoder_raw, contract
        self.bindings = worker.validate_decoder(decoder_raw, contract.digest(decoder_raw))

    def bind(self, clip):
        self.worker.verify_scope()
        descriptor = self.bindings[clip["audio_id"]]["descriptor"]
        bound = self.worker.bind_wave(str(self.inputs / "audio"), self.worker.WaveExpectation(
            clip["audio_id"], clip["wav_sha256"] + ".wav", clip["wav_sha256"],
            descriptor["wav"]["bytes"], descriptor["raw_pcm16"]["frame_count"]))
        _require(bound.descriptor == descriptor, "full bound PCM unchanged")
        return bound

    def infer(self, bound, directory):
        return self.adapter.infer(bound, self.contract.digest(self.decoder_raw), {})


class _WhisperSession:
    def __init__(self, recognizer, common, inputs):
        self.recognizer, self.common, self.inputs = recognizer, common, inputs

    def bind(self, clip):
        _local("runtime_scope").verify_runtime_scope()
        return self.common.wav_bytes(self.inputs / clip["audio_path"], clip["wav_sha256"])

    def infer(self, bound, directory):
        pcm, frames = bound
        return None, self.recognizer.decode(pcm, frames, directory)


def _construct_native(family, runtime, inputs, output, job, run_id, head):
    """Only these audited model loaders and native adapters are selectable."""
    global _NATIVE_CONSTRUCTED
    _production_gate()
    _require(not _NATIVE_CONSTRUCTED, "fresh process for exactly one native model load")
    _NATIVE_CONSTRUCTED = True
    _require(runtime == Path("/runtime") and inputs == Path("/input") and output == Path("/output"),
             "fixed production model/input/output mounts")
    contract = _local("campaign_contract")
    locks = _local("native_locks")
    if family == "whisper":
        adapter = _fixed_module("whisper_adapter")
        base = ROOT / "research/source-screen32-whisper-v1"
        source_raw, asset_raw = locks.build_whisper_locks(runtime, output, code_root=ROOT,
            official_source_lock=contract.read_json(base / "source-lock.json"),
            verified_asset_lock=contract.read_json(runtime / "asset-lock.json"), executed_head_sha=head, run_id=run_id)
        recognizer = _local("whisper_binding").construct_whisper(adapter, runtime, output, source_raw, asset_raw)
        common = sys.modules.get("common")
        _require(type(common) is types.ModuleType and vars(common).get("__file__") == str(base / "common.py"),
                 "fixed Whisper common origin")
        return _WhisperSession(recognizer, common, inputs)
    _require(family in ("qwen06", "sensevoice"), "fixed native family")
    worker = _fixed_module("retained_worker")
    worker.verify_scope()
    os.environ.update(worker.THREAD_ENV)
    os.environ.update(worker.OFFLINE_ENV)
    worker.prepare_caches()
    worker.verify_sources()
    worker.import_helpers()
    _local("campaign_contract")  # Reject every foreign cached retained-helper origin.
    _, job_raw, decoder_raw = worker.bind_inputs(inputs)
    _require(job_raw == contract.canonical(job), "audio input differs from frozen blind job")
    verified = _local("setup_adapter").verify_runtime(runtime, "asr")
    runtime_lock = contract.read_json(worker.RETAINED / "runtime-lock.json")
    extension = locks.verify_installed_extension_sources(runtime, runtime_lock) if family == "qwen06" else None
    lock = worker.decode(worker.read_regular(worker.RETAINED / "model-locks.json", 65536))[family]
    declaration = worker.execution_contract(family, lock, decoder_raw)
    contract_sha = contract.write_once(output / "contract.json", declaration)
    contract.write_once(output / "decoder-inputs.json", worker.decode(decoder_raw, 1024 ** 2))
    loader = worker.load_qwen_after_approval if family == "qwen06" else worker.load_sense_after_approval
    model_root = runtime / "models" / family
    args = (str(model_root), lock["asset_lock"], lock["asset_lock_sha256"], lock["source_lock"],
            lock["source_lock_sha256"], declaration, contract_sha, decoder_raw)
    execution_policy = None
    if family == "sensevoice":
        optional = _local("optional_tools")
        with optional.unavailable_ffmpeg(runtime, contract.read_json(output / "cpu-process-guard.json"),
                lambda value: contract.write_once(output / "optional-tool-probes.json", value),
                register_code=register_optional_probe_code):
            loaded = loader(*args)
        execution_policy = contract.canonical(optional.policy_config())
    else:
        loaded = loader(*args)
    contract.write_once(output / "model-load.json", loaded.receipt)
    native = _fixed_module("native_adapter")
    frontend = None
    if family == "sensevoice":
        contract.write_once(output / "native-frontend-config.json", native.sense_effective(loaded, model_root)[1])
        frontend = _read(output / "native-frontend-config.json")
    source_raw, asset_raw = locks.build_native_locks(runtime, lock, output, family=family, code_root=ROOT,
        loaded_receipt=loaded.receipt, verified_runtime=verified, executed_head_sha=head, run_id=run_id,
        installed_source_inventory=extension, frontend_config_bytes=frontend, execution_policy_bytes=execution_policy)
    adapter = native.NativeCaptureAdapter(loaded, model_root, source_raw, asset_raw,
        lambda name, value: contract.write_once(output / name, value), frontend_config_bytes=frontend)
    return _QwenSenseSession(worker, adapter, inputs, decoder_raw, contract)


def _execute_family(family, mode, *, runtime, inputs, output, frozen, policy_path,
                    job, execution_plan, input_binding, run_id, head, policy_sha256):
    """Internal orchestration; public production API never accepts dependencies."""
    _require(family in FAMILIES and mode in MODES, "fixed family/mode")
    contract, receipts = _local("campaign_contract"), _local("worker_receipts")
    clips = contract.planned_clips(job, execution_plan, family)
    contract.validate_input_binding(input_binding, job, execution_plan)
    _require(bool(clips), "zero eligible cells forbids model load")
    _diagnostic_stage("load_model")
    session = _construct_native(family, runtime, inputs, output, job, run_id, head)
    if mode == "prepare":
        _require(not any(output.glob("clip-*")), "prepare cannot contain native attempts")
        _diagnostic_stage("prepare_receipt")
        result = {"schema": "k1-asr18-worker-result-v1", "family": family, "mode": mode,
                  "status": "prepared_no_recognition", "audio_forward_calls": 0, "completed": 0,
                  "audio_ids": [], "job_sha256": contract.digest(contract.canonical(job)),
                  "execution_plan_sha256": contract.digest(contract.canonical(execution_plan))}
        contract.write_once(output / "prepared.json", result)
        return result
    _diagnostic_stage("compare_frozen")
    _compare_frozen(output, frozen)
    _diagnostic_stage("bind_predecode")
    binding = receipts.bind_predecode(recognizer_id=family, policy_path=policy_path,
        expected_policy_sha256=policy_sha256, frozen_dir=frozen, current_dir=output,
        run_id=run_id, executed_head_sha=head, job=job, execution_plan=execution_plan, input_binding=input_binding)
    completed = []
    for position, clip in enumerate(clips, 1):
        _diagnostic_stage("bind_audio", position)
        bound = session.bind(clip)
        directory = output / clip["audio_id"]
        _diagnostic_stage("start_attempt", position)
        # emit_start exclusively creates and fsyncs the directory and receipt.
        # Any write/fsync uncertainty is terminal; never recreate an identity.
        attempt = receipts.emit_start(binding, directory, wav_path=inputs / clip["audio_path"],
            expected_file_sha256=clip["wav_sha256"], audio_id=clip["audio_id"],
            request_id="k1asr18-" + run_id + "-" + family + "-" + clip["audio_id"])
        try:
            receipts.mark_native_enter(attempt)
            _diagnostic_stage("infer", position)
            row, raw = _native_call(session, bound, directory)
            receipts.mark_native_return(attempt)
            _diagnostic_stage("complete_attempt", position)
            receipts.emit_completion(attempt, row=row, raw=raw)
        except BaseException:
            # Cleanup is conservative too: neither cleanup failure nor timeout
            # creates attempt credit, and no later eligible clip is reached.
            receipts.fail_attempt(attempt)
            raise
        completed.append(clip["audio_id"])
    return {"schema": "k1-asr18-worker-result-v1", "family": family, "mode": mode,
            "status": "all_planned_native_complete", "audio_forward_calls": len(completed),
            "completed": len(completed), "audio_ids": completed,
            "job_sha256": contract.digest(contract.canonical(job)),
            "execution_plan_sha256": contract.digest(contract.canonical(execution_plan))}


def _validate_preload_audio(inputs, job, execution_plan, input_binding):
    """Remeasure every eligible anonymous WAV/PCM before any model construction."""
    contract, receipts = _local("campaign_contract"), _local("worker_receipts")
    by_id = contract.validate_blind_job(job, execution_plan)
    contract.validate_input_binding(input_binding, job, execution_plan)
    for row in input_binding["audio"]:
        clip = by_id[row["audio_id"]]
        measured = receipts._wave(inputs / clip["audio_path"], clip["wav_sha256"])
        _require(measured == {"file_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"],
                              "audio_frames": row["audio_frames"]}, "preload actual WAV/PCM/frame authority drift")


def run_family(family, mode):
    """Fixed production entrypoint. Current disabled release stops immediately."""
    global _FAILURE_FAMILY, _FAILURE_MODE, _PUBLIC_ENTRY_USED
    _require(family in FAMILIES and mode in MODES, "fixed family/mode")
    _FAILURE_FAMILY, _FAILURE_MODE = family, mode
    _diagnostic_stage("release")
    release = _production_gate()
    _require(not _PUBLIC_ENTRY_USED, "one isolated family/mode per fresh worker process")
    _PUBLIC_ENTRY_USED = True
    contract = _local("campaign_contract")
    freeze = _json(Path("/authority/execution-freeze.json"))
    _verify_closed_sources(freeze)
    _diagnostic_stage("scope")
    scope = _local("runtime_scope").verify_runtime_scope()
    output, inputs = Path("/output"), Path("/input")
    _require(output.is_dir() and not output.is_symlink() and not list(output.iterdir()), "fresh output directory")
    _diagnostic_stage("inputs")
    job = _json(inputs / "job.json", 65536)
    execution_plan = _json(Path("/order/execution-plan.json"), 65536)
    input_binding = _json(Path("/order/worker-input-binding.json"), 65536)
    _require({p.name for p in Path("/order").iterdir()} ==
             {"execution-plan.json", "worker-input-binding.json"}, "only anonymous order and binding mounted")
    for name in ("/input", "/order", "/authority", "/runtime") + (("/policy", "/frozen") if mode == "decode" else ()):
        _require(not Path(name).is_symlink() and bool(os.statvfs(name).f_flag & os.ST_RDONLY),
                 "read-only admitted worker mount required")
    contract.validate_blind_job(job, execution_plan)
    _require(bool(job["clips"]), "zero eligible cells forbids model load")
    _require({p.name for p in inputs.iterdir()} == {"job.json", "audio"}, "audio-only input root")
    _require({p.name for p in (inputs / "audio").iterdir()} ==
             {r["wav_sha256"] + ".wav" for r in job["clips"]}, "exact eligible audio roster")
    # Root-authorized bindings and the global one-use claim are checked before
    # every model load, not merely at the first native audio invocation.
    contract.validate_worker_admission(release, input_binding, job, execution_plan)
    _validate_preload_audio(inputs, job, execution_plan, input_binding)
    if mode == "decode":
        contract.validate_decode_proof(_json(Path("/policy/budget-proof.json")),
            job=job, execution_plan=execution_plan, input_binding=input_binding,
            policy_path=Path("/policy/policy.json"), family=family)
    _diagnostic_stage("guard")
    guard = _local("cpu_budget").install_inference_guard(family)
    run_id, head = os.environ.get("KWS_RUN_ID", ""), os.environ.get("KWS_EXECUTED_HEAD_SHA", "")
    _require(re.fullmatch(r"[1-9][0-9]{0,19}", run_id) and re.fullmatch(r"[0-9a-f]{40}", head), "actual run/source identity")
    contract.write_once(output / "kernel-scope.json", scope)
    contract.write_once(output / "cpu-process-guard.json", guard)
    contract.write_once(output / "blind-job.json", job)
    contract.write_once(output / "execution-plan.json", execution_plan)
    signal.signal(signal.SIGALRM, _alarm)
    result = _execute_family(family, mode, runtime=Path("/runtime"), inputs=inputs, output=output,
        frozen=Path("/frozen"), policy_path=Path("/policy/policy.json"), job=job, execution_plan=execution_plan,
        input_binding=input_binding, run_id=run_id, head=head, policy_sha256=os.environ.get("KWS_POLICY_SHA256", ""))
    _diagnostic_stage("worker_complete")
    contract.write_once(output / "worker-complete.json", result)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    _require(type(argv) is list and len(argv) == 2, "only family and mode arguments")
    return run_family(argv[0], argv[1])


def run():
    try:
        main()
        return 0
    except BaseException as error:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        try:
            _local("campaign_contract").write_once(Path("/output/worker-failure.json"),
                {"status": "failed_no_retry", "error_type": _safe_error_type(error),
                 "diagnostic": _failure_diagnostic(error)})
        except BaseException:
            pass
        return 1


_register_diagnostic_codes()
if __name__ == "__main__":
    raise SystemExit(run())
