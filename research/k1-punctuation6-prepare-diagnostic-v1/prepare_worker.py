"""Fixed constructor/config diagnostic. No mode, decode loop, WAV or ASR claim.

Imports unchanged retained modules and selects constructors only. The runtime
contains native inference libraries, so zero-forward confidence is specifically
bounded by the closed reviewed call graph and pre-call guard, not a counter literal.
"""
from pathlib import Path
import _thread
import os
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare_contract as c

_MODEL_STARTED = False
_FORBIDDEN_APPLICATION = frozenset(('bind_inputs', '_validate_preload_audio', '_native_call',
    'run_family', 'emit_start', 'mark_native_enter', 'infer_qwen_once_after_approval',
    'infer_sense_once_after_approval', '_qwen', '_sense', 'infer'))


class NoForward:
    """Fail before reviewed Python inference entry bodies; no tensor contents logged.

    Does not independently prove arbitrary native C-extension behavior. The
    independently reviewed source closure must exclude implicit native warmup.
    """
    def __init__(self):
        self.attempts = 0
        self.started = False
        self.completed = False
        self.bound_profile = self.profile
        self.old_main = None
        self.old_thread = None
        self._exiting = False
        self._exit_code = type(self).__exit__.__code__
        self._setprofile_builtin = sys.setprofile
        self._thread_builtins = tuple(getattr(_thread, name) for name in
            ('start_new_thread', 'start_new', 'start_joinable_thread') if hasattr(_thread, name))
        self._thread_start_code = threading.Thread.start.__code__
        self._python_api_codes = (threading.setprofile.__code__,)
        if hasattr(threading, 'setprofile_all_threads'):
            self._python_api_codes += (threading.setprofile_all_threads.__code__,)

    def _hardstop(self):
        self.attempts += 1
        # Exceptions in profile callbacks clear the hook and may be caught.
        # The fixed reviewed runtime gets no recovery path after this boundary.
        try:
            os.write(1, b'{"schema":"k1-prepare-native-entry-blocked-v1","status":"stopped_before_forbidden_native_body","exit_code":91}\n')
        finally:
            os._exit(91)

    def profile(self, frame, event, arg):
        if event == 'c_call':
            if arg is self._setprofile_builtin:
                # The fixed source-controlled exit is the ONLY permitted API
                # modification, including calls through a pre-saved builtin.
                own_exit = (self._exiting is True and frame.f_code is self._exit_code
                    and frame.f_locals.get('self') is self)
                if not own_exit:
                    self._hardstop()
            elif any(arg is function for function in self._thread_builtins):
                # Standard Thread bootstrap inherits threading.setprofile before
                # its target runs. Direct low-level starts do not provide this
                # boundary, and are outside this fixed-source diagnostic scope.
                inherited = (frame.f_code is self._thread_start_code and
                    threading.getprofile() is self.bound_profile)
                if not inherited:
                    self._hardstop()
            return
        if event != 'call':
            return
        if any(frame.f_code is code for code in self._python_api_codes):
            self._hardstop()
        name = frame.f_code.co_name
        module = frame.f_globals.get('__name__', '')
        filename = frame.f_code.co_filename
        if type(module) is not str:
            self._hardstop()
        application = filename.startswith('/code/') and name in _FORBIDDEN_APPLICATION
        backend = module.startswith(('qwen_asr.', 'transformers.', 'funasr.', 'torch.nn.'))
        # Tokenizer.transcribe is an integer cached_property and Tokenizer.decode
        # is text-only. Neither is a neural entrypoint. Match locked Whisper
        # neural modules rather than names across the entire whisper package.
        whisper_neural = ((module == 'whisper.transcribe' and name == 'transcribe') or
            (module == 'whisper.decoding' and name in ('decode', 'run', '_main_loop', '_get_audio_features')) or
            (module == 'whisper.model' and name == 'forward') or
            (filename.endswith('/source-screen32-whisper-v1/adapter.py') and
                frame.f_code.co_qualname == 'FixedWhisper.decode'))
        if application or whisper_neural or (backend and name in ('forward', 'generate', 'inference', 'infer',
                'transcribe', '_call_impl', '_wrapped_call_impl')):
            self._hardstop()

    def __enter__(self):
        self.old_main, self.old_thread = sys.getprofile(), threading.getprofile()
        c.require(self.old_main is None and self.old_thread is None, 'unexpected_profile_hook')
        self.started = True
        threading.setprofile(self.bound_profile)
        sys.setprofile(self.bound_profile)
        return self

    def __exit__(self, kind, value, tb):
        intact = sys.getprofile() is self.bound_profile and threading.getprofile() is self.bound_profile
        self._exiting = True
        try:
            sys.setprofile(self.old_main)
            threading.setprofile(self.old_thread)
        finally:
            self._exiting = False
        if kind is None:
            c.require(intact and self.attempts == 0, 'prepare_guard_lifetime_or_attempt')
            self.completed = True
        return False

    def receipt(self):
        c.require(self.started and self.completed and self.attempts == 0, 'guard_not_completed')
        return {'schema': 'k1-prepare-forward-guard-v1', 'forbidden_attempts': self.attempts,
            'python_entry_guard': 'active_through_constructor_and_config',
            'coverage': 'reviewed_fixed_python_native_and_module_forward_entrypoints',
            'universal_native_extension_proof': False}


def _shared(name):
    return c.fixed_module(c.OLD + name + '.py', name)


def construct_family(family, files, packet, run_id, head):
    """Exactly one original constructor path; no callback or alternate dependency."""
    global _MODEL_STARTED
    c.release_first()  # Independent second gate, before every constructor import.
    _, _, admitted_claim, admitted_files, admitted_packet = c.worker_admission(family)
    c.require(files == admitted_files and packet == admitted_packet and
        run_id == str(admitted_claim['identity']['run_id']) and
        head == admitted_claim['identity']['executed_head_sha'], 'constructor_fresh_authority_binding')
    c.require(family in c.FAMILIES and not _MODEL_STARTED, 'fresh_single_model_process')
    c.require(Path('/runtime').is_dir() and Path('/output').is_dir() and not Path('/input').exists(),
              'fixed_audio_absent_runtime')
    _MODEL_STARTED = True
    runtime, output = Path('/runtime'), Path('/output')
    contract, locks = _shared('campaign_contract'), _shared('native_locks')
    if family == 'whisper':
        adapter = c.fixed_module('research/source-screen32-whisper-v1/adapter.py', 'prepare_whisper_adapter')
        base = c.ROOT / 'research/source-screen32-whisper-v1'
        source_raw, asset_raw = locks.build_whisper_locks(runtime, output, code_root=c.ROOT,
            official_source_lock=contract.read_json(base / 'source-lock.json'),
            verified_asset_lock=contract.read_json(runtime / 'asset-lock.json'),
            executed_head_sha=head, run_id=run_id)
        _shared('whisper_binding').construct_whisper(adapter, runtime, output, source_raw, asset_raw)
        return
    worker = c.fixed_module('research/source-screen32-v1/asr_worker.py', 'prepare_retained_worker')
    worker.verify_scope()
    os.environ.update(worker.THREAD_ENV)
    os.environ.update(worker.OFFLINE_ENV)
    worker.prepare_caches()
    worker.verify_sources()
    worker.import_helpers()
    # Actual descriptor packet was measured on the host. Never bind_inputs here.
    decoder_raw = files['decoder-inputs.json']
    worker.validate_decoder(decoder_raw, contract.digest(decoder_raw))
    verified = _shared('setup_adapter').verify_runtime(runtime, 'asr')
    runtime_lock = contract.read_json(worker.RETAINED / 'runtime-lock.json')
    extension = locks.verify_installed_extension_sources(runtime, runtime_lock) if family == 'qwen06' else None
    lock = worker.decode(worker.read_regular(worker.RETAINED / 'model-locks.json', 65536))[family]
    declaration = worker.execution_contract(family, lock, decoder_raw)
    # This unchanged object is loader compatibility metadata, NOT a grant to decode.
    declaration_sha = contract.write_once(output / 'contract.json', declaration)
    contract.write_once(output / 'decoder-inputs.json', worker.decode(decoder_raw, c.MIB))
    loader = worker.load_qwen_after_approval if family == 'qwen06' else worker.load_sense_after_approval
    model_root = runtime / 'models' / family
    args = (str(model_root), lock['asset_lock'], lock['asset_lock_sha256'], lock['source_lock'],
        lock['source_lock_sha256'], declaration, declaration_sha, decoder_raw)
    execution_policy = None
    if family == 'sensevoice':
        optional = _shared('optional_tools')
        with optional.unavailable_ffmpeg(runtime, contract.read_json(output / 'cpu-process-guard.json'),
                lambda value: contract.write_once(output / 'optional-tool-probes.json', value)):
            loaded = loader(*args)
        execution_policy = contract.canonical(optional.policy_config())
    else:
        loaded = loader(*args)
    contract.write_once(output / 'model-load.json', loaded.receipt)
    native = c.fixed_module('research/source-screen32-v1/native_v2_adapter.py', 'prepare_native_adapter')
    frontend = None
    if family == 'sensevoice':
        contract.write_once(output / 'native-frontend-config.json', native.sense_effective(loaded, model_root)[1])
        frontend = contract.read_bytes(output / 'native-frontend-config.json')
    source_raw, asset_raw = locks.build_native_locks(runtime, lock, output, family=family,
        code_root=c.ROOT, loaded_receipt=loaded.receipt, verified_runtime=verified,
        executed_head_sha=head, run_id=run_id, installed_source_inventory=extension,
        frontend_config_bytes=frontend, execution_policy_bytes=execution_policy)
    native.NativeCaptureAdapter(loaded, model_root, source_raw, asset_raw,
        lambda name, value: contract.write_once(output / name, value), frontend_config_bytes=frontend)


def main(argv=None):
    c.release_first()  # Reject before args, scope, packet imports or model modules.
    argv = sys.argv[1:] if argv is None else argv
    c.require(type(argv) is list and len(argv) == 1 and argv[0] in c.FAMILIES,
              'only_one_fixed_prepare_family_no_mode')
    family = argv[0]
    release, freeze, claim, files, packet = c.worker_admission(family)
    output = Path('/output')
    c.require(output.is_dir() and not output.is_symlink() and not list(output.iterdir()),
              'fresh_prepare_output')
    scope = _shared('runtime_scope').verify_runtime_scope()
    process_guard = _shared('cpu_budget').install_inference_guard(family)
    contract = _shared('campaign_contract')
    contract.write_once(output / 'kernel-scope.json', scope)
    contract.write_once(output / 'cpu-process-guard.json', process_guard)
    started = time.monotonic()
    def alarm(_signum, _frame):
        raise TimeoutError('prepare_constructor_wall_limit')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 600)
    guard = NoForward()
    try:
        with guard:
            construct_family(family, files, packet, str(claim['identity']['run_id']),
                claim['identity']['executed_head_sha'])
        # No clip files, native starts, completed attempts or transcripts are emitted.
        c.require(not any(p.name.startswith('clip-') for p in output.iterdir()), 'native_attempt_output_prohibited')
        result = {'schema': 'k1-prepare-worker-result-v1', 'identity': claim['identity'],
            'family': family, 'status': 'prepared_configuration_only', 'model_loads_started': 1,
            'model_loads_completed': 1, 'maximum_new_asr_calls': 0, 'native_decode_attempts': 0,
            'audio_forward_calls': 0, 'audio_mounted': False, 'recognition_result': False,
            'legacy_execution_contract_grants_asr': False,
            'metadata_packet_sha256': claim['metadata_packet_sha256'],
            'source_freeze_sha256': claim['source_freeze_sha256'], 'guard': guard.receipt(),
            'wall_seconds': time.monotonic() - started}
        contract.write_once(output / 'prepare-receipt.json', result)
        contract.write_once(output / 'worker-complete.json', result)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # Parent supervisor retains bounded stage/exit metadata. Never dump error strings.
        print('{"schema":"k1-prepare-worker-stop-v1","status":"failed_no_retry"}', flush=True)
        raise SystemExit(2)
