"""Serial, one-shot container backend. Importing this file never runs Docker.

All security/resource enforcement delegates to pinned, unchanged retained guards.
Only explicitly listed files enter each read-only code mount. No setup logs,
runtime contents, local paths or exception messages enter returned summaries.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import threading
import time
import types

NEW = 'research/kw1-preset-supplement-v1'
PILOT = 'research/tts-quality-pilot-v1'
SCREEN = 'research/source-screen32-v1'
ASR_LOCK = 'research/tts-pilot-screening-v1/source-lock.json'
BOUNDARY = 'research/tts-pilot-screening-v1/boundary.py'
SCREEN_HELPER = 'research/tts-pilot-screening-v1/screen.py'
IMAGE_LOCK = SCREEN + '/container-lock.json'
GIB = 1024**3
MEMORY = 12*GIB
GLOBAL_SECONDS = 18000
STAGES = ('image', 'setup_qwen06', 'tts_qwen06', 'setup_qwen17', 'tts_qwen17',
          'setup_primary', 'qwen_asr', 'sensevoice', 'setup_whisper', 'whisper_small')
TTS_FILES = tuple('research/qwen6_tts/' + name for name in (
    'setup_locked.py', 'runtime-lock.json', 'model-locks.json',
    'setup_diagnostics.py', 'wheel_identity.py', 'import_preflight.py',
    'runtime_gate.py', 'generate_six.py')) + (
    SCREEN + '/setup_adapter.py', SCREEN + '/runtime_scope.py',
    PILOT + '/setup_adapter.py', PILOT + '/runtime_scope.py', NEW + '/worker.py')
# This fixed set is also checked against the unchanged retained 35-file lock.
ASR_FILES = ('research/qwen6_asr/FUNASR_MODEL_LICENSE', 'research/qwen6_asr/NOTICE.md', 'research/qwen6_asr/core/asr6_contract.py', 'research/qwen6_asr/core/asr_stage/__init__.py', 'research/qwen6_asr/core/asr_stage/adapters.py', 'research/qwen6_asr/core/asr_stage/architecture.py', 'research/qwen6_asr/core/asr_stage/assets.py', 'research/qwen6_asr/core/asr_stage/decoding.py', 'research/qwen6_asr/core/asr_stage/execution.py', 'research/qwen6_asr/core/asr_stage/runtime_support.py', 'research/qwen6_asr/core/pcm/pcm_binding.py', 'research/qwen6_asr/core/runtime_version_gate.py', 'research/qwen6_asr/core/sense_binding.py', 'research/qwen6_asr/core/vendor/__init__.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/__init__.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/checkpoint_contract.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/exact_apply.py', 'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py', 'research/qwen6_asr/import_preflight.py', 'research/qwen6_asr/model-locks.json', 'research/qwen6_asr/runtime-lock.json', 'research/qwen6_asr/sense_parser.py', 'research/qwen6_asr/setup_diagnostics.py', 'research/qwen6_asr/setup_locked.py', 'research/qwen6_asr/wheel_identity.py', 'research/source-screen32-v1/asr_worker.py', 'research/source-screen32-v1/runtime_scope.py', 'research/source-screen32-v1/sense_adapter.py', 'research/source-screen32-v1/setup_adapter.py', 'research/source-screen32-whisper-v1/adapter.py', 'research/source-screen32-whisper-v1/common.py', 'research/source-screen32-whisper-v1/runtime-lock.json', 'research/source-screen32-whisper-v1/source-lock.json', 'research/source-screen32-whisper-v1/setup_runtime.py', 'tools/asr_native_contracts.py')
ASR_NEW = (NEW + '/blind_asr_worker.py', NEW + '/blind_pcm.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def claim(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    sync_dir(Path(path).parent)


def regular_bytes(path, maximum=8*1024**2):
    path = Path(path).absolute()
    require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
            'unaliased canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                before.st_size <= maximum, 'bounded unaliased regular file required')
        with os.fdopen(os.dup(descriptor), 'rb') as stream:
            raw = stream.read(maximum+1)
        after = os.fstat(descriptor)
        entry = path.lstat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        require(stamp(before) == stamp(after) == stamp(entry) and len(raw) == after.st_size,
                'file changed during read')
        return raw
    finally:
        os.close(descriptor)


def document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def dependency_paths():
    """All source paths the caller must pin before constructing this backend."""
    return sorted(set(TTS_FILES + ASR_FILES + ASR_NEW + (BOUNDARY, SCREEN_HELPER,
        ASR_LOCK, IMAGE_LOCK, NEW + '/docker_backend.py', NEW + '/plan.json', PILOT + '/qwen06-model-lock.json',
        PILOT + '/qwen17-model-lock.json')))


def _load(name, path, raw):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class DockerBackend:
    """A fresh work directory is mandatory, including after interruption/failure.

    The caller owns outer campaign admission and the full 16-slot ledger. This
    backend cannot resume or retry and never imports a model on the host.
    Test doubles override _host_capacity, _pull_image and _supervise only.
    """

    def __init__(self, repo, work, deadline, *, source_lock, boundary=None):
        self.repo = Path(repo).absolute()
        self.work = Path(work).absolute()
        require(self.repo.is_dir() and not any(p.is_symlink() for p in (self.repo, *self.repo.parents)),
                'unaliased repository required')
        require(not self.work.exists() and not self.work.is_symlink() and
                not any(p.is_symlink() for p in self.work.parents), 'fresh backend directory required; no resume')
        require(type(deadline) in (int, float) and math.isfinite(deadline), 'finite monotonic deadline required')
        self.deadline = min(deadline, time.monotonic() + GLOBAL_SECONDS)
        self.work.mkdir(parents=True, exist_ok=False)
        claim(self.work/'backend.claim', {'schema': 'kw1-backend-claim-v1', 'attempt': 1})
        self._failed = False
        self._next = 0
        self._mutex = threading.Lock()
        self._summaries = []
        self._runtime_cleanups = []
        st=self.work.stat();self._work_identity=(st.st_dev,st.st_ino)
        self._staged = {}
        self._diagnostic_paths = set()
        self._plan_hash = None
        self._save_summary()
        try:
            raw = regular_bytes(source_lock) if isinstance(source_lock, (str, Path)) else canonical(source_lock)
            lock = document(raw)
            require(isinstance(lock, dict) and isinstance(lock.get('files'), dict), 'source lock files required')
            self.files = dict(lock['files'])
            for name, digest in self.files.items():
                self._safe_name(name)
                require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'source hash required')
            require(set(dependency_paths()) <= self.files.keys(), 'required source pin missing')
            for name in dependency_paths():
                self._source(name)
            retained_lock = document(self._source(ASR_LOCK))
            require(retained_lock.get('schema') == 'tts-pilot-asr-source-lock-v1' and
                    set(retained_lock.get('files', {})) == set(ASR_FILES), 'retained exact ASR allowlist required')
            require(all(self.files[name] == digest for name, digest in retained_lock['files'].items()),
                    'retained ASR pins differ')
            self.image_lock = document(self._source(IMAGE_LOCK))
            require(self.image_lock.get('platform') == 'linux/amd64' and
                    re.fullmatch(r'docker.io/library/python@sha256:[0-9a-f]{64}', self.image_lock.get('image', '')),
                    'pinned official image required')
            self.image = self.image_lock['image']
            if boundary is None:
                # Load pinned stdlib-only host helpers without leaking sys.path or
                # accidentally binding an unrelated generic module named screen.
                screen = _load('_kw1_retained_screen', self.repo/SCREEN_HELPER, self._source(SCREEN_HELPER))
                missing = object()
                old = sys.modules.get('screen', missing)
                try:
                    sys.modules['screen'] = screen
                    self.boundary = _load('_kw1_retained_boundary', self.repo/BOUNDARY, self._source(BOUNDARY))
                finally:
                    if old is missing:
                        sys.modules.pop('screen', None)
                    else:
                        sys.modules['screen'] = old
            else:
                self.boundary = boundary
            self._save_summary()
        except BaseException:
            self._failed = True
            self._save_summary()
            raise

    @staticmethod
    def _safe_name(name):
        require(isinstance(name, str) and name and not name.startswith('/') and
                '\\' not in name and PurePosixPath(name).as_posix() == name and
                all(part not in ('', '.', '..') for part in name.split('/')), 'safe source path required')

    def _source(self, name):
        self._safe_name(name)
        raw = regular_bytes(self.repo/name)
        require(name in self.files and sha(raw) == self.files[name], 'locked source drift')
        return raw

    def _save_summary(self):
        save(self.work/'summary.json', self.public_summary())

    def public_summary(self):
        """Positive-allowlisted evidence only; never raw exceptions/setup output."""
        stages = []
        for row in self._summaries:
            label = row['stage']
            setup = label.startswith('setup_')
            limits = dict(global_wall_seconds=GLOBAL_SECONDS,
                stage_wall_seconds=180 if label == 'image' else (1200 if setup else 4800),
                host_available_memory_min_bytes=14*GIB, host_available_disk_min_bytes=25*GIB)
            if label != 'image':
                limits.update(memory_bytes=MEMORY, memory_swap_bytes=MEMORY, cpus=4, pids=256,
                    readonly_root=True, readonly_code=True, readonly_runtime=not setup,
                    nonroot=True, network='bridge' if setup else 'none')
                if not setup:
                    limits.update(load_wall_seconds=1200, call_wall_seconds=300)
            safe = dict(stage=label, status=row['status'], attempt=1, limits=limits,
                        container_configuration_verified=row.get('container_configuration_verified', False),
                        memory_observation=dict(process_peak_rss_kib=None,cgroup_memory_peak_bytes=None,
                            scope='host supervisor does not sample worker RSS or cgroup peak; configured limits are not measurements',status='unknown'))
            if 'wall_seconds' in row:
                safe['wall_seconds'] = row['wall_seconds']
            for key in ('exit_code', 'oom_killed'):
                if key in row:
                    safe[key] = row[key]
            if 'host' in row:
                safe['host'] = dict(row['host'])
            stages.append(safe)
        return dict(schema='kw1-container-summary-v1',
            status='stopped_no_retry' if self._failed else ('complete' if self._next == len(STAGES) else 'started'),
            stages=stages, runtime_cleanups=list(self._runtime_cleanups), max_wall_seconds=GLOBAL_SECONDS, max_attempts_per_stage=1,
            raw_setup_exported=False, api_tokens_passed=False)

    def _capture_container_state(self, path):
        """Read only typed/fixed fields from the retained bounded diagnostic."""
        if not path.exists():
            return
        try:
            value = document(regular_bytes(path, 65536))
            if not isinstance(value, dict) or value.get('schema') != 'screen32-container-diagnostic-v1':
                return
            row = self._summaries[-1]
            operations = value.get('operations', {})
            if isinstance(operations, dict) and isinstance(operations.get('inspect_configuration'), dict):
                row['container_configuration_verified'] = operations['inspect_configuration'].get('status') == 'complete'
            state = value.get('container_state', {})
            if isinstance(state, dict):
                if type(state.get('ExitCode')) is int and -255 <= state['ExitCode'] <= 255:
                    row['exit_code'] = state['ExitCode']
                if type(state.get('OOMKilled')) is bool:
                    row['oom_killed'] = state['OOMKilled']
        except (OSError, ValueError, TypeError):
            # The unchanged supervisor owns diagnostic write enforcement. This
            # optional safe projection must not replace its primary failure.
            return

    def _host_capacity(self):
        host = self.boundary.host_capacity()
        free = shutil.disk_usage(self.work).free
        require(free >= 25*GIB, '25 GiB disk reserve required before every stage')
        require(type(host.get('available_memory_bytes')) is int and host['available_memory_bytes'] >= 14*GIB,
                '14 GiB available host memory required')
        return dict(available_memory_bytes=host['available_memory_bytes'], available_disk_bytes=free,
                    cpu_count=host['cpu_count'], cgroup_version='2')

    def _pull_image(self):
        return self.boundary.pull_image(self.image_lock, self.work/'image-transfer.json')

    def _supervise(self, args, name, output, stage, deadline):
        return self.boundary.supervise_container(args, name, output, stage, deadline)

    def _stage(self, label, action):
        require(not self._failed and self._next < len(STAGES) and STAGES[self._next] == label,
                'serial stage order; consumed stages cannot restart')
        directory = self.work/'stages'/label
        directory.mkdir(parents=True, exist_ok=False)
        claim(directory/'stage.claim', dict(stage=label, attempt=1))
        record = dict(stage=label, status='started', attempt=1)
        self._summaries.append(record)
        started = time.monotonic()
        try:
            self._save_summary()
            require(started < self.deadline, 'global 18000 second wall limit')
            host = self._host_capacity()
            # Test doubles cannot silently weaken these numeric requirements.
            require(type(host.get('available_memory_bytes')) is int and host['available_memory_bytes'] >= 14*GIB
                    and type(host.get('available_disk_bytes')) is int and host['available_disk_bytes'] >= 25*GIB,
                    'host reserves unavailable')
            record['host'] = {key: host[key] for key in ('available_memory_bytes', 'available_disk_bytes')}
            require(time.monotonic() < self.deadline, 'global wall limit before stage execution')
            result = action(directory)
            require(time.monotonic() <= self.deadline, 'global wall limit exceeded')
            record['status'] = 'complete'
            self._next += 1
            return result
        except BaseException as error:
            self._failed = True
            record['status'] = 'failed_no_retry'
            record['error_type'] = type(error).__name__
            raise
        finally:
            record['wall_seconds'] = max(0, time.monotonic()-started)
            self._save_summary()

    def _stage_code(self, label, names, aliases=None, extra=None):
        code = self.work/'code'/label
        code.mkdir(parents=True, exist_ok=False)
        manifest = []
        aliases = aliases or {}
        extra = extra or {}
        require(not (set(names) & aliases.keys()) and not (set(names) & extra.keys()) and
                not (aliases.keys() & extra.keys()), 'unique staged destinations required')
        sources = {name: (name, self._source(name)) for name in names}
        sources.update({name: (origin, self._source(origin)) for name, origin in aliases.items()})
        sources.update({name: (name, raw) for name, raw in extra.items()})
        for name, (origin, raw) in sorted(sources.items()):
            self._safe_name(name)
            require(sha(raw) == self.files[origin], 'staged source pin mismatch')
            path = code/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            manifest.append(dict(path=name, source_path=origin, sha256=sha(raw), bytes=len(raw)))
        sync_dir(code)
        self._staged[label] = dict(code=code, manifest=manifest)
        save(self.work/('source-manifest-'+label+'.json'), dict(schema='kw1-staged-source-v1', files=manifest))
        self._verify_code(label)
        return code

    def _verify_code(self, label):
        staged = self._staged[label]
        code = staged['code']
        expected = {row['path']: row for row in staged['manifest']}
        observed = set()
        for path in code.rglob('*'):
            require(not path.is_symlink(), 'staged symlink prohibited')
            if path.is_dir():
                continue
            rel = path.relative_to(code).as_posix()
            require(rel in expected, 'unallowlisted staged file')
            raw = regular_bytes(path)
            require(sha(raw) == expected[rel]['sha256'] and len(raw) == expected[rel]['bytes'], 'staged source changed')
            observed.add(rel)
        require(observed == set(expected), 'missing staged source')

    def _launch(self, label, code_label, runtime, output, *, setup=False, engine=None, recognizer=None):
        self._verify_code(code_label)
        code = self._staged[code_label]['code']
        output = Path(output).absolute()
        require(not any(p.is_symlink() for p in (output, *output.parents)) and not output.exists(),
                'fresh inference/setup output required')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        retained_stage = 'setup' if setup else ('tts' if engine else ('qwen06' if recognizer == 'qwen_asr' else 'sensevoice'))
        diagnostic = output.parent/('container-'+retained_stage+'.json')
        require(diagnostic not in self._diagnostic_paths and not diagnostic.exists(), 'unique stage diagnostic parent required')
        self._diagnostic_paths.add(diagnostic)
        name = 'kw1-'+sha(str(self.work).encode())[:12]+'-'+label.replace('_', '-')
        args = self.boundary.container_args(name, self.image, code, runtime, output,
            'tts' if engine else 'asr', retained_stage,
            None if engine else (code if setup else self._input))
        split = args.index(self.image)
        if engine:
            args[split:split] = ['--env', 'TQP_ENGINE='+engine]
        start = args.index(self.image)+1
        if setup and engine:
            command = ['python', '-I', '-B', '/code/'+PILOT+'/setup_adapter.py',
                       '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
        elif setup and recognizer == 'whisper_small':
            command = ['python', '-I', '-B', '/code/research/source-screen32-whisper-v1/setup_runtime.py']
        elif setup:
            command = args[start:]
        elif engine:
            command = ['/runtime/venv/bin/python', '-I', '-B', '/code/'+NEW+'/worker.py',
                '--engine', engine, '--plan', '/code/'+NEW+'/plan.json', '--runtime', '/runtime',
                '--model-lock', '/code/'+PILOT+'/'+engine+'-model-lock.json', '--output', '/output']
        else:
            command = ['/runtime/venv/bin/python', '-I', '-B', '/code/'+NEW+'/blind_asr_worker.py',
                '--recognizer', recognizer, '--input', '/input', '--output', '/output', '--runtime', '/runtime']
        args[start:] = command
        try:
            result = self._supervise(args, name, output, retained_stage,
                min(self.deadline, time.monotonic()+(1200 if setup else 4800)))
        finally:
            self._capture_container_state(diagnostic)
        require(isinstance(result, dict) and result.get('status') == 'complete' and
                result.get('exit_code') == 0, 'container completion required')
        return dict(status='complete', container=dict(memory_bytes=MEMORY, memory_swap_bytes=MEMORY,
            cpus=4, pids=256, nonroot=True, readonly_root=True, readonly_code=True,
            readonly_runtime=not setup, network='bridge' if setup else 'none'),
            source_manifest=[dict(row) for row in self._staged[code_label]['manifest']])

    def _enter(self):
        require(self._mutex.acquire(blocking=False), 'concurrent execution prohibited')
        if self._failed:
            self._mutex.release()
            raise ValueError('backend terminal; retry/resume prohibited')

    def _image(self):
        if self._next == 0:
            self._stage('image', lambda _: self._pull_image())

    def tts(self, engine, plan, output):
        self._enter()
        try:
            require(engine in ('qwen06', 'qwen17'), 'unknown TTS engine')
            self._image()
            def setup(directory):
                raw = self._source(NEW+'/plan.json')
                supplied = plan if isinstance(plan, dict) else document(regular_bytes(plan))
                require(canonical(supplied) == canonical(document(raw)), 'frozen plan drift')
                require(self._plan_hash is None or self._plan_hash == sha(raw), 'plan changed between engines')
                self._plan_hash = sha(raw)
                selected = PILOT+'/'+engine+'-model-lock.json'
                self._stage_code(engine, TTS_FILES+(selected,),
                    aliases={PILOT+'/qwen-model-lock.json': selected}, extra={NEW+'/plan.json': raw})
                runtime = self.work/('runtime-'+engine)
                runtime.mkdir(exist_ok=False)
                return self._launch('setup_'+engine, engine, runtime, directory/'output', setup=True, engine=engine)
            self._stage('setup_'+engine, setup)
            result=self._stage('tts_'+engine, lambda _: self._launch('tts_'+engine, engine,
                self.work/('runtime-'+engine), output, engine=engine))
            self._cleanup_runtime('runtime-'+engine)
            return result
        except BaseException:
            self._failed = True
            self._save_summary()
            raise
        finally:
            self._mutex.release()

    @staticmethod
    def _read_mountinfo():
        with open('/proc/self/mountinfo','rb') as stream:raw=stream.read(2*1024**2+1)
        require(0<len(raw)<=2*1024**2,'bounded mountinfo required')
        return raw.decode('utf-8','strict')

    @staticmethod
    def _mountpoints(raw):
        require(type(raw) is str and raw and len(raw)<=2*1024**2,'mountinfo text required')
        result=[]
        for line in raw.splitlines():
            halves=line.split(' - ');require(len(halves)==2,'mountinfo separator')
            fields=halves[0].split(' ')
            require(len(fields)>=6 and fields[0].isdigit() and fields[1].isdigit() and re.fullmatch(r'[0-9]+:[0-9]+',fields[2]),'mountinfo fields')
            encoded=fields[4]
            def unescape(match):
                value=match.group(1);require(value in ('040','011','012','134'),'mountinfo escape')
                return chr(int(value,8))
            # Reject malformed escapes instead of interpreting a different path.
            require(re.fullmatch(r'(?:[^\\]|\\(?:040|011|012|134))*',encoded),'mountinfo path encoding')
            point=re.sub(r'\\([0-7]{3})',unescape,encoded)
            require(point.startswith('/') and '\x00' not in point and os.path.normpath(point)==point,'canonical mountpoint')
            result.append(Path(point))
        require(result,'mountinfo empty')
        return result

    def _cleanup_runtime(self, name):
        """Remove only this fresh campaign's completed runtime, never evidence.

        Called only after retained supervise_container returned, including its
        cleanup proof. No cleanup is attempted after unknown container teardown.
        """
        allowed={'runtime-qwen06','runtime-qwen17','runtime-primary','runtime-whisper'}
        require(name in allowed,'exact runtime cleanup name required')
        terminal={'runtime-qwen06':'tts_qwen06','runtime-qwen17':'tts_qwen17','runtime-primary':'sensevoice','runtime-whisper':'whisper_small'}[name]
        require(not self._failed and any(r['stage']==terminal and r['status']=='complete' for r in self._summaries),'runtime still needed or container completion unknown')
        path=self.work/name
        require(not any(p.is_symlink() for p in (self.work,*self.work.parents,path)), 'cleanup root symlink')
        st=self.work.stat();require((st.st_dev,st.st_ino)==self._work_identity,'campaign root identity changed')
        require(path.parent==self.work and path.is_dir() and not os.path.ismount(path),'owned runtime directory required')
        require(shutil.rmtree.avoids_symlink_attacks,'fd-safe runtime removal required')
        mounts=self._mountpoints(self._read_mountinfo())
        require(not any(m==path or path in m.parents for m in mounts),'runtime contains mountpoint including same-device bind')
        require(not any(r['runtime']==name for r in self._runtime_cleanups),'runtime cleanup already consumed')
        total=0;seen=set()
        for root,dirs,files in os.walk(path,followlinks=False):
            directory=Path(root);dst=directory.lstat()
            require(dst.st_dev==st.st_dev and not os.path.ismount(directory),'nested runtime mount forbidden')
            for item in files:
                fp=directory/item;fst=fp.lstat()
                if stat.S_ISREG(fst.st_mode) and (fst.st_dev,fst.st_ino) not in seen:
                    total+=fst.st_size;seen.add((fst.st_dev,fst.st_ino))
        record=dict(runtime=name,status='started',logical_regular_bytes=total,
                    scope='only this campaign owned runtime; claims/results/source code retained')
        self._runtime_cleanups.append(record);self._save_summary()
        try:
            shutil.rmtree(path)
            require(not os.path.lexists(path),'runtime removal incomplete')
            record['status']='removed_after_container_cleanup'
        except BaseException:
            record['status']='failed_no_retry';self._failed=True;raise
        finally:self._save_summary()

    def _blind_input(self, source, destination):
        """Reject metadata/path leakage, then snapshot only exact anonymous PCM."""
        source = Path(source).absolute()
        require(source.is_dir() and not any(p.is_symlink() for p in (source, *source.parents)),
                'unaliased blind input directory required')
        raw = regular_bytes(source/'job.json', 32768)
        job = document(raw)
        require(isinstance(job, dict) and set(job) == {'schema', 'clips'} and
                job['schema'] == 'kw1-blind-pcm-job-v1' and isinstance(job['clips'], list) and
                1 <= len(job['clips']) <= 16, 'blind PCM job required')
        names = {'job.json'}
        rows = job['clips']
        keys = {'opaque_id', 'file', 'sha256', 'bytes', 'frames', 'sample_rate_hz', 'channels', 'sample_format'}
        ids = []
        snapshots = {'job.json': raw}
        for row in rows:
            require(isinstance(row, dict) and set(row) == keys, 'blind clip fields required')
            oid = row['opaque_id']
            require(isinstance(oid, str) and re.fullmatch('[0-9a-f]{64}', oid) and
                    row['sha256'] == oid and row['file'] == oid+'.pcm', 'anonymous PCM identity required')
            require(all(type(row[key]) is int for key in ('bytes', 'frames', 'sample_rate_hz', 'channels')) and
                    row['sample_rate_hz'] == 16000 and row['channels'] == 1 and row['sample_format'] == 's16le' and
                    0 < row['frames'] <= 192000 and row['bytes'] == 2*row['frames'], 'PCM geometry required')
            pcm = regular_bytes(source/row['file'], 384000)
            require(len(pcm) == row['bytes'] and sha(pcm) == oid, 'blind PCM content identity')
            ids.append(oid); names.add(row['file']); snapshots[row['file']] = pcm
        require(ids == sorted(set(ids)), 'sorted unique anonymous input required')
        require({path.name for path in source.iterdir()} == names, 'input contains unallowlisted material')
        destination.mkdir(parents=True, exist_ok=False)
        for name, content in snapshots.items():
            with (destination/name).open('xb') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
        sync_dir(destination)
        return sha(raw)

    def asr(self, recognizer, input_dir, output):
        self._enter()
        try:
            require(recognizer in ('qwen_asr', 'sensevoice', 'whisper_small'), 'unknown ASR recognizer')
            require(self._next >= 5, 'both TTS stages must complete before ASR')
            if recognizer == 'qwen_asr':
                def setup(directory):
                    self._stage_code('asr', ASR_FILES+ASR_NEW)
                    self._input = self.work/'blind-input'
                    self._input_hash = self._blind_input(input_dir, self._input)
                    runtime = self.work/'runtime-primary'
                    runtime.mkdir(exist_ok=False)
                    return self._launch('setup_primary', 'asr', runtime, directory/'output', setup=True)
                self._stage('setup_primary', setup)
            elif recognizer == 'whisper_small':
                def setup_whisper(directory):
                    runtime = self.work/'runtime-whisper'
                    runtime.mkdir(exist_ok=False)
                    return self._launch('setup_whisper', 'asr', runtime, directory/'output',
                                        setup=True, recognizer='whisper_small')
                self._stage('setup_whisper', setup_whisper)
            def inference(_):
                # Independently compare all caller/snapshot PCM for each stage;
                # keep a unique per-stage checked snapshot as the mount source.
                checked = self.work/('input-'+recognizer)
                require(self._blind_input(input_dir, checked) == self._input_hash, 'blind cohort changed')
                self._input = checked
                runtime = self.work/('runtime-whisper' if recognizer == 'whisper_small' else 'runtime-primary')
                return self._launch(recognizer, 'asr', runtime, output, recognizer=recognizer)
            result=self._stage(recognizer, inference)
            if recognizer=='sensevoice':self._cleanup_runtime('runtime-primary')
            elif recognizer=='whisper_small':self._cleanup_runtime('runtime-whisper')
            return result
        except BaseException:
            self._failed = True
            self._save_summary()
            raise
        finally:
            self._mutex.release()
