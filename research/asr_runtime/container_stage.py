#!/usr/bin/env python3
"""Container-only stages. No model loading, user audio, decoding or inference."""
import errno
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time
import tarfile
from pathlib import PurePosixPath

INPUT = Path('/inputs')
OUTPUT = Path('/output')
WORK = Path('/work')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for b in iter(lambda: src.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(command, timeout=600):
    # The outer controller records bounded container logs and kills the whole
    # container on deadline; this subprocess timeout is an additional safeguard.
    print(json.dumps({'event': 'command', 'argv': command}), flush=True)
    subprocess.run(command, check=True, timeout=timeout, stdin=subprocess.DEVNULL)


def fail_write(path):
    try:
        with path.open('xb') as out:
            out.write(b'probe')
    except OSError as exc:
        need(exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM), 'write probe failed for an unexpected reason')
        return exc.errno
    raise RuntimeError('forbidden filesystem write succeeded')


def runtime_identity(work=WORK):
    """Validate numeric identity and private writable paths without passwd edits."""
    ids = {'uid': os.getuid(), 'gid': os.getgid(), 'euid': os.geteuid(), 'egid': os.getegid()}
    need(all(value == 1000 for value in ids.values()), 'nonroot real/effective UID/GID mismatch')
    username = 'kws-asr-runtime'
    expected = {'USER': username, 'LOGNAME': username, 'HOME': str(work / 'home'),
                'XDG_CACHE_HOME': str(work / 'cache'),
                'TORCHINDUCTOR_CACHE_DIR': str(work / 'cache/torchinductor')}
    need(all(os.environ.get(key) == value for key, value in expected.items()),
         'fixed runtime identity/home/cache environment mismatch')
    # The official image need not have a passwd entry for numeric UID 1000.
    # getpass honors LOGNAME/USER; this is a process label, not an OS account.
    need(getpass.getuser() == username, 'runtime process username mismatch')
    need(work.is_dir() and not work.is_symlink(), 'invalid runtime work directory')
    for directory in (work / 'home', work / 'cache', work / 'cache/torchinductor'):
        need(not directory.is_symlink(), 'linked runtime home/cache directory')
        directory.mkdir(exist_ok=True)
        probe = directory / '.kws-write-probe'
        with probe.open('xb') as out:
            out.write(b'probe')
        probe.unlink()
    return dict(ids, process_username=username, process_username_source='fixed_environment_not_passwd',
                home=expected['HOME'], xdg_cache_home=expected['XDG_CACHE_HOME'],
                torchinductor_cache_dir=expected['TORCHINDUCTOR_CACHE_DIR'])


def preflight():
    admission = json.loads((INPUT / 'admission.json').read_text())
    limits = admission['limits']
    need(sys.version.split()[0] == admission['image']['python_version'], 'Python patch version mismatch')
    need(platform.libc_ver() == ('glibc', admission['runtime_root_policy']['target_glibc_maximum']), 'container glibc target mismatch')
    identity = runtime_identity()
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    need(int(status['CapEff'].strip(), 16) == 0, 'effective capabilities present')
    need(status['NoNewPrivs'].strip() == '1' and status['Seccomp'].strip() == '2', 'NNP/seccomp not effective')
    cgroup = Path('/sys/fs/cgroup')
    memory = int((cgroup / 'memory.max').read_text())
    swap = int((cgroup / 'memory.swap.max').read_text())
    quota, period = map(int, (cgroup / 'cpu.max').read_text().split())
    pids = int((cgroup / 'pids.max').read_text())
    need(memory == limits['container_memory_bytes'] and swap == 0 and quota == limits['container_cpus'] * period and
         pids == limits['container_pids'], 'effective cgroup limits mismatch')
    need(all(line.split()[1] != '00000000' for line in Path('/proc/net/route').read_text().splitlines()[1:]), 'default IPv4 route present')
    network_errno = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        try:
            probe.connect(('192.0.2.1', 9))
        except OSError as exc:
            network_errno = exc.errno
        else:
            raise RuntimeError('external reserved-address connect succeeded')
    need(network_errno in (errno.ENETUNREACH, errno.EHOSTUNREACH), 'network probe did not prove unreachable route')
    result = {'root_write_errno': fail_write(Path('/.kws-forbidden')),
              'input_write_errno': fail_write(INPUT / '.kws-forbidden'), 'external_connect_errno': network_errno,
              **identity, 'memory_max': memory, 'swap_max': swap, 'cpu_quota': quota, 'cpu_period': period,
              'pids_max': pids, 'python_version': sys.version.split()[0]}
    result['image_tools'] = {name: shutil.which(name) for name in ('cc', 'gcc', 'make', 'ld', 'ffmpeg', 'sox')}
    need(result['image_tools']['cc'] and result['image_tools']['make'], 'image lacks required build tools')
    result['media_tool_interpretation'] = 'Presence only; no audio or model capability claim'
    Path('/tmp/kws-write-probe').write_bytes(b'probe')
    (OUTPUT / 'write-probe').write_bytes(b'probe')
    # A fixed stdlib child must observe the same inherited isolation.
    run([sys.executable, '-I', '-S', '-c',
         "import getpass,os,pathlib; assert os.getuid()==os.geteuid()==os.getgid()==os.getegid()==1000; "
         "assert getpass.getuser()=='kws-asr-runtime'; assert os.environ['HOME']=='/work/home'; "
         "assert os.environ['TORCHINDUCTOR_CACHE_DIR']=='/work/cache/torchinductor'; "
         "assert 'NoNewPrivs:\\t1' in pathlib.Path('/proc/self/status').read_text()"], 5)
    return result


def installed_env(requirements):
    env = WORK / 'venv'
    run([sys.executable, '-I', '-m', 'venv', str(env)])
    python = str(env / 'bin/python')
    run([python, '-I', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-cache-dir', '--require-hashes',
         '--only-binary=:all:', '--find-links=/inputs/artifacts', '-r', str(INPUT / requirements)], 900)
    run([python, '-I', '-m', 'pip', '--isolated', 'check'], 120)
    return python


def extract_source(source, project, destination_base):
    """Read only regular members into a fresh fixed path; never call tar.extract."""
    target = Path(destination_base) / project['source_root']
    need(not target.exists() and not target.is_symlink(), 'source extraction directory is not fresh')
    target.mkdir(parents=True)
    payloads = {}
    with tarfile.open(source, 'r:gz') as archive:
        members = archive.getmembers()
        need(len(members) == project['archive_members'], 'source member count changed')
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            need(not path.is_absolute() and '..' not in path.parts and path.parts and
                 path.parts[0] == project['source_root'] and '\\' not in member.name and
                 str(path) == member.name.rstrip('/'), 'unsafe source path')
            need(member.name not in seen and (member.isfile() or member.isdir()), 'duplicate or nonregular source member')
            seen.add(member.name)
        for member in members:
            relative = PurePosixPath(member.name).parts[1:]
            destination = target.joinpath(*relative)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            need(relative, 'source root cannot be a regular file')
            destination.parent.mkdir(parents=True, exist_ok=True)
            h, actual = hashlib.sha256(), 0
            with archive.extractfile(member) as src, destination.open('xb') as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    actual += len(chunk)
                    need(actual <= member.size, 'source member exceeds declared bytes')
                    dst.write(chunk)
                    h.update(chunk)
            need(actual == member.size, 'source member size mismatch')
            destination.chmod(member.mode & 0o777)
            os.utime(destination, (1704067200, 1704067200))
            payloads[member.name] = {'bytes': actual, 'sha256': h.hexdigest()}
    manifest = (json.dumps(payloads, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
    need(hashlib.sha256(manifest).hexdigest() == project['source_payload_manifest_sha256'], 'source member identity changed')
    return target


def build():
    # No source executes until the complete host artifact audit and base preflight
    # have succeeded. Build requirements are the independently reviewed closure.
    python = installed_env('build-requirements.txt')
    plan = json.loads((INPUT / 'build-plan.json').read_text())
    for project in plan['sources']:
        source = INPUT / 'artifacts' / project['filename']
        need(digest(source) == project['sha256'], 'source changed inside container')
        fixed_source = extract_source(source, project, WORK / 'sources')
        run([python, '-I', '-m', 'pip', '--isolated', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
             '--no-cache-dir', '--wheel-dir=/output', str(fixed_source)], 600)
    # --no-deps here is only wheel construction, never runtime installation.
    return {'built_source_count': len(plan['sources'])}


PROBE = r'''
import importlib, importlib.metadata, json, resource
from pathlib import Path
import torch
expected = json.loads(Path('/inputs/runtime-versions.json').read_text())
actual = {name: importlib.metadata.version(name) for name in expected}
assert actual == expected, 'installed package versions differ from complete closure'
plan = json.loads(Path('/inputs/build-plan.json').read_text())
modules = ['torch', 'numpy', 'safetensors', 'transformers', 'qwen_asr', 'funasr', 'kaldi_native_fbank'] + plan['required_native_imports']
if 'torchaudio' in expected:
    modules.append('torchaudio')
for name in modules:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(Path('/work/venv')), 'import escaped isolated environment'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
a = torch.arange(16, dtype=torch.float32, device='cpu').reshape(4, 4)
b = torch.eye(4, dtype=torch.float32, device='cpu')
assert torch.equal(a @ b, a)
x = a.to(torch.bfloat16)
assert torch.equal((x @ b.to(torch.bfloat16)).float(), a)
# Small representative operations only, not an ASR/KWS/TTS forward pass.
y = torch.nn.functional.layer_norm(x, (4,))
z = torch.nn.functional.conv1d(torch.ones(1, 2, 8, dtype=torch.bfloat16), torch.ones(2, 2, 3, dtype=torch.bfloat16))
assert y.dtype == torch.bfloat16 and z.dtype == torch.bfloat16
assert torch.isfinite(y).all() and torch.isfinite(z).all()
record = {'schema': 'kws.asr-runtime-cpu-probe.v1', 'event':'cpu_probe', 'imports': modules, 'torchaudio_status': 'imported_locked_package' if 'torchaudio' in expected else 'not_in_locked_knf_route', 'installed_versions': actual, 'torch_version': torch.__version__, 'torch_build_cuda': torch.version.cuda,
                 'device':'cpu', 'float32_matmul':True, 'bf16_matmul_layer_norm_conv1d':True,
                 'interpretation':'Tiny CPU operations only; no model compatibility or latency claim',
                 'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'target_speech_model_weights_loaded':0, 'user_audio_loaded':0, 'speech_model_forward_calls':0,
                 'dependency_import_side_effects':'Package-bundled tokenizers/statistical state may initialize; not counted as target ASR/TTS/KWS weights. No unlocked asset download permitted.'}
Path('/output/cpu-probe.json').write_text(json.dumps(record, sort_keys=True) + '\n')
print(json.dumps(record), flush=True)
'''


def runtime():
    python = installed_env('runtime-requirements.txt')
    script = WORK / 'probe.py'
    script.write_text(PROBE)
    run([python, '-I', str(script)], 300)
    return {'cpu_probe_completed': True, 'cpu_probe_sha256': digest(OUTPUT / 'cpu-probe.json'),
            'target_speech_model_weights_loaded': 0, 'user_audio_loaded': 0}


def main():
    stage = sys.argv[1]
    need(stage in ('preflight', 'build-a', 'build-b', 'runtime'), 'unknown stage')
    started = time.monotonic()
    evidence = preflight()
    if stage.startswith('build-'):
        evidence.update(build())
    if stage == 'runtime':
        evidence.update(runtime())
    for name in ('memory.peak', 'memory.events', 'cpu.stat', 'pids.peak'):
        p = Path('/sys/fs/cgroup') / name
        evidence[name] = p.read_text().strip() if p.exists() else None
    contracts = ('admission.json', 'inventory.json', 'build-plan.json', 'container_stage.py',
                 'build-requirements.txt', 'runtime-requirements.txt', 'runtime-versions.json')
    evidence.update({'schema': 'kws.asr-runtime-container-stage.v1', 'stage': stage,
                     'target_speech_model_weights_loaded': 0, 'user_audio_loaded': 0,
                     'input_contracts_sha256': {n: digest(INPUT / n) for n in contracts if (INPUT / n).is_file()},
                     'wall_seconds': time.monotonic() - started,
                     'child_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
    data = (json.dumps(evidence, sort_keys=True, indent=2) + '\n').encode()
    need(len(data) <= 65536, 'stage receipt too large')
    (OUTPUT / 'stage-receipt.json').write_bytes(data)
    print(json.dumps({'event': 'stage_complete', 'stage': stage, 'receipt_sha256': hashlib.sha256(data).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
