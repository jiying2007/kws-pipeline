#!/usr/bin/env python3
"""Owner-local strong-R1 FSMN x86/ARM actual model inference replay.

Source-only code. Original/quantized model weights, manifest, and raw PCM remain
owner-held and must NEVER be committed to this public repository.
This measures deterministic stream decisions/clock identity, NOT independent FAR
or device resource/power/SSC305 qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

MODEL_SHA = 'e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb'
FRONTEND_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
MANIFEST_SHA = '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab'
TRACE_NAME = 'owner_arm_trace.c'
MAX_OUTPUT = 4 * 1024 * 1024  # per runner, fail on unbounded output


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, wanted: str, size: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or symlinked source input')
    if size is not None and path.stat().st_size != size:
        raise ValueError('input byte count mismatch')
    if digest(path) != wanted:
        raise ValueError('input SHA256 mismatch')


def load_manifest(path: Path) -> dict[int, dict]:
    verify_file(path, MANIFEST_SHA)
    if path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError('invalid bounded manifest')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'kws-r1-arm-input-recovery-v1':
        raise ValueError('unexpected original input-manifest schema')
    rows = data.get('records')
    if not isinstance(rows, list) or len(rows) != 134:
        raise ValueError('expected 134 original bound PCM identities')
    by_id = {}
    for row in rows:
        i = row.get('index')
        if type(i) is not int or not 0 <= i < 134 or i in by_id:
            raise ValueError('invalid or duplicate PCM index')
        sha = row.get('pcm_sha256')
        n = row.get('pcm_bytes')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid PCM hash')
        if type(n) is not int or not 0 < n <= 1_000_000 or n % 2:
            raise ValueError('invalid PCM byte count')
        if row.get('split') not in ('train', 'development'):
            raise ValueError('invalid saved role')
        by_id[i] = row
    if len(by_id) != 134 or set(by_id) != set(range(134)):
        raise ValueError('incomplete original PCM index')
    return by_id


def parse_indices(spec: str) -> list[int]:
    if spec == 'all':
        return list(range(134))
    values = spec.split(',')
    if not 1 <= len(values) <= 134:
        raise ValueError('invalid index count')
    if any(not v.isdecimal() or len(v)>3 for v in values):
        raise ValueError('invalid PCM index syntax')
    values = [int(v) for v in values]
    if len(values) != len(set(values)) or any(v < 0 or v >= 134 for v in values):
        raise ValueError('invalid or duplicate PCM indices')
    return values


def parse_trace(raw: bytes) -> tuple[list[tuple], list[tuple]]:
    if len(raw) > MAX_OUTPUT:
        raise ValueError('unbounded model trace')
    frame = []
    events = []
    for line in raw.decode('ascii').splitlines():
        parts = line.split(',')
        if parts[0] == 'F' and len(parts) == 5:
            numbers = tuple(int(x) for x in parts[1:])
            if min(numbers) < 0 or numbers[3] >= 2601 or numbers[1] > numbers[2]:
                raise ValueError('invalid frame trace')
            if numbers[0] != len(frame):
                raise ValueError('nonconsecutive frame trace')
            frame.append(numbers)
        elif parts[0] == 'E' and len(parts) == 7:
            numbers = tuple(int(x) for x in parts[1:])
            if (not frame or numbers[1] not in (1, 2) or numbers[4] > numbers[5]
                    or numbers[2] != len(frame)-1 or numbers[3] > numbers[2]):
                raise ValueError('invalid keyword event trace')
            events.append(numbers)
        else:
            raise ValueError('unrecognized trace data')
    if not frame:
        raise ValueError('no actual model frames')
    return frame, events


def checked(argv: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    out = subprocess.run(argv, capture_output=True, timeout=timeout, env=env, check=True)
    if len(out.stdout)>MAX_OUTPUT or len(out.stderr)>MAX_OUTPUT:
        raise ValueError('unbounded external process output')
    return out


def build_sdk(source: Path, out: Path, compiler: str, toolchain: Path | None) -> tuple[Path,Path]:
    args = ['cmake', '-S', str(source), '-B', str(out), '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_TESTING=OFF', '-DCMAKE_C_COMPILER='+compiler]
    if toolchain:
        args.append('-DCMAKE_TOOLCHAIN_FILE='+str(toolchain))
    checked(args, 90)
    checked(['cmake', '--build', str(out), '--parallel', '2'], 120)
    return out/'libfsmn_sdk_static.a', source/'include/fsmn_sdk.h'


def build_trace(source: Path, build: Path, cc: str, owner_script: Path, arch: str) -> Path:
    lib,header = build/'libfsmn_sdk_static.a', source/'include/fsmn_sdk.h'
    if not lib.is_file() or not header.is_file():
        raise ValueError('missing SDK build objects')
    target = build/'owner_r1_trace'
    cmd = [cc, '-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
           '-fno-fast-math','-ffp-contract=off']
    if arch == 'arm':
        cmd += ['-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']
    cmd += ['-I'+str(source/'include'),str(owner_script),str(lib),'-lm','-o',str(target)]
    checked(cmd, 90)
    return target


def verify_arm_elf(readelf: Path, executable: Path) -> None:
    if executable.is_symlink() or not executable.is_file():
        raise ValueError('no compiled ARM executable')
    header = checked([str(readelf), '-h', '-W', str(executable)], 10).stdout.decode('ascii')
    if not (re.search(r'Class:\s+ELF32', header)
            and re.search(r'Machine:\s+ARM\b', header)
            and 'hard-float ABI' in header):
        raise ValueError('wrong ARM target/ABI; cannot claim emulator verification')


def load_quant(repo: Path):
    path = repo / 'research/r1_stream_c/quant_head8/materialize.py'
    spec = importlib.util.spec_from_file_location('r1_owner_arm_quant',path)
    if spec is None or spec.loader is None:
        raise ValueError('missing pinned public quant generator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(args) -> dict:
    repo = args.repo.resolve()
    original = args.reference_sdk.resolve()
    owner_model = args.model.resolve()
    frontend = args.frontend.resolve()
    pcm_dir = args.pcm_dir.resolve()
    output = args.output.resolve(strict=False)
    manifest = args.manifest.resolve()
    qemu = args.emulator.resolve()
    sysroot = args.sysroot.resolve()
    armcc = args.arm_gcc.resolve()
    readelf = args.readelf.resolve()
    source_c = Path(__file__).with_name(TRACE_NAME)
    indices = parse_indices(args.indices)
    if (not repo.is_dir() or output.exists()
            or any(output.is_relative_to(p) for p in (repo,original,pcm_dir))
            or any(p.is_symlink() for p in [output,*output.parents])):
        raise ValueError('must use a fresh non-Git output path')
    if not all(p.is_file() and not p.is_symlink() for p in [qemu,armcc,readelf,source_c]):
        raise ValueError('missing bounded QEMU/compiler/trace source')
    if not (sysroot / 'lib/ld-linux-armhf.so.3').is_file() and not (sysroot / 'lib/arm-linux-gnueabihf/ld-linux-armhf.so.3').is_file():
        raise ValueError('missing ARM hardfloat sysroot loader')
    verify_file(owner_model,MODEL_SHA,3028860)
    verify_file(frontend,FRONTEND_SHA,83840)
    rows = load_manifest(manifest)
    sources = load_quant(repo)
    for path,wanted in sources.SOURCES.items():
        verify_file(original/path,wanted)
    selected=[]
    for i in indices:
        f=pcm_dir/f'{i:03}.pcm'
        verify_file(f,rows[i]['pcm_sha256'],rows[i]['pcm_bytes'])
        selected.append((i,f))
    if args.timeout < 1 or args.timeout > 600:
        raise ValueError('per-model timeout must be between 1 and 600s')
    if not shutil.which('gcc') or not shutil.which('cmake'):
        raise ValueError('host gcc and cmake required')
    # A read-only preflight has completed. Owner-only receipts/PCM traces.
    os.umask(0o077)
    sys.dont_write_bytecode = True
    output.mkdir(parents=True,mode=0o700)
    started=time.monotonic()
    receipt={'schema':'r1-owner-arm-model-replay-v1','status':'STARTED_NOT_QUALIFIED',
             'model_sha256':MODEL_SHA,'frontend_sha256':FRONTEND_SHA,
             'source_scope':'owner-local-only','original_fp32_torch_gate':'FAIL_UNCHANGED',
             'physical_ssc305':False,'acoustic_far_frr':'NOT_RUN',
             'frame_logit_numeric_tolerance':'NOT_MEASURED',
             'attempted_indices':indices,'completed':[]}
    try:
        # Copy only the public reference source; never copy embedded owner weights.
        def ignore(path,names):
            return [x for x in names if x in ('weights','bin','build','__pycache__','notes')]
        src_ref=output/'ref-source'
        shutil.copytree(original,src_ref,ignore=ignore)
        ref_host_build=output/'host-ref-build'
        ref_arm_build=output/'arm-ref-build'
        host_model_lib,_=build_sdk(src_ref,ref_host_build,'gcc',None)
        arm_model_lib,_=build_sdk(src_ref,ref_arm_build,str(armcc),src_ref/'cmake/armv7-hardfloat.cmake')
        assert host_model_lib.is_file() and arm_model_lib.is_file()
        trace_host=build_trace(src_ref,ref_host_build,'gcc',source_c,'host')
        trace_arm_ref=build_trace(src_ref,ref_arm_build,str(armcc),source_c,'arm')
        binaries={'host_ref':trace_host,'arm_ref':trace_arm_ref}
        verify_arm_elf(readelf,trace_arm_ref)
        models={'host_ref':owner_model,'arm_ref':owner_model}
        for kernel in ('scalar','par4'):
            src=output/(f'arm-{kernel}-source')
            sources.materialize_from_repository(repo,src,model=owner_model,frontend=frontend,kernel=kernel)
            b=output/(f'arm-{kernel}-build')
            lib,_=build_sdk(src,b,str(armcc),src/'cmake/armv7-hardfloat.cmake')
            assert lib.is_file()
            binaries['arm_'+kernel]=build_trace(src,b,str(armcc),source_c,'arm')
            verify_arm_elf(readelf,binaries['arm_'+kernel])
            models['arm_'+kernel]=src/'weights/model.r1hq8'
        result_root=output/'private-traces'
        result_root.mkdir()
        for index,pcm in selected:
            traces={}
            status={}
            for label in ('host_ref','arm_ref','arm_scalar','arm_par4'):
                runner=[str(binaries[label]),str(models[label]),str(frontend),str(pcm)]
                if label.startswith('arm_'):
                    runner=[str(qemu),'-L',str(sysroot),*runner]
                p=checked(runner,args.timeout)
                ff,ev=parse_trace(p.stdout)
                trace_path=result_root/(f'{index:03}-{label}.txt')
                trace_path.write_bytes(p.stdout)
                traces[label]=(ff,ev)
                status[label]={'frames':len(ff),'events':len(ev),'trace_sha256':digest(trace_path),
                               'trace_bytes':trace_path.stat().st_size}
            ref=traces['host_ref']
            parity={label:{'frame_and_clock_equal':ff==ref[0],
                           'full_keyword_events_equal':ev==ref[1]}
                    for label,(ff,ev) in traces.items() if label!='host_ref'}
            row={'index':index,'pcm_sha256':rows[index]['pcm_sha256'],
                 'comparison':parity,'variants':status}
            receipt['completed'].append(row)
            (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
            if not all(v['frame_and_clock_equal'] and v['full_keyword_events_equal'] for v in parity.values()):
                raise ValueError('ARM model replay diverged from strong host reference')
        receipt['status']='SOURCE_ARCH_BOUND_REPLAY_PASS_NOT_BOARD_QUALIFICATION'
    except Exception as exc:
        receipt['status']='FAILED_OR_BLOCKED_RETAIN_EVIDENCE'
        receipt['failure_type']=type(exc).__name__
        (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        raise
    finally:
        receipt['wall_seconds']=round(time.monotonic()-started,3)
        (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return {'status':receipt['status'],'completed_rows':len(receipt['completed']),
            'physical_ssc305':False,'acoustic_far_frr':'NOT_RUN',
            'private_output':str(output/'RECEIPT.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['repo','reference-sdk','model','frontend','pcm-dir','manifest','output','emulator','sysroot','arm-gcc','readelf']:
        p.add_argument('--'+flag,required=True,type=Path)
    p.add_argument('--indices',default='0,100,133',
                   help='Comma-separated saved index list or all; default bounded three-source smoke')
    p.add_argument('--timeout',type=int,default=180,help='per process timeout seconds (1..600)')
    a=p.parse_args()
    try:
        result=execute(a)
    except (ValueError, OSError, subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
        p.exit(2, f'OWNER_REPLAY_FAILED_CLOSED: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
