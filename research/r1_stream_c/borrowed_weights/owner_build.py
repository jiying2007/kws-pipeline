#!/usr/bin/env python3
"""Offline host and Cortex-A32 build of a materialized immutable-view SDK."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCES = ('core/r1d_stream.c', 'core/r1d_pcm.c', 'core/reference64_fbank.c',
           'core/r1dfsmn.c', 'core/sha256.c', 'src/fsmn_sdk.c', 'src/fsmn_detector.c')
FLAGS = ('-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard', '-fPIC')

def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sdk', type=Path, required=True)
    ap.add_argument('--toolchain', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sdk, toolchain, output = (getattr(args, n).absolute() for n in ('sdk', 'toolchain', 'output'))
    if output.exists():
        raise ValueError('fresh build directory required')
    output.mkdir(parents=True)
    logs = output / 'logs'
    logs.mkdir()
    commands = []
    receipt = {'schema': 'r1-borrowed-offline-build-v1', 'status': 'BUILDING',
               'model_executions': 0, 'source_files': {}, 'variants': {}, 'commands': commands}
    for p in sorted(sdk.rglob('*')):
        if p.is_file():
            if p.is_symlink():
                raise ValueError('symlink in generated SDK')
            receipt['source_files'][str(p.relative_to(sdk))] = identity(p)

    def run(argv, env, name):
        commands.append([str(x) for x in argv])
        result = subprocess.run(argv, capture_output=True, env=env, timeout=90)
        (logs / (name + '.stdout')).write_bytes(result.stdout)
        (logs / (name + '.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(name + ' failed; retained logs')
        return result.stdout.decode()

    try:
        for arch in ('host', 'arm'):
            env = dict(os.environ)
            env.pop('LD_LIBRARY_PATH', None)
            flags = list(FLAGS)
            if arch == 'arm':
                env['PATH'] = str(toolchain / 'usr/bin') + os.pathsep + env['PATH']
                env['LD_LIBRARY_PATH'] = str(toolchain / 'usr/lib/x86_64-linux-gnu')
                cc = toolchain / 'usr/bin/arm-linux-gnueabihf-gcc-13'
                ar = toolchain / 'usr/bin/arm-linux-gnueabihf-ar'
                readelf = toolchain / 'usr/bin/arm-linux-gnueabihf-readelf'
                flags += ['--sysroot=' + str(toolchain), '-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']
            else:
                cc, ar, readelf = (Path('/usr/bin') / x for x in ('gcc', 'ar', 'readelf'))
            target = output / arch
            target.mkdir()
            objects = []
            for index, source in enumerate(SOURCES):
                obj = target / (Path(source).stem + '.o')
                run([cc, *flags, '-I' + str(sdk / 'include'), '-I' + str(sdk / 'core'),
                     '-c', sdk / source, '-o', obj], env, f'{arch}-object-{index}')
                objects.append(obj)
            lib = target / 'libfsmn_sdk_static.a'
            run([ar, 'rcsD', lib, *objects], env, arch + '-ar')
            shared = target / 'libfsmn_sdk.so.0.1.0'
            run([cc, *flags, '-shared', '-Wl,-soname,libfsmn_sdk.so.0', *objects, '-lm', '-o', shared], env, arch + '-shared')
            binaries = {}
            for name in ('fsmn_run', 'fsmn_run_view', 'fsmn_trace_view', 'fsmn_init_bench'):
                source = sdk / 'src' / (name + '.c')
                binary = target / name
                run([cc, *flags, '-I' + str(sdk / 'include'), source, lib, '-lm', '-o', binary], env, arch + '-' + name)
                headers = run([readelf, '-h', '-W', binary], env, arch + '-' + name + '-elf')
                loader = run([readelf, '-l', '-W', binary], env, arch + '-' + name + '-loader')
                versions = run([readelf, '-V', '-W', binary], env, arch + '-' + name + '-versions')
                if arch == 'arm' and not (re.search(r'Class:\s+ELF32', headers) and re.search(r'Machine:\s+ARM\b', headers)
                    and 'hard-float ABI' in headers and 'little endian' in headers):
                    raise ValueError('ARM ABI mismatch')
                binaries[name] = dict(identity(binary),
                    interpreter=re.findall(r'Requesting program interpreter: ([^\]]+)', loader),
                    glibc_versions=sorted(set(re.findall(r'\bGLIBC_\d+(?:\.\d+)+', versions))))
            receipt['variants'][arch] = {'compiler': run([cc, '--version'], env, arch + '-version').splitlines()[0],
                'compiler_sha256': identity(cc)['sha256'], 'flags': flags, 'binaries': binaries,
                'static_library': identity(lib), 'shared_library': identity(shared)}
        receipt['status'] = 'HOST_AND_ARM_BUILT_NO_MODEL_EXECUTION'
    except BaseException as exc:
        receipt['status'] = 'FAILED_RETAINED_NO_MODEL_EXECUTION'
        receipt['exception'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        receipt['build_script'] = identity(Path(__file__))
        (output / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'variants': receipt['variants']}, sort_keys=True))

if __name__ == '__main__':
    main()
