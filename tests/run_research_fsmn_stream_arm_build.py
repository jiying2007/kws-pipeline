#!/usr/bin/env python3
"""Opt-in full B ARM hard-float compile/link check. No binary/model execution."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def execute(root):
    if any(name in os.environ for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT")):
        raise RuntimeError("loader overrides are forbidden")
    source = root / 'research/fsmn2599-stream-c/build.py'
    spec = importlib.util.spec_from_file_location('fsmn_stream_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    binding = build.validate_sources(root)
    cc = shutil.which('arm-linux-gnueabihf-gcc', path='/usr/bin:/bin')
    cxx = shutil.which('arm-linux-gnueabihf-g++', path='/usr/bin:/bin')
    nm = shutil.which('arm-linux-gnueabihf-nm', path='/usr/bin:/bin')
    if not all((cc, cxx, nm)):
        raise RuntimeError('ARM hard-float gcc, g++ and nm are required; no skip/pass fallback')
    flags = ['-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']
    module, decoder = root / build.MODULE, root / build.DECODER
    with tempfile.TemporaryDirectory(prefix='b-full-arm-build-') as temporary:
        temp = Path(temporary)
        vendor = temp / 'include/vendor'
        vendor.mkdir(parents=True)
        (vendor / 'fft_twiddles.h').symlink_to(root / build.FFT)
        include = ['-I'+str(module), '-I'+str(decoder), '-I'+str(temp / 'include')]
        sources = [module / name for name in ('b_stream_adapter.c', 'canonical_pcm.c',
                   'canonical_fbank.c', 'fsmn2599.c', 'sha256.c', 'softmax2599.c')]
        sources.append(decoder / 'kws3.c')
        library = temp / 'libb_stream_adapter.so'
        started, cpu = time.monotonic(), 0.0
        commands = [
            [cc, *build.FLAGS, *flags, '-fPIC', '-shared', '-Wl,-z,defs', *include,
             *map(str, sources), '-lm', '-o', str(library)],
            [cxx, '-std=c++11', '-Wall', '-Wextra', '-Werror', '-pedantic', *flags,
             *include, str(module / 'test_cpp_link.cpp'), str(library), '-lm',
             '-o', str(temp / 'cpp_link')],
        ]
        for command in commands:
            _, spent = build.run_bounded(command, temp, 60-(time.monotonic()-started), 60-cpu)
            cpu += spent
        output, _ = build.run_bounded([nm, '-D', '--defined-only', str(library)], temp, 5, 5)
        exports = {line.split()[-1] for line in output.splitlines() if line.strip()}
        expected = build.ADAPTER | build.DECODER_EXPORTS | build.CORE_EXPORTS
        if exports != expected:
            raise RuntimeError('full B ARM exported symbols differ from fixed public ABI')
        if build.validate_sources(root) != binding:
            raise RuntimeError('source binding changed during cross-build')
    return {'schema': 1, 'full_B_ARM_compile_link': True, 'cpp_link': True,
            'compiler_invocations': 2, 'exports': sorted(exports),
            'source_binding_sha256': binding, 'target': 'cortex-a32/neon-vfpv4/hard',
            'binary_executions': 0, 'model_loads': 0, 'qualification_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    print(json.dumps(execute(Path(__file__).resolve().parents[1]), sort_keys=True))


if __name__ == '__main__':
    main()
