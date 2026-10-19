#!/usr/bin/env python3
"""Explicit bounded public mock/toy matrix. No audio or learned payload reads."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

M = Path(__file__).resolve().parent
ROOT = M.parents[1]
D = ROOT / 'research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018'
spec = importlib.util.spec_from_file_location('bounded', M / 'build.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
MODES = [('default', []), ('profile', ['-DKWS_B_PROFILE']),
         ('block2', ['-DKWS_AFFINE_BLOCK2']),
         ('profile_block2', ['-DKWS_B_PROFILE', '-DKWS_AFFINE_BLOCK2'])]

def execute():
    if any(k in os.environ for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT')):
        raise RuntimeError('loader overrides forbidden')
    binding = guard.validate_sources(ROOT)
    with tempfile.TemporaryDirectory(prefix='b-optin-check-') as directory:
        tmp = Path(directory); vendor = tmp / 'include/vendor'; vendor.mkdir(parents=True)
        (vendor / 'fft_twiddles.h').symlink_to(ROOT / guard.FFT)
        # Extract the tested candidate body from production, never maintain a second copy.
        source = (M / 'fsmn2599.c').read_text()
        begin = source.index('#ifdef KWS_AFFINE_BLOCK2\n') + len('#ifdef KWS_AFFINE_BLOCK2\n')
        body = source[begin:source.index('#else\n', begin)]
        (tmp / 'affine_block2.inc').write_text(body.replace('static void affine(', 'static __attribute__((noinline)) void affine_block2('))
        common = ['/usr/bin/gcc', *guard.FLAGS, '-I'+str(M), '-I'+str(D), '-I'+str(tmp / 'include'), '-I'+str(tmp)]
        sources = [str(M / n) for n in ('b_stream_adapter.c', 'canonical_pcm.c', 'canonical_fbank.c', 'fsmn2599.c', 'sha256.c', 'softmax2599.c')] + [str(D / 'kws3.c')]
        commands = []; programs = []; libraries = []
        for name, flags in MODES:
            lib = tmp / (name+'.so'); libraries.append((name, lib, '-DKWS_B_PROFILE' in flags))
            extra = [str(M / 'b_profile.c')] if '-DKWS_B_PROFILE' in flags else []
            commands.append(common+flags+['-shared', '-fPIC', '-Wl,-z,defs', *sources, *extra, '-lm', '-o', str(lib)])
            commands.append(['/usr/bin/g++', '-std=c++11', '-Wall', '-Wextra', '-Werror', '-pedantic', *flags,
                             '-I'+str(M), '-I'+str(D), str(M / 'test_cpp_link.cpp'), str(lib), '-lm', '-o', str(tmp / (name+'_cpp'))])
            for test, inputs in [('contract', [str(M / 'test_profile_contract.c'), str(lib)]), ('mock', [str(M / 'test_b_stream_adapter.c'), str(M / 'canonical_pcm.c')])]:
                program = name+'_'+test; programs.append(program)
                commands.append(common+flags+inputs+['-lm', '-o', str(tmp / program)])
        for name, inputs in [('unit', ['test_profile_unit.c']), ('overhead', ['test_timer_overhead.c', 'b_profile.c']), ('affine', ['test_affine_block2.c'])]:
            commands.append(common+[str(M / n) for n in inputs]+['-lm', '-o', str(tmp / name)]); programs.append(name)
        start = time.monotonic(); cpu = 0.
        for cmd in commands:
            _, spent = guard.run_bounded(cmd, tmp, 60-(time.monotonic()-start), 60-cpu); cpu += spent
        outputs = {}
        for name in programs:
            outputs[name], _ = guard.run_bounded([str(tmp / name)], tmp, 10, 10)
        for test in ('contract', 'mock'):
            if len({outputs[name+'_'+test] for name, _ in MODES}) != 1:
                raise RuntimeError('opt-in observable mock/argument behavior differs')
        modes = {}
        for name, lib, profiled in libraries:
            out, _ = guard.run_bounded(['/usr/bin/nm', '-D', '--defined-only', str(lib)], tmp, 5, 5)
            symbols = {line.split()[-1] for line in out.splitlines() if line.strip()}
            expected = guard.ADAPTER | guard.DECODER_EXPORTS | guard.CORE_EXPORTS
            if profiled: expected |= {'b_profile_get', 'b_profile_reset', 'b_profile_size'}
            if symbols != expected: raise RuntimeError('unexpected opt-in ABI')
            modes[name] = {'sha256': hashlib.sha256(lib.read_bytes()).hexdigest(), 'exports': sorted(symbols)}
        if guard.validate_sources(ROOT) != binding: raise RuntimeError('source identity changed')
        return {'status':'PUBLIC_MOCK_TOY_PASS', 'compiler_invocations':len(commands), 'program_invocations':len(programs), 'compile_cpu_seconds':cpu, 'source_binding_sha256':binding, 'modes':modes, 'outputs':outputs, 'model_loads':0, 'model_forwards':0, 'audio_reads':0, 'qualification_allowed':False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if args.execute: print(json.dumps(execute(), indent=2))
    else: parser.print_help()
