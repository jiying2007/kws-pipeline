"""Bounded synthetic-only research C decoder build/differential contract.
No model, audio, real posterior, downloads, or production runtime integration.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018'
PUBLIC_SYMBOLS = set(('kws3_create kws3_destroy kws3_storage_bytes kws3_reset '
    'kws3_step kws3_step_f32 kws3_process kws3_finish kws3_get_summary '
    'kws3_status_string kws3_keyword kws3_create_with_threshold '
    'kws3_get_threshold_status').split())

def check_threshold_completion(text):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    expected = dict(fixtures_passed=8, completed_rows=569, step_attempts=572,
                    beam_updates=949, windows=21, direct_merge_calls=11)
    if (not rows or rows[-1].get('status') != 'PASS_DECODER_THRESHOLD_SYNTHETIC_ONLY'
            or any(row.get('status') == 'FAIL' for row in rows)
            or any(rows[-1].get(key) != value for key, value in expected.items())):
        raise RuntimeError('bounded threshold fixture completion/count mismatch')

def check_production_symbols(text):
    names = [line.split()[-1].split('@')[0] for line in text.splitlines() if line.split()]
    actual = [name for name in names if name.startswith('kws3_')]
    if len(actual) != 13 or set(actual) != PUBLIC_SYMBOLS or any('kws3_test' in name for name in names):
        raise RuntimeError('production symbols differ from the 13-symbol ABI; no test hooks allowed')

def run(command, *, env=None, timeout=60):
    result = subprocess.run(command, cwd=SOURCE, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    print(result.stdout, end='')
    if result.returncode:
        raise RuntimeError(f'synthetic subprocess failed ({result.returncode}): {command[0]}')
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitizers', action='store_true',
                        help='Require ASan/UBSan and LeakSanitizer on both fixture executables; no fallback.')
    args = parser.parse_args()
    common = [args.compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
              '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard']
    with tempfile.TemporaryDirectory(prefix='research-kws3-') as temporary:
        output = Path(temporary)
        smoke = output / 'smoke'
        library = output / 'libkws3_test.so'
        threshold = output / 'threshold'
        production = output / 'libkws3.so'
        flags = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitizers else []
        run(common + ['-O1', '-g'] + flags + ['test_synthetic.c', '-lm', '-o', str(smoke)], timeout=45)
        env = dict(os.environ)
        if args.sanitizers:
            env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
            env['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
        text = run([str(smoke)], env=env)
        if 'synthetic groups=8 ' not in text:
            raise RuntimeError('eight-group C smoke completion missing')
        run(common + ['-O2', '-fPIC', '-shared', '-DKWS3_TESTING', 'kws3.c', '-lm', '-o', str(library)], timeout=45)
        text = run([sys.executable, '-S', '-B', 'test_differential.py', '--library', str(library),
                    '--approved-small-synthetic-only'])
        if not re.search(r'Ran 28 tests', text) or 'model_calls 0 real_posterior_rows 0' not in text:
            raise RuntimeError('bounded 28-method synthetic differential completion missing')
        run(common + ['-O1', '-g'] + flags + ['test_threshold.c', '-lm', '-o', str(threshold)], timeout=45)
        check_threshold_completion(run([str(threshold)], env=env))
        # Production is a distinct build: KWS3_TESTING must never enter it.
        run(common + ['-O2', '-fPIC', '-shared', 'kws3.c', '-lm', '-o', str(production)], timeout=45)
        check_production_symbols(run(['nm', '-D', '--defined-only', str(production)], timeout=10))
    print('PASS: four builds; eight legacy C smoke groups; 28 differential methods (25 C-backed); '
          'eight threshold fixture groups (569 rows, 572 attempts, 949 beam updates, 21 windows, 11 direct merges); '
          '13 production symbols without test hooks; '
          'zero model/audio/real-posterior inputs; sanitizers=' + str(args.sanitizers))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
