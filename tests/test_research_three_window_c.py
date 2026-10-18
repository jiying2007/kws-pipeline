"""Bounded synthetic-only research C decoder build/differential contract.
No model, audio, real posterior, downloads, or production runtime integration.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018'

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
                        help='Require ASan/UBSan and LeakSanitizer on the smoke executable; no fallback.')
    args = parser.parse_args()
    common = [args.compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
              '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard']
    with tempfile.TemporaryDirectory(prefix='research-kws3-') as temporary:
        output = Path(temporary)
        smoke = output / 'smoke'
        library = output / 'libkws3_test.so'
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
    print('PASS: two builds; eight C smoke groups; 28 differential methods (25 C-backed); '
          'zero model/audio/real-posterior inputs; sanitizers=' + str(args.sanitizers))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
