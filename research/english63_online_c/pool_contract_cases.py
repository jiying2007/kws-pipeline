#!/usr/bin/env python3
"""Compile/run only invented pooling arrays, model parameters and PCM."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
FLAGS = ('-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
         '-Wall', '-Wextra', '-Werror', '-pedantic')
PUBLIC_SYMBOLS = ('en63_state_bytes', 'en63_init', 'en63_reset',
                  'en63_discontinuity', 'en63_feed', 'en63_finish')
FORBIDDEN = {'malloc', 'calloc', 'realloc', 'free', 'fopen', 'fread', 'fwrite',
             'printf', 'pthread_create', 'pthread_mutex_lock', 'fork', 'execv', 'system'}


def run(compiler, flags, directory):
    objects = []
    for mode in (0, 1):
        prefix = 'pool' + str(mode)
        target = directory / (prefix + '.o')
        defines = ['-DEN63_POOL_FIXTURE_OBJECT=1',
                   '-DEN63_POOL_TIME_MAJOR=' + str(mode),
                   '-DEN63_POOL_FIXTURE_ENTRY=' + prefix + '_pool']
        defines += ['-D' + name + '=' + prefix + '_' + name for name in PUBLIC_SYMBOLS]
        subprocess.run([compiler, *flags, *defines, '-c', str(HERE / 'pool_contract_cases.c'),
                        '-o', str(target)], check=True, timeout=120)
        names = {line.split()[-1] for line in subprocess.check_output(['nm', '-u', str(target)], text=True).splitlines()}
        if names & FORBIDDEN:
            raise ValueError('Pooling core forbidden symbols: ' + str(names & FORBIDDEN))
        objects.append(target)
    executable = directory / 'pool-contract-cases'
    subprocess.run([compiler, *flags, str(HERE / 'pool_contract_cases.c'),
                    *map(str, objects), '-lm', '-o', str(executable)], check=True, timeout=120)
    subprocess.run([str(executable)], check=True, timeout=120)
    for value in ('2', '-1', '1+1', 'not_an_integer_macro'):
        result = subprocess.run([compiler, *flags, '-DEN63_POOL_TIME_MAJOR=' + value,
                                 '-c', str(HERE / 'en63_stream.c'), '-o', str(directory / 'reject.o')],
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode == 0 or 'EN63_POOL_TIME_MAJOR' not in result.stderr:
            raise ValueError('Invalid pooling macro was not rejected: ' + value)
    for option in ('-ffast-math', '-ffinite-math-only'):
        result = subprocess.run([compiler, *flags, option, '-DEN63_POOL_TIME_MAJOR=1',
                                 '-c', str(HERE / 'en63_stream.c'), '-o', str(directory / 'reject.o')],
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode == 0 or 'EN63 pooling requires strict' not in result.stderr:
            raise ValueError('Unsafe floating-point mode was not rejected: ' + option)
    print('PASS pooling core has no allocation/thread/lock/stdio/process symbols; invalid macro/fast-math rejected')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise ValueError('Pooling source-test compiler unavailable')
    flags = list(FLAGS)
    if args.sanitize:
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    with tempfile.TemporaryDirectory(prefix='en63-pool-source-') as temporary:
        run(compiler, flags, Path(temporary))


if __name__ == '__main__':
    main()
