#!/usr/bin/env python3
"""Compile/run invented numeric fixtures only; no saved project model execution."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise ValueError('C source-test compiler unavailable')
    with tempfile.TemporaryDirectory(prefix='en63-c-source-') as temp:
        temp = Path(temp)
        flags = ['-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
                 '-Wall', '-Wextra', '-Werror', '-pedantic']
        if args.sanitize:
            flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        subprocess.run([compiler, *flags, str(HERE/'en63_stream.c'), str(HERE/'synthetic_cases.c'),
                        '-lm', '-o', str(temp/'synthetic')], check=True)
        subprocess.run([str(temp/'synthetic')], check=True, timeout=60)
        subprocess.run([compiler, *flags, '-c', str(HERE/'en63_stream.c'), '-o', str(temp/'core.o')], check=True)
        symbols = subprocess.check_output(['nm', '-u', str(temp/'core.o')], text=True)
        names = {line.split()[-1] for line in symbols.splitlines()}
        forbidden = {'malloc', 'calloc', 'realloc', 'free', 'fopen', 'fread', 'fwrite', 'printf',
                     'pthread_create', 'pthread_mutex_lock', 'fork', 'execv', 'system'}
        if names & forbidden:
            raise ValueError('C data-plane forbidden symbols: '+str(names & forbidden))
        print('PASS core object has no allocation/thread/lock/stdio/process symbols')
        if not args.sanitize:
            # A fake constant classifier, not either retained project checkpoint.
            def row(kind, name, values):
                return 'static const '+kind+' '+name+'['+str(len(values))+']={'+','.join(map(str, values))+'};\n'
            header = '#include "en63_stream.h"\n'
            for name, values in (('w', [1]*400), ('mean', [0]*32), ('std', [1]*32), ('mel', [1]*32)):
                header += row('float', name, values)
            for name, values in (('start', [0]*32), ('count', [1]*32), ('offset', list(range(33)))):
                header += row('uint16_t', name, values)
            for kind, size, tag in (('gru', 6531, 1), ('dsc_tcn', 4803, 2)):
                header += f'static const float {kind}_p[{size}]={{[{size-2}]=4}};\n'
                header += f'static const en63_model en63_{kind}_model={{ {tag}, {size}, {kind}_p, w, mean, std, start, count, offset, mel, 32, "synthetic" }};\n'
            header += '#define EN63_FRONTEND_TABLE_BYTES 2178u\n'
            (temp/'en63_generated.h').write_text(header)
            subprocess.run([compiler, *flags, '-I', str(HERE), '-I', str(temp), str(HERE/'en63_stream.c'),
                            str(HERE/'replay_cli.c'), '-lm', '-o', str(temp/'fake-replay')], check=True)
            (temp/'fake.pcm').write_bytes(b'\xe8\x03'*39123)
            (temp/'inputs.txt').write_text(str(temp/'fake.pcm')+'\n')
            parent_buffer = bytearray(96*1024*1024)
            result = subprocess.check_output([str(temp/'fake-replay'), '--supervise', 'gru', 'continuous', '160',
                                              str(temp/'inputs.txt'), '-'], text=True, timeout=60)
            stats = json.loads(result)
            parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if not (stats['process_peak_rss_kib'] < parent_peak/2 and stats['frames'] == 243
                    and stats['pcm_samples'] == 39123 and stats['eof_events'] == 0
                    and not stats['trace_io_enabled']):
                raise ValueError('Native RSS isolation / synthetic CLI counters failed: '+str(stats))
            print('PASS fake-model native child peak RSS', stats['process_peak_rss_kib'],
                  'KiB versus Python parent', parent_peak, 'KiB; no project model executed')
            del parent_buffer


if __name__ == '__main__':
    main()
