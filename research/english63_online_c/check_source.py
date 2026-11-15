#!/usr/bin/env python3
"""Compile/run invented numeric fixtures only; no saved project model execution."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import struct
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
        screen_spec = importlib.util.spec_from_file_location('en63_librispeech_source_fixture', HERE/'librispeech_screen_cases.py')
        screen_contract = importlib.util.module_from_spec(screen_spec)
        screen_spec.loader.exec_module(screen_contract)
        screen_contract.run(compiler, flags, temp)
        pool_spec = importlib.util.spec_from_file_location('en63_pool_source_fixture', HERE/'pool_contract_cases.py')
        pool_contract = importlib.util.module_from_spec(pool_spec)
        pool_spec.loader.exec_module(pool_contract)
        pool_contract.run(compiler, flags, temp)
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
            if (HERE/'en63_generated.h').exists() or (HERE/'en63_generated.h').is_symlink():
                raise ValueError('Invented fixture header would be shadowed by a source-directory header')
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
            # Exercise the new driver only with this invented header and PCM.
            # No original header/checkpoint/audio is opened by this source test.
            spec = importlib.util.spec_from_file_location('en63_state_source_fixture', HERE/'state_replay.py')
            state_contract = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(state_contract)
            subprocess.run([compiler, *flags, '-I', str(HERE), '-I', str(temp), str(HERE/'en63_stream.c'),
                            str(HERE/'state_cases.c'), '-lm', '-o', str(temp/'fake-state-cases')], check=True)
            invented = temp/'invented-state-input.pcm'
            invented.write_bytes(b'\xe8\x03'*64123 + b'\x30\xf8'*64123)
            for kind in ('gru', 'dsc_tcn'):
                trace = temp/(kind+'-invented-state.trace')
                raw = subprocess.check_output([str(temp/'fake-state-cases'), '--supervise', kind,
                                               str(invented), str(trace)], text=True, timeout=60)
                body, lifetime = [json.loads(line) for line in raw.splitlines()]
                if not (body['forward_frames'] == 4773 and body['supplied_samples_including_replays'] == 769475
                        and body['native_checks_passed'] is True and lifetime['child_exit_code'] == 0
                        and lifetime['child_signaled'] is False
                        and lifetime['process_lifetime_peak_rss_kib'] >= body['inference_end_high_water_rss_kib'] > 0):
                    raise ValueError('Invented state harness did not close its fixed execution budget')
                groups = state_contract.read_trace(trace.read_bytes())
                for actual, expected, frame_base, sample_base in (
                        (2, 0, 0, 0), (3, 1, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0),
                        (8, 9, 0, 399), (11, 12, 1, 400), (14, 15, 201, 32401)):
                    if not state_contract.identical(groups[actual], groups[expected], frame_base, sample_base):
                        raise ValueError('Invented state reset/interleave/gap trace comparison failed')
                print('PASS invented state harness', kind, '4773 frames / 769475 supplied samples;',
                      'original checkpoint and original audio executions = 0')
            # The new paired profile driver is exercised with this same invented
            # header and a constant invented PCM stream, never retained inputs.
            subprocess.run([compiler, *flags, '-DEN63_POOL_TIME_MAJOR=0', '-I', str(HERE), '-I', str(temp),
                            str(HERE/'en63_stream.c'), str(HERE/'pool_profile.c'), '-lm',
                            '-o', str(temp/'fake-pool-profile')], check=True)
            profile_pcm = temp/'invented-pool-profile.pcm'
            profile_pcm.write_bytes(b'\xe8\x03'*2101760)
            cpu = min(os.sched_getaffinity(0))
            full_trace = temp/'invented-pool-profile.trace'
            record = struct.Struct('<IQ38fI')
            expected_signature = None
            expected_events = None
            for phase in ('trace', 'warmup', 'timed'):
                raw = subprocess.check_output([str(temp/'fake-pool-profile'), '--supervise', 'gru', phase,
                                               str(profile_pcm), str(full_trace) if phase == 'trace' else '-',
                                               str(cpu)], text=True, timeout=30)
                lines = raw.splitlines()
                if len(lines) != 2:
                    raise ValueError('Invented profile must retain both worker and lifetime JSON lines')
                body, lifetime = map(json.loads, lines)
                expected = {'schema': 'english63-pool-native-pass-v1', 'kind': 'gru', 'variant': 0,
                            'phase': phase, 'affinity_cpu': cpu, 'affinity_single_cpu': True,
                            'pcm_samples': 2101760, 'frames': 13134, 'feed_calls': 13136,
                            'input_pcm_bytes': 4203520, 'trace_io_enabled': phase == 'trace',
                            'eof_padding_samples': 0, 'eof_events': 0,
                            'timed_region': 'preloaded-pcm-en63_feed-and-constant-signature-callback'}
                for key, value in expected.items():
                    if type(body.get(key)) is not type(value) or body[key] != value:
                        raise ValueError('Invented profile identity/count mismatch: '+key)
                integers = ('proposal_count', 'state_bytes', 'weight_parameter_bytes', 'frontend_table_bytes',
                            'cpu_ns', 'wall_ns', 'inference_end_high_water_rss_kib')
                if any(type(body.get(key)) is not int or body[key] < 0 for key in integers):
                    raise ValueError('Invented profile must report integer resource/counter values')
                if not (lifetime.get('schema') == 'english63-pool-child-lifetime-v1' and
                        type(lifetime.get('child_exit_code')) is int and lifetime['child_exit_code'] == 0 and
                        lifetime.get('child_signaled') is False and
                        type(lifetime.get('process_lifetime_peak_rss_kib')) is int and
                        lifetime['process_lifetime_peak_rss_kib'] >= body['inference_end_high_water_rss_kib'] > 0 and
                        all(type(lifetime.get(key)) is int and lifetime[key] >= 0
                            for key in ('child_user_us', 'child_system_us'))):
                    raise ValueError('Invented profile child exit/lifetime RSS accounting failed')
                feed_ns = body.get('feed_ns')
                if phase == 'timed':
                    if not (type(feed_ns) is list and len(feed_ns) == 13136 and
                            all(type(value) is int and value > 0 for value in feed_ns) and
                            body['cpu_ns'] > 0 and body['wall_ns'] >= sum(feed_ns) > 0):
                        raise ValueError('Invented profile raw feed timings/CPU clock failed')
                elif feed_ns != [] or body['cpu_ns'] != 0 or body['wall_ns'] != 0:
                    raise ValueError('Untimed invented phases must not pretend to have CPU/feed measurements')
                if phase == 'trace':
                    trace_bytes = full_trace.read_bytes()
                    if trace_bytes[:16] != b'EN63TRC1'+struct.pack('<II', 1, 168) or len(trace_bytes) != 16+13134*168:
                        raise ValueError('Invented profile complete binary trace layout/count failed')
                    signature = 14695981039346656037
                    for value in trace_bytes[16:]:
                        signature = ((signature ^ value)*1099511628211) & ((1 << 64)-1)
                    expected_signature = format(signature, '016x')
                    expected_events = 0
                    for index, row in enumerate(record.iter_unpack(trace_bytes[16:])):
                        if not (row[0] == 0 and row[1] == 400+index*160 and
                                all(math.isfinite(value) for value in row[2:-1]) and row[-1] in (0, 1, 2)):
                            raise ValueError('Invented profile trace clock/finiteness/proposal failed')
                        expected_events += row[-1] != 0
                if (body.get('signature64') != expected_signature or body['proposal_count'] != expected_events
                        or expected_events is None or expected_events <= 0):
                    raise ValueError('Invented trace/warmup/timed complete-payload FNV/event identity failed')
                print('PASS invented profile phase', phase, '13134 frames / 13136 calls; complete-payload FNV',
                      body['signature64'], 'matches saved invented trace; original inputs executed = 0')


if __name__ == '__main__':
    main()
