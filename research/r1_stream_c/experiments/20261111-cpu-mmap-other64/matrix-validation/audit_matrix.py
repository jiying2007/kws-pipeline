#!/usr/bin/env python3
"""Independently audit retained matrix bytes; never run a binary or model.

Only the explicitly named fresh output is written. Raw PCM, labels, individual
receipts and trace rows are not emitted. This verifies retained evidence, not
physical hardware identity, a complete-logit contract or acoustic qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import struct

ROOT = Path('/workspace/scratch/fec7ccd5bb4b')
OLD = Path('/workspace/scratch/1e1fe64498b2')
INPUTS = OLD/'r1-local/evidence-original/KWS_FSMN_R1_EVIDENCE_20261110/inputs'
SAVED = OLD/'r1-local/evidence-qemu'
HEAD = '13661a38064247d38bca88287e48db85bf2ccd79'
PLAN_SHA = '1074eadcdb1783b4f50abc11c26aeaf15f09a3f23775361957864c54c2e801b2'
BUILD_SHA = 'e3b5335d67d9db5c43878ad2c520c3493ddb52c183e6662c2a3efd689f88b477'
INPUT_SHA = '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab'
SAVED_SHA = '8a3d0fbd2822c383b40db2159785419a3b85ab926f06a14c3b3da2a056dd1e44'
MODEL_SHA = '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
FRONTEND_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
SOURCE_PINS = {
    'borrowed_weights/materialize.py': 'baa9b40d9b3df55b3e30077e69124985c551b70dc380a1ba6dd3c61e5ea516c3',
    'cpu_opt/materialize.py': '822aa6381a8569aa4350beaa363a8d104bdaa70d6ac5d463ace535ed2f3d7350',
    'quant_head8/materialize.py': 'c44aa8c0521cf1b6b7f2789eeb5d9983a2ebdcb354a9c06de82e03d06a4e3ac4',
    'quant_head8/pack_model.py': 'ad564f0a60f2ab49e05e884302cf131669a42951a435b6efaed6f422a7e0603b',
    'quant_head8/owner_arm_trace.c': '7e947f7c1534ce191886b3506abb235d9342192409122525c552435c81c4891f',
    'borrowed_weights/init_bench.c': '6ccff60211c2b5119a9142f8a276d86781ec2936e65e1a84beee9e44f74ceae2',
}
KERNELS = ('par4', 'block2', 'block4')
ORDERS = (KERNELS, ('block2', 'block4', 'par4'), ('block4', 'par4', 'block2'))
COUNT_KEYS = ('sdk_model_frames', 'keyword_events', 'keyword1_events', 'keyword2_events')
MEM_KEYS = ('proc_self_pss_kib', 'proc_self_private_dirty_kib')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'nonregular or linked evidence: '+path.name)
    return path


def sha(path):
    digest = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def load(path, cap=4194304):
    need(regular(path).stat().st_size <= cap, 'JSON size cap')
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid)


def verify(path, identity):
    need(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
         'retained file identity changed: '+path.name)


def inventory(directory, expected):
    need(not directory.is_symlink(), 'linked evidence directory')
    found = {p.name for p in directory.iterdir()}
    need(found == expected, 'missing or additional members in '+directory.name)
    for name in found:
        regular(directory/name)


def count_trace(data, samples):
    need(0 < len(data) <= 65536 and data.endswith(b'\n'), 'trace bound/termination')
    frames = []
    events = [0, 0]
    for line in data.decode('ascii').splitlines():
        fields = line.split(',')
        need(fields[0] in ('F', 'E'), 'unknown trace row')
        need(len(fields) == (5 if fields[0] == 'F' else 7), 'trace arity')
        need(all(re.fullmatch(r'0|[1-9][0-9]*', x) for x in fields[1:]), 'trace integer')
        values = [int(x) for x in fields[1:]]
        if fields[0] == 'F':
            frame, source, delivery, token = values
            need(frame == len(frames) and source == 720+480*frame, 'frame identity/grid')
            need(delivery == min(((source+319)//320)*320, samples) and
                 source <= delivery <= samples and token < 2601, 'frame clock/token')
            frames.append((source, delivery))
        else:
            segment, word, frame, first, source, delivery = values
            need(segment == 0 and word in (1, 2) and first <= frame < len(frames),
                 'event identity')
            need((source, delivery) == frames[frame] and frame-first <= 150, 'event clock/span')
            events[word-1] += 1
    need(len(frames) == max(0, (samples-720)//480+1), 'frame denominator')
    return dict(zip(COUNT_KEYS, (len(frames), sum(events), *events)))


def receipt(path, pcm_bytes, counts, storage, arch):
    row = load(path, 65536)
    need(row.get('schema') == 'r1-strong-fsmn-offline-board-resource-v1', 'receipt schema')
    need(row.get('model_kind') == 'source-hash-checked-by-sdk', 'model check evidence')
    need(row.get('model_storage') == storage and row.get('model_caller_heap_bytes') ==
         (0 if storage == 'read_only_mmap' else 1960140), 'storage mismatch')
    need(all(row.get(k) is False for k in ('device_qualified', 'live_afe', 'independent_far_frr')),
         'qualification flags changed')
    need(row.get('power_mw') is None and row.get('stack_high_water_bytes') is None,
         'unmeasured metrics changed')
    need(row.get('pcm_bytes') == pcm_bytes and row.get('feed_chunk_bytes') == 640 and
         row.get('feed_calls') == (pcm_bytes+639)//640, 'input/feed denominator')
    need(all(type(row.get(k)) is int and row[k] == v for k, v in counts.items()), 'receipt counts')
    need(row.get('arch') == 'x86_64' if arch == 'host' else
         isinstance(row.get('arch'), str) and row['arch'].startswith('arm'), 'receipt architecture')
    for key in ('audio_seconds', 'offline_feed_cpu_seconds', 'offline_feed_wall_seconds',
                'offline_rtf', 'cpu_seconds_per_audio_second', 'model_init_wall_ms'):
        need(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] > 0,
             'nonpositive/nonfinite resource metric')
    need(abs(row['audio_seconds']-pcm_bytes/32000) <= 1e-9, 'audio duration arithmetic')
    for target, numerator in (('offline_rtf', 'offline_feed_wall_seconds'),
                              ('cpu_seconds_per_audio_second', 'offline_feed_cpu_seconds')):
        need(abs(row[target]-row[numerator]/row['audio_seconds']) <= 2e-9, 'resource ratio arithmetic')
    p95, maximum = row.get('feed_p95_histogram_upper_us'), row.get('feed_max_us')
    need(type(p95) is int and 0 < p95 <= 2**31 and not p95 & (p95-1), 'p95 bucket')
    need(type(maximum) is int and maximum >= 0 and p95 <= 1 << max(0, (max(1, maximum)-1).bit_length()),
         'p95 exceeds maximum bucket')
    need(type(row.get('process_peak_rss_kib')) is int and row['process_peak_rss_kib'] >= 0, 'RSS metric')
    for key in MEM_KEYS:
        need(key in row and (row[key] is None or type(row[key]) is int and row[key] >= 0), 'memory metric')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'matrix-execution/AUDIT.json')
    args = parser.parse_args()
    need(not args.output.exists(), 'fresh output required; do not overwrite an audit')
    execution, build = ROOT/'matrix-execution', ROOT/'matrix-build'
    need(sha(execution/'PLAN.json') == PLAN_SHA and sha(build/'BUILD_RECEIPT.json') == BUILD_SHA,
         'fixed plan or build receipt changed')
    plan, compiled, terminal = (load(p) for p in
        (execution/'PLAN.json', build/'BUILD_RECEIPT.json', execution/'TERMINAL.json'))
    need(plan['source_head'] == compiled['source_head'] == HEAD, 'source commit')
    need(compiled['status'] == 'BUILT_NO_MODEL_EXECUTION' and compiled['model_executions'] == 0, 'build terminal')
    need(sha(ROOT/'matrix-validation/run_matrix.py') == plan['run_script_sha256'], 'run script drift')
    verify(ROOT/'matrix-validation/build_matrix.py', compiled['build_script'])
    for key in ('source_probe', 'wrapper'):
        verify(Path(compiled[key]['path']), compiled[key])
    for relative, digest in SOURCE_PINS.items():
        need(sha(ROOT/'kws-source/research/r1_stream_c'/relative) == digest, 'source generator drift')
    generated_count = 0
    for kernel in KERNELS:
        sdk = build/'sdk'/kernel
        actual = {p.relative_to(sdk).as_posix() for p in sdk.rglob('*') if not p.is_dir()}
        need(actual == set(compiled['generated_sources'][kernel]), 'SDK file inventory drift')
        for name, identity in compiled['generated_sources'][kernel].items():
            verify(sdk/name, identity)
            generated_count += 1
        for arch in ('host', 'arm'):
            record = compiled['builds'][kernel+'_'+arch]
            verify(build/kernel/arch/'libfsmn_sdk_static.a', record['static_library'])
            compiler = Path('/usr/bin/gcc') if arch == 'host' else OLD/'r1-toolchain/toolchain/usr/bin/arm-linux-gnueabihf-gcc-13'
            need(sha(compiler.resolve()) == record['compiler_sha256'], 'compiler drift')
            for item in record['binaries'].values():
                binary = Path(item['path']); verify(binary, item)
                elf = binary.read_bytes()[:52]
                need(elf[:4] == b'\x7fELF' and elf[5] == 1 and elf[4] == (2 if arch == 'host' else 1), 'ELF type')
                need(struct.unpack_from('<H', elf, 18)[0] == (62 if arch == 'host' else 40), 'ELF machine')
                if arch == 'arm':
                    need(struct.unpack_from('<I', elf, 36)[0] & 0x400, 'ARM hard-float ABI')
    build_logs = {f'{i:03d}.{ext}' for i in range(len(compiled['commands'])) for ext in ('stdout', 'stderr')}
    inventory(build/'logs', build_logs)
    for name in build_logs:
        data = (build/'logs'/name).read_bytes()
        need(not data if name.endswith('.stderr') else not re.search(rb'(^|\n)(Traceback|FAILED|.*error:)', data),
             'build log error')
    need(sha(INPUTS/'INPUT_MANIFEST.json') == INPUT_SHA and sha(SAVED/'FILE_MANIFEST.json') == SAVED_SHA,
         'input or baseline manifest drift')
    records, saved_index = load(INPUTS/'INPUT_MANIFEST.json')['records'], load(SAVED/'FILE_MANIFEST.json')
    need(len(records) == 134 and [r['index'] for r in records] == list(range(134)), '134-input identity')
    need(sha(Path(plan['model_path'])) == MODEL_SHA and sha(Path(plan['frontend_path'])) == FRONTEND_SHA,
         'model/frontend post-run drift')
    need(sha(Path(plan['qemu']['path'])) == plan['qemu']['sha256'], 'QEMU drift')
    baseline, expected = {}, {}
    for r in records:
        i = r['index']; pcm = INPUTS/'pcm'/f'{i:03d}.pcm'
        need(pcm.stat().st_size == r['pcm_bytes'] and sha(pcm) == r['pcm_sha256'], 'PCM post-run drift')
        path = SAVED/f'private-traces/{i:03d}-arm_par4.txt'
        verify(path, saved_index[f'private-traces/{i:03d}-arm_par4.txt'])
        baseline[i] = path.read_bytes(); expected[i] = count_trace(baseline[i], r['samples'])
    schedule = [(f'parity-{a}-{k}-{i:03d}', a, k, i, 'capture', 'mmap')
                for a in ('host', 'arm') for k in KERNELS for i in range(134)]
    schedule += [(f'host-perf-{n:02d}-{k}', 'host', k, 49, 'plain', 'mmap')
                 for n in range(9) for k in ORDERS[n % 3]]
    schedule += [(f'host-memory-{n:02d}-{m}', 'host', 'par4', 49, 'plain', m)
                 for n in range(6) for m in (('copy', 'mmap') if n % 2 == 0 else ('mmap', 'copy'))]
    names = [row[0] for row in schedule]
    need(len(names) == len(set(names)) == 843, 'audit schedule denominator')
    for folder, suffix in (('claims', '.json'), ('receipts', '.json'), ('logs', '.stderr')):
        inventory(execution/folder, {n+suffix for n in names})
    inventory(execution/'traces', {n+'.txt' for n in names[:804]})
    need(terminal['status'] == 'PASS_FIXED_FE_MATRIX_HOST_SCREEN_ONLY' and 'failure' not in terminal,
         'unsuccessful terminal')
    need(terminal['plan_sha256'] == PLAN_SHA and terminal['claimed'] == names and
         [r['name'] for r in terminal['completed_invocations']] == names and
         [r['name'] for r in terminal['completed']] == names[:804] and
         terminal['capture_completed'] == 804 and terminal['all_claims_completed'] is True,
         'terminal denominator/order/identity')
    need(terminal['physical_ssc305'] == terminal['independent_far_frr'] == 'NOT_RUN' and
         terminal['full_logit_parity'] == 'NOT_TESTED' and terminal['model_promotion_authorized'] is False and
         terminal['original_fp32_numeric_gate'] == 'FAIL_UNCHANGED', 'scope drift')
    after = {'model_sha256': MODEL_SHA, 'frontend_sha256': FRONTEND_SHA,
             'input_manifest_sha256': INPUT_SHA, 'saved_trace_manifest_sha256': SAVED_SHA,
             'verified_pcm_rows': 134, 'all_unchanged': True}
    need(terminal['post_run_inputs'] == after, 'post-run receipt mismatch')
    totals, raw, receipt_hashes = {}, {}, set()
    for number, (name, arch, kernel, index, kind, mode) in enumerate(schedule):
        claim = load(execution/'claims'/(name+'.json'), 65536)
        binary = compiled['builds'][kernel+'_'+arch]['binaries'][kind]['path']
        argv = [binary, plan['model_path'], plan['frontend_path'], str(INPUTS/'pcm'/f'{index:03d}.pcm')]
        if mode == 'mmap': argv += ['--mmap']
        if arch == 'arm': argv = [plan['qemu']['path'], '-L', plan['sysroot'], *argv]
        trace_path = execution/'traces'/(name+'.txt')
        need(claim['name'] == name and claim['attempt'] == 1 and claim['argv'] == argv and
             claim['trace'] == (str(trace_path) if kind == 'capture' else None), 'claim invocation mismatch')
        need((execution/'logs'/(name+'.stderr')).stat().st_size == 0, 'nonempty invocation stderr')
        row = receipt(execution/'receipts'/(name+'.json'), records[index]['pcm_bytes'], expected[index],
                      'read_only_mmap' if mode == 'mmap' else 'sdk_owned_copy', arch)
        raw[name] = row
        record = terminal['completed_invocations'][number]
        expected_scope = ('captured_fe_parity_no_performance_claim' if kind == 'capture' else
                          'uninstrumented_host_cpu_screen' if name.startswith('host-perf-') else
                          'uninstrumented_host_memory_only_no_speed_claim')
        receipt_digest = sha(execution/'receipts'/(name+'.json'))
        need(receipt_digest not in receipt_hashes, 'duplicate raw receipt payload')
        receipt_hashes.add(receipt_digest)
        need(record['index'] == index and record['model_storage'] == row['model_storage'] and
             record['scope'] == expected_scope and record['receipt_sha256'] == receipt_digest and
             record['expected_counts'] == expected[index] and record['counts_match_saved'] is True,
             'invocation completion binding')
        if kind == 'capture':
            data = trace_path.read_bytes()
            need(data == baseline[index] and count_trace(data, records[index]['samples']) == expected[index],
                 'actual trace differs from retained original')
            completed = terminal['completed'][number]
            need(completed['arch'] == arch and completed['kernel'] == kernel and completed['index'] == index and
                 completed['trace_sha256'] == sha(trace_path) and completed['receipt_sha256'] == record['receipt_sha256'] and
                 completed['fe_trace_equal'] is True and completed['counts_match_saved'] is True and
                 all(completed[k] == expected[index][k] for k in COUNT_KEYS), 'capture completion binding')
            group = totals.setdefault(arch+'_'+kernel, dict(clips=0, **{k: 0 for k in COUNT_KEYS}))
            group['clips'] += 1
            for key in COUNT_KEYS: group[key] += expected[index][key]
    for group in totals.values():
        need(group == dict(clips=134, **dict(zip(COUNT_KEYS, (12534, 88, 44, 44)))), 'six-group totals')
    cpu = {k: [raw[f'host-perf-{n:02d}-{k}']['cpu_seconds_per_audio_second'] for n in range(9)] for k in KERNELS}
    p95 = {k: [raw[f'host-perf-{n:02d}-{k}']['feed_p95_histogram_upper_us'] for n in range(9)] for k in KERNELS}
    medians = {k: statistics.median(v) for k, v in cpu.items()}
    savings = {k: statistics.median(100*(1-c/b) for c, b in zip(cpu[k], cpu['par4'])) for k in KERNELS[1:]}
    wins = {k: sum(c < b for c, b in zip(cpu[k], cpu['par4'])) for k in KERNELS[1:]}
    hp = terminal['host_perf']
    need(hp['rounds'] == 9 and hp['pcm_index'] == 49 and hp['instrumented_or_qemu_measurements_used'] == 0 and
         hp['cpu_seconds_per_audio_second'] == cpu and hp['p95_upper_us'] == p95 and
         hp['median_cpu_seconds_per_audio_second'] == medians and hp['paired_median_cpu_savings_pct'] == savings,
         'CPU statistics mismatch')
    memory = {m: {k: [raw[f'host-memory-{n:02d}-{m}'][k] for n in range(6)] for k in MEM_KEYS} for m in ('copy', 'mmap')}
    memory_medians = {m: {k: statistics.median(v) if all(x is not None for x in v) else None
                         for k, v in values.items()} for m, values in memory.items()}
    deltas = {k: [c-m for c, m in zip(memory['copy'][k], memory['mmap'][k])]
              if all(v is not None for mode in memory for v in memory[mode][k]) else None for k in MEM_KEYS}
    delta_medians = {k: statistics.median(v) if v is not None else None for k, v in deltas.items()}
    hm = terminal['host_memory']
    need(hm['pairs'] == 6 and hm['invocations'] == 12 and hm['kernel'] == 'par4' and hm['pcm_index'] == 49 and
         hm['own_process_observations_kib'] == memory and hm['own_process_medians_kib'] == memory_medians and
         hm['paired_copy_minus_mmap_kib'] == deltas and hm['median_copy_minus_mmap_kib'] == delta_medians and
         hm['peak_rss_used_for_savings'] is False and hm['memory_pair_timing_used_for_speedup'] is False,
         'memory statistics mismatch')
    result = {'schema': 'r1-mmap-matrix-independent-byte-audit-v1', 'status': 'PASS_RETAINED_EVIDENCE_AUDIT',
        'source_head': HEAD, 'audit_script_sha256': sha(Path(__file__)), 'plan_sha256': PLAN_SHA,
        'build_receipt_sha256': BUILD_SHA, 'terminal_sha256': sha(execution/'TERMINAL.json'),
        'generated_source_files_verified': generated_count, 'binaries_verified': 12,
        'model_sha256': MODEL_SHA, 'frontend_sha256': FRONTEND_SHA,
        'input_manifest_sha256': INPUT_SHA, 'saved_trace_manifest_sha256': SAVED_SHA,
        'source_build_input_drift': False, 'unique_inputs': 134, 'unique_completed_claims': 843,
        'capture_invocations': 804, 'host_cpu_invocations': 27, 'host_memory_invocations': 12,
        'mismatching_traces': 0, 'nonempty_invocation_stderr': 0, 'build_stderr_errors': 0,
        'groups': totals, 'host_cpu': {'rounds': 9, 'paired_wins_of_9': wins,
            'median_cpu_seconds_per_audio_second': medians, 'paired_median_cpu_savings_pct': savings,
            'median_feed_p95_upper_us': {k: statistics.median(v) for k, v in p95.items()}},
        'host_memory': {'pairs': 6, 'own_process_medians_kib': memory_medians,
                        'median_copy_minus_mmap_kib': delta_medians},
        'model_input_posthash_unchanged': True, 'new_model_calls_in_audit': 0,
        'full_logit_parity': 'NOT_TESTED', 'physical_ssc305': 'NOT_RUN',
        'independent_far_frr': 'NOT_RUN', 'original_fp32_numeric_gate': 'FAIL_UNCHANGED',
        'model_promotion_authorized': False,
        'limitations': ['Host timing is not SSC305 evidence; QEMU/capture timings excluded.',
            'Receipt-only CPU counts do not establish per-frame parity for the 39 plain runs.',
            'Numeric process exits were checked by the hash-bound runner, not separately stored per invocation.',
            'PSS/Private_Dirty are sampled after streaming, not peak RSS or a runtime speed claim.']}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
