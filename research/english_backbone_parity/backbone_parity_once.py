"""Once-guarded numerical verification of the fixed shared backbone only.

This separate protocol consumes the original successful static C export and
three frozen artificial signals. It executes no head, training, recognition or
real audio, and cannot establish full KWS, FAR/FRR, or SSC305 qualification.
Default and --template only report a disabled source contract.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
REPOSITORY = 'jiying2007/kws-pipeline'
SCHEMA = 'english-fixed-shared-backbone-parity-v1'
PINS_SCHEMA = 'english-fixed-shared-backbone-parity-pins-v1'
PROFILE = 'FIXED_SHARED_BACKBONE_ONLY'
STATIC_SOURCE_HEAD = 'cfc69dcd7535ea909403f10a91bcfe26ccc1b331'
STATIC_SCHEMA = 'english-fixed-frontend-static-export-clip-bound-fix-v2'
MAX_ARTIFACT_ZIP = 256 * 1024**2
GENERATED_FILES = ('fixed_models.c', 'fixed_models.h', 'export_manifest.json')
REFERENCE_FILES = ('weak_embedding_contract.py', 'causal_embedding.py')
FRONTEND_FILES = ('fixed_frontend.h',
 'fixed_frontend.c',
 'export_fixed_frontend.py',
 'frontend_replay.c',
 'test_fixed_frontend.py',
 'README.md')
ASSETS = {'melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'),
 'embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f')}
FALSE_FLAGS = ('training_admitted',
 'formal_training_admitted',
 'human_gold',
 'trusted_negative',
 'pronunciation_verified',
 'model_promotion_allowed',
 'qualification_allowed')
SINE_CYCLE = (0,
 1282,
 2531,
 3719,
 4815,
 5793,
 6627,
 7299,
 7791,
 8091,
 8192,
 8091,
 7791,
 7299,
 6627,
 5793,
 4815,
 3719,
 2531,
 1282,
 0,
 -1282,
 -2531,
 -3719,
 -4815,
 -5793,
 -6627,
 -7299,
 -7791,
 -8091,
 -8192,
 -8091,
 -7791,
 -7299,
 -6627,
 -5793,
 -4815,
 -3719,
 -2531,
 -1282)
PCM_SHA = {'zero': '55873fecc61a79e87ca550c7072e38ccdd7ecb600ace286fe4717952a97c42b0',
 'sine400': 'f81681f58003475f0e81e7625f1bf7d76fb018585d3757c0e3083af83c1a8b01',
 'lcg14': '4e97af4846861db16ce7461517d405f60dca4473b515ae1efcb0f505221142dc'}
CHUNKS = (1, 1279, 257, 4096)
SAMPLES = (16000,
 17280,
 18560,
 19840,
 21120,
 22400,
 23680,
 24960,
 26240,
 27520,
 28800,
 30080,
 31360,
 32640,
 33920,
 35200,
 36480,
 37760,
 39040,
 40320,
 41600,
 42880,
 44160,
 45440,
 46720,
 48000)
STAGE_COUNTS = {1: 3104, 2: 2432, 3: 96}
TRACE_MAGIC = b'EFFTR01\x00'
LIMITS = {'raw_mel': (0.01, 2e-05), 'normalized_mel': (0.001, 2e-05), 'embedding': (0.001, 0.0002)}
PLAN = {'protocol': 'FIXED_SHARED_BACKBONE_ONLY',
 'scope': 'THREE_FIXED_ARTIFICIAL_SIGNALS_PCM_TO_MEL_TO_NORMALIZED_MEL_TO_96_EMBEDDING_ONLY',
 'pcm16_samples_per_case': 48000,
 'sample_rate': 16000,
 'pcm_sha256': {'zero': '55873fecc61a79e87ca550c7072e38ccdd7ecb600ace286fe4717952a97c42b0',
                'sine400': 'f81681f58003475f0e81e7625f1bf7d76fb018585d3757c0e3083af83c1a8b01',
                'lcg14': '4e97af4846861db16ce7461517d405f60dca4473b515ae1efcb0f505221142dc'},
 'signals': {'zero': 'all signed PCM16 zeros',
             'sine400': 'fixed 40-integer cycle, 400Hz, peak8192, repeated1200',
             'lcg14': 'update uint32 seed0x12345678 by1664525*x+1013904223; (x>>18)-8192'},
 'chunk_pattern_samples': [1, 1279, 257, 4096],
 'reset_between_cases': True,
 'eof_padding_samples': 0,
 'first_embedding_sample': 16000,
 'embedding_hop_samples': 1280,
 'cases': 3,
 'embedding_windows_per_backend': 78,
 'maximum_reference_onnx_session_runs': 156,
 'maximum_c_processes_per_backend': 3,
 'error_rule': 'abs(candidate-reference) <= atol + rtol*abs(reference), every finite element',
 'limits': {'raw_mel': {'atol': 0.01, 'rtol': 2e-05},
            'normalized_mel': {'atol': 0.001, 'rtol': 2e-05},
            'embedding': {'atol': 0.001, 'rtol': 0.0002}},
 'exact_embedding_clock_identity': True,
 'compile_timeout_seconds': 180,
 'replay_timeout_seconds': 300,
 'total_timeout_seconds': 1800,
 'rationale': 'The three frontend limits and artificial signals are unchanged from the independently '
              'declared full-head plan. This separate protocol checks only the shared backbone; no head is '
              'executed or qualified, and bounds are not widened after results.',
 'head_evaluations': 0,
 'optimizer_updates': 0,
 'tts_asr_phone_calls': 0,
 'real_audio_cases': 0,
 'full_head_plan_unchanged_sha256': 'cdb8cce44b3ba88c92afeaf2b1d64124190c46198091f375d036a099964a16fb'}
FIXED_PLAN_SHA = '5657e4387909bf7b47ba4ed7adec723005a8428357bd75d3515c3ddfe26e550b'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def regular(path, maximum=256 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
         path.stat().st_size <= maximum, 'REGULAR_BOUNDED_FILE:' + str(path))
    raw = path.read_bytes()
    need(len(raw) <= maximum, 'FILE_GREW')
    return raw


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(path):
    return json.loads(regular(path, 32 * 1024**2), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return sha(raw)


def hex_id(value, size):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % size, value) is not None


def member_path(name):
    return (type(name) is str and '\\' not in name and not PurePosixPath(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')))


def artifact_pin(pin):
    need(type(pin.get('artifact_id')) is int and pin['artifact_id'] > 0 and
         type(pin.get('zip_bytes')) is int and 0 < pin['zip_bytes'] <= MAX_ARTIFACT_ZIP and
         hex_id(pin.get('zip_sha256'), 64) and member_path(pin.get('result_member')) and
         PurePosixPath(pin['result_member']).name == 'RESULT.json' and
         hex_id(pin.get('result_sha256'), 64), 'ACTUAL_ARTIFACT_ZIP_AND_RESULT_PINS')


def synthetic_pcm():
    state, noise = 0x12345678, []
    for _ in range(48000):
        state = (1664525 * state + 1013904223) & 0xffffffff
        noise.append((state >> 18) - 8192)
    values = {'zero': (0,) * 48000, 'sine400': SINE_CYCLE * 1200, 'lcg14': noise}
    result = {name: struct.pack('<48000h', *samples) for name, samples in values.items()}
    need({name: sha(raw) for name, raw in result.items()} == PCM_SHA, 'FIXED_ARTIFICIAL_PCM_IDENTITIES')
    return result


def chunks(raw):
    need(len(raw) == 96000, 'THREE_SECOND_PCM_BYTES')
    offset = index = 0
    while offset < 48000:
        end = min(48000, offset + CHUNKS[index % len(CHUNKS)])
        yield struct.unpack('<%dh' % (end - offset), raw[2 * offset:2 * end])
        offset, index = end, index + 1


def trace_bytes(rows):
    output = bytearray(TRACE_MAGIC)
    for row in rows:
        for stage, count in STAGE_COUNTS.items():
            values = row[stage]
            need(len(values) == count and all(math.isfinite(x) for x in values), 'FINITE_TRACE_VALUES')
            output.extend(struct.pack('<IIQ', stage, count, row['sample']))
            output.extend(struct.pack('<%df' % count, *values))
    return bytes(output)


def errors(reference, candidate, samples, atol, rtol):
    need(len(reference) == len(candidate) == len(samples), 'COMPARISON_ROW_COUNT')
    maximum = {'abs': (0.0, None), 'relative': (0.0, None), 'ratio': (0.0, None)}
    violations = count = zero_mismatches = nonzero_count = 0
    for index, (expected, actual, sample) in enumerate(zip(reference, candidate, samples)):
        need(len(expected) == len(actual), 'COMPARISON_ELEMENT_COUNT')
        for component, (a, b) in enumerate(zip(expected, actual)):
            need(math.isfinite(a) and math.isfinite(b), 'FINITE_COMPARISON')
            delta, allowed = abs(b - a), atol + rtol * abs(a)
            count += 1
            violations += int(delta > allowed)
            zero_mismatches += int(a == 0 and delta != 0)
            nonzero_count += int(a != 0)
            where = {'embedding_index': index, 'sample': sample, 'component': component,
                     'reference': a, 'candidate': b}
            values = [('abs', delta), ('ratio', delta / allowed)]
            if a:
                values.append(('relative', delta / abs(a)))
            for key, value in values:
                if maximum[key][1] is None or value > maximum[key][0]:
                    maximum[key] = (value, where)
    return {'atol': atol, 'rtol': rtol, 'elements': count, 'violations': violations,
            'max_abs_error': maximum['abs'][0], 'max_abs_at': maximum['abs'][1],
            'max_relative_error_nonzero_reference': maximum['relative'][0] if nonzero_count else None,
            'max_relative_at': maximum['relative'][1], 'zero_reference_mismatch_count': zero_mismatches,
            'nonzero_reference_elements': nonzero_count,
            'max_tolerance_ratio': maximum['ratio'][0], 'max_tolerance_ratio_at': maximum['ratio'][1]}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_archive(path, pin):
    raw = regular(path, MAX_ARTIFACT_ZIP)
    need(len(raw) == pin['zip_bytes'] and sha(raw) == pin['zip_sha256'], 'EXACT_ARTIFACT_ZIP_BYTES')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members = archive.infolist()
    need(len(members) == len({m.filename for m in members}) and
         sum(m.file_size for m in members) <= 1024**3, 'UNIQUE_BOUNDED_ARTIFACT_MEMBERS')
    for item in members:
        name = item.filename[:-1] if item.is_dir() else item.filename
        need(member_path(name) and not item.flag_bits & 1 and
             stat.S_IFMT(item.external_attr >> 16) in
             ((0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)), 'SAFE_ARTIFACT_MEMBER')
    return archive


def archive_bytes(archive, member, limit):
    info = archive.getinfo(member)
    need(not info.is_dir() and 0 < info.file_size <= limit, 'BOUNDED_REQUIRED_ARCHIVE_MEMBER')
    return archive.read(member)


def json_bytes(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def static_export_archive(args, pins, frontend_hashes):
    """Return original generated bytes; no parser export or model inference."""
    wanted = pins['static_export']
    with checked_archive(args.static_export_zip, wanted) as archive:
        base = PurePosixPath(wanted['result_member']).parent
        need(str(base / 'FAILURE.json') not in archive.namelist(), 'COMPLETED_STATIC_EXPORT_WITHOUT_FAILURE')
        raw = archive_bytes(archive, wanted['result_member'], 32 * 1024**2)
        need(sha(raw) == wanted['result_sha256'], 'EXACT_STATIC_EXPORT_RESULT_BYTES')
        result = json_bytes(raw)
        need(result.get('schema') == STATIC_SCHEMA and result.get('status') == 'EXPORTED_STATIC_GRAPHS_ONLY' and
             result.get('repository') == 'jiying2007/kws-pipeline' and
             result.get('run_id') == wanted['run_id'] and result.get('run_attempt') == 1 and
             result.get('source_head') == wanted['github_sha'] == STATIC_SOURCE_HEAD and
             result.get('export_attempted') is True and
             result.get('parser_versions') == {'numpy': '1.26.4', 'onnx': '1.17.0'} and
             all(result.get(k) == 0 for k in ('model_calls', 'onnx_inference_calls', 'c_inference_calls',
                                             'optimizer_updates', 'tts_asr_phone_calls')) and
             result.get('numerical_parity_established') is False and result.get('ssc305_qualification') is False,
             'COMPLETED_ZERO_INFERENCE_STATIC_EXPORT_REQUIRED')
        assets = {name: {'bytes': size, 'sha256': digest} for name, (size, digest) in ASSETS.items()}
        need(result.get('assets') == result.get('expected_assets') == assets, 'STATIC_EXPORT_SAME_ASSETS')
        prefix = args.frontend_source.resolve().relative_to(args.pipeline_source.resolve()).as_posix()
        code = result.get('code_files')
        need(type(code) is dict and {prefix + '/' + name for name in FRONTEND_FILES} <= set(code),
             'STATIC_EXPORT_FULL_FRONTEND_SOURCE_BINDING')
        for name, entry in code.items():
            need(member_path(name) and type(entry) is dict and set(entry) == {'bytes', 'sha256'},
                 'STATIC_EXPORT_SOURCE_PATH')
            source_raw = regular(args.pipeline_source / name, 1024**2)
            need(entry == {'bytes': len(source_raw), 'sha256': sha(source_raw)}, 'STATIC_EXPORT_SOURCE_BYTES')
        need(all(code[prefix + '/' + name]['sha256'] == digest for name, digest in frontend_hashes.items()),
             'STATIC_EXPORT_USED_EXACT_FRONTEND')
        generated = {}
        expected = result.get('generated')
        need(type(expected) is dict and set(expected) == {'generated/' + name for name in GENERATED_FILES},
             'STATIC_EXPORT_EXACT_GENERATED_ROSTER')
        for name, entry in wanted['generated'].items():
            need(entry['member'] == str(base / 'generated' / name), 'EXACT_STATIC_GENERATED_MEMBER_PATH')
            content = archive_bytes(archive, entry['member'], 64 * 1024**2)
            identity = {'bytes': len(content), 'sha256': sha(content)}
            need(identity == expected['generated/' + name] ==
                 {k: entry[k] for k in ('bytes', 'sha256')}, 'EXACT_STATIC_GENERATED_BYTES')
            generated[name] = content
    manifest = json_bytes(generated['export_manifest.json'])
    need(manifest.get('schema') == 'fixed-feature-pure-export-v1' and
         manifest.get('real_inference_performed') is False and manifest.get('numerical_parity_established') is False and
         manifest.get('ssc305_qualification') is False and
         manifest.get('generated_c_sha256') == sha(generated['fixed_models.c']) and
         set(manifest.get('graphs', {})) == set(ASSETS) and set(result.get('graphs', {})) == set(ASSETS),
         'ORIGINAL_STATIC_GENERATED_MANIFEST')
    for name, (size, digest) in ASSETS.items():
        graph = manifest['graphs'][name]
        need(graph.get('asset_sha256') == digest and graph.get('asset_bytes') == size and
             result['graphs'][name] == {k: graph[k] for k in
                ('asset_sha256', 'asset_bytes', 'input', 'output', 'constant_bytes', 'workspace_bytes')},
             'STATIC_GRAPH_ASSET_AND_GEOMETRY_BINDING')
    return result, generated


def timed_process(command, timeout, log):
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    try:
        result = subprocess.run([str(value) for value in command], timeout=timeout,
                                capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired as error:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        once(log, {'status': 'TIMEOUT', 'timeout_seconds': timeout,
                   'wall_seconds': time.perf_counter() - start,
                   'host_child_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime})
        raise TimeoutError('BOUNDED_CHILD_PROCESS_TIMEOUT') from error
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {'returncode': result.returncode, 'wall_seconds': time.perf_counter() - start,
              'host_child_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
              'stdout': result.stdout[:32768], 'stderr': result.stderr[:32768],
              'scope': 'whole host process including startup and trace I/O; ARM includes QEMU emulation'}
    once(log, report)
    need(result.returncode == 0, 'CHILD_PROCESS_FAILED:' + str(log.name))
    return report


def timeout_handler(_signum, _frame):
    raise TimeoutError('TOTAL_VERIFICATION_DEADLINE')


STATIC_EXPORT = {
    'run_id': '43823297841', 'github_sha': STATIC_SOURCE_HEAD, 'run_attempt': 1,
    'artifact_id': 14205188962, 'zip_bytes': 2722450,
    'zip_sha256': '1ffc968da201e07186929f05d9d8c651e18390a92f5436b6cdfec3a3cb1e9ac0',
    'result_member': 'RESULT.json',
    'result_sha256': '4b2dcca15d047333aab39fc4c345592213c86d6d0dec4849175a84918471bb08',
    'generated': {
        'fixed_models.c': {'member': 'generated/fixed_models.c', 'bytes': 13464364,
            'sha256': '9bf740a692370dfed02f034f87f6f0dfcaac42f3f83152c79fd27088ccc71941'},
        'fixed_models.h': {'member': 'generated/fixed_models.h', 'bytes': 165,
            'sha256': '0fb2bae330f87ea685e0ce713879872c3d4372c1544e62e9c58a539b00c2eeb1'},
        'export_manifest.json': {'member': 'generated/export_manifest.json', 'bytes': 144095,
            'sha256': 'b94549aec824a8eddb014e19f01629d8733c35f809d763927b712df845b5677b'}}}


def pins_template():
    """Only confirmed static evidence is populated; no execution is enabled."""
    return {'schema': PINS_SCHEMA, 'profile': PROFILE, 'enabled': False,
        'main_before': STATIC_SOURCE_HEAD, 'worker_code_sha256': None, 'plan_sha256': FIXED_PLAN_SHA,
        'frontend_sources_sha256': None, 'reference_sources_sha256': None,
        'static_export': json.loads(canonical(STATIC_EXPORT)), 'no_retry': True,
        'head_evaluations': 0, 'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0,
        'head_parity_established': False, 'kws_model_delivered': False, 'ssc305_qualification': False,
        **dict.fromkeys(FALSE_FLAGS, False)}


def validate_pins(pins):
    need(type(pins) is dict and pins.get('schema') == PINS_SCHEMA and pins.get('enabled') is True,
         'BACKBONE_VERIFICATION_DISABLED')
    template = pins_template()
    dynamic = {'enabled', 'worker_code_sha256', 'frontend_sources_sha256', 'reference_sources_sha256'}
    need(set(pins) == set(template) and all(pins[k] == v for k, v in template.items() if k not in dynamic) and
         sha(canonical(PLAN)) == pins['plan_sha256'] == FIXED_PLAN_SHA and
         hex_id(pins['worker_code_sha256'], 64) and all(pins[k] is False for k in FALSE_FLAGS),
         'FIXED_BACKBONE_SCOPE_PLAN_AND_STATIC_EVIDENCE')
    for field, expected in (('frontend_sources_sha256', FRONTEND_FILES), ('reference_sources_sha256', REFERENCE_FILES)):
        need(type(pins[field]) is dict and set(pins[field]) == set(expected) and
             all(hex_id(value, 64) for value in pins[field].values()), 'COMPLETE_FIXED_SOURCE_PINS:' + field)


def execution_guard(pins, env, event, worker_sha):
    """Pure first-successor guard; source tests pass dictionaries, not fake env."""
    validate_pins(pins)
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPOSITORY and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and re.fullmatch('[1-9][0-9]*', env.get('GITHUB_RUN_ID', '')) and
         env['GITHUB_RUN_ID'] != STATIC_EXPORT['run_id'] and event.get('ref') == 'refs/heads/main' and
         event.get('before') == pins['main_before'] == STATIC_SOURCE_HEAD and
         event.get('after') == env.get('GITHUB_SHA') and hex_id(event.get('after'), 40) and
         event['after'] != STATIC_SOURCE_HEAD and event.get('repository', {}).get('full_name') == REPOSITORY and
         event.get('repository', {}).get('private') is False and worker_sha == pins['worker_code_sha256'],
         'FIRST_NEW_PUBLIC_PIPELINE_MAIN_ATTEMPT_ONLY')


def parse_trace(raw, samples=SAMPLES):
    """Exactly three feature stages: a head stage or extra EOF record fails."""
    row_bytes = sum(16 + 4 * count for count in STAGE_COUNTS.values())
    need(len(raw) == 8 + len(samples) * row_bytes and raw[:8] == TRACE_MAGIC, 'TRACE_LENGTH_OR_MAGIC')
    rows, position = [], 8
    for index, sample in enumerate(samples):
        row = {'sample': sample}
        for stage, count in STAGE_COUNTS.items():
            header = struct.unpack_from('<IIQ', raw, position)
            need(header == (stage, count, sample), 'TRACE_STAGE_COUNT_CLOCK:%d' % index)
            position += 16
            values = struct.unpack_from('<%df' % count, raw, position)
            position += 4 * count
            need(all(math.isfinite(x) for x in values), 'NONFINITE_FEATURE_TRACE')
            row[stage] = values
        rows.append(row)
    return rows


def comparisons(reference, candidate):
    need(len(reference) == len(candidate) and
         all(a['sample'] == b['sample'] for a, b in zip(reference, candidate)), 'EXACT_EMBEDDING_CLOCKS')
    clocks = [row['sample'] for row in reference]
    metrics = {name: errors([r[stage] for r in reference], [r[stage] for r in candidate], clocks, *LIMITS[name])
               for stage, name in ((1, 'raw_mel'), (2, 'normalized_mel'), (3, 'embedding'))}
    return {'checked_stages': ['raw_mel', 'normalized_mel', 'embedding'], 'metrics': metrics,
            'exact_embedding_clocks': True, 'head_checked': False,
            'passed': all(item['violations'] == 0 for item in metrics.values())}


def replay_summary(stdout, rows):
    summary = json_bytes(stdout)
    need(summary.get('samples') == 48000 and summary.get('embeddings') == len(rows) == 26 and
         summary.get('head_enabled') is False and summary.get('head_ready') == summary.get('events') == 0 and
         summary.get('ssc305_qualification') is False and
         all(type(summary.get(key)) is int and 0 < summary[key] <= 256 * 1024**2
             for key in ('workspace_bytes', 'frontend_state_bytes')), 'BACKBONE_ONLY_C_PROCESS_ACCOUNTING')
    return summary


def source_preflight(args, pins, expected_head):
    pipeline, frontend = args.pipeline_source.resolve(), args.frontend_source.resolve()
    reference = pipeline / 'research/english_final_embedding'
    need(pipeline.is_dir() and frontend.is_relative_to(pipeline) and
         Path(__file__).resolve().is_relative_to(pipeline), 'ALL_EXECUTED_SOURCES_IN_PIPELINE_CHECKOUT')
    commit = subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip()
    need(commit == expected_head and commit != STATIC_SOURCE_HEAD and not subprocess.check_output(
        ['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(), 'EXACT_CLEAN_EXECUTION_HEAD')
    front_hashes = {name: sha(regular(frontend / name, 1024**2)) for name in FRONTEND_FILES}
    reference_hashes = {name: sha(regular(reference / name, 256 * 1024)) for name in REFERENCE_FILES}
    need(front_hashes == pins['frontend_sources_sha256'] and
         reference_hashes == pins['reference_sources_sha256'], 'EXACT_FRONTEND_AND_REFERENCE_SOURCE_BYTES')
    paths = [str((frontend / name).relative_to(pipeline)) for name in FRONTEND_FILES]
    paths += [str((reference / name).relative_to(pipeline)) for name in REFERENCE_FILES]
    paths += [str(Path(__file__).resolve().relative_to(pipeline))]
    subprocess.run(['git', '-C', str(pipeline), 'ls-files', '--error-unmatch', *paths],
                   check=True, stdout=subprocess.DEVNULL)
    exported, generated = static_export_archive(args, pins, front_hashes)
    for name, (size, digest) in ASSETS.items():
        raw = regular(args.assets / name, 2 * 1024**2)
        need(len(raw) == size and sha(raw) == digest, 'SAME_TWO_PINNED_ONNX_ASSETS')
    return reference, frontend, generated, {'profile': PROFILE, 'verification_pipeline_commit': commit,
        'static_export_source_commit': STATIC_SOURCE_HEAD, 'frontend_sources_sha256': front_hashes,
        'reference_sources_sha256': reference_hashes, 'static_export': pins['static_export'],
        'static_export_result': exported, 'consumed_original_static_generated_bytes': True,
        'head_used': False, 'head_weights_used': False, 'training_run_required': False}


def build_replays(frontend, generated, output):
    tools = {name: shutil.which(name) for name in ('gcc', 'arm-linux-gnueabihf-gcc', 'qemu-arm')}
    need(all(tools.values()), 'NATIVE_ARM_COMPILER_AND_QEMU_REQUIRED')
    programs, build = {}, {}
    for name, compiler, static, machine in (('native', 'gcc', [], 62),
                                           ('arm', 'arm-linux-gnueabihf-gcc', ['-static'], 40)):
        program = output / ('backbone-' + name)
        command = [tools[compiler], '-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off',
                   '-Wall', '-Wextra', '-Werror', *static, '-I', frontend, '-I', generated,
                   frontend / 'fixed_frontend.c', frontend / 'frontend_replay.c',
                   generated / 'fixed_models.c', '-lm', '-o', program]
        build[name] = timed_process(command, PLAN['compile_timeout_seconds'], output / ('compile-' + name + '.json'))
        raw = regular(program, 64 * 1024**2)
        need(raw[:4] == b'\x7fELF' and raw[5] == 1 and int.from_bytes(raw[18:20], 'little') == machine,
             'EXPECTED_NATIVE_OR_ARM_ELF')
        programs[name] = [str(program)] if name == 'native' else [tools['qemu-arm'], str(program)]
        build[name]['executable_sha256'] = sha(raw)
    build['tool_versions'] = {name: subprocess.check_output([path, '--version'], text=True).splitlines()[0]
                              for name, path in tools.items()}
    build['head_compiled_or_linked'] = False
    once(output / 'BUILD.json', build)
    return programs


def bounded_ort_call(session, inputs, accounting):
    need(accounting['onnx_session_attempts'] < PLAN['maximum_reference_onnx_session_runs'],
         'REFERENCE_SESSION_BUDGET_EXHAUSTED')
    accounting['onnx_session_attempts'] += 1
    values = session.run(None, inputs)
    accounting['onnx_session_completions'] += 1
    return values


def reference_trace(raw, embedding, causal, accounting):
    import numpy as np
    rows = []
    def callback(pcm):
        mel = bounded_ort_call(embedding.mel,
            {'input': np.asarray(pcm, dtype=np.float32).reshape(1, 16000)}, accounting)[0]
        need(mel.shape == (1, 1, 97, 32) and mel.dtype == np.float32 and bool(np.isfinite(mel).all()),
             'REFERENCE_MEL_GEOMETRY')
        normalized = ((mel[0, 0, -76:, :] / np.float32(10)) + np.float32(2)).reshape(1, 76, 32, 1)
        vector = bounded_ort_call(embedding.embedding, {'input_1': normalized}, accounting)[0]
        need(vector.shape == (1, 1, 1, 96) and vector.dtype == np.float32 and bool(np.isfinite(vector).all()),
             'REFERENCE_EMBEDDING_GEOMETRY')
        rows.append({'sample': 16000 + len(rows) * 1280, 1: mel.reshape(-1).tolist(),
                     2: normalized.reshape(-1).tolist(), 3: vector.reshape(-1).tolist()})
        accounting['reference_frontend_windows_completed'] += 1
        return vector.reshape(96)
    stream, observed_clocks = causal.CausalFeatureStream(callback), []
    cpu, start = time.process_time(), time.perf_counter()
    for chunk in chunks(raw):
        observed_clocks.extend(r['available_sample'] for r in stream.accept(chunk))
    need(stream.observed == 48000 and not stream.finish() and tuple(observed_clocks) == SAMPLES and
         tuple(row['sample'] for row in rows) == SAMPLES, 'REFERENCE_ACTUAL_CLOCKS_AND_NO_EOF_PADDING')
    return rows, {'wall_seconds': time.perf_counter() - start, 'process_cpu_seconds': time.process_time() - cpu,
        'scope': 'loaded ORT sessions plus Python feature stream; excludes session initialization and file I/O'}


def manifest(output):
    files = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(regular(p))}
             for p in sorted(output.rglob('*')) if p.is_file()}
    once(output / 'MANIFEST.json', {'schema': SCHEMA, 'profile': PROFILE, 'scope': PLAN['scope'],
        'plan': PLAN, 'plan_sha256': FIXED_PLAN_SHA, 'files': files, 'test_backend': False,
        'head_parity_established': False, 'kws_model_delivered': False, 'ssc305_qualification': False,
        **dict.fromkeys(FALSE_FLAGS, False)})


def execute(args):
    pins = read_json(args.pins)
    validate_pins(pins)
    execution_guard(pins, os.environ, read_json(Path(os.environ.get('GITHUB_EVENT_PATH', ''))),
                    sha(regular(Path(__file__), 256 * 1024)))
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'FRESH_BACKBONE_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'profile': PROFILE, 'run_id': os.environ['GITHUB_RUN_ID'],
        'github_sha': os.environ['GITHUB_SHA'], 'run_attempt': '1', 'pins_sha256': sha(canonical(pins)),
        'plan_sha256': FIXED_PLAN_SHA, 'plan': PLAN, 'test_backend': False, 'no_retry': True,
        'head_parity_established': False, 'kws_model_delivered': False, 'ssc305_qualification': False,
        **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    stage, reports = 'source-and-static-export-preflight', []
    accounting = {'onnx_session_attempts': 0, 'onnx_session_completions': 0,
        'reference_frontend_windows_completed': 0,
        'c_process_attempts': {'native': 0, 'arm': 0}, 'c_process_completions': {'native': 0, 'arm': 0},
        'c_frontend_windows_completed': {'native': 0, 'arm': 0},
        'head_evaluations': 0, 'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PLAN['total_timeout_seconds'])
    try:
        reference_source, frontend, generated, source = source_preflight(args, pins, os.environ['GITHUB_SHA'])
        once(output / 'SOURCE_BINDING.json', source)
        stage = 'original-static-bytes-and-backbone-only-build'
        (output / 'generated').mkdir(mode=0o700)
        for name, raw in generated.items():
            with (output / 'generated' / name).open('xb') as stream:
                stream.write(raw)
        programs = build_replays(frontend, output / 'generated', output)
        sys.path.insert(0, str(reference_source))
        causal = importlib.import_module('causal_embedding')
        need(all(Path(sys.modules[name].__file__).resolve() == reference_source / (name + '.py')
                 for name in ('causal_embedding', 'weak_embedding_contract')), 'PINNED_FEATURE_REFERENCE_MODULES')
        import numpy as np
        import onnx
        import onnxruntime as ort
        versions = {'numpy': np.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}
        need(versions == {'numpy': '1.26.4', 'onnx': '1.17.0', 'onnxruntime': '1.20.1'}, 'FIXED_REFERENCE_RUNTIME')
        once(output / 'RUNTIME.json', {'versions': versions, 'python': sys.version,
            'timing_is_profiling_only_not_matched_benchmark_or_target_measurement': True})
        embedding = causal.FrozenOnnxEmbedding(args.assets)
        once(output / 'GEOMETRY.json', {k: embedding.geometry[k] for k in
            ('status', 'convolutions', 'selected_mel_input_support', 'mel_frames', 'support_end_before_available')})
        for case, raw in synthetic_pcm().items():
            directory = output / case
            directory.mkdir()
            pcm = directory / 'input.pcm'
            pcm.write_bytes(raw)
            once(directory / 'STARTED.json', {'case': case, 'pcm_sha256': sha(raw), 'samples': 48000,
                'maximum_reference_session_runs': 52, 'embeddings_per_backend': 26, 'head_evaluations': 0})
            stage = 'reference:' + case
            reference, timing = reference_trace(raw, embedding, causal, accounting)
            (directory / 'reference.bin').write_bytes(trace_bytes(reference))
            reference = parse_trace(regular(directory / 'reference.bin', 1024**2))
            once(directory / 'reference-timing.json', timing)
            for backend, runner in programs.items():
                stage = backend + ':' + case
                need(accounting['c_process_attempts'][backend] < PLAN['maximum_c_processes_per_backend'],
                     'C_PROCESS_BUDGET_EXHAUSTED')
                once(directory / (backend + '-started.json'), {'case': case, 'backend': backend,
                    'head_enabled': False, 'no_retry': True, 'pcm_sha256': sha(raw)})
                accounting['c_process_attempts'][backend] += 1
                trace = directory / (backend + '.bin')
                timing = timed_process([*runner, pcm, trace, ','.join(map(str, CHUNKS))],
                    PLAN['replay_timeout_seconds'], directory / (backend + '-timing.json'))
                actual = parse_trace(regular(trace, 1024**2))
                summary = replay_summary(timing['stdout'], actual)
                accounting['c_process_completions'][backend] += 1
                accounting['c_frontend_windows_completed'][backend] += len(actual)
                report = {'case': case, 'backend': backend, 'pcm_sha256': sha(raw), 'embeddings': len(actual),
                    'backbone': comparisons(reference, actual), 'process_summary': summary, 'timing': timing,
                    'head_checked': False}
                report['passed'] = report['backbone']['passed']
                reports.append(report)
                once(directory / (backend + '-comparison.json'), report)
        need(accounting['onnx_session_attempts'] == accounting['onnx_session_completions'] == 156 and
             accounting['reference_frontend_windows_completed'] == 78 and len(reports) == 6 and
             all(accounting['c_process_attempts'][name] == accounting['c_process_completions'][name] == 3 and
                 accounting['c_frontend_windows_completed'][name] == 78 for name in ('native', 'arm')),
             'EXACT_PREDECLARED_BACKBONE_WORK_ACCOUNTING')
        passed = all(report['passed'] for report in reports)
        once(output / 'RESULT.json', {**started,
            'status': 'EXECUTED_BACKBONE_WITHIN_PREDECLARED_LIMITS' if passed else 'EXECUTED_BACKBONE_WITH_NUMERICAL_MISMATCH',
            'passed': passed, 'accounting': accounting, 'comparisons': reports, 'source_binding': source,
            'head_parity': 'NOT_RUN_NO_HEAD_IN_PROTOCOL', 'kws_weights': 'NOT_USED_OR_PRODUCED',
            'independent_frr': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'word_end_latency': 'NOT_RUN',
            'ssc305': 'NOT_RUN', 'arm_scope': 'FIXED_SHARED_BACKBONE_UNDER_HOST_QEMU_ONLY'})
        manifest(output)
        return 0 if passed else 2
    except Exception as error:
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': str(error)[:2000],
            'accounting': accounting, 'completed_comparisons': reports})
        manifest(output)
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--template', action='store_true')
    cli.add_argument('--pins', type=Path, default=HERE / 'backbone_parity_pins.json')
    for name in ('pipeline-source', 'frontend-source', 'assets', 'static-export-zip', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.execute:
        print(canonical({'schema': SCHEMA, 'profile': PROFILE, 'status': 'SOURCE_CONTRACT_ONLY',
            'execution_enabled': False, 'plan_sha256': FIXED_PLAN_SHA, 'plan': PLAN, 'model_calls': 0,
            'head_parity_established': False, 'kws_model_delivered': False, **dict.fromkeys(FALSE_FLAGS, False)}).decode())
        return 0
    need(all(getattr(args, name) is not None for name in
             ('pipeline_source', 'frontend_source', 'assets', 'static_export_zip', 'output')), 'EXPLICIT_INPUTS_REQUIRED')
    return execute(args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
