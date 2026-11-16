"""One bounded verification of a new exact-mel512 C implementation.

Consumes saved PCM, reference traces and generated C from one completed public
baseline. No new reference inference, model download, training or recognition.
Default mode is a source contract; execution requires the first public main
successor, explicit frozen pins, clean tracked sources and attempt one.
"""
from __future__ import annotations
import argparse
import hashlib
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
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
REPOSITORY = 'jiying2007/kws-pipeline'
PROFILE = 'EXACT_MEL512_C_SPECIALIZATION_SAVED_REFERENCE'
SCHEMA = 'english-exact-mel512-specialization-v1'
PINS_SCHEMA = 'english-exact-mel512-specialization-pins-v1'
BASELINE_HEAD = 'cd3e07ebc25505648d10166a806a05afade37703'
BASELINE_PLAN_SHA = '5657e4387909bf7b47ba4ed7adec723005a8428357bd75d3515c3ddfe26e550b'
BASELINE = {
    'run_id': '43829765872', 'github_sha': BASELINE_HEAD, 'run_attempt': 1,
    'artifact_id': 14208696183, 'artifact_name': 'english-backbone-parity-43829765872',
    'zip_bytes': 27306350,
    'zip_sha256': 'dfe28da9c5a35fef3768f7958ff7945e587645554d838dcbb4aa1d0508a9db1f',
    'result_member': 'backbone-parity-result/RESULT.json',
    'result_sha256': '75d48edbd790f17c55c887b4fb4af9980bdeea3f3925530183822eac89f63737',
    'zip_members': 61, 'manifest_files': 49,
}
FRONTEND_DIR = 'research/english_final_embedding_frontend'
PROFILE_DIR = 'research/english_mel_specialization'
CANDIDATE = PROFILE_DIR + '/fixed_frontend_mel.c'
HELPER_PATH = 'research/english_backbone_parity/backbone_parity_once.py'
FROZEN_SOURCES = {
    FRONTEND_DIR + '/fixed_frontend.c': 'c45821d94887d82a6b9f22375875210aacd9a80f6653a520f6761062bf2aee40',
    FRONTEND_DIR + '/fixed_frontend.h': '55ffd7edffd06744f9146088c4431635cd57498195cb05019df4e0d476e6cef9',
    FRONTEND_DIR + '/frontend_replay.c': 'a359e666f34fbe863dfa0ec3ca39be2b3adc8ce5bea62ce75c4ec596fb9f9535',
    HELPER_PATH: '5f077922dbebf018a026a96340e417464bcc71e248c36d9990a7de6e0933a376',
    'research/english_backbone_parity/backbone_parity_pins.json':
        '6c12d8d8bb49e9ec991dfff8272d70891c844939dd2bc0ef431aca48133f47b0',
}
GENERATED = {
    'fixed_models.c': {'bytes': 13464364, 'sha256': '9bf740a692370dfed02f034f87f6f0dfcaac42f3f83152c79fd27088ccc71941'},
    'fixed_models.h': {'bytes': 165, 'sha256': '0fb2bae330f87ea685e0ce713879872c3d4372c1544e62e9c58a539b00c2eeb1'},
    'export_manifest.json': {'bytes': 144095, 'sha256': 'b94549aec824a8eddb014e19f01629d8733c35f809d763927b712df845b5677b'},
}
CASES = ('zero', 'sine400', 'lcg14')
BACKENDS = ('native', 'arm')
PCM_SHA = {
    'zero': '55873fecc61a79e87ca550c7072e38ccdd7ecb600ace286fe4717952a97c42b0',
    'sine400': 'f81681f58003475f0e81e7625f1bf7d76fb018585d3757c0e3083af83c1a8b01',
    'lcg14': '4e97af4846861db16ce7461517d405f60dca4473b515ae1efcb0f505221142dc',
}
CHUNKS = (1, 1279, 257, 4096)
LIMITS = {'raw_mel': (0.01, 2e-5), 'normalized_mel': (0.001, 2e-5), 'embedding': (0.001, 2e-4)}
BUILD_FLAGS = ('-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror')
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
    'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed',
    'head_parity_established', 'kws_model_delivered', 'ssc305_qualification')
EXPECTED_SUMMARIES = {backend: {'samples': 48000, 'embeddings': 26, 'head_ready': 0,
    'events': 0, 'workspace_bytes': 528384, 'frontend_state_bytes': size,
    'head_enabled': False, 'ssc305_qualification': False}
    for backend, size in (('native', 32080), ('arm', 32056))}
PLAN = {
    'protocol': PROFILE,
    'purpose': 'NEW_C_IMPLEMENTATION_BYTE_EQUIVALENCE_AND_SINGLE_PASS_PROFILING',
    'baseline_run_id': BASELINE['run_id'], 'baseline_plan_sha256': BASELINE_PLAN_SHA,
    'candidate': 'only exact mel Conv [1,1,16000] x [257,1,512], stride160/dilation1/pad0/group1/no-bias',
    'scope': 'PCM_TO_MEL_TO_NORMALIZED_MEL_TO_96_EMBEDDING_WITHOUT_HEAD',
    'cases': list(CASES), 'pcm_sha256': PCM_SHA, 'pcm16_samples_per_case': 48000,
    'sample_rate': 16000, 'chunk_pattern_samples': list(CHUNKS),
    'first_embedding_sample': 16000, 'embedding_hop_samples': 1280,
    'reset_between_cases': True, 'eof_padding_samples': 0,
    'embedding_windows_per_backend': 78,
    'maximum_build_processes_per_backend': 1, 'maximum_c_processes_per_backend': 3,
    'new_reference_onnx_session_runs': 0, 'new_model_asset_downloads': 0,
    'old_implementation_processes': 0,
    'required_same_backend_baseline_trace_identity': 'BIT_FOR_BIT_INCLUDING_HEADERS_AND_SIGNED_ZERO',
    'error_rule': 'abs(candidate-saved_ORT) <= atol + rtol*abs(saved_ORT), every finite element',
    'limits': {key: {'atol': value[0], 'rtol': value[1]} for key, value in LIMITS.items()},
    'exact_embedding_clock_identity': True, 'unchanged_process_summaries': EXPECTED_SUMMARIES,
    'compile_flags': list(BUILD_FLAGS), 'compile_timeout_seconds': 180,
    'replay_timeout_seconds': 300, 'total_timeout_seconds': 1800,
    'profiling': 'whole-process elapsed and host child CPU; historical comparison is not a matched benchmark; ARM includes QEMU',
    'speed_threshold': None, 'threshold_search': False,
    'head_evaluations': 0, 'optimizer_updates': 0, 'tts_asr_phone_calls': 0,
    'real_audio_cases': 0, 'no_retry': True,
}
FIXED_PLAN_SHA = 'cc5cd38e2aa96b37672c10d61b8db2ac2364f3d794a0d9eb5f0afe961a2622aa'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def json_bytes(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def regular(path, maximum=64 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
         path.stat().st_size <= maximum, 'REGULAR_BOUNDED_FILE:' + str(path))
    raw = path.read_bytes()
    need(len(raw) <= maximum, 'FILE_GREW')
    return raw


def read_json(path):
    return json_bytes(regular(path, 32 * 1024**2))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def hex_id(value, size):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % size, value) is not None


def member_path(name):
    return type(name) is str and '\\' not in name and not PurePosixPath(name).is_absolute() and \
        all(part not in ('', '.', '..') for part in name.split('/'))


def pins_template():
    return {'schema': PINS_SCHEMA, 'profile': PROFILE, 'enabled': False,
        'main_before': BASELINE_HEAD, 'worker_code_sha256': None,
        'candidate_source_sha256': None, 'plan_sha256': FIXED_PLAN_SHA,
        'baseline': json.loads(canonical(BASELINE)), 'frozen_sources_sha256': FROZEN_SOURCES.copy(),
        'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}


def validate_pins(pins):
    need(type(pins) is dict and pins.get('enabled') is True, 'MEL_SPECIALIZATION_DISABLED')
    template = pins_template()
    dynamic = {'enabled', 'worker_code_sha256', 'candidate_source_sha256'}
    need(set(pins) == set(template) and
         all(canonical(pins[key]) == canonical(value) for key, value in template.items() if key not in dynamic) and
         sha(canonical(PLAN)) == pins['plan_sha256'] == FIXED_PLAN_SHA and
         hex_id(pins['worker_code_sha256'], 64) and hex_id(pins['candidate_source_sha256'], 64) and
         pins['candidate_source_sha256'] != FROZEN_SOURCES[FRONTEND_DIR + '/fixed_frontend.c'],
         'FIXED_NEW_IMPLEMENTATION_PLAN_AND_BASELINE')


def execution_guard(pins, env, event, worker_sha):
    """Pure guard: source tests supply dictionaries, never a spoofed execution env."""
    validate_pins(pins)
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPOSITORY and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and re.fullmatch('[1-9][0-9]*', env.get('GITHUB_RUN_ID', '')) and
         env['GITHUB_RUN_ID'] != BASELINE['run_id'] and event.get('ref') == 'refs/heads/main' and
         event.get('before') == pins['main_before'] == BASELINE_HEAD and
         event.get('after') == env.get('GITHUB_SHA') and hex_id(event.get('after'), 40) and
         event['after'] != BASELINE_HEAD and event.get('repository', {}).get('full_name') == REPOSITORY and
         event.get('repository', {}).get('private') is False and worker_sha == pins['worker_code_sha256'],
         'FIRST_NEW_PUBLIC_MAIN_ATTEMPT_ONLY')


def baseline_api_evidence(run, artifact):
    need(type(run.get('id')) is int and str(run['id']) == BASELINE['run_id'] and
         run.get('head_sha') == BASELINE_HEAD and type(run.get('run_attempt')) is int and run['run_attempt'] == 1 and
         run.get('event') == 'push' and run.get('head_branch') == 'main' and
         run.get('status') == 'completed' and run.get('conclusion') == 'success' and
         run.get('repository', {}).get('full_name') == REPOSITORY and
         run.get('repository', {}).get('private') is False, 'COMPLETED_ORIGINAL_PUBLIC_BASELINE_RUN')
    need(type(artifact.get('id')) is int and artifact['id'] == BASELINE['artifact_id'] and
         artifact.get('name') == BASELINE['artifact_name'] and
         type(artifact.get('size_in_bytes')) is int and artifact['size_in_bytes'] == BASELINE['zip_bytes'] and
         artifact.get('digest') == 'sha256:' + BASELINE['zip_sha256'] and artifact.get('expired') is False and
         artifact.get('workflow_run', {}).get('id') == int(BASELINE['run_id']) and
         artifact['workflow_run'].get('head_sha') == BASELINE_HEAD and
         artifact['workflow_run'].get('head_branch') == 'main', 'EXACT_ORIGINAL_BASELINE_ARTIFACT_METADATA')


def checked_archive(path, pin):
    raw = regular(path)
    need(identity(raw) == {'bytes': pin['zip_bytes'], 'sha256': pin['zip_sha256']}, 'EXACT_BASELINE_ZIP_BYTES')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members = archive.infolist()
    need(len(members) == pin['zip_members'] == len({m.filename for m in members}) and
         sum(m.file_size for m in members) <= 256 * 1024**2, 'UNIQUE_BOUNDED_BASELINE_MEMBERS')
    for item in members:
        name = item.filename[:-1] if item.is_dir() else item.filename
        need(member_path(name) and not item.flag_bits & 1 and
             stat.S_IFMT(item.external_attr >> 16) in
             ((0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)), 'SAFE_BASELINE_MEMBER')
    return archive


def archive_bytes(archive, member, limit=64 * 1024**2):
    info = archive.getinfo(member)
    need(not info.is_dir() and 0 < info.file_size <= limit, 'BOUNDED_REQUIRED_BASELINE_MEMBER')
    return archive.read(member)


def validate_baseline_result(result):
    expected = {'onnx_session_attempts': 156, 'onnx_session_completions': 156,
        'reference_frontend_windows_completed': 78,
        'c_process_attempts': dict.fromkeys(BACKENDS, 3),
        'c_process_completions': dict.fromkeys(BACKENDS, 3),
        'c_frontend_windows_completed': dict.fromkeys(BACKENDS, 78),
        'head_evaluations': 0, 'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}
    need(result.get('schema') == 'english-fixed-shared-backbone-parity-v1' and
         result.get('profile') == 'FIXED_SHARED_BACKBONE_ONLY' and
         result.get('status') == 'EXECUTED_BACKBONE_WITHIN_PREDECLARED_LIMITS' and
         result.get('run_id') == BASELINE['run_id'] and result.get('github_sha') == BASELINE_HEAD and
         result.get('run_attempt') == '1' and result.get('passed') is True and
         result.get('test_backend') is False and result.get('no_retry') is True and
         result.get('plan_sha256') == BASELINE_PLAN_SHA == sha(canonical(result.get('plan'))) and
         canonical(result.get('accounting')) == canonical(expected) and
         all(result.get(key) is False for key in FALSE_FLAGS), 'COMPLETE_ACTUAL_BASELINE_RESULT')
    reports = result.get('comparisons')
    need(type(reports) is list and len(reports) == 6 and
         {(r.get('case'), r.get('backend')) for r in reports} == {(c, b) for c in CASES for b in BACKENDS} and
         all(r.get('passed') is True and r.get('head_checked') is False and
             r.get('embeddings') == 26 and r.get('pcm_sha256') == PCM_SHA[r['case']] and
             r.get('backbone', {}).get('passed') is True and
             canonical(r.get('process_summary')) == canonical(EXPECTED_SUMMARIES[r['backend']])
             for r in reports), 'SIX_COMPLETE_BASELINE_COMPARISONS')


def read_baseline(path, helper):
    """Read and bind saved evidence only. Never imports or calls a model runtime."""
    with checked_archive(path, BASELINE) as archive:
        base = str(PurePosixPath(BASELINE['result_member']).parent)
        result_raw = archive_bytes(archive, BASELINE['result_member'], 1024**2)
        need(sha(result_raw) == BASELINE['result_sha256'] and
             base + '/FAILURE.json' not in archive.namelist(), 'EXACT_SUCCESSFUL_BASELINE_RESULT_BYTES')
        result = json_bytes(result_raw)
        validate_baseline_result(result)
        manifest_raw = archive_bytes(archive, base + '/MANIFEST.json', 1024**2)
        manifest = json_bytes(manifest_raw)
        files = manifest.get('files')
        expected_paths = {name[len(base) + 1:] for name in archive.namelist()
            if name.startswith(base + '/') and name != base + '/MANIFEST.json'}
        need(manifest.get('schema') == result['schema'] and
             manifest.get('profile') == result['profile'] and manifest.get('plan_sha256') == BASELINE_PLAN_SHA and
             canonical(manifest.get('plan')) == canonical(result['plan']) and type(files) is dict and
             len(files) == BASELINE['manifest_files'] and set(files) == expected_paths and
             all(manifest.get(key) is False for key in FALSE_FLAGS), 'CLOSED_BASELINE_MANIFEST')
        for name, entry in files.items():
            need(member_path(name) and type(entry) is dict and set(entry) == {'bytes', 'sha256'} and
                 canonical(identity(archive_bytes(archive, base + '/' + name))) == canonical(entry),
                 'BASELINE_MANIFEST_EXACT_BYTES:' + name)
        binding = json_bytes(archive_bytes(archive, base + '/SOURCE_BINDING.json', 1024**2))
        need(canonical(binding) == canonical(result.get('source_binding')) and
             binding.get('verification_pipeline_commit') == BASELINE_HEAD and
             binding.get('consumed_original_static_generated_bytes') is True and
             binding.get('head_used') is False and binding.get('head_weights_used') is False,
             'BASELINE_ORIGINAL_SOURCE_BINDING')
        for name in ('fixed_frontend.c', 'fixed_frontend.h', 'frontend_replay.c'):
            need(binding['frontend_sources_sha256'][name] == FROZEN_SOURCES[FRONTEND_DIR + '/' + name],
                 'BASELINE_FRONTEND_SOURCE_IDENTITY')
        generated = {name: archive_bytes(archive, base + '/generated/' + name) for name in GENERATED}
        need(all(canonical(identity(raw)) == canonical(GENERATED[name]) and
                 canonical(identity(raw)) == canonical({key: binding['static_export']['generated'][name][key]
                     for key in ('bytes', 'sha256')}) for name, raw in generated.items()),
             'ORIGINAL_GENERATED_C_HEADER_MANIFEST_BYTES')
        graph_manifest = json_bytes(generated['export_manifest.json'])
        need(graph_manifest.get('generated_c_sha256') == sha(generated['fixed_models.c']) and
             graph_manifest.get('schema') == 'fixed-feature-pure-export-v1' and
             graph_manifest.get('real_inference_performed') is False and
             graph_manifest.get('numerical_parity_established') is False and
             graph_manifest.get('ssc305_qualification') is False, 'ORIGINAL_GENERATED_MANIFEST')
        cases = {}
        for case in CASES:
            pcm = archive_bytes(archive, base + '/' + case + '/input.pcm', 96000)
            need(len(pcm) == 96000 and sha(pcm) == PCM_SHA[case], 'ORIGINAL_ARTIFICIAL_PCM_BYTES')
            reference = archive_bytes(archive, base + '/' + case + '/reference.bin', 1024**2)
            helper.parse_trace(reference)
            entry = {'pcm': pcm, 'reference': reference, 'old': {}}
            for backend in BACKENDS:
                trace = archive_bytes(archive, base + '/' + case + '/' + backend + '.bin', 1024**2)
                rows = helper.parse_trace(trace)
                timing = json_bytes(archive_bytes(archive, base + '/' + case + '/' + backend + '-timing.json', 1024**2))
                report = next(r for r in result['comparisons'] if r['case'] == case and r['backend'] == backend)
                need(canonical(timing) == canonical(report['timing']) and
                     canonical(helper.replay_summary(timing['stdout'], rows)) == canonical(EXPECTED_SUMMARIES[backend]),
                     'BASELINE_RAW_TRACE_AND_TIMING_RECEIPTS')
                valid_timing(timing)
                entry['old'][backend] = {'trace': trace, 'timing': timing,
                    'process_summary': report['process_summary']}
            cases[case] = entry
    return {'result': result, 'result_identity': identity(result_raw), 'manifest_identity': identity(manifest_raw),
        'generated': generated, 'cases': cases, 'closed_manifest_files': len(files)}


def load_trace_helpers(pipeline):
    path = Path(pipeline) / HELPER_PATH
    need(sha(regular(path, 1024**2)) == FROZEN_SOURCES[HELPER_PATH], 'FROZEN_PURE_TRACE_HELPER_SOURCE')
    spec = importlib.util.spec_from_file_location('_frozen_backbone_trace_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.PROFILE == 'FIXED_SHARED_BACKBONE_ONLY' and module.FIXED_PLAN_SHA == BASELINE_PLAN_SHA and
         sha(canonical(module.PLAN)) == BASELINE_PLAN_SHA and module.LIMITS == LIMITS and
         tuple(module.CHUNKS) == CHUNKS and module.PCM_SHA == PCM_SHA, 'UNCHANGED_TRACE_CLOCK_AND_ERROR_LIMITS')
    return module


def source_preflight(pipeline, pins, expected_head, pins_path):
    pipeline = Path(pipeline).absolute()
    need(pipeline.is_dir() and Path(__file__).resolve() == pipeline / PROFILE_DIR / 'mel_specialization_once.py' and
         Path(pins_path).resolve() == pipeline / PROFILE_DIR / 'mel_specialization_pins.json',
         'EXECUTED_WORKER_AND_PINS_IN_FIXED_CHECKOUT_PATHS')
    head = subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip()
    need(head == expected_head and head != BASELINE_HEAD and not subprocess.check_output(
        ['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(), 'EXACT_CLEAN_EXECUTION_HEAD')
    sources = {name: identity(regular(pipeline / name, 1024**2)) for name in FROZEN_SOURCES}
    need(all(sources[name]['sha256'] == digest for name, digest in FROZEN_SOURCES.items()),
         'ORIGINAL_FRONTEND_AND_BASELINE_SOURCES_UNCHANGED')
    candidate = identity(regular(pipeline / CANDIDATE, 1024**2))
    need(candidate['sha256'] == pins['candidate_source_sha256'], 'EXACT_NEW_CANDIDATE_BYTES')
    worker = identity(regular(Path(__file__), 1024**2))
    need(worker['sha256'] == pins['worker_code_sha256'], 'EXACT_WORKER_BYTES')
    paths = [*FROZEN_SOURCES, CANDIDATE, PROFILE_DIR + '/mel_specialization_once.py',
             PROFILE_DIR + '/mel_specialization_pins.json']
    subprocess.run(['git', '-C', str(pipeline), 'ls-files', '--error-unmatch', *paths],
                   check=True, stdout=subprocess.DEVNULL, timeout=30)
    return {'execution_pipeline_commit': head, 'baseline_pipeline_commit': BASELINE_HEAD,
        'frozen_sources': sources, 'candidate': {CANDIDATE: candidate},
        'worker': {PROFILE_DIR + '/mel_specialization_once.py': worker},
        'pins_file_identity': identity(regular(pins_path, 1024**2)),
        'baseline': BASELINE, 'generated': GENERATED, 'head_compiled_or_linked': False}


def valid_timing(value):
    need(type(value.get('returncode')) is int and value['returncode'] == 0 and
         all(type(value.get(key)) in (int, float) and math.isfinite(value[key]) and value[key] > 0
             for key in ('wall_seconds', 'host_child_cpu_seconds')), 'VALID_COMPLETED_PROCESS_TIMING')


def bit_identity(old, new):
    equal = old == new
    position = None if equal else next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
    return {'passed': equal, 'old_trace': identity(old), 'new_trace': identity(new),
            'first_different_byte': position, 'rule': 'SAME_BACKEND_FULL_TRACE_BYTES'}


def compare_candidate(helper, case, backend, raw, timing, saved):
    reference = helper.parse_trace(saved['reference'])
    rows = helper.parse_trace(raw)
    summary = helper.replay_summary(timing['stdout'], rows)
    unchanged = canonical(summary) == canonical(saved['old'][backend]['process_summary']) == canonical(EXPECTED_SUMMARIES[backend])
    exact = bit_identity(saved['old'][backend]['trace'], raw)
    numeric = helper.comparisons(reference, rows)
    valid_timing(timing)
    prior = saved['old'][backend]['timing']
    valid_timing(prior)
    return {'case': case, 'backend': backend, 'embeddings': len(rows), 'pcm_sha256': sha(saved['pcm']),
        'saved_reference_trace': identity(saved['reference']), 'bit_identity': exact,
        'saved_ort_comparison': numeric, 'process_summary': summary, 'unchanged_workspace_and_state': unchanged,
        'timing': timing, 'baseline_timing': prior,
        'historical_wall_ratio_baseline_over_candidate': prior['wall_seconds'] / timing['wall_seconds'],
        'historical_cpu_ratio_baseline_over_candidate': prior['host_child_cpu_seconds'] / timing['host_child_cpu_seconds'],
        'profiling_scope': PLAN['profiling'], 'speed_is_not_an_acceptance_gate': True,
        'head_checked': False, 'passed': exact['passed'] and numeric['passed'] and unchanged}


def fresh_accounting():
    return {'new_onnx_session_attempts': 0, 'new_onnx_session_completions': 0,
        'new_model_asset_downloads': 0, 'old_implementation_processes': 0,
        'build_attempts': dict.fromkeys(BACKENDS, 0), 'build_completions': dict.fromkeys(BACKENDS, 0),
        'c_process_attempts': dict.fromkeys(BACKENDS, 0), 'c_process_completions': dict.fromkeys(BACKENDS, 0),
        'c_frontend_windows_completed': dict.fromkeys(BACKENDS, 0),
        'head_evaluations': 0, 'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}


def reserve_attempt(accounting, kind, backend):
    need(backend in BACKENDS and kind in ('build', 'c_process'), 'DECLARED_PROCESS_KIND_AND_BACKEND')
    limit = 1 if kind == 'build' else PLAN['maximum_c_processes_per_backend']
    need(accounting[kind + '_attempts'][backend] < limit, 'PROCESS_BUDGET_EXHAUSTED')
    accounting[kind + '_attempts'][backend] += 1


def timed_process(command, timeout, log):
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    process, stdout, stderr = None, '', ''
    error = None
    try:
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
            env={k: v for k, v in os.environ.items() if k not in ('GH_TOKEN', 'GITHUB_TOKEN')})
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as caught:
        error = caught
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {'status': 'COMPLETED' if error is None else 'FAILED_OR_INTERRUPTED_NO_RETRY',
        'returncode': process.returncode if process is not None else None,
        'wall_seconds': time.perf_counter() - start,
        'host_child_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        'stdout': stdout[:32768], 'stderr': stderr[:32768], 'timeout_seconds': timeout,
        'scope': PLAN['profiling'], 'command': [str(x) for x in command], 'retry_count': 0,
        'error_type': type(error).__name__ if error else None}
    once(log, report)
    if error is not None:
        raise error
    need(report['returncode'] == 0, 'CHILD_PROCESS_FAILED:' + Path(log).name)
    return report


def build_command(compiler, backend, pipeline, generated, program):
    need(backend in BACKENDS, 'DECLARED_BUILD_BACKEND')
    pipeline, generated = Path(pipeline), Path(generated)
    frontend = pipeline / FRONTEND_DIR
    return [compiler, *BUILD_FLAGS, *(['-static'] if backend == 'arm' else []),
        '-I', frontend, '-I', generated, pipeline / CANDIDATE,
        frontend / 'frontend_replay.c', generated / 'fixed_models.c', '-lm', '-o', program]


def build_replays(pipeline, generated, output, accounting):
    tools = {name: shutil.which(name) for name in ('gcc', 'arm-linux-gnueabihf-gcc', 'qemu-arm')}
    need(all(tools.values()), 'NATIVE_ARM_COMPILER_AND_QEMU_REQUIRED')
    programs, build = {}, {}
    for backend, compiler, machine in (('native', 'gcc', 62), ('arm', 'arm-linux-gnueabihf-gcc', 40)):
        program = output / ('mel-specialization-' + backend)
        command = build_command(tools[compiler], backend, pipeline, generated, program)
        reserve_attempt(accounting, 'build', backend)
        once(output / ('compile-' + backend + '-started.json'), {'backend': backend,
            'command': [str(x) for x in command], 'no_retry': True, 'attempt': 1, 'head_enabled': False})
        build[backend] = timed_process(command, PLAN['compile_timeout_seconds'], output / ('compile-' + backend + '.json'))
        accounting['build_completions'][backend] += 1
        raw = regular(program)
        need(len(raw) >= 20 and raw[:4] == b'\x7fELF' and raw[5] == 1 and
             int.from_bytes(raw[18:20], 'little') == machine, 'EXPECTED_NATIVE_OR_ARM_ELF')
        programs[backend] = [str(program)] if backend == 'native' else [tools['qemu-arm'], str(program)]
        build[backend]['executable_identity'] = identity(raw)
    build['tool_versions'] = {name: subprocess.check_output([path, '--version'], text=True, timeout=10).splitlines()[0]
                             for name, path in tools.items()}
    build['head_compiled_or_linked'] = False
    once(output / 'BUILD.json', build)
    return programs


def output_manifest(output):
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.json':
            files[path.relative_to(output).as_posix()] = identity(regular(path))
    once(output / 'MANIFEST.json', {'schema': SCHEMA, 'profile': PROFILE, 'plan_sha256': FIXED_PLAN_SHA,
        'plan': PLAN, 'files': files, **dict.fromkeys(FALSE_FLAGS, False)})


def timeout_handler(_signum, _frame):
    raise TimeoutError('TOTAL_SPECIALIZATION_DEADLINE')


def execute(args):
    pins = read_json(args.pins)
    execution_guard(pins, os.environ, read_json(os.environ.get('GITHUB_EVENT_PATH', '')),
                    sha(regular(Path(__file__), 1024**2)))
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'FRESH_SPECIALIZATION_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'profile': PROFILE, 'run_id': os.environ['GITHUB_RUN_ID'],
        'github_sha': os.environ['GITHUB_SHA'], 'run_attempt': '1',
        'pins_sha256': sha(canonical(pins)), 'plan_sha256': FIXED_PLAN_SHA, 'plan': PLAN,
        'test_backend': False, 'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    accounting, reports, stage = fresh_accounting(), [], 'source-and-saved-baseline-preflight'
    handlers = {signum: signal.signal(signum, timeout_handler) for signum in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(PLAN['total_timeout_seconds'])
    try:
        source = source_preflight(args.pipeline_source, pins, os.environ['GITHUB_SHA'], args.pins)
        run_raw, artifact_raw = regular(args.baseline_run_json, 4 * 1024**2), regular(args.baseline_artifact_json, 4 * 1024**2)
        baseline_api_evidence(json_bytes(run_raw), json_bytes(artifact_raw))
        helper = load_trace_helpers(args.pipeline_source)
        baseline = read_baseline(args.baseline_zip, helper)
        source.update({'baseline_run_api': identity(run_raw), 'baseline_artifact_api': identity(artifact_raw),
            'baseline_result': baseline['result_identity'], 'baseline_manifest': baseline['manifest_identity'],
            'closed_baseline_manifest_files': baseline['closed_manifest_files'],
            'saved_reference_only': True, 'new_onnx_session_runs': 0, 'new_model_asset_downloads': 0})
        once(output / 'SOURCE_BINDING.json', source)
        once(output / 'BASELINE_RESULT.json', baseline['result'])
        (output / 'generated').mkdir(mode=0o700)
        for name, raw in baseline['generated'].items():
            with (output / 'generated' / name).open('xb') as stream:
                stream.write(raw)
        stage = 'new-candidate-only-build'
        programs = build_replays(args.pipeline_source, output / 'generated', output, accounting)
        for case in CASES:
            saved, directory = baseline['cases'][case], output / case
            directory.mkdir(mode=0o700)
            for name, raw in {'input.pcm': saved['pcm'], 'reference.bin': saved['reference'],
                **{backend + '-baseline.bin': saved['old'][backend]['trace'] for backend in BACKENDS}}.items():
                with (directory / name).open('xb') as stream:
                    stream.write(raw)
            for backend in BACKENDS:
                stage = backend + ':' + case
                trace = directory / (backend + '.bin')
                reserve_attempt(accounting, 'c_process', backend)
                once(directory / (backend + '-started.json'), {'case': case, 'backend': backend,
                    'pcm': identity(saved['pcm']), 'old_trace': identity(saved['old'][backend]['trace']),
                    'attempt': 1, 'no_retry': True, 'head_enabled': False,
                    'candidate_source_sha256': pins['candidate_source_sha256']})
                timing = timed_process([*programs[backend], directory / 'input.pcm', trace, ','.join(map(str, CHUNKS))],
                    PLAN['replay_timeout_seconds'], directory / (backend + '-timing.json'))
                accounting['c_process_completions'][backend] += 1
                raw = regular(trace, 1024**2)
                report = compare_candidate(helper, case, backend, raw, timing, saved)
                accounting['c_frontend_windows_completed'][backend] += report['embeddings']
                reports.append(report)
                once(directory / (backend + '-comparison.json'), report)
        need(len(reports) == 6 and all(accounting['build_attempts'][b] == accounting['build_completions'][b] == 1 and
             accounting['c_process_attempts'][b] == accounting['c_process_completions'][b] == 3 and
             accounting['c_frontend_windows_completed'][b] == 78 for b in BACKENDS) and
             all(accounting[key] == 0 for key in ('new_onnx_session_attempts', 'new_onnx_session_completions',
                 'new_model_asset_downloads', 'old_implementation_processes', 'head_evaluations',
                 'optimizer_updates', 'new_tts_asr_phone_calls', 'real_audio_cases')), 'EXACT_NEW_IMPLEMENTATION_BUDGET')
        passed = all(report['passed'] for report in reports)
        signal.alarm(0)
        once(output / 'RESULT.json', {**started,
            'status': 'EXECUTED_NEW_MEL_IMPLEMENTATION_WITHIN_FIXED_LIMITS' if passed else 'EXECUTED_NEW_MEL_IMPLEMENTATION_WITH_MISMATCH',
            'passed': passed, 'accounting': accounting, 'comparisons': reports, 'source_binding': source,
            'baseline_implementation_reexecuted': False, 'reference_inference_reexecuted': False,
            'head_parity': 'NOT_RUN_NO_HEAD', 'ssc305': 'NOT_RUN', 'hardware_realtime': 'NOT_ESTABLISHED',
            'independent_far': 'NOT_RUN', 'independent_frr': 'NOT_RUN', 'word_end_latency': 'NOT_RUN'})
        output_manifest(output)
        return 0 if passed else 2
    except BaseException as error:
        signal.alarm(0)
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': str(error)[:2000],
            'accounting': accounting, 'completed_comparisons': reports})
        output_manifest(output)
        return 2
    finally:
        signal.alarm(0)
        for signum, old in handlers.items():
            signal.signal(signum, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--template', action='store_true')
    parser.add_argument('--pins', type=Path, default=HERE / 'mel_specialization_pins.json')
    for name in ('pipeline-source', 'baseline-zip', 'baseline-run-json', 'baseline-artifact-json', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), sort_keys=True, indent=2))
        return 0
    if not args.execute:
        need(sha(canonical(PLAN)) == FIXED_PLAN_SHA, 'FIXED_SPECIALIZATION_PLAN')
        print(canonical({'schema': SCHEMA, 'profile': PROFILE, 'status': 'SOURCE_CONTRACT_ONLY',
            'execution_enabled': False, 'plan_sha256': FIXED_PLAN_SHA, 'plan': PLAN,
            'new_onnx_session_runs': 0, 'c_processes': 0, **dict.fromkeys(FALSE_FLAGS, False)}).decode())
        return 0
    need(all(getattr(args, key) is not None for key in
        ('pipeline_source', 'baseline_zip', 'baseline_run_json', 'baseline_artifact_json', 'output')), 'EXPLICIT_INPUTS_REQUIRED')
    return execute(args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
