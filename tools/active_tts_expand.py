#!/usr/bin/env python3
"""Prepare bounded D1 continuation jobs; preparation never executes a model.

The original 256-cell plan and first eight outputs are immutable inputs. A
single ledger under the pinned runtime prevents another stage from consuming
the same cell again. A claimed cell without a terminal artifact blocks reuse;
quality failures with retained raw output are terminal and are never retried.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import struct
import subprocess
import sys
import time
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_tts_smoke as common
from research_job_lock import heavy_model_slot

SCHEMA = 'active-tts-extension-stage-v1'
JOB_SCHEMA = 'active-tts-extension-job-v1'
MAX_CHUNK = 28
MAX_STAGE_CALLS = 56
STAGE_MINUTES = 20
JOB_SECONDS = 600
RECOVERY_QUEUE_MINUTES = 19.99
RECOVERY_QUEUE_NS = 1_199_400_000_000
OWN_SOURCES = ('tools/active_tts_expand.py', 'tools/active_tts_smoke.py',
               'tools/research_job_lock.py', 'tools/research_no_network.py', 'tools/research_runtime.py')
ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z')
CELL_ID = re.compile(r'bulk-[0-9a-f]{24}\Z')
OUTPUTS = ('summary.json', 'inventory.json', 'invocation.json', 'outcomes.json',
           'network-guard.json', 'model-loaded.json', 'generation/plan-lock.json')
ASR_INPUT_CONTRACT = {'channels': 1, 'sample_rate_hz': 24000, 'encoding': 'IEEE_FLOAT32',
                      'minimum_frames': 1, 'maximum_frames': 720000, 'all_finite': True,
                      'maximum_absolute_value': 1_000_000}


def require(value, label):
    common.require(value, label)


def digest(value):
    return hashlib.sha256(common.canonical(value)).hexdigest()


def queue_digest(value):
    # research_queue canonical bytes include a trailing newline. Bulk plan and
    # cell identities intentionally keep their existing no-newline convention.
    return hashlib.sha256(common.canonical(value) + b'\n').hexdigest()


def absolute(path):
    path = Path(path).absolute()
    require(not any(item.is_symlink() for item in (path, *path.parents)), 'PATH_ALIAS')
    return path


def atomic_bytes(path, body):
    """Publish complete fsynced bytes exactly once, including across processes."""
    path = absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def once(path, value):
    atomic_bytes(path, common.canonical(value) + b'\n')


@contextmanager
def ledger_lock(root):
    root = absolute(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = absolute(root / 'writer.lock')
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'LEDGER_LOCK_NOT_REGULAR')
    with os.fdopen(descriptor, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_files(root, records):
    root = absolute(root)
    for name, expected in records.items():
        path = absolute(root / common.relative(name))
        require(common.record(path) == expected, 'RETAINED_FILE_CHANGED:' + name)


def expected_start(cell, ref, plan_sha):
    return {'schema': 'kws-bulk-clone-start-v1', 'cell_id': cell['cell_id'],
            'reference_id': ref['reference_id'], 'reference_wav_sha256': ref['wav_sha256'],
            'plan_sha256': plan_sha, 'source_family_id': cell['source_family_id'],
            'seed': cell['seed'], 'attempts': 1, 'training_admitted': False}


def cell_identity(binding, cell, ref):
    require(type(cell['cell_id']) is str and CELL_ID.fullmatch(cell['cell_id']), 'SAFE_CELL_ID_REQUIRED')
    return digest({'binding': binding, 'cell': cell, 'reference': ref})


def verify_parent(recipe_path, recipe_sha):
    recipe_path = absolute(recipe_path)
    require(common.sha_file(recipe_path) == recipe_sha, 'BASE_RECIPE_CHANGED')
    recipe = common.read(recipe_path)
    require(recipe['schema'] == 'active-tts-smoke-v1' and recipe['kind'] == 'd1-first-eight' and
            recipe['max_calls'] == 8 and recipe['max_retries'] == 0 and recipe['threads'] == 4 and
            recipe['timeout_seconds'] == 600 and recipe['queue_supervision_required'] is True and
            recipe['generation'] == common.GENERATION and
            recipe['network_policy'] == 'native-seccomp-deny-before-ML-import-v1', 'BASE_RECIPE_SCOPE')
    prepared = absolute(recipe['prepared'])
    for key, name in (('plan', 'plan.json'), ('source_manifest', 'source-manifest.json'),
                      ('model_assets', 'model-assets.json'), ('runtime_setup', 'runtime-setup.json'),
                      ('runtime_preflight', 'runtime-preflight.json')):
        require(common.record(prepared / name) == recipe[key], 'BASE_INPUT_CHANGED:' + name)
    require(set(recipe['producer_source']) == set(common.SOURCE_FILES), 'BASE_PRODUCER_ROSTER')
    check_files(prepared / 'producer-source', recipe['producer_source'])
    # Validate the retained snapshot even if the original development checkout
    # has moved. Executed shared helpers must still be byte-identical.
    entries = [Path(name) for name in recipe['execution_source'] if name.endswith('/tools/active_tts_smoke.py')]
    require(len(entries) == 1, 'BASE_EXECUTION_SOURCE_ENTRY')
    origin = entries[0].parents[1]
    for name, expected in recipe['execution_source'].items():
        relative = Path(name).relative_to(origin)
        require(common.record(prepared / 'execution-source' / relative) == expected,
                'BASE_EXECUTION_SNAPSHOT_CHANGED')
        if str(relative) in OWN_SOURCES[1:]:
            require(common.record(common.ROOT / relative) == expected, 'SHARED_GENERATION_SOURCE_CHANGED')
    plan = common.read(prepared / 'plan.json')
    require(common.selection(plan) == recipe['selected_cells'], 'BASE_SELECTION_CHANGED')
    bulk = common.load_bulk(prepared / 'producer-source')
    common.verify_source_plan(bulk, prepared, plan)
    binding = {'plan_canonical_sha256': hashlib.sha256(bulk.p.canonical(plan)).hexdigest(),
               'plan_file_sha256': recipe['plan']['sha256'], 'model_assets_sha256': recipe['model_assets']['sha256'],
               'source_manifest_sha256': recipe['source_manifest']['sha256'],
               'producer_source_sha256': digest(recipe['producer_source']),
               'runtime_setup_sha256': recipe['runtime_setup']['sha256'], 'generation': recipe['generation']}
    return recipe, prepared, plan, bulk, binding


def verify_runtime(prepared):
    """Check pinned distribution metadata/source without relocating old receipts."""
    source = prepared / 'execution-source'
    manifest = common.research_runtime.load_manifest(source / 'configs/runtime/research-runtime.json')
    common.research_runtime.check_interpreter(manifest)
    common.research_runtime.check_packages(manifest, 'tts')
    expected = common.read(prepared / 'runtime-setup.json')
    require(expected['status'] == 'complete', 'TTS_SETUP_INCOMPLETE')
    for name, version in expected['installed_packages'].items():
        require(importlib.metadata.version(name) == version, 'RUNTIME_PACKAGE_CHANGED:' + name)
    for row in common.read(source / 'research/qwen6_tts/model-locks.json')['qwen_tts']['source_lock']:
        path = importlib.metadata.distribution(row['distribution']).locate_file(row['relative_path'])
        require(common.sha_file(path) == row['sha256'], 'TTS_API_SOURCE_CHANGED')


def verify_base_job(job_root, expected_result_sha, recipe_sha, plan, binding):
    """Authenticate all first-eight files through the supplied terminal pin."""
    job_root = absolute(job_root)
    require(common.sha_file(job_root / 'result.json') == expected_result_sha, 'BASE_TERMINAL_CHANGED')
    result = common.read(job_root / 'result.json')
    require(result['status'] == 'completed' and result['exit_code'] == 0 and
            result['recipe_sha256'] == recipe_sha and
            result['claim_sha256'] == common.sha_file(job_root / 'claim.json'), 'BASE_JOB_NOT_COMPLETE')
    output = job_root / 'outputs'
    check_files(output, result['outputs'])
    inventory = common.read(output / 'inventory.json')['files']
    check_files(output, inventory)
    actual = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
    require(actual == set(inventory) | {'inventory.json', 'summary.json'}, 'BASE_OUTPUT_ROSTER_CHANGED')
    summary = common.read(output / 'summary.json')
    require(summary['generation_calls'] == summary['total_started'] == summary['total_completed'] == 8 and
            summary['recipe_sha256'] == recipe_sha, 'BASE_EIGHT_CALLS_REQUIRED')
    refs = {ref['reference_id']: ref for ref in plan['references']}
    records = {}
    for cell in plan['cells'][:8]:
        name = cell['cell_id']; ref = refs[cell['reference_id']]
        generated = output / 'generation/cells' / name
        native = output / 'native' / name
        require(common.read(generated / 'started.json') == expected_start(cell, ref, binding['plan_canonical_sha256']),
                'BASE_START_CHANGED')
        completed = common.read(generated / 'completed.json')
        require(completed['status'] == 'GENERATED_PENDING_ASR' and not (generated / 'failed.json').exists() and
                completed['audio']['native_sha256'] == common.sha_file(generated / 'native-f32.wav') and
                completed['audio']['wav_sha256'] == common.sha_file(generated / 'audio-16k.wav'), 'BASE_MEDIA_CHANGED')
        raw = common.read(native / 'native.json'); call = common.read(native / 'call.json')
        require(raw['array'] == common.record(native / 'returned-array.npy') and
                raw['wav'] == common.record(native / 'returned-native.wav') and
                call['generation'] == binding['generation'] and call['seed'] == cell['seed'] and
                call['reference_wav_sha256'] == ref['wav_sha256'] and call['reference_pcm_sha256'] == ref['pcm_sha256'],
                'BASE_RAW_OR_CONFIGURATION_CHANGED')
        files = {key: value for key, value in inventory.items()
                 if key.startswith('generation/cells/' + name + '/') or key.startswith('native/' + name + '/')}
        records[name] = {'schema': 'active-tts-plan-cell-v1', 'origin': 'reused-first-eight', 'cell_id': name,
            'identity_sha256': cell_identity(binding, cell, ref), 'status': 'GENERATED_PENDING_ASR',
            'output_root': str(output), 'files': files,
            'native_wav': 'generation/cells/' + name + '/native-f32.wav',
            'generation_root': str(output / 'generation'), 'base_terminal_sha256': expected_result_sha,
            'technical_failure_reason': None, 'new_generation_calls': 0, 'training_admitted': False}
    require({p.name for p in (output / 'generation/cells').iterdir()} == set(records), 'BASE_EXTRA_CELLS')
    return records


def verify_terminal_media(value):
    name = value['cell_id']
    require(type(name) is str and CELL_ID.fullmatch(name), 'SAFE_CELL_ID_REQUIRED')
    root = absolute(value['output_root'])
    require(value['generation_root'] == str(root / 'generation'), 'CELL_GENERATION_ROOT_CHANGED')
    check_files(root, value['files'])
    native = 'native/' + name + '/'
    generated = 'generation/cells/' + name + '/'
    required = {native + suffix for suffix in ('call.json', 'native.json', 'returned.json', 'returned-array.npy')}
    required.add(generated + 'started.json')
    require(required <= set(value['files']), 'CELL_RAW_OUTPUT_MISSING')
    require(all(key.startswith(native) or key.startswith(generated) for key in value['files']), 'CELL_FILE_SCOPE')
    actual = {path.relative_to(root).as_posix() for location in (root / native, root / generated)
              for path in location.rglob('*') if path.is_file()}
    require(actual == set(value['files']), 'CELL_FILE_ROSTER_CHANGED')
    raw = common.read(root / native / 'native.json')
    require(raw['schema'] == 'active-tts-native-return-v1' and
            raw['array'] == value['files'][native + 'returned-array.npy'] and
            raw['technical_gate_applied'] is False, 'CELL_NATIVE_METADATA_CHANGED')
    if raw['wav'] is None:
        require(value['native_wav'] is None and value['status'] == 'FAILED_NO_RETRY' and
                raw.get('wav_unavailable_reason') and native + 'returned-native.wav' not in value['files'],
                'MISSING_WAV_WITHOUT_EXPLICIT_FAILURE')
    else:
        require(raw['wav'] == value['files'][native + 'returned-native.wav'] and
                value['native_wav'] in value['files'], 'CELL_NATIVE_METADATA_CHANGED')
    complete = root / generated / 'completed.json'; failed = root / generated / 'failed.json'
    if value['status'] == 'GENERATED_PENDING_ASR':
        require(complete.is_file() and not failed.exists() and
                value['native_wav'] == generated + 'native-f32.wav', 'CELL_COMPLETION_CONTRADICTION')
        row = common.read(complete)
        require(row['schema'] == 'kws-bulk-clone-complete-v1' and row['cell_id'] == name and
                row['status'] == value['status'] and row['training_admitted'] is False and
                row['qualification_allowed'] is False and row['actual_text'] is None and
                row['audio']['native_sha256'] == value['files'][generated + 'native-f32.wav']['sha256'] and
                row['audio']['wav_sha256'] == value['files'][generated + 'audio-16k.wav']['sha256'],
                'CELL_COMPLETED_MEDIA_CHANGED')
    else:
        require(failed.is_file() and not complete.exists() and
                value['native_wav'] in (None, native + 'returned-native.wav'), 'CELL_FAILURE_CONTRADICTION')
        row = common.read(failed)
        require(row['schema'] == 'kws-bulk-clone-failure-v1' and row['cell_id'] == name and
                row['status'] == 'FAILED_NO_RETRY' and row['training_admitted'] is False,
                'CELL_FAILED_RECEIPT_CHANGED')


def retain_extension_native(folder, audio, rate):
    """Save the returned array even if its shape cannot form a decoder WAV."""
    import numpy as np
    value = np.asarray(audio)
    target = folder / 'returned-array.npy'
    with target.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    reason = None
    if value.ndim != 1 or value.dtype.kind != 'f':
        reason = 'NATIVE_ARRAY_NOT_ONE_DIMENSIONAL_FLOAT'
    elif type(rate) is not int or rate != 24000:
        reason = 'NATIVE_RATE_NOT_24000'
    wav_record = None
    if reason is None:
        import soundfile as sf
        wav = folder / 'returned-native.wav'
        with wav.open('xb') as stream:
            sf.write(stream, value, rate, format='WAV', subtype='FLOAT')
            stream.flush(); os.fsync(stream.fileno())
        wav_record = common.record(wav)
    result = {'schema': 'active-tts-native-return-v1', 'array': common.record(target),
              'wav': wav_record, 'dtype': value.dtype.str, 'shape': list(value.shape),
              'sample_rate_hz': rate, 'samples_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
              'all_finite': bool(np.isfinite(value).all()) if value.dtype.kind in 'fciu' else False,
              'technical_gate_applied': False}
    if reason is not None:
        result['wav_unavailable_reason'] = reason
    once(folder / 'native.json', result)
    return result


def asr_eligibility(value):
    """Check bounded decoder input geometry, independently of the quality gate."""
    if value['native_wav'] is None:
        return {'eligible': False, 'reason': 'NO_DECODABLE_NATIVE_WAV'}
    path = absolute(Path(value['output_root']) / common.relative(value['native_wav']))
    require(common.record(path) == value['files'][value['native_wav']], 'ASR_NATIVE_BYTES_CHANGED')
    try:
        require(44 <= path.stat().st_size <= 16_000_000, 'ASR_WAV_FILE_BOUND')
        body = path.read_bytes()
        require(body[:4] == b'RIFF' and body[8:12] == b'WAVE' and
                struct.unpack_from('<I', body, 4)[0] + 8 == len(body), 'ASR_RIFF_GEOMETRY')
        offset = 12; fmt = data = None
        while offset < len(body):
            require(offset + 8 <= len(body), 'ASR_RIFF_CHUNK_BOUND')
            name = body[offset:offset + 4]; size = struct.unpack_from('<I', body, offset + 4)[0]
            offset += 8
            require(offset + size <= len(body), 'ASR_RIFF_CHUNK_BOUND')
            chunk = body[offset:offset + size]
            if name == b'fmt ':
                require(fmt is None and len(chunk) >= 16, 'ASR_FORMAT_CHUNK')
                fmt = struct.unpack_from('<HHIIHH', chunk)
            elif name == b'data':
                require(data is None, 'ASR_DUPLICATE_DATA_CHUNK'); data = chunk
            offset += size + size % 2
        require(offset == len(body) and fmt == (3, 1, 24000, 96000, 4, 32) and data is not None,
                'ASR_MONO_FLOAT_24000_REQUIRED')
        require(len(data) % 4 == 0 and 1 <= len(data) // 4 <= ASR_INPUT_CONTRACT['maximum_frames'], 'ASR_FRAME_BOUND')
        values = struct.iter_unpack('<f', data); maximum = 0.0
        for (sample,) in values:
            require(math.isfinite(sample), 'ASR_NONFINITE_SAMPLE')
            maximum = max(maximum, abs(sample))
        # A very wide numeric bound prevents resampling/model overflow without
        # treating ordinary clipping or loudness failure as non-decodable.
        require(maximum <= ASR_INPUT_CONTRACT['maximum_absolute_value'], 'ASR_UNSAFE_NUMERIC_RANGE')
        return {'eligible': True, 'reason': None, 'frames': len(data) // 4,
                'native_wav_sha256': value['files'][value['native_wav']]['sha256']}
    except (ValueError, struct.error) as error:
        return {'eligible': False, 'reason': str(error)}


def read_cell(ledger, cell_id, identity):
    require(type(cell_id) is str and CELL_ID.fullmatch(cell_id), 'SAFE_CELL_ID_REQUIRED')
    path = absolute(Path(ledger) / 'cells' / (cell_id + '.json'))
    claim = absolute(Path(ledger) / 'claims' / (cell_id + '.json'))
    if path.exists():
        value = common.read(path)
        require(value['schema'] == 'active-tts-plan-cell-v1' and value['cell_id'] == cell_id and
                value['identity_sha256'] == identity and value['origin'] in ('reused-first-eight', 'extension-call') and
                value['training_admitted'] is False and value['status'] in
                ('GENERATED_PENDING_ASR', 'FAILED_NO_RETRY'), 'CELL_IDENTITY_CHANGED')
        if value['origin'] != 'reused-first-eight':
            require(claim.is_file() and common.sha_file(claim) == value['claim_sha256'], 'CELL_CLAIM_CHANGED')
            old = common.read(claim)
            require(old['schema'] == 'active-tts-plan-call-claim-v1' and
                    old['identity_sha256'] == identity and old['cell_id'] == cell_id and old['attempt'] == 1 and
                    value['new_generation_calls'] == 1,
                    'CELL_CLAIM_IDENTITY_CHANGED')
        else:
            require(not claim.exists() and value['new_generation_calls'] == 0, 'IMPORTED_CELL_CLAIM_CONTRADICTION')
            base = common.read(Path(ledger) / 'base.json')
            require(base['schema'] == 'active-tts-plan-first-eight-v1' and
                    base['records'].get(cell_id) == value and
                    value['base_terminal_sha256'] == base['base_terminal_sha256'], 'IMPORTED_CELL_NOT_IN_BASELINE')
        verify_terminal_media(value)
        return value
    if claim.exists():
        value = common.read(claim)
        require(value['cell_id'] == cell_id and value['identity_sha256'] == identity, 'CLAIM_IDENTITY_CHANGED')
        raise ValueError('CONSUMED_WITHOUT_TERMINAL:' + cell_id)
    return None


def partition_cells(plan, completed, target_count, chunk_size=MAX_CHUNK):
    cells = plan['cells']
    require(len(cells) == 256 and len({row['cell_id'] for row in cells}) == 256, 'UNIQUE_256_PLAN_REQUIRED')
    require(type(target_count) is int and 8 <= target_count <= 256 and
            type(chunk_size) is int and 1 <= chunk_size <= MAX_CHUNK, 'BOUNDED_TARGET_AND_CHUNK')
    require(set(completed) <= {row['cell_id'] for row in cells}, 'FOREIGN_COMPLETED_CELL')
    pending = [row['cell_id'] for row in cells[:target_count] if row['cell_id'] not in completed]
    require(len(pending) <= MAX_STAGE_CALLS, 'STAGE_MAXIMUM_56_NEW_CALLS')
    chunks = [pending[index:index + chunk_size] for index in range(0, len(pending), chunk_size)]
    require(len(chunks) <= 2, 'STAGE_MAXIMUM_TWO_SERIAL_JOBS')
    return chunks


def consume_cell(ledger, cell, ref, binding, owner, callback):
    """The callback is reached only after exclusive durable call publication."""
    identity = cell_identity(binding, cell, ref)
    cached = read_cell(ledger, cell['cell_id'], identity)
    if cached is not None:
        return cached, False
    claim_path = Path(ledger) / 'claims' / (cell['cell_id'] + '.json')
    claim = {'schema': 'active-tts-plan-call-claim-v1', 'cell_id': cell['cell_id'],
             'identity_sha256': identity, 'attempt': 1, 'owner': owner,
             'started_at': datetime.now(timezone.utc).isoformat()}
    once(claim_path, claim)  # Racing stages cannot both reach the callback.
    value = callback(cell, ref)
    value.update(schema='active-tts-plan-cell-v1', origin='extension-call', cell_id=cell['cell_id'],
                 identity_sha256=identity, claim_sha256=common.sha_file(claim_path),
                 new_generation_calls=1, training_admitted=False)
    require(value['status'] in ('GENERATED_PENDING_ASR', 'FAILED_NO_RETRY'), 'CELL_TERMINAL_STATUS')
    verify_terminal_media(value)
    once(Path(ledger) / 'cells' / (cell['cell_id'] + '.json'), value)
    return value, True


def job_recipe(stage, index):
    path = Path(stage['prepared']) / 'stage.json'
    return {'schema': JOB_SCHEMA, 'stage': str(path), 'stage_sha256': common.sha_file(path),
            'job_id': 'chunk-' + str(index + 1).zfill(3), 'chunk_index': index,
            'cells': stage['chunks'][index], 'maximum_new_calls': len(stage['chunks'][index]),
            'timeout_seconds': JOB_SECONDS}


def queue_request(stage):
    attempt = stage.get('infrastructure_attempt', 1)
    queue_minutes = stage.get('queue_budget_minutes', STAGE_MINUTES)
    require(stage.get('stage_budget_minutes', STAGE_MINUTES) == STAGE_MINUTES and
            ((attempt == 1 and queue_minutes == STAGE_MINUTES and stage.get('recovery') is None) or
             (attempt == 2 and queue_minutes == RECOVERY_QUEUE_MINUTES and stage.get('recovery') is not None)),
            'FIXED_STAGE_AND_RECOVERY_QUEUE_BUDGET')
    jobs = []
    for index in range(len(stage['chunks'])):
        recipe = job_recipe(stage, index)
        path = Path(stage['prepared']) / 'jobs' / (recipe['job_id'] + '.json')
        require(common.read(path) == recipe, 'STAGE_JOB_RECIPE_CHANGED')
        jobs.append({'job_id': recipe['job_id'], 'recipe': str(path), 'recipe_sha256': common.sha_file(path),
            'argv': [stage['runtime_python'], '-I', '-B', str(Path(stage['execution_root']) / OWN_SOURCES[0]),
                     'run', '--recipe', '{recipe}', '--output', '{output_dir}', '--heavy-lock', stage['heavy_lock']],
            'cwd': stage['execution_root'], 'depends_on': [] if index == 0 else ['chunk-' + str(index).zfill(3)],
            'timeout_minutes': JOB_SECONDS // 60, 'outputs': list(OUTPUTS)})
    return {'schema': 'kws-local-research-queue-v1', 'queue_id': stage['stage_id'],
            'global_budget_minutes': queue_minutes, 'jobs': jobs}


def queue_state(stage, *, require_created=False):
    """Read the actual queue's frozen request, claims and receipts, without running it."""
    request = queue_request(stage)
    request_path = Path(stage['prepared']) / 'queue-request.json'
    require(common.read(request_path) == request, 'STAGE_QUEUE_REQUEST_CHANGED')
    root = absolute(stage['queue_root'])
    if not root.exists():
        require(not require_created, 'DURABLE_QUEUE_REQUIRED')
        return request, {job['job_id']: {'status': 'not-started', 'actual_elapsed_ns': None}
                         for job in request['jobs']}
    queue = common.read(root / 'queue.json')
    require((root / 'request.json').read_bytes() == request_path.read_bytes() and
            queue['schema'] == request['schema'] and queue['request_sha256'] == common.sha_file(request_path) and
            queue['heavy_lock'] == stage['heavy_lock'] and queue['automatic_retries'] == 0 and
            queue['historical_jobs_imported'] == 0 and
            queue['active_plan_sha256'] == common.sha_file(root / 'active-plan.json'), 'FROZEN_QUEUE_CHANGED')
    queue_sha = common.sha_file(root / 'queue.json'); states = {}; actual_before = 0
    for definition in request['jobs']:
        name = definition['job_id']; folder = root / 'jobs' / name
        require(common.sha_file(root / 'recipes' / (name + '.json')) == definition['recipe_sha256'],
                'FROZEN_QUEUE_RECIPE_CHANGED')
        claim_path = folder / 'claim.json'; terminal = folder / 'result.json'
        if not claim_path.exists():
            require(not terminal.exists(), 'QUEUE_RESULT_WITHOUT_CLAIM')
            states[name] = {'status': 'not-started', 'actual_elapsed_ns': None}; continue
        binding = {'job_id': name, 'queue_sha256': queue_sha,
                   'job_definition_sha256': queue_digest(definition), 'recipe_sha256': definition['recipe_sha256']}
        claim = common.read(claim_path)
        require(claim['schema'] == 'kws-local-research-job-claim-v1' and
                all(claim.get(k) == v for k, v in binding.items()) and claim['attempt'] == 1 and
                claim['automatic_retry'] is False and type(claim['allowed_runner_nanoseconds']) is int and
                0 < claim['allowed_runner_nanoseconds'] <= JOB_SECONDS * 10**9 and
                claim['deadline_monotonic_ns'] == claim['started_monotonic_ns'] + claim['allowed_runner_nanoseconds'],
                'QUEUE_CLAIM_IDENTITY_OR_BUDGET_CHANGED')
        require(all(states[prior]['status'] == 'completed' for prior in definition['depends_on']),
                'QUEUE_PREREQUISITE_NOT_COMPLETE')
        queue_budget_ns = RECOVERY_QUEUE_NS if stage.get('infrastructure_attempt', 1) == 2 else STAGE_MINUTES * 60 * 10**9
        require(claim['global_actual_before_nanoseconds'] == actual_before and
                claim['allowed_runner_nanoseconds'] <= queue_budget_ns - actual_before,
                'QUEUE_CUMULATIVE_BUDGET_CHANGED')
        dependencies = {prior: {'receipt': str(root / 'jobs' / prior / 'result.json'),
            'receipt_sha256': common.sha_file(root / 'jobs' / prior / 'result.json'),
            'outputs_root': str(root / 'jobs' / prior / 'outputs'), 'outputs': states[prior]['outputs']}
            for prior in definition['depends_on']}
        require(common.read(folder / 'dependencies.json') == dependencies and
                claim['dependencies_sha256'] == queue_digest(dependencies), 'QUEUE_DEPENDENCIES_CHANGED')
        if not terminal.exists():
            states[name] = {'status': 'claimed-without-terminal', 'actual_elapsed_ns': None, 'claim': claim}
            continue
        row = common.read(terminal)
        require(row['schema'] == 'kws-local-research-job-result-v1' and
                row['claim_sha256'] == common.sha_file(claim_path) and
                all(row.get(k) == v for k, v in binding.items()) and
                row['status'] in ('completed', 'failed', 'timed_out', 'unknown'), 'QUEUE_TERMINAL_IDENTITY_CHANGED')
        elapsed = row['actual_elapsed_ns']
        require(elapsed is None if row['status'] == 'unknown' else type(elapsed) is int and elapsed >= 0,
                'QUEUE_ACTUAL_ELAPSED_INVALID')
        if elapsed is not None:
            actual_before += elapsed
        if row['status'] == 'completed':
            require(row['exit_code'] == 0 and set(row['outputs']) == set(OUTPUTS), 'QUEUE_COMPLETED_OUTPUTS')
            check_files(folder / 'outputs', row['outputs'])
        states[name] = {**row, 'claim': claim}
    return request, states


def freeze_history(path):
    """Retain the old unknown duration and actual completed queue receipts separately."""
    path = absolute(path); budget = common.read(path)
    require(budget['schema'] == 'active-tts-cross-attempt-budget-v1' and
            budget['actual_total_complete'] is False and budget['old_attempt']['status'] == 'unknown' and
            budget['old_attempt']['actual_elapsed_seconds'] is None, 'HISTORICAL_UNKNOWN_MUST_REMAIN_UNKNOWN')
    old = absolute(Path(budget['old_attempt']['queue']) / 'jobs/smoke8/result.json')
    require(common.sha_file(old) == budget['old_attempt']['terminal_receipt_sha256'] and
            common.read(old)['status'] == 'unknown' and common.read(old)['actual_elapsed_ns'] is None,
            'HISTORICAL_UNKNOWN_RECEIPT_CHANGED')
    receipts = {str(path): common.record(path), str(old): common.record(old)}; known = 0
    for job in budget['new_attempt']['jobs']:
        result = absolute(Path(budget['new_attempt']['queue']) / 'jobs' / job['id'] / 'result.json')
        row = common.read(result)
        require(row['recipe_sha256'] == job['recipe_sha256'] and row['status'] == 'completed' and
                type(row['actual_elapsed_ns']) is int and row['actual_elapsed_ns'] >= 0, 'HISTORICAL_TERMINAL_CHANGED')
        known += row['actual_elapsed_ns']; receipts[str(result)] = common.record(result)
    return {'schema': 'active-tts-extension-history-v1', 'files': receipts,
            'historical_known_job_nanoseconds': known, 'historical_actual_total_nanoseconds': None,
            'historical_actual_total_complete': False,
            'unknown_conservative_reservation_seconds': budget['old_attempt']['conservative_reservation_seconds'],
            'historical_conservative_allocation_seconds': budget['total_conservative_allocation_seconds'],
            'budget': budget}


def freeze_completed_base_history(job_root, result_sha, recipe_sha):
    """Record one verified successful base job, without inventing older attempts.

    The caller has already run verify_base_job against the complete first-eight
    media. This history describes that job only, never all project/runtime work.
    """
    root = absolute(job_root)
    result_path, claim_path = root / 'result.json', root / 'claim.json'
    require(common.sha_file(result_path) == result_sha, 'COMPLETED_BASE_TERMINAL_CHANGED')
    result, claim = common.read(result_path), common.read(claim_path)
    require(result['schema'] == 'kws-local-research-job-result-v1' and
            result['status'] == 'completed' and result['exit_code'] == 0 and
            result['recipe_sha256'] == recipe_sha and
            result['claim_sha256'] == common.sha_file(claim_path) and
            result['actual_duration_known'] is True and result['automatic_retry'] is False and
            type(result['actual_elapsed_ns']) is int and result['actual_elapsed_ns'] >= 0,
            'COMPLETED_BASE_ACTUAL_DURATION_REQUIRED')
    require(claim['schema'] == 'kws-local-research-job-claim-v1' and
            claim['job_id'] == root.name and claim['attempt'] == 1 and claim['automatic_retry'] is False and
            all(claim[key] == result[key] for key in
                ('job_id', 'queue_sha256', 'job_definition_sha256', 'recipe_sha256', 'allowed_runner_nanoseconds')) and
            claim['allowed_runner_nanoseconds'] == JOB_SECONDS * 10**9 and
            claim['deadline_monotonic_ns'] == claim['started_monotonic_ns'] + claim['allowed_runner_nanoseconds'],
            'COMPLETED_BASE_CLAIM_BINDING')
    known = result['actual_elapsed_ns']
    return {'schema': 'active-tts-completed-base-history-v1',
            'scope': 'verified-first-eight-job-only', 'base_job': str(root),
            'base_result_sha256': result_sha, 'base_recipe_sha256': recipe_sha,
            'files': {str(path): common.record(path) for path in (result_path, claim_path)},
            'historical_known_job_nanoseconds': known, 'historical_actual_total_nanoseconds': known,
            'historical_actual_total_complete': True, 'unknown_conservative_reservation_seconds': 0,
            'historical_conservative_allocation_seconds': JOB_SECONDS,
            'project_or_runtime_lifetime_total_claimed': False}


def freeze_recovery(path, stage_sha, result_sha):
    """Admit one separately named recovery only after a proven pre-model failure."""
    path = absolute(path)
    require(common.sha_file(path) == stage_sha, 'PRIOR_STAGE_CHANGED')
    prior = common.read(path)
    require(not (Path(prior['ledger']) / 'recoveries' / (stage_sha + '.json')).exists(),
            'RECOVERY_ALREADY_ASSIGNED_NO_THIRD_ATTEMPT')
    require(prior['schema'] == SCHEMA and prior.get('infrastructure_attempt', 1) == 1 and
            prior['stage_budget_minutes'] == STAGE_MINUTES and
            prior.get('queue_budget_minutes', STAGE_MINUTES) == STAGE_MINUTES and prior.get('recovery') is None,
            'ONLY_ONE_INFRASTRUCTURE_RECOVERY_ALLOWED')
    check_files(prior['execution_root'], prior['execution_source'])
    check_files(Path(prior['prepared']) / 'execution-source', prior['execution_source'])
    request, states = queue_state(prior, require_created=True)
    first_id = request['jobs'][0]['job_id']; first = states[first_id]
    result_path = absolute(Path(prior['queue_root']) / 'jobs' / first_id / 'result.json')
    require(common.sha_file(result_path) == result_sha and first['status'] == 'failed' and
            first['exit_code'] != 0 and type(first['actual_elapsed_ns']) is int and
            first['actual_elapsed_ns'] + RECOVERY_QUEUE_NS <= STAGE_MINUTES * 60 * 10**9 and
            all(states[job['job_id']]['status'] == 'not-started' for job in request['jobs'][1:]),
            'RECOVERY_REQUIRES_BOUNDED_PRE_MODEL_FAILURE')
    evidence = {str(path): common.record(path), str(result_path): common.record(result_path)}
    for job in request['jobs']:
        folder = Path(prior['queue_root']) / 'jobs' / job['job_id']
        require(not list((folder / 'outputs').iterdir()), 'PRIOR_MODEL_OR_OUTPUT_ACTIVITY')
    for name in ('claim.json', 'stderr.log', 'stdout.log'):
        file = result_path.parent / name; evidence[str(file)] = common.record(file)
    for name, expected in first['logs'].items():
        require(common.record(result_path.parent / name) == expected, 'PRIOR_FAILURE_LOG_CHANGED')
    for names in prior['chunks']:
        for name in names:
            require(not (Path(prior['ledger']) / 'claims' / (name + '.json')).exists() and
                    not (Path(prior['ledger']) / 'cells' / (name + '.json')).exists(),
                    'PRIOR_CELL_CONSUMPTION_PREVENTS_INFRASTRUCTURE_RECOVERY')
    return {'schema': 'active-tts-pre-model-recovery-v1', 'prior_stage': str(path),
            'prior_stage_sha256': stage_sha, 'prior_result': str(result_path), 'prior_result_sha256': result_sha,
            'prior_source_revision': prior['source_revision'], 'prior_execution_source': prior['execution_source'],
            'evidence': evidence, 'actual_elapsed_nanoseconds': first['actual_elapsed_ns'],
            'maximum_new_queue_nanoseconds': RECOVERY_QUEUE_NS,
            'maximum_combined_nanoseconds': first['actual_elapsed_ns'] + RECOVERY_QUEUE_NS,
            'model_loads': 0, 'generation_calls': 0, 'cell_claims': 0, 'retry_same_attempt': False}


def verify_recovery(stage):
    recovery = stage.get('recovery')
    if recovery is None:
        require(stage.get('infrastructure_attempt', 1) == 1 and
                stage.get('queue_budget_minutes', STAGE_MINUTES) == STAGE_MINUTES, 'RECOVERY_METADATA_REQUIRED')
        return
    require(stage['infrastructure_attempt'] == 2 and stage['queue_budget_minutes'] == RECOVERY_QUEUE_MINUTES and
            recovery['schema'] == 'active-tts-pre-model-recovery-v1' and recovery['model_loads'] == 0 and
            recovery['generation_calls'] == 0 and recovery['cell_claims'] == 0 and
            recovery['maximum_new_queue_nanoseconds'] == RECOVERY_QUEUE_NS and
            recovery['actual_elapsed_nanoseconds'] + RECOVERY_QUEUE_NS == recovery['maximum_combined_nanoseconds'] <=
            STAGE_MINUTES * 60 * 10**9, 'FIXED_RECOVERY_BUDGET_CHANGED')
    require(common.read(Path(stage['ledger']) / 'recoveries' / (recovery['prior_stage_sha256'] + '.json')) ==
            recovery_registration(stage), 'RECOVERY_REGISTRATION_CHANGED')
    for name, expected in recovery['evidence'].items():
        require(common.record(absolute(name)) == expected, 'RECOVERY_EVIDENCE_CHANGED')
    prior = common.read(recovery['prior_stage'])
    require(common.sha_file(recovery['prior_stage']) == recovery['prior_stage_sha256'] and
            common.sha_file(recovery['prior_result']) == recovery['prior_result_sha256'] and
            prior.get('infrastructure_attempt', 1) == 1 and prior.get('recovery') is None and
            prior['source_revision'] == recovery['prior_source_revision'] and
            prior['execution_source'] == recovery['prior_execution_source'] and
            prior['binding'] == stage['binding'] and prior['target_cells'] == stage['target_cells'] and
            prior['chunks'] == stage['chunks'] and prior['ledger'] == stage['ledger'] and
            prior['heavy_lock'] == stage['heavy_lock'], 'RECOVERY_STAGE_BINDING_CHANGED')
    check_files(prior['execution_root'], prior['execution_source'])
    check_files(Path(prior['prepared']) / 'execution-source', prior['execution_source'])
    _, states = queue_state(prior, require_created=True)
    require(states['chunk-001']['status'] == 'failed' and
            states['chunk-001']['actual_elapsed_ns'] == recovery['actual_elapsed_nanoseconds'] and
            all(row['status'] == 'not-started' for name, row in states.items() if name != 'chunk-001'),
            'PRIOR_QUEUE_RESUMED_OR_CHANGED')
    for name in states:
        require(not list((Path(prior['queue_root']) / 'jobs' / name / 'outputs').iterdir()), 'PRIOR_OUTPUTS_CHANGED')


def recovery_registration(stage):
    return {'schema': 'active-tts-single-recovery-v1',
            'prior_stage_sha256': stage['recovery']['prior_stage_sha256'],
            'recovery_stage': str(Path(stage['prepared']) / 'stage.json'),
            'recovery_stage_sha256': common.sha_file(Path(stage['prepared']) / 'stage.json')}


def prepare(args):
    require(type(args.stage_id) is str and ID.fullmatch(args.stage_id), 'SAFE_STAGE_ID_REQUIRED')
    require(type(args.target_count) is int and 8 <= args.target_count <= 256, 'BOUNDED_TARGET_AND_CHUNK')
    recipe, prepared, plan, bulk, binding = verify_parent(args.base_recipe, args.base_recipe_sha256)
    completed_base = getattr(args, 'completed_base_history', False)
    require(type(completed_base) is bool and completed_base != (args.prior_budget is not None),
            'EXACTLY_ONE_HISTORICAL_EVIDENCE_MODE')
    prior_args = (args.prior_stage, args.prior_stage_sha256, args.prior_result_sha256)
    require(all(value is None for value in prior_args) or all(value is not None for value in prior_args),
            'RECOVERY_STAGE_AND_RESULT_PINS_REQUIRED')
    recovery = None if args.prior_stage is None else freeze_recovery(*prior_args)
    if recovery is not None:
        prior = common.read(args.prior_stage)
        require(prior['binding'] == binding and prior['target_count'] == args.target_count and
                prior['heavy_lock'] == str(absolute(args.heavy_lock)) and prior['stage_id'] != args.stage_id,
                'NEW_RECOVERY_STAGE_SAME_INPUTS_REQUIRED')
        with heavy_model_slot(args.heavy_lock):
            pass  # The stopped attempt has actually released the shared slot.
    reused = verify_base_job(args.base_job, args.base_result_sha256, args.base_recipe_sha256, plan, binding)
    # Validate history before publishing any shared ledger/preparation files.
    history = (freeze_completed_base_history(args.base_job, args.base_result_sha256, args.base_recipe_sha256)
               if completed_base else freeze_history(args.prior_budget))
    ledger = absolute(Path(recipe['runtime']) / 'plan-ledgers' / binding['plan_canonical_sha256'])
    refs = {ref['reference_id']: ref for ref in plan['references']}
    with ledger_lock(ledger):
        if recovery is not None:
            require(not (ledger / 'recoveries' / (recovery['prior_stage_sha256'] + '.json')).exists(),
                    'RECOVERY_ALREADY_ASSIGNED_NO_THIRD_ATTEMPT')
        base = {'schema': 'active-tts-plan-first-eight-v1', 'base_terminal_sha256': args.base_result_sha256,
                'records': reused}
        if (ledger / 'base.json').exists():
            require(common.read(ledger / 'base.json') == base, 'BASE_LEDGER_IMPORT_CHANGED')
        else:
            once(ledger / 'base.json', base)
        ledger_binding = {'schema': 'active-tts-plan-ledger-v1', 'binding': binding,
                          'base': common.record(ledger / 'base.json')}
        if (ledger / 'binding.json').exists():
            require(common.read(ledger / 'binding.json') == ledger_binding, 'PLAN_LEDGER_CHANGED')
        else:
            once(ledger / 'binding.json', ledger_binding)
        for name, value in reused.items():
            target = ledger / 'cells' / (name + '.json')
            if target.exists():
                require(common.read(target) == value, 'BASE_LEDGER_IMPORT_CHANGED')
            else:
                require(not (ledger / 'claims' / (name + '.json')).exists(), 'BASE_ALREADY_CLAIMED')
                once(target, value)
        completed = {}
        for cell in plan['cells'][:args.target_count]:
            value = read_cell(ledger, cell['cell_id'], cell_identity(binding, cell, refs[cell['reference_id']]))
            if value is not None:
                completed[cell['cell_id']] = value
        chunks = partition_cells(plan, completed, args.target_count)
        require(chunks, 'REQUESTED_PREFIX_ALREADY_COMPLETE_NO_MODEL_WORK_NEEDED')
        output = absolute(args.output)
        require(not output.exists(), 'NEW_PREPARATION_DIRECTORY_REQUIRED')
        output.mkdir(parents=True, mode=0o700)
        once(output / 'history.json', history)
        source = {}
        for name in OWN_SOURCES:
            body = (common.ROOT / name).read_bytes()
            atomic_bytes(output / 'execution-source' / name, body)
            source[name] = common.record(common.ROOT / name)
        stage = {'schema': SCHEMA, 'stage_id': args.stage_id, 'prepared': str(output),
            'base_recipe': str(absolute(args.base_recipe)), 'base_recipe_sha256': args.base_recipe_sha256,
            'base_job': str(absolute(args.base_job)), 'base_result_sha256': args.base_result_sha256,
            'binding': binding, 'ledger': str(ledger), 'heavy_lock': str(absolute(args.heavy_lock)),
            'runtime': recipe['runtime'], 'queue_root': str(output / 'queue'),
            'target_count': args.target_count, 'target_cells': [row['cell_id'] for row in plan['cells'][:args.target_count]],
            'initial_reused_cells': sorted(completed), 'chunks': chunks,
            'maximum_new_calls': sum(map(len, chunks)), 'maximum_job_seconds': JOB_SECONDS,
            'stage_budget_minutes': STAGE_MINUTES, 'max_retries': 0,
            'infrastructure_attempt': 1 if recovery is None else 2,
            'queue_budget_minutes': STAGE_MINUTES if recovery is None else RECOVERY_QUEUE_MINUTES,
            'recovery': recovery,
            'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=common.ROOT, text=True).strip(),
            'execution_root': str(common.ROOT), 'history': common.record(output / 'history.json'),
            'execution_source': source, 'runtime_python': str(Path(recipe['runtime']) / 'venv/bin/python'),
            'requires_review_before_queue_execution': True, 'models_executed_during_prepare': 0,
            'training_admitted': False, 'paired_cases_included': False}
        once(output / 'stage.json', stage)
        if recovery is not None:
            once(ledger / 'recoveries' / (recovery['prior_stage_sha256'] + '.json'), recovery_registration(stage))
        for index in range(len(chunks)):
            job = job_recipe(stage, index)
            once(output / 'jobs' / (job['job_id'] + '.json'), job)
        once(output / 'queue-request.json', queue_request(stage))
        return {'status': 'prepared-not-executed', 'stage': str(output / 'stage.json'),
                'stage_sha256': common.sha_file(output / 'stage.json'), 'new_call_cap': stage['maximum_new_calls'],
                'chunks': list(map(len, chunks)), 'reused': len(completed), 'model_calls': 0}


def load_stage(path):
    path = absolute(path); stage = common.read(path)
    require(stage['schema'] == SCHEMA and stage['stage_budget_minutes'] == STAGE_MINUTES and
            stage['maximum_job_seconds'] == JOB_SECONDS and stage['max_retries'] == 0 and
            type(stage['stage_id']) is str and ID.fullmatch(stage['stage_id']) and
            type(stage['target_count']) is int and 8 <= stage['target_count'] <= 256 and
            stage['execution_root'] == str(common.ROOT) and
            path == absolute(Path(stage['prepared']) / 'stage.json') and
            stage['queue_root'] == str(Path(stage['prepared']) / 'queue') and
            stage['runtime_python'] == str(Path(stage['runtime']) / 'venv/bin/python') and
            stage['training_admitted'] is False and stage['paired_cases_included'] is False and
            stage['models_executed_during_prepare'] == 0 and
            1 <= len(stage['chunks']) <= 2 and all(1 <= len(chunk) <= MAX_CHUNK for chunk in stage['chunks']) and
            sum(map(len, stage['chunks'])) == stage['maximum_new_calls'] <= MAX_STAGE_CALLS,
            'STAGE_SCOPE_CHANGED')
    check_files(common.ROOT, stage['execution_source'])
    check_files(Path(stage['prepared']) / 'execution-source', stage['execution_source'])
    require(set(stage['execution_source']) == set(OWN_SOURCES), 'STAGE_SOURCE_ROSTER')
    recipe, prepared, plan, bulk, binding = verify_parent(stage['base_recipe'], stage['base_recipe_sha256'])
    require(binding == stage['binding'] and stage['ledger'] ==
            str(Path(recipe['runtime']).absolute() / 'plan-ledgers' / binding['plan_canonical_sha256']), 'STAGE_BINDING_CHANGED')
    base = verify_base_job(stage['base_job'], stage['base_result_sha256'], stage['base_recipe_sha256'], plan, binding)
    require(common.read(Path(stage['ledger']) / 'base.json') ==
            {'schema': 'active-tts-plan-first-eight-v1', 'base_terminal_sha256': stage['base_result_sha256'],
             'records': base}, 'BASE_LEDGER_IMPORT_CHANGED')
    require(common.read(Path(stage['ledger']) / 'binding.json') ==
            {'schema': 'active-tts-plan-ledger-v1', 'binding': binding,
             'base': common.record(Path(stage['ledger']) / 'base.json')}, 'PLAN_LEDGER_CHANGED')
    history_path = Path(stage['prepared']) / 'history.json'
    require(common.record(history_path) == stage['history'], 'HISTORY_SNAPSHOT_CHANGED')
    history = common.read(history_path)
    require(history['schema'] in ('active-tts-extension-history-v1', 'active-tts-completed-base-history-v1'),
            'HISTORY_SCHEMA_CHANGED')
    if history['schema'] == 'active-tts-completed-base-history-v1':
        require(history == freeze_completed_base_history(stage['base_job'], stage['base_result_sha256'],
                                                         stage['base_recipe_sha256']), 'COMPLETED_BASE_HISTORY_CHANGED')
    for name, expected in history['files'].items():
        require(common.record(absolute(name)) == expected, 'HISTORICAL_RECEIPT_CHANGED')
    target = [row['cell_id'] for row in plan['cells'][:stage['target_count']]]
    selected = [name for chunk in stage['chunks'] for name in chunk]
    require(target == stage['target_cells'] and len(set(selected)) == len(selected) and
            len(set(stage['initial_reused_cells'])) == len(stage['initial_reused_cells']) and
            not set(selected) & set(stage['initial_reused_cells']) and
            set(selected) | set(stage['initial_reused_cells']) == set(target), 'STAGE_CELL_PARTITION_CHANGED')
    verify_recovery(stage)
    return stage, recipe, prepared, plan, bulk


def verify_job(path):
    job = common.read(path)
    require(job['schema'] == JOB_SCHEMA and common.sha_file(job['stage']) == job['stage_sha256'], 'JOB_STAGE_CHANGED')
    stage, parent, prepared, plan, bulk = load_stage(job['stage'])
    index = job['chunk_index']
    require(type(index) is int and 0 <= index < len(stage['chunks']) and job == job_recipe(stage, index),
            'JOB_CHUNK_CHANGED')
    return job, stage, parent, prepared, plan, bulk


def outcomes(job, stage, new_ids, reused_ids):
    rows = []
    for name in job['cells']:
        record_path = Path(stage['ledger']) / 'cells' / (name + '.json')
        value = common.read(record_path)
        claim = None if value['origin'] == 'reused-first-eight' else common.read(
            Path(stage['ledger']) / 'claims' / (name + '.json'))
        if name in new_ids:
            require(claim is not None and claim['owner'] == {
                'stage_sha256': job['stage_sha256'], 'job_id': job['job_id'],
                'job_recipe_sha256': common.sha_file(Path(stage['prepared']) / 'jobs' / (job['job_id'] + '.json'))},
                'NEW_CALL_OWNER_CHANGED')
        rows.append({'cell_id': name, 'ledger_record_sha256': common.sha_file(record_path),
                     'record': value, 'claim': claim})
    return {'schema': 'active-tts-stage-outcomes-v1', 'stage': job['stage'],
            'stage_sha256': job['stage_sha256'], 'job_id': job['job_id'],
            'job_recipe_sha256': common.sha_file(Path(stage['prepared']) / 'jobs' / (job['job_id'] + '.json')),
            'plan_canonical_sha256': stage['binding']['plan_canonical_sha256'], 'binding': stage['binding'],
            'selected_cells': job['cells'], 'new_call_ids': new_ids, 'reused_cell_ids': reused_ids, 'rows': rows}


def verify_job_outputs(job, stage, output, plan):
    summary = common.read(output / 'summary.json')
    require(summary['schema'] == 'active-tts-extension-result-v1' and summary['status'] == 'completed' and
            summary['job_recipe_sha256'] == common.sha_file(Path(stage['prepared']) / 'jobs' / (job['job_id'] + '.json')) and
            summary['stage_sha256'] == job['stage_sha256'] and
            summary['inventory'] == common.record(output / 'inventory.json') and
            summary['outcomes'] == common.record(output / 'outcomes.json'), 'CACHED_JOB_RECEIPTS_CHANGED')
    inventory = common.read(output / 'inventory.json')['files']
    check_files(output, inventory)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(actual == set(inventory) | {'inventory.json', 'summary.json'}, 'JOB_OUTPUT_ROSTER_CHANGED')
    new = summary['new_call_ids']; reused = summary['reused_cell_ids']
    require(len(set(new)) == len(new) and len(set(reused)) == len(reused) and not set(new) & set(reused) and
            set(new) | set(reused) == set(job['cells']) and len(new) == summary['new_generation_calls'],
            'CACHED_CALL_COUNTS_CHANGED')
    cells = {row['cell_id']: row for row in plan['cells']}; refs = {row['reference_id']: row for row in plan['references']}
    for name in job['cells']:
        cell = cells[name]
        require(read_cell(stage['ledger'], name, cell_identity(stage['binding'], cell, refs[cell['reference_id']])) is not None,
                'CACHED_CELL_MISSING')
    expected = outcomes(job, stage, new, reused)
    require(common.read(output / 'outcomes.json') == expected and summary['cells'] == [
        {'cell_id': row['cell_id'], 'status': row['record']['status'], 'ledger_record_sha256': row['ledger_record_sha256']}
        for row in expected['rows']], 'CACHED_CELL_BINDINGS_CHANGED')
    return summary


def run(args):
    job, stage, parent, prepared, plan, bulk = verify_job(args.recipe)
    output = absolute(args.output)
    require(output == Path(stage['queue_root']) / 'jobs' / job['job_id'] / 'outputs' and
            absolute(args.heavy_lock) == Path(stage['heavy_lock']), 'JOB_OUTPUT_OR_LOCK_CHANGED')
    require(os.environ.get('KWS_RESEARCH_RECIPE_SHA256') == common.sha_file(args.recipe), 'QUEUE_RECIPE_BINDING')
    _, states = queue_state(stage, require_created=True)
    state = states[job['job_id']]
    require(common.sha_file(args.recipe) == common.sha_file(
        Path(stage['prepared']) / 'jobs' / (job['job_id'] + '.json')), 'RUN_RECIPE_CHANGED')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    cells = {row['cell_id']: row for row in plan['cells']}; refs = {row['reference_id']: row for row in plan['references']}
    if (output / 'invocation.json').exists():
        # Queue resume handles completed jobs; this entry never restarts a
        # partially executed job even when some unclaimed cells remain.
        require((output / 'summary.json').is_file(), 'INVOCATION_ALREADY_CONSUMED_NO_AUTOMATIC_RESTART')
        require(state['status'] == 'completed', 'COMPLETED_QUEUE_RECEIPT_REQUIRED_FOR_CACHE')
        verify_job_outputs(job, stage, output, plan)
        return {'status': 'completed-cache-reused', 'new_model_calls': 0, 'summary_sha256': common.sha_file(output / 'summary.json')}
    require(state['status'] == 'claimed-without-terminal' and
            state['claim']['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
            state['claim']['deadline_monotonic_ns'] > time.monotonic_ns(), 'LIVE_QUEUE_CLAIM_REQUIRED')
    with heavy_model_slot(stage['heavy_lock']) as lease, ledger_lock(stage['ledger']):
        require(lease.supervised, 'DURABLE_QUEUE_REQUIRED')
        # Resolve every identity before loading; any prior incomplete claim
        # blocks this whole invocation rather than silently skipping the cell.
        existing = {}
        for name in job['cells']:
            cell = cells[name]
            existing[name] = read_cell(stage['ledger'], name, cell_identity(stage['binding'], cell, refs[cell['reference_id']]))
        started = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
        once(output / 'invocation.json', {'job_recipe_sha256': common.sha_file(args.recipe),
            'stage_sha256': job['stage_sha256'], 'selected_cells': job['cells'], 'inherited': lease.inherited,
            'supervised': lease.supervised, 'process_group': os.getpgrp(), 'calls_allowed': len(job['cells'])})
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '4'
        common.research_no_network.set_offline_environment()
        once(output / 'network-guard.json', common.research_no_network.install())
        once(output / 'generation/plan-lock.json', {'schema': 'kws-bulk-clone-execution-lock-v1',
             'plan_sha256': stage['binding']['plan_canonical_sha256'], 'training_admitted': False, 'model_id': plan['model_id']})
        pending = [name for name, value in existing.items() if value is None]
        delegate = backend = None
        if pending:
            # The same installed role and package sources as the original run.
            verify_runtime(prepared)
            import onnxruntime
            onnxruntime.disable_telemetry_events()
            bulk.asset_lock(Path(parent['runtime']) / 'models/base', common.read(prepared / 'model-assets.json'), plan['model_revision'])
            import torch
            torch.set_num_threads(4); torch.set_num_interop_threads(1)
            delegate = bulk.QwenBulkClone(Path(parent['runtime']) / 'models/base', device='cpu')
            before = time.monotonic(); model = delegate._model()
            require(model.model.tts_model_type == 'base', 'BASE_MODEL_REQUIRED')
            once(output / 'model-loaded.json', {'status': 'loaded', 'seconds': time.monotonic() - before,
                'parameter_dtype': str(next(model.model.parameters()).dtype), 'parameter_device': str(next(model.model.parameters()).device),
                'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads()})
            backend = common.BoundedBackend(delegate, output,
                {'selected_cells': job['cells'], 'max_calls': len(job['cells']), 'generation': stage['binding']['generation']},
                native_writer=retain_extension_native)
        else:
            once(output / 'model-loaded.json', {'status': 'not-loaded-all-cells-reused', 'model_loads': 0})

        def generate(cell, ref):
            require(time.monotonic() - started < JOB_SECONDS and
                    time.monotonic_ns() < state['claim']['deadline_monotonic_ns'], 'JOB_SOFT_DEADLINE')
            folder = output / 'generation/cells' / cell['cell_id']
            once(folder / 'started.json', expected_start(cell, ref, stage['binding']['plan_canonical_sha256']))
            # The shared backend saves the exact raw array and FLOAT WAV before
            # this technical gate. Inference exceptions abort, leaving a claim.
            audio, rate = backend.synthesize(cell, ref, prepared / 'reference-assets')
            raw = common.read(output / 'native' / cell['cell_id'] / 'native.json')
            try:
                require(raw['wav'] is not None, raw.get('wav_unavailable_reason', 'NO_NATIVE_WAV'))
                metrics = bulk.media.quality(audio, rate)
                rendered = bulk.media._write_waves(folder, audio, rate, metrics)
            except ValueError as error:
                once(folder / 'failed.json', {'schema': 'kws-bulk-clone-failure-v1', 'cell_id': cell['cell_id'],
                    'status': 'FAILED_NO_RETRY', 'error_type': type(error).__name__, 'training_admitted': False})
                status = 'FAILED_NO_RETRY'; reason = str(error)
                native_wav = 'native/' + cell['cell_id'] + '/returned-native.wav' if raw['wav'] is not None else None
            else:
                once(folder / 'completed.json', {'schema': 'kws-bulk-clone-complete-v1', 'cell_id': cell['cell_id'],
                    'status': 'GENERATED_PENDING_ASR', 'technical': metrics, 'audio': rendered,
                    'source_family_id': cell['source_family_id'], 'intended_kind': cell['intended_kind'],
                    'actual_text': None, 'training_admitted': False, 'qualification_allowed': False})
                status = 'GENERATED_PENDING_ASR'; reason = None
                native_wav = 'generation/cells/' + cell['cell_id'] + '/native-f32.wav'
            files = {}
            for location in (folder, output / 'native' / cell['cell_id']):
                files.update({p.relative_to(output).as_posix(): common.record(p) for p in location.rglob('*') if p.is_file()})
            return {'status': status, 'technical_failure_reason': reason, 'output_root': str(output),
                    'files': files, 'native_wav': native_wav, 'generation_root': str(output / 'generation')}

        observations = []; new_ids = []; reused_ids = []
        for name in job['cells']:
            cell = cells[name]
            value, invoked = consume_cell(stage['ledger'], cell, refs[cell['reference_id']], stage['binding'],
                {'stage_sha256': job['stage_sha256'], 'job_id': job['job_id'], 'job_recipe_sha256': common.sha_file(args.recipe)}, generate)
            (new_ids if invoked else reused_ids).append(name)
            observations.append({'cell_id': name, 'status': value['status'],
                'ledger_record_sha256': common.sha_file(Path(stage['ledger']) / 'cells' / (name + '.json'))})
        require((backend.calls if backend is not None else 0) == len(new_ids) <= len(job['cells']), 'ACTUAL_GENERATION_CALL_BUDGET')
        once(output / 'outcomes.json', outcomes(job, stage, new_ids, reused_ids))
        end = resource.getrusage(resource.RUSAGE_SELF)
        once(output / 'inventory.json', {'files': {p.relative_to(output).as_posix(): common.record(p)
            for p in sorted(output.rglob('*')) if p.is_file()}})
        summary = {'schema': 'active-tts-extension-result-v1', 'status': 'completed',
            'job_recipe_sha256': common.sha_file(args.recipe), 'stage_sha256': job['stage_sha256'],
            'new_generation_calls': len(new_ids), 'new_call_ids': new_ids, 'reused_cell_ids': reused_ids,
            'cells': observations, 'prompt_creations': delegate.prompt_creations if delegate else 0,
            'wall_seconds': time.monotonic() - started, 'user_cpu_seconds': end.ru_utime - usage.ru_utime,
            'system_cpu_seconds': end.ru_stime - usage.ru_stime, 'peak_rss_kib': end.ru_maxrss,
            'asr_calls': 0, 'training_admitted': False, 'qualification_allowed': False,
            'inventory': common.record(output / 'inventory.json'), 'outcomes': common.record(output / 'outcomes.json')}
        once(output / 'summary.json', summary)
        return summary


def inspect_stage(path):
    stage, parent, prepared, plan, bulk = load_stage(path)
    refs = {ref['reference_id']: ref for ref in plan['references']}
    records = {}; unknown = []; pending = []
    for cell in plan['cells'][:stage['target_count']]:
        try:
            value = read_cell(stage['ledger'], cell['cell_id'], cell_identity(stage['binding'], cell, refs[cell['reference_id']]))
        except ValueError as error:
            if str(error).startswith('CONSUMED_WITHOUT_TERMINAL:'):
                unknown.append(cell['cell_id']); continue
            raise
        if value is None:
            pending.append(cell['cell_id'])
        else:
            records[cell['cell_id']] = value
    _, states = queue_state(stage)
    known_ns = sum(row['actual_elapsed_ns'] for row in states.values() if row['actual_elapsed_ns'] is not None)
    complete_duration = all(row['status'] not in ('unknown', 'claimed-without-terminal') for row in states.values())
    jobs = {name: {'status': row['status'], 'actual_elapsed_ns': row['actual_elapsed_ns']} for name, row in states.items()}
    for index in range(len(stage['chunks'])):
        job = job_recipe(stage, index)
        if states[job['job_id']]['status'] == 'completed':
            verify_job_outputs(job, stage, Path(stage['queue_root']) / 'jobs' / job['job_id'] / 'outputs', plan)
    history = common.read(Path(stage['prepared']) / 'history.json')
    prior_ns = 0 if stage.get('recovery') is None else stage['recovery']['actual_elapsed_nanoseconds']
    return {'schema': 'active-tts-extension-status-v1', 'target_count': stage['target_count'],
        'verified_terminal_cells': len(records), 'pending_cells': pending, 'unknown_cells': unknown,
        'jobs': jobs, 'actual_known_job_nanoseconds': known_ns, 'actual_total_is_complete': complete_duration,
        'stage_budget_nanoseconds': STAGE_MINUTES * 60 * 10**9,
        'queue_budget_nanoseconds': STAGE_MINUTES * 60 * 10**9 if stage.get('recovery') is None else RECOVERY_QUEUE_NS,
        'prior_extension_actual_elapsed_nanoseconds': prior_ns,
        'combined_extension_known_nanoseconds': prior_ns + known_ns,
        'combined_extension_actual_total_is_complete': complete_duration,
        'budget_scope': 'new jobs in this frozen extension stage; historical generation receipts remain separate',
        'historical_known_job_nanoseconds': history['historical_known_job_nanoseconds'],
        'historical_actual_total_nanoseconds': history['historical_actual_total_nanoseconds'],
        'historical_actual_total_complete': history['historical_actual_total_complete'],
        'historical_scope': history.get('scope', 'retained-cross-attempt-budget'),
        'historical_unknown_conservative_reservation_seconds': history['unknown_conservative_reservation_seconds'],
        'records': records, 'training_admitted': False}


def export_asr(path):
    stage, parent, prepared, plan, bulk = load_stage(path)
    state = inspect_stage(path)
    require(not state['pending_cells'] and not state['unknown_cells'] and
            all(job['status'] == 'completed' for job in state['jobs'].values()), 'COMPLETE_STAGE_REQUIRED_FOR_EXPORT')
    target = Path(stage['prepared']) / 'asr-exports'
    target.mkdir(exist_ok=True, mode=0o700)
    partitions = []; new_calls = set(); excluded = []; job_outcomes = []
    for index in range(len(stage['chunks'])):
        name = 'chunk-' + str(index + 1).zfill(3)
        output = Path(stage['queue_root']) / 'jobs' / name / 'outputs'
        summary = common.read(output / 'summary.json')
        new_ids = summary['new_call_ids']
        require(len(new_ids) <= 64 and len(set(new_ids)) == len(new_ids) and not new_calls & set(new_ids) and
                set(new_ids) <= set(stage['chunks'][index]), 'ASR_PARTITION_SCOPE')
        new_calls.update(new_ids)
        outcome_pin = {'path': str(output / 'outcomes.json'), **common.record(output / 'outcomes.json')}
        recipe_pin = {'path': str(Path(stage['prepared']) / 'jobs' / (name + '.json')),
                      **common.record(Path(stage['prepared']) / 'jobs' / (name + '.json'))}
        job_outcomes.append({'job_id': name, 'outcomes': outcome_pin, 'job_recipe': recipe_pin})
        ids = []
        for identity in new_ids:
            record = state['records'][identity]; eligibility = asr_eligibility(record)
            if eligibility['eligible']:
                ids.append(identity)
            else:
                excluded.append({'cell_id': identity, 'job_id': name, 'reason': eligibility['reason'],
                    'record': record, 'ledger_record_sha256': common.sha_file(Path(stage['ledger']) / 'cells' / (identity + '.json')),
                    'outcomes': outcome_pin, 'asr_calls': 0, 'observed_text': None, 'training_admitted': False})
        if not ids:
            continue
        archive = target / (name + '-native.zip')
        # Deterministic ZIP members; intended text is never put into this input.
        import io
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as stream:
            for identity in ids:
                row = state['records'][identity]
                body = (Path(row['output_root']) / row['native_wav']).read_bytes()
                info = zipfile.ZipInfo('cells/' + identity + '/native-f32.wav')
                info.external_attr = 0o600 << 16
                stream.writestr(info, body)
        body = buffer.getvalue()
        if archive.exists():
            require(archive.read_bytes() == body, 'EXISTING_ASR_ARCHIVE_CHANGED')
        else:
            atomic_bytes(archive, body)
        partitions.append({'job_id': name, 'cell_ids': ids, 'archive': str(archive),
            **common.record(archive), 'generation_root': str(output / 'generation'),
            'outcomes': outcome_pin, 'job_recipe': recipe_pin,
            'technical_failed_cells': [identity for identity in ids if state['records'][identity]['status'] == 'FAILED_NO_RETRY'],
            'recognizer_context_text': None, 'weak_view_requires_explicit_failed_source_support':
                any(state['records'][identity]['status'] == 'FAILED_NO_RETRY' for identity in ids)})
    reused = [name for name in stage['target_cells'] if name not in new_calls]
    index = {'schema': 'active-tts-asr-partitions-v1', 'stage': str(absolute(path)), 'stage_sha256': common.sha_file(path),
        'plan_file_sha256': stage['binding']['plan_file_sha256'],
        'plan_canonical_sha256': stage['binding']['plan_canonical_sha256'],
        'plan': str(prepared / 'plan.json'), 'partitions': partitions,
        'job_outcomes': job_outcomes, 'excluded_cells': excluded, 'asr_input_contract': ASR_INPUT_CONTRACT,
        'new_generation_cell_count': len(new_calls), 'asr_eligible_new_cell_count': len(new_calls) - len(excluded),
        'asr_excluded_new_cell_count': len(excluded),
        'reused_cells_without_new_asr_request': reused,
        'reused_bindings': [{'cell_id': name, 'record': state['records'][name],
                            'ledger_record_sha256': common.sha_file(Path(stage['ledger']) / 'cells' / (name + '.json'))}
                           for name in reused],
        'aggregate_cell_count': stage['target_count'], 'paired_cases_included': False,
        'asr_calls_executed': 0, 'training_admitted': False}
    path = target / 'index.json'
    if path.exists():
        require(common.read(path) == index, 'EXISTING_ASR_INDEX_CHANGED')
    else:
        once(path, index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('base-recipe', 'base-job', 'output', 'heavy-lock'):
        prep.add_argument('--' + name, type=Path, required=True)
    history = prep.add_mutually_exclusive_group(required=True)
    history.add_argument('--prior-budget', type=Path)
    history.add_argument('--completed-base-history', action='store_true',
                         help='Retain only the authenticated completed first-eight job; no invented unknown history.')
    for name in ('base-recipe-sha256', 'base-result-sha256', 'stage-id'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--target-count', type=int, default=64)
    prep.add_argument('--prior-stage', type=Path)
    prep.add_argument('--prior-stage-sha256')
    prep.add_argument('--prior-result-sha256')
    run_cli = sub.add_parser('run')
    for name in ('recipe', 'output', 'heavy-lock'):
        run_cli.add_argument('--' + name, type=Path, required=True)
    for name in ('status', 'export-asr'):
        sub.add_parser(name).add_argument('--stage', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args)
    elif args.command == 'run':
        result = run(args)
    elif args.command == 'status':
        result = inspect_stage(args.stage); result.pop('records')
    else:
        result = export_asr(args.stage)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
