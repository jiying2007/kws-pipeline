"""Independent zero-ASR preparation authority. Stdlib only.

Legacy loader declarations are compatibility inputs, never execution authority.
This module neither imports ML nor reads WAV bytes nor creates an ASR claim.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = 'research/k1-punctuation6-prepare-diagnostic-v1/'
OLD = 'research/k1-punctuation6-asr18-v1/'
EXPERIMENT = 'k1-punctuation6-prepare-diagnostic-v1'
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
PACKET_NAMES = frozenset(('job.json', 'decoder-inputs.json', 'worker-input-binding.json',
                          'execution-plan.json', 'host-binding.json'))
MIB = 1024 ** 2
SHA = re.compile(r'[0-9a-f]{64}\Z')
FIXED = {'schema': 'k1-prepare-diagnostic-release-v1', 'experiment': EXPERIMENT,
         'source_ref': 'research/' + EXPERIMENT, 'evidence_ref': 'evidence/' + EXPERIMENT,
         'workflow': '.github/workflows/' + EXPERIMENT + '.yml', 'job_key': 'prepare_diagnostic',
         'max_hosted_runs': 1, 'run_attempt': 1, 'retries': 0, 'max_model_constructions': 3,
         'max_model_constructions_per_family': 1, 'max_new_asr_calls': 0,
         'old_calls_allowed': 0, 'audio_forward_allowed': False}
BINDING_PINS = {'input_freeze_sha256': '143e5241e4b9a52da272294862a6ded1447354865883ec5f5e8cd172742b455e',
    'execution_plan_sha256': '6efa37e506b1b063809c5cccb6d4a9cd1e8995403499a66ab1b5184ee576f7ef',
    'worker_input_binding_sha256': 'f8471d2472402c60a9339f2787e749eeb816e679db94b36d1c9c447c7c646b65',
    'screening_receipt_sha256': '3588366d127632e1889f98f433eb5b7333840d5371c9280e29f771b1a3ee3c16',
    'tts_authority_sha256': '45f2f712479ee8035cd22b5102d1e5980e86873ce2852cdf587a54c746785c0e'}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(raw):
    require(type(raw) is bytes, 'bytes_required')
    return hashlib.sha256(raw).hexdigest()


def decode(raw, cap=MIB):
    require(type(raw) is bytes and len(raw) <= cap, 'metadata_size')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, 'nonfinite_json'))
    require(canonical(value) == raw, 'canonical_metadata')
    return value


def regular(path, cap=MIB):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'aliased_path')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 <= before.st_size <= cap, 'bounded_regular_file')
        chunks = []
        while True:
            block = os.read(fd, min(65536, cap + 1 - sum(map(len, chunks))))
            if not block:
                break
            chunks.append(block)
            require(sum(map(len, chunks)) <= cap, 'file_size_changed')
        body = b''.join(chunks)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_nlink)
        require(len(body) == before.st_size and stamp(before) == stamp(os.fstat(fd)) ==
                stamp(path.stat(follow_symlinks=False)), 'file_changed')
        return body
    finally:
        os.close(fd)


def validate_release(value):
    require(type(value) is dict and set(value) == set(FIXED) | set(BINDING_PINS) |
            {'enabled', 'independent_source_review', 'source_closure_ready',
             'prior_prepare_history_sha256'}, 'release_fields')
    for key, expected in {**FIXED, **BINDING_PINS}.items():
        require(type(value[key]) is type(expected) and value[key] == expected, 'release_' + key)
    for name in ('enabled', 'independent_source_review', 'source_closure_ready'):
        require(type(value[name]) is bool, 'release_boolean')
    require(type(value['prior_prepare_history_sha256']) is str and
            SHA.fullmatch(value['prior_prepare_history_sha256']), 'prepare_history_identity')
    require(value['enabled'] is True, 'prepare_execution_disabled')
    require(value['independent_source_review'] is True and value['source_closure_ready'] is True,
            'prepare_review_or_closure_incomplete')
    return value


def release_first():
    # First operational gate; do not accept a caller-selected release file.
    return validate_release(decode(regular(HERE / 'prepare-release.json', 65536)))


def fixed_module(relative, name):
    require(relative in {OLD + n + '.py' for n in ('campaign_contract', 'cpu_budget',
        'runtime_scope', 'setup_adapter', 'native_locks', 'whisper_binding', 'optional_tools', 'runtime_session')} |
        {'research/source-screen32-v1/asr_worker.py', 'research/source-screen32-v1/native_v2_adapter.py',
         'research/source-screen32-whisper-v1/adapter.py'}, 'fixed_dependency')
    path = ROOT / relative
    existing = sys.modules.get(name)
    require(existing is None or Path(vars(existing).get('__file__', '/missing')) == path,
            'foreign_module_cache')
    if existing is not None:
        return existing
    for directory in (ROOT / OLD, ROOT / 'tools'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'fixed_module_spec')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    require(Path(module.__file__) == path, 'fixed_module_origin')
    return module


def validate_metadata_packet(files, release):
    require(type(files) is dict and set(files) == PACKET_NAMES, 'exact_metadata_packet')
    values = {name: decode(raw) for name, raw in files.items()}
    job, plan, binding = (values[n] for n in ('job.json', 'execution-plan.json', 'worker-input-binding.json'))
    host = values['host-binding.json']
    require(type(host) is dict and set(host) == {'schema', 'identity', 'binding_pins', 'members',
        'wav_files_validated', 'worker_audio_mounted', 'asr_permission_granted'}, 'host_binding_fields')
    require(host['schema'] == 'k1-prepare-host-binding-v1' and host['binding_pins'] == BINDING_PINS and
        type(host['wav_files_validated']) is int and host['wav_files_validated'] == 6 and
        host['worker_audio_mounted'] is False and host['asr_permission_granted'] is False,
        'host_binding_scope')
    require(canonical(host['members']) == canonical({name: {'bytes': len(files[name]), 'sha256': digest(files[name])}
        for name in sorted(PACKET_NAMES - {'host-binding.json'})}), 'host_packet_hashes')
    for name, key in (('execution-plan.json', 'execution_plan_sha256'),
                      ('worker-input-binding.json', 'worker_input_binding_sha256')):
        require(digest(files[name]) == release[key] == BINDING_PINS[key], 'frozen_binding_hash')
    c = fixed_module(OLD + 'campaign_contract.py', 'campaign_contract')
    c.validate_input_binding(binding, job, plan)
    require(len(job['clips']) == 6 and plan['family_order'] == list(FAMILIES), 'fixed_six_three')
    worker = fixed_module('research/source-screen32-v1/asr_worker.py', 'prepare_retained_worker')
    # Only stdlib descriptor validator is needed; no prepare/import_helpers/model.
    core = ROOT / 'research/qwen6_asr/core'
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from pcm.pcm_binding import validate_pcm_descriptor
    worker.validate_pcm_descriptor = validate_pcm_descriptor
    descriptors = worker.validate_decoder(files['decoder-inputs.json'], digest(files['decoder-inputs.json']))
    decoder = values['decoder-inputs.json']
    require(decoder['source_manifest_sha256'] == digest(files['job.json']) and
        set(descriptors) == {row['audio_id'] for row in job['clips']}, 'descriptor_job_binding')
    for row in binding['audio']:
        descriptor = descriptors[row['audio_id']]['descriptor']
        require(descriptor['wav']['sha256'] == row['wav_sha256'] and
            descriptor['raw_pcm16']['sha256'] == row['pcm_sha256'] and
            descriptor['raw_pcm16']['frame_count'] == row['audio_frames'], 'actual_wav_pcm_binding')
    return values


def worker_admission(family):
    release = release_first()
    require(family in FAMILIES and type(family) is str, 'fixed_prepare_family')
    require(ROOT == Path('/code') and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            'fixed_isolated_worker_root')
    for path in ('/code', '/runtime', '/metadata', '/authority'):
        require(not Path(path).is_symlink() and bool(os.statvfs(path).f_flag & os.ST_RDONLY),
                'readonly_worker_mount')
    require(not Path('/input').exists(), 'audio_mount_prohibited')
    authority = Path('/authority')
    require({p.name for p in authority.iterdir()} == {'prepare-freeze.json', 'prepare-claim.json'},
            'prepare_authority_roster')
    freeze_raw = regular(authority / 'prepare-freeze.json', 8 * MIB)
    # Freeze may exceed metadata cap; complete canonical JSON is still required.
    freeze = decode(freeze_raw, 8 * MIB)
    require(digest(freeze_raw) == os.environ.get('KWS_PREPARE_FREEZE_SHA256'),
            'independently_bound_prepare_freeze')
    require(freeze.get('schema') == 'k1-prepare-source-freeze-v1' and
            freeze.get('experiment') == EXPERIMENT and set(freeze) == {'schema', 'experiment', 'worker_code',
                'inference_code', 'setup_code', 'host_code'}, 'prepare_freeze_identity')
    require(freeze['worker_code'] == freeze['inference_code'] and
            all(freeze['host_code'].get(k) == v for k, v in freeze['worker_code'].items()),
            'worker_inference_host_closure')
    roster = freeze.get('worker_code')
    require(type(roster) is dict and roster, 'closed_worker_roster')
    observed = set()
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        require(not any((Path(base) / n).is_symlink() for n in dirs + names), 'source_alias')
        for name in names:
            relative = (Path(base) / name).relative_to(ROOT).as_posix()
            observed.add(relative)
    require(observed == set(roster), 'complete_closed_worker_source')
    for name, expected in roster.items():
        require(type(name) is str and not name.startswith('/') and
                all(p not in ('', '.', '..') for p in name.split('/')), 'source_relative_path')
        raw = regular(ROOT / name, 8 * MIB)
        require(canonical(expected) == canonical({'bytes': len(raw), 'sha256': digest(raw)}), 'source_identity')
    claim_raw = regular(authority / 'prepare-claim.json', 65536)
    require(digest(claim_raw) == os.environ.get('KWS_PREPARE_CLAIM_SHA256'), 'prepare_claim_hash')
    claim = decode(claim_raw)
    required = {'schema', 'experiment', 'identity', 'source_freeze_sha256', 'release_sha256',
        'metadata_packet_sha256', 'maximum_model_constructions', 'maximum_new_asr_calls',
        'prior_prepare_history_sha256'}
    require(set(claim) == required and claim['schema'] == 'k1-prepare-once-claim-v1' and
        claim['experiment'] == EXPERIMENT and claim['maximum_model_constructions'] == 3 and
        type(claim['maximum_model_constructions']) is int and
        claim['maximum_new_asr_calls'] == 0 and type(claim['maximum_new_asr_calls']) is int and
        claim['prior_prepare_history_sha256'] == release['prior_prepare_history_sha256'] and
        claim['source_freeze_sha256'] == digest(freeze_raw) and
        claim['release_sha256'] == digest(canonical(release)), 'prepare_claim_only')
    identity = claim['identity']
    require(type(identity) is dict and set(identity) == {'repository', 'run_id', 'run_attempt', 'job_key', 'executed_head_sha'} and
        identity['repository'] == 'jiying2007/kws-pipeline' and identity['job_key'] == 'prepare_diagnostic' and
        type(identity['run_id']) is int and identity['run_id'] > 0 and
        type(identity['run_attempt']) is int and identity['run_attempt'] == 1 and
        str(identity['run_id']) == os.environ.get('KWS_RUN_ID') and
        re.fullmatch('[0-9a-f]{40}', identity['executed_head_sha']) and
        identity['executed_head_sha'] == os.environ.get('KWS_EXECUTED_HEAD_SHA'), 'prepare_run_identity')
    metadata = Path('/metadata')
    require({p.name for p in metadata.iterdir()} == PACKET_NAMES, 'metadata_only_mount')
    files = {name: regular(metadata / name) for name in PACKET_NAMES}
    inventory = {name: {'bytes': len(raw), 'sha256': digest(raw)} for name, raw in sorted(files.items())}
    require(claim['metadata_packet_sha256'] == digest(canonical(inventory)), 'claimed_metadata_packet')
    packet = validate_metadata_packet(files, release)
    require(packet['host-binding.json']['identity'] == identity, 'same_host_worker_identity')
    return release, freeze, claim, files, packet
