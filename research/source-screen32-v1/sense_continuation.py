"""Host-only, literal v3 evidence binding for one bounded SenseVoice continuation.

No acquisition, installation, model invocation, or caller-selected source paths.
This module and its producer mapping must never be staged in blind ASR code.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import wave

EXPERIMENT = 'qwen16-sensevoice-continuation-v1'
BRANCH = 'research/' + EXPERIMENT
REFERENCE = 'research/source-screen32-v1/evidence/run-38018029787'
LOCK_SHA256 = '22743f09b085e4fae20c90aa2bf30db9b865b1d386e1917e85cd8a5b6c2d702e'
LOCK_CANONICAL_SHA256 = 'ace7581bbed1225a37fd1e90c68c4ff9c6da5a516f3493eea2f7f09bfa244365'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def _lock_identity(lock, host):
    host.require(hashlib.sha256(_canonical(lock)).hexdigest() == LOCK_CANONICAL_SHA256,
                 'literal continuation lock identity')


def load_lock(host):
    path = host.HERE / 'continuation-lock.json'
    _regular(path, host)
    host.require(host.file_hash(path) == LOCK_SHA256, 'literal continuation lock bytes')
    lock = host.bounded_json(path)
    _lock_identity(lock, host)
    return lock


def _regular(path, host):
    path = Path(path)
    host.require(not any(p.is_symlink() for p in (path, *path.parents)), 'no continuation path aliases')
    host.require(path.is_file() and path.stat().st_nlink == 1, 'regular unaliased continuation file')


def _tree(directory, members, host):
    """Check every retained archive member, including absence of extra paths."""
    host.require(directory.is_dir() and not directory.is_symlink(), 'retained directory')
    expected_dirs = {str(p) for name in members for p in Path(name).parents if str(p) != '.'}
    files, dirs = set(), set()
    for path in directory.rglob('*'):
        name = path.relative_to(directory).as_posix()
        host.require(not path.is_symlink(), 'no retained aliases')
        if path.is_dir():
            dirs.add(name)
        else:
            _regular(path, host); files.add(name)
    host.require(files == set(members) and dirs == expected_dirs, 'exact retained member set')
    for name, item in members.items():
        host.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'retained member path')
        path = directory / name
        host.require(path.stat().st_size == item['bytes'] and host.file_hash(path) == item['sha256'],
                     'retained member bytes/hash drift')


def verify_retained(lock, host):
    """Authenticate both complete original artifacts before any resource acquisition."""
    _lock_identity(lock, host)
    base = host.ROOT / REFERENCE
    for name, digest in lock['reference_sha256'].items():
        path = base / name; _regular(path, host)
        host.require(host.file_hash(path) == digest, 'fixed original evidence hash')
    provenance = host.bounded_json(base / 'provenance.json')
    source = lock['source_base']
    host.require(provenance['run_id'] == source['run_id'] and
                 provenance['executed_head_sha'] == source['head_sha'] and
                 provenance['source_freeze_sha256'] == source['source_freeze_sha256'] and
                 provenance['source_frozen_files'] == 66, 'original source/run binding')
    for stage, count in (('generation', 58), ('asr', 108)):
        members = provenance['archives'][stage]['exact_members']
        host.require(len(members) == count, 'original artifact denominator')
        _tree(base / stage, members, host)
        manifest = host.bounded_json(base / stage / 'artifact-freeze.json')
        host.require(manifest['schema'] == 'screen32-evidence-freeze-v1' and
                     manifest['human_gold'] is False and manifest['training_admitted'] is False and
                     manifest['incomplete_temporary_files_excluded'] == [] and
                     manifest['files'] == {name: item['sha256'] for name, item in members.items()
                                          if name != 'artifact-freeze.json'}, 'original raw artifact freeze')
        receipt = host.bounded_json(base / stage / 'host-receipt.json')
        host.require(str(receipt['run_id']) == str(source['run_id']) and
                     receipt['head_sha'] == source['head_sha'], 'original artifact producer identity')
    job = host.bounded_json(base / 'blind/job.json', 65536)
    expected = [{'audio_id': row['audio_id'], 'audio_path': row['audio_path'],
                 'wav_sha256': row['wav_sha256']} for row in lock['clips']]
    host.require(job == {'schema': 'blind-source-screen32-job-v1', 'clips': expected}, 'fixed sixteen blind clips')
    _tree(base / 'blind', {'job.json': provenance['archives']['blind']['exact_members']['job.json']}, host)
    generation = host.bounded_json(base / 'generation/tts/tts-receipt.json')
    ledger = generation['ledger']
    host.require(len(ledger) == 32 and all(row['status'] == 'NOT_RUN' and row['attempts'] == 0
                 for row in ledger[16:]), 'original generation denominator')
    by_producer = {row['cell_id']: row for row in ledger[:16]}
    host.require(len(by_producer) == 16 and len({row['producer_id'] for row in lock['clips']}) == 16,
                 'unique original producer mapping')
    for clip in lock['clips']:
        row = by_producer[clip['producer_id']]
        host.require(row['status'] == 'GENERATED' and row['attempts'] == 1 and
                     row['audio']['wav_sha256'] == clip['wav_sha256'] and
                     row['audio']['pcm_sha256'] == clip['pcm_sha256'] and
                     row['audio']['frames'] == clip['frames'] and
                     clip['producer_path'] == 'generation/tts/' + row['cell_id'] + '.wav',
                     'exact original producer/WAV mapping')
        path = base / clip['producer_path']
        host.require(path.stat().st_size == clip['bytes'] and host.file_hash(path) == clip['wav_sha256'] and
                     provenance['archives']['blind']['exact_members'][clip['audio_path']] ==
                     {'bytes': clip['bytes'], 'sha256': clip['wav_sha256']}, 'original blind WAV binding')
        with wave.open(str(path), 'rb') as wav:
            host.require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) ==
                         (16000, 1, 2, clip['frames']), 'original PCM geometry')
            host.require(hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() == clip['pcm_sha256'],
                         'original decoded PCM hash')
    for model in ('qwen06', 'sensevoice'):
        rows = host.bounded_json(base / 'asr' / model / 'terminal-outcomes.json')
        _terminal_rows(rows, job, host)
        for index, row in enumerate(rows):
            status = 'success' if model == 'qwen06' else ('failed_no_retry' if index == 0 else 'not_run')
            host.require(row['status'] == status and row['started_marker_present'] is (status != 'not_run'),
                         'original per-clip terminal attempt accounting')
        directory = base / 'asr' / model / model
        raw = host.bounded_json(directory / 'raw-freeze.json')
        host.require(raw['schema'] == 'bounded-source-screen-asr-raw-freeze-v1' and raw['labels_joined'] is False,
                     'original label-free raw freeze')
        host.require(set(raw['files']) == {p.name for p in directory.iterdir() if p.name != 'raw-freeze.json'},
                     'original model raw membership')
        for name, digest in raw['files'].items():
            host.require(Path(name).name == name and host.file_hash(directory / name) == digest,
                         'original model raw hash')
        host.require(host.bounded_json(directory / 'blind-job.json') == job, 'both original decoder jobs match input')
    primary = host.bounded_json(base / 'asr/primary-raw-freeze.json')
    host.require(primary['terminal_outcomes_sha256'] == {
        model: lock['reference_sha256']['asr/' + model + '/terminal-outcomes.json']
        for model in ('qwen06', 'sensevoice')} and primary['intended_text_joined'] is False and
        primary['human_gold'] is False and primary['training_admitted'] is False and
        host.file_hash(base / 'asr/primary-raw.json') == primary['primary_raw_sha256'], 'original primary raw freeze')
    return job, provenance


def verify_compatibility(lock, host):
    """Reuse only historical setup/kernel proof, never claim a new source-wide probe."""
    _lock_identity(lock, host)
    proof = lock['probe']; path = host.HERE / proof['original_freeze']; _regular(path, host)
    host.require(host.file_hash(path) == proof['original_freeze_sha256'], 'historical sixty-six source freeze')
    original = host.bounded_json(path)
    host.require(len(original['files']) == 66, 'historical source denominator')
    for name, digest in proof['unchanged_files'].items():
        path = host.HERE / name; _regular(path, host)
        host.require(original['files']['research/source-screen32-v1/' + name] == digest and
                     host.file_hash(path) == digest, 'unchanged setup/kernel source required')
    path = host.HERE / 'hosted_run.py'; _regular(path, host)
    tree = ast.parse(path.read_text())
    for name, digest in proof['host_function_ast_sha256'].items():
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        host.require(len(matches) == 1 and hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()
                     == digest, 'host safety AST identity: ' + name)
    container = host.bounded_json(host.HERE / 'container-lock.json')
    host.require(host.setup_contract_sha256(container['image']) == proof['setup_contract_sha256'],
                 'historical setup create contract')
    receipt = host.verify_successful_probe(proof['reference'], original, proof['original_freeze_sha256'])
    return {'schema': 'screen32-sense-continuation-probe-compatibility-v1',
            'status': 'limited_setup_kernel_compatibility_verified', 'probe_run_id': receipt['run_id'],
            'original_source_freeze_sha256': proof['original_freeze_sha256'],
            'original_probe_artifact_freeze_sha256': proof['reference']['artifact_freeze_sha256'],
            'setup_contract_sha256': proof['setup_contract_sha256'],
            'unchanged_files': proof['unchanged_files'], 'host_function_ast_sha256': proof['host_function_ast_sha256'],
            'new_whole_source_freeze_probed': False, 'new_probe_executed': False,
            'asr_setup_input_bytes': lock['limits']['asr_setup_input_bytes'],
            'scope': 'historical setup create/start/kernel/cleanup only; decoder changes are not probe-certified'}


def verify_admission(env, host):
    release = host.bounded_json(host.HERE / 'continuation-release.json')
    host.require(set(release) == {'schema', 'experiment', 'approved', 'source_freeze_sha256'} and
                 release['schema'] == 'screen32-sense-continuation-release-v1' and
                 release['experiment'] == EXPERIMENT and release['approved'] is True,
                 'exact reviewed continuation admission required')
    freeze = host.verify_first_created_source(env, BRANCH, 1, release['source_freeze_sha256'],
              'source-screen32-sense-continuation-v1.yml', 'continuation-workflow.yml')
    lock = load_lock(host)
    for name in ('sense_continuation.py', 'continuation-lock.json'):
        relative = 'research/source-screen32-v1/' + name
        host.require(relative in freeze['files'] and
                     all(relative not in names for names in freeze['profiles'].values()),
                     'continuation producer context is host-only')
    verify_retained(lock, host)
    compatibility = verify_compatibility(lock, host)
    return freeze, lock, compatibility


def prepare_input(destination, lock, host):
    job, _ = verify_retained(lock, host)
    destination = Path(destination)
    host.require(not os.path.lexists(destination) and
                 not any(p.is_symlink() for p in destination.parents), 'fresh blind destination required')
    destination.mkdir(); (destination / 'audio').mkdir()
    base = host.ROOT / REFERENCE
    shutil.copyfile(base / 'blind/job.json', destination / 'job.json')
    for row in lock['clips']:
        shutil.copyfile(base / row['producer_path'], destination / row['audio_path'])
    host.require(host.file_hash(destination / 'job.json') == lock['reference_sha256']['blind/job.json'],
                 'byte-exact original job copy')
    host.require(host.validate_blind_input(destination) == job, 'exact continuation handoff')
    return job


def _qwen_members(provenance):
    return {name[len('qwen06/'):]: item for name, item in provenance['archives']['asr']['exact_members'].items()
            if name.startswith('qwen06/')}


def retain_prior_qwen(root, lock, host):
    _, provenance = verify_retained(lock, host)
    root = Path(root)
    host.require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'regular output root')
    destination = root / 'qwen06'
    host.require(not os.path.lexists(destination), 'fresh Qwen reuse destination required')
    shutil.copytree(host.ROOT / REFERENCE / 'asr/qwen06', destination)
    members = _qwen_members(provenance); _tree(destination, members, host)
    value = {'schema': 'screen32-sense-continuation-provenance-v1', 'experiment': EXPERIMENT,
             'source_base': lock['source_base'], 'continuation_lock_sha256': LOCK_SHA256,
             'blind_job_sha256': lock['reference_sha256']['blind/job.json'],
             'reused_qwen': {'source_directory': REFERENCE + '/asr/qwen06', 'byte_exact': True,
                            'files': members, 'terminal_outcomes_sha256':
                            lock['reference_sha256']['asr/qwen06/terminal-outcomes.json']},
             'new_qwen_calls': 0, 'new_tts_calls': 0, 'new_whisper_calls': 0,
             'human_gold': False, 'training_admitted': False, 'shipping_approved': False}
    _write_once(root / 'continuation-provenance.json', value)
    return value


def _terminal_rows(rows, job, host):
    host.require(type(rows) is list and len(rows) == 16 and
                 [r['opaque_id'] for r in rows] == [r['audio_id'] for r in job['clips']],
                 'full sixteen terminal denominator')
    for row, clip in zip(rows, job['clips']):
        host.require(row['wav_sha256'] == clip['wav_sha256'] and
                     row['status'] in ('success', 'failed_no_retry', 'not_run') and
                     row['human_gold'] is False and row['training_admitted'] is False and
                     type(row['started_marker_present']) is bool, 'terminal identity/status boundary')
        if row['status'] == 'not_run':
            host.require(row['started_marker_present'] is False and row['raw_text'] is None and
                         row['execution_receipt_sha256'] is None, 'untouched terminal accounting')
        elif row['status'] == 'success':
            host.require(row['started_marker_present'] is True and row['validation'] == 'validated_receipt' and
                         type(row['raw_text']) is str and type(row['execution_receipt_sha256']) is str,
                         'no unvalidated success promotion')
        else:
            host.require(row['raw_text'] is None, 'failed terminal result is not a transcript')


def _write_once(path, value):
    with path.open('xb') as stream:
        stream.write(_canonical(value)); stream.flush(); os.fsync(stream.fileno())


def finalize_cross_run(root, input_job, lock, host):
    job, provenance = verify_retained(lock, host)
    host.require(_canonical(input_job) == _canonical(job), 'fixed continuation job identity')
    root = Path(root)
    host.require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'regular output root')
    _tree(root / 'qwen06', _qwen_members(provenance), host)
    terminal = root / 'sensevoice/terminal-outcomes.json'; _regular(terminal, host)
    rows = host.bounded_json(terminal); _terminal_rows(rows, job, host)
    stopped = False
    for row in rows:
        host.require(not stopped or row['status'] == 'not_run', 'continuation stops at first failure or untouched clip')
        stopped = stopped or row['status'] != 'success'
    prior = host.bounded_json(host.ROOT / REFERENCE / 'asr/sensevoice/terminal-outcomes.json')
    entries = []
    for clip, previous, current in zip(lock['clips'], prior, rows):
        old = clip['prior_sensevoice_attempts']; new = int(current['status'] != 'not_run')
        host.require(old == int(previous['status'] != 'not_run') and old + new <= (2 if old else 1),
                     'per-clip cumulative attempt cap')
        entries.append({'opaque_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'],
                        'prior_status': previous['status'], 'prior_attempts': old,
                        'new_status': current['status'], 'new_attempts': new,
                        'cumulative_attempts': old + new, 'cumulative_attempt_cap': 2 if old else 1,
                        'human_gold': False, 'training_admitted': False})
    new_calls = sum(row['new_attempts'] for row in entries)
    host.require(new_calls <= 16 and sum(row['prior_attempts'] for row in entries) == 1,
                 'bounded cross-run call accounting')
    value = {'schema': 'screen32-sense-continuation-ledger-v1', 'experiment': EXPERIMENT,
             'source_base': lock['source_base'], 'denominator': 16, 'maximum_new_sensevoice_calls': 16,
             'prior_sensevoice_calls': 1, 'new_sensevoice_calls': new_calls,
             'counts_scope': 'Conservative consumed clip attempts; actual model forward completion may be unconfirmed',
             'cumulative_sensevoice_calls': 1 + new_calls, 'new_qwen_calls': 0, 'new_tts_calls': 0,
             'new_whisper_calls': 0, 'rows': entries, 'new_terminal_outcomes_sha256': host.file_hash(terminal),
             'prior_terminal_outcomes_sha256': lock['reference_sha256']['asr/sensevoice/terminal-outcomes.json'],
             'human_review': 'PENDING', 'human_gold': False, 'training_admitted': False,
             'shipping_approved': False, 'acoustic_completeness': 'UNKNOWN'}
    _write_once(root / 'continuation-ledger.json', value)
    return value
