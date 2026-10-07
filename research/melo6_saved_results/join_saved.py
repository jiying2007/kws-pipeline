"""Read-only Melo6 saved-result adapter. No model imports, inference or playback.

Use the publisher's verified source checkout, head and release SHA. An ASR
artifact must already exist. This adapter does not launch or recover execution.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
from types import ModuleType
import zipfile

import compare_melo6 as comparison
import saved_verifier as saved

HERE = Path(__file__).resolve().parent
MODELS = ('sensevoice', 'qwen06')
EXPERIMENT = 'melo-six-phrase-source-screen-asr6-v1'
PLAN_SHA = 'a7a78e618152167c05dd8c65eeda546100db66634e08ce1c5cae63fd9467a5db'
TTS_SHA = '12bccfeb0620e3476de0eb440251da5570377b4dd800393e1e7163d2018a09c0'
BLIND_SHA = '382db1d3e795d7a1b4cd16ba6f4f4467252082f5c04069c3de6b7b70f138c9b2'
INPUT_SHA = 'dbf5e0a82978f298ea97c015b86fe200bf419b3d425aea9970b54e38b9b67e10'
JOB_SHA = 'eeecb1e4d5d0bd36d1c13e462b061526103aa94c6088c054b56ff03766664113'
AUDIT_SHA = '076cee1e6d89228570a9c30e98d384a3957c04cb1eee53a6514f616883494a0c'
COMPARE_SHA = '427ffce6fd334461d9c2c79662a96d7dbdd54624256ef11272e63e70720114de'
GENERATION_SHA = 'e5201e5f4c45eaa319a9cb61241cb816465cf4bc9ac2a9a7cf0436beccfcc3fc'
IDS = [f'clip-{i:06d}' for i in range(1, 7)]
SOURCE_IDS = [f'melo6-{i:03d}' for i in range(1, 7)]
audio = comparison.helpers()
require, digest = audio.require, audio.digest
saved._audio, saved.require, saved.digest = audio, require, digest
saved.MODELS = saved.MODEL_DIRS = MODELS
read, decode, tree = saved._read, saved._json, saved._tree


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()


def pure_functions(path, constants, functions, namespace):
    """Use the existing source definitions without importing an executable runner."""
    raw = read(path)
    parsed = ast.parse(raw, filename=str(path))
    selected = [n for n in parsed.body if
                (isinstance(n, ast.FunctionDef) and n.name in functions) or
                (isinstance(n, ast.Assign) and len(n.targets) == 1 and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants)]
    require(len(selected) == len(constants) + len(functions), 'Missing pure source helpers')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def configure(source_root, release_sha256):
    """Verify the exact caller-supplied published release and component bytes."""
    audio.valid_hash(release_sha256)
    source_root = saved._safe(source_root)
    release_raw = read(source_root / 'research/melo6-source-screen-release.json')
    require(digest(release_raw) == release_sha256, 'Publisher release identity mismatch')
    release = decode(release_raw)
    fields = {'schema', 'approved', 'experiment_id', 'tts_candidate_sha256', 'asr_candidate_sha256',
              'plan_sha256', 'blind_archive_sha256', 'blind_freeze_sha256', 'max_tts_calls', 'max_asr_calls'}
    require(set(release) == fields and release['schema'] == 'melo6-source-screen-release-v1'
            and release['approved'] is True and release['experiment_id'] == 'melo-six-phrase-source-screen-v1',
            'Wrong release scope')
    require(release['tts_candidate_sha256'] == TTS_SHA and release['plan_sha256'] == PLAN_SHA
            and release['blind_archive_sha256'] == BLIND_SHA and release['blind_freeze_sha256'] == INPUT_SHA
            and release['max_tts_calls'] == 6 and release['max_asr_calls'] == 12, 'Wrong fixed Melo pilot')
    audio.valid_hash(release['asr_candidate_sha256'])
    source = source_root / 'research/melo6_asr'
    files = tree(source, 20 * 1024**2)
    freeze_raw = files['candidate-freeze.json']
    require(digest(freeze_raw) == release['asr_candidate_sha256'], 'ASR candidate identity mismatch')
    freeze = decode(freeze_raw)
    require(set(freeze) == {'schema', 'component', 'files'} and freeze['component'] == 'asr'
            and freeze['schema'] == 'melo6-component-code-freeze-v1', 'ASR candidate schema')
    require(set(files) == set(freeze['files']) | {'candidate-freeze.json'}, 'ASR source membership mismatch')
    for name, expected in freeze['files'].items():
        require(digest(files[name]) == expected, 'ASR source hash mismatch: ' + name)
    require(read(source_root / '.github/workflows/melo6-source-screen.yml') == files['workflow.yml'],
            'Published workflow differs from candidate')
    require(digest(read(HERE / 'compare_melo6.py')) == COMPARE_SHA, 'Frozen comparison drift')
    # The PCM module is stdlib-only. Importing it performs no I/O or execution.
    pcm = ModuleType('_melo_saved_pcm')
    sys.modules[pcm.__name__] = pcm
    exec(compile(files['core/pcm/pcm_binding.py'], 'verified/pcm_binding.py', 'exec'), pcm.__dict__)
    helpers = {'Path': Path, 'json': json, 'hashlib': hashlib, 're': re,
               'validate_pcm_descriptor': pcm.validate_pcm_descriptor}
    pure_functions(source / 'core/asr6_contract.py',
                   {'EXPERIMENT', 'SHA', 'BATCH_IDS', 'MAX_FRAMES', 'MAX_JOB_BYTES', 'MAX_WAV_BYTES', 'MAX_ARCHIVE_BYTES'},
                   {'canonical', 'digest', 'unique', 'decode', 'validate_job', 'validate_input_freeze', 'validate_decoder'}, helpers)
    pure_functions(source / 'run_asr6.py', set(), {'recover_rows', 'adapt'}, helpers)
    require(helpers['EXPERIMENT'] == EXPERIMENT and helpers['MAX_FRAMES'] == 160000, 'Wrong ASR identity/input bound')
    saved._helpers = helpers
    return release, files, pcm


def verify_blind(tts_root, artifact_files, job):
    """Check actual opaque audio bytes before reading generation or intent rows."""
    raw = read(Path(tts_root) / 'blind/blind-inputs.zip', 2 * 1024**2)
    freeze_raw = read(Path(tts_root) / 'blind/blind-input-freeze.json')
    require(digest(raw) == BLIND_SHA and digest(freeze_raw) == INPUT_SHA, 'Frozen Melo blind input mismatch')
    require(artifact_files['blind-input-freeze.json'] == freeze_raw, 'ASR input freeze differs from producer')
    freeze = saved._helpers['validate_input_freeze'](freeze_raw, INPUT_SHA)
    expected = {r['path']: r for r in freeze['files']}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == 7 and set(archive.namelist()) == set(expected) and not archive.comment,
                'Blind ZIP membership')
        payloads = {}
        for info in infos:
            require(not info.is_dir() and not info.flag_bits & 1 and not info.extra and not info.comment
                    and info.file_size == expected[info.filename]['bytes'], 'Blind ZIP member')
            payloads[info.filename] = archive.read(info)
            require(digest(payloads[info.filename]) == expected[info.filename]['sha256'], 'Blind ZIP member hash')
    require(payloads['job.json'] == artifact_files['blind-job.json'] and digest(payloads['job.json']) == JOB_SHA,
            'ASR job differs from fixed producer job')
    return {clip['audio_id']: payloads[clip['audio_path']] for clip in job['clips']}


def verify_rows(files, resources, job, adapted, source_files, pcm, waves):
    """Bind recovered rows, final statuses, input samples and durable receipts."""
    locks = decode(source_files['model-locks.json'])
    final = decode(files['final-status.json'])
    require(final['schema'] == 'asr6-final-status-v1' and set(final['models']) == set(MODELS), 'Final status models')
    rows, counts = {}, {}
    for model in MODELS:
        prefix = 'raw/' + model + '/'
        members = {n[len(prefix):]: raw for n, raw in files.items() if n.startswith(prefix)}
        rows[model] = [r['raw_output'] for r in adapted[model]]
        require([r['opaque_id'] for r in rows[model]] == IDS, 'Full recovered denominator')
        require(len(final['models'][model]) == 6, 'Final status denominator')
        inputs, contract = None, None
        if 'decoder-inputs.json' in members:
            inputs = saved._helpers['validate_decoder'](members['decoder-inputs.json'], digest(members['decoder-inputs.json']))
            require(inputs['source_manifest_sha256'] == JOB_SHA and inputs['input_freeze_sha256'] == INPUT_SHA,
                    'Decoder input identity mismatch')
            for clip in inputs['clips']:
                oid = clip['opaque_id']
                d = clip['descriptor']
                wave = waves[oid]
                frames = (len(wave) - 44) // 2
                header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', len(wave)-8, b'WAVE', b'fmt ',
                                     16, 1, 1, 16000, 32000, 2, 16, b'data', frames*2)
                require(wave[:44] == header and len(wave) == 44 + frames*2 and 0 < frames <= 160000,
                        'Actual PCM WAV format/bound')
                floats = b''.join(struct.pack('<f', x[0] / 32768) for x in struct.iter_unpack('<h', wave[44:]))
                require(d['wav']['sha256'] == digest(wave) and d['wav']['bytes'] == len(wave)
                        and d['raw_pcm16']['sha256'] == digest(wave[44:])
                        and d['float32_pcm']['sha256'] == digest(floats)
                        and d['raw_pcm16']['frame_count'] == d['float32_pcm']['frame_count'] == frames,
                        'Actual WAV/PCM/float input binding')
        if 'contract.json' in members:
            contract = decode(members['contract.json'])
            wanted = locks[model]
            require(inputs is not None and contract['experiment_id'] == EXPERIMENT
                    and contract['run_id'] == EXPERIMENT + '-' + model and contract['model_id'] == wanted['model_id']
                    and contract['candidate_count'] == 6 and contract['batch_size'] == 1 and contract['model_dtype'] == 'float32'
                    and contract['source_manifest_sha256'] == JOB_SHA and contract['input_freeze_sha256'] == INPUT_SHA
                    and contract['decoder_manifest_sha256'] == digest(members['decoder-inputs.json'])
                    and all(contract[k] == wanted[k] for k in ('asset_lock_sha256', 'source_lock_sha256', 'expected_schema_sha256')),
                    'Model contract identity/scope')
        attempts = 0
        for row, status, clip in zip(rows[model], final['models'][model], job['clips']):
            oid = clip['audio_id']
            keys = ('opaque_id', 'wav_sha256', 'status', 'completeness', 'quality_flags', 'execution_receipt_sha256')
            require(all(status[k] == row[k] for k in keys) and row['wav_sha256'] == clip['wav_sha256'],
                    'Final status/recovered row mismatch')
            require(status['raw_text_present'] is (row['raw_text'] is not None)
                    and status['raw_text_characters'] == (len(row['raw_text']) if row['raw_text'] is not None else None)
                    and status['recovery_provenance'] == row['recovery_provenance']
                    and status['recovery_warnings'] == row.get('recovery_warnings', []), 'Recovery projection mismatch')
            started = members.get(oid + '.started.json')
            receipt = members.get(oid + '.receipt.json')
            if started:
                attempts += 1
                claim = decode(started)
                require(contract is not None and claim['schema'] == 'asr6-clip-started-v1', 'Started claim without contract')
                require(claim['opaque_id'] == oid and claim['wav_sha256'] == clip['wav_sha256']
                        and claim['execution_contract_sha256'] == digest(members['contract.json'])
                        and claim['decoder_manifest_sha256'] == digest(members['decoder-inputs.json'])
                        and claim['input_binding_sha256'] == inputs['clips_index'][oid]['binding_sha256']
                        and claim['model'] == {'model_id': locks[model]['model_id'], 'revision': locks[model]['asset_lock']['revision'],
                                             'run_id': EXPERIMENT + '-' + model}, 'Started claim identity')
            if row['execution_receipt_sha256'] is not None:
                require(receipt is not None and digest(receipt) == row['execution_receipt_sha256'], 'Completion receipt hash')
                value = decode(receipt)
                require(started is not None and value['schema'] == 'asr6-clip-execution-receipt-v1'
                        and all(value[k] == row[k] for k in keys if k != 'execution_receipt_sha256')
                        and value['raw_text'] == row['raw_text']
                        and all(value[k] == claim[k] for k in ('execution_contract_sha256', 'decoder_manifest_sha256',
                                                             'input_binding_sha256', 'model'))
                        and value['decoder_evidence_sha256'] == digest(members[oid + '.decoder.json']), 'Completion evidence identity')
            require(row['status'] != 'success' or (started is not None and row['execution_receipt_sha256'] is not None),
                    'Success lacks durable started/completion evidence')
        counts[model] = {'saved_started_claims': attempts, 'statuses': dict(Counter(r['status'] for r in rows[model])),
                         'complete_success_rows': sum(r['status'] == 'success' and r['completeness'] == 'complete' for r in rows[model])}
    return rows, counts


def verify_generation(tts_root, plan_path, audit_path, job, waves):
    """Read generation/intent inputs only after both saved recognizer freezes."""
    audit_raw = read(audit_path)
    require(digest(audit_raw) == AUDIT_SHA, 'Independent recovery audit identity')
    audit = decode(audit_raw)
    require(audit['audit_status'] == 'PASS_SIX_SAVED_GENERATION_AND_BLIND_HANDOFF', 'Independent audit result')
    # Revalidate the saved bytes underlying the retained audit, without replaying conversion.
    for row in audit['public_projection']['recovery_evidence_allowlist']:
        raw = read(Path(tts_root) / row['path'])
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'Audited generation evidence changed: ' + row['path'])
    generation = tree(Path(tts_root) / 'generation', 20 * 1024**2)
    freeze_raw = generation['generation-freeze.json']
    require(digest(freeze_raw) == GENERATION_SHA, 'Generation freeze identity')
    freeze = decode(freeze_raw)
    expected = {r['path']: r for r in freeze['files']}
    require(len(expected) == 19 and set(generation) == set(expected) | {'generation-freeze.json'}, 'Generation membership')
    for name, row in expected.items():
        require(len(generation[name]) == row['bytes'] and digest(generation[name]) == row['sha256'], 'Generation frozen file')
    plan_raw = read(plan_path)
    require(digest(plan_raw) == PLAN_SHA, 'Frozen execution plan identity')
    plan = decode(plan_raw)
    require(plan['phrase_order'] == list(comparison.TEXTS), 'Frozen phrase order')
    receipt = decode(generation['generation-receipt.json'])
    require(receipt['plan_sha256'] == PLAN_SHA and receipt['attempts_consumed'] == 6
            and receipt['status'] == 'six_candidates_generated' and len(receipt['generation_rows']) == 6, 'Generation receipt scope')
    for source_id, row, clip in zip(SOURCE_IDS, receipt['generation_rows'], job['clips']):
        require(row['source_id'] == source_id and row['status'] == 'generated_candidate'
                and row['pcm_wav_sha256'] == clip['wav_sha256']
                and generation[source_id + '.pcm16.wav'] == waves[clip['audio_id']], 'Generation-to-blind association')
    return [{'intended_text': text} for text in plan['phrase_order']], audit


def resource_summary(resources, audit, counts):
    fields = ('returncode', 'termination_reason', 'wall_seconds', 'wall_limit_seconds',
              'reaped_children_cpu_seconds', 'sample_count', 'sample_interval_seconds',
              'affinity_core_limit', 'cpu_limit_seconds_per_process', 'rss_threshold_bytes', 'max_sampled')
    return {'ASR': {name: {k: resources[name].get(k, 'UNKNOWN') for k in fields} if name in resources else
                           {'status': 'not_run_or_no_stage_receipt'} for name in ('setup', *MODELS)},
            'ASR_runner_wall_seconds': resources.get('runner_wall_seconds', 'UNKNOWN'),
            'ASR_controlled_download_bytes': resources.get('controlled_download_bytes', 'UNKNOWN'),
            'TTS_controlled_download_bytes': 'UNKNOWN_NOT_RECOUNTED_BY_THIS_ADAPTER',
            'TTS_saved_audit_resources': audit['resources'], 'TTS_attempt_accounting': audit['attempt_accounting'],
            'ASR_attempt_accounting': counts, 'actual_max_sample_gap': 'UNKNOWN',
            'limits': ['RSS/disk maxima are sampled observations, not continuous peaks or hard caps.',
                       'Sample interval is nominal sleep; actual timestamp gaps are not saved.',
                       'CPU limits are per process; reaped CPU is not a hard aggregate process-tree limit.',
                       'Model stages include loading and all six decodes, not pure decode or KWS performance.',
                       'Controlled download bytes cover the recorded package/model paths, not every network socket.',
                       'Started claims are saved attempt evidence, not an independent system-wide call census.']}


def compare_saved(asr_root, source_root, release_sha256, source_head, tts_root, plan_path, audit_path):
    require(re.fullmatch(r'[0-9a-f]{40}', source_head) is not None, 'Publisher source head required')
    release, source_files, pcm = configure(source_root, release_sha256)
    # The unchanged saved verifier checks all three freeze layers before recovery.
    files, resources, job, blind, adapted = saved._verify_asr(asr_root, TTS_SHA, release['asr_candidate_sha256'])
    require(resources['preregistered_plan_sha256'] == PLAN_SHA and resources['blind_job_sha256'] == JOB_SHA
            and resources['blind_input_freeze_sha256'] == INPUT_SHA and resources['blind_archive_sha256'] == BLIND_SHA,
            'Wrong experiment/input identities')
    require(resources['retries'] == 0 and resources['requested_clips'] == 6 and resources['recognizers'] == 2
            and resources['max_asr_calls'] == 12 and resources['raw_frozen_before_plan_join'] is True, 'Wrong attempt/scope budget')
    waves = verify_blind(tts_root, files, job)
    rows, counts = verify_rows(files, resources, job, adapted, source_files, pcm, waves)
    intents, audit = verify_generation(tts_root, plan_path, audit_path, job, waves)
    require(tree(asr_root, 20 * 1024**2) == files, 'ASR bytes changed before intent join')
    binding = {'plan_sha256': PLAN_SHA, 'tts_generation_freeze_sha256': GENERATION_SHA, 'blind_job_sha256': JOB_SHA,
               **{name + '_raw_freeze_sha256': digest(files['raw/' + name + '/model-raw-freeze.json']) for name in MODELS}}
    result = comparison.compare_verified_saved(intents, job['clips'], rows, binding)
    result['saved_artifact_verification'] = {
        'both_model_raw_freezes_verified_before_private_read': True,
        'source_head_from_publisher': source_head, 'source_head_remote_binding_checked_here': False,
        'release_sha256': release_sha256, 'asr_candidate_sha256': release['asr_candidate_sha256'],
        'tts_candidate_sha256': TTS_SHA, 'independent_TTS_audit_sha256': AUDIT_SHA,
        'asr_artifact_freeze_sha256': digest(files['artifact-freeze.json']), 'raw_freeze_sha256': digest(files['raw-freeze.json']),
        'comparison_source_sha256': COMPARE_SHA, 'actual_WAV_PCM_float32_identities_verified': True,
        'source_scope': 'Publisher supplies verified head/release pairing; local candidate bytes verified here.',
        'scope': 'Saved bytes, identities, recovery and comparison; no replay of model behavior or full setup audit.'}
    result['recovered_ASR_rows'] = rows
    result['resource_summary'] = resource_summary(resources, audit, counts)
    result['counts'] = {'generated_candidates': 6, 'ASR_rows_including_failed_and_not_run': 12,
                        'saved_ASR_started_claims': sum(r['saved_started_claims'] for r in counts.values()),
                        'both_ASR_exact_intent_matches': sum(r['machine_text_agreement_only'] for r in result['clips']),
                        'training_admitted': 0, 'human_labels': 0, 'comparison_model_calls': 0}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asr-root', 'source-root', 'release-sha256', 'source-head', 'tts-root', 'plan', 'tts-audit', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output = saved._safe(args.out)
    roots = [saved._safe(p) for p in (args.asr_root, args.source_root, args.tts_root)]
    require(not output.exists() and output not in (saved._safe(args.plan), saved._safe(args.tts_audit))
            and not any(output == root or root in output.parents for root in roots), 'Output must be new and outside inputs')
    result = compare_saved(args.asr_root, args.source_root, args.release_sha256, args.source_head,
                           args.tts_root, args.plan, args.tts_audit)
    with output.open('xb') as stream:
        stream.write(audio.encode_json(result))
    print(json.dumps({'report': str(output), 'overall_result': result['overall_result'], 'counts': result['counts']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
