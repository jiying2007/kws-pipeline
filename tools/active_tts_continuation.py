#!/usr/bin/env python3
"""One bounded continuation of never-claimed cells after a real TTS timeout.

The original queue and its consumed unknown are immutable. This module never
relabels that unknown as a technical failure or turns an old timeout into a
completed job. Inspection, preparation and export use only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_tts_expand as e
import research_queue as q
from research_job_lock import heavy_model_slot
from research_tts_handoff import DATA_SOURCES

c = e.common
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'active-tts-unclaimed-continuation-v2'
JOB_SCHEMA = 'active-tts-unclaimed-job-v2'
CHECKPOINT_SCHEMA = 'active-tts-accounted-checkpoint-v2'
OLD_NS = 600_186_647_856
NEW_NS = 1_794_000_000_000
COMBINED_NS = 2_400_000_000_000
JOB_NS = 600_000_000_000
OLD_SOURCES = (*e.OWN_SOURCES, 'tools/research_queue.py')
SOURCES = (*OLD_SOURCES, 'tools/research_tts_partial.py', 'tools/research_tts_handoff.py', 'tools/active_tts_continuation.py',
    'configs/research/active-plan-20261105.json')
OUTPUTS = e.OUTPUTS
ML_MODULES = ('numpy', 'torch', 'onnxruntime', 'qwen_tts', 'soundfile', 'scipy')

require = e.require
once = e.once


def pin(path):
    path = q.safe_path(path)
    return {'path': str(path), **q.file_record(path)}


def check_pin(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'} and
        pin(value['path']) == value, 'CONTINUATION_PIN_CHANGED')
    return Path(value['path'])


def read(path):
    return q.loads(q.read_bytes(path, 32 * 1024 * 1024))


def pinned(value):
    return read(check_pin(value))


def inventory(root):
    root = q.safe_path(root)
    return {p.relative_to(root).as_posix(): q.file_record(p)
        for p in sorted(root.rglob('*')) if p.is_file()}


def source_records(root=ROOT):
    return {name: q.file_record(Path(root) / name) for name in SOURCES}


def verify_protocol(protocol):
    require(protocol['schema'] == 'kws-d1-post-timeout-continuation-predeclaration-v1' and
        protocol['new_generation_maximum_calls'] == 58 and protocol['pause_after_jobs'] == 3 and
        protocol['known_returned_cells'] == 197 and protocol['old_stage_returned_cells'] == 21 and
        protocol['old_stage_consumed_calls'] == 22 and
        protocol['first_pause_requires_ASR_accounted232_before_last_two_jobs'] is True and
        protocol['unknown_receives_asr_or_native_call'] is False and
        protocol['unknown_becomes_failed_native'] is False and
        protocol['unknown_has_observed_text_or_pcm'] is False and protocol['unknown_is_negative'] is False,
        'CONTINUATION_PROTOCOL_SCOPE')
    budget = protocol['budget']
    require(budget['prior_actual_elapsed_ns'] == OLD_NS and budget['new_queue_global_minutes'] == 29.9 and
        budget['new_queue_global_ns'] == NEW_NS and budget['combined_allocation_ns'] == COMBINED_NS and
        budget['per_job_maximum_ns'] == JOB_NS and
        budget['combined_declared_actual_ceiling_ns'] == OLD_NS + NEW_NS and
        budget['cleanup_reserve_ns'] == COMBINED_NS - OLD_NS - NEW_NS,
        'FIXED_CONTINUATION_BUDGET_REQUIRED')
    for key in ('old_tts_stage', 'old_queue_terminal', 'partial_audit', 'recovery_contract_review',
                'original_training_protocol'):
        check_pin(protocol[key])


def old_helper(protocol_pin, source_root, source_files, *, proof=None, timeout=30):
    request = {'protocol': protocol_pin, 'source_root': str(source_root), 'source_files': source_files,
        'mode': 'freeze' if proof is None else 'verify'}
    if proof is not None:
        request['proof'] = proof
    # A new interpreter keeps the old controller/module globals separate from
    # this queue. It inherits the process group and is bounded by the caller.
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_tts_partial.py')],
        input=q.canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, check=False, start_new_session=False)
    require(result.returncode == 0, 'ORIGINAL_PARTIAL_VERIFICATION_FAILED:' + result.stderr.decode()[-4000:])
    require(len(result.stdout) <= 4 * 1024 * 1024, 'PARTIAL_PROOF_RESPONSE_BOUND')
    value = q.loads(result.stdout)
    require(value['ml_imports'] == 0 and value['pgid'] == os.getpgrp(), 'PARTIAL_HELPER_PROCESS_SCOPE')
    return value['proof']


def data_helper(verifier, sealed_pin=None):
    request_body = {'source_root': verifier['source_root'], 'source_files': verifier['files'],
        'sealed_stage': sealed_pin, 'expected_count': 232, 'action': 'verify' if sealed_pin else 'preflight'}
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_tts_handoff.py')],
        input=q.canonical(request_body), capture_output=True, timeout=60 if sealed_pin else 15,
        check=False, start_new_session=False)
    require(result.returncode == 0, 'ASR232_FULL_SOURCE_REPLAY_FAILED:' + result.stderr.decode()[-4000:])
    verified = q.loads(result.stdout)
    require(verified.pop('pgid') == os.getpgrp() and type(verified.pop('pid')) is int and
        verified['source_preflight_passed'] is True and verified['ml_imports'] == verified['new_model_calls'] == 0 and
        verified['verified'] == verified['verification_performed'] == (sealed_pin is not None) and
        verified['source_root'] == verifier['source_root'] and verified['source_files'] == verifier['files'] and
        verified['sealed_stage'] == sealed_pin and verified['expected_count'] == 232,
        'ASR232_VERIFIER_PROCESS_OR_SOURCE_CHANGED')
    return verified


def verifier_contract(prepared, control):
    return {'schema': 'active-tts-asr232-late-source-contract-v2',
        'entrypoint': 'automation.active_asr_weak_view.verify_accounted_seal',
        'source_files': list(DATA_SOURCES), 'expected_count': 232,
        'success_contract': {'schema': 'd1-accounted-selective-seal-verification-v3',
            'status': 'verified', 'verified': True, 'accounted_prefix_count': 232, 'new_model_calls': 0,
            'ml_imports': 0, 'full_replay_completed': True},
        'control_source_root': str(Path(control) / 'source-snapshot/data'),
        'retained_source_root': str(Path(prepared) / 'asr-verifier-source'),
        'freeze_after_actual_asr232_completion': True,
        'must_match_all_new_view_producers': True, 'live_source_fallback': False}


def verify_late_source(d, verifier, views):
    contract = d['asr232_verifier_contract']
    require(set(verifier) == {'source_root', 'control_source_root', 'files', 'local_commit',
        'public_commit', 'git_tree'} and verifier['source_root'] == contract['retained_source_root'] and
        verifier['control_source_root'] == contract['control_source_root'] and
        set(verifier['files']) == set(DATA_SOURCES) and all(type(verifier[key]) is str and
            len(verifier[key]) == 40 and all(ch in '0123456789abcdef' for ch in verifier[key])
            for key in ('local_commit', 'public_commit', 'git_tree')), 'ASR232_LATE_SOURCE_SCOPE')
    for root in (verifier['source_root'], verifier['control_source_root']):
        require(inventory(root) == verifier['files'], 'ASR232_FROZEN_SOURCE_BYTES_CHANGED')
    expected = {name: row['sha256'] for name, row in verifier['files'].items()}
    for entry in views:
        root = Path(entry['path']); manifest = read(root / 'MANIFEST.json')
        require(c.sha_file(root / 'MANIFEST.json') == entry['manifest_sha256'] and
            c.sha_file(root / 'report.json') == manifest['report.json'], 'ASR232_PRODUCER_VIEW_CHANGED')
        report = read(root / 'report.json')
        require(report['producer']['source_files_sha256'] == expected, 'ASR232_VERIFIER_NOT_ACTUAL_VIEW_PRODUCER')


def prior176(spec_pin, plan):
    spec = pinned(spec_pin)
    require(spec['schema'] == 'd1-sealed-prior-evidence-v1' and spec['prefix_count'] == 176 and
        spec['plan_sha256'] == e.digest(plan) and spec['excluded_exports'] == [], 'PRIOR176_SCOPE')
    ids = []
    for entry in spec['views']:
        root = q.safe_path(entry['path']); manifest = root / 'MANIFEST.json'
        require(q.file_record(manifest)['sha256'] == entry['manifest_sha256'], 'PRIOR176_MANIFEST_CHANGED')
        record = read(manifest)
        require(q.file_record(root / 'report.json')['sha256'] == record['report.json'], 'PRIOR176_REPORT_CHANGED')
        report = read(root / 'report.json')
        require(report['source_adapter']['plan_sha256'] == spec['plan_sha256'], 'PRIOR176_REPORT_PLAN')
        for row in report['items']:
            source = row['source']
            require(source['source_plan_sha256'] == spec['plan_sha256'], 'PRIOR176_SOURCE_PLAN')
            ids.append(source['source_id'])
    expected = [row['cell_id'] for row in plan['cells'][:176]]
    require(len(ids) == len(set(ids)) == 176 and set(ids) == set(expected), 'PRIOR176_EXACT_IDS')
    return spec


def chunks_for(plan):
    ids = [row['cell_id'] for row in plan['cells']]
    require(len(ids) == len(set(ids)) == 256, 'EXACT_256_PLAN')
    return [ids[198:210], ids[210:222], ids[222:232], ids[232:244], ids[244:256]]


def expected_job(declaration, index):
    return {'schema': JOB_SCHEMA, 'declaration': str(Path(declaration['prepared']) / 'declaration.json'),
        'declaration_sha256': c.sha_file(Path(declaration['prepared']) / 'declaration.json'),
        'job_id': 'continuation-' + str(index + 1).zfill(3), 'chunk_index': index,
        'cells': declaration['chunks'][index], 'maximum_new_calls': len(declaration['chunks'][index]),
        'timeout_seconds': 600, 'requires_asr232_handoff': index >= 3}


def queue_request(declaration):
    jobs = []
    for index in range(5):
        job = expected_job(declaration, index)
        path = Path(declaration['prepared']) / 'jobs' / (job['job_id'] + '.json')
        require(read(path) == job, 'CONTINUATION_JOB_CHANGED')
        jobs.append({'job_id': job['job_id'], 'recipe': str(path), 'recipe_sha256': c.sha_file(path),
            'argv': [declaration['runtime_python'], '-I', '-B', str(ROOT / 'tools/active_tts_continuation.py'),
                'run', '--recipe', '{recipe}', '--output', '{output_dir}', '--heavy-lock', declaration['heavy_lock']],
            'cwd': str(ROOT), 'depends_on': [] if index == 0 else ['continuation-' + str(index).zfill(3)],
            'timeout_minutes': 10, 'outputs': list(OUTPUTS)})
    return {'schema': q.SCHEMA, 'queue_id': declaration['continuation_id'], 'global_budget_minutes': 29.9, 'jobs': jobs}


def registration(declaration):
    prepared = Path(declaration['prepared'])
    return {'schema': 'active-tts-single-continuation-registration-v2',
        'declaration': pin(prepared / 'declaration.json'), 'queue_request': pin(prepared / 'queue-request.json'),
        'protocol_declaration': declaration['protocol_declaration'],
        'original_stage_sha256': declaration['old_stage']['sha256'],
        'original_terminal_sha256': declaration['old_terminal']['sha256'],
        'never_claimed_cell_ids': [name for chunk in declaration['chunks'] for name in chunk],
        'permanently_consumed_unknown_ids': declaration['consumed_unknown_ids'],
        'maximum_new_calls': 58, 'automatic_retry': False, 'further_continuation_allowed': False,
        'new_queue_budget_ns': NEW_NS, 'old_actual_ns': OLD_NS, 'combined_allocation_ns': COMBINED_NS}


def verify_registration(declaration):
    expected = registration(declaration)
    require(read(declaration['registration_path']) == expected and
        read(Path(declaration['prepared']) / 'registration.json') == expected and
        q.file_record(declaration['registration_path']) == q.file_record(Path(declaration['prepared']) / 'registration.json'),
        'SINGLE_CONTINUATION_REGISTRATION_CHANGED')


def publish_registration(declaration):
    value = registration(declaration)
    once(Path(declaration['prepared']) / 'registration.json', value)
    once(declaration['registration_path'], value)


def load_declaration(path, *, verify_old=True):
    path = q.safe_path(path); d = read(path)
    require(d['schema'] == SCHEMA and d['execution_root'] == str(ROOT) and
        d['execution_source'] == source_records() and q.ID.fullmatch(d['continuation_id']) and
        path == Path(d['prepared']) / 'declaration.json' and d['queue_root'] == str(Path(d['prepared']) / 'queue') and
        d['maximum_new_calls'] == 58 and d['new_queue_global_minutes'] == 29.9 and
        d['per_job_maximum_seconds'] == 600 and d['combined_allocation_ns'] == COMBINED_NS and
        d['prior_actual_elapsed_ns'] == OLD_NS and d['pause_after_jobs'] == 3 and
        d['max_retries'] == 0 and d['further_continuation_allowed'] is False and
        d['paired_cases_included'] is False and d['training_admitted'] is False and
        d['runtime_python'] == str(Path(d['runtime']) / 'venv/bin/python') and
        d['asr232_gate_path'] == str(Path(d['prepared']) / 'asr232-gate.json') and
        Path(d['asr232_control']).is_absolute(), 'CONTINUATION_SCOPE_CHANGED')
    e.check_files(Path(d['prepared']) / 'execution-source', d['execution_source'])
    require(d['asr232_verifier_contract'] == verifier_contract(d['prepared'], d['asr232_control']),
        'ASR232_VERIFIER_CONTRACT_CHANGED')
    protocol = pinned(d['protocol_declaration']); verify_protocol(protocol)
    proof = pinned(d['old_partial_proof'])
    require(d['old_stage'] == protocol['old_tts_stage'] == proof['old_stage'] and
        d['old_terminal'] == protocol['old_queue_terminal'] == proof['old_terminal'] and
        proof['protocol_declaration'] == d['protocol_declaration'], 'CONTINUATION_OLD_BINDING')
    stage = pinned(d['old_stage'])
    parent, prepared, plan, bulk, binding = e.verify_parent(stage['base_recipe'], stage['base_recipe_sha256'])
    require(d['binding'] == binding == proof['binding'] and d['ledger'] == stage['ledger'] == proof['ledger'] and
        d['heavy_lock'] == stage['heavy_lock'] and d['runtime'] == parent['runtime'] and
        d['chunks'] == protocol['new_chunks'] == chunks_for(plan) and
        [n for chunk in d['chunks'] for n in chunk] == protocol['never_claimed_ids'] and
        d['consumed_unknown_ids'] == protocol['consumed_unknown_ids'] == [plan['cells'][197]['cell_id']] and
        d['registration_path'] == str(Path(d['ledger']) / 'continuations' / (d['old_stage']['sha256'] + '.json')),
        'CONTINUATION_FIXED_ROSTER_CHANGED')
    require(proof['old_source_root'] == stage['execution_root'] and
        proof['old_source_files'] == {name: q.file_record(Path(stage['execution_root']) / name) for name in OLD_SOURCES},
        'ORIGINAL_GENERATION_SOURCE_CHANGED')
    prior176(d['prior_evidence'], plan)
    if verify_old:
        require(old_helper(d['protocol_declaration'], proof['old_source_root'], proof['old_source_files'],
            proof=d['old_partial_proof']) == proof, 'ORIGINAL_PARTIAL_PROOF_CHANGED')
    require(read(Path(d['prepared']) / 'queue-request.json') == queue_request(d), 'CONTINUATION_QUEUE_REQUEST_CHANGED')
    verify_registration(d)
    return d, proof, parent, prepared, plan, bulk


def queue_state(d, *, required=False):
    root = Path(d['queue_root']); expected = queue_request(d)
    if not root.exists():
        require(not required, 'CONTINUATION_QUEUE_NOT_CREATED')
        return {job['job_id']: {'status': 'pending'} for job in expected['jobs']}
    root, queue, request, queue_sha = q.load_queue(root)
    require(request == expected and (root / 'request.json').read_bytes() ==
        (Path(d['prepared']) / 'queue-request.json').read_bytes() and queue['heavy_lock'] == d['heavy_lock'] and
        queue['automatic_retries'] == 0 and queue['historical_jobs_imported'] == 0,
        'CONTINUATION_ACTUAL_QUEUE_CHANGED')
    states = {}; actual_before = 0
    for job in request['jobs']:
        name = job['job_id']; folder = root / 'jobs' / name
        state = q.result_for(root, job, queue_sha, check_active=False)
        if state['status'] == 'pending':
            states[name] = state
            continue
        claim = read(folder / 'claim.json')
        require(claim['schema'] == 'kws-local-research-job-claim-v1' and claim['automatic_retry'] is False and
            type(claim['allowed_runner_nanoseconds']) is int and
            claim['allowed_runner_nanoseconds'] == min(JOB_NS, NEW_NS - actual_before) and
            claim['global_actual_before_nanoseconds'] == actual_before and
            claim['deadline_monotonic_ns'] == claim['started_monotonic_ns'] + claim['allowed_runner_nanoseconds'],
            'CONTINUATION_CUMULATIVE_BUDGET_CHANGED')
        require(all(states[name]['status'] == 'completed' for name in job['depends_on']), 'CONTINUATION_PREREQUISITE')
        dependencies = {name: {'receipt': str(root / 'jobs' / name / 'result.json'),
            'receipt_sha256': c.sha_file(root / 'jobs' / name / 'result.json'),
            'outputs_root': str(root / 'jobs' / name / 'outputs'), 'outputs': states[name]['outputs']}
            for name in job['depends_on']}
        require(read(folder / 'dependencies.json') == dependencies and
            claim['dependencies_sha256'] == q.sha(q.canonical(dependencies)), 'CONTINUATION_ACTUAL_DEPENDENCIES')
        if not (folder / 'result.json').exists():
            state = {'status': 'claimed-without-terminal', 'actual_elapsed_ns': None}
        elif state['actual_elapsed_ns'] is not None:
            actual_before += state['actual_elapsed_ns']
        states[name] = {**state, 'claim': claim}
    return states


def budget(states, count):
    rows = [states['continuation-' + str(i + 1).zfill(3)] for i in range(count)]
    complete = all(row['status'] == 'completed' for row in rows)
    known = sum(row.get('actual_elapsed_ns') or 0 for row in rows)
    return {'old_actual_elapsed_ns': OLD_NS, 'new_actual_known_ns': known,
        'combined_actual_known_ns': OLD_NS + known, 'new_actual_total_complete': complete,
        'new_queue_allocation_ns': NEW_NS, 'combined_allocation_ns': COMBINED_NS,
        'combined_declared_ceiling_ns': OLD_NS + NEW_NS,
        'cleanup_reserve_ns': COMBINED_NS - OLD_NS - NEW_NS,
        'new_queue_overrun_ns': max(0, known - NEW_NS),
        'observed_combined_overrun_ns': max(0, OLD_NS + known - COMBINED_NS),
        'selected_job_count': count,
        'completed_job_count': sum(row['status'] == 'completed' for row in rows),
        'within_new_queue_allocation': known <= NEW_NS,
        'within_combined_allocation': OLD_NS + known <= COMBINED_NS,
        'historical_original_smoke_actual_total_ns': None,
        'historical_original_smoke_actual_total_complete': False}


def prepare(args):
    require(type(args.continuation_id) is str and q.ID.fullmatch(args.continuation_id), 'CONTINUATION_SAFE_ID')
    require(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
        cwd=ROOT, text=True) == '', 'FREEZE_SOURCE_BEFORE_PREPARE')
    protocol_pin = pin(args.protocol)
    require(protocol_pin['sha256'] == args.protocol_sha256, 'CONTINUATION_PROTOCOL_PIN')
    protocol = pinned(protocol_pin); verify_protocol(protocol)
    old_stage = pinned(protocol['old_tts_stage'])
    source_root = q.safe_path(old_stage['execution_root'])
    source_files = {name: q.file_record(source_root / name) for name in OLD_SOURCES}
    prepared_root = q.safe_path(args.output)
    require(not prepared_root.exists(), 'NEW_CONTINUATION_PREPARED_ROOT_REQUIRED')
    prior_pin = pin(args.prior_evidence)
    require(prior_pin['sha256'] == args.prior_sha256, 'CONTINUATION_PRIOR_PIN')
    with e.ledger_lock(old_stage['ledger']):
        proof = old_helper(protocol_pin, source_root, source_files)
        plan = pinned(proof['plan']); prior176(prior_pin, plan)
        reg_path = Path(old_stage['ledger']) / 'continuations' / (protocol['old_tts_stage']['sha256'] + '.json')
        require(not reg_path.exists(), 'CONTINUATION_ALREADY_REGISTERED_NO_THIRD_ATTEMPT')
        parent, _, _, _, binding = e.verify_parent(old_stage['base_recipe'], old_stage['base_recipe_sha256'])
        prepared_root.mkdir(mode=0o700)
        once(prepared_root / 'old-partial-proof.json', proof)
        sources = source_records()
        for name in sources:
            e.atomic_bytes(prepared_root / 'execution-source' / name, (ROOT / name).read_bytes())
        d = {'schema': SCHEMA, 'continuation_id': args.continuation_id, 'prepared': str(prepared_root),
            'execution_root': str(ROOT), 'execution_source': sources,
            'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'protocol_declaration': protocol_pin, 'old_stage': protocol['old_tts_stage'],
            'old_terminal': protocol['old_queue_terminal'], 'old_partial_proof': pin(prepared_root / 'old-partial-proof.json'),
            'prior_evidence': prior_pin, 'binding': binding, 'chunks': chunks_for(plan),
            'consumed_unknown_ids': protocol['consumed_unknown_ids'], 'maximum_new_calls': 58,
            'new_queue_global_minutes': 29.9, 'per_job_maximum_seconds': 600,
            'combined_allocation_ns': COMBINED_NS, 'prior_actual_elapsed_ns': OLD_NS, 'pause_after_jobs': 3,
            'max_retries': 0, 'further_continuation_allowed': False, 'paired_cases_included': False,
            'training_admitted': False, 'runtime': parent['runtime'], 'runtime_python': old_stage['runtime_python'],
            'heavy_lock': old_stage['heavy_lock'], 'ledger': old_stage['ledger'],
            'queue_root': str(prepared_root / 'queue'), 'registration_path': str(reg_path),
            'asr232_gate_path': str(prepared_root / 'asr232-gate.json'),
            'asr232_control': str(q.safe_path(args.asr232_control)),
            'asr232_verifier_contract': verifier_contract(prepared_root, q.safe_path(args.asr232_control))}
        once(prepared_root / 'declaration.json', d)
        for index in range(5):
            job = expected_job(d, index)
            once(prepared_root / 'jobs' / (job['job_id'] + '.json'), job)
        request = queue_request(d)
        q.validate_request(request)
        q.validate_plan(read(ROOT / 'configs/research/active-plan-20261105.json'))
        once(prepared_root / 'queue-request.json', request)
        # Registration is the last publication. No alternative prepared root
        # or third continuation can reserve this original timed-out stage.
        publish_registration(d)
    return inspect(prepared_root / 'declaration.json')


def inspect(path):
    d, proof, _, _, plan, _ = load_declaration(path)
    states = queue_state(d); cells = {r['cell_id']: r for r in plan['cells']}
    refs = {r['reference_id']: r for r in plan['references']}
    returned = []; pending = []; unknown = []
    for name in [name for chunk in d['chunks'] for name in chunk]:
        cell = cells[name]
        try:
            row = e.read_cell(d['ledger'], name, e.cell_identity(d['binding'], cell, refs[cell['reference_id']]))
        except ValueError as error:
            if str(error) != 'CONSUMED_WITHOUT_TERMINAL:' + name:
                raise
            unknown.append(name)
        else:
            (pending if row is None else returned).append(name)
    return {'schema': 'active-tts-continuation-status-v2', 'declaration': pin(path),
        'old_returned': 197, 'old_unobservable_consumed': d['consumed_unknown_ids'],
        'new_returned_ids': returned, 'new_pending_ids': pending, 'new_consumed_unknown_ids': unknown,
        'jobs': {name: {'status': row['status'], 'actual_elapsed_ns': row.get('actual_elapsed_ns')}
            for name, row in states.items()}, 'budget': budget(states, 5), 'new_model_calls_during_inspection': 0}


def verify_call(d, cell, ref, record):
    output = Path(record['output_root']); name = cell['cell_id']
    call = read(output / 'native' / name / 'call.json')
    mode = 'ICL' if ref['reference_text_status'] in ('verified', 'dataset-provided-content-checked') else 'x-vector-only'
    require(call['schema'] == 'active-tts-generation-call-v1' and call['cell_id'] == name and
        call['source_family_id'] == cell['source_family_id'] and call['reference_id'] == ref['reference_id'] and
        call['reference_wav_sha256'] == ref['wav_sha256'] and call['reference_pcm_sha256'] == ref['pcm_sha256'] and
        call['seed'] == cell['seed'] and call['generation'] == d['binding']['generation'] and
        call['intended_text'] == cell['prompt'] and call['reference_mode'] == mode and
        call['reference_text_passed'] == (mode == 'ICL') and call['attempt'] == 1 and
        read(output / 'generation/cells' / name / 'started.json') ==
        e.expected_start(cell, ref, d['binding']['plan_canonical_sha256']), 'CONTINUATION_CALL_CHANGED')


def returned_row(d, job, cell, ref):
    name = cell['cell_id']; ledger = Path(d['ledger'])
    value = e.read_cell(ledger, name, e.cell_identity(d['binding'], cell, ref))
    require(value is not None and value['origin'] == 'extension-call', 'CONTINUATION_RETURN_REQUIRED')
    rp = pin(ledger / 'cells' / (name + '.json')); cp = pin(ledger / 'claims' / (name + '.json'))
    claim = pinned(cp)
    require(value['claim_sha256'] == cp['sha256'] and rp['sha256'] == e.queue_digest(value) and
        claim['owner'] == {'stage_sha256': job['declaration_sha256'], 'job_id': job['job_id'],
            'job_recipe_sha256': c.sha_file(Path(d['prepared']) / 'jobs' / (job['job_id'] + '.json'))} and
        value['output_root'] == str(Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs'),
        'CONTINUATION_RETURN_OWNER_CHANGED')
    verify_call(d, cell, ref, value)
    return {'cell_id': name, 'origin': 'continuation-completed', 'owner_id': job['job_id'],
        'ledger_record_sha256': rp['sha256'], 'record_pin': rp, 'claim_pin': cp,
        'record': value, 'claim': claim, 'eligibility': e.asr_eligibility(value)}


def expected_outcomes(d, job, plan):
    cells = {r['cell_id']: r for r in plan['cells']}; refs = {r['reference_id']: r for r in plan['references']}
    return {'schema': 'active-tts-continuation-outcomes-v2',
        'declaration': pin(Path(d['prepared']) / 'declaration.json'),
        'registration': pin(d['registration_path']), 'job_id': job['job_id'],
        'job_recipe': pin(Path(d['prepared']) / 'jobs' / (job['job_id'] + '.json')),
        'plan_canonical_sha256': d['binding']['plan_canonical_sha256'], 'binding': d['binding'],
        'selected_cells': job['cells'], 'new_call_ids': job['cells'], 'reused_cell_ids': [],
        'rows': [returned_row(d, job, cells[name], refs[cells[name]['reference_id']]) for name in job['cells']]}


def verify_outputs(d, job, plan):
    output = Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs'
    summary = read(output / 'summary.json'); recorded = read(output / 'inventory.json')['files']
    e.check_files(output, recorded)
    require(set(inventory(output)) == set(recorded) | {'inventory.json', 'summary.json'}, 'CONTINUATION_OUTPUT_ROSTER')
    expected = expected_outcomes(d, job, plan)
    require(summary['schema'] == 'active-tts-continuation-result-v2' and summary['status'] == 'completed' and
        summary['job_recipe_sha256'] == expected['job_recipe']['sha256'] and
        summary['declaration_sha256'] == job['declaration_sha256'] and
        summary['new_generation_calls'] == len(job['cells']) and summary['new_call_ids'] == job['cells'] and
        summary['reused_cell_ids'] == [] and summary['inventory'] == q.file_record(output / 'inventory.json') and
        summary['outcomes'] == q.file_record(output / 'outcomes.json') and
        read(output / 'outcomes.json') == expected, 'CONTINUATION_FINISHED_OUTPUTS_CHANGED')
    return expected


def new_owner(d, job):
    root = Path(d['queue_root']); folder = root / 'jobs' / job['job_id']
    return {'origin': 'continuation-completed', 'job_id': job['job_id'],
        'stage': pin(Path(d['prepared']) / 'declaration.json'),
        'job_recipe': pin(root / 'recipes' / (job['job_id'] + '.json')), 'queue_root': str(root),
        'queue_files': {leaf: pin(root / leaf) for leaf in ('queue.json', 'request.json', 'active-plan.json')},
        'job_root': str(folder), 'job_files': inventory(folder), 'result': pin(folder / 'result.json'),
        'outputs_root': str(folder / 'outputs'), 'source_root': str(ROOT), 'source_files': d['execution_source']}


def build_checkpoint(d, proof, plan, prefix, previous=None):
    require(prefix in (232, 256), 'CHECKPOINT_PREFIX_SCOPE')
    states = queue_state(d, required=True); count = 3 if prefix == 232 else 5
    owners = {'original-232-chunk-001': proof['owner']}; rows = list(proof['returned_rows'])
    for index in range(count):
        job = expected_job(d, index)
        require(states[job['job_id']]['status'] == 'completed', 'CHECKPOINT_REQUIRES_ACTUAL_COMPLETED_JOB')
        value = verify_outputs(d, job, plan)
        owners[job['job_id']] = new_owner(d, job)
        rows.extend(value['rows'])
    ids = [cell['cell_id'] for cell in plan['cells']]
    require([row['cell_id'] for row in rows] == ids[176:197] + ids[198:prefix], 'CHECKPOINT_RETURNED_ROSTER')
    unknown = proof['unobservable_consumed']
    eligible = [row['cell_id'] for row in rows if row['eligibility']['eligible']]
    excluded = [{'cell_id': row['cell_id'], 'reason': row['eligibility']['reason'],
        'ledger_record_sha256': row['ledger_record_sha256']} for row in rows if not row['eligibility']['eligible']]
    result = {'schema': CHECKPOINT_SCHEMA, 'declaration': pin(Path(d['prepared']) / 'declaration.json'),
        'registration': pin(d['registration_path']), 'protocol_declaration': d['protocol_declaration'],
        'prior_evidence': d['prior_evidence'], 'old_partial_proof': d['old_partial_proof'],
        'plan': proof['plan'], 'plan_canonical_sha256': d['binding']['plan_canonical_sha256'],
        'binding': d['binding'], 'prefix_count': prefix, 'prior_count': 176,
        'planned_cell_ids': ids[:prefix], 'prior_cell_ids': ids[:176], 'owners': owners,
        'returned_rows': rows, 'unobservable_consumed': unknown, 'eligible_cell_ids': eligible,
        'excluded_native_rows': excluded, 'accounted_cell_ids': ids[:prefix],
        'budget': budget(states, count), 'previous_checkpoint': previous,
        'new_model_calls_during_checkpoint': 0, 'training_admitted': False}
    require(len(rows) + len(unknown) + 176 == prefix and len(unknown) == 1 and
        set(eligible).isdisjoint(row['cell_id'] for row in unknown), 'CHECKPOINT_ACCOUNTED_UNION')
    if prefix == 232:
        require(previous is None, 'FIRST_CHECKPOINT_HAS_NO_PREVIOUS')
    else:
        old = pinned(previous)
        require(old['prefix_count'] == 232 and old == build_checkpoint(d, proof, plan, 232),
            'PREVIOUS_CHECKPOINT_CHANGED')
    return result


def verify_checkpoint(path, expected_sha=None):
    if expected_sha is not None:
        require(c.sha_file(path) == expected_sha, 'CHECKPOINT_FILE_PIN_CHANGED')
    value = read(path)
    require(value['schema'] == CHECKPOINT_SCHEMA, 'CHECKPOINT_SCHEMA')
    d, proof, _, _, plan, _ = load_declaration(check_pin(value['declaration']))
    require(value == build_checkpoint(d, proof, plan, value['prefix_count'], value['previous_checkpoint']),
        'CHECKPOINT_CONTENT_CHANGED')
    require(not any(name in sys.modules for name in ML_MODULES), 'CHECKPOINT_ML_IMPORT')
    return value


def checkpoint(args):
    d, proof, _, _, plan, _ = load_declaration(args.declaration)
    previous = pin(args.previous_checkpoint) if args.previous_checkpoint else None
    if previous is not None:
        verify_checkpoint(previous['path'], previous['sha256'])
    states = queue_state(d, required=True)
    if args.prefix_count == 232:
        require(all(states['continuation-' + str(i).zfill(3)]['status'] == 'pending' for i in (4, 5)),
            'INITIAL_CHECKPOINT_MUST_PRECEDE_LAST_TWO_JOBS')
    value = build_checkpoint(d, proof, plan, args.prefix_count, previous)
    once(args.output, value)
    return {'checkpoint': pin(args.output), 'returned': len(value['returned_rows']),
        'eligible': len(value['eligible_cell_ids']), 'unknown': len(value['unobservable_consumed']),
        'budget': value['budget']}


def export(args):
    value = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)
    output = q.safe_path(args.output); require(not output.exists(), 'NEW_EXPORT_ROOT_REQUIRED')
    reused = []
    if value['previous_checkpoint'] is not None:
        previous = verify_checkpoint(value['previous_checkpoint']['path'], value['previous_checkpoint']['sha256'])
        reused = [row['cell_id'] for row in previous['returned_rows']]
    rows = {row['cell_id']: row for row in value['returned_rows']}
    ids = [name for name in value['eligible_cell_ids'] if name not in reused]
    output.mkdir(parents=True, mode=0o700); partitions = []
    for index, offset in enumerate(range(0, len(ids), 28)):
        selected = ids[offset:offset + 28]; name = 'partition-' + str(index + 1).zfill(3)
        archive = output / (name + '.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for cell_id in selected:
                record = rows[cell_id]['record']
                path = Path(record['output_root']) / record['native_wav']
                require(q.file_record(path) == record['files'][record['native_wav']], 'EXPORT_NATIVE_CHANGED')
                stream.write(path, 'cells/' + cell_id + '/native-f32.wav')
        with zipfile.ZipFile(archive) as stream:
            require(stream.namelist() == ['cells/' + n + '/native-f32.wav' for n in selected] and
                stream.testzip() is None, 'EXPORT_ZIP_ROSTER_OR_CRC')
            for cell_id in selected:
                record = rows[cell_id]['record']
                require(hashlib.sha256(stream.read('cells/' + cell_id + '/native-f32.wav')).hexdigest() ==
                    record['files'][record['native_wav']]['sha256'], 'EXPORT_ZIP_NATIVE_HASH')
        partitions.append({'partition_id': name, 'archive': pin(archive), 'cell_ids': selected})
    result = {'schema': 'active-tts-accounted-asr-partitions-v2', 'checkpoint': pin(args.checkpoint),
        'plan_canonical_sha256': value['plan_canonical_sha256'], 'prefix_count': value['prefix_count'],
        'partitions': partitions, 'new_eligible_cell_ids': ids, 'reused_cell_ids': reused,
        'excluded_native_rows': [row for row in value['excluded_native_rows'] if row['cell_id'] not in reused],
        'unobservable_consumed_ids': [row['cell_id'] for row in value['unobservable_consumed']],
        'asr_calls_executed': 0, 'training_admitted': False}
    once(output / 'index.json', result)
    return {'index': pin(output / 'index.json'), 'partitions': partitions}


def external_asr_queue(d, proof):
    root = Path(d['asr232_control']) / 'queue'
    queue = read(root / 'queue.json')
    if queue['source_sha256'] == q.source_identity():
        _, queue, request, queue_sha = q.load_queue(root)
        states = {job['job_id']: q.result_for(root, job, queue_sha, check_active=False) for job in request['jobs']}
    else:
        old_sources = proof['old_source_files']
        require(queue['source_sha256'] == {name: old_sources['tools/' + name]['sha256']
            for name in ('research_queue.py', 'research_job_lock.py')}, 'ASR_CONTROLLER_SOURCE_NOT_PINNED')
        request_body = {'action': 'queue', 'source_root': proof['old_source_root'],
            'source_files': old_sources, 'queue_root': str(root)}
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_tts_partial.py')],
            input=q.canonical(request_body), capture_output=True, timeout=30, check=False, start_new_session=False)
        require(result.returncode == 0, 'ASR_OLD_CONTROLLER_VERIFICATION_FAILED:' + result.stderr.decode()[-2000:])
        value = q.loads(result.stdout)
        require(value['pgid'] == os.getpgrp() and value['ml_imports'] == 0, 'ASR_CONTROLLER_HELPER_SCOPE')
        request, states = value['request'], value['jobs']
        require(queue == value['queue'], 'ASR_QUEUE_CHANGED_DURING_READ')
    require(queue['heavy_lock'] == d['heavy_lock'] and queue['automatic_retries'] == 0 and
        queue['historical_jobs_imported'] == 0 and len(states) == len(request['jobs']) and
        all(row['status'] == 'completed' and row['exit_code'] == 0 and row['actual_duration_known'] is True
            for row in states.values()), 'ASR232_REAL_COMPLETED_QUEUE_REQUIRED')
    before = 0
    for job in request['jobs']:
        folder = root / 'jobs' / job['job_id']; claim = read(folder / 'claim.json')
        dependencies = {name: {'receipt': str(root / 'jobs' / name / 'result.json'),
            'receipt_sha256': states[name]['receipt_sha256'], 'outputs_root': str(root / 'jobs' / name / 'outputs'),
            'outputs': states[name]['outputs']} for name in job['depends_on']}
        require(claim['automatic_retry'] is False and claim['global_actual_before_nanoseconds'] == before and
            0 < claim['allowed_runner_nanoseconds'] <= min(q.minutes_ns(job['timeout_minutes']),
                q.minutes_ns(request['global_budget_minutes']) - before) and
            claim['deadline_monotonic_ns'] == claim['started_monotonic_ns'] + claim['allowed_runner_nanoseconds'] and
            read(folder / 'dependencies.json') == dependencies and
            claim['dependencies_sha256'] == q.sha(q.canonical(dependencies)), 'ASR_QUEUE_ACTUAL_CLAIM_OR_BUDGET')
        before += states[job['job_id']]['actual_elapsed_ns']
    return queue, request, states


def handoff_evidence(d, proof, checkpoint_pin, verifier):
    cp = verify_checkpoint(checkpoint_pin['path'], checkpoint_pin['sha256'])
    require(cp['prefix_count'] == 232 and cp['declaration'] == pin(Path(d['prepared']) / 'declaration.json'),
        'ASR_HANDOFF_CHECKPOINT232_REQUIRED')
    root = Path(d['asr232_control'])
    sealed_pin = pin(root / 'sealed-stage.json'); prior_pin = pin(root / 'accounted-prior.json')
    sealed = pinned(sealed_pin); prior = pinned(prior_pin)
    verify_late_source(d, verifier, sealed['views'])
    verified = data_helper(verifier, sealed_pin)
    require(sealed['schema'] == 'd1-accounted-selective-stage-closed-v3' and
        sealed['accounted_prefix_count'] == 232 and sealed['continuation_checkpoint'] == checkpoint_pin and
        sealed['accounted_prior_evidence'] == prior_pin and sealed['queue_root'] == str(root / 'queue') and
        sealed['full_replay_completed'] is True and sealed['new_model_calls_in_sealing'] == 0 and
        sealed['unknown_asr_requests'] == 0 and sealed['accounted_complete'] is True and
        sealed['complete_terminal_coverage'] is False, 'ASR232_SEALED_SCOPE')
    require(prior['schema'] == 'd1-accounted-prior-evidence-v2' and prior['accounted_prefix_count'] == 232 and
        prior['plan_sha256'] == cp['plan_canonical_sha256'] and prior['plan_file_sha256'] == cp['plan']['sha256'] and
        prior['parent_evidence'] == d['prior_evidence'] and prior['continuation_checkpoint'] == checkpoint_pin and
        prior['unobservable_consumed'] == cp['unobservable_consumed'] and
        prior['excluded_returned'] == cp['excluded_native_rows'], 'ASR232_ACCOUNTED_PRIOR_SCOPE')
    prior_views = pinned(d['prior_evidence'])['views']
    require(prior['views'][:len(prior_views)] == prior_views and
        prior['views'][len(prior_views):] == [{'path': entry['path'], 'manifest_sha256': entry['manifest_sha256']}
            for entry in sealed['views']], 'ASR232_PRIOR_VIEW_PINS_CHANGED')
    observed = []
    for entry in prior['views']:
        package = Path(entry['path']); manifest = read(package / 'MANIFEST.json')
        require(c.sha_file(package / 'MANIFEST.json') == entry['manifest_sha256'] and
            c.sha_file(package / 'report.json') == manifest['report.json'], 'ASR232_VIEW_BYTES_CHANGED')
        report = read(package / 'report.json')
        require(report['source_adapter']['plan_sha256'] == cp['plan_canonical_sha256'], 'ASR232_VIEW_PLAN')
        for row in report['items']:
            require(row['source']['source_plan_sha256'] == cp['plan_canonical_sha256'], 'ASR232_SOURCE_PLAN')
            observed.append(row['source']['source_id'])
    expected_ids = set(cp['prior_cell_ids']) | set(cp['eligible_cell_ids'])
    require(len(observed) == len(set(observed)) == prior['observed_source_count'] ==
        sealed['observed_prefix_count'] and set(observed) == expected_ids and
        sealed['new_observed_audio'] == len(cp['eligible_cell_ids']), 'ASR232_OBSERVABLE_UNION')
    queue, request, states = external_asr_queue(d, proof)
    require(sealed['queue_state']['status'] == 'completed' and sealed['queue_state']['jobs'] == states,
        'ASR232_TERMINAL_SNAPSHOT_CHANGED')
    require({entry['job_id'] for entry in sealed['views']} == set(states) and all(
        set(entry['consumers']) == {'rnn-four-token', 'fsmn-fulltext'} and all(
            set(counts) == {'train', 'development'} and all(type(n) is int and n >= 0 for n in counts.values())
            for counts in entry['consumers'].values()) for entry in sealed['views']), 'ASR232_FOUR_LOADERS_REQUIRED')
    return {'schema': 'active-tts-asr232-handoff-v2', 'declaration': pin(Path(d['prepared']) / 'declaration.json'),
        'checkpoint': checkpoint_pin, 'sealed_stage': sealed_pin, 'accounted_prior': prior_pin,
        'queue_files': {name: pin(root / 'queue' / name) for name in ('queue.json', 'request.json', 'active-plan.json')},
        'queue_source_identity': queue['source_sha256'],
        'terminals': {name: pin(root / 'queue/jobs' / name / 'result.json') for name in states},
        'verifier_source': verifier, 'data_verification': verified,
        'unknown_remains_unobservable': True, 'new_model_calls_during_gate': 0}


def verify_handoff(d, proof):
    value = read(d['asr232_gate_path'])
    require(value == handoff_evidence(d, proof, value['checkpoint'], value['verifier_source']), 'ASR232_HANDOFF_CHANGED')
    return value


def seal_handoff(args):
    d, proof, _, _, _, _ = load_declaration(args.declaration)
    cp = pin(args.checkpoint)
    require(cp['sha256'] == args.checkpoint_sha256, 'ASR232_HANDOFF_CHECKPOINT_PIN')
    states = queue_state(d, required=True)
    require(all(states['continuation-' + str(index).zfill(3)]['status'] == 'pending' for index in (4, 5)),
        'HANDOFF_MUST_PRECEDE_LAST_TWO_JOB_CLAIMS')
    # The source does not exist at TTS declaration time. It is taken only from
    # this declaration's actual ASR control snapshot after real completion,
    # then matched to every new weak-view producer. There is no live fallback.
    external_asr_queue(d, proof)
    source = Path(d['asr232_verifier_contract']['control_source_root'])
    destination = Path(d['asr232_verifier_contract']['retained_source_root'])
    files = inventory(source)
    require(set(files) == set(DATA_SOURCES), 'ASR232_CONTROL_SEVEN_SOURCES_REQUIRED')
    for name in DATA_SOURCES:
        target = destination / name
        if target.exists():
            require(q.file_record(target) == files[name], 'ASR232_PARTIAL_SOURCE_FREEZE_CHANGED')
        else:
            e.atomic_bytes(target, (source / name).read_bytes())
    verifier = {'source_root': str(destination), 'control_source_root': str(source), 'files': files,
        'local_commit': args.data_source_revision, 'public_commit': args.data_public_commit,
        'git_tree': args.data_source_tree}
    value = handoff_evidence(d, proof, cp, verifier)
    once(d['asr232_gate_path'], value)
    return {'handoff': pin(d['asr232_gate_path']), 'new_model_calls': 0}


def create_backend(d, parent, prepared, plan, bulk, output, job):
    e.verify_runtime(prepared)
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    bulk.asset_lock(Path(parent['runtime']) / 'models/base', read(prepared / 'model-assets.json'), plan['model_revision'])
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    delegate = bulk.QwenBulkClone(Path(parent['runtime']) / 'models/base', device='cpu')
    before = time.monotonic(); model = delegate._model()
    require(model.model.tts_model_type == 'base', 'BASE_MODEL_REQUIRED')
    once(output / 'model-loaded.json', {'status': 'loaded', 'seconds': time.monotonic() - before,
        'parameter_dtype': str(next(model.model.parameters()).dtype),
        'parameter_device': str(next(model.model.parameters()).device),
        'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads()})
    backend = c.BoundedBackend(delegate, output,
        {'selected_cells': job['cells'], 'max_calls': len(job['cells']), 'generation': d['binding']['generation']},
        native_writer=e.retain_extension_native)
    return delegate, backend


def generate_callback(d, job, prepared, bulk, output, backend, deadline_ns, started):
    def generate(cell, ref):
        require(time.monotonic() - started < 600 and time.monotonic_ns() < deadline_ns, 'JOB_SOFT_DEADLINE')
        folder = output / 'generation/cells' / cell['cell_id']
        once(folder / 'started.json', e.expected_start(cell, ref, d['binding']['plan_canonical_sha256']))
        audio, rate = backend.synthesize(cell, ref, prepared / 'reference-assets')
        raw = read(output / 'native' / cell['cell_id'] / 'native.json')
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
            files.update({p.relative_to(output).as_posix(): q.file_record(p) for p in location.rglob('*') if p.is_file()})
        return {'status': status, 'technical_failure_reason': reason, 'output_root': str(output),
            'files': files, 'native_wav': native_wav, 'generation_root': str(output / 'generation')}
    return generate


def run(args):
    job = read(args.recipe)
    require(job['schema'] == JOB_SCHEMA and c.sha_file(job['declaration']) == job['declaration_sha256'],
        'CONTINUATION_JOB_DECLARATION_CHANGED')
    d, proof, parent, prepared, plan, bulk = load_declaration(job['declaration'])
    require(type(job['chunk_index']) is int and 0 <= job['chunk_index'] < 5 and
        job == expected_job(d, job['chunk_index']), 'CONTINUATION_SELECTED_JOB_CHANGED')
    output = q.safe_path(args.output)
    require(output == Path(d['queue_root']) / 'jobs' / job['job_id'] / 'outputs' and
        q.safe_path(args.heavy_lock) == Path(d['heavy_lock']) and
        os.environ.get('KWS_RESEARCH_RECIPE_SHA256') == c.sha_file(args.recipe), 'CONTINUATION_RUN_SCOPE')
    state = queue_state(d, required=True)[job['job_id']]
    if (output / 'invocation.json').exists():
        require((output / 'summary.json').is_file() and state['status'] == 'completed',
            'CONSUMED_INVOCATION_NO_RESTART')
        verify_outputs(d, job, plan)
        return {'status': 'completed-cache-reused', 'new_model_calls': 0}
    require(state['status'] == 'claimed-without-terminal' and
        state['claim']['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
        state['claim']['deadline_monotonic_ns'] > time.monotonic_ns(), 'LIVE_CONTINUATION_QUEUE_REQUIRED')
    if job['requires_asr232_handoff']:
        verify_handoff(d, proof)
    cells = {row['cell_id']: row for row in plan['cells']}; refs = {row['reference_id']: row for row in plan['references']}
    with heavy_model_slot(d['heavy_lock']) as lease, e.ledger_lock(d['ledger']):
        require(lease.supervised, 'SUPERVISED_CONTINUATION_REQUIRED')
        verify_registration(d)
        for name in job['cells']:
            cell = cells[name]
            require(e.read_cell(d['ledger'], name, e.cell_identity(d['binding'], cell,
                refs[cell['reference_id']])) is None, 'ONLY_NEVER_CLAIMED_CELLS_CAN_RUN')
        started = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
        once(output / 'invocation.json', {'job_recipe_sha256': c.sha_file(args.recipe),
            'declaration_sha256': job['declaration_sha256'], 'registration': pin(d['registration_path']),
            'selected_cells': job['cells'], 'inherited': lease.inherited, 'supervised': lease.supervised,
            'process_group': os.getpgrp(), 'calls_allowed': len(job['cells'])})
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '4'
        require(not any(name in sys.modules for name in ML_MODULES), 'ML_IMPORT_BEFORE_NETWORK_GUARD')
        c.research_no_network.set_offline_environment()
        once(output / 'network-guard.json', c.research_no_network.install())
        once(output / 'generation/plan-lock.json', {'schema': 'kws-bulk-clone-execution-lock-v1',
            'plan_sha256': d['binding']['plan_canonical_sha256'], 'training_admitted': False, 'model_id': plan['model_id']})
        delegate, backend = create_backend(d, parent, prepared, plan, bulk, output, job)
        callback = generate_callback(d, job, prepared, bulk, output, backend,
            state['claim']['deadline_monotonic_ns'], started)
        for name in job['cells']:
            cell = cells[name]
            _, invoked = e.consume_cell(d['ledger'], cell, refs[cell['reference_id']], d['binding'],
                {'stage_sha256': job['declaration_sha256'], 'job_id': job['job_id'],
                    'job_recipe_sha256': c.sha_file(args.recipe)}, callback)
            require(invoked, 'CONTINUATION_CACHED_ID_FORBIDDEN_IN_NEW_ROSTER')
        require(backend.calls == len(job['cells']), 'CONTINUATION_ACTUAL_CALL_COUNT')
        once(output / 'outcomes.json', expected_outcomes(d, job, plan))
        end = resource.getrusage(resource.RUSAGE_SELF)
        once(output / 'inventory.json', {'files': inventory(output)})
        summary = {'schema': 'active-tts-continuation-result-v2', 'status': 'completed',
            'job_recipe_sha256': c.sha_file(args.recipe), 'declaration_sha256': job['declaration_sha256'],
            'new_generation_calls': backend.calls, 'new_call_ids': job['cells'], 'reused_cell_ids': [],
            'prompt_creations': delegate.prompt_creations, 'wall_seconds': time.monotonic() - started,
            'user_cpu_seconds': end.ru_utime - usage.ru_utime, 'system_cpu_seconds': end.ru_stime - usage.ru_stime,
            'peak_rss_kib': end.ru_maxrss, 'asr_calls': 0, 'training_admitted': False, 'qualification_allowed': False,
            'inventory': q.file_record(output / 'inventory.json'), 'outcomes': q.file_record(output / 'outcomes.json')}
        once(output / 'summary.json', summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('protocol', 'protocol-sha256', 'prior-evidence', 'prior-sha256', 'continuation-id', 'output',
                 'asr232-control'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('inspect'); p.add_argument('--declaration', required=True)
    p = sub.add_parser('run')
    for name in ('recipe', 'output', 'heavy-lock'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('checkpoint')
    p.add_argument('--declaration', required=True); p.add_argument('--output', required=True)
    p.add_argument('--prefix-count', type=int, choices=(232, 256), required=True)
    p.add_argument('--previous-checkpoint')
    p = sub.add_parser('verify-checkpoint')
    p.add_argument('--checkpoint', required=True); p.add_argument('--checkpoint-sha256', required=True)
    p = sub.add_parser('export')
    for name in ('checkpoint', 'checkpoint-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('seal-asr-gate')
    for name in ('declaration', 'checkpoint', 'checkpoint-sha256', 'data-source-revision',
                 'data-public-commit', 'data-source-tree'):
        p.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args)
    elif args.command == 'inspect':
        result = inspect(args.declaration)
    elif args.command == 'run':
        result = run(args)
    elif args.command == 'checkpoint':
        result = checkpoint(args)
    elif args.command == 'export':
        result = export(args)
    elif args.command == 'seal-asr-gate':
        result = seal_handoff(args)
    else:
        value = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)
        result = {'schema': 'active-tts-accounted-verification-v2', 'checkpoint': pin(args.checkpoint),
            'accounted_ids': value['accounted_cell_ids'], 'returned_ids': [r['cell_id'] for r in value['returned_rows']],
            'eligible_ids': value['eligible_cell_ids'], 'excluded_ids': [r['cell_id'] for r in value['excluded_native_rows']],
            'unknown_ids': [r['cell_id'] for r in value['unobservable_consumed']], 'source_files': source_records(),
            'new_model_calls': 0, 'ml_imports': 0}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
