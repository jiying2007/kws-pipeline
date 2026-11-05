#!/usr/bin/env python3
"""Verify a frozen TTS timeout in its original, isolated stdlib environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

OLD_SOURCES = ('tools/active_tts_expand.py', 'tools/active_tts_smoke.py',
    'tools/research_job_lock.py', 'tools/research_no_network.py',
    'tools/research_runtime.py', 'tools/research_queue.py')
ML_MODULES = ('numpy', 'torch', 'onnxruntime', 'qwen_tts', 'soundfile', 'scipy')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    body = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    require(len(body) <= 2 * 1024 * 1024, 'PARTIAL_REQUEST_BOUND')
    request = json.loads(body)
    source = Path(request['source_root']).absolute()
    require(not any(p.is_symlink() for p in (source, *source.parents)), 'PARTIAL_SOURCE_ALIAS')
    require(set(request['source_files']) == set(OLD_SOURCES), 'PARTIAL_SOURCE_ROSTER')
    for name, expected in request['source_files'].items():
        path = source / name
        require(not any(p.is_symlink() for p in (path, *path.parents)) and
            path.stat().st_size == expected['bytes'] and
            hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], 'PARTIAL_SOURCE_CHANGED')
    sys.path[:0] = [str(source / 'tools'), str(source)]
    import active_tts_expand as e
    import research_queue as q
    c = e.common
    require(Path(e.__file__).resolve() == source / OLD_SOURCES[0] and
        Path(q.__file__).resolve() == source / OLD_SOURCES[-1], 'PARTIAL_IMPORT_SOURCE')
    if request.get('action') == 'queue':
        root, queue, definition, queue_sha = q.load_queue(request['queue_root'])
        states = {job['job_id']: q.result_for(root, job, queue_sha, check_active=False)
            for job in definition['jobs']}
        require(not any(name in sys.modules for name in ML_MODULES), 'PARTIAL_ML_IMPORT')
        print(json.dumps({'queue': queue, 'request': definition, 'jobs': states,
            'pid': os.getpid(), 'pgid': os.getpgrp(), 'ml_imports': 0},
            ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
        return 0

    def pin(path):
        path = q.safe_path(path)
        return {'path': str(path), **q.file_record(path)}

    def pinned(value):
        require(set(value) == {'path', 'bytes', 'sha256'} and pin(value['path']) == value, 'PARTIAL_PIN_CHANGED')
        return q.loads(q.read_bytes(value['path'], 32 * 1024 * 1024))

    def inventory(root):
        root = q.safe_path(root)
        return {p.relative_to(root).as_posix(): q.file_record(p)
            for p in sorted(root.rglob('*')) if p.is_file()}

    protocol = pinned(request['protocol'])
    require(protocol['schema'] == 'kws-d1-post-timeout-continuation-predeclaration-v1' and
        protocol['new_generation_maximum_calls'] == 58 and protocol['pause_after_jobs'] == 3,
        'PARTIAL_PROTOCOL_SCOPE')
    old_stage = protocol['old_tts_stage']; pinned(old_stage)
    stage, parent, prepared, plan, _ = e.load_stage(old_stage['path'])
    require(stage['source_revision'] == '43ffea59304121691a84424236543c00a56de588' and
        stage['target_count'] == 232 and len(stage['initial_reused_cells']) == 176 and
        stage['execution_source'] == {name: request['source_files'][name] for name in e.OWN_SOURCES},
        'PARTIAL_ORIGINAL_STAGE_SCOPE')
    ids = [row['cell_id'] for row in plan['cells']]
    require(len(ids) == len(set(ids)) == 256 and set(stage['initial_reused_cells']) == set(ids[:176]) and
        stage['chunks'] == [ids[176:204], ids[204:232]] and
        protocol['never_claimed_ids'] == ids[198:] and protocol['consumed_unknown_ids'] == [ids[197]],
        'PARTIAL_ORIGINAL_ROSTER')
    root, _, definitions, queue_sha = q.load_queue(stage['queue_root'])
    require([j['job_id'] for j in definitions['jobs']] == ['chunk-001', 'chunk-002'], 'PARTIAL_QUEUE_ROSTER')
    states = [q.result_for(root, job, queue_sha, check_active=False) for job in definitions['jobs']]
    _, original_states = e.queue_state(stage, require_created=True)
    terminal = protocol['old_queue_terminal']; pinned(terminal)
    require(terminal['path'] == str(root / 'jobs/chunk-001/result.json') and
        states[0]['status'] == original_states['chunk-001']['status'] == 'timed_out' and
        states[0]['exit_code'] == -9 and states[0]['actual_duration_known'] is True and
        states[0]['actual_elapsed_ns'] == 600_186_647_856 and
        states[0]['receipt_sha256'] == terminal['sha256'] and states[1] == {'status': 'pending'},
        'PARTIAL_TRUE_TIMEOUT_REQUIRED')
    audit = pinned(protocol['partial_audit'])
    require(audit['schema'] == 'active-tts-timed-out-partial-independent-readback-v1' and
        audit['stage'] == old_stage and audit['new_terminal_cells'] == 21 and
        audit['unknown_cell_ids'] == [ids[197]] and audit['native_arrays_read_back'] == 21 and
        audit['derived_pcm_read_back'] == 18 and audit['model_calls_during_audit'] == 0 and
        inventory(Path(stage['prepared'])) == audit['original_stage_files'], 'PARTIAL_ORIGINAL_FILES_CHANGED')
    ledger = Path(stage['ledger']); refs = {r['reference_id']: r for r in plan['references']}
    prior = []; returned = []
    for index, cell in enumerate(plan['cells'][:197]):
        name = cell['cell_id']; ref = refs[cell['reference_id']]
        value = e.read_cell(ledger, name, e.cell_identity(stage['binding'], cell, ref))
        require(value is not None, 'PARTIAL_PREVIOUS_RETURN_MISSING')
        rp = pin(ledger / 'cells' / (name + '.json'))
        require(rp['sha256'] == hashlib.sha256(c.canonical(value) + b'\n').hexdigest(), 'PARTIAL_RECORD_CANONICAL')
        if index < 176:
            prior.append({'cell_id': name, 'record_pin': rp})
            continue
        cp = pin(ledger / 'claims' / (name + '.json')); claim = pinned(cp)
        expected_owner = {'stage_sha256': old_stage['sha256'], 'job_id': 'chunk-001',
            'job_recipe_sha256': definitions['jobs'][0]['recipe_sha256']}
        require(claim['owner'] == expected_owner and value['claim_sha256'] == cp['sha256'], 'PARTIAL_RETURN_OWNER')
        output = Path(value['output_root'])
        require(output == root / 'jobs/chunk-001/outputs', 'PARTIAL_RETURN_OUTPUT')
        call = c.read(output / 'native' / name / 'call.json')
        mode = 'ICL' if ref['reference_text_status'] in ('verified', 'dataset-provided-content-checked') else 'x-vector-only'
        require(call['cell_id'] == name and call['source_family_id'] == cell['source_family_id'] and
            call['reference_id'] == ref['reference_id'] and call['reference_wav_sha256'] == ref['wav_sha256'] and
            call['reference_pcm_sha256'] == ref['pcm_sha256'] and call['seed'] == cell['seed'] and
            call['generation'] == stage['binding']['generation'] and call['intended_text'] == cell['prompt'] and
            call['reference_mode'] == mode and call['reference_text_passed'] == (mode == 'ICL') and call['attempt'] == 1 and
            c.read(output / 'generation/cells' / name / 'started.json') ==
            e.expected_start(cell, ref, stage['binding']['plan_canonical_sha256']), 'PARTIAL_CALL_BINDING')
        returned.append({'cell_id': name, 'origin': 'original-timed-out', 'owner_id': 'original-232-chunk-001',
            'ledger_record_sha256': rp['sha256'], 'record_pin': rp, 'claim_pin': cp,
            'record': value, 'claim': claim, 'eligibility': e.asr_eligibility(value)})
    cell = plan['cells'][197]; name = cell['cell_id']; ref = refs[cell['reference_id']]
    identity = e.cell_identity(stage['binding'], cell, ref)
    cp = pin(ledger / 'claims' / (name + '.json')); claim = pinned(cp)
    require(claim['schema'] == 'active-tts-plan-call-claim-v1' and claim['cell_id'] == name and
        claim['identity_sha256'] == identity and claim['attempt'] == 1 and claim['owner'] == expected_owner,
        'PARTIAL_UNKNOWN_CLAIM')
    output = root / 'jobs/chunk-001/outputs'
    artifacts = {p.relative_to(output).as_posix(): q.file_record(p)
        for folder in (output / 'native' / name, output / 'generation/cells' / name)
        for p in sorted(folder.rglob('*')) if p.is_file()}
    require(set(artifacts) == {'native/' + name + '/call.json', 'generation/cells/' + name + '/started.json'},
        'PARTIAL_UNKNOWN_HAS_RETURN_OR_EXTRA_ARTIFACT')
    call = c.read(output / 'native' / name / 'call.json')
    mode = 'ICL' if ref['reference_text_status'] in ('verified', 'dataset-provided-content-checked') else 'x-vector-only'
    require(call['schema'] == 'active-tts-generation-call-v1' and call['cell_id'] == name and
        call['reference_id'] == ref['reference_id'] and call['source_family_id'] == cell['source_family_id'] and
        call['seed'] == cell['seed'] and call['generation'] == stage['binding']['generation'] and
        call['intended_text'] == cell['prompt'] and call['reference_wav_sha256'] == ref['wav_sha256'] and
        call['reference_pcm_sha256'] == ref['pcm_sha256'] and call['reference_mode'] == mode and
        call['reference_text_passed'] == (mode == 'ICL') and call['attempt'] == 1 and
        c.read(output / 'generation/cells' / name / 'started.json') ==
        e.expected_start(cell, ref, stage['binding']['plan_canonical_sha256']), 'PARTIAL_UNKNOWN_CALL_BINDING')
    absent = [str(ledger / 'cells' / (name + '.json'))] + [str(output / 'native' / name / leaf)
        for leaf in ('returned-array.npy', 'returned-native.wav', 'native.json', 'returned.json')] + [
        str(output / 'generation/cells' / name / leaf) for leaf in
        ('completed.json', 'failed.json', 'native-f32.wav', 'audio-16k.wav')]
    require(all(not Path(path).exists() for path in absent), 'PARTIAL_UNKNOWN_BECAME_RETURNED')
    unknown = {'cell_id': name, 'origin': 'original-timed-out', 'status': 'unobservable-consumed',
        'owner_id': 'original-232-chunk-001', 'identity_sha256': identity, 'claim_pin': cp, 'claim': claim,
        'output_root': str(output), 'artifacts': artifacts, 'absent_paths': absent,
        'model_return_known': False, 'may_retry': False, 'asr_calls': 0, 'native_system_calls': 0,
        'observed_text': None, 'training_admitted': False}
    job_root = root / 'jobs/chunk-001'
    owner = {'origin': 'original-timed-out', 'job_id': 'chunk-001', 'stage': old_stage,
        'job_recipe': pin(root / 'recipes/chunk-001.json'), 'queue_root': str(root),
        'queue_files': {leaf: pin(root / leaf) for leaf in ('queue.json', 'request.json', 'active-plan.json')},
        'job_root': str(job_root), 'job_files': inventory(job_root), 'result': terminal,
        'outputs_root': str(output), 'source_root': str(source), 'source_files': request['source_files']}
    proof = {'schema': 'active-tts-original-partial-proof-v2', 'protocol_declaration': request['protocol'],
        'old_stage': old_stage, 'old_terminal': terminal, 'prior_returned': prior,
        'returned_rows': returned, 'unobservable_consumed': [unknown], 'owner': owner,
        'never_claimed_ids_at_registration': ids[198:], 'binding': stage['binding'],
        'plan': pin(prepared / 'plan.json'), 'ledger': str(ledger),
        'old_source_root': str(source), 'old_source_files': request['source_files']}
    if request['mode'] == 'freeze':
        for cell in plan['cells'][198:]:
            require(e.read_cell(ledger, cell['cell_id'], e.cell_identity(stage['binding'], cell,
                refs[cell['reference_id']])) is None, 'PARTIAL_NEW_ROSTER_ALREADY_CONSUMED')
    else:
        require(request['mode'] == 'verify' and pinned(request['proof']) == proof, 'PARTIAL_FROZEN_PROOF_CHANGED')
    require(not any(name in sys.modules for name in ML_MODULES), 'PARTIAL_ML_IMPORT')
    print(json.dumps({'proof': proof, 'pid': os.getpid(), 'pgid': os.getpgrp(), 'ml_imports': 0},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
