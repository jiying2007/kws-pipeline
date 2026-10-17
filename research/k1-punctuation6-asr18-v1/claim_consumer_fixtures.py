"""Generated current-source owner fixtures, with fake API/kernel/model receipts.

These are synthetic tests, never actual campaign evidence. Every source, freeze,
release, control, Git object and response hash is computed from the bytes used.
"""
import copy
import json
from pathlib import Path

import campaign_contract as c
import evidence_transport as t
import prepare_freeze
from test_control_fixture import historical_source_tree
from test_bound_claim_contract import fixture as owner_fixture

ROOT = Path(__file__).resolve().parents[2]
NEW = 'research/k1-punctuation6-asr18-v1/'
STAMP = '2026-10-13T00:00:00Z'
NOTICE = 'Synthetic local fake API and native receipts; not actual inference evidence.'


def bound_public_fixture(*, base_files=None, run=None, scope=None):
    if base_files is None:
        from test_asr18_host import public_fixture
        files, run = public_fixture()
    else:
        c.require(type(base_files) is dict and type(run) is dict, 'explicit synthetic base files and run')
        files, run = copy.deepcopy(base_files), copy.deepcopy(run)
    release = t.decode(files['source/' + NEW + 'execution-release.json'])
    # An explicit scope is only for coherent negative fixtures; no live gate is bypassed.
    scope = c.current_scope() if scope is None else dict(scope)
    release.update(experiment=scope['campaign'],source_ref=scope['source_branch'],
        evidence_ref=scope['result_branch'],workflow=scope['workflow'],job_key=scope['job_key'])
    control = c.disabled_claim_control()
    control.update({key: True for key in c.CLAIM_APPROVALS})
    control.update(campaign=release['experiment'], source_ref=release['source_ref'],
        evidence_ref=release['evidence_ref'], workflow=release['workflow'],
        claim_ref='refs/heads/claims/' + release['experiment'])
    control_raw = json.dumps(control, indent=2).encode() + b'\n'
    release.update(schema=c.BOUND_RELEASE_SCHEMA, claim_admission_mode=c.CURRENT_CLAIM_MODE,
        claim_admission_control_sha256=c.digest(control_raw))
    release['prior_public_claim_history_sha256'] = c.digest(c.canonical(c.bound_prior_history_identity(control)))
    release_raw = json.dumps(release, indent=2).encode() + b'\n'
    with historical_source_tree(ROOT) as historical:
        freeze = prepare_freeze.build(historical)
        source_files = {name: files.get('source/' + name, c.read_bytes(historical / name)) for name in freeze['host_code']}
    publication=t.decode(source_files[NEW+'publication-release.json'])
    publication.update(enabled=True,root_publication_release=True,independent_source_review=True)
    source_files[NEW+'publication-release.json']=c.canonical(publication)
    source_files[NEW + 'execution-release.json'] = release_raw
    source_files[NEW + c.CLAIM_CONTROL_FILE] = control_raw
    for leaf in ('claim_admission.py',):
        name = NEW + leaf
        source_files[name] = c.read_bytes(ROOT / name)
        freeze['host_code'][name] = {}
    freeze['host_code'][NEW + c.CLAIM_CONTROL_FILE] = {}
    for role in ('host_code', 'setup_code', 'inference_code'):
        freeze[role] = {name: {'bytes': len(source_files[name]), 'sha256': c.digest(source_files[name])}
            for name in freeze[role]}
    freeze_raw = c.canonical(freeze)
    source_files[NEW + 'execution-freeze.json'] = freeze_raw
    rows = [{'path': name, 'mode': '100644', 'type': 'blob', 'sha': t.git_sha(raw)}
        for name, raw in sorted(source_files.items())]
    tree = t.tree_identity(rows)[0]
    person = 'Synthetic Fixture <fixture@example.invalid> 1791849600 +0000'
    commit_raw = ('tree ' + tree + '\nauthor ' + person + '\ncommitter ' + person +
        '\n\nSynthetic current public source closure; zero-model fixture only\n').encode()
    head = t.git_sha(commit_raw, 'commit')
    old_head = run['executed_head_sha']
    def rebind(value):
        if type(value) is dict:return {key:rebind(item) for key,item in value.items()}
        if type(value) is list:return [rebind(item) for item in value]
        return head if value == old_head else value
    run = rebind(run)
    files = {name: c.canonical(rebind(t.decode(raw))) for name,raw in files.items() if not name.startswith('source/')}
    files.update({'source/' + name:raw for name,raw in source_files.items()})
    files['authority/execution-freeze.json'] = freeze_raw
    proof = owner_fixture(count=2, release=release, release_raw=release_raw, control=control,
        control_raw=control_raw, run=run, source_tree=tree, source_timestamp=STAMP,
        source_freeze_sha256=c.digest(freeze_raw), numeric_job_id=456, workflow_id=9)
    history = proof['admission']['host_history']
    preflight = {'schema':'k1-asr18-run-preflight-v2','identity':run,'numeric_job_id':456,
        'workflow_id':9,'workflow':control['workflow'],'source_branch':control['source_ref'],
        'evidence_branch':control['evidence_ref'],'source_branch_workflow_runs':history['total_count'],
        'history':history,'complete_api_history':{'total_count':history['total_count'],'workflow_runs':history['rows']},
        'evidence_ref_absent':True,'prior_public_claim_history_sha256':release['prior_public_claim_history_sha256'],
        'public_standard_hosted_ubuntu24':True,'api_metadata_accounting':{
            'gets':6,'response_body_bytes':len(proof['history_raw']),'response_body_cap_bytes':c.MIB,
            'scope':'public_api_metadata_separate_from_model_audio_bodies','hard_wire_byte_cap':False}}
    files['run-preflight.json'] = c.canonical(preflight)
    files['admission.json'] = c.canonical(proof['admission'])
    host = t.decode(files['host-receipt.json']);host['executed_tree_sha']=tree;host['source_freeze_sha256']=c.digest(freeze_raw)
    files['host-receipt.json'] = c.canonical(host)
    return {'files':files,'identity':run,'source_tree':tree,'source_rows':rows,'source_commit_raw':commit_raw,
        'source_commit_timestamp':STAMP,'owner_fixture':proof,'synthetic_notice':NOTICE}


def execute_bound_fixture(fixture, env):
    """Use real issuer, durable private handoff and execute with fake source reads."""
    import os
    from unittest.mock import patch
    import hosted_run as h
    import claim_admission
    import test_claim_admission_host as handoff
    owner=fixture['owner_fixture'];folder=env['RUNNER_TEMP'];files=fixture['files']
    original_read=h.read_bytes
    def source_read(path, **kwargs):
        name=str(Path(path).relative_to(h.ROOT)) if Path(path).is_relative_to(h.ROOT) else None
        if name is not None and 'source/'+name in files:return files['source/'+name]
        return original_read(path,**kwargs)
    with patch.object(os,'getuid',new=os.geteuid),patch.object(claim_admission.secrets,'token_hex',return_value=c.digest(b'public-test-fixture-private-nonce')):
        with patch.dict(os.environ,handoff.environment(owner,folder,token=True),clear=True):
            ready,wire=handoff.issue(owner,folder)
        with patch.dict(os.environ,handoff.environment(owner,folder),clear=True):
            preflight,permit,public=handoff.consume(owner,folder)
        if ready is not True or len(wire.calls)!=13 or len(public.calls)!=6:raise AssertionError('real bounded fixture admission')
    with patch.object(h,'read_bytes',side_effect=source_read):
        return h.execute(env,fixture['identity'],owner['release'],t.decode(files['authority/execution-freeze.json']),
            owner['identity']['freeze_sha256'],fixture['source_tree'],preflight,
            admission_permit=permit,timestamp=fixture['source_commit_timestamp'])
