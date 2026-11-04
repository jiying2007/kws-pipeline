"""Fresh 64-call public-preset research campaign; no references or cloning."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = 'qwen-nonclone64-20261104-v1'
BRANCH = 'research/' + CAMPAIGN
WORKFLOW = '.github/workflows/' + CAMPAIGN + '.yml'
ENGINE = 'qwen06'
MAX_CALLS = 64
WALL_SECONDS = 4800
VOICES = ('Vivian', 'Serena', 'Uncle_Fu', 'Dylan')
PHRASES = (('你好小窝', 'positive-intent', 1),
           ('小窝小窝', 'positive-intent', 2),
           ('你好小好', 'near-miss-intent', None),
           ('你好小你', 'near-miss-intent', None),
           ('小窝你好', 'near-miss-intent', None),
           ('小好小窝', 'near-miss-intent', None),
           ('你好你好', 'near-miss-intent', None),
           ('小小窝窝', 'near-miss-intent', None))
POLICY = {
    'schema': 'machine-weak-label-train-only-intent-v1',
    'use': 'controlled-noncommercial-research',
    'intended_split': 'train_only',
    'intended_label_tier': 'machine_weak_label',
    'requested_text_is_observed_transcript': False,
    'future_admission_requires': ['independent_audio_only_screen',
        'native_runtime_capture', 'independent_trust_root', 'oov_negative_coverage'],
    'training_admitted': False, 'human_gold': False,
    'evaluation_or_production_use': False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def plan():
    require(len(PHRASES) == 8 and len(VOICES) == 4, 'FIXED_FACTORS')
    positives = [text for text, kind, key in PHRASES if key is not None]
    for text, kind, key in PHRASES:
        require(set(text) <= set('你好小窝'), 'TRANSCRIPT_VOCABULARY')
        if key is None:
            require(not any(wake in text for wake in positives), 'NEGATIVE_CONTAINS_WAKE')
    return [dict(cell_id=CAMPAIGN+'-%02d' % i, global_ordinal=i,
                 text=text, intended_kind=kind, intended_keyword_id=keyword,
                 speaker=voice, lane=lane, repeat=repeat, seed=2026110400+i,
                 attempts=1, label_authority='requested_text_only',
                 intended_split='train_only', intended_label_tier='machine_weak_label',
                 training_admitted=False)
            for i, (lane, voice, text, kind, keyword, repeat) in enumerate(
                (lane, voice, text, kind, keyword, repeat)
                for lane, voice in enumerate(VOICES)
                for text, kind, keyword in PHRASES for repeat in range(2))]


def lane_plan(lane):
    require(type(lane) is int and 0 <= lane < 4, 'FIXED_LANE')
    return plan()[lane*16:(lane+1)*16]


def verify_event(env, event):
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline',
                'GITHUB_REF': 'refs/heads/'+BRANCH, 'GITHUB_EVENT_NAME': 'push',
                'GITHUB_RUN_NUMBER': '1', 'GITHUB_RUN_ATTEMPT': '1'}
    require(all(env.get(k) == v for k, v in expected.items()), 'FRESH_CAMPAIGN_IDENTITY')
    require(re.fullmatch('[0-9a-f]{40}', env.get('GITHUB_SHA', '')) is not None and
            re.fullmatch('[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')) is not None,
            'SOURCE_RUN_IDENTITY')
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'] and
            event.get('repository', {}).get('private') is False, 'FIRST_PUBLIC_BRANCH_ONLY')


def verify_sources(root):
    lock = json.loads((root / 'research/qwen-nonclone64-20261104/source-lock.json').read_text())
    require(lock.get('schema') == 'qwen-nonclone-source-freeze-v1' and
            lock.get('campaign') == CAMPAIGN, 'SOURCE_LOCK_SCHEMA')
    files = lock['files']
    require(type(files) is dict and 15 <= len(files) <= 40, 'SOURCE_LOCK_COUNT')
    for name, digest in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and
                name.startswith(('research/', '.github/workflows/')) and
                re.fullmatch('[0-9a-f]{64}', digest) is not None, 'SOURCE_LOCK_PATH')
        path = root / name
        require(path.is_file() and not path.is_symlink() and
                hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'SOURCE_BYTES_CHANGED')
    return lock


def staged_sources(root, destination, lock):
    destination.mkdir()
    for name in lock['files']:
        if not name.startswith('research/'):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    # The verified existing setup adapter accepts this fixed staged basename.
    old = destination / 'research/tts-quality-pilot-v1'
    shutil.copyfile(old / 'qwen06-model-lock.json', old / 'qwen-model-lock.json')


def reconcile_lane(output, host_status, lane):
    cells = lane_plan(lane)
    """Reconstruct the fixed denominator from claims after container cleanup."""
    generation = output/'generation.json'
    raw = {}
    if generation.exists():
        require(not generation.is_symlink() and generation.stat().st_size <= 1024**2,
                'GENERATION_RECEIPT_BUDGET')
        raw = json.loads(generation.read_text())
        require(raw.get('campaign') == CAMPAIGN and raw.get('planned') == cells and raw.get('lane') == lane,
                'GENERATION_PLAN_CHANGED')
    outcomes = raw.get('outcomes', [])
    require(type(outcomes) is list and len(outcomes) <= 16, 'OUTCOME_COUNT')
    indexed = {}
    for row in outcomes:
        require(type(row) is dict and row.get('cell_id') not in indexed, 'OUTCOME_IDENTITY')
        indexed[row['cell_id']] = row
    require(set(indexed) <= {c['cell_id'] for c in cells}, 'UNKNOWN_OUTCOME')
    allowed_claims = {cell['cell_id']+'.claim' for cell in cells}
    require({p.name for p in output.glob('*.claim')} <= allowed_claims, 'UNKNOWN_CLAIM')
    claimed_prefix = [i for i, cell in enumerate(cells) if (output/(cell['cell_id']+'.claim')).exists()]
    require(claimed_prefix == list(range(len(claimed_prefix))), 'NONPREFIX_CLAIMS')
    terminal = []
    for cell in cells:
        ident=cell['cell_id']; claim=output/(ident+'.claim'); row=indexed.get(ident)
        claimed=claim.exists()
        claim_complete=False
        if claimed:
            require(claim.is_file() and not claim.is_symlink() and claim.stat().st_nlink == 1 and
                    claim.stat().st_size <= 128, 'ATTEMPT_CLAIM_CHANGED')
            claim_complete=claim.read_text() == ident
        state='not_run'
        if claimed:
            state='failed_no_retry'
            if claim_complete and row and row.get('status') == 'generated':
                wav=output/(ident+'.raw-float.wav')
                require(wav.is_file() and not wav.is_symlink() and wav.stat().st_size <= 8*1024**2 and
                        hashlib.sha256(wav.read_bytes()).hexdigest() == row.get('raw_wav_sha256'),
                        'GENERATED_WAV_CHANGED')
                state=('generated_technically_valid' if row.get('metrics',{}).get('technically_valid') is True
                       else 'generated_technically_invalid')
        else:
            require(row is None, 'OUTCOME_WITHOUT_ATTEMPT_CLAIM')
        terminal.append({'cell_id':ident,'status':state,'attempt_claimed':claimed,
                         'training_admitted':False})
    result={'campaign':CAMPAIGN,'host_status':host_status,
            'worker_reported_generation_status':raw.get('status','unavailable'),
            'outcomes':terminal,'planned_cells':16,
            'claimed_attempts':sum(r['attempt_claimed'] for r in terminal),
            'generated_files_verified':sum(r['status'].startswith('generated_') for r in terminal),
            'all_lane_technically_valid':all(r['status']=='generated_technically_valid' for r in terminal),
            'training_admitted':False,'human_gold':False,
            'runtime_verification_summary_present':(output/'runtime-verification.json').is_file()}
    (output/'terminal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def reconcile_terminal(work, host_status):
    lanes = [reconcile_lane(work/('lane-%d' % lane)/'output', host_status, lane)
             for lane in range(4)]
    outcomes = [row for lane in lanes for row in lane['outcomes']]
    require(len(outcomes) == MAX_CALLS, 'FIXED_DENOMINATOR')
    result = dict(campaign=CAMPAIGN, host_status=host_status, planned_cells=MAX_CALLS,
        outcomes=outcomes, policy=POLICY,
        claimed_attempts=sum(row['attempt_claimed'] for row in outcomes),
        generated_files_verified=sum(row['status'].startswith('generated_') for row in outcomes),
        all_64_technically_valid=all(row['status']=='generated_technically_valid' for row in outcomes),
        training_admitted=False, human_gold=False)
    (work/'terminal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def public_names():
    names = {'host-receipt.json','image-transfer.json','terminal.json','policy.json',
             'container-setup.json'}
    for lane in range(4):
        prefix = 'lane-%d/' % lane
        names.add(prefix+'container-tts.json')
        for name in ('generation.json','terminal.json','runtime-verification.json',
                     'runtime-observations.json','worker-status.json','progress.json'):
            names.add(prefix+'output/'+name)
        for cell in lane_plan(lane):
            names.add(prefix+'output/'+cell['cell_id']+'.raw-float.wav')
    return names


def export_public(work):
    """Positive filename allowlist: synthetic WAV and known JSON, never claims/logs."""
    dest = work/'artifact'
    dest.mkdir(exist_ok=False)
    files = {}; total = 0
    for name in sorted(public_names()):
        source = work/name
        if not source.exists() and not source.is_symlink():
            continue
        require(all(not parent.is_symlink() for parent in source.parents if parent != work.parent),
                'PUBLIC_PARENT_ALIAS')
        require(source.is_file() and not source.is_symlink() and source.stat().st_nlink == 1,
                'PUBLIC_FILE_TYPE')
        maximum = 12*24000*4+4096 if source.suffix == '.wav' else 1024**2
        require(source.stat().st_size <= maximum, 'PUBLIC_FILE_SIZE')
        total += source.stat().st_size
        require(total <= 95*1024**2, 'PUBLIC_ARTIFACT_BUDGET')
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (dest/'artifact-index.json').write_text(json.dumps(dict(campaign=CAMPAIGN,files=files,
        source_kind='synthetic-public-preset',training_admitted=False,human_gold=False),indent=2)+'\n')
    require(sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) <= 96*1024**2,
            'PUBLIC_ARTIFACT_BUDGET')


def run(env):
    # Admission precedes imports of retained launch code and every external action.
    event_path = Path(env['GITHUB_EVENT_PATH'])
    require(event_path.stat().st_size <= 2*1024**2, 'EVENT_BUDGET')
    verify_event(env, json.loads(event_path.read_text()))
    lock = verify_sources(ROOT)
    sys.path.insert(0, str(ROOT/'research/source-screen32-v1'))
    import hosted_run as retained
    work = ROOT / ('nonclone-'+CAMPAIGN)
    work.mkdir(exist_ok=False)
    runtime = work/'runtime'
    runtime.mkdir()
    (work/'setup').mkdir()
    for lane in range(4):
        (work/('lane-%d' % lane)/'output').mkdir(parents=True)
    (work/'policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    receipt = {'campaign': CAMPAIGN, 'head_sha': env['GITHUB_SHA'],
        'run_id': env['GITHUB_RUN_ID'], 'status': 'started',
        'max_model_calls': MAX_CALLS, 'new_asr_calls': 0, 'new_training_runs': 0,
        'reference_voice_inputs': 0, 'retry': False,
        'training_admitted': False, 'human_gold': False,
        'source_lock_sha256': hashlib.sha256((HERE/'source-lock.json').read_bytes()).hexdigest()}
    try:
        require(shutil.disk_usage(work).free >= 16*1024**3, 'DISK_RESERVE')
        receipt['host_capacity'] = retained.host_capacity()
        image_lock = json.loads((ROOT/'research/source-screen32-v1/container-lock.json').read_text())
        retained.pull_image(image_lock, work/'image-transfer.json')
        code = work/'code'
        staged_sources(ROOT, code, lock)
        for lane in (-1, 0, 1, 2, 3):
            require(deadline-time.monotonic() >= 120, 'ABSOLUTE_DEADLINE_RESERVE')
            stage = 'setup' if lane == -1 else 'tts'
            output = work/'setup' if lane == -1 else work/('lane-%d' % lane)/'output'
            retained.host_capacity()
            require(deadline-time.monotonic() >= 120, 'ABSOLUTE_DEADLINE_RESERVE')
            name = CAMPAIGN+'-'+env['GITHUB_RUN_ID']+'-'+('setup' if lane == -1 else 'lane-%d' % lane)
            args = retained.container_args(name, image_lock['image'], code, runtime, output, 'tts', stage)
            pos = args.index(image_lock['image'])
            args[pos:pos] = ['--env', 'TQP_ENGINE=qwen06']
            pos = args.index(image_lock['image'])+1
            if stage == 'setup':
                args[pos:] = ['python', '-I', '-B', '/code/research/tts-quality-pilot-v1/setup_adapter.py',
                              '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
            else:
                args[pos:] = ['/runtime/venv/bin/python', '-I', '-B',
                              '/code/research/qwen-nonclone64-20261104/worker.py', str(lane)]
            require(deadline-time.monotonic() >= 120, 'ABSOLUTE_DEADLINE_RESERVE')
            receipt['setup' if lane == -1 else 'lane-%d' % lane] = retained.supervise_container(args, name, output, stage,
                min(deadline-60, time.monotonic()+1200) if stage == 'setup' else deadline-60)
        terminal = reconcile_terminal(work, 'generation_checks_pending')
        require(terminal['all_64_technically_valid'], 'INCOMPLETE_64_CELL_CAMPAIGN')
        receipt['status'] = 'generated_pending_independent_screen'
    except BaseException as exc:
        receipt.update(status='stopped_no_retry', error_type=type(exc).__name__)
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic()-started
        retained.atomic(work/'host-receipt.json', receipt)
        try:
            reconcile_terminal(work, receipt['status'])
        except BaseException as exc:
            receipt['terminal_reconciliation']='failed_no_retry'
            receipt['terminal_error_type']=type(exc).__name__
            retained.atomic(work/'host-receipt.json', receipt)
            raise
        finally:
            export_public(work)


if __name__ == '__main__':
    try:
        run(os.environ)
    except BaseException as exc:
        print(json.dumps({'campaign':CAMPAIGN,'status':'stopped_no_retry','error_type':type(exc).__name__}))
        raise SystemExit(1)
