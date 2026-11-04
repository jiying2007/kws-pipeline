"""Fresh four-call public-preset research campaign; no references or cloning."""
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
CAMPAIGN = 'qwen-nonclone-pilot-20261104-v1'
BRANCH = 'research/' + CAMPAIGN
WORKFLOW = '.github/workflows/' + CAMPAIGN + '.yml'
ENGINE = 'qwen06'
MAX_CALLS = 4
WALL_SECONDS = 2400
PHRASES = (('你好小窝', 'positive-intent', 1),
           ('小窝小窝', 'positive-intent', 2),
           ('你好小屋', 'near-miss-intent', None),
           ('今天晚上吃什么', 'ordinary-negative-intent', None))


def require(value, message):
    if not value:
        raise ValueError(message)


def plan():
    return [dict(cell_id=CAMPAIGN+'-%02d' % i, text=text,
                 intended_kind=kind, intended_keyword_id=keyword,
                 speaker='Vivian', seed=20261104+i, attempts=1,
                 label_authority='requested_text_only', training_admitted=False)
            for i, (text, kind, keyword) in enumerate(PHRASES)]


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
    lock = json.loads((root / 'research/qwen-nonclone-pilot-20261104/source-lock.json').read_text())
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


def reconcile_terminal(output, host_status):
    """Reconstruct the fixed denominator from claims after container cleanup."""
    generation = output/'generation.json'
    raw = {}
    if generation.exists():
        require(not generation.is_symlink() and generation.stat().st_size <= 1024**2,
                'GENERATION_RECEIPT_BUDGET')
        raw = json.loads(generation.read_text())
        require(raw.get('campaign') == CAMPAIGN and raw.get('planned') == plan(),
                'GENERATION_PLAN_CHANGED')
    outcomes = raw.get('outcomes', [])
    require(type(outcomes) is list and len(outcomes) <= MAX_CALLS, 'OUTCOME_COUNT')
    indexed = {}
    for row in outcomes:
        require(type(row) is dict and row.get('cell_id') not in indexed, 'OUTCOME_IDENTITY')
        indexed[row['cell_id']] = row
    require(set(indexed) <= {c['cell_id'] for c in plan()}, 'UNKNOWN_OUTCOME')
    terminal = []
    for cell in plan():
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
            'outcomes':terminal,'planned_cells':MAX_CALLS,
            'claimed_attempts':sum(r['attempt_claimed'] for r in terminal),
            'generated_files_verified':sum(r['status'].startswith('generated_') for r in terminal),
            'all_four_technically_valid':all(r['status']=='generated_technically_valid' for r in terminal),
            'training_admitted':False,'human_gold':False,
            'runtime_verification_summary_present':(output/'runtime-verification.json').is_file()}
    (output/'terminal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def export_public(work):
    """Only this synthetic preset campaign's bounded audio/receipts, no logs/weights."""
    dest = work / 'artifact'
    dest.mkdir(exist_ok=False)
    candidates = list(work.glob('*.json')) + list((work/'output').glob('*'))
    files = {}
    for source in candidates:
        if source.suffix not in ('.json', '.wav', '.claim'):
            continue
        require(source.is_file() and not source.is_symlink() and source.stat().st_nlink == 1,
                'PUBLIC_FILE_TYPE')
        require(source.stat().st_size <= 8*1024**2, 'PUBLIC_FILE_SIZE')
        relative = source.relative_to(work)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    require(sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) <= 32*1024**2,
            'PUBLIC_ARTIFACT_BUDGET')
    (dest/'artifact-index.json').write_text(json.dumps({
        'campaign': CAMPAIGN, 'files': files, 'source_kind': 'synthetic-public-preset',
        'training_admitted': False, 'human_gold': False}, indent=2)+'\n')


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
    output, runtime = work/'output', work/'runtime'
    output.mkdir(); runtime.mkdir()
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
        for stage in ('setup', 'tts'):
            retained.host_capacity()
            name = CAMPAIGN+'-'+env['GITHUB_RUN_ID']+'-'+stage
            args = retained.container_args(name, image_lock['image'], code, runtime, output, 'tts', stage)
            pos = args.index(image_lock['image'])
            args[pos:pos] = ['--env', 'TQP_ENGINE=qwen06']
            pos = args.index(image_lock['image'])+1
            if stage == 'setup':
                args[pos:] = ['python', '-I', '-B', '/code/research/tts-quality-pilot-v1/setup_adapter.py',
                              '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
            else:
                args[pos:] = ['/runtime/venv/bin/python', '-I', '-B',
                              '/code/research/qwen-nonclone-pilot-20261104/worker.py']
            receipt[stage] = retained.supervise_container(args, name, output, stage,
                min(deadline, time.monotonic() + (1200 if stage == 'setup' else 1500)))
        receipt['status'] = 'generated_pending_independent_screen'
    except BaseException as exc:
        receipt.update(status='stopped_no_retry', error_type=type(exc).__name__)
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic()-started
        retained.atomic(work/'host-receipt.json', receipt)
        try:
            reconcile_terminal(output, receipt['status'])
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
