"""Third separately reviewed campaign; disarmed source is never an activation."""
from pathlib import Path
import json
import importlib.util
import os
import re
import sys
import subprocess
import time
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
from common import JOB_SHA, LIMITS, BOUNDARY, require, read_json, file_hash, identity, regular
import campaign_host
_spec = importlib.util.spec_from_file_location('third_campaign_freeze_builder', HERE / 'prepare_freeze.py')
freeze_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_builder)
EXPERIMENT = 'qwen16-whisper-small-third-attempt-3-v1'
BRANCH = 'research/' + EXPERIMENT
WORKFLOW = '.github/workflows/source-screen32-whisper-third-attempt-3-v1.yml'
PRIOR = (
    (38105471896, '2c01ce5e0dbbc79dfda5ee59d6cc734eec995db882ac22e9465cf99ef2f93504',
     'qwen16-whisper-small-third-once-v1', '48a8135fc00489075ffeb456bb51aea6afef3cdd',
     'fdfb8a9040a10ec380ac7a658662d2d2935296301ee4269ac0ce79309cc48c2e'),
    (38107706348, '7a0966e920cd703a948fe7e4828e133d5f09f36581fe1471fc4e93dd89e68a01',
     'qwen16-whisper-small-third-attempt-2-v1', '733fbb90a98b3d39242594cebea378f790bc5775',
     '543bc71585522dc318b2c954ee715b779b696876a5030b04f921a36214870f2a'))
INFERENCE_CODE = sorted({'research/source-screen32-whisper-v1/' + n for n in
    ('common.py', 'adapter.py', 'worker.py', 'setup_runtime.py', 'runtime-lock.json', 'source-lock.json')} |
    {'tools/asr_native_contracts.py', 'research/source-screen32-v1/runtime_scope.py'})
SETUP_CODE = sorted(set(INFERENCE_CODE) | {'research/source-screen32-whisper-v1/attempt-3/' + n for n in
    ('bounded_host.py', 'bounded_connect.py', 'pip_acquire.py', 'phased_setup.py')})


def verify_prior():
    job_path = ROOT / 'research/source-screen32-v1/evidence/run-38018029787/blind/job.json'
    require(file_hash(job_path) == JOB_SHA, 'fixed sixteen job')
    clips = read_json(job_path)['clips']; require(len(clips) == 16, 'sixteen clips')
    records = []
    for run, digest, experiment, head, source_freeze in PRIOR:
        directory = BASE / ('evidence/run-' + str(run))
        require(file_hash(directory / 'provenance.json') == digest, 'immutable prior provenance')
        p = read_json(directory / 'provenance.json')
        require(p['run_id'] == run and p['head_sha'] == head and p['source_freeze_sha256'] == source_freeze and
                p['run_attempt'] == p['run_number'] == 1 and p['attempts'] == 0 and p['not_run'] == 16,
                'prior source/run identity')
        for name, expected in p['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'prior path')
            require(identity(directory / name) == expected, 'immutable prior bytes')
        receipt = read_json(directory / 'host-receipt.json')
        require(receipt['experiment'] == experiment and receipt['head_sha'] == head and
                receipt['source_freeze_sha256'] == source_freeze and receipt['run_id'] == str(run) and
                receipt['status'] == 'stopped_no_retry' and receipt['attempts'] == receipt['completed'] == 0 and
                receipt['stages'] == {'setup': 'failed_no_retry', 'whisper': 'not_run'} and
                receipt['limits'] == LIMITS and receipt['terminal_validation_errors'] == 0, 'prior setup-only failure')
        rows = read_json(directory / 'terminal-outcomes.json')
        require(len(rows) == 16, 'prior denominator')
        for row, clip in zip(rows, clips):
            require(row == {'audio_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'],
                'attempts': 0, 'status': 'not_run', 'raw_output_sha256': None, 'raw_text': None,
                'stop_reason': None, 'validation_errors': [], **BOUNDARY} and
                type(row['attempts']) is int, 'prior zero audio attempts')
        records.append({'run_id': run, 'provenance_sha256': digest, 'head_sha': head,
                        'source_freeze_sha256': source_freeze, 'audio_attempts': 0})
    return {'schema': 'whisper-screen-campaign-attempt-v1', 'campaign_attempt': 3,
            'prior_runs': records, 'same_model_and_input': True, 'automatic_retry': False}


def checkout_head():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, timeout=10)
    return result.stdout.decode('ascii').strip()


def admission(env):
    release = read_json(HERE / 'execution-release.json')
    require(set(release) == {'schema', 'experiment', 'campaign_attempt', 'prior_run_ids', 'approved',
                            'source_freeze_sha256'} and
            release['schema'] == 'whisper-screen-third-execution-release-v1' and
            release['experiment'] == EXPERIMENT and release['campaign_attempt'] == 3 and
            release['prior_run_ids'] == [p[0] for p in PRIOR] and release['approved'] is True,
            'reviewed third activation required')
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + BRANCH,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux'}
    require(all(env.get(k) == v for k, v in expected.items()), 'one-shot public hosted identity')
    require(re.fullmatch('[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch('[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')) and
            env['GITHUB_RUN_ID'] not in {str(p[0]) for p in PRIOR}, 'fresh immutable run/source')
    require(checkout_head() == env['GITHUB_SHA'], 'checked-out exact source head')
    event = read_json(env['GITHUB_EVENT_PATH'])
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('before') == '0' * 40 and event.get('ref') == expected['GITHUB_REF'] and
            event.get('after') == env['GITHUB_SHA'] and event.get('head_commit', {}).get('id') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('private') is False and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'], 'fresh branch exact head')
    freeze_path = HERE / 'execution-freeze.json'
    require(file_hash(freeze_path) == release['source_freeze_sha256'], 'third source freeze identity')
    freeze = read_json(freeze_path)
    require(set(freeze) == {'schema', 'files', 'inference_code', 'setup_code'} and
            freeze['schema'] == 'whisper-screen-third-source-freeze-v1', 'third freeze schema')
    require(set(freeze['files']) == {p.relative_to(ROOT).as_posix() for p in freeze_builder.source_paths()},
            'complete frozen source closure')
    for name, digest in freeze['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                name.startswith(('research/', 'tools/', 'tests/')), 'freeze path')
        regular(ROOT / name); require(file_hash(ROOT / name) == digest, 'source drift')
    require(freeze['inference_code'] == INFERENCE_CODE and freeze['setup_code'] == SETUP_CODE and
            set(SETUP_CODE) <= set(freeze['files']), 'separate exact inference/setup mounts')
    require((ROOT / WORKFLOW).read_bytes() == (HERE / 'workflow.yml').read_bytes(), 'workflow drift')
    for old, workflow in ((BASE, 'source-screen32-whisper-third-v1.yml'),
                          (BASE / 'attempt-2', 'source-screen32-whisper-third-attempt-2-v1.yml')):
        require(read_json(old / 'execution-release.json')['approved'] is False and
                not (ROOT / '.github/workflows' / workflow).exists(), 'prior campaigns disarmed')
    return freeze, verify_prior()


def main(env=None):
    started = time.monotonic(); env = os.environ if env is None else env
    freeze, campaign = admission(env)
    return campaign_host.execute(env, freeze, started, EXPERIMENT,
                                 file_hash(HERE / 'execution-freeze.json'), campaign)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'stopped_no_retry', 'error_type': type(error).__name__}))
        raise SystemExit(1)
