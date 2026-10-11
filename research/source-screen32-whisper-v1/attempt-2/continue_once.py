"""Second, separately approved campaign attempt; never a retry of the old workflow."""
from pathlib import Path
import json
import os
import re
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from common import (EXPERIMENT as FIRST_EXPERIMENT, JOB_SHA, BOUNDARY, LIMITS,
                    require, read_json, file_hash, identity, regular)
import hosted_run as runtime

EXPERIMENT = 'qwen16-whisper-small-third-attempt-2-v1'
BRANCH = 'research/' + EXPERIMENT
WORKFLOW = '.github/workflows/source-screen32-whisper-third-attempt-2-v1.yml'
PRIOR_RUN = 38105471896
PRIOR_PROVENANCE_SHA = '2c01ce5e0dbbc79dfda5ee59d6cc734eec995db882ac22e9465cf99ef2f93504'


def verify_prior():
    """Exact retained first-run bytes, plus zero audio attempts, before acquisition."""
    directory = BASE / 'evidence/run-38105471896'
    provenance_path = directory / 'provenance.json'
    require(file_hash(provenance_path) == PRIOR_PROVENANCE_SHA, 'fixed first-run provenance')
    provenance = read_json(provenance_path)
    require(provenance['run_id'] == PRIOR_RUN and provenance['attempts'] == 0 and
            provenance['not_run'] == 16 and len(provenance['files']) == 7,
            'prior campaign run identity')
    for name, expected in provenance['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'prior evidence path')
        require(identity(directory / name) == expected, 'immutable prior artifact bytes')
    receipt = read_json(directory / 'host-receipt.json')
    require(receipt['experiment'] == FIRST_EXPERIMENT and receipt['run_id'] == str(PRIOR_RUN) and
            receipt['status'] == 'stopped_no_retry' and receipt['attempts'] == 0 and receipt['completed'] == 0 and
            receipt['stages'] == {'setup': 'failed_no_retry', 'whisper': 'not_run'} and
            receipt['limits'] == LIMITS and receipt['terminal_validation_errors'] == 0,
            'prior setup-only failure without audio attempts')
    # Bind the denominator to the same retained anonymous job, without reading audio.
    job_path = ROOT / 'research/source-screen32-v1/evidence/run-38018029787/blind/job.json'
    require(file_hash(job_path) == JOB_SHA, 'same fixed sixteen source job')
    clips = read_json(job_path)['clips']
    rows = read_json(directory / 'terminal-outcomes.json')
    require(len(rows) == len(clips) == 16, 'prior fixed denominator')
    for row, clip in zip(rows, clips):
        require(row['audio_id'] == clip['audio_id'] and row['wav_sha256'] == clip['wav_sha256'] and
                type(row['attempts']) is int and row['attempts'] == 0 and row['status'] == 'not_run' and
                row['raw_output_sha256'] is None and row['raw_text'] is None and row['stop_reason'] is None and
                row['validation_errors'] == [] and all(row[k] == v for k, v in BOUNDARY.items()),
                'prior zero-attempt terminal rows')
    return {'schema': 'whisper-screen-campaign-attempt-v1', 'campaign_attempt': 2,
            'prior_run_id': PRIOR_RUN, 'prior_audio_attempts': 0,
            'prior_provenance_sha256': PRIOR_PROVENANCE_SHA,
            'prior_terminal_outcomes_sha256': provenance['terminal_outcomes_sha256'],
            'prior_source_freeze_sha256': provenance['source_freeze_sha256'],
            'same_model_and_input': True, 'automatic_retry': False}


def admission(env):
    release = read_json(HERE / 'execution-release.json')
    require(set(release) == {'schema', 'experiment', 'campaign_attempt', 'prior_run_id',
                            'approved', 'source_freeze_sha256'} and
            release['schema'] == 'whisper-screen-second-execution-release-v1' and
            release['experiment'] == EXPERIMENT and release['campaign_attempt'] == 2 and
            release['prior_run_id'] == PRIOR_RUN and release['approved'] is True,
            'reviewed second-attempt activation required')
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + BRANCH,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux'}
    require(all(env.get(k) == v for k, v in expected.items()), 'separate one-shot free public hosted identity')
    require(re.fullmatch('[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch('[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')) and
            env['GITHUB_RUN_ID'] != str(PRIOR_RUN), 'new immutable run/source identity')
    event = read_json(env['GITHUB_EVENT_PATH'])
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('private') is False and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'],
            'first public creation of dedicated second-attempt branch required')
    freeze_path = HERE / 'execution-freeze.json'
    require(file_hash(freeze_path) == release['source_freeze_sha256'], 'reviewed second source freeze')
    freeze = read_json(freeze_path)
    require(set(freeze) == {'schema', 'files', 'blind_code'} and
            freeze['schema'] == 'whisper-screen-second-source-freeze-v1', 'second source freeze schema')
    for name, digest in freeze['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                name.startswith(('research/', 'tools/', 'tests/')), 'source freeze path')
        regular(ROOT / name)
        require(file_hash(ROOT / name) == digest, 'reviewed source drift')
    require((ROOT / WORKFLOW).read_bytes() == (HERE / 'workflow.yml').read_bytes(), 'second workflow drift')
    require(not (ROOT / '.github/workflows/source-screen32-whisper-third-v1.yml').exists(),
            'first workflow remains historical, not activated')
    require(read_json(BASE / 'execution-release.json')['approved'] is False, 'first gate stays disarmed')
    original = read_json(BASE / 'execution-freeze.json')
    require(set(freeze['blind_code']) == set(original['blind_code']) and len(freeze['blind_code']) == 8 and
            set(freeze['blind_code']) <= set(freeze['files']), 'unchanged blind eight-file mount')
    # First-run evidence is host-only; no prior outputs or campaign metadata enter inference.
    require(all('/attempt-2/' not in p and '/evidence/' not in p for p in freeze['blind_code']),
            'no campaign or peer context in ASR mount')
    campaign = verify_prior()
    return freeze, campaign


def main(env=None):
    started = time.monotonic()
    env = os.environ if env is None else env
    freeze, campaign = admission(env)
    return runtime.execute(env, freeze, started, EXPERIMENT,
                           file_hash(HERE / 'execution-freeze.json'), campaign)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'stopped_no_retry', 'error_type': type(error).__name__}))
        raise SystemExit(1)
