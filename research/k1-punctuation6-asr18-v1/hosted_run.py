"""Fixed, default-disabled K1 six-input/eighteen-slot trusted host.

No Docker/runtime/model/network action occurs before the checked-in release
booleans pass. A real release additionally requires immutable Git/API source,
complete once-only history, root-approved new-input authority and actual bytes.
Private screening inventory is neither read nor emitted by this program.
"""
from pathlib import Path
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import campaign_contract as c
import asr18_metadata as m
from campaign_contract import require, canonical, digest, read_json, read_bytes, write_once

REPOSITORY = 'jiying2007/kws-pipeline'
NEW = 'research/k1-punctuation6-asr18-v1/'
WORKFLOW = '.github/workflows/k1-punctuation6-asr18-v1.yml'
SOURCE_REF = 'research/k1-punctuation6-asr18-r1'
EVIDENCE_REF = 'evidence/k1-punctuation6-asr18-r1'
HOST_STAGES = ('admitted', 'input', 'setup', 'prepare', 'policy_freeze', 'capacity_proof', 'decode', 'cleanup', 'package')
SHARED = frozenset(('prepared.json','run-identity.json','kernel-scope.json','cpu-process-guard.json',
    'blind-job.json','execution-plan.json','progress.json','contract.json','decoder-inputs.json',
    'model-load.json','native-source-lock.json','native-asset-lock.json','native-config.json',
    'native-resolved-generation-config.json','native-frontend-config.json','optional-tool-probes.json',
    'load-receipt.json','worker-complete.json','worker-failure.json'))
CLIP = frozenset(('started.json','completed.json','raw-output.json','raw-native-output.json',
    'raw-transcript.json','raw-native-row.json','input-capture.json','pre-finalize.json'))
ROOT_PUBLIC = frozenset(('host-receipt.json','terminal-outcomes.json','run-preflight.json','input-freeze.json',
    'public-screening-receipt.json','new-cell-projection.json','acquisition-receipt.json','image-transfer.json',
    'asr-setup-public.json','cpu-receipt.json','cleanup-receipt.json','supervision-receipt.json'))


def _observe(observer,stage):
    if observer is not None:
        try:observer.stage(stage)
        except BaseException:pass  # Diagnostics cannot truncate cleanup or replace primary errors.


def release_first():
    release = read_json(HERE / 'execution-release.json', 65536)
    c.validate_release_shape(release)
    require(all(release[key] is True for key in c.RELEASE_APPROVALS), 'execution release disabled or review absent')
    require(all(release[key] is not None for key in c.RELEASE_AUTHORITIES), 'actual authority unavailable')
    return release


class PublicAPI:
    """Public, fixed GitHub read endpoint; no token, redirect or retry."""
    def __init__(self):
        from raw_data_reader import NoRedirect
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.used = set(); self.response_body_bytes = 0
    def accounting(self):
        return {'gets':len(self.used),'response_body_bytes':self.response_body_bytes,'response_body_cap_bytes':c.MIB,
            'scope':'public_api_metadata_separate_from_model_audio_bodies','hard_wire_byte_cap':False}
    def get(self, path, *, absent=False):
        require(type(path) is str and path.startswith('/repos/' + REPOSITORY + '/') and path not in self.used,
                'one fixed public API read per URL')
        require(len(self.used)<6 and self.response_body_bytes<c.MIB,'fixed six metadata GETs and1MiB aggregate body cap')
        self.used.add(path)
        url = 'https://api.github.com' + path
        request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'Accept-Encoding': 'identity'}, method='GET')
        response = None
        try:
            try: response = self.opener.open(request, timeout=30)
            except urllib.error.HTTPError as error: response = error
            allowance=min(512*1024,c.MIB-self.response_body_bytes)
            raw = response.read(allowance+1); self.response_body_bytes+=len(raw)
            require(len(raw)<=allowance and self.response_body_bytes<=c.MIB and response.geturl() == url and
                    response.headers.get('Content-Encoding', 'identity') == 'identity', 'bounded unredirected API result')
            status = response.getcode()
            if absent:
                require(status == 404, 'evidence ref already exists or absence unverified')
                return None
            require(status == 200, 'public API authority unavailable')
            return c.decode_json(raw, require_canonical=False)
        finally:
            if response is not None: response.close()


def event_identity(env, event):
    fixed = {'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_REF': 'refs/heads/' + SOURCE_REF,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_JOB': 'asr18',
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
    require(all(env.get(k) == v for k, v in fixed.items()), 'fixed public standard ubuntu24.04 first-attempt job')
    head, run = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    require(c.HEAD.fullmatch(head) and re.fullmatch('[1-9][0-9]{0,19}', run), 'actual run/source identity')
    require(int(run) != 38207026177, 'sealed earlier campaign run cannot become a new authority')
    require(env.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/' + WORKFLOW + '@refs/heads/' + SOURCE_REF and
            env.get('GITHUB_WORKFLOW_SHA') == head, 'fixed source workflow identity')
    require(type(event) is dict and event.get('created') is True and event.get('deleted') is False and
        event.get('forced') is False and event.get('before') == '0' * 40 and event.get('after') == head and
        event.get('ref') == 'refs/heads/' + SOURCE_REF and
        event.get('repository', {}).get('full_name') == REPOSITORY and
        event.get('repository', {}).get('private') is False, 'first public source creation event')
    return {'repository': REPOSITORY, 'run_id': int(run), 'run_attempt': 1,
            'job_key': 'asr18', 'executed_head_sha': head}


def _unique_page(api, path, key):
    # Exactly one matching run/job is required, so any total above one is already
    # a terminal replay/identity rejection. No general history paginator exists.
    value=api.get(path+('&' if '?' in path else '?')+'per_page=100&page=1')
    require(type(value) is dict and type(value.get('total_count')) is int and value['total_count']==1 and
        type(value.get(key)) is list and len(value[key])==1,'complete unique counted API page required')
    return value[key]


def prior_history_identity():
    return {'schema': 'k1-asr18-prior-public-claim-history-v1', 'repository': REPOSITORY,
            'source_branch': SOURCE_REF, 'workflow': WORKFLOW, 'prior_runs': [],
            'prior_runs_scope': 'only_new_source_branch_and_fixed_workflow',
            'sealed_prior_campaigns': [{'observation': 'frozen_existing_public_failure_observation', 'experiment': 'k1-punctuation6-asr18-v1',
                'source_branch': 'research/k1-punctuation6-asr18-v1',
                'result_branch': 'evidence/k1-punctuation6-asr18-v1',
                'run_id': 38207026177, 'run_attempt': 1,
                'executed_head_sha': '0f3cc89a3235c4b3af294dc6b5816ff9877c0f42',
                'planned_slots': 18, 'actual_consumed_attempts': None, 'reusable_slots': 0,
                'status': 'failed_consumption_unknown_sealed'}]}


def api_preflight(api, run, release):
    base = '/repos/' + REPOSITORY
    current = api.get(base + '/actions/runs/' + str(run['run_id']))
    require(current.get('id') == run['run_id'] and type(current.get('run_attempt')) is int and current.get('run_attempt') == 1 and
        current.get('status')=='in_progress' and current.get('conclusion') is None and
        current.get('event') == 'push' and current.get('head_branch') == SOURCE_REF and
        current.get('head_sha') == run['executed_head_sha'] and current.get('path') == WORKFLOW and
        current.get('repository', {}).get('full_name') == REPOSITORY and
        current.get('repository', {}).get('private') is False and
        current.get('head_repository', {}).get('full_name') == REPOSITORY and
        type(current.get('workflow_id')) is int and current['workflow_id'] > 0, 'actual current API run identity')
    workflow = api.get(base + '/actions/workflows/' + str(current['workflow_id']))
    require(workflow.get('id') == current['workflow_id'] and workflow.get('path') == WORKFLOW, 'actual workflow path')
    branch = urllib.parse.quote(SOURCE_REF, safe='')
    history = _unique_page(api, base + '/actions/workflows/' + str(current['workflow_id']) + '/runs?branch=' + branch, 'workflow_runs')
    require(len(history) == 1 and history[0].get('id') == run['run_id'] and
            history[0].get('head_sha') == run['executed_head_sha'] and history[0].get('run_attempt') == 1,
            'unique complete workflow/source-branch history required; deleted refs never reset authority')
    require(digest(canonical(prior_history_identity())) == release['prior_public_claim_history_sha256'],
            'independently released prior public claim history')
    jobs = _unique_page(api, base + '/actions/runs/' + str(run['run_id']) + '/attempts/1/jobs', 'jobs')
    require(len(jobs) == 1, 'single fixed public hosted job')
    job = jobs[0]
    require(job.get('name') == 'asr18' and type(job.get('id')) is int and job['id'] > 0 and
        job.get('run_id') == run['run_id'] and job.get('status')=='in_progress' and job.get('conclusion') is None and job.get('head_sha') == run['executed_head_sha'] and
        job.get('labels') == ['ubuntu-24.04'] and type(job.get('runner_name')) is str and
        job['runner_name'].startswith('GitHub Actions ') and job.get('runner_group_name') == 'GitHub Actions',
        'actual standard hosted ubuntu24.04 runner')
    ref = api.get(base + '/git/ref/heads/' + SOURCE_REF)
    require(ref.get('ref') == 'refs/heads/' + SOURCE_REF and
            ref.get('object', {}).get('type') == 'commit' and ref['object'].get('sha') == run['executed_head_sha'], 'actual source ref/head')
    api.get(base + '/git/ref/heads/' + EVIDENCE_REF, absent=True)
    return {'schema': 'k1-asr18-run-preflight-v1', 'identity': run, 'numeric_job_id': job['id'],
            'workflow': WORKFLOW, 'source_branch': SOURCE_REF, 'evidence_branch': EVIDENCE_REF,
            'source_branch_workflow_runs': 1, 'evidence_ref_absent': True,
            'prior_public_claim_history_sha256': release['prior_public_claim_history_sha256'],
            'public_standard_hosted_ubuntu24': True, 'api_metadata_accounting':api.accounting()}


def git_read(*args):
    process = subprocess.run(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false', '-c',
        'core.hooksPath=/dev/null', *args], cwd=ROOT, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent',
        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}, capture_output=True, timeout=30, check=True)
    require(len(process.stdout) <= 16 * c.MIB and len(process.stderr) <= 65536, 'bounded git observation')
    return process.stdout


def source_authority(run, release):
    """Expected freeze comes from the fixed immutable Git blob, never its local candidate."""
    head = run['executed_head_sha']
    require(git_read('rev-parse', 'HEAD').decode().strip() == head and
            git_read('status', '--porcelain=v1', '--untracked-files=all') == b'', 'clean actual executed Git source required')
    tree = git_read('rev-parse', head + '^{tree}').decode().strip()
    require(c.HEAD.fullmatch(tree), 'actual committed source tree')
    frozen_bytes = git_read('cat-file', 'blob', head + ':' + NEW + 'execution-freeze.json')
    freeze = c.decode_json(frozen_bytes)
    expected_sha = digest(frozen_bytes)
    require(read_bytes(HERE / 'execution-freeze.json') == frozen_bytes, 'working source freeze differs from independent immutable blob')
    c.require_execution_release(release, freeze, expected_freeze_sha256=expected_sha)
    require(read_bytes(HERE / 'execution-release.json') == git_read('cat-file', 'blob', head + ':' + NEW + 'execution-release.json'), 'committed exact release bytes')
    from prepare_freeze import declared_rosters
    rosters = declared_rosters(ROOT)
    require(set(freeze) == {'schema', *rosters} and freeze['schema'] == 'k1-asr18-source-freeze-v1', 'complete exact source freeze')
    for role, names in rosters.items():
        require(set(freeze[role]) == names, 'declared complete source role roster')
        for name, identity in freeze[role].items():
            require(c.identity(ROOT / name) == identity, 'actual closed source byte drift')
    require(read_bytes(ROOT / WORKFLOW) == read_bytes(HERE / 'workflow.yml') ==
            git_read('cat-file', 'blob', head + ':' + WORKFLOW), 'active workflow is exact committed inert template')
    return freeze, expected_sha, tree


def source_claim(run, release, preflight, freeze_sha):
    claim = {'schema': 'k1-asr18-once-source-claim-v1', 'identity': run,
        'source_freeze_sha256': freeze_sha, 'release_sha256': digest(canonical(release)),
        'source_branch': SOURCE_REF, 'result_branch': EVIDENCE_REF, 'workflow': WORKFLOW,
        'numeric_job_id': preflight['numeric_job_id'],
        'prior_public_claim_history_sha256': release['prior_public_claim_history_sha256'],
        'api_preflight_verified': True, 'source_branch_workflow_runs': 1,
        'maximum_new_asr_calls': 18, 'old_calls_allowed': 0, 'new_audio_forwards_at_claim': 0}
    c.validate_source_claim(canonical(claim), expected_sha256=digest(canonical(claim)), release=release,
        source_freeze_sha256=freeze_sha, run_id=str(run['run_id']), executed_head_sha=run['executed_head_sha'])
    return claim


def screening_policy_identity(source_freeze):
    # The reviewed algorithm's original bytes define this policy identity. It
    # comes from the independently checked committed source closure, never a
    # self-declared receipt field, path argument or newly generated policy hash.
    name=NEW+'asr18_metadata.py'
    expected=source_freeze['host_code'][name]
    require(c.exact_json(expected,source_freeze['inference_code'][name]),'one frozen screening algorithm identity')
    require(c.exact_json(c.identity(ROOT/name),expected),'reviewed screening algorithm source bytes drift')
    return expected


def input_authorities(release, source_freeze):
    freeze = read_json(HERE / 'input-freeze.json', 128 * 1024)
    require(set(freeze) == {'schema','execution_enabled','actual_new_cells','actual_blind_job',
        'actual_opaque_order','actual_screening_receipt','actual_worker_binding'} and
        freeze['schema'] == 'k1-asr18-input-freeze-v1' and freeze['execution_enabled'] is False and
        all(freeze[key] is not None for key in set(freeze) - {'schema','execution_enabled'}), 'actual fixed input authority required')
    require(digest(canonical(freeze)) == release['actual_input_freeze_sha256'], 'released actual input freeze')
    authority = read_json(HERE / 'tts-source-authority.json', 2 * c.MIB)
    from raw_data_reader import validate_authority
    validate_authority(authority, expected_authority_sha256=release['actual_tts_authority_sha256'])
    receipt = freeze['actual_screening_receipt']
    require(digest(canonical(receipt)) == release['actual_screening_receipt_sha256'] and
            freeze['actual_new_cells'] == receipt['cells'], 'all six released new cells and screening receipt')
    policy=screening_policy_identity(source_freeze)
    require(receipt['screening_policy_sha256']==policy['sha256'],'screening receipt policy differs from reviewed source algorithm')
    c.validate_worker_admission(release, freeze['actual_worker_binding'], freeze['actual_blind_job'], freeze['actual_opaque_order'])
    return freeze, authority


def verify_download(release, frozen_input, authority, *, expected_policy_sha256):
    from raw_data_reader import RawDataReader
    from input_binding import bind_complete_input_package
    c.sha(expected_policy_sha256)
    reader = RawDataReader()
    fetched = reader.read_complete(authority, expected_authority_sha256=release['actual_tts_authority_sha256'])
    bound = bind_complete_input_package(fetched['archive'], authority, canonical(frozen_input['actual_screening_receipt']),
        expected_authority_sha256=release['actual_tts_authority_sha256'],
        expected_source_freeze_sha256=authority['input']['source_freeze_sha256'],
        expected_receipt_sha256=release['actual_screening_receipt_sha256'],
        expected_policy_sha256=expected_policy_sha256)
    verified = bound['verification']
    for actual, expected in ((verified['projection']['cells'], frozen_input['actual_new_cells']),
        (verified['blind']['job'], frozen_input['actual_blind_job']),
        (verified['blind']['execution_plan'], frozen_input['actual_opaque_order']),
        (verified['worker_input_binding'], frozen_input['actual_worker_binding'])):
        require(c.exact_json(actual, expected), 'remeasured complete input/anonymous authority drift')
    require(verified['blind']['eligible_cells'] > 0, 'no eligible audio means no setup/model/claim')
    return bound, fetched['accounting']


def projection_for(cells):
    value = {'schema':'k1-asr18-untrusted-metadata-projection-v1','experiment':m.EXPERIMENT,
        'status':'metadata_only_external_custody_and_review_required','execution_enabled':False,'cells':cells}
    m._validate_projection(value)
    return value


def _completion(directory, slot, run, policy_sha):
    return _completion_bytes(lambda name: read_bytes(directory / name, cap=8*c.MIB),
        read_bytes(directory.parent / 'native-config.json', cap=8*c.MIB), slot, run, policy_sha)


def _completion_bytes(read, config_raw, slot, run, policy_sha, binding=None):
    started_raw = read('started.json')
    started = c.decode_json(started_raw)
    completed = c.decode_json(read('completed.json'))
    require(started.get('schema') == 'retained-asr-started-v1' and
        completed.get('schema') == 'retained-asr-completed-v1' and completed.get('status') == 'success' and
        completed.get('decoder_completed') is True and completed.get('started_sha256') == digest(started_raw), 'complete source-bound native pair')
    common = {k:v for k,v in started.items() if k != 'schema'}
    require(all(c.exact_json(completed.get(k),v) for k,v in common.items()), 'completion/start exact binding')
    expected = {'recognizer_id':slot['family'],'source_id':'k1asr18-'+slot['wav_sha256'],
        'request_id':'k1asr18-'+str(run['run_id'])+'-'+slot['family']+'-'+slot['audio_id'],
        'file_sha256':slot['wav_sha256'],'pcm_sha256':slot['pcm_sha256'],'audio_frames':slot['audio_frames'],
        'run_id':str(run['run_id']),'executed_head_sha':run['executed_head_sha'],
        'recognizer_policy_sha256':policy_sha}
    require(all(started.get(k) == v for k,v in expected.items()) and started.get('audio_id')==slot['audio_id'] and
        started.get('run_attempt')==1 and type(started.get('run_attempt')) is int, 'fixed actual slot and run authority')
    if binding is not None:
        require(all(started.get(key)==value for key,value in {
            'job_sha256':binding['job_sha256'],'execution_plan_sha256':binding['execution_plan_sha256'],
            'worker_input_binding_sha256':digest(canonical(binding)),'screening_receipt_sha256':binding['screening_receipt_sha256'],
            'tts_authority_sha256':binding['tts_authority_sha256']}.items()), 'actual released input authority on every start')
    require(completed.get('processed_start_frame') == 0 and completed.get('processed_end_frame') == slot['audio_frames'], 'full source coverage')
    transcript_raw = read('raw-transcript.json')
    native_raw = read('raw-native-output.json')
    require(completed['raw_output_sha256'] == digest(transcript_raw) and
        completed['raw_native_output_sha256'] == digest(native_raw), 'unmodified complete native output binding')
    transcript = c.decode_json(transcript_raw); native = c.decode_json(native_raw)
    require(transcript.get('request_id') == started['request_id'] and transcript.get('truncated') is False and
        transcript.get('output_scope') == 'complete-decoder-transcript' and
        type(transcript.get('actual_text')) is str and len(transcript['actual_text']) <= 32768, 'bounded complete actual transcript')
    import worker_receipts as wr
    config = c.decode_json(config_raw)
    wr.validate_native_config(config); wr.validate_native_capture(config, completed['native_capture'])
    require(completed['native_capture'] == native.get('native_capture'), 'native capture body drift')
    require(transcript.get('native_output')=={'path':'raw-native-output.json','bytes':len(native_raw),'sha256':digest(native_raw)},
        'full raw native output reference')
    row=None
    if transcript.get('native_row') is not None:
        row_raw=read('raw-native-row.json'); row=c.decode_json(row_raw)
        require(transcript['native_row']=={'path':'raw-native-row.json','bytes':len(row_raw),'sha256':digest(row_raw)},'actual native row reference')
    require(transcript['actual_text']==wr._actual_text(slot['family'],config,row,native),'actual unmodified native text only')
    if slot['family']=='whisper': require(read('raw-output.json')==native_raw,'original Whisper raw output retained')
    return completed


def terminal_accounting(root, projection, job, plan, run, successful_families):
    """Any created slot directory consumes one; a failed final write never succeeds.

    Earlier slots can be accepted only when a later sequential slot exists or a
    source-bound failure index proves the worker advanced after durable return.
    The failed current slot is never promoted by completed.json existence.
    """
    ledger = c.initial_ledger(projection)
    if not job['clips']:
        return {'schema':'k1-asr18-terminal-outcomes-v1','identity':run,
            'projection_sha256':digest(canonical(projection)), 'slots':ledger,'accounting_valid':True}
    slots = c.fixed_slots(projection, job, plan); valid = True; stopped = False
    policy = root / 'policy/policy.json'; policy_sha = c.identity(policy)['sha256'] if policy.exists() else None
    for family in c.FAMILIES:
        family_slots = [(i,s) for i,s in enumerate(slots) if s['family'] == family and s['disposition'] == 'eligible']
        folder = root / 'output' / family
        directories = [s['audio_id'] for _,s in family_slots if (folder/s['audio_id']).exists()]
        if family in successful_families and len(directories)!=len(family_slots): valid=False
        if folder.exists() and any(p.name.startswith('clip-') and p.is_dir() and p.name not in {s['audio_id'] for _,s in family_slots} for p in folder.iterdir()): valid=False
        require(all(not (folder/x).is_symlink() and (folder/x).is_dir() for x in directories), 'uncertain slot must be unaliased directory')
        failure_index = None
        if (folder / 'worker-failure.json').exists():
            fail = read_json(folder / 'worker-failure.json', 16384); diag = fail.get('diagnostic', {})
            if diag.get('family') == family and diag.get('mode') == 'decode' and type(diag.get('index')) is int and 1 <= diag['index'] <= len(family_slots):
                failure_index = diag['index']
        for position, (i,slot) in enumerate(family_slots,1):
            consumed = slot['audio_id'] in directories
            if not consumed:
                stopped = True
                continue
            later_exists = any(s['audio_id'] in directories for _,s in family_slots[position:])
            durable_prefix = family in successful_families or later_exists or (failure_index is not None and position < failure_index)
            complete = False
            if durable_prefix and not stopped:
                try:
                    _completion(folder / slot['audio_id'], slot, run, policy_sha); complete = True
                except BaseException:
                    valid = False
            if stopped: valid = False
            ledger[i] = {'id':ledger[i]['id'],'status':'complete' if complete else 'failed_no_retry','attempts':1}
            if not complete: stopped = True
        # A failed family, including cleanup failure, never gives later families credit.
        if family not in successful_families: stopped = True
    try: c.validate_ledger(projection, ledger)
    except ValueError: valid = False
    return {'schema':'k1-asr18-terminal-outcomes-v1','identity':run,
        'projection_sha256':digest(canonical(projection)), 'slots':ledger,'accounting_valid':valid}


def validate_worker_complete(folder, family, mode, job, plan):
    value = read_json(folder/'worker-complete.json',65536)
    expected = {'schema':'k1-asr18-worker-result-v1','family':family,'mode':mode,
        'status':'prepared_no_recognition' if mode=='prepare' else 'all_planned_native_complete',
        'audio_forward_calls':0 if mode=='prepare' else len(job['clips']),
        'completed':0 if mode=='prepare' else len(job['clips']),
        'audio_ids':[] if mode=='prepare' else plan['audio_order'],
        'job_sha256':digest(canonical(job)), 'execution_plan_sha256':digest(canonical(plan))}
    require(c.exact_json(value,expected), 'source-bound actual worker success')
    require(not (folder/'worker-failure.json').exists(), 'failure cannot coexist with accepted worker success')
    if mode=='prepare':
        require(c.exact_json(read_json(folder/'prepared.json'),expected) and not any(folder.glob('clip-*')), 'prepare is zero-forward')
    return value


def _walk_files(root):
    result = set()
    require(root.is_dir() and not root.is_symlink(), 'public root directory')
    for base, dirs, files in os.walk(root, followlinks=False):
        require(not any((Path(base)/name).is_symlink() for name in dirs+files), 'public member alias prohibited')
        for name in files:
            item = Path(base)/name
            read_bytes(item, cap=8*c.MIB)
            result.add(item.relative_to(root).as_posix())
    return result


def _native_roster(root, storage, family):
    base = f'{storage}/{family}/'; folder = root/storage/family
    if not folder.exists(): return set()
    allowed = {base+name for name in SHARED}
    for index in range(1,7):
        opaque = f'clip-{index:06d}'
        allowed |= {base+opaque+'/'+name for name in CLIP}
        allowed |= {base+opaque+f'/update-{step:03d}.json' for step in range(1,225)}
        allowed |= {base+opaque+suffix for suffix in ('.native-observation.json','.native-generation-return.json',
            '.native-tokenizer-input.json','.native-inference-return.json','.native-returned-raw.json')}
    for name,field in (('native-source-lock.json','files'),('native-asset-lock.json','assets')):
        if not (folder/name).exists(): continue
        lock=read_json(folder/name,8*c.MIB)
        rows=lock[field]+(lock.get('empty_source_files',[]) if field=='files' else [])
        for row in rows:
            if field=='assets' and row.get('body_retained') is not True: continue
            member=row['path']; prefix='output/'+family+'/'
            require(type(member) is str and member.startswith(prefix) and not any(p in ('','..','.') for p in member.split('/')), 'same family exact retained body')
            physical=base+member[len(prefix):]
            require(c.identity(root/physical)=={'bytes':row['bytes'],'sha256':row['sha256']}, 'complete actual retained source/assets')
            allowed.add(physical)
    observed={base+name for name in _walk_files(folder)}
    require(observed<=allowed, 'unknown worker member cannot be dropped or published')
    return observed


def collect_public(root):
    """Fixed public evidence trees only; exclude runtime, original audio and gates."""
    root=Path(root); names=set()
    for name in ROOT_PUBLIC:
        if (root/name).exists(): read_json(root/name,8*c.MIB); names.add(name)
    fixed={'authority':{'execution-freeze.json','source-claim.json'},
        'order':{'execution-plan.json','worker-input-binding.json'},
        'policy':{'policy.json','policy-freeze.json','budget-proof.json'}}
    for folder,allowed in fixed.items():
        if (root/folder).exists():
            observed=_walk_files(root/folder)
            require(observed<=allowed, 'only fixed public authority/policy/order members')
            names|={folder+'/'+name for name in observed}
    for storage in ('output','frozen'):
        if not (root/storage).exists(): continue
        require({p.name for p in (root/storage).iterdir()} <= set(c.FAMILIES), 'fixed output families')
        for family in c.FAMILIES: names|=_native_roster(root,storage,family)
    if (root/'source').exists():
        freeze=read_json(root/'authority/execution-freeze.json')
        expected={name:row for name,row in freeze['host_code'].items()}
        expected[NEW+'execution-freeze.json']=c.identity(root/'authority/execution-freeze.json')
        require(_walk_files(root/'source')==set(expected), 'whole actual public source closure required')
        for name,identity in expected.items(): require(c.identity(root/'source'/name)==identity, 'complete public source identity')
        names|={'source/'+name for name in expected}
    if (root/'whisper-setup').exists():
        from runtime_session import whisper_public_names
        observed=_walk_files(root/'whisper-setup')
        require(observed<=whisper_public_names(), 'exact bounded Whisper setup metadata')
        names|={'whisper-setup/'+name for name in observed}
    require(sum(c.identity(root/name)['bytes'] for name in names)<=c.LIMITS['output_bytes'], 'complete public evidence bound; never prune to fit')
    return sorted(names)


def public_setup_members():
    from runtime_session import whisper_public_names
    return {'whisper-setup/'+name for name in whisper_public_names()}


def host_receipt(run, freeze_sha, tree, screening_sha):
    return {'schema':'k1-asr18-host-receipt-v1','identity':run,'boundary':c.BOUNDARY,'limits':c.LIMITS,
        'status':'started','stage':'admitted','source_freeze_sha256':freeze_sha,'executed_tree_sha':tree,
        'source_claim_sha256':None,'screening_receipt_sha256':screening_sha,'input_verified':False,
        'stages':{name:'not_run' for name in ('setup',)+tuple(f+'_'+mode for mode in ('prepare','decode') for f in c.FAMILIES)},
        'all3_policy_frozen_before_recognition':False,'primary_error':None,'cleanup_error':None,
        'combined_setup_seconds':0.0,'engine_wall_seconds':{f:0.0 for f in c.FAMILIES},'wall_seconds':0.0,
        'accounting_valid':True,'cleanup_verified':False}


def _exact(value, keys, reason):
    require(type(value) is dict and set(value)==set(keys),reason)


def _safe_failure(value):
    if value is None: return
    require(type(value) is dict and set(value)<={'error_type','returncode','errno','frames','frames_truncated'} and
        value.get('error_type') in ('TimeoutExpired','CalledProcessError','TimeoutError','ValueError','TypeError',
            'KeyError','OSError','RuntimeError','KeyboardInterrupt','SystemExit','OtherError'), 'bounded error type only')
    for key in ('returncode','errno'):
        if key in value: require(type(value[key]) is int and -255<=value[key]<=255,'bounded numeric OS status')
    if 'frames_truncated' in value: require(type(value['frames_truncated']) is bool,'bounded truncation flag')
    require(type(value.get('frames',[])) is list,'bounded frame list')
    for frame in value.get('frames',[]):
        _exact(frame,{'file','function','line'},'bounded source frame')
        require(frame['file'] in ('<redacted>','research/k1-punctuation6-asr18-v1/resource_host.py') and
            (frame['line'] is None or type(frame['line']) is int and 1<=frame['line']<=1000000) and
            type(frame['function']) is str and re.fullmatch(r'(?:<redacted>|[A-Za-z_][A-Za-z0-9_]*)',frame['function']), 'source-bound frame only')
    require(len(value.get('frames',[]))<=8 and len(canonical(value))<=2048,'bounded failure shape')



def _prepared_evidence(files, prefix, family, run, job, plan, policy_rows):
    import worker_receipts as wr
    mandatory=set(wr.FILES)|{'prepared.json','worker-complete.json','kernel-scope.json',
        'cpu-process-guard.json','blind-job.json','execution-plan.json'}
    require(all(prefix+name in files for name in mandatory),'whole actual prepared metadata and lock closure')
    require(prefix+'worker-failure.json' not in files,'prepare failure cannot satisfy all3 policy freeze')
    source,model,config=[c.decode_json(files[prefix+name],require_canonical=False) for name in wr.FILES]
    require(source.get('schema')=='retained-asr-source-lock-v1' and model.get('schema')=='retained-asr-model-lock-v1', 'prepared native lock schemas')
    wr._inventory(source.get('files'));wr._inventory(model.get('assets'));wr.validate_native_config(config)
    require(source.get('executed_head_sha')==run['executed_head_sha'] and source.get('workflow_run_id')==str(run['run_id']), 'prepared actual run/source closure')
    require(config['source_lock_sha256']==digest(files[prefix+'native-source-lock.json']) and
        config['asset_lock_sha256']==digest(files[prefix+'native-asset-lock.json']) and
        config['adapter_id']==wr.ADAPTERS[family] and config['model_family']==model['model_family'] and
        model['model_sha256']==digest(canonical(model['assets'])), 'prepared measured config/model/source identity')
    identity={'recognizer_id':family,**{k:model[k] for k in wr.IDENTITY},'native_decode_config':config,
        'native_decode_config_sha256':digest(files[prefix+'native-config.json'])}
    require(c.exact_json(identity,policy_rows[family]), 'complete policy and actual prepared identity')
    expected={'schema':'k1-asr18-worker-result-v1','family':family,'mode':'prepare','status':'prepared_no_recognition',
        'audio_forward_calls':0,'completed':0,'audio_ids':[],'job_sha256':digest(canonical(job)),
        'execution_plan_sha256':digest(canonical(plan))}
    require(all(c.exact_json(c.decode_json(files[prefix+name]),expected) for name in ('prepared.json','worker-complete.json')),
        'durable complete prepare-only result')
    require(c.exact_json(c.decode_json(files[prefix+'blind-job.json']),job) and
        c.exact_json(c.decode_json(files[prefix+'execution-plan.json']),plan),'prepared exact anonymous inputs')
    require(not any(name.startswith(prefix+'clip-') for name in files),'no prepared forward traces')


def validate_public_evidence(files, run_identity, *, phase='final'):
    """Revalidate in-memory archive members, including failures and all18 rows.

    Source/native file identities and the binary/audio exclusion are checked by
    the transport before this function. No caller-selected validator is used.
    """
    require(phase in ('prepared','final') and type(files) is dict,'fixed evidence phase')
    def value(name):
        require(name in files,'required public evidence missing')
        return c.decode_json(files[name], require_canonical=False)
    required={'host-receipt.json','run-preflight.json','input-freeze.json','public-screening-receipt.json',
        'new-cell-projection.json','authority/execution-freeze.json'}
    require(required<=set(files),'complete source-bound evidence core')
    host=value('host-receipt.json'); template=host_receipt(run_identity,'0'*64,'0'*40,'0'*64)
    _exact(host,template,'exact public host schema')
    require(host['schema']==template['schema'] and c.exact_json(host['identity'],run_identity) and
        c.exact_json(host['boundary'],c.BOUNDARY) and c.exact_json(host['limits'],c.LIMITS),'fixed host identity/boundary/caps')
    require(host['stage'] in HOST_STAGES and host['status'] in ('started','stopped_no_retry','all_planned_native_complete','cleanup_unconfirmed'), 'fixed host terminal status')
    for key in ('input_verified','all3_policy_frozen_before_recognition','accounting_valid','cleanup_verified'):
        require(type(host[key]) is bool,'explicit boolean host observation')
    require(set(host['stages'])==set(template['stages']) and all(v in ('not_run','started','complete','failed') for v in host['stages'].values()), 'fixed host stage roster')
    for key in ('combined_setup_seconds','wall_seconds'):
        require(type(host[key]) in (int,float) and 0<=host[key]<=86400,'bounded observed wall duration')
    require(set(host['engine_wall_seconds'])==set(c.FAMILIES) and all(type(v) in (int,float) and 0<=v<=86400 for v in host['engine_wall_seconds'].values()),'family wall observations')
    _safe_failure(host['primary_error']); _safe_failure(host['cleanup_error'])
    freeze=value('authority/execution-freeze.json'); freeze_sha=digest(canonical(freeze))
    require(host['source_freeze_sha256']==freeze_sha and c.HEAD.fullmatch(host['executed_tree_sha']), 'actual source freeze/head binding')
    release=c.decode_json(files['source/'+NEW+'execution-release.json'], require_canonical=False)
    c.require_execution_release(release,freeze,expected_freeze_sha256=freeze_sha)
    preflight=value('run-preflight.json')
    expected={'schema':'k1-asr18-run-preflight-v1','identity':run_identity,'numeric_job_id':preflight.get('numeric_job_id'),
        'workflow':WORKFLOW,'source_branch':SOURCE_REF,'evidence_branch':EVIDENCE_REF,'source_branch_workflow_runs':1,
        'evidence_ref_absent':True,'prior_public_claim_history_sha256':release['prior_public_claim_history_sha256'],
        'public_standard_hosted_ubuntu24':True,'api_metadata_accounting':preflight.get('api_metadata_accounting')}
    require(c.exact_json(preflight,expected) and type(preflight['numeric_job_id']) is int and preflight['numeric_job_id']>0, 'exact actual preflight observation')
    api_counts=preflight['api_metadata_accounting']
    _exact(api_counts,{'gets','response_body_bytes','response_body_cap_bytes','scope','hard_wire_byte_cap'},'fixed metadata acquisition accounting')
    require(type(api_counts['gets']) is int and api_counts['gets']==6 and type(api_counts['response_body_bytes']) is int and
        0<=api_counts['response_body_bytes']<=c.MIB and api_counts['response_body_cap_bytes']==c.MIB and
        api_counts['scope']=='public_api_metadata_separate_from_model_audio_bodies' and api_counts['hard_wire_byte_cap'] is False,
        'actual metadata bodies separately counted within remaining6GiB reserve')
    inp=value('input-freeze.json'); receipt=value('public-screening-receipt.json'); projection=value('new-cell-projection.json')
    _exact(inp,{'schema','execution_enabled','actual_new_cells','actual_blind_job','actual_opaque_order',
        'actual_screening_receipt','actual_worker_binding'},'exact public input freeze')
    require(inp['schema']=='k1-asr18-input-freeze-v1' and inp['execution_enabled'] is False and
        digest(canonical(inp))==release['actual_input_freeze_sha256'] and
        c.exact_json(inp['actual_screening_receipt'],receipt) and c.exact_json(inp['actual_new_cells'],projection['cells']), 'frozen full new input binding')
    _exact(receipt,{'schema','execution_enabled','tts_authority','tts_authority_sha256','screening_policy_sha256','cells'},'public screening fields only')
    require(receipt['schema']=='k1-asr18-public-screening-receipt-v1' and receipt['execution_enabled'] is False and
        digest(canonical(receipt))==release['actual_screening_receipt_sha256']==host['screening_receipt_sha256'] and
        receipt['tts_authority_sha256']==release['actual_tts_authority_sha256'] and receipt['cells']==projection['cells'], 'immutable root-reviewed screening receipt')
    m.validate_source_authority_shape(receipt['tts_authority']); c.sha(receipt['screening_policy_sha256'])
    policy_name=NEW+'asr18_metadata.py'
    policy_body=files['source/'+policy_name]
    policy_identity={'bytes':len(policy_body),'sha256':digest(policy_body)}
    require(c.exact_json(policy_identity,freeze['host_code'][policy_name]) and
        c.exact_json(policy_identity,freeze['inference_code'][policy_name]) and
        receipt['screening_policy_sha256']==policy_identity['sha256'],
        'public screening policy bound to exact archived reviewed algorithm bytes')
    m._validate_projection(projection)
    job,plan,binding=inp['actual_blind_job'],inp['actual_opaque_order'],inp['actual_worker_binding']
    c.validate_worker_admission(release,binding,job,plan)
    blind=m.blind_metadata(projection)
    require(c.exact_json(job,blind['job']) and c.exact_json(plan,blind['execution_plan']), 'fixed eligible-only anonymous order')
    if 'authority/source-claim.json' in files:
        c.validate_source_claim(files['authority/source-claim.json'], expected_sha256=host['source_claim_sha256'],
            release=release,source_freeze_sha256=freeze_sha,run_id=str(run_identity['run_id']),executed_head_sha=run_identity['executed_head_sha'])
        require(host['input_verified'] is True,'no claim before complete bytes/screening admission')
    else:
        require(host['source_claim_sha256'] is None and not any(name.startswith(('output/','frozen/')) for name in files),'no native work without durable source claim')
    for name,expected in (('order/execution-plan.json',plan),('order/worker-input-binding.json',binding)):
        if name in files: require(c.exact_json(value(name),expected),'exact opaque mounted authority')
    policy_rows=None
    if 'policy/policy.json' in files:
        import worker_receipts as wr
        policy_rows=wr._policy(files['policy/policy.json'])
    if host['all3_policy_frozen_before_recognition']:
        require(policy_rows is not None and 'policy/policy-freeze.json' in files,'all3 prepared policy proof')
        for family in c.FAMILIES:
            prefix=('frozen/' if 'frozen/'+family+'/prepared.json' in files else 'output/')+family+'/'
            _prepared_evidence(files,prefix,family,run_identity,job,plan,policy_rows)
    if phase=='prepared':
        require(host['input_verified'] and host['all3_policy_frozen_before_recognition'] and
            not any(re.search(r'/clip-00000[1-6](?:/|\.)',name) for name in files),'all-three prepare proof before any forward')
        for family in c.FAMILIES:
            prepared=value('output/'+family+'/prepared.json')
            require(prepared.get('audio_forward_calls')==0 and prepared.get('mode')=='prepare','no prepare audio forwards')
    if phase=='final':
        terminal=value('terminal-outcomes.json')
        _exact(terminal,{'schema','identity','projection_sha256','slots','accounting_valid'},'exact full denominator terminal receipt')
        require(terminal['schema']=='k1-asr18-terminal-outcomes-v1' and c.exact_json(terminal['identity'],run_identity) and
            terminal['projection_sha256']==digest(canonical(projection)) and terminal['accounting_valid'] is host['accounting_valid'], 'terminal source/new-input identity')
        if terminal['accounting_valid']: c.validate_ledger(projection,terminal['slots'])
        else:
            require(len(terminal['slots'])==18 and host['status']!='all_planned_native_complete','invalid accounting retained only as failure')
            for actual,slot in zip(terminal['slots'],m.slots()):
                _exact(actual,{'id','status','attempts'},'failed accounting fixed row')
                require(actual['id']==slot['slot_id'] and actual['status'] in ('not_run','not_run_duplicate','not_run_quality','complete','failed_no_retry') and
                    type(actual['attempts']) is int and actual['attempts'] in (0,1),'bounded failed accounting observation')
        slots=c.fixed_slots(projection,job,plan)
        for slot,outcome in zip(slots,terminal['slots']):
            if outcome['status']=='complete':
                family=slot['family']; family_prefix='output/'+family+'/'
                family_rows=[(s,o) for s,o in zip(slots,terminal['slots']) if s['family']==family and s['disposition']=='eligible']
                position=[s['audio_id'] for s,o in family_rows].index(slot['audio_id'])+1
                success=host['stages'][family+'_decode']=='complete'
                if success:
                    expected_result={'schema':'k1-asr18-worker-result-v1','family':family,'mode':'decode',
                        'status':'all_planned_native_complete','audio_forward_calls':len(job['clips']),
                        'completed':len(job['clips']),'audio_ids':plan['audio_order'],
                        'job_sha256':digest(canonical(job)),'execution_plan_sha256':digest(canonical(plan))}
                    require(c.exact_json(value(family_prefix+'worker-complete.json'),expected_result) and
                        family_prefix+'worker-failure.json' not in files,'durable acknowledged full decode success')
                later_consumed=any(o['attempts']==1 for s,o in family_rows[position:])
                failure_index=None
                if family_prefix+'worker-failure.json' in files:
                    diagnostic=value(family_prefix+'worker-failure.json').get('diagnostic',{})
                    if diagnostic.get('family')==family and diagnostic.get('mode')=='decode': failure_index=diagnostic.get('index')
                advanced=type(failure_index) is int and position<failure_index<=len(family_rows)
                require(success or later_consumed or advanced,'failed current completion residue never proves durable success')
                prefix=family_prefix+slot['audio_id']+'/'
                _completion_bytes(lambda leaf: files[prefix+leaf],files['output/'+slot['family']+'/native-config.json'],
                    slot,run_identity,digest(files['policy/policy.json']),binding)
            if outcome['attempts']==0 and slot['audio_id'] is not None:
                prefix='output/'+slot['family']+'/'+slot['audio_id']+'/'
                require(not any(name.startswith(prefix) for name in files),'unreported native evidence cannot be not_run')
        if host['status']=='all_planned_native_complete':
            require(host['cleanup_verified'] and host['accounting_valid'] and
                all(row['status'] in ('complete','not_run_duplicate','not_run_quality') for row in terminal['slots']), 'success requires all eligible completed and cleanup')
    if 'policy/policy-freeze.json' in files:
        frozen=value('policy/policy-freeze.json')
        _exact(frozen,{'schema','identity','recognizer_policy_sha256','native_calls_started','prepared_config_files'},'exact all3 policy freeze')
        require(frozen['schema']=='k1-asr18-all3-policy-freeze-v1' and c.exact_json(frozen['identity'],run_identity) and
            frozen['native_calls_started']==0 and type(frozen['native_calls_started']) is int and
            frozen['recognizer_policy_sha256']==digest(files['policy/policy.json']) and set(frozen['prepared_config_files'])==set(c.FAMILIES),'zero-forward policy binding')
        for family,expected in frozen['prepared_config_files'].items():
            key=('frozen/' if 'frozen/'+family+'/native-config.json' in files else 'output/')+family+'/native-config.json'
            require(expected=={'bytes':len(files[key]),'sha256':digest(files[key])},'actual prepared family config')
    if 'policy/budget-proof.json' in files:
        proof=value('policy/budget-proof.json')
        hashes={'recognizer_policy_sha256':digest(files['policy/policy.json']),'worker_input_binding_sha256':digest(canonical(binding)),
            'execution_plan_sha256':digest(canonical(plan)),'job_sha256':digest(canonical(job))}
        fixed={'schema':'k1-asr18-predecode-package-budget-v1','status':'passed_before_any_audio_forward','new_native_calls':0,
            'eligible_clip_count':len(job['clips']),'whisper_full_prefix_steps_per_clip':224,
            'transcript_reserve_characters_each_copy':32768,'transcript_copies_per_clip':2,'extra_json_reserve_bytes':c.MIB}
        caps={'compressed_bytes':12*c.MIB,'expanded_member_bytes':32*c.MIB,'actual_prepared_members':4096,'worst_fixture_members':4096}
        _exact(proof,set(hashes)|set(fixed)|set(caps),'exact prepared capacity proof')
        require(all(c.exact_json(proof[k],v) for k,v in {**hashes,**fixed}.items()) and
            all(type(proof[k]) is int and 1<=proof[k]<=v for k,v in caps.items()),'actual package capacity/input binding')
    if 'asr-setup-public.json' in files:
        setup=value('asr-setup-public.json')
        _exact(setup,{'schema','status','model_calls','actual_download_bytes','setup_receipt_sha256'},'bounded ASR setup public summary')
        require(setup['schema']=='k1-asr18-setup-summary-v1' and setup['status']=='complete' and setup['model_calls']==0 and
            type(setup['actual_download_bytes']) is int and 0<=setup['actual_download_bytes']<=4*c.GIB,'setup complete response bytes')
        c.sha(setup['setup_receipt_sha256'])
    if 'acquisition-receipt.json' in files:
        acq=value('acquisition-receipt.json')
        _exact(acq,{'schema','data','asr_response_body_bytes','whisper','reserved_total_bytes','cap_bytes','hard_wire_byte_cap'},'mixed scope acquisition receipt')
        require(acq['schema']=='k1-asr18-reserved-acquisition-v1' and acq['reserved_total_bytes']==c.LIMITS['acquisition_reserved_bytes'] and
            acq['cap_bytes']==6*c.GIB and acq['hard_wire_byte_cap'] is False,'mixed observation never hard allwire claim')
        require(acq['asr_response_body_bytes'] is None or type(acq['asr_response_body_bytes']) is int and 0<=acq['asr_response_body_bytes']<=4*c.GIB,'ASR actual body reservation')
        data=acq['data']
        _exact(data,{'gets','response_body_bytes','response_body_reservation_bytes','retries','redirects','extra_source_gets','failed','execution_enabled'},'exact acquisition response scope')
        require(type(data['gets']) is int and 1<=data['gets']<=49 and type(data['response_body_bytes']) is int and
            0<data['response_body_bytes']<=16*c.MIB and data['response_body_reservation_bytes']==16*c.MIB and
            all(type(data[k]) is int and data[k]==0 for k in ('retries','redirects','extra_source_gets')) and
            data['failed'] is False and data['execution_enabled'] is False,'actual bounded one-pass complete input bodies')
        if acq['whisper'] is not None:
            w=acq['whisper']
            _exact(w,{'schema','status','aggregate_cap','total_read_bytes','asset_count','network_teardown_complete','model_calls','setup_deadline_seconds'},'fixed Whisper acquisition observation')
            require(w['schema']=='bounded-host-setup-v1' and w['status']=='complete' and w['aggregate_cap']==1536*c.MIB and
                type(w['total_read_bytes']) is int and 0<=w['total_read_bytes']<=1536*c.MIB and
                type(w['asset_count']) is int and w['asset_count']==24 and w['network_teardown_complete'] is True and
                type(w['model_calls']) is int and w['model_calls']==0 and w['setup_deadline_seconds']==1200,'mixed bounded Whisper receive observation')
    return True


def execute(env, run, release, freeze, freeze_sha, tree, preflight, observer=None):
    from resource_host import MeteredDocker, recognition_stages, atomic, safe_error
    import runtime_session as runtime
    import input_binding
    import evidence_transport as transport
    def observe(stage):
        _observe(observer,stage)
    started=time.monotonic()
    def interrupt(_signum,_frame): raise TimeoutError('fixed host ceiling or interruption')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM): signal.signal(sig,interrupt)
    signal.setitimer(signal.ITIMER_REAL,c.LIMITS['host_seconds'])
    observe('workspace')
    root=c.durable_mkdir(Path(env['RUNNER_TEMP'])/(c.EXPERIMENT+'-'+str(run['run_id'])))
    # Frozen public control plane and source failures remain packageable before claim.
    observe('input_authorities')
    frozen_input,authority=input_authorities(release,freeze)
    screening_policy_sha=screening_policy_identity(freeze)['sha256']
    projection=projection_for(frozen_input['actual_new_cells'])
    job,plan=frozen_input['actual_blind_job'],frozen_input['actual_opaque_order']
    binding=frozen_input['actual_worker_binding']
    receipt=host_receipt(run,freeze_sha,tree,release['actual_screening_receipt_sha256'])
    write_once(root/'run-preflight.json',preflight)
    write_once(root/'input-freeze.json',frozen_input)
    write_once(root/'public-screening-receipt.json',frozen_input['actual_screening_receipt'])
    write_once(root/'new-cell-projection.json',projection)
    c.durable_mkdir(root/'authority'); write_once(root/'authority/execution-freeze.json',freeze)
    observe('source_stage')
    runtime.stage_source(freeze['host_code'],root/'source')
    write_once(root/'source'/NEW/'execution-freeze.json',freeze)
    atomic(root/'host-receipt.json',receipt)
    docker=None; successful=[]; failed=False
    try:
        observe('input')
        receipt['stage']='input'; atomic(root/'host-receipt.json',receipt)
        require(shutil.disk_usage(root).free>=c.LIMITS['initial_free_bytes'],'12GiB initial free reserve before acquisition')
        bound,data_accounting=verify_download(release,frozen_input,authority,expected_policy_sha256=screening_policy_sha)
        receipt['input_verified']=True
        material=input_binding.stage_worker_inputs(root/'input',bound['verification'],bound['eligible_audio'])
        c.durable_mkdir(root/'order')
        write_once(root/'order/execution-plan.json',plan)
        write_once(root/'order/worker-input-binding.json',binding)
        observe('source_claim')
        claim=source_claim(run,release,preflight,freeze_sha)
        claim_sha=write_once(root/'authority/source-claim.json',claim)
        receipt['source_claim_sha256']=claim_sha
        acquisition={'schema':'k1-asr18-reserved-acquisition-v1','data':data_accounting,
            'asr_response_body_bytes':None,'whisper':None,
            'reserved_total_bytes':c.LIMITS['acquisition_reserved_bytes'],'cap_bytes':6*c.GIB,'hard_wire_byte_cap':False}
        atomic(root/'acquisition-receipt.json',acquisition)
        docker=MeteredDocker(root,started+c.LIMITS['host_seconds'])
        observe('setup')
        receipt['stage']='setup'; receipt['stages']['setup']='started'; atomic(root/'host-receipt.json',receipt)
        setup=runtime.setup_runtimes(root,freeze,docker,root/'input',run)
        receipt['combined_setup_seconds']=setup['seconds']; receipt['stages']['setup']='complete'
        acquisition.update(asr_response_body_bytes=setup['asr_body_bytes'],whisper=setup['whisper'])
        atomic(root/'acquisition-receipt.json',acquisition)
        c.durable_mkdir(root/'output'); c.durable_mkdir(root/'frozen'); c.durable_mkdir(root/'policy')
        def worker(family,mode,seconds):
            out=c.durable_mkdir(root/'output'/family)
            name=c.EXPERIMENT+'-'+str(run['run_id'])+'-'+family+'-'+mode
            args=runtime.worker_arguments(name=name,image=setup['image'],code=setup['code'],
                runtime=setup['runtimes'][family],output=out,inputs=root/'input',family=family,mode=mode,
                order=root/'order',authority=root/'authority',run_identity=run,freeze_sha=freeze_sha,claim_sha=claim_sha,
                frozen=root/'frozen'/family if mode=='decode' else None,policy=root/'policy' if mode=='decode' else None)
            docker.stage='setup' if mode=='prepare' else family
            runtime.supervise(docker,args,out,family=family,mode=mode,deadline=min(docker.deadline,time.monotonic()+seconds))
            validate_worker_complete(out,family,mode,job,plan)
        def prepare(family,seconds):
            observe('prepare_'+family)
            receipt['stage']='prepare'; receipt['stages'][family+'_prepare']='started'; atomic(root/'host-receipt.json',receipt)
            worker(family,'prepare',seconds)
            receipt['stages'][family+'_prepare']='complete'; atomic(root/'host-receipt.json',receipt)
        def freeze_policy():
            import worker_receipts
            observe('policy_freeze')
            receipt['stage']='policy_freeze'
            policy=worker_receipts.build_policy({family:root/'output'/family for family in c.FAMILIES})
            policy_sha=write_once(root/'policy/policy.json',policy)
            write_once(root/'policy/policy-freeze.json',{'schema':'k1-asr18-all3-policy-freeze-v1','identity':run,
                'recognizer_policy_sha256':policy_sha,'native_calls_started':0,
                'prepared_config_files':{family:c.identity(root/'output'/family/'native-config.json') for family in c.FAMILIES}})
            receipt['all3_policy_frozen_before_recognition']=True; receipt['stage']='capacity_proof'
            atomic(root/'host-receipt.json',receipt)
            observe('capacity_proof')
            proof=transport.package_budget_preflight(root,run,job=job,execution_plan=plan,input_binding=binding,policy_path=root/'policy/policy.json')
            c.validate_decode_proof(proof,job=job,execution_plan=plan,input_binding=binding,policy_path=root/'policy/policy.json',family='qwen06')
            write_once(root/'policy/budget-proof.json',proof)
            for family in c.FAMILIES:
                (root/'output'/family).rename(root/'frozen'/family)
                # Canonical worker files already use read-only durable writes.
                directory=root/'frozen'/family
                for name in _walk_files(directory):
                    os.chmod(directory/name,0o444,follow_symlinks=False)
        def decode(family,seconds):
            observe('decode_'+family)
            receipt['stage']='decode'; receipt['stages'][family+'_decode']='started'; atomic(root/'host-receipt.json',receipt)
            worker(family,'decode',seconds)
            successful.append(family)
            receipt['stages'][family+'_decode']='complete'; atomic(root/'host-receipt.json',receipt)
        recognition_stages(prepare,freeze_policy,decode,setup_seconds=setup['seconds'],record=receipt)
        receipt['status']='all_planned_native_complete'
    except BaseException as error:
        if observer is not None:
            try:observer.failure(error)
            except BaseException:pass
        failed=True; receipt['status']='stopped_no_retry'; receipt['primary_error']=safe_error(error)
        for key,value in receipt['stages'].items():
            if value=='started': receipt['stages'][key]='failed'
    finally:
        observe('cleanup')
        receipt['stage']='cleanup'
        if docker is not None:
            try: docker.finish(); receipt['cleanup_verified']=True
            except BaseException as error:
                if observer is not None:
                    try:observer.failure(error)
                    except BaseException:pass
                failed=True; receipt['status']='cleanup_unconfirmed'; receipt['cleanup_error']=safe_error(error)
        else: receipt['cleanup_verified']=True
        observe('terminal_accounting')
        terminal=terminal_accounting(root,projection,job,plan,run,successful)
        receipt['accounting_valid']=terminal['accounting_valid']; failed |= not terminal['accounting_valid']
        if failed and receipt['status']=='all_planned_native_complete': receipt['status']='stopped_no_retry'
        observe('terminal_write')
        write_once(root/'terminal-outcomes.json',terminal)
        receipt['wall_seconds']=time.monotonic()-started; receipt['stage']='package'
        observe('host_receipt_write')
        atomic(root/'host-receipt.json',receipt)
        observe('package_build')
        payload,_=transport.build_package(root,run)
        observe('package_write')
        transport.write_package(env['RUNNER_TEMP'],payload,run)
        # No audio/text/paths, arbitrary exceptions, tokens or raw container logs.
        print(json.dumps({'schema':'k1-asr18-local-evidence-ready-v1','archive_bytes':len(payload),
            'archive_sha256':digest(payload),'native_run_failed':bool(failed),
            'publication':'pending_separate_trusted_host_step'},sort_keys=True),flush=True)
        signal.setitimer(signal.ITIMER_REAL,0)
    return int(failed)


def main(argv=None):
    from failure_observer import Observation
    observer=Observation(role='host')
    try:
        # First operational gate remains before event/Git/API/claim/model/token.
        _observe(observer,'release')
        release=release_first()
        _observe(observer,'arguments')
        require((sys.argv[1:] if argv is None else argv)==[],'fixed source entry; no alternate config path')
        env=dict(os.environ)
        _observe(observer,'event_read')
        event=read_json(Path(env['GITHUB_EVENT_PATH']),c.MIB)
        _observe(observer,'context')
        run=event_identity(env,event)
        observer.attach(run,env['RUNNER_TEMP'])
        _observe(observer,'api_preflight')
        preflight=api_preflight(PublicAPI(),run,release)
        _observe(observer,'source_authority')
        freeze,freeze_sha,tree=source_authority(run,release)
        result=execute(env,run,release,freeze,freeze_sha,tree,preflight,observer=observer)
    except BaseException as error:
        try:observer.failure(error)
        except BaseException:pass
        try:observer.finish(success=False)
        except BaseException:pass
        raise
    _observe(observer,'complete')
    recorded=observer.finish(success=result==0)
    return result if recorded else 2


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        print('{"status":"blocked_no_retry"}')
        raise SystemExit(2)
