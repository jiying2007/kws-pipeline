"""Qwen12 prospective native admission: stdlib-only, strict new scope."""
from pathlib import Path
import hashlib
import io
import json
import os
import re
import stat
import wave

EXPERIMENT = 'qwen12-native-admission-v1'
BRANCH = 'research/qwen12-native-admission-remaining35-v1'
# Fixed public identities only. This is not a user-selectable resume cursor.
PUBLIC_INPUTS_SHA256 = 'cd56eb66043a487c151107445bdccbe627f958110b142bea341bf221193e14fe'
PUBLIC_INPUTS_CANONICAL_SHA256 = '5bac33659f146b27f0f1b92f5adf50a38d5fccea4d4882bad693e953c8f893ac'
ORIGINAL_JOB_SHA256 = '44a82464e5d7808bb7636b7a050356cf66767040f6737b264e9e7016c58fc404'
ORIGINAL_JOB_BYTES = 2415
COVERAGE_CONTRACT_SHA256 = '83a0b0473049ca1b784274be5703ff38afd5e7d6922277350db3732265a2b128'
ORIGINAL_SOURCES = (('clip-000001',
  '02e864a17e72e72ad62d13f6c04b18788a59ef08d121f6a676c89ff1b39cc915',
  'a3829a5a2a25aa4adee963582ff19a0b850629b3c1db5aedf9bba0816fcf4bae',
  35840),
 ('clip-000002',
  '06fd2d9f2186ad9c7d4386bbbfe13241a141bc792c7ea8e634b74cadea5e2980',
  '6db96131a891c95e12769defa9bee0c55ab74316c0da45d138626ca2301aeea4',
  25600),
 ('clip-000003',
  '3660c72e7db7392d4a430b71cc224b65527037129ef45d4387627dc6e262ef25',
  '636705536d4d78fc892cc7bbfae3dca227b26d494157b3c480d63268c5cd8631',
  33280),
 ('clip-000004',
  '514bca4b4dc77a10dac3d66f38231dbd7203e193fa5d515857db6826813bcda5',
  'f2c212d2a54f124535a6ac7ce6e2fc272a99dd878030c8765064dca8f30ec037',
  30720),
 ('clip-000005',
  '7100eaf6994960cee1d5da192020ffcdfae367d17b401c9b9efbacf7172a5ddd',
  'd56f7352a238d43e30a1019d1fc8b8bb3064d5e671556519af54b8386119002a',
  46080),
 ('clip-000006',
  '7c091117d6ef772c960bd818c4188a1ece8db9219f540d35ecb6850493697af4',
  '86e51482ca5b41c81e7c10e3eaf52df4be6bb3979e0725ed328dc35c5c085d47',
  28160),
 ('clip-000007',
  'a9ddaa06fa6133cd9acdb37a761b0e3cc6b66e48a18c82a97b159738bdbc8229',
  '1f23e6b5c70a59b93142dddea93dc86765657fe7f3f42804a94cd970dfe511da',
  40960),
 ('clip-000008',
  'ae352e520daff27d394d726e3f35ea46b1745c6d8c0a335685e1f9b770227e20',
  '366f8ea0c0e0d31103bcbdc42675481a4827be8b2991e1ed691c4ee7fb3010af',
  25600),
 ('clip-000009',
  'b01a04cc168f8c9de314c9ddb22f3f191e6c0a7138ac78c3cf63831d2bbbf26e',
  '423b2c1b9502a6a877fba41e24f18b8e965e2db0ac78c8a9a8832d601d5743bc',
  32000),
 ('clip-000010',
  'b6902e14b699ca504f34d7b4bdba66a4c9805a0fd3403f5633cbb891872d9d51',
  '61dcec8fc60753686a363a3ace40109a26df37cf556733c046b8c6cbc757ce02',
  29440),
 ('clip-000011',
  'db85999b9d7a36f2a8633a58e5279c2ed5dda8f61f8bf7cf45c72beef6871261',
  '4d5db9285d3bd10fae13345b4ccd3ad081b748257eff03d1c01e211865b7926d',
  29440),
 ('clip-000012',
  'f96edce564f94ea7919c6de63613e88fe89ae07f01ad907097c4af095c3685db',
  '453ad37fc9203ea9796b6933bd538a331944cb140218e39035ead2682b320d8d',
  30720))
# Copies are returned to callers; retained originals are immutable canonical bytes.
_PREDECESSOR_BYTES = b'[{"archive_bytes":1597173,"archive_sha256":"1992682c1a1301a43c06143db0fd034a9ed976252f55bef4c32799fff13dae13","completed":0,"error":0,"evidence_commit":"8e05b507e8239ed70191c56ff13216690d6c3049","executed_head_sha":"3528118f3cdedf391519a5f8c3365ac15eac16a8","job_id":114515750086,"job_key":"qwen12","not_run":36,"repository":"jiying2007/kws-pipeline","run_attempt":1,"run_id":38154869005,"source_freeze_sha256":"f6d24914a930a94776fa4051f54c3d4b87569ff355d1ffa687016c8f6e6988e5","source_tree_sha":"3b754bcc98511971c1c21382f6a48d01dab38e9c","started":0,"started_members":[],"terminal_outcomes":{"bytes":12373,"path":"terminal-outcomes.json","sha256":"d86b904ad806cd614a65e20bc3d7e2193ae6758a683307ed4eda40ea945f03dc"}},{"archive_bytes":86754,"archive_sha256":"9b063d2c92ccf88d67678b032d391bf01e0ceb1a626d5cf23a3f11c0fd250d39","completed":0,"error":0,"evidence_commit":"d843cc59113c7fd3ac89a0b4be9db71e5950ea20","executed_head_sha":"e9297163a21f8f7c2f960301ff05f1894617d2cf","job_id":114531695348,"job_key":"qwen12","not_run":36,"repository":"jiying2007/kws-pipeline","run_attempt":1,"run_id":38160295526,"source_freeze_sha256":"4b4fcb7dbabde3f4306cb0c363b7298be2c66382bd38359c6701d29a3a282342","source_tree_sha":"20e9ff3b33b894807ee1973810afb7645df5d7bd","started":0,"started_members":[],"terminal_outcomes":{"bytes":12373,"path":"terminal-outcomes.json","sha256":"d86b904ad806cd614a65e20bc3d7e2193ae6758a683307ed4eda40ea945f03dc"}},{"archive_bytes":2656120,"archive_sha256":"140a18c7429b16674b5a1dc8e1f65f4a1be6b0dd4ecb324b0a4c1ccaff11d157","completed":0,"error":0,"evidence_commit":"43113646c12fc9d6c112481bab6818c96f2ce789","executed_head_sha":"42b333f8b96c4c9fcca63c9e275d26a4519737c8","job_id":114551789796,"job_key":"qwen12","not_run":36,"repository":"jiying2007/kws-pipeline","run_attempt":1,"run_id":38167052146,"source_freeze_sha256":"ee821f44e1ac4bba36089eea751b5f6ae075b8c7bf4a614c49a7e4f21b50cc96","source_tree_sha":"6e91e77b1e60e19956664da2eab80f0a929e02af","started":0,"started_members":[],"terminal_outcomes":{"bytes":12373,"path":"terminal-outcomes.json","sha256":"d86b904ad806cd614a65e20bc3d7e2193ae6758a683307ed4eda40ea945f03dc"}},{"archive_bytes":4210336,"archive_sha256":"9d371985530e8d5e32c915b8853f6e945ded1bfdbb6dad0b53fcbba4772e5bf5","completed":0,"error":1,"evidence_commit":"8c25805e226ca23d55bd698b52ba5b45acd9195a","executed_head_sha":"80009f412987aeaaaf401894677299724ed0490d","job_id":114572506731,"job_key":"qwen12","not_run":35,"repository":"jiying2007/kws-pipeline","run_attempt":1,"run_id":38173984595,"source_freeze_sha256":"2888f0a1a60abf11e3c8a12caf71085a859fb7a3636f6825b163afdf6809060c","source_tree_sha":"b6722270d54de525556c602b92b1a29ce9ebd0ef","started":1,"started_members":[{"bytes":930,"path":"output/qwen06/clip-000001/started.json","sha256":"0f8886197d615cdd2cc240cff5f46e194380950856d82e495eb3458f019a6c38"}],"terminal_outcomes":{"bytes":12371,"path":"terminal-outcomes.json","sha256":"04fa63ff6dbc12f3e64934ba0ed708ea74d994d7375b9b648dd4201cd45fe45f"}}]'
_HISTORICAL_BYTES = b'{"archive_sha256":"9d371985530e8d5e32c915b8853f6e945ded1bfdbb6dad0b53fcbba4772e5bf5","audio_frames":35840,"audio_id":"clip-000001","evidence_commit":"8c25805e226ca23d55bd698b52ba5b45acd9195a","executed_head_sha":"80009f412987aeaaaf401894677299724ed0490d","family":"qwen06","job_id":114572506731,"model_lock_sha256":"051c0091294246642882676c1d6e26ecb4508436392c2ad1dcaec56ee5f0a1da","native_decode_config_sha256":"70828c08d53241e2170e5483137d9d518935e5358103b7941cf5ec302be927fb","pcm_sha256":"a3829a5a2a25aa4adee963582ff19a0b850629b3c1db5aedf9bba0816fcf4bae","recognizer_policy_sha256":"fa697f0e93852716d7309e6206bd056eb0135099d58af49f3236e1af8842238e","request_id":"qwen12-38173984595-qwen06-clip-000001","run_attempt":1,"run_id":"38173984595","source_id":"qwen12-02e864a17e72e72ad62d13f6c04b18788a59ef08d121f6a676c89ff1b39cc915","source_lock_sha256":"8a8ce266b03288f370797df3fc628349c468e9804180b3f3d7b6a3773e3af2b3","started_member":{"bytes":930,"path":"output/qwen06/clip-000001/started.json","sha256":"0f8886197d615cdd2cc240cff5f46e194380950856d82e495eb3458f019a6c38"},"status":"error","wav_sha256":"02e864a17e72e72ad62d13f6c04b18788a59ef08d121f6a676c89ff1b39cc915"}'
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
MIB = 1024**2
GIB = 1024**3
LIMITS = {'clips': 12, 'campaign_slots': 36, 'historical_starts': 1, 'new_starts': 35, 'cumulative_starts': 36, 'workspace_bytes': 10*GIB,
          'initial_free_bytes': 12*GIB, 'acquisition_reserved_bytes': 4*GIB+1536*MIB+128*MIB+MIB,
          'acquisition_cap_bytes': 6*GIB, 'output_bytes': 32*MIB,
          'setup_seconds': 1800, 'engine_seconds': 600, 'clip_seconds': 120,
          'host_seconds': 3600, 'workflow_minutes': 70,
          'cpu_threshold_seconds': {'setup': 3600, 'qwen06': 600, 'sensevoice': 600, 'whisper': 1800},
          'cpu_total_threshold_seconds': 6600, 'cpu_poll_seconds': 0.2,
          'kill_command_seconds': 5}
BOUNDARY = {'scope': EXPERIMENT, 'training_admitted': False, 'human_gold': False,
            'shipping_approved': False, 'acoustic_verbatim_completeness': 'unknown'}
SOURCE_COMMIT = '7af8f8597b0b7fbe8761c9a2400f2e4028395551'
DISPATCH_APPROVALS = ('enabled', 'new_transcripts_publication_approved',
    'budget_review_approved', 'source_review_approved', 'independent_review_approved',
    'bounded_evidence_transport_readback_approved', 'prior_campaign_history_and_unique_claim_approved')
# A fixed obligation roster, never declarations that future runtime work passed.
REQUIRED_RUNTIME_PRECONDITIONS = (
    'current_run_host_capacity_and_kernel_scope_checks',
    'current_run_both_runtime_installs_before_prepare',
    'current_run_all_three_prepare_only_loads_with_kernel_cpu_guards_and_zero_audio_forward',
    'current_run_full_policy_and_prepare_configs_frozen_before_first_audio_forward',
    'current_run_actual_complete_closure_worsttrace_dry_package_before_first_audio_forward',
    'each_fresh_decode_worker_kernel_cpu_guard_and_frozen_config_recheck_before_its_first_audio_forward')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def exact_json(value, expected):
    """Exact fixed JSON values, including container and boolean/integer types."""
    if type(value) is not type(expected):return False
    if type(expected) is dict:
        return set(value)==set(expected) and all(type(key) is str and exact_json(value[key],expected[key]) for key in expected)
    if type(expected) is list:
        return len(value)==len(expected) and all(exact_json(a,b) for a,b in zip(value,expected))
    return value==expected


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path, cap=32*MIB):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'path alias')
    s = path.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= cap, 'regular bounded file')
    return s


def read_json(path, cap=32*MIB):
    regular(path, cap)
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key'); out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def write_once(path, value):
    raw = canonical(value); require(len(raw) <= LIMITS['output_bytes'], 'JSON output bound')
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return digest(raw)


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    raw = canonical(value); require(len(raw) <= LIMITS['output_bytes'], 'JSON output bound')
    with tmp.open('wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(path):
    return {'bytes': regular(path).st_size, 'sha256': digest(Path(path).read_bytes())}


def validate_public(value):
    require(type(value) is dict and set(value) == {'schema', 'count', 'audio_bytes', 'frames',
        'sample_rate_hz', 'channels', 'sample_width_bytes', 'records'}, 'public fields only')
    require(value['schema'] == 'qwen12-existing-public-audio-v1' and value['count'] == 12 and
        value['audio_bytes'] == 776208 and value['frames'] == 387840 and
        (value['sample_rate_hz'], value['channels'], value['sample_width_bytes']) == (16000,1,2), 'exact12 corpus')
    rows = value['records']; require(type(rows) is list and len(rows) == 12, 'exact12 records')
    hashes = []
    for r in rows:
        require(type(r) is dict and set(r) == {'repository','commit','path','bytes','sha256',
            'git_blob_sha1','raw_url','pcm_sha256','frames'}, 'public metadata projection')
        require(r['repository'] == 'jiying2007/kws-data' and r['commit'] == SOURCE_COMMIT, 'existing public source')
        require(re.fullmatch(r'datasets/qwen3-reviewed-v1/audio/[A-Za-z0-9_-]+\.wav', r['path']), 'public WAV path')
        require(r['raw_url'] == 'https://raw.githubusercontent.com/'+r['repository']+'/'+r['commit']+'/'+r['path'], 'exact source URL')
        require(all(type(r[k]) is str and re.fullmatch('[a-f0-9]{64}',r[k]) for k in ('sha256','pcm_sha256'))
                and re.fullmatch('[a-f0-9]{40}',r['git_blob_sha1']), 'source hashes')
        require(type(r['frames']) is int and 0 < r['frames'] <= 192000 and
                type(r['bytes']) is int and r['bytes'] == 44+2*r['frames'], 'full canonical WAV frames')
        hashes.append(r['sha256'])
    require(hashes == sorted(set(hashes)) and sum(r['bytes'] for r in rows) == value['audio_bytes']
            and sum(r['frames'] for r in rows) == value['frames'], 'ordered unique exact audio')
    require(digest(canonical(value)) == PUBLIC_INPUTS_CANONICAL_SHA256, 'exact original public inputs')
    return rows


def validate_wav(raw, row):
    require(type(raw) is bytes and len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'WAV file identity')
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob_sha1'], 'public Git object')
    with wave.open(io.BytesIO(raw), 'rb') as w:
        require((w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype(),w.getnframes()) ==
            (1,2,16000,'NONE',row['frames']), 'full PCM framing')
        pcm = w.readframes(w.getnframes()); require(w.readframes(1) == b'', 'WAV EOF')
    require(len(pcm) == 2*row['frames'] and digest(pcm) == row['pcm_sha256'], 'full PCM identity')
    return pcm


def blind_job(rows):
    return {'schema':'blind-source-screen32-job-v1','clips':[
        {'audio_id':f'clip-{i:06d}','audio_path':'audio/'+r['sha256']+'.wav','wav_sha256':r['sha256']}
        for i,r in enumerate(rows,1)]}


def original_job():
    return {'schema': 'blind-source-screen32-job-v1', 'clips': [
        {'audio_id': audio_id, 'audio_path': 'audio/'+wav+'.wav', 'wav_sha256': wav}
        for audio_id, wav, pcm, frames in ORIGINAL_SOURCES]}


def validate_original_job(job):
    raw = canonical(job)
    require(len(raw) == ORIGINAL_JOB_BYTES and digest(raw) == ORIGINAL_JOB_SHA256
            and exact_json(job,original_job()), 'exact original anonymous job required')
    return job


def historical_slot():
    return json.loads(_HISTORICAL_BYTES)


def slot_key(row):
    return (row['family'], row['wav_sha256'], row['pcm_sha256'])


def original_slots(job):
    validate_original_job(job)
    return [{'family': family, 'audio_id': audio_id, 'wav_sha256': wav,
             'pcm_sha256': pcm, 'audio_frames': frames, 'source_id': 'qwen12-'+wav}
            for family in FAMILIES for audio_id, wav, pcm, frames in ORIGINAL_SOURCES]


def remaining_plan(job, family=None):
    require(family is None or family in FAMILIES, 'fixed family required')
    historical = historical_slot()
    rows = [row for row in original_slots(job) if slot_key(row) != slot_key(historical)]
    require(len(rows) == 35 and len({slot_key(row) for row in rows}) == 35
            and {f: sum(row['family'] == f for row in rows) for f in FAMILIES}
                == {'qwen06': 11, 'sensevoice': 12, 'whisper': 12}, 'fixed remaining35 roster')
    return rows if family is None else [row for row in rows if row['family'] == family]


def remaining_slot_contract():
    return {'schema': 'qwen12-fixed-remaining35-v1', 'executions': json.loads(_PREDECESSOR_BYTES),
            'public_inputs_sha256': PUBLIC_INPUTS_SHA256, 'original_job_sha256': ORIGINAL_JOB_SHA256,
            'original_job_bytes': ORIGINAL_JOB_BYTES, 'source_repository': 'jiying2007/kws-data',
            'source_commit': SOURCE_COMMIT, 'coverage_contract_sha256': COVERAGE_CONTRACT_SHA256,
            'historical_consumed': [historical_slot()], 'planned_new_slots': remaining_plan(original_job()),
            'campaign_slots': 36, 'historical_starts': 1, 'maximum_new_starts': 35,
            'maximum_cumulative_starts': 36}


def planned_clips(job, family):
    allowed = remaining_plan(job, family)
    return [dict(job['clips'][int(row['audio_id'].split('-')[1])-1]) for row in allowed]


def validate_start_slot(job, family, *, audio_id, measured, source_id, request_id, run_id):
    # Check exclusion before aliases: a different PCM declaration is identity drift,
    # never permission to reuse the historical WAV under another logical key.
    historical = historical_slot()
    require(not (family == historical['family'] and measured['file_sha256'] == historical['wav_sha256']),
            'historical consumed slot cannot be attempted')
    rows = [row for row in remaining_plan(job, family) if row['audio_id'] == audio_id]
    require(len(rows) == 1, 'original allowed audio ID required')
    row = rows[0]
    require(canonical(measured) == canonical({'file_sha256': row['wav_sha256'],
            'pcm_sha256': row['pcm_sha256'], 'audio_frames': row['audio_frames']})
            and source_id == row['source_id'], 'original WAV PCM frames source binding required')
    require(request_id == 'qwen12-'+run_id+'-'+family+'-'+audio_id, 'exact original slot request identity required')
    return row


def validate_blind(root):
    root = Path(root); require(root.is_dir() and not root.is_symlink(), 'blind root')
    require({p.name for p in root.iterdir()} == {'job.json','audio'}, 'no source mapping or labels')
    job = read_json(root/'job.json',65536)
    require(set(job) == {'schema','clips'} and job['schema']=='blind-source-screen32-job-v1' and
        type(job['clips']) is list and len(job['clips']) == 12, 'blind exact12')
    validate_original_job(job)
    require((root/'job.json').read_bytes() == canonical(job), 'canonical original job bytes')
    hashes = []
    for i,r in enumerate(job['clips'],1):
        require(type(r) is dict and set(r)=={'audio_id','audio_path','wav_sha256'}, 'blind fields')
        h=r['wav_sha256']; require(type(h) is str and re.fullmatch('[0-9a-f]{64}',h), 'blind WAV hash')
        require(r['audio_id']==f'clip-{i:06d}' and r['audio_path']=='audio/'+h+'.wav', 'anonymous order')
        p=root/r['audio_path']; regular(p,MIB); require(digest(p.read_bytes())==h,'blind WAV bytes')
        hashes.append(h)
    require(hashes == sorted(set(hashes)) and not (root/'audio').is_symlink() and
        {p.name for p in (root/'audio').iterdir()} == {h+'.wav' for h in hashes}, 'exact blind members')
    return job


def fetch_blind(value, root, opener):
    """Injected no-retry HTTPS opener; one exact public body per audio."""
    rows=validate_public(value); root=Path(root); root.mkdir(mode=0o700); (root/'audio').mkdir(mode=0o700)
    seen=0
    for row in rows:
        with opener.open(row['raw_url'],timeout=30) as response:
            require(response.status==200 and response.geturl()==row['raw_url'], 'public audio no redirect')
            require(response.headers.get('Content-Encoding','identity')=='identity','identity audio response')
            raw=response.read(row['bytes']+1); seen+=len(raw)
            require(seen<=MIB,'audio acquisition reservation')
        validate_wav(raw,row)
        with (root/'audio'/(row['sha256']+'.wav')).open('xb') as f: f.write(raw)
    write_once(root/'job.json',blind_job(rows)); validate_blind(root)
    return {'audio_body_bytes':seen,'audio_attempts':12,'retry_count':0}


def fresh_run(env, event, release):
    require(type(release) is dict and all(release.get(key) is True for key in DISPATCH_APPROVALS),
            'release disabled or missing explicit dispatch approval')
    preconditions = release.get('required_runtime_preconditions')
    require(type(preconditions) is list and all(type(value) is str for value in preconditions)
            and preconditions == list(REQUIRED_RUNTIME_PRECONDITIONS)
            and release.get('evidence_transport') == 'github-repository-zip-chunks-v1'
            and 'activation_blockers' not in release
            and 'maximum_log_api_readback_approved' not in release, 'fixed required runtime preconditions or evidence transport drift')
    require('predecessor_zero_recognition' not in release and
            exact_json(release.get('remaining_slot_contract'), remaining_slot_contract()),
            'exact immutable predecessor and remaining35 contract required')
    require(env.get('GITHUB_REPOSITORY')=='jiying2007/kws-pipeline' and
        env.get('GITHUB_REF')=='refs/heads/'+BRANCH and env.get('GITHUB_EVENT_NAME')=='push' and
        env.get('GITHUB_RUN_ATTEMPT')=='1' and re.fullmatch('[1-9][0-9]*',env.get('GITHUB_RUN_NUMBER','')) and
        re.fullmatch('[1-9][0-9]{0,19}',env.get('GITHUB_RUN_ID','')) and
        re.fullmatch('[0-9a-f]{40}',env.get('GITHUB_SHA','')), 'first unique real Actions run')
    require(int(env['GITHUB_RUN_ID']) not in {row['run_id'] for row in json.loads(_PREDECESSOR_BYTES)},
            'historical execution cannot be reused')
    require(event.get('created') is True and event.get('repository',{}).get('private') is False and
            event.get('after')==env['GITHUB_SHA'], 'new public branch only')
    return {'run_id':env['GITHUB_RUN_ID'],'head_sha':env['GITHUB_SHA'],'run_attempt':1,'run_number':int(env['GITHUB_RUN_NUMBER'])}


def validate_output_members(job, outputs, family=None):
    """Reject unexpected or historical per-call output rather than ignoring it."""
    allowed = {(row['family'], row['audio_id']) for row in remaining_plan(job)}
    root = Path(outputs)
    require(root.is_dir() and not root.is_symlink(), 'unaliased output root')
    require(family is None or family in FAMILIES, 'fixed output family')
    directories=[(family,root)] if family is not None else [(p.name,p) for p in root.iterdir()]
    shared={'prepared.json','run-identity.json','kernel-scope.json','cpu-process-guard.json','blind-job.json',
            'progress.json','contract.json','decoder-inputs.json','model-load.json','native-source-lock.json',
            'native-asset-lock.json','native-config.json','native-resolved-generation-config.json',
            'native-frontend-config.json','optional-tool-probes.json','load-receipt.json',
            'worker-complete.json','worker-failure.json'}
    clip_files={'started.json','completed.json','raw-output.json','raw-native-output.json',
                'raw-transcript.json','raw-native-row.json','input-capture.json','pre-finalize.json'}
    observations={'.native-observation.json','.native-generation-return.json','.native-tokenizer-input.json',
                  '.native-inference-return.json','.native-returned-raw.json'}
    for family_name,directory in directories:
        require(family_name in FAMILIES and directory.is_dir() and not directory.is_symlink(),
                'unknown output family')
        for path in directory.iterdir():
            require(not path.is_symlink(), 'output member alias')
            if path.is_dir() and path.name not in ('retained-source', 'retained-assets'):
                require((family_name, path.name) in allowed, 'unknown or historical output slot')
                for member in path.iterdir():
                    require(member.is_file() and not member.is_symlink() and
                            (member.name in clip_files or re.fullmatch(r'update-(?:00[1-9]|0[1-9][0-9]|[1-3][0-9]{2}|4[0-3][0-9]|44[0-8])\.json',member.name)),
                            'unknown per-call output member')
            if path.name.startswith('clip-'):
                audio_id = path.name.split('.', 1)[0]
                require((family_name, audio_id) in allowed, 'unknown or historical output slot')
                if path.is_file():require(path.name[len(audio_id):] in observations,'unknown observation output member')
            elif path.is_file():require(path.name in shared,'unknown family output member')


def denominator(job, outputs, *, isolated_family=None):
    """Cumulative36 accounting: one immutable old error, at most35 new attempts."""
    validate_output_members(job, outputs, isolated_family)
    result=[]
    historical=historical_slot()
    for slot in original_slots(job):
        family=slot['family'];audio_id=slot['audio_id'];wav=slot['wav_sha256']
        row={**slot, 'attempts':0, 'status':'not_run', 'origin':'current_run',
             'source_resolver_admission':False, **BOUNDARY}
        if slot_key(slot)==slot_key(historical):
            row.update(attempts=1,status='error',origin='historical',actual_text=None,
                       historical_evidence=historical)
            result.append(row);continue
        p=(Path(outputs) if family==isolated_family else Path(outputs)/family)/audio_id
        if isolated_family is not None and family != isolated_family:
            result.append(row);continue
        start=p/'started.json';done=p/'completed.json';raw=p/'raw-transcript.json'
        if p.exists() and any(p.iterdir()):
            row.update(attempts=1,status='error')
            if start.exists() and done.exists() and raw.exists():
                s=read_json(start);d=read_json(done)
                if (s.get('schema')=='retained-asr-started-v1' and s.get('file_sha256')==wav
                    and s.get('pcm_sha256')==slot['pcm_sha256'] and type(s.get('audio_frames')) is int
                    and s['audio_frames']==slot['audio_frames'] and s.get('audio_id')==audio_id
                    and s.get('source_id')==slot['source_id'] and s.get('recognizer_id')==family
                    and s.get('request_id')=='qwen12-'+str(s.get('run_id'))+'-'+family+'-'+audio_id
                    and s.get('original_job_sha256')==ORIGINAL_JOB_SHA256
                    and s.get('remaining_plan_sha256')==digest(canonical(remaining_plan(job,family)))
                    and d.get('started_sha256')==identity(start)['sha256']
                    and d.get('raw_output_sha256')==identity(raw)['sha256']
                    and all(canonical(d.get(key))==canonical(value) for key,value in s.items() if key!='schema')
                    and d.get('schema')=='retained-asr-completed-v1' and d.get('status')=='success'
                    and d.get('decoder_completed') is True):
                    row['status']='native_v2_complete'
        result.append(row)
    require(len(result)==36 and sum(r['attempts'] for r in result)<=36,'full cumulative denominator bound')
    return result


def current_family_counts(job, outputs, family, *, isolated=False):
    rows=[r for r in denominator(job,outputs,isolated_family=family if isolated else None) if r['family']==family and r['origin']=='current_run']
    return {'calls':sum(r['attempts'] for r in rows),
            'completed':sum(r['status']=='native_v2_complete' for r in rows),
            'audio_ids':[r['audio_id'] for r in rows if r['status']=='native_v2_complete']}


def terminal_summary(rows):
    current=[row for row in rows if row['origin']=='current_run']
    complete=sum(row['status']=='native_v2_complete' for row in current)
    triplets=sum(all(any(row['audio_id']==audio_id and row['family']==family and
                        row['status']=='native_v2_complete' for row in rows) for family in FAMILIES)
                 for audio_id, wav, pcm, frames in ORIGINAL_SOURCES)
    return {'campaign_slots':36,'historical_started':1,'historical_errors':1,
            'current_started':sum(row['attempts'] for row in current),'current_completed':complete,
            'current_errors':sum(row['status']=='error' for row in current),
            'not_run':sum(row['status']=='not_run' for row in current),
            'cumulative_started':1+sum(row['attempts'] for row in current),
            'complete_triplets':triplets,'planned35_complete':complete==35,
            'training_admitted':False,'source_resolver_admission':False}
