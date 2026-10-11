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
BRANCH = 'research/qwen12-native-admission-recovery2'
# Both consumed public predecessors remain bound; neither identity may be reused.
# Bind public zero-recognition records; do not infer historical failure causes.
PREDECESSOR_ZERO_RECOGNITION = {
    'executions': [
        {'run_id': 38154869005, 'run_attempt': 1,
         'executed_head_sha': '3528118f3cdedf391519a5f8c3365ac15eac16a8',
         'evidence_commit': '8e05b507e8239ed70191c56ff13216690d6c3049',
         'archive_sha256': '1992682c1a1301a43c06143db0fd034a9ed976252f55bef4c32799fff13dae13',
         'started': 0, 'completed': 0, 'not_run': 36},
        {'run_id': 38160295526, 'run_attempt': 1,
         'executed_head_sha': 'e9297163a21f8f7c2f960301ff05f1894617d2cf',
         'evidence_commit': 'd843cc59113c7fd3ac89a0b4be9db71e5950ea20',
         'archive_sha256': '9b063d2c92ccf88d67678b032d391bf01e0ceb1a626d5cf23a3f11c0fd250d39',
         'started': 0, 'completed': 0, 'not_run': 36},
    ],
    'cross_execution_recognition_limit': 36,
}
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
MIB = 1024**2
GIB = 1024**3
LIMITS = {'clips': 12, 'attempts': 36, 'workspace_bytes': 10*GIB,
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


def validate_blind(root):
    root = Path(root); require(root.is_dir() and not root.is_symlink(), 'blind root')
    require({p.name for p in root.iterdir()} == {'job.json','audio'}, 'no source mapping or labels')
    job = read_json(root/'job.json',65536)
    require(set(job) == {'schema','clips'} and job['schema']=='blind-source-screen32-job-v1' and
        type(job['clips']) is list and len(job['clips']) == 12, 'blind exact12')
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
    require(canonical(release.get('predecessor_zero_recognition')) == canonical(PREDECESSOR_ZERO_RECOGNITION),
            'exact immutable predecessor and cumulative recognition cap required')
    require(env.get('GITHUB_REPOSITORY')=='jiying2007/kws-pipeline' and
        env.get('GITHUB_REF')=='refs/heads/'+BRANCH and env.get('GITHUB_EVENT_NAME')=='push' and
        env.get('GITHUB_RUN_ATTEMPT')=='1' and re.fullmatch('[1-9][0-9]*',env.get('GITHUB_RUN_NUMBER','')) and
        re.fullmatch('[1-9][0-9]{0,19}',env.get('GITHUB_RUN_ID','')) and
        re.fullmatch('[0-9a-f]{40}',env.get('GITHUB_SHA','')), 'first unique real Actions run')
    require(event.get('created') is True and event.get('repository',{}).get('private') is False and
            event.get('after')==env['GITHUB_SHA'], 'new public branch only')
    return {'run_id':env['GITHUB_RUN_ID'],'head_sha':env['GITHUB_SHA'],'run_attempt':1,'run_number':int(env['GITHUB_RUN_NUMBER'])}


def denominator(job, outputs):
    """36 host accounting rows, never source-resolver/training admission authority."""
    result=[]
    for family in FAMILIES:
        for clip in job['clips']:
            p=Path(outputs)/family/clip['audio_id']; start=p/'started.json'; done=p/'completed.json'; raw=p/'raw-transcript.json'
            row={'family':family,'audio_id':clip['audio_id'],'wav_sha256':clip['wav_sha256'],
                 'attempts':0,'status':'not_run','source_resolver_admission':False,**BOUNDARY}
            if start.exists() or done.exists() or raw.exists() or (p/'raw-native-output.json').exists() or (p/'raw-output.json').exists():
                row.update(attempts=1,status='error')
                if start.exists() and done.exists() and raw.exists():
                    s=read_json(start); d=read_json(done)
                    if (s.get('schema')=='retained-asr-started-v1' and s.get('file_sha256')==clip['wav_sha256']
                        and s.get('source_id')=='qwen12-'+clip['wav_sha256'] and s.get('recognizer_id')==family
                        and d.get('started_sha256')==identity(start)['sha256'] and
                        d.get('raw_output_sha256')==identity(raw)['sha256'] and d.get('schema')=='retained-asr-completed-v1'
                        and d.get('status')=='success' and d.get('decoder_completed') is True):
                        row['status']='native_v2_complete'
            result.append(row)
    require(len(result)==36,'full denominator')
    return result
