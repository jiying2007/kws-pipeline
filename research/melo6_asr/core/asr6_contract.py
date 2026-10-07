"""Six newly generated blind PCM inputs, dynamically bound to the TTS job output."""
import hashlib
import json
import re
from pathlib import Path
from pcm.pcm_binding import WaveExpectation,bind_wave,validate_pcm_descriptor

EXPERIMENT='melo-six-phrase-source-screen-asr6-v1'
BATCH_IDS={'all6':[f'clip-{i:06d}' for i in range(1,7)]}
SHA=re.compile(r'[0-9a-f]{64}\Z')
MAX_FRAMES=160000
MAX_WAV_BYTES=44+2*MAX_FRAMES
MAX_JOB_BYTES=65536
MAX_ARCHIVE_BYTES=2*1024**2

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()

def digest(raw):return hashlib.sha256(raw).hexdigest()

def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('Duplicate JSON key')
        out[key]=value
    return out

def decode(raw,sha,max_bytes=8*1024**2):
    if type(raw) is not bytes or len(raw)>max_bytes or type(sha) is not str or not SHA.fullmatch(sha) or digest(raw)!=sha:
        raise ValueError('Frozen bytes mismatch')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def validate_job(raw,expected_sha):
    job=decode(raw,expected_sha,MAX_JOB_BYTES)
    if type(job) is not dict or set(job)!={'schema','clips'} or job['schema']!='blind-asr-job-v1':raise ValueError('Blind job schema')
    if type(job['clips']) is not list or len(job['clips'])!=6 or [r.get('audio_id') for r in job['clips']]!=BATCH_IDS['all6']:raise ValueError('Exactly6 requested IDs required')
    hashes=set()
    for row in job['clips']:
        if set(row)!={'audio_id','audio_path','wav_sha256'} or type(row['wav_sha256']) is not str or not SHA.fullmatch(row['wav_sha256']):raise ValueError('Only blind audio fields allowed')
        if row['audio_path']!='audio/'+row['wav_sha256']+'.wav' or row['wav_sha256'] in hashes:raise ValueError('Duplicate or nonopaque audio path')
        hashes.add(row['wav_sha256'])
    return job

def validate_input_freeze(raw,expected_sha):
    freeze=decode(raw,expected_sha,MAX_JOB_BYTES)
    if type(freeze) is not dict or set(freeze)!={'schema','job_sha256','files'} or freeze['schema']!='qwen6-blind-input-freeze-v1':raise ValueError('Producer freeze schema')
    if type(freeze['job_sha256']) is not str or not SHA.fullmatch(freeze['job_sha256']):raise ValueError('Producer job hash')
    if type(freeze['files']) is not list or len(freeze['files'])!=7:raise ValueError('Exactly7 blind input files required')
    names=set();wavs=0
    for row in freeze['files']:
        if type(row) is not dict or set(row)!={'path','bytes','sha256'}:raise ValueError('Producer freeze fields')
        name=row['path']
        if type(name) is not str or name in names or type(row['bytes']) is not int:raise ValueError('Producer freeze membership')
        if type(row['sha256']) is not str or not SHA.fullmatch(row['sha256']):raise ValueError('Producer file hash')
        if name=='job.json':
            if not 0<row['bytes']<=MAX_JOB_BYTES or row['sha256']!=freeze['job_sha256']:raise ValueError('Producer job binding')
        else:
            if name!='audio/'+row['sha256']+'.wav' or not 46<=row['bytes']<=MAX_WAV_BYTES:raise ValueError('Producer WAV bounds/path')
            wavs+=1
        names.add(name)
    if 'job.json' not in names or wavs!=6:raise ValueError('Producer blind membership')
    return freeze

def bind_job(input_root,job_sha256,input_freeze_sha256):
    root=Path(input_root);job=validate_job((root/'job.json').read_bytes(),job_sha256);clips=[]
    if sorted(p.name for p in (root/'audio').iterdir())!=sorted(r['wav_sha256']+'.wav' for r in job['clips']):raise ValueError('Extra/missing WAV')
    for row in job['clips']:
        path=root/row['audio_path'];size=path.stat().st_size
        if not 46<=size<=MAX_WAV_BYTES:raise ValueError('PCM input size limit')
        bound=bind_wave(str(root/'audio'),WaveExpectation(row['audio_id'],path.name,row['wav_sha256'],size,(size-44)//2))
        clips.append({'opaque_id':bound.opaque_id,'binding_sha256':bound.descriptor_sha256,'descriptor':bound.descriptor})
    result={'schema':'generated-asr6-decoder-inputs-v1','experiment_id':EXPERIMENT,'source_manifest_sha256':job_sha256,
        'input_freeze_sha256':input_freeze_sha256,'scope':'audio_only_no_plan_context','order':BATCH_IDS['all6'],'clips':clips}
    validate_decoder(canonical(result),digest(canonical(result)));return result

def validate_decoder(raw,sha):
    value=decode(raw,sha)
    if set(value)!={'schema','experiment_id','source_manifest_sha256','input_freeze_sha256','scope','order','clips'}:raise ValueError('Decoder fields')
    if value['schema']!='generated-asr6-decoder-inputs-v1' or value['experiment_id']!=EXPERIMENT or value['scope']!='audio_only_no_plan_context':raise ValueError('ASR6 decoder scope')
    if any(type(value[k]) is not str or not SHA.fullmatch(value[k]) for k in ('source_manifest_sha256','input_freeze_sha256')):raise ValueError('Generated input identities')
    if value['order']!=BATCH_IDS['all6'] or len(value['clips'])!=6:raise ValueError('ASR6 exact denominator')
    index={};hashes=set()
    for row in value['clips']:
        if set(row)!={'opaque_id','binding_sha256','descriptor'}:raise ValueError('Decoder row fields')
        d=row['descriptor'];validate_pcm_descriptor(d)
        if not 1<=d['raw_pcm16']['frame_count']<=MAX_FRAMES:raise ValueError('ASR6 maximum duration')
        if d['opaque_id']!=row['opaque_id'] or row['opaque_id'] in index or digest(canonical(d))!=row['binding_sha256'] or d['wav']['sha256'] in hashes:raise ValueError('PCM binding')
        index[row['opaque_id']]=row;hashes.add(d['wav']['sha256'])
    if list(index)!=value['order']:raise ValueError('ASR6 order')
    return {**value,'clips_index':index}
