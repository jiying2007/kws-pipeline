"""Fixed public blind input contract; no human labels or target context."""
import hashlib
import json
import re
from pathlib import Path
from pcm.pcm_binding import WaveExpectation, bind_wave, validate_pcm_descriptor

EXPERIMENT = 'fixed30-exposed-asr-regression-v1'
JOB_SHA256 = '08ebc361dafa395b408db04679cb63dd6cc1dad804a13caf0132b0051b89a969'
ARCHIVE_SHA256 = 'c5b466d8091f06fcdf6579770d9962b8fc44449287b9608ceb59de8f7c688e02'
BATCH_IDS = {'all30': [f'clip-{i:06d}' for i in range(1,31)]}
SHA = re.compile(r'[0-9a-f]{64}\Z')

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('Duplicate JSON key')
        out[key]=value
    return out

def decode(raw,sha):
    if type(raw) is not bytes or len(raw)>8*1024**2 or digest(raw)!=sha:
        raise ValueError('Frozen JSON bytes changed')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def validate_job(raw):
    job=decode(raw,JOB_SHA256)
    if set(job)!={'schema','clips'} or job['schema']!='blind-asr-job-v1': raise ValueError('Blind job schema')
    if len(job['clips'])!=30 or [r.get('audio_id') for r in job['clips']]!=BATCH_IDS['all30']: raise ValueError('Exact30 required')
    hashes=set()
    for row in job['clips']:
        if set(row)!={'audio_id','audio_path','wav_sha256'} or not SHA.fullmatch(row['wav_sha256']): raise ValueError('Audio-only fields required')
        if row['audio_path']!='audio/'+row['wav_sha256']+'.wav' or row['wav_sha256'] in hashes: raise ValueError('Duplicate/path drift')
        hashes.add(row['wav_sha256'])
    return job

def bind_job(input_root):
    root=Path(input_root); job=validate_job((root/'job.json').read_bytes()); clips=[]
    if sorted(p.name for p in (root/'audio').iterdir())!=sorted(r['wav_sha256']+'.wav' for r in job['clips']): raise ValueError('Extra/missing WAV')
    for row in job['clips']:
        path=root/row['audio_path']; size=path.stat().st_size
        bound=bind_wave(str(root/'audio'),WaveExpectation(row['audio_id'],path.name,row['wav_sha256'],size,(size-44)//2))
        clips.append({'opaque_id':bound.opaque_id,'binding_sha256':bound.descriptor_sha256,'descriptor':bound.descriptor})
    result={'schema':'fixed30-decoder-inputs-v1','experiment_id':EXPERIMENT,'source_manifest_sha256':JOB_SHA256,
            'scope':'audio_only_no_human_gold','order':BATCH_IDS['all30'],'clips':clips}
    validate_decoder(canonical(result),digest(canonical(result)))
    return result

def validate_decoder(raw,sha):
    value=decode(raw,sha)
    if set(value)!={'schema','experiment_id','source_manifest_sha256','scope','order','clips'}: raise ValueError('Decoder fields')
    if value['schema']!='fixed30-decoder-inputs-v1' or value['experiment_id']!=EXPERIMENT or value['source_manifest_sha256']!=JOB_SHA256 or value['scope']!='audio_only_no_human_gold': raise ValueError('Decoder scope')
    if value['order']!=BATCH_IDS['all30'] or len(value['clips'])!=30: raise ValueError('Decoder count')
    idx={}; hashes=set()
    for row in value['clips']:
        if set(row)!={'opaque_id','binding_sha256','descriptor'}: raise ValueError('Decoder row fields')
        d=row['descriptor'];validate_pcm_descriptor(d)
        if d['opaque_id']!=row['opaque_id'] or row['opaque_id'] in idx or digest(canonical(d))!=row['binding_sha256'] or d['wav']['sha256'] in hashes: raise ValueError('Decoder binding')
        idx[row['opaque_id']]=row;hashes.add(d['wav']['sha256'])
    if list(idx)!=value['order']: raise ValueError('Decoder order')
    return {**value,'clips_index':idx}
