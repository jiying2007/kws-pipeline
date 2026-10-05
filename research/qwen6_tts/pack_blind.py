"""Export exactly six normalized WAVs without text, voice IDs or generation metadata."""
import argparse,hashlib,json,os,pathlib,wave,zipfile
from generate_six import validate_plan,sha,require
ROOT=pathlib.Path(__file__).resolve().parent

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()

def pack(generation,out):
    frozen=json.loads((generation/'generation-freeze.json').read_text())
    require(frozen['schema']=='qwen6-generation-freeze-v1','Generation freeze required before blind export')
    require({p.name for p in generation.iterdir()}=={r['path'] for r in frozen['files']}|{'generation-freeze.json'},'Generation file set changed')
    for item in frozen['files']:
        p=generation/item['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Frozen generation bytes changed')
    plan=json.loads((ROOT/'plan.json').read_text());rows=validate_plan(plan)
    receipt=json.loads((generation/'generation-receipt.json').read_text())
    require(receipt['status']=='six_candidates_generated' and len(receipt['generation_rows'])==6,'Incomplete generation cannot enter ASR')
    require(receipt['plan_sha256']==sha(ROOT/'plan.json'),'Generation plan mismatch')
    contents={};clips=[]
    for i,(row,got) in enumerate(zip(rows,receipt['generation_rows'])):
        require(got['source_id']==row['source_id'] and got['status']=='generated_candidate','Missing or reordered cell')
        path=generation/(row['source_id']+'.wav');require(path.is_file() and not path.is_symlink(),'Missing regular WAV')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        require(44<len(raw)<=384044 and digest==got['audio_sha256'],'Derived WAV identity/budget')
        with wave.open(str(path),'rb') as f:
            require((f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getcomptype())==(16000,1,2,'NONE'),'Derived WAV format')
            require(0<f.getnframes()<=192000 and len(raw)==44+2*f.getnframes(),'Canonical WAV frame bound')
            require(hashlib.sha256(f.readframes(f.getnframes())).hexdigest()==got['pcm_sha256'],'PCM identity mismatch')
        name=f'audio/{digest}.wav';require(name not in contents,'Duplicate generated waveform')
        contents[name]=raw;clips.append({'audio_id':f'clip-{i+1:06d}','audio_path':name,'wav_sha256':digest})
    contents['job.json']=canonical({'schema':'blind-asr-job-v1','clips':clips})
    freeze={'schema':'qwen6-blind-input-freeze-v1','job_sha256':hashlib.sha256(contents['job.json']).hexdigest(),
            'files':[{'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in sorted(contents.items())]}
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'blind-inputs.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for name,raw in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,5,0,0,0));info.external_attr=0o100644<<16;z.writestr(info,raw)
    require(archive.stat().st_size<=4*1024**2,'Blind archive exceeds4MiB')
    freeze_path=out/'blind-input-freeze.json';freeze_path.write_bytes(canonical(freeze))
    result={'blind_archive_sha256':sha(archive),'blind_freeze_sha256':sha(freeze_path),'blind_job_sha256':freeze['job_sha256']}
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            for key,value in result.items():f.write(f'{key}={value}\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);a=p.parse_args();print(json.dumps(pack(a.generation,a.out)))
