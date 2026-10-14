"""New pilot accounting. No ML imports, no historical execution controls."""
import os
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata
import wave

FAMILIES = ('qwen06', 'sensevoice', 'whisper')
TARGETS = ('你好小窝', '小窝小窝')
MAX_SLOTS = 48

def require(ok, why):
    if not ok: raise ValueError(why)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def write(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())

def normalize(text):
    require(type(text) is str, 'transcript must be text')
    # Never map homophones or synonyms to the target; punctuation only.
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace() and not unicodedata.category(c).startswith('P'))

def validate_manifest(data):
    from campaign import expected_plan
    require(data['schema']=='tts-quality-pilot-asr-input-v1','new manifest only')
    rows=data['slots'];require(type(rows) is list and len(rows)==48,'fixed 48 design slots')
    plan=expected_plan()
    expected={c['cell_id']:(e,c['text']) for e,cells in plan.items() for c in cells}
    expected.update({f'tqp-cosy3-{i:02d}':('cosy3',TARGETS[(i//2)%2]) for i in range(16)})
    require(len({r['sample_id'] for r in rows})==48 and {r['sample_id'] for r in rows}==set(expected),'exact pilot slot IDs')
    for row in rows:
        require((row['engine'],row['target_text'])==expected[row['sample_id']],'slot engine/target identity')
        require(row['status'] in ('success','failed','not_run','blocked'),'slot status')
        if row['engine']=='cosy3':require(row['status']=='blocked','Cosy reference remains unapproved')
    return data

def manifest(path):
    return validate_manifest(json.loads(Path(path).read_bytes()))

def validate_wav(path, expected):
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'unaliased WAV')
    require(path.stat().st_size <= 384044, '12 second maximum')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'audio digest')
    with wave.open(str(path)) as w:
        require((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype()) == (16000,1,2,'NONE'), 'derived PCM16 mono 16k only')
        require(0 < w.getnframes() <= 192000, 'full input maximum')

def pack(data, audio_root, dest, subset=None):
    """Chunk per producer lane <=16. No label/path leakage into model input."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=False)
    mapping=[]
    for engine in ('qwen06','qwen17','cosy3'):
        selected=[r for r in data['slots'] if r['engine']==engine and r['status']=='success' and (subset is None or r['sample_id'] in subset)]
        if not selected: continue
        batch=dest/engine; (batch/'audio').mkdir(parents=True)
        # Identical bytes decode once, but every original slot stays in the denominator.
        byhash={}
        for row in selected:
            rel=Path(row['audio_relpath']); require(not rel.is_absolute() and '..' not in rel.parts, 'relative audio path')
            source=Path(audio_root)/rel; validate_wav(source,row['wav_sha256'])
            byhash.setdefault(row['wav_sha256'],[]).append(row)
        clips=[]
        for i,(sha,rows) in enumerate(sorted(byhash.items()),1):
            oid=f'clip-{i:06d}'
            shutil.copyfile(Path(audio_root)/rows[0]['audio_relpath'],batch/'audio'/f'{sha}.wav')
            clips.append({'audio_id':oid,'audio_path':f'audio/{sha}.wav','wav_sha256':sha})
            mapping.extend({'sample_id':r['sample_id'],'batch':engine,'opaque_id':oid,'wav_sha256':sha} for r in rows)
        write(batch/'job.json',{'schema':'blind-source-screen32-job-v1','clips':clips})
    # Host-only mapping MUST NOT be mounted into recognizer.
    write(dest/'mapping.json',mapping)
    return mapping

def is_ok(row):
    return row is not None and row.get('status')=='success' and isinstance(row.get('raw_text'),str)

def disagreement(first, second):
    # Operational failures are not linguistic disagreement and never trigger retries.
    return is_ok(first) and is_ok(second) and normalize(first['raw_text']) != normalize(second['raw_text'])

def result_index(rows):
    index={}
    for r in rows:
        key=(r['sample_id'],r['family'])
        require(r['family'] in FAMILIES and key not in index, 'unique recognizer slot')
        require(r.get('attempt',1)==1, 'no retries')
        index[key]=r
    return index

def third_subset(data, results):
    idx=result_index(results)
    return [r['sample_id'] for r in data['slots'] if disagreement(idx.get((r['sample_id'],'qwen06')),idx.get((r['sample_id'],'sensevoice')))]

def score(data, results):
    validate_manifest(data)
    idx=result_index(results); ids={r['sample_id'] for r in data['slots']}
    require(all(k[0] in ids for k in idx), 'unknown result slot')
    scored=[]
    for slot in data['slots']:
        sid=slot['sample_id']; target=normalize(slot['target_text'])
        a,b,c=[idx.get((sid,f)) for f in FAMILIES]
        needs=disagreement(a,b)
        require(c is None or needs, 'third recognizer only for successful primary disagreement')
        successful=[r for r in (a,b,c) if is_ok(r)]
        matches=sum(normalize(r['raw_text'])==target for r in successful)
        agreement=is_ok(a) and is_ok(b) and normalize(a['raw_text'])==normalize(b['raw_text'])
        # Majority is review-only. High precision tier requires both independent first-pass decoders.
        target_consensus=bool(agreement and normalize(a['raw_text'])==target)
        quality=slot.get('audio_quality_pass') is True and slot['status']=='success'
        complete=all(r.get('completeness')=='complete' and not r.get('quality_flags') for r in (a,b) if r) and len(successful)>=2
        overall=quality and target_consensus and complete
        scored.append({'sample_id':sid,'engine':slot['engine'],'audio_quality_pass':quality,
            'consensus_kind':'two-model-consensus' if agreement else 'three-model-disagreement' if is_ok(c) else 'primary-disagreement' if needs else 'unresolved','primary_text_agreement':bool(agreement),'target_text_consensus':target_consensus,
            'target_vote_count':matches,'third_needed':needs,'third_status':c.get('status') if c else 'not_run' if needs else 'not_needed',
            'overall_qualified':overall,'tier':'machine_high_confidence_candidate' if overall else 'review' if matches>=2 else 'reject_or_unresolved',
            'human_gold':False,'training_admitted':False,
            'asr_slots':{f:(idx[(sid,f)]['status'] if (sid,f) in idx else 'not_needed' if f=='whisper' and not needs else 'not_run') for f in FAMILIES}})
    def metrics(rows):
        n=len(rows)
        return {k:{'numerator':sum(r[k] for r in rows),'denominator':n,'rate':sum(r[k] for r in rows)/n if n else None} for k in ('audio_quality_pass','primary_text_agreement','target_text_consensus','overall_qualified')}
    return {'schema':'tts-pilot-screening-report-v1','slots':scored,'metrics':metrics(scored),
      'per_engine':{e:metrics([r for r in scored if r['engine']==e]) for e in ('qwen06','qwen17','cosy3')},
      'denominators':{'design_slots':len(scored),'authorized_executable_slots':sum(r['engine']!='cosy3' for r in data['slots']),'generated_decodable_slots':sum(r['status']=='success' for r in data['slots']),'blocked_reference_slots':sum(r['engine']=='cosy3' and r['status']=='blocked' for r in data['slots'])},'executable_metrics':metrics([r for r in scored if r['engine']!='cosy3']),'max_asr_slots':len(scored)*3,'human_gold':False,'training_admitted':False,
      'confidence_note':'Machine tier is a screening heuristic, not calibrated probability or human truth; held-out evaluation excluded.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('pack','third','score'));p.add_argument('--manifest',required=True);p.add_argument('--results');p.add_argument('--audio-root');p.add_argument('--output',required=True);a=p.parse_args()
    data=manifest(a.manifest)
    if a.mode=='pack': pack(data,a.audio_root,a.output)
    else:
        results=json.loads(Path(a.results).read_bytes())
        if a.mode=='third': pack(data,a.audio_root,a.output,set(third_subset(data,results)))
        else: write(a.output,score(data,results))
if __name__=='__main__':main()
