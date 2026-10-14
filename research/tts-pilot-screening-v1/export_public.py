"""Positive-list public artifact export. Never archive a runtime/output tree."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from screen import manifest,score,write,require,validate_wav,FAMILIES
from terminal import read

RESULT_FIELDS=('sample_id','family','attempt','status','raw_text','completeness','quality_flags')
SLOT_FIELDS=('sample_id','engine','target_text','status','audio_quality_pass','wav_sha256','native_wav_sha256','failure_type','reason')

def export(host_root,derived_root,destination):
    host_root,derived_root,destination=map(Path,(host_root,derived_root,destination))
    require(not destination.exists(),'fresh public destination')
    data=manifest(derived_root/'manifest.json')
    rows=json.loads(read(host_root/'results.json'))
    require(type(rows) is list and len(rows)<=144,'bounded result count')
    public=[]
    for row in rows:
        require(set(row)==set(RESULT_FIELDS),'reviewed result schema only')
        require(row['family'] in FAMILIES and row['attempt']==1,'result family/attempt')
        require(row['status'] in ('success','error','timeout','failed_no_retry','not_run'),'result status')
        require(row['raw_text'] is None or type(row['raw_text']) is str and len(row['raw_text'])<=32768,'bounded ASR text')
        require(row['completeness'] in ('complete','incomplete','unknown'),'decoder completeness')
        require(type(row['quality_flags']) is list and len(row['quality_flags'])<=32 and all(type(x) is str and len(x)<=128 for x in row['quality_flags']),'bounded quality flags')
        public.append({k:row[k] for k in RESULT_FIELDS})
    report=score(data,public) # Do not trust or copy arbitrary files called report.json.
    destination.mkdir(parents=True)
    write(destination/'results.json',public);write(destination/'report.json',report)
    write(destination/'generation-receipts.json',[{k:r[k] for k in SLOT_FIELDS if k in r} for r in data['slots']])
    for row in data['slots']:
        if row['engine'] not in ('qwen06','qwen17') or row['status']!='success':continue
        require(row['audio_relpath']==row['sample_id']+'.wav','preset-derived filename only')
        source=derived_root/row['audio_relpath'];validate_wav(source,row['wav_sha256'])
        audio=destination/'audio';audio.mkdir(exist_ok=True);shutil.copyfile(source,audio/source.name)
    # Explicit projections only: no glob, raw setup logs, models, runtime, code,
    # cache, arbitrary JSON, or unreviewed raw decoder payload is copied.
    return destination
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host-root',required=True);p.add_argument('--derived-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.host_root,a.derived_root,a.output)
