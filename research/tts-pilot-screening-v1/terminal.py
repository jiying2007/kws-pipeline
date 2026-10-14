"""Host-only terminal accounting from immutable completed clip receipts."""
import hashlib,json
from pathlib import Path
from screen import require,write

def read(path):
    require(not any(p.is_symlink() for p in (path,*path.parents)) and path.stat().st_size<=8*1024**2,'bounded unaliased evidence')
    return path.read_bytes()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def whisper_rows(input_dir,out):
    input_dir,out=Path(input_dir),Path(out);jobraw=read(input_dir/'job.json');job=json.loads(jobraw);rows=[]
    for clip in job['clips']:
        d=out/clip['audio_id'];start=d/'started.json';done=d/'completed.json';rawpath=d/'raw-output.json'
        attempted=any(p.exists() for p in (start,done,rawpath))
        row={'opaque_id':clip['audio_id'],'wav_sha256':clip['wav_sha256'],'status':'failed_no_retry' if attempted else 'not_run','raw_text':None,'completeness':'unknown','quality_flags':[]}
        if attempted:
            try:
                sr,cr,rr=(read(p) for p in (start,done,rawpath));s,c,r=(json.loads(x) for x in (sr,cr,rr))
                require(s=={'attempt':1,'opaque_id':clip['audio_id'],'wav_sha256':clip['wav_sha256'],'job_sha256':sha(jobraw),'config_sha256':sha(read(out/'native-config.json'))},'start identity')
                require(c['started_sha256']==sha(sr) and c['raw_sha256']==sha(rr),'completion binding')
                require(c['opaque_id']==clip['audio_id'] and c['wav_sha256']==clip['wav_sha256'],'audio binding')
                require(c['status']=='success' and c['completeness']=='complete' and c['quality_flags']==[] and c['raw_text']==r['raw_text'] and type(r['raw_text']) is str,'transcript binding')
                require(r['decoder_completion_validated'] is True and r['full_input_processed'] is True,'decoder completion')
                row.update(status='success',raw_text=r['raw_text'],completeness='complete')
            except (OSError,ValueError,KeyError,TypeError):pass
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True);write(out/'terminal-outcomes.json',rows)
    return rows
