"""Variable <=16 blind subset with retained locked native Whisper adapter."""
from pathlib import Path
import json,sys,signal,time,wave,hashlib,os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'source-screen32-v1'))
from blind_utils import require,write,validate_wav
from runtime_scope import verify_runtime_scope

def progress(output,stage,index):
    tmp=output/'progress.tmp'
    with tmp.open('w') as f:
        json.dump({'stage':stage,'index':index,'started_monotonic':time.monotonic()},f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,output/'progress.json')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def alarm(*_): raise TimeoutError('single attempt deadline')

def main():
    verify_runtime_scope()
    sys.path.insert(0,str(HERE.parent/'source-screen32-whisper-v1'))
    from adapter import FixedWhisper
    output=Path('/output'); job=json.loads(Path('/input/job.json').read_bytes())
    require(job['schema']=='blind-source-screen32-job-v1' and 0<len(job['clips'])<=16,'blind subset')
    require(all(set(r)=={'audio_id','audio_path','wav_sha256'} for r in job['clips']),'no prompt fields')
    rows=[dict(opaque_id=r['audio_id'],wav_sha256=r['wav_sha256'],status='not_run',raw_text=None,completeness='unknown',quality_flags=[]) for r in job['clips']]
    signal.signal(signal.SIGALRM,alarm)
    try:
        progress(output,'load',0)
        signal.alarm(1200); model=FixedWhisper('/runtime',output);signal.alarm(0)
        for index,(clip,row) in enumerate(zip(job['clips'],rows),1):
            verify_runtime_scope();path=Path('/input')/clip['audio_path'];validate_wav(path,clip['wav_sha256'])
            d=output/clip['audio_id'];d.mkdir();write(d/'started.json',{'attempt':1,'wav_sha256':clip['wav_sha256'],'opaque_id':clip['audio_id'],'job_sha256':digest(Path('/input/job.json')),'config_sha256':digest(output/'native-config.json')})
            progress(output,'cell',index)
            row['status']='failed_no_retry';signal.alarm(300)
            with wave.open(str(path)) as wav: frames=wav.getnframes();pcm=wav.readframes(frames)
            raw=model.decode(pcm,frames,d);signal.alarm(0)
            require(raw['decoder_completion_validated'] is True,'natural decoder completion')
            row.update(status='success',raw_text=raw['raw_text'],completeness='complete')
            write(d/'completed.json',dict(row,started_sha256=digest(d/'started.json'),raw_sha256=digest(d/'raw-output.json')))
    finally:
        signal.alarm(0);write(output/'outcomes.json',rows)
if __name__=='__main__':main()
