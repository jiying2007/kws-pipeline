"""Fixed native-to-PCM projection, never speech inference, trimming or gain repair."""
import hashlib,json
from pathlib import Path

def require(ok, why):
    if not ok: raise ValueError(why)

def derive(source, output):
    import numpy as np
    import scipy
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    require(np.__version__=='2.3.5' and scipy.__version__=='1.17.0','exact numeric versions')
    source,output=Path(source),Path(output);output.mkdir(exist_ok=False)
    report=json.loads((source/'independent-verification.json').read_text())
    require(report['artifact_zip_sha256']=='0b598d492683e865126a8ab2f9c66c722deae89a40c507fc843a694a37bf3540' and report['run_id']==41803323981,'exact generated input')
    require(len(report['waveforms'])==4,'fixed four')
    rows=[]
    for row in report['waveforms']:
        p=source/'output'/(row['cell_id']+'.raw-float.wav')
        require(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],'native file hash')
        rate,x=wavfile.read(p)
        require(rate==24000 and x.dtype==np.float32 and x.ndim==1 and len(x)==row['frames'] and np.isfinite(x).all(),'native values')
        y=resample_poly(x,2,3,window=('kaiser',5.0),padtype='constant')
        require(len(y)==(len(x)*2+2)//3 and np.isfinite(y).all() and float(np.max(np.abs(y)))<1,'no clipping or partial audio')
        pcm=np.floor(y.astype(np.float64)*32768).astype('<i2')
        dest=output/(row['cell_id']+'.wav');wavfile.write(dest,16000,pcm)
        rows.append(dict(sample_id=row['cell_id'],requested_text=row['requested_text'],native_sha256=row['sha256'],native_frames=len(x),derived_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),frames=len(pcm),training_admitted=False))
    clips=[];mapping=[];(output/'blind'/'audio').mkdir(parents=True)
    for i,row in enumerate(sorted(rows,key=lambda r:r['derived_sha256']),1):
        sha=row['derived_sha256'];oid='clip-%06d'%i
        raw=(output/(row['sample_id']+'.wav')).read_bytes();(output/'blind'/'audio'/(sha+'.wav')).write_bytes(raw)
        clips.append(dict(audio_id=oid,audio_path='audio/'+sha+'.wav',wav_sha256=sha))
        mapping.append(dict(row,opaque_id=oid))
    (output/'blind'/'job.json').write_text(json.dumps(dict(schema='blind-source-screen32-job-v1',clips=clips),sort_keys=True,separators=(',',':')))
    (output/'mapping.json').write_text(json.dumps(dict(source_run_id=41803323981,source_head='adc2047807289dcba1ff7774c4340797fa0ae57e',method='scipy.signal.resample_poly',up=2,down=3,window=['kaiser',5.0],padtype='constant',numpy_version=np.__version__,scipy_version=scipy.__version__,quantization='floor_float64_times_32768_to_int16',gain_normalization=False,trimming=False,rows=mapping),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    import sys
    derive(sys.argv[1],sys.argv[2])
