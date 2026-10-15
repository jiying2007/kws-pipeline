"""Host-only rational resampling; raw native files remain unchanged."""
from pathlib import Path
from execution_common import require,read,sha,write

def derive(slots,native_rows,native_roots,output):
    import numpy as np
    import scipy
    require(np.__version__ == '2.3.2' and scipy.__version__ == '1.17.1', 'exact hash-locked host numeric versions required')
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from worker import verify_native
    output=Path(output);output.mkdir(exist_ok=False)
    require(len(slots)==len(native_rows)==16,'all sixteen native completions required')
    seen_native=set();seen_values=set();seen_pcm=set();mapping=[];clips=[]
    for cell,row in zip(slots,native_rows):
        require(all(row.get(k)==v for k,v in cell.items()) and row.get('status')=='completed' and row.get('attempts')==1,'native row bound to exact planned cell')
        require(row.get('metrics',{}).get('technically_valid') is True,'native technical gate')
        require(row.get('file')==cell['slot_id']+'.raw-float.wav','native filename')
        path=Path(native_roots[cell['engine']])/row['file']; raw=read(path)
        require(sha(raw)==row.get('raw_wav_sha256'),'native bytes identity')
        verify_native(raw,row['native'])
        rate,x=wavfile.read(path)
        require(rate==24000 and x.dtype==np.float32 and x.ndim==1 and 0<len(x)<=288000 and np.isfinite(x).all(),'native numeric geometry')
        value_sha=sha(x.astype('<f4').tobytes());require(value_sha==row['native']['source_values_sha256'],'native values identity')
        require(sha(raw) not in seen_native and value_sha not in seen_values,'duplicate native across campaign')
        y=resample_poly(x,2,3,window=('kaiser',5.0),padtype='constant')
        require(len(y)==(2*len(x)+2)//3 and np.isfinite(y).all() and float(np.max(np.abs(y)))<1,'derived geometry/clipping')
        pcm=np.floor(y.astype(np.float64)*32768).astype('<i2').tobytes();digest=sha(pcm)
        require(digest not in seen_pcm and any(pcm),'duplicate or silent derived PCM')
        (output/(digest+'.pcm')).write_bytes(pcm)
        clip=dict(opaque_id=digest,file=digest+'.pcm',sha256=digest,bytes=len(pcm),frames=len(pcm)//2,sample_rate_hz=16000,channels=1,sample_format='s16le')
        clips.append(clip);mapping.append(dict(slot_id=cell['slot_id'],native_sha256=sha(raw),native_values_sha256=value_sha,derived=clip,
            derivation=dict(method='scipy.signal.resample_poly',up=2,down=3,window=['kaiser',5.0],padtype='constant',quantization='floor_float64_times_32768_to_int16',numpy_version=np.__version__,scipy_version=scipy.__version__,gain_normalization=False,trimming=False)))
        seen_native.add(sha(raw));seen_values.add(value_sha);seen_pcm.add(digest)
    write(output/'job.json',dict(schema='kw1-blind-pcm-job-v1',clips=sorted(clips,key=lambda c:c['opaque_id'])))
    return mapping
