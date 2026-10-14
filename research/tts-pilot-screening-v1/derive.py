"""Offline native-to-ASR projection. Never normalizes gain, trims or regenerates."""
import argparse,hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from screen import require,write,TARGETS
from campaign import expected_plan,MODELS

def derive(plan_path,artifact_root,output):
    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    from scipy.io import wavfile
    plan=json.loads(Path(plan_path).read_bytes());require(plan==expected_plan(),'exact frozen pilot design');output=Path(output);output.mkdir(parents=True,exist_ok=False)
    slots=[]
    for engine in ('qwen06','qwen17'):
        cells=plan[engine];require(len(cells)==16,'fixed 16 per Qwen lane')
        folder=Path(artifact_root)/('tqp-'+engine)/'artifact'
        receipt_path=folder/(engine+'-receipt.json');receipts=[receipt_path] if receipt_path.exists() else []
        found={}
        for receipt in receipts:
            receipt_data=json.loads(receipt.read_bytes())
            require(receipt_data.get('engine')==engine and (receipt_data.get('model_id'),receipt_data.get('revision'))==tuple(MODELS[engine]),'TTS model receipt binding')
            for r in receipt_data.get('outcomes',[]):
                expected=next((c for c in cells if c['cell_id']==r.get('cell_id')),None)
                require(expected is not None and all(r.get(k)==v for k,v in expected.items()) and r.get('attempts')==1,'TTS plan row binding')
                require(r['cell_id'] not in found,'duplicate generation slot, cannot rerun')
                found[r['cell_id']]=r
        for cell in cells:
            row={'sample_id':cell['cell_id'],'engine':engine,'target_text':cell['text'],'status':'not_run','audio_quality_pass':False}
            r=found.get(cell['cell_id'])
            if r:
                row['status']='failed'
                if r.get('status')=='generated':
                    try:
                        src=folder/(cell['cell_id']+'.raw-float.wav');raw=src.read_bytes()
                        require(hashlib.sha256(raw).hexdigest()==r['raw_wav_sha256'],'native hash')
                        require(len(raw)<=2*1024**2,'bounded native WAV')
                        sr,samples=wavfile.read(src);require(sr==24000 and samples.dtype==np.float32 and samples.ndim==1,'native float geometry')
                        require(0<len(samples)<=288000 and np.isfinite(samples).all(),'native samples')
                        require(r.get('native')=={'sample_rate_hz':24000,'channels':1,'frames':len(samples),'subtype':'FLOAT','source_values_sha256':hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest()},'native receipt geometry/value binding')
                        # Deterministic rational resampling; original preserved. No amplitude normalization.
                        y=resample_poly(samples,2,3,window=('kaiser',5.0),padtype='constant')
                        require(len(y)==(len(samples)*2+2)//3,'resampled complete frame count')
                        require(np.isfinite(y).all() and float(np.max(np.abs(y)))<1,'resampling would clip; preserve failure')
                        dest=output/(cell['cell_id']+'.wav');wavfile.write(dest,16000,np.floor(y.astype(np.float64)*32768).astype('<i2'))
                        row.update(status='success',audio_relpath=dest.name,wav_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),native_wav_sha256=r['raw_wav_sha256'],audio_quality_pass=r.get('metrics',{}).get('technically_valid') is True,
                            derivation={'method':'scipy.signal.resample_poly','up':2,'down':3,'window':['kaiser',5.0],'padtype':'constant','gain_normalization':False,'trimming':False,'scipy_version':scipy.__version__,'quantization':'floor_float64_times_32768_to_int16'})
                    except Exception as error:row['failure_type']=type(error).__name__
            slots.append(row)
    for i in range(16): slots.append({'sample_id':f'tqp-cosy3-{i:02d}','engine':'cosy3','target_text':TARGETS[(i//2)%2],'status':'blocked','audio_quality_pass':False,'reason':'authorized_reference_voices_missing'})
    write(output/'manifest.json',{'schema':'tts-quality-pilot-asr-input-v1','slots':slots})
    return slots
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--artifact-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();derive(a.plan,a.artifact_root,a.output)
