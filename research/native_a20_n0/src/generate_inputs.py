#!/usr/bin/env python3
"""Deterministic integer nonspeech generation and signal QA, never candidate DSP."""
import array, hashlib, json, math, pathlib, struct, sys, time, wave
from geometry import geometry
ROOT=pathlib.Path(__file__).resolve().parents[1]
RECIPE_SHA='68515c3f50518b1bdb6852ec8e2ffef67667bb71bc78688337f560e8e7e5a559'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f: f.write(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def round32(v):return (v+16)//32 if v>=0 else -((-v+16)//32)
def samples(row,n):
    prefix=row['domain'].encode('ascii')+b'\0'+bytes.fromhex(row['seed_hex'])
    history=[0]*32; total=0; index=0
    for counter in range((n+15)//16):
        block=hashlib.sha256(prefix+struct.pack('<Q',counter)).digest()
        for u in struct.unpack('<16H',block):
            if index>=n:return
            v=(u&255)-(u>>8)
            if row['recording']=='n0-colored-ma32':
                slot=index%32;total-=history[slot];history[slot]=16*v;total+=history[slot];v=round32(total)
            elif row['recording']=='n0-sparse-transients':
                rel=index-11*16000
                if rel>=0:
                    k,j=divmod(rel,23*16000)
                    if k<13 and j<320:
                        h=j%160;a=(4096,6144,8192)[k%3]
                        pulse=(a*min(h+1,160-h))//80
                        if (k%2)^(j>=160):pulse=-pulse
                        v+=pulse
            yield v;index+=1

def qa(pcm):
    data=pcm.read_bytes(); values=array.array('h');values.frombytes(data)
    if sys.byteorder!='little': values.byteswap()
    n=len(values);s=sum(values);sq=sum(x*x for x in values)
    peak=max(abs(min(values)),abs(max(values)))
    diff=sum((values[i]-values[i-1])**2 for i in range(1,n))
    lag1=sum(values[i]*values[i-1] for i in range(1,n))
    blocks=[hashlib.sha256(data[j:j+32000]).hexdigest() for j in range(0,len(data),32000)]
    return {'samples':n,'minimum_pcm':min(values),'maximum_pcm':max(values),'peak_abs_pcm':peak,'sum_pcm':s,'sum_square_pcm':sq,'mean_pcm':s/n,'rms_pcm':math.sqrt(sq/n),'rms_dbfs_reference_32768':20*math.log10(math.sqrt(sq/n)/32768),'full_scale_clipping_samples':sum(x in (-32768,32767) for x in values),'first_difference_sum_square':diff,'lag1_uncentered_product_sum':lag1,'one_second_block_hashes':blocks,'unique_one_second_blocks':len(set(blocks)),'spectral_qa':'NOT_COMPUTED_NO_FFT; spectral shape defined by integer recipe, not measured bandlimit','semantic_evidence':'generated exclusively from deterministic integer pseudorandom noise and explicit nonlinguistic transient recipe; no source clips or speech synthesis'}

def main():
    start=time.monotonic();cpu=time.process_time()
    assert sha(ROOT/'GENERATOR-RECIPE.json')==RECIPE_SHA
    recipe=json.loads((ROOT/'GENERATOR-RECIPE.json').read_text())
    save(ROOT/'metadata/generator-before-generation-freeze.json',{'schema':'a20-n0-generator-prefreeze-v1','recipe_sha256':RECIPE_SHA,'generator_sha256':sha(pathlib.Path(__file__)),'geometry_source_sha256':sha(ROOT/'src/geometry.py'),'candidate_runs':0,'reason':'fixed complete generator and seeds before generating any input or observing any model output'})
    rows=[]; gs=[]
    for row in recipe['streams']:
        name=row['recording']; n=recipe['samples_per_stream'];pcm=ROOT/'inputs/pcm'/f'{name}.pcm';wav=ROOT/'inputs/audio'/f'{name}.wav'
        assert not pcm.exists() and not wav.exists()
        with pcm.open('xb') as f:
            buffer=[]
            for v in samples(row,n):
                assert -32768<=v<=32767
                buffer.append(v)
                if len(buffer)==16000:f.write(struct.pack('<16000h',*buffer));buffer=[]
            if buffer:f.write(struct.pack('<'+'h'*len(buffer),*buffer))
        with wav.open('xb') as f:
            f.write(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+2*n,b'WAVE',b'fmt ',16,1,1,16000,32000,2,16,b'data',2*n))
            with pcm.open('rb') as p:
                while chunk:=p.read(1048576):f.write(chunk)
        assert pcm.stat().st_size==2*n and wav.stat().st_size==44+2*n
        with wave.open(str(wav),'rb') as f:
            assert (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getnframes(),f.getcomptype())==(1,2,16000,n,'NONE')
            assert hashlib.sha256(f.readframes(n)).hexdigest()==sha(pcm) and f.readframes(1)==b''
        q=qa(pcm);assert q['peak_abs_pcm']<=row['peak_bound_pcm'] and q['full_scale_clipping_samples']==0 and q['unique_one_second_blocks']==300
        item={'recording':name,'kind':'constructed_nonspeech_negative_control','domain':row['domain'],'seed_hex':row['seed_hex'],'sample_rate_hz':16000,'channels':1,'sample_width_bytes':2,'frames':n,'duration_s':300,'wav_path':str(wav.relative_to(ROOT)),'pcm_path':str(pcm.relative_to(ROOT)),'wav_bytes':wav.stat().st_size,'pcm_bytes':pcm.stat().st_size,'wav_sha256':sha(wav),'pcm_sha256':sha(pcm),'recipe_sha256':RECIPE_SHA,'verified':True,'formal_far_qualification_allowed':False,'signal_qa':q}
        rows.append(item);gs.append({'recording':name,**geometry(n)})
        print(json.dumps({k:item[k] for k in ['recording','frames','wav_sha256','pcm_sha256']},sort_keys=True),flush=True)
    save(ROOT/'metadata/verified-inputs.json',{'schema':'a20-n0-verified-generated-inputs-v1','verification':'PASS','audio_count':3,'frames':14400000,'duration_s':900,'audio_bytes':28800132,'recipe_sha256':RECIPE_SHA,'candidate_outputs_seen':False,'rows':rows})
    keys=['frames','feed_calls','full_feeds','tail_feeds','finish_calls','callbacks','fbank_rows','model_rows','decoder_input_rows']
    save(ROOT/'metadata/geometry.json',{'schema':'a20-n0-source-derived-integer-geometry-v1','method':'pure integer port of donor_splice_plan_call/accept; no FFT/frontend/model or decoder executed','totals':{k:sum(x[k] for x in gs) for k in keys},'rows':gs})
    lines=['/* Fixed three complete N0 streams; metadata only. */','typedef struct {const char *id,*pcm,*wav_sha,*pcm_sha; size_t frames;} clip_spec;','static const clip_spec clips[3] = {']
    for r in rows:lines.append('{'+','.join(json.dumps(r[k]) for k in ['recording','pcm_path','wav_sha256','pcm_sha256'])+','+str(r['frames'])+'},')
    with (ROOT/'src/inputs.h').open('x') as f:f.write('\n'.join(lines+['};'])+'\n')
    save(ROOT/'metadata/generation-resource.json',{'wall_s':time.monotonic()-start,'process_cpu_s':time.process_time()-cpu,'scope':'input generation plus header/hash/time-domain QA only; separate from future candidate resource budget','candidate_runs':0,'frontend_fft_runs':0,'decoder_runs':0})
if __name__=='__main__':main()
