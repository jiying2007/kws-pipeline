#!/usr/bin/env python3
"""V6 read-only original-PCM + saved CTC forensics; no model forwards/optimizer."""
from __future__ import annotations
import argparse, ast, hashlib, io, json, math, struct, wave, zipfile
from pathlib import Path
SOURCE_SHA='530fccdd25abca9e76331dd8dad1f2440181158e140f2279ee952833a8d220f6'
PAIR_SHA='7a771ce8947f26b3a14c97213c3aa279056dd89968990e49330da48955c347a0'

def need(x,why):
    if not x: raise ValueError(why)

def sha(x): return hashlib.sha256(x).hexdigest()

def npy(x):
    need(x[:6]==b'\x93NUMPY','NPY_MAGIC')
    v=x[6:8];need(v in (b'\x01\x00',b'\x02\x00',b'\x03\x00'),'NPY_VERSION')
    off=10 if v==b'\x01\x00' else 12
    n=struct.unpack_from('<H' if off==10 else '<I',x,8)[0]
    need(0<n<10000 and off+n<=len(x),'NPY_HEADER')
    h=ast.literal_eval(x[off:off+n].decode('latin1'));shape=h.get('shape')
    need(h.get('descr') in ('<f4','=f4') and h.get('fortran_order') is False
         and type(shape) is tuple and len(shape)==3 and shape[0]==1
         and type(shape[1]) is int and shape[1]>0 and shape[2]==4,'NPY_FLOAT32_4TOKEN')
    raw=x[off+n:];need(len(raw)==shape[1]*16,'NPY_BYTES')
    vals=[a[0] for a in struct.iter_unpack('<f',raw)]
    need(all(math.isfinite(v) for v in vals),'FINITE')
    return [vals[i:i+4] for i in range(0,len(vals),4)]

def greedy(rows):
    t=[max(range(4),key=lambda i:x[i]) for x in rows];out=[];prev=-1
    for v in t:
        if v and v!=prev:out.append(v)
        prev=v
    return t,out

def ratio(pcm):
    s=[v[0]/32768 for v in struct.iter_unpack('<h',pcm)]
    need(len(s)>=6400,'PCM_LEN')
    rms=lambda a:math.sqrt(sum(x*x for x in a)/len(a))
    entire=rms(s);need(entire>0,'EMPTY_SOURCE')
    return rms(s[:400])/entire

def audit(source_path, pair_path):
    need(source_path.is_file() and pair_path.is_file(),'ARCHIVES_REQUIRED')
    need(sha(source_path.read_bytes())==SOURCE_SHA and sha(pair_path.read_bytes())==PAIR_SHA,'PINNED_SHA256')
    with zipfile.ZipFile(source_path) as s,zipfile.ZipFile(pair_path) as p:
        need(set(p.namelist())=={'RESULT.json','ADMISSION.json','STARTED.json','ORIGINAL_FRAME_LOGITS.npz'},'PAIR_FILES')
        manifest=json.loads(s.read('MANIFEST.json'));result=json.loads(p.read('RESULT.json'))
        need(manifest['completed']==manifest['denominator']==len(manifest['rows'])==32,'SOURCE_32')
        need(result['status']=='FRESH_SYNTHETIC_PAIRED_DEV_DIAGNOSTIC_NO_PROMOTION'
             and result['product_promotion_allowed'] is False
             and result['model_selection_allowed'] is False
             and result['cal_locked_opened'] is False
             and result['new_optimizer_updates']==0,'NO_PROMOTION_OR_TRAIN')
        logits=p.read('ORIGINAL_FRAME_LOGITS.npz')
        need(sha(logits)==result['actual_original_frame_logits_sha256'],'ORIGINAL_LOGITS_SHA')
        source={r['id']:r for r in manifest['rows']};rows=result['per_original']
        need(len(rows)==14 and len({r['id'] for r in rows})==14
             and sum(r['positive'] for r in rows)==6
             and len({r['source_group_id'] for r in rows})==8
             and len({r['source_group_id'] for r in rows if r['positive']})==4,
             'SELECTED_6_8_FOUR_POSITIVE_GROUPS')
        first={m:{'positive':0,'ordinary':0} for m in ('B0','B1')}
        missing={m:0 for m in ('B0','B1')};ratios=[];per=[]
        with zipfile.ZipFile(io.BytesIO(logits)) as z:
            need(set(z.namelist())=={r['id']+'__'+m+'.npy' for r in rows for m in first},'EXACT_LOGITS')
            for r in rows:
                orig=source[r['id']]
                need(r['role']==orig['role']=='development'
                     and r['source_group_id']==orig['source_group_id']
                     and r['pcm_sha256']==orig['pcm_sha256']
                     and orig['requested_class']==(2 if r['positive'] else 0),'SOURCE_ID')
                wav=s.read('wav/'+r['id']+'.wav')
                need(sha(wav)==orig['wav_sha256'],'WAV_SHA')
                with wave.open(io.BytesIO(wav)) as w:
                    need((w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000),'PCM16')
                    pcm=w.readframes(w.getnframes())
                need(sha(pcm)==r['pcm_sha256'],'PCM_SHA')
                rr=ratio(pcm);ratios.append(rr)
                details={'id':r['id'],'positive':r['positive'],'first_25ms_rms_ratio':round(rr,6)}
                for m in first:
                    f=npy(z.read(r['id']+'__'+m+'.npy'));tokens,collapsed=greedy(f)
                    need(collapsed==r['conditions']['original'][m]['ctc_tokens'],'GREEDY_TOKEN_EQUIVALENCE')
                    if tokens[0]==1:first[m]['positive' if r['positive'] else 'ordinary']+=1
                    half=r['conditions']['half_gain'][m]
                    if r['positive'] and not half['event'] and 2 not in half['ctc_tokens']:missing[m]+=1
                    details[m]={'first_token':tokens[0],'original':collapsed,'half_gain':half['ctc_tokens']}
                per.append(details)
        need(all(first[m]=={'positive':6,'ordinary':8} for m in first),'14_OF_14_START_HEY')
        need(missing=={'B0':5,'B1':5},'FIVE_HALF_GAIN_MISSES_LACK_MEH')
        need(max(ratios)<0.05,'LOW_ENERGY_FIRST_25MS')
        need(result['comparison']['original']['B0']['positive_hits']==5
             and result['comparison']['original']['B1']['positive_hits']==6,'FROZEN_BASELINE_HITS')
        return {'schema':'english-v6-saved-ctc-forensics-v1','status':'SAVED_EVIDENCE_ONLY_NO_NEW_FORWARDS',
                'source_artifact':14543254675,'source_zip_sha256':SOURCE_SHA,
                'paired_artifact':14585918123,'paired_run':44640947580,
                'paired_zip_sha256':PAIR_SHA,'positive':6,'ordinary':8,'unknown_excluded':18,
                'first_frame_hey':first,'half_gain_positive_miss_missing_meh':missing,
                'max_first_25ms_rms_ratio':round(max(ratios),6),
                'root_cause':'NOT_PROVEN_STARTUP_HEY_SHORTCUT_SUSPECTED',
                'half_gain_frame_logits':'NOT_SAVED',
                'training':False,'model_forwards':False,'product_promotion':False,
                'human_far_frr':'NOT_MEASURED','physical_ssc305':'NOT_RUN','per_original':per}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-zip',required=True,type=Path)
    p.add_argument('--paired-zip',required=True,type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();r=audit(a.source_zip,a.paired_zip)
    text=json.dumps(r,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    else:print(text,end='')

if __name__=='__main__':main()
