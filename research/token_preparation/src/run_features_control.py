"""One bounded native32/official20/A20-one-forward preparation. No training."""
import argparse
import ctypes as C
import hashlib
from pathlib import Path
import sys
import wave
from common import check_context, read_json, require, digest, safe, write_bytes, write_json, replace_json
from geometry import geometry
from safe_failure import progress
from contracts import D20, Q12, TOKENS, STATE_SHA, cohort_weights, encode, minimum_frames

ROOT=Path(__file__).resolve().parents[1]


class NativeCall(C.Structure):
    _fields_=[('call_index',C.c_uint64),('available_samples',C.c_uint64),
        ('call_samples',C.c_size_t),('waveform_samples',C.c_size_t),('fbank_rows',C.c_size_t),
        ('splice_rows',C.c_size_t),('selected_rows',C.c_size_t),('is_final_short',C.c_uint32),
        ('wave_samples',C.c_uint32),('feature_count',C.c_uint32),('offset',C.c_uint32),
        ('centers',C.c_uint64*11)]


def bind_native(path):
    lib=C.CDLL(str(path))
    lib.a20_recipe_workspace_bytes.restype=C.c_size_t
    for name in ('a20_recipe_rows_count','a20_recipe_calls_count'):
        fn=getattr(lib,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_size_t
    lib.a20_recipe_rows_pointer.argtypes=[C.c_void_p];lib.a20_recipe_rows_pointer.restype=C.c_void_p
    lib.a20_recipe_calls_pointer.argtypes=[C.c_void_p];lib.a20_recipe_calls_pointer.restype=C.POINTER(NativeCall)
    lib.a20_recipe_execute.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.c_size_t];lib.a20_recipe_execute.restype=C.c_int
    return lib


def validate_rows(rows):
    require(len(rows)==32 and len({r['recording']for r in rows})==32,'exact32')
    require(all(r['role']=='train'for r in rows),'train only')
    cohort_weights([r['cohort']for r in rows])
    require([r['cohort']for r in rows[:20]]==[D20]*20,'D20 first')
    require([r['cohort']for r in rows[20:]]==[Q12]*12,'Qwen12 last')
    require(sum(r['model_rows']for r in rows)==1812 and max(r['model_rows']for r in rows)==95,'feature geometry')
    for r in rows:
        require(r['voice']in ('Vivian','Uncle_Fu','Dylan'),'no development voices')
        require(r['target_ids']==encode(r['text'])and len(r['target_ids'])in (3,4),'targets')
        require(r['model_rows']>=minimum_frames(r['target_ids']),'CTC feasible')
        require(r['audio_path']=='audio/'+r['recording']+'.wav','audio file scope')


def read_pcm(root,row):
    path=safe(root,row['audio_path']);require(digest(path)==row['wav_sha256'],'WAV drift')
    with wave.open(str(path))as w:
        require((w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,row['frames']),'WAV schema')
        pcm=w.readframes(w.getnframes())
    require(len(pcm)==2*row['frames']and hashlib.sha256(pcm).hexdigest()==row['pcm_sha256'],'PCM drift')
    return pcm


def native_one(lib,pcm,row):
    size=lib.a20_recipe_workspace_bytes();require(70456<size<262144,'bounded frontend workspace')
    workspace=(C.c_uint64*((size+7)//8))()
    audio=(C.c_int16*(len(pcm)//2)).from_buffer_copy(pcm)
    require(lib.a20_recipe_execute(workspace,audio,len(audio))==0,'native frontend failure')
    count=lib.a20_recipe_rows_count(workspace);calls=lib.a20_recipe_calls_count(workspace)
    require(count==row['model_rows']and calls==(row['frames']+4799)//4800,'native rows/calls')
    raw=C.string_at(lib.a20_recipe_rows_pointer(workspace),count*1600)
    source=lib.a20_recipe_calls_pointer(workspace);observed=[]
    for i in range(calls):
        c=source[i];d={name:int(getattr(c,name))for name,_ in NativeCall._fields_ if name!='centers'}
        d['centers']=list(c.centers)[:d['selected_rows']];observed.append(d)
    expected=geometry(row['frames'])['callback_plan']
    for a,b in zip(observed,expected):
        require(all(a[k]==b[k]for k in a),'native observed callback geometry')
    return raw,observed


def official_one(pcm,accept_wave,observations,row):
    import numpy as np
    import torch
    from types import SimpleNamespace
    state=SimpleNamespace(sample_rate=16000,wave_remained=np.array([]),num_mel_bins=80,
        frame_length=25,frame_shift=10,downsampling=3,context_expansion=True,left_context=2,right_context=2,
        feature_remained=None,feats_ctx_offset=0,device=torch.device('cpu'))
    chunks=[];calls=[];total_fbank=0;available=0
    for start in range(0,len(pcm),9600):
        actual=pcm[start:start+9600];sample_count=len(actual)//2
        wave_before=len(state.wave_remained)
        feat_before=0 if state.feature_remained is None else len(state.feature_remained)
        offset_before=state.feats_ctx_offset
        observations.clear()
        result=accept_wave(state,actual)
        available+=sample_count
        if observations:
            require(len(observations)==1,'one official fbank call per canonical input')
            obs=observations[0];n=obs['rows'];require(obs['samples']==wave_before+sample_count,'official waveform observation')
            splice_rows=(2 if feat_before==0 else feat_before)+n-4
            centers=[j if feat_before==0 else total_fbank-feat_before+j+2 for j in range(offset_before,splice_rows,3)]
        else:
            n=splice_rows=0;centers=[]
        selected=0 if result is None else len(result)
        require(selected==len(centers),'official selected count and derived centers')
        total_fbank+=n
        d=dict(call_index=len(calls),available_samples=available,call_samples=sample_count,
            waveform_samples=wave_before+sample_count,fbank_rows=n,splice_rows=splice_rows,selected_rows=selected,
            is_final_short=int(sample_count<4800),wave_samples=len(state.wave_remained),
            feature_count=0 if state.feature_remained is None else len(state.feature_remained),offset=state.feats_ctx_offset,
            centers=centers)
        expected=geometry(row['frames'])['callback_plan'][len(calls)]
        require(all(d[k]==expected[k]for k in d),'official observed state geometry')
        calls.append(d)
        if selected:
            require(result.dtype==torch.float32 and result.shape[1]==400 and torch.isfinite(result).all().item(),'official feature tensor')
            chunks.append(result)
    require(chunks,'official no features')
    tensor=torch.cat(chunks,0)
    require(tuple(tensor.shape)==(row['model_rows'],400),'official feature shape')
    return tensor.numpy().tobytes(),calls


def check_synthetic_ctc():
    import math
    import torch
    from contracts import ctc_nll
    probs=[[.1,.2,.3,.1,.2,.1],[.2,.2,.1,.1,.1,.3],[.3,.1,.1,.2,.2,.1],[.2,.1,.1,.1,.2,.3]]
    logs=[[math.log(p)for p in r]for r in probs]
    rows=[]
    for target in ([1,1],[1,2,1,2],[4,3,4],[]):
        got=torch.nn.functional.ctc_loss(torch.tensor(logs,dtype=torch.float64).unsqueeze(1),torch.tensor(target,dtype=torch.long),
            torch.tensor([4]),torch.tensor([len(target)]),blank=0,reduction='none',zero_infinity=False)
        error=abs(float(got[0])-ctc_nll(logs,target));require(error<=1e-11,'synthetic CTC mathematical agreement')
        rows.append(dict(target=target,absolute_error=error))
    impossible=torch.nn.functional.ctc_loss(torch.tensor(logs[:2],dtype=torch.float64).unsqueeze(1),
        torch.tensor([1,1]),torch.tensor([2]),torch.tensor([2]),blank=0,reduction='none',zero_infinity=False)
    require(torch.isinf(impossible).all().item(),'impossible CTC not zeroed')
    return dict(status='PASS',fixtures=rows,impossible_alignment_is_infinite=True,model_calls=0)


def run(context):
    check_context(ROOT,context,'features_control')
    progress(ROOT,'features_control','CONFIGURE_CPU')
    from control import configure_cpu,load_a20,initialization_control
    configure_cpu()
    import numpy as np
    import torch
    require(sys.byteorder=='little','little endian')
    out=ROOT/'work/artifact';out.mkdir(exist_ok=True)
    manifest=read_json(ROOT/'metadata/TRAIN32.json');rows=manifest['rows'];validate_rows(rows)
    source_root=ROOT/'work/inputs'
    for e in read_json(ROOT/'metadata/PUBLIC-INPUTS.json')['outputs']:
        p=safe(source_root,e['path']);require(p.stat().st_size==e['bytes']and digest(p)==e['sha256'],'input source identity')
    libpath=ROOT/'work/build/feature_collector.so'
    build=read_json(ROOT/'work/artifact/build.json');require(digest(libpath)==build['sha256'],'compiled frontend identity')
    progress(ROOT,'features_control','SYNTHETIC_CTC')
    synthetic=check_synthetic_ctc();write_json(out/'synthetic-ctc.json',synthetic)
    lib=bind_native(libpath)
    native=[];features=[]
    for r in rows:
        progress(ROOT,'features_control','NATIVE_FEATURES',r['recording'])
        pcm=read_pcm(source_root,r);raw,calls=native_one(lib,pcm,r)
        a=np.frombuffer(raw,dtype='<f4').reshape(r['model_rows'],400);require(np.isfinite(a).all(),'finite native features')
        path='native/'+r['recording']+'.f32le';write_bytes(out/path,raw,152000)
        native.append(dict(recording=r['recording'],file=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=list(a.shape),
                           wav_sha256=r['wav_sha256'],pcm_sha256=r['pcm_sha256'],normalization='PRE_CMVN',callbacks=calls))
        features.append(a.copy())
        replace_json(out/'native-features.json',dict(schema='a20-native32-feature-evidence-v1',rows=native,frontend_passes=len(native),model_calls=0,status='IN_PROGRESS'))
    replace_json(out/'native-features.json',dict(schema='a20-native32-feature-evidence-v1',rows=native,frontend_passes=32,model_calls=0))
    # Hash comparison FIRST. No historical tensors are invented or numerical gate relaxed.
    progress(ROOT,'features_control','OFFICIAL_REFERENCE_LOAD')
    from reference_feature_adapter import build_feature_only_reference
    observations=[]
    accept=build_feature_only_reference(source_root/'reference/upstream/torchaudio/kaldi.py',
        source_root/'reference/upstream/wekws/bin/stream_kws_ctc.py',observer=lambda samples,result:observations.append(dict(samples=samples,rows=len(result))))
    historical=read_json(ROOT/'metadata/HISTORICAL-D20-FEATURES.json')['rows']
    require([r['source_id']for r in historical]==[r['recording']for r in rows[:20]],'original D20 record order')
    official=[];comparisons=[]
    for i,r in enumerate(rows[:20]):
        progress(ROOT,'features_control','OFFICIAL_FEATURE_RECONSTRUCTION',r['recording'])
        raw,calls=official_one(read_pcm(source_root,r),accept,observations,r)
        h=hashlib.sha256(raw).hexdigest();old=historical[i]
        historical_match=h==old['feature_sha256']
        require(old['model_frames']==r['model_rows']and old['samples']==r['frames']and old['target']==r['target_ids'],'historical schema')
        require(old['frame_chunks']==[c['selected_rows']for c in calls if c['selected_rows']],'historical nonempty chunks')
        a=np.frombuffer(raw,dtype='<f4').reshape(r['model_rows'],400)
        path='official/'+r['recording']+'.f32le';write_bytes(out/path,raw,152000)
        official.append(dict(recording=r['recording'],file=path,bytes=len(raw),sha256=h,shape=list(a.shape),callbacks=calls,
            centers_provenance='derived from observed original accept_wave state and actual fbank rows; upstream does not emit centers',
            historical_feature_sha256=old['feature_sha256'],historical_sha_match=historical_match))
        delta=np.abs(features[i].astype(np.float64)-a.astype(np.float64))
        comparisons.append(dict(recording=r['recording'],historical_sha_match=historical_match,native_sha256=native[i]['sha256'],
            reconstruction_sha256=h,historical_sha256=old['feature_sha256'],max_absolute=float(delta.max()),
            rms_absolute=float(np.sqrt(np.mean(delta*delta))),p50_absolute=float(np.quantile(delta,.5)),
            p95_absolute=float(np.quantile(delta,.95)),p99_absolute=float(np.quantile(delta,.99)),
            equality_count=int((delta==0).sum()),elements=int(delta.size),acceptance='DESCRIPTIVE_COMPATIBILITY_ONLY_NO_NEW_TOLERANCE'))
        replace_json(out/'official-features.json',dict(schema='a20-official20-reconstruction-v1',rows=official,frontend_passes=len(official),model_calls=0,status='IN_PROGRESS'))
        replace_json(out/'feature-comparison.json',dict(schema='a20-feature-comparison-v1',rows=comparisons,status='IN_PROGRESS'))
    replace_json(out/'official-features.json',dict(schema='a20-official20-reconstruction-v1',rows=official,frontend_passes=20,model_calls=0))
    replace_json(out/'feature-comparison.json',dict(schema='a20-feature-comparison-v1',rows=comparisons,
        adopted_training_input='canonical-native PRE-CMVN, explicitly new input semantic',historical_tensors_available=False,
        official_reconstruction_is_not_math_authority=True))
    x=torch.zeros((32,95,400),dtype=torch.float32)
    for i,a in enumerate(features):x[i,:len(a)]=torch.from_numpy(a)
    bindings=read_json(ROOT/'metadata/CONTROL-BINDINGS.json')
    progress(ROOT,'features_control','LOAD_ORIGINAL_A20')
    model=load_a20(source_root,bindings)
    progress(ROOT,'features_control','CONTROL_FORWARD_ONE')
    result=initialization_control(model,x,rows)
    progress(ROOT,'features_control','SAVE_CONTROL_EVIDENCE')
    raw=result.pop('logits').contiguous().numpy().tobytes()
    require(len(raw)==72960,'one control raw size')
    write_bytes(out/'a20-initial-logits.f32le',raw,72960)
    result.update(schema='a20-one-forward-initialization-control-v1',source_order=[r['recording']for r in rows],
        raw_file='a20-initial-logits.f32le',raw_shape=[32,95,6],raw_sha256=hashlib.sha256(raw).hexdigest(),
        native_feature_hashes=[r['sha256']for r in native],valid_slices='first lengths[i] rows in each saved padded source; preserve complete padded tensor',
        torch_version=torch.__version__,numpy_version=np.__version__,python_version=sys.version.split()[0])
    write_json(out/'initial-control.json',result)
    write_json(out/'preparation-result.json',dict(schema='a20-integrated-preparation-result-v1',status='COMPLETE_PENDING_SAVED_AUDIT',
        native_waveform_passes=32,official_waveform_passes=20,model_forwards=1,optimizer_constructions=0,backwards=0,
        updates=0,decoder_calls=0,development_calls=0,F_arm_calls=0,training_authorized=False,
        checkpoint_state_sha256=STATE_SHA,source_freeze_sha256=context['source_freeze_sha256']))
    print('Preparation complete; native32 + official20 + originalA20 forward1; training remains disabled')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-context',type=Path);args=p.parse_args()
    if args.execute_context is None:raise SystemExit('PREPARATION_ONLY: no execution context')
    from safe_failure import save
    try:run(read_json(args.execute_context))
    except BaseException as error:
        save(ROOT,error,'features_control','FEATURE_CONTROL_SETUP')
        raise SystemExit(1)
