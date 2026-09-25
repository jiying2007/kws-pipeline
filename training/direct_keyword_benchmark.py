#!/usr/bin/env python3
"""Development-only whole-keyword benchmark on the shipping RNN/frontend/runtime.

The acoustic vocabulary is {blank, wake1, wake2}; keyword packs are one token per
wake. This is an architecture-control experiment, never a shipping/default change.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random, subprocess, sys, wave

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'eval'),str(ROOT/'tools')]
from model import TinyStreamingRNN
import torch
from frontend import features
from frozen_speech_ablation import verify_pool,resolve,sha,write
from startup_context_training import active_span,read_pcm
from startup_context_evaluation import pcm,wav,collect,measure,shifted
from preceding_context_evaluation import select_prior,speech_prefix
from wake_pressure_balance import keyword_target_sequences
from train_ctc import Manifest,training_environment,vocab_fingerprint,load_tokens
from training_state import state_identity
from synthetic_audio import activity_bounds

POLICY='direct-whole-keyword-final-window-v2'
PREFIX_HOPS=(0,25,50)
TAIL_SAMPLES=3200
BLANK_WEIGHT=0.30

def direct_class(row:dict)->int:
    expected=row.get('expected',[])
    if not expected:return 0
    ids={int(e['keyword_id']) for e in expected}
    if len(ids)!=1 or next(iter(ids)) not in (1,2):raise ValueError('direct benchmark requires one configured wake id')
    return next(iter(ids))

def write_vocab(root:pathlib.Path)->tuple[pathlib.Path,pathlib.Path,pathlib.Path]:
    tokens=root/'direct.tokens.txt';keywords=root/'direct.keywords.tsv';pack=root/'direct.keywords.kwk'
    tokens.write_text('<blk> 0\nwake1 1\nwake2 2\n',encoding='utf-8')
    keywords.write_text('1\t你好小窝\t0.55\twake1\n2\t小窝小窝\t0.55\twake2\n',encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'tools/compile_keywords.py'),'--tokens',str(tokens),'--keywords',str(keywords),'--out-pack',str(pack)],check=True,timeout=30)
    return tokens,keywords,pack

def frame_loss(log_probs:torch.Tensor,lengths:list[int],spans:list[tuple[int,int]],classes:list[int],end_frames:int=4)->torch.Tensor:
    T,B,V=log_probs.shape
    if V!=3 or not(len(lengths)==len(spans)==len(classes)==B):raise ValueError('invalid direct loss input')
    if type(end_frames) is not int or not 1<=end_frames<=32:raise ValueError('invalid end supervision window')
    valid=torch.zeros((T,B),dtype=torch.bool);pos=torch.zeros((T,B),dtype=torch.bool);cls=torch.tensor(classes,dtype=torch.long)
    positive=cls>0
    for j,(n,(a,b),c) in enumerate(zip(lengths,spans,classes)):
        if type(n) is not int or not 0<=a<b<=n<=T or c not in (0,1,2):raise ValueError('invalid direct loss geometry')
        valid[:n,j]=True
        if c:pos[max(a,b-end_frames):b,j]=True
    blank_nll=-log_probs[:,:,0]
    chosen=log_probs.permute(1,0,2)[torch.arange(B)[:,None],torch.arange(T)[None,:],cls[:,None]].T
    class_nll=-chosen;neg=valid&~pos
    neg_loss=(blank_nll*neg).sum(0)/neg.sum(0).clamp_min(1)
    pos_loss=(class_nll*pos).sum(0)/pos.sum(0).clamp_min(1)
    sample=torch.where(positive,pos_loss+BLANK_WEIGHT*neg_loss,neg_loss)
    return sample.mean()

def train(pool_root:pathlib.Path,pool_sha:str,output:pathlib.Path,seed:int,epochs:int,end_frames:int=4)->None:
    if type(seed) is not int or type(epochs) is not int or not 1<=epochs<=400:raise ValueError('invalid bounded direct recipe')
    if type(end_frames) is not int or not 1<=end_frames<=32:raise ValueError('invalid end supervision window')
    pool=verify_pool(pool_root,pool_sha);rows=[r for r in pool['rows'] if r['split']=='train']
    if len(rows)!=128:raise ValueError('direct benchmark expects frozen 128-row train split')
    output.mkdir(parents=True,exist_ok=False);tokens,_,_=write_vocab(output)
    manifest=output/'source-train.tsv';manifest.write_text(''.join(f"{resolve(pool_root,r['path'])}\t{' '.join(map(str,r['target_ids']))}\n" for r in rows),encoding='utf-8')
    corpus=Manifest([manifest],32,5,'logmel').corpus_identity
    cache={}
    for i,r in enumerate(rows):
        raw=read_pcm(resolve(pool_root,r['path']))
        for prefix in PREFIX_HOPS:
            signal=torch.cat((torch.zeros(prefix*320,dtype=torch.int16),raw,torch.zeros(TAIL_SAMPLES,dtype=torch.int16)))
            x=features(signal.float()/32768,feature_dim=32);cache[i,prefix]=(x,active_span(signal,len(x)),direct_class(r))
    shuffle=random.Random(seed);context=random.Random(seed+1901);ids=list(range(len(rows)))
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True);model=TinyStreamingRNN(32,64,3);initial=state_identity(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];batch_digest=hashlib.sha256();context_digest=hashlib.sha256()
    for epoch in range(1,epochs+1):
        shuffle.shuffle(ids);total=0.;batches=0
        for offset in range(0,len(ids),16):
            selected=[(i,context.choice(PREFIX_HOPS)) for i in ids[offset:offset+16]];batch_digest.update(json.dumps([epoch,[i for i,_ in selected]]).encode());context_digest.update(json.dumps([epoch,selected]).encode())
            group=[cache[k] for k in selected];xs=[g[0] for g in group]
            logits=model(torch.nn.utils.rnn.pad_sequence(xs,batch_first=True));loss=frame_loss(logits.log_softmax(-1),[len(x) for x in xs],[g[1] for g in group],[g[2] for g in group],end_frames)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();total+=float(loss.detach());batches+=1
        history.append({'epoch':epoch,'loss':total/batches,'ctc':0.0,'ordered':0.0,'margin':0.0,'completion':0.0,'release':0.0,'ordered_token_accuracy':0.0})
        if epoch%50==0:print(json.dumps({'seed':seed,'epoch':epoch,'loss':history[-1]['loss']}),flush=True)
    env=training_environment();env['training_code_sha256']['training/direct_keyword_benchmark.py']=sha(pathlib.Path(__file__))
    zero={k:0.0 for k in ('ordered_token_loss_weight','keyword_sequence_margin_loss_weight','prefix_completion_loss_weight','recurrent_release_loss_weight')}
    recipe={'policy':POLICY,'development_only':True,'release_authority':False,'pool_sha256':pool_sha,'seed':seed,'epochs':epochs,'prefix_hops':list(PREFIX_HOPS),'tail_samples':TAIL_SAMPLES,'end_supervision_frames':end_frames,'blank_weight':BLANK_WEIGHT,'threshold':0.55,'model':'shipping-rnn-32x64-logmel-vocab3'}
    cp={'state_dict':model.state_dict(),'float_state_identity':state_identity(model.state_dict()),'initial_float_state_identity':initial,'development_recipe':recipe,'batch_order_sha256':batch_digest.hexdigest(),'context_order_sha256':context_digest.hexdigest(),'feature_dim':32,'hidden_dim':64,'vocab_size':3,'vocab_fingerprint':vocab_fingerprint(load_tokens(tokens)),'tokens_sha256':sha(tokens),'frame_length_samples':400,'frame_hop_samples':320,'frontend_spec_version':2,'frontend_name':'logmel','frontend_kind':0,'training_examples':len(rows),'training_manifests':[{'name':manifest.name,'sha256':sha(manifest)}],'training_corpus_identity':corpus,'seed':seed,'epochs':epochs,'epoch_history':history,'batch_size':16,'learning_rate':.001,'optimizer':'AdamW','weight_decay':.0001,'grad_clip_norm':5.0,'training_environment':env,'auxiliary_loss_weights':zero,**zero}
    torch.save(cp,output/'model.pt');subprocess.run([sys.executable,str(ROOT/'training/export_model.py'),'--checkpoint',str(output/'model.pt'),'--tokens',str(tokens),'--output',str(output/'model.kwm')],check=True,timeout=60)
    write(output/'recipe.json',recipe)

def same_voice_pairs(rows:list[dict])->dict[int,tuple[int,int]]:
    def voice(r):return r.get('source_provenance',{}).get('voice_id')
    lookup={}
    for i,r in enumerate(rows):
        t=tuple(r['target_ids'])
        if len(t)==2 and voice(r):lookup.setdefault((voice(r),t),i)
    out={}
    for i,r in enumerate(rows):
        t=tuple(r['target_ids'])
        if len(t)==4 and voice(r):
            a=lookup.get((voice(r),t[:2]));b=lookup.get((voice(r),t[2:]))
            if a is not None and b is not None:out[i]=(a,b)
    return out

def evaluate(pool_root:pathlib.Path,pool_sha:str,model:pathlib.Path,output:pathlib.Path,runner:pathlib.Path,train_only:bool=False)->dict:
    pool=verify_pool(pool_root,pool_sha);output.mkdir(parents=True,exist_ok=False);tokens,_,pack=write_vocab(output);keywords=keyword_target_sequences(pool_root/'tokens.example.txt',pool_root/'zh_cn_example.tsv');result={'policy':POLICY,'development_only':True,'release_authority':False,'model_sha256':sha(model),'pool_sha256':pool_sha,'splits':{},'pause':{}}
    for split in (('train',) if train_only else ('train','calibration','test')):
        rows=[r for r in pool['rows'] if r['split']==split];refs=[];dets=[];prefs=[];pd=[]
        for i,r in enumerate(rows):
            raw=pcm(resolve(pool_root,r['path']));name=f'{split}-{i}';path=output/'scratch.wav';wav(path,b'\0\0'*16000+raw);ref={'recording':name,'duration_s':(16000+len(raw)//2)/16000,'expected':shifted(r['expected'],16000)};found=collect(runner,model,pack,path,name);refs.append(ref);dets+=found
            prior=select_prior(pool['rows'],r,i,keywords);pre=speech_prefix(pcm(resolve(pool_root,prior['path'])));wav(path,pre+raw);ref2={'recording':name+'p','duration_s':(len(pre)//2+len(raw)//2)/16000,'expected':shifted(r['expected'],len(pre)//2)};prefs.append(ref2);pd+=collect(runner,model,pack,path,name+'p')
        result['splits'][split]={'silence1s':measure(refs,dets),'prior':measure(prefs,pd)}
    for split in (('train',) if train_only else ('train','calibration')):
        rows=[r for r in pool['rows'] if r['split']==split];pairs=same_voice_pairs(rows);result['pause'][split]={}
        for gap in (3200,6400):
            refs=[];dets=[]
            for i,(a,b) in pairs.items():
                left=pcm(resolve(pool_root,rows[a]['path']));right=pcm(resolve(pool_root,rows[b]['path']));pad=(-len(left)//2)%320+gap;audio=left+b'\0\0'*pad+right;start,end=activity_bounds(list(__import__('struct').unpack('<'+'h'*(len(audio)//2),audio)));name=f'{split}-p-{i}';path=output/'scratch.wav';wav(path,b'\0\0'*16000+audio);exp=[{'keyword_id':e['keyword_id'],'start_s':(16000+start)/16000,'end_s':(16000+end)/16000} for e in rows[i]['expected']];ref={'recording':name,'duration_s':(16000+len(audio)//2)/16000,'expected':exp};refs.append(ref);dets+=collect(runner,model,pack,path,name)
            result['pause'][split][str(gap//16)]=measure(refs,dets)
    (output/'scratch.wav').unlink(missing_ok=True);write(output/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('train');a.add_argument('--pool',type=pathlib.Path,required=True);a.add_argument('--pool-sha',required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--epochs',type=int,default=200);a.add_argument('--end-frames',type=int,default=4)
    b=sub.add_parser('evaluate');b.add_argument('--pool',type=pathlib.Path,required=True);b.add_argument('--pool-sha',required=True);b.add_argument('--model',type=pathlib.Path,required=True);b.add_argument('--output',type=pathlib.Path,required=True);b.add_argument('--runner',type=pathlib.Path,required=True);b.add_argument('--train-only',action='store_true')
    q=p.parse_args();train(q.pool.resolve(),q.pool_sha,q.output.resolve(),q.seed,q.epochs,q.end_frames) if q.cmd=='train' else evaluate(q.pool.resolve(),q.pool_sha,q.model.resolve(),q.output.resolve(),q.runner.resolve(),q.train_only)
if __name__=='__main__':main()
