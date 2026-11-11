#!/usr/bin/env python3
"""Independent retained-byte audit of the single teacher pass. No inference."""
from __future__ import annotations
import hashlib,json,math,os,struct
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN_SHA='df2bd5b9fed99809f211672714f00a95a33db12dbc4c1475ba108f940145339d'
IDS=(0,1027,1401,1462,2093)

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    need(path.is_file() and not path.is_symlink(),'nonregular audit input')
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON');d[k]=v
        return d
    def bad(value):raise ValueError('nonfinite JSON')
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=bad)
def main():
    out=HERE/'EXTRACTION_AUDIT.json';need(not out.exists(),'fresh audit required')
    need(sha(HERE/'PLAN.json')==PLAN_SHA,'predeclared plan drift')
    original=load(HERE/'PLAN.json');execution=HERE/'extraction'
    plan=load(execution/'PLAN.json');result=load(execution/'RESULT.json');build=load(HERE/'build/BUILD_RECEIPT.json')
    need(plan['predeclared_plan_sha256']==build['predeclared_plan_sha256']==PLAN_SHA,'plan bindings')
    need(result['plan_sha256']==sha(execution/'PLAN.json') and
         plan['build_receipt_sha256']==sha(HERE/'build/BUILD_RECEIPT.json'),'result/build binding')
    need(all(plan[k]==v for k,v in original.items()),'predeclared policy changed')
    need(build['status']=='BUILT_NO_INFERENCE' and build['model_calls']==0,'build state')
    need(build['source_sha256']==original['source_sha256'],'build source record')
    for name,expected in original['source_sha256'].items():need(sha(HERE/name)==expected,'source bytes changed')
    binary=Path(build['binary']['path'])
    need(plan['binary']==build['binary'] and sha(binary)==build['binary']['sha256'],'binary binding')
    need(sha(Path(original['static_sdk']['path']))==original['static_sdk']['sha256']==build['static_sdk_sha256'],
         'static SDK binding')
    need('-Wl,--wrap=r1dfsmn_step' in build['command'] and
         (HERE/'build/stderr.log').stat().st_size==0,'build command/log')
    for kind in ('model','frontend'):
        need(sha(Path(plan[kind]['path']))==plan[kind]['sha256']==result[kind+'_sha256'],kind+' binding')
    inputs=Path(plan['input_root']);saved=Path(plan['saved_root'])
    need(sha(inputs/'INPUT_MANIFEST.json')==plan['input_manifest_sha256']==result['input_manifest_sha256'],'input manifest')
    need(sha(saved/'FILE_MANIFEST.json')==plan['saved_trace_manifest_sha256']==result['saved_trace_manifest_sha256'],'trace manifest')
    need(sha(inputs/'original-observations.jsonl')==plan['observation_labels_sha256'],'label provenance')
    rows=load(inputs/'INPUT_MANIFEST.json')['records'];traces=load(saved/'FILE_MANIFEST.json')
    observations=[json.loads(line) for line in (inputs/'original-observations.jsonl').read_text().splitlines()]
    labels={r['source_id']:r for r in observations};need(len(labels)==134,'label inventory')
    need(result['status']=='PASS_FIXED_TEACHER_EXTRACTION' and result['extraction_count']==134 and
         result['actual_frames']==12534 and result['post_input_model_frontend_hashes_unchanged'] is True,'extraction terminal')
    need(result['full_logits_persisted'] is False and result['model_promotion_authorized'] is False and
         result['training']==0 and result['physical_ssc305']==result['independent_far_frr']=='NOT_RUN','scope boundary')
    need(result['claimed']==list(range(134)) and [r['index'] for r in result['records']]==list(range(134)),
         'claim/completion denominator')
    for folder,suffix in [('hidden','.r1h'),('traces','.txt'),('receipts','.json'),('claims','.json'),('logs','.stderr')]:
        need({p.name for p in (execution/folder).iterdir()}=={f'{i:03d}{suffix}' for i in range(134)},'unexpected inventory')
    header=struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*IDS,1,0)
    parser=struct.Struct('<3Q2I146d')
    totals={'train':{'clips':0,'frames':0},'development':{'clips':0,'frames':0},'unknown':{'clips':0,'frames':0}}
    events=hidden_values=0
    for i,(source,descriptor) in enumerate(zip(rows,result['records'])):
        need(source['index']==i and descriptor['path']==f'hidden/{i:03d}.r1h','ordered record mapping')
        pcm=inputs/'pcm'/f'{i:03d}.pcm'
        need(pcm.stat().st_size==source['pcm_bytes'] and sha(pcm)==source['pcm_sha256']==descriptor['pcm_sha256'],'PCM identity')
        name=f'private-traces/{i:03d}-arm_par4.txt';reference=saved/name
        need(sha(reference)==traces[name]['sha256'] and reference.stat().st_size==traces[name]['bytes'],'baseline identity')
        trace=execution/'traces'/f'{i:03d}.txt';trace_data=trace.read_bytes()
        need(trace_data==reference.read_bytes() and sha(trace)==descriptor['trace_sha256'],'F/E parity')
        path=execution/descriptor['path'];data=path.read_bytes()
        need(len(data)==descriptor['bytes'] and sha(path)==descriptor['sha256'],'binary byte binding')
        need(data[:64]==header and len(data)>64 and (len(data)-64)%1200==0,'binary schema')
        frame_text=[];frames=0
        for frame,fields in enumerate(parser.iter_unpack(data[64:])):
            index,src,dst,best,other_id,*values=fields
            need(index==frame and src==720+480*frame and dst==min(((src+319)//320)*320,source['samples']),'clocks')
            need(all(math.isfinite(v) for v in values),'finite embedding/logit values')
            need(best<2601 and other_id<2601 and other_id not in IDS,'token identities')
            exact=values[140:145];other=values[145]
            k=min(range(5),key=lambda k:(-exact[k],IDS[k]));value=exact[k]
            want=IDS[k] if value>other else other_id if value<other else min(IDS[k],other_id)
            need(best==want,'full-winner reduction/tie identity')
            frame_text.append(f'F,{index},{src},{dst},{best}\n'.encode());frames+=1;hidden_values+=140
        need(frames==descriptor['frames']==source['feature_shape'][0]==max(0,(source['samples']-720)//480+1),'frame denominator')
        need(b''.join(frame_text)==b''.join(x for x in trace_data.splitlines(keepends=True) if x.startswith(b'F,')),
             'binary frame records vs textual original trace')
        receipt_path=execution/'receipts'/f'{i:03d}.json';receipt=load(receipt_path)
        need(sha(receipt_path)==descriptor['receipt_sha256'] and receipt['frames']==receipt['teacher_steps']==frames and
             receipt['pcm_bytes']==source['pcm_bytes'] and receipt['full_logits_persisted'] is False,'receipt bindings')
        event_rows=[r.split(b',') for r in trace_data.splitlines() if r.startswith(b'E,')]
        need(receipt['events']==len(event_rows) and receipt['keyword1_events']==sum(r[2]==b'1' for r in event_rows) and
             receipt['keyword2_events']==sum(r[2]==b'2' for r in event_rows),'event denominator')
        events+=len(event_rows)
        claim=load(execution/'claims'/f'{i:03d}.json')
        need(claim=={'index':i,'attempt':1,'argv':[str(binary),plan['model']['path'],plan['frontend']['path'],
             str(pcm),str(path),str(trace)],'pcm_sha256':source['pcm_sha256']},'actual invocation claim')
        need((execution/'logs'/f'{i:03d}.stderr').stat().st_size==0 and descriptor['returncode']==0 and
             descriptor['fe_trace_equal'] is True,'completion status')
        label=labels[source['source_id']];need(label['split']==source['split'],'split source binding')
        role='unknown' if label['ctc_weight']==0 else label['split']
        need((role=='unknown')==(i in (96,97,98,99))==descriptor['fit_excluded_unknown'],'unknown exclusion')
        totals[role]['clips']+=1;totals[role]['frames']+=frames
    need(totals=={'train':{'clips':96,'frames':9079},'development':{'clips':34,'frames':3176},
                  'unknown':{'clips':4,'frames':279}},'role denominator')
    aggregate={'schema':'r1-hidden140-independent-byte-audit-v1','status':'PASS_RETAINED_EXTRACTION_AUDIT',
        'audit_script_sha256':sha(Path(__file__)),'predeclared_plan_sha256':PLAN_SHA,
        'execution_plan_sha256':sha(execution/'PLAN.json'),'result_sha256':sha(execution/'RESULT.json'),
        'build_receipt_sha256':sha(HERE/'build/BUILD_RECEIPT.json'),'unique_clips':134,'teacher_frames':12534,
        'retained_hidden_values':hidden_values,'retained_events':events,'mismatching_fe_traces':0,
        'row_winner_and_clock_failures':0,'groups':totals,'source_input_model_frontend_drift':False,
        'unknown_clips_retained_but_excluded_from_fit':4,'full_logits_persisted':False,
        'neural_forward_calls_in_audit':0,'training_in_audit':0,
        'embedding_semantics':'stage19 copied by hash-bound wrapper using unchanged public core trace API; no new neural recomputation',
        'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN','model_promotion_authorized':False}
    with out.open('x') as f:json.dump(aggregate,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(aggregate,sort_keys=True))
if __name__=='__main__':main()
