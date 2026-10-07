"""Verify immutable saved evidence offline; never compile or call a model/decoder.

Dependencies must be explicitly prepared first. Temporary files contain only
saved data, reconstructed source and deterministic WAV bytes.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, math, shutil, struct, subprocess, sys, tempfile, zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
BINDINGS_SHA256='85e987ecd0e9ad5d69c886fcf49a2089dd7485829d13fec3820ad32d5d737f03'

def require(ok,message):
    if not ok:raise ValueError(message)
def descriptor(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(root,name,expected):
    root=Path(root);p=root/name
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'missing/nonlocal file: '+name)
    raw=p.read_bytes();require(descriptor(raw)==expected,'size/hash mismatch: '+name)
    return raw
def read(p):return json.loads(p.read_bytes())
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def jsonl(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rows))
def sha(p):return descriptor(p.read_bytes())['sha256']
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def apply_delta(raw,delta):
    require(descriptor(raw)==delta['original'],'source delta base mismatch')
    text=raw.decode();previous=0
    for e in delta['edits']:
        a,b=e['start'],e['end'];require(type(a) is int and type(b) is int and previous<=a<=b<=len(text),'overlapping source delta')
        require(text[a:b]==e['old'],'source delta old text mismatch');previous=b
    for e in reversed(delta['edits']):text=text[:e['start']]+e['new']+text[e['end']:]
    raw=text.encode();require(descriptor(raw)==delta['result'],'source delta result mismatch');return raw
def make_wav(original,leading,tail):
    raw=bytearray(original[:44]+bytes(2*leading)+original[44:]+bytes(2*tail))
    struct.pack_into('<I',raw,4,len(raw)-8);struct.pack_into('<I',raw,40,len(raw)-44)
    return bytes(raw)

def compare_token_support(computed,recorded):
    # libm may change binary64 roundoff while the checked binary32 result stays
    # identical. Exclude exactly this diagnostic; never replace its local value.
    current=copy.deepcopy(computed);historical=copy.deepcopy(recorded)
    key='max_binary64_absolute_difference'
    local_error=current['numeric_check'].pop(key)
    recorded_error=historical['numeric_check'].pop(key)
    for error in (local_error,recorded_error):
        require(type(error) in (float,int) and math.isfinite(error) and error>=0,'invalid libm diagnostic')
    require(current['numeric_check']['binary32_reference_disagreements']==0,'FP32 reference disagreement')
    require(current==historical,'M2 token support drift')
    return dict(field='numeric_check.'+key,local_value=local_error,recorded_value=recorded_error,
                scope='Platform-dependent binary64 roundoff diagnostic; not claimed reproduced. All remaining fields compare exactly; all 888 FP32 values must agree with 80-digit reference.')

@contextmanager
def staged(data_dir,dependencies):
    binding_raw=(ROOT/'metadata/bindings.json').read_bytes()
    require(hashlib.sha256(binding_raw).hexdigest()==BINDINGS_SHA256,'bindings changed')
    binding=json.loads(binding_raw)
    require({p.name for p in Path(data_dir).iterdir()}==set(binding['data_files']),'closed data file set differs')
    saved={n:checked(data_dir,n,d) for n,d in binding['data_files'].items()}
    dep={r['name']:checked(dependencies,r['name'],{k:r[k] for k in ('bytes','sha256')}) for r in binding['dependencies']}
    sources={n:checked(ROOT,n,d) for n,d in binding['copied_source'].items()}
    with tempfile.TemporaryDirectory(prefix='melo-leading-saved-') as directory:
        work=Path(directory)
        for name,raw in sources.items():p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        for name,delta in binding['source_deltas'].items():
            p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(apply_delta(dep['core/'+p.name],delta))
        for n in ('manifest.json','geometry.json'):
            p=work/'metadata'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(saved[n])
        for n in ('score_events.py','context_policy.py'):
            p=work/'scorer'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(dep['scorer/'+n])
        manifest=json.loads(saved['manifest.json'])
        receipt=dict(model_sha256=binding['acquisition_bindings']['model_sha256'],library_sha256=binding['acquisition_bindings']['library_sha256'],saved_logs={})
        for group,name,path in [('original_eof','original.raw.jsonl','controls/original_A20.raw.jsonl'),('positive300','positive300.raw.jsonl','controls/positive300/original_A20.raw.jsonl'),('negative300','negative300.raw.jsonl','controls/negative300/original_A20.raw.jsonl')]:
            raw=dep['controls/'+name];p=work/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            receipt['saved_logs'][group]={'path':path,**descriptor(raw)}
        write(work/'controls/identities.json',receipt)
        with zipfile.ZipFile(Path(dependencies)/'blind-inputs.zip') as z:
            for row in manifest['rows']:
                name=row['recording'];original=z.read(binding['archive_audio_paths'][name])
                require(hashlib.sha256(original).hexdigest()==row['original_wav_sha256'],'original WAV drift')
                for path,raw in [('controls/data/'+name+'.wav',original),('data/'+name+'.wav',make_wav(original,24000,4800)),('controls/'+('positive300' if row['label'] else 'negative300')+'/data/'+name+'.wav',make_wav(original,0,4800))]:
                    p=work/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        oldpath=sys.path[:];names=['prepare','geometry','derived_binding','compare_context'];prior={n:sys.modules.get(n) for n in names}
        sys.path.insert(0,str(work/'src'))
        for n in names:sys.modules.pop(n,None)
        try:
            yield binding,saved,dep,work,manifest
        finally:
            sys.path[:]=oldpath
            for n,value in prior.items():
                if value is None:sys.modules.pop(n,None)
                else:sys.modules[n]=value

def score_condition(work,name,rows,records,appended):
    out=work/'analysis'/name;out.mkdir(parents=True)
    references=[dict(recording=r['recording'],duration_s=r['frames']/16000,audio_sha256=r['wav_sha256'],annotation_status='complete',expected_keywords=[r['label']] if r['label'] else []) for r in rows]
    detections=[dict(recording=c['recording'],keyword_id=c['keyword'],time_s=c['available_samples']/16000,confidence=c['score']) for c in records if c.get('kind')=='callback' and c.get('state')==1]
    jsonl(out/'references.jsonl',references);jsonl(out/'detections.jsonl',detections)
    policy=dict(sample_rate_hz=16000,feed_chunk_samples=4800,appended_context_samples=appended,appended_context_kind='digital-zero' if appended else 'none',reset_frontend='per-recording',reset_model='per-recording',reset_decoder='per-recording',reset_clocks='per-recording',eof_partial_chunk='process-retained',eof_flush=False,eof_padding='none')
    write(out/'context.json',dict(schema='eval-context-v1',references_sha256=sha(out/'references.jsonl'),detections_sha256=sha(out/'detections.jsonl'),recordings={r['recording']:dict(role='positive' if r['label'] else 'negative',audio_sha256=r['wav_sha256'],policy=policy) for r in rows}))
    command=[sys.executable,'-B',str(work/'scorer/score_events.py'),'--references',str(out/'references.jsonl'),'--detections',str(out/'detections.jsonl'),'--clip-presence','--context-manifest',str(out/'context.json'),'--require-matched-context','--summary',str(out/'summary.json')]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=30)
    require(proc.returncode==0,'pinned stateless scorer failed: '+proc.stderr)
    return read(out/'summary.json')

def verify(data_dir,dependencies):
    with staged(data_dir,dependencies) as (binding,saved,dep,work,manifest):
        derived=load('leading_byte_binding',work/'src/derived_binding.py').verify_derived_inputs(manifest,work)
        integer=load('leading_integer_geometry',work/'src/geometry.py')
        geometry=json.loads(saved['geometry.json'])
        for row,g in zip(manifest['rows'],geometry['rows']):require({'recording':row['recording'],**integer.geometry(row['frames'])}==g,'integer geometry drift')
        scorer=load('leading_frozen_trace_validator',work/'src/score_endpoint.py')
        result=scorer.score(Path(data_dir)/'original_A20.raw.jsonl',work/'metadata/manifest.json',work/'metadata/geometry.json',binding['acquisition_bindings'])
        require(result['source_hashes']['raw_sha256']==binding['projections']['original_A20.raw.jsonl']['public']['sha256'],'public raw binding')
        result['source_hashes']['raw_sha256']=binding['projections']['original_A20.raw.jsonl']['original']['sha256']
        reproduced=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        require(reproduced==saved['observations.json'],'historical observations failed exact reproduction')
        report=json.loads(saved['report.json']);require(report['paired']==result['paired_context_observations'],'paired report drift')
        require(report['derived_input_binding']==derived,'leading byte binding report drift')
        originals=[];tails=[]
        for row in manifest['rows']:
            name=row['recording'];original=work/'controls/data'/f'{name}.wav';tail=work/'controls'/('positive300' if row['label'] else 'negative300')/'data'/f'{name}.wav'
            originals.append(dict(recording=name,label=row['label'],frames=row['original_frames'],wav_sha256=sha(original)))
            tails.append(dict(recording=name,label=row['label'],frames=row['original_frames']+4800,wav_sha256=sha(tail)))
        parse=lambda raw:[json.loads(line) for line in raw.splitlines()]
        records=parse(saved['original_A20.raw.jsonl'])
        conditions=[('original_eof',originals,parse(dep['controls/original.raw.jsonl']),0),('tail300',tails,parse(dep['controls/positive300.raw.jsonl'])+parse(dep['controls/negative300.raw.jsonl']),4800),('leading1500_tail300',manifest['rows'],records,4800)]
        counts=[]
        for (name,rows,raw,appended),historical in zip(conditions,report['conditions']):
            summary=score_condition(work,name,rows,raw,appended)
            require(name==historical['condition'] and summary==historical['report'],'PR485 summary drift: '+name)
            counts.append(dict(condition=name,positive_clips_with_target=summary['positive_clips_with_target'],negative_clips_with_events=summary['negative_clips_with_events']))
        platform_diagnostic=None
        if 'M2-token-support.json' in saved:
            module=load('leading_m2_token_support',ROOT/'src/m2_token_support.py')
            support=module.analyze(Path(dependencies)/'controls/original.raw.jsonl',Path(dependencies)/'controls/positive300.raw.jsonl',Path(data_dir)/'original_A20.raw.jsonl')
            platform_diagnostic=compare_token_support(support,json.loads(saved['M2-token-support.json']))
        end=result['native_run_end']
        return dict(status='PASS_SAVED_LEADING_CONTEXT',conditions=counts,callbacks=end['callbacks'],model_rows=end['model_rows'],decoder_rows_searched=end['decoder_rows_decoded'],observations_bytes_reproduced=len(reproduced),stateless_PR485_cli_invocations=3,new_model_frontend_native_decoder_calls=0,m2_platform_diagnostic=platform_diagnostic)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--dependencies',type=Path,required=True);a=p.parse_args()
    try:print(json.dumps(verify(a.data_dir,a.dependencies),sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError) as e:raise SystemExit('VERIFICATION FAILED: '+str(e))
