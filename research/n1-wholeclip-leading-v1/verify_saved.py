"""Verify saved evidence only; standard library, no native/model execution or network."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,struct,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def require(ok,why):
    if not ok:raise ValueError(why)
def check(p,row):require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'identity mismatch: '+p.name)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify(data_dir,source_dir,scorer_dir):
    data_dir,source_dir,scorer_dir=map(Path,(data_dir,source_dir,scorer_dir))
    bind=load(ROOT/'metadata/bindings.json')
    for row in bind['source_files']:check(ROOT/row['path'],row)
    for row in bind['data_files']:check(data_dir/row['path'],row)
    for row in bind['scorer']['files']:check(scorer_dir/Path(row['path']).name,row)
    preserve=load(data_dir/'source-preservation.json')
    source=source_dir/Path(preserve['original_pcm16_wav']['path']).name
    check(source,preserve['original_pcm16_wav']);original=source.read_bytes()
    require(original[:4]==b'RIFF' and original[8:20]==b'WAVEfmt \x10\0\0\0','canonical WAV header')
    require(struct.unpack_from('<HHIIHH',original,20)==(1,1,16000,32000,2,16),'mono 16k PCM16 required')
    require(original[36:40]==b'data' and len(original)==44+47556,'complete source PCM size')
    require(hashlib.sha256(original[44:]).hexdigest()==preserve['original_pcm16_wav']['pcm_sha256'],'source PCM hash')
    human=source_dir/Path(preserve['human_adjudication']['path']).name
    require(sha(human)==preserve['human_adjudication']['sha256'],'human revision hash')
    derived=bytearray(original[:44]+bytes(48000)+original[44:]+bytes(9600))
    struct.pack_into('<I',derived,4,len(derived)-8);struct.pack_into('<I',derived,40,len(derived)-44)
    require(hashlib.sha256(derived).hexdigest()==preserve['derived_wav_sha256'],'derived WAV hash')
    require(hashlib.sha256(derived[44:]).hexdigest()==preserve['derived_pcm_sha256'],'derived PCM hash')
    require(derived[48044:48044+47556]==original[44:],'unchanged complete source middle')
    g=module('n1_saved_geometry',ROOT/'src/geometry.py')
    geometry=load(data_dir/'geometry.json')
    require(dict(recording='N1',**g.geometry(52578))==geometry['rows'][0],'independent integer geometry')
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp);(t/'src').mkdir();(t/'metadata').mkdir()
        shutil.copyfile(ROOT/'src/score_endpoint.py',t/'src/score_endpoint.py')
        for name in ['manifest.json','geometry.json']:shutil.copyfile(data_dir/name,t/'metadata'/name)
        scorer=module('n1_saved_endpoint',t/'src/score_endpoint.py')
        obs=scorer.score(data_dir/'original_A20.raw.jsonl',data_dir/'manifest.json',data_dir/'geometry.json',bind['acquisition_bindings'])
        raw_mapping=next(r for r in bind['projection_mappings'] if r['public'].endswith('/original_A20.raw.jsonl'))
        require(obs['source_hashes']['raw_sha256']==raw_mapping['public_identity']['sha256'],'projected raw hash')
        obs['source_hashes']['raw_sha256']=raw_mapping['original_identity']['sha256']
        require(obs==load(data_dir/'observations.json'),'saved observations reproduce exactly with declared raw-hash normalization')
        result=t/'summary.json'
        argv=[sys.executable,'-B',str(scorer_dir/'score_events.py'),'--references',str(data_dir/'pr485-references.jsonl'),'--detections',str(data_dir/'pr485-detections.jsonl'),'--clip-presence','--context-manifest',str(data_dir/'pr485-context.json'),'--summary',str(result)]
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        require(p.returncode==0,'saved PR485 scorer failed: '+p.stderr)
        summary=load(result);require(summary==load(data_dir/'pr485-summary.json'),'unchanged PR485 summary mismatch')
        require(summary['context']['status']=='CONTEXT_UNVERIFIED' and not summary['context']['matched_context'],'single-positive context guard')
        require(summary['context']['unknown']==['both positive and negative recordings required'],'context unknowns changed')
    report=load(data_dir/'report.json')
    require(report['N1']==obs['streams'][0]['whole_clip_counts'] and report['events']==obs['streams'][0]['events'],'whole-clip report mismatch')
    require(report['PR485']['report']==summary,'report scorer summary mismatch')
    require(not report['qualification'] and not report['causal_effect_claim'] and not report['word_interval_score'] and report['word_end_latency'] is None,'unsupported claim')
    return dict(status='PASS_SAVED_ONLY',data_files=len(bind['data_files']),source_files=len(bind['source_files']),numeric_trace_reproduced=True,source_PCM_unchanged=True,context='CONTEXT_UNVERIFIED',model_frontend_decoder_calls=0,network_requests=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--n1-source-dir',required=True,type=Path);p.add_argument('--scorer-dir',required=True,type=Path);a=p.parse_args()
    print(json.dumps(verify(a.data_dir,a.n1_source_dir,a.scorer_dir),sort_keys=True))
if __name__=='__main__':main()
