#!/usr/bin/env python3
"""No-model synthetic checks for the separate trace facade and row adapter."""
import argparse, hashlib, io, json, os, pathlib, subprocess, sys, tempfile
HERE=pathlib.Path(__file__).resolve().parent
FLAGS=['-std=c11','-Wall','-Wextra','-Werror','-DNDEBUG','-O2','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard']
def require(ok,message):
    if not ok: raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=pathlib.Path,required=True);p.add_argument('--cc',choices=['gcc','clang'],default='gcc');a=p.parse_args();root=a.repo_root.resolve()
    # The root provides pinned source reconstruction; this script provides only new trace code.
    package=root/'research/bounded-local-retention-2026-10-20'
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [package/'candidate.patch',package/'activation/research_mode.c',package/'activation/exports.map']}
    with tempfile.TemporaryDirectory(prefix='trace-synthetic-') as tmp:
        tmp=pathlib.Path(tmp);out=tmp/'trace'
        subprocess.run([sys.executable,str(HERE/'build_trace.py'),'--mode','bounded-retention-trace-v1','--repo-root',str(root),'--output',str(out),'--cc',a.cc],check=True,stdout=subprocess.PIPE,timeout=60)
        subprocess.run([sys.executable,str(package/'activation/build_research.py'),'--mode','bounded-retention-research-v1','--repo-root',str(root),'--output',str(tmp/'aggregate'),'--cc',a.cc],check=True,stdout=subprocess.PIPE,timeout=60)
        for name in ['test_native','test_rows']:
            subprocess.run([a.cc,*FLAGS,'-I'+str(out),'-I'+str(HERE),str(HERE/(name+'.c')),str(out/'trace.so'),'-Wl,-rpath,'+str(out),'-pthread','-lm','-o',str(tmp/name)],check=True,capture_output=True,timeout=30)
        subprocess.run([str(tmp/'test_native')],check=True,timeout=30)
        result=subprocess.run([str(tmp/'test_rows')],check=True,capture_output=True,timeout=30)
        failed=subprocess.run([str(tmp/'test_rows'),'100'],capture_output=True,timeout=30)
        require(failed.returncode==2,'bounded writer must fail closed')
        # Invented wrapper is a synthetic fixture, not a scientific execution receipt.
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        require(len(rows)==3,'synthetic row count')
        staging=HERE.parents[2]
        sys.path.insert(0,str(staging))
        from tools.link_retention_events import validate_stream,COUNTERS
        header={'type':'header','schema':'retention-event-linkage-v1','run_id':'invented','recording_id':'invented-blank','arm':'candidate','threshold':.7,'retention_rows':5,'expected_rows':3,'max_bytes':1000000,'max_records':10000,'max_row_records':32768}
        for n in ['pcm_sha256','plan_sha256','candidate_sha256','library_sha256','harness_sha256','ledger_abi_sha256']:header[n]='0'*64
        footer={'type':'footer','valid':True,'dropped':0,'pending':0,'exit_code':0,'sink_closed':True,'rows_committed':3,'sequence_count':sum(len(r['records']) for r in rows),'events_emitted':0,'summary':dict.fromkeys(COUNTERS,0),'active_windows':[]};footer['summary']['rows_committed']=3
        trace=(json.dumps(header)+'\n').encode()+result.stdout+(json.dumps(footer)+'\n').encode()
        linked=validate_stream(io.BytesIO(trace));require(linked['valid'] and linked['retained_path_events']==0,'native serializer/parser integration')
        # Invented token probabilities exercise a native low -> strong episode;
        # no acoustic model, waveform or real utterance is involved.
        result=subprocess.run([str(tmp/'test_rows'),'1000000','retention'],check=True,capture_output=True,timeout=30)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        require(len(rows)==10,'native retention row count')
        header['expected_rows']=10
        footer.update(rows_committed=10,sequence_count=sum(len(r['records']) for r in rows),events_emitted=1)
        footer['summary'].update(rows_committed=10,episodes_started=1,strong_closed=1,peak_active_episodes=1)
        trace=(json.dumps(header)+'\n').encode()+result.stdout+(json.dumps(footer)+'\n').encode()
        episodes=[]
        linked=validate_stream(io.BytesIO(trace),emit_episode=episodes.append)
        require(linked['valid'] and linked['retained_path_events']==1 and len(episodes)==1,'native low/strong serializer/parser integration')
        require(rows[3]['records'][5]['kind']=='retention_started' and rows[4]['events'][0]['event_index']==0,'native retention fixture expectations')
    require(before=={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in before},'aggregate or candidate source modified')
    print(json.dumps({'status':'PASS_SOURCE_SYNTHETIC','compiler':a.cc,'real_audio_reads':0,'model_calls':0,'production_changes':False,'native_serializer_parser_integration':True,'retention_branch_native_coverage':'invented_low_to_strong'}))
if __name__=='__main__':main()
