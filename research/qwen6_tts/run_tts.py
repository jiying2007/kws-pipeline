"""Two bounded child stages using the previously executed ASR CPU supervisor."""
import argparse,json,os,pathlib,sys,time
from supervision import supervise
from release_gate import verify_release
from setup_diagnostics import stderr_diagnostic
ROOT=pathlib.Path(__file__).resolve().parent
GIB=1024**3

def isolated(python,script,args):
    return [str(python),'-I','-B','-c',
            'import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name="__main__")',str(ROOT/script),*map(str,args)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-reviewed-six',action='store_true');p.add_argument('--release',required=True);a=p.parse_args()
    if not a.execute_reviewed_six:raise ValueError('Reviewed execution flag required')
    release=verify_release(ROOT,'tts',a.release)
    if os.environ.get('GITHUB_RUN_ATTEMPT','1')!='1':raise ValueError('No reruns or retries')
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=False)
    started=time.monotonic();deadline=started+2400;measurements={'candidate_sha256':release['tts_candidate_sha256'],'asr_candidate_sha256':release['asr_candidate_sha256'],'preregistered_plan_sha256':release['plan_sha256'],'max_generation_calls':6,'retries':0,'recovery':release['recovery']}
    env=dict(os.environ,ORT_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
    stages=[('setup',sys.executable,'setup_locked.py',['--root',runtime],1200,2400,4*GIB,True),
            ('generation',runtime/'venv/bin/python','generate_six.py',['--model-dir',runtime/'models/qwen_tts','--out',runtime/'generation','--execute-reviewed-six'],1200,3600,7*GIB,False)]
    try:
        for name,python,script,args,wall,cpu,rss,is_setup in stages:
            stderr_path=runtime/(name+'.stderr')
            receipt=supervise(isolated(python,script,args),cwd=ROOT,env=env,wall_seconds=min(wall,deadline-time.monotonic()),cpu_seconds=cpu,rss_bytes=rss,cores=4,workspace=runtime,installed=runtime/'venv',buildtmp=runtime/'buildtmp',log_path=runtime/(name+'.log'),job_deadline=deadline,setup=is_setup,stderr_path=stderr_path)
            receipt['stderr_diagnostic']=stderr_diagnostic(stderr_path)
            measurements[name]=receipt
            if receipt['returncode']!=0:raise RuntimeError(name+' stopped: '+receipt['termination_reason'])
        from pack_blind import pack,canonical
        from generate_six import sha
        directory=runtime/'generation'
        files=[{'path':str(p.relative_to(directory)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.iterdir()) if p.is_file()]
        if len(files)!=19 or sum(r['bytes'] for r in files)>16*1024**2:raise ValueError('Generation artifact file/byte cap')
        freeze={'schema':'qwen6-generation-freeze-v1','files':files}
        (directory/'generation-freeze.json').write_bytes(canonical(freeze))
        measurements['blind']=pack(runtime/'generation',runtime/'blind-artifact')
        measurements['status']='six_generated_and_blind_exported'
    except BaseException as error:
        measurements.update(status='stopped_no_retry',error_class=type(error).__name__,error_code='TTS_SUPERVISED_STAGE_FAILED');raise
    finally:
        measurements['wall_seconds']=time.monotonic()-started
        (runtime/'resources.json').write_text(json.dumps(measurements,indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'stopped_no_retry','error_class':type(error).__name__,'error_code':'TTS_ENTRY_OR_STAGE_FAILED'}),file=sys.stderr)
        raise SystemExit(1)
