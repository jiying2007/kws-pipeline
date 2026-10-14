"""New public-branch campaign; probe only. No ASR or historical attempts used."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'source-screen32-v1'))
import hosted_run as retained

def main():
    engine=sys.argv[1]
    if engine not in ('qwen06','qwen17'):raise ValueError('unknown engine')
    work=ROOT/('tqp-'+engine);work.mkdir(exist_ok=False)
    output=work/'output';output.mkdir()
    receipt={'campaign':'tts-quality-probe-20261014-v1','engine':engine,'max_calls':1,
             'asr_calls':0,'model_load_started':False,'retry':False,'status':'started','sha':os.environ.get('GITHUB_SHA')}
    deadline=time.monotonic()+2400
    try:
        if (os.environ.get('GITHUB_REPOSITORY')!='jiying2007/kws-pipeline' or
            os.environ.get('GITHUB_REF')!='refs/heads/research/tts-quality-probe-20261014-v1' or
            os.environ.get('GITHUB_RUN_ATTEMPT')!='1' or
            os.environ.get('GITHUB_RUN_NUMBER')!='1' or
            os.environ.get('GITHUB_EVENT_NAME')!='push'):
            raise ValueError('new campaign first-attempt push required')
        event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        if (event.get('created') is not True or event.get('deleted') is not False or
            event.get('after') != os.environ.get('GITHUB_SHA') or
            event.get('repository',{}).get('full_name') != 'jiying2007/kws-pipeline' or
            event.get('repository',{}).get('private') is not False):
            raise ValueError('new public branch required for zero-cost route')
        receipt['host']=retained.host_capacity()
        image_lock=json.loads((HERE.parent/'source-screen32-v1/container-lock.json').read_text())
        retained.pull_image(image_lock,work/'image-transfer.json')
        code=work/'code';(code/'research').mkdir(parents=True)
        for name in ('tts-quality-pilot-v1','source-screen32-v1','qwen6_tts','qwen6_asr'):
            shutil.copytree(HERE.parent/name,code/'research'/name,ignore=shutil.ignore_patterns('__pycache__','evidence','history','runtime'))
        staged=code/'research/tts-quality-pilot-v1'
        shutil.copyfile(staged/(engine+'-model-lock.json'),staged/'qwen-model-lock.json')
        runtime=work/'runtime';runtime.mkdir()
        for stage in ('setup','tts'):
            name='tqp-'+engine+'-'+stage+'-'+os.environ['GITHUB_RUN_ID']
            args=retained.container_args(name,image_lock['image'],code,runtime,output,'tts',stage)
            split=args.index(image_lock['image']);args[split:split]=['--env','TQP_ENGINE='+engine]
            start=args.index(image_lock['image'])+1
            if stage=='setup':
                args[start:]=['python','-I','-B','/code/research/tts-quality-pilot-v1/setup_adapter.py','--profile','tts','--stage','setup','--runtime','/runtime']
            else:
                args[start:]=['/runtime/venv/bin/python','-I','-B','/code/research/tts-quality-pilot-v1/worker.py']
            if stage=='tts': receipt['model_load_started']=True
            receipt[stage]=retained.supervise_container(args,name,output,stage,min(deadline,time.monotonic()+(1200 if stage=='setup' else 1500)))
        receipt['status']='probe_completed_pending_human_review'
    except BaseException as e:
        receipt['status']='stopped_no_retry';receipt['error_type']=type(e).__name__
        raise
    finally:
        retained.atomic(work/'host-receipt.json',receipt)
        # Only metadata, claims and audio of authorized public preset synthesis.
        # No packages, weights, setup logs or raw exception text are exported.
        public=work/'artifact';public.mkdir(exist_ok=True)
        for path in list(work.glob('*.json'))+list(output.glob('*.json'))+list(output.glob('*.claim'))+list(output.glob('*.wav')):
            if path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024**2:
                shutil.copyfile(path,public/path.name)
        if sum(p.stat().st_size for p in public.iterdir())>32*1024**2:raise ValueError('artifact cap exceeded')
if __name__=='__main__':
    try: main()
    except BaseException as e:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(e).__name__}),file=sys.stderr)
        raise SystemExit(1)
