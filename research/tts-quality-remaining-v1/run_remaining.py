"""First-push, one-shot remaining host. No token is needed or passed to containers."""
from pathlib import Path
import json,os,shutil,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'source-screen32-v1'))
import hosted_run as retained
import remaining as r
from resource_receipt import host_summary

def identity(env,event):
    expected={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/research/tts-quality-remaining-20261014-v1',
              'GITHUB_EVENT_NAME':'push','GITHUB_RUN_NUMBER':'1','GITHUB_RUN_ATTEMPT':'1'}
    r.require(all(env.get(k)==v for k,v in expected.items()),'first remaining campaign identity')
    r.require(event.get('created') is True and event.get('deleted') is False and event.get('after')==env.get('GITHUB_SHA') and
              event.get('repository',{}).get('full_name')==expected['GITHUB_REPOSITORY'] and event.get('repository',{}).get('private') is False,'public first branch event')

def export(work,output,engine):
    public=work/'artifact';public.mkdir(exist_ok=True)
    fixed={'host-receipt.json','image-transfer.json','container-setup.json','container-tts.json',
           'worker-status.json','progress.json','runtime-observations.json','probe-binding.json',
           'remaining-worker-resources.json','remaining-host-resource-status.json',
           'remaining.claim','execution.claim',engine+'-receipt.json'}
    fixed|={f'tqp-{engine}-{i:02d}{suffix}' for i in range(16) for suffix in ('.claim','.raw-float.wav')}
    for name in sorted(fixed):
        options=[p for p in (work/name,output/name) if p.is_file()]
        r.require(len(options)<=1,'duplicate public evidence source')
        if options:
            p=options[0];r.require(not p.is_symlink() and p.stat().st_size<=8*1024**2,'public file bound')
            shutil.copyfile(p,public/name)
    r.require(sum(p.stat().st_size for p in public.iterdir())<=64*1024**2,'public artifact total bound')

def main():
    engine=sys.argv[1];r.require(engine in r.ENGINES,'engine')
    work=ROOT/('tqremaining-'+engine);output=work/'output'
    receipt={'campaign':'tts-quality-remaining-20261014-v1','engine':engine,'status':'started','phase':'remaining',
             'max_new_calls':15,'prior_probe_calls':1,'max_total_calls':16,'asr_calls':0,'retry':False,
             'sha':os.environ.get('GITHUB_SHA'),'model_load_started':False}
    deadline=time.monotonic()+6000
    try:
        identity(os.environ,r.document(os.environ['GITHUB_EVENT_PATH']))
        r.require(work.is_dir(),'admission required')
        with (work/'host.claim').open('x') as f:f.write(engine);f.flush();os.fsync(f.fileno())
        admission=r.document(work/'artifact/admission-status.json')
        b=r.document(HERE/('binding-'+engine+'.json'))
        r.require(admission.get('status')=='admitted' and admission.get('engine')==engine and admission.get('archive_sha256')==b['artifact_archive_sha256'],'authenticated admission missing')
        inputs=work/'inputs'
        r.prepare(b,engine,inputs/'incoming',ROOT,inputs/'probe.zip',HERE/'source-manifest.json',output)
        receipt['host']=retained.host_capacity()
        image_lock=r.document(HERE.parent/'source-screen32-v1/container-lock.json')
        retained.pull_image(image_lock,work/'image-transfer.json')
        code=work/'code';(code/'research').mkdir(parents=True)
        for name in ('tts-quality-pilot-v1','source-screen32-v1','qwen6_tts','qwen6_asr'):
            shutil.copytree(HERE.parent/name,code/'research'/name,ignore=shutil.ignore_patterns('__pycache__','evidence','history','runtime'))
        # Preserve an exact separate read-only probe source for post-load revalidation.
        shutil.copytree(code/'research',code/'probe-source/research')
        workflow=Path('.github/workflows/tts-quality-probe-20261014-v1.yml')
        (code/'probe-source'/workflow.parent).mkdir(parents=True)
        shutil.copyfile(ROOT/workflow,code/'probe-source'/workflow)
        shutil.copytree(HERE,code/'research/tts-quality-remaining-v1',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(inputs,code/'inputs')
        staged=code/'research/tts-quality-pilot-v1'
        shutil.copyfile(staged/(engine+'-model-lock.json'),staged/'qwen-model-lock.json')
        runtime=work/'runtime';runtime.mkdir()
        for stage in ('setup','tts'):
            name='tqremaining-'+engine+'-'+stage+'-'+os.environ['GITHUB_RUN_ID']
            args=retained.container_args(name,image_lock['image'],code,runtime,output,'tts',stage)
            split=args.index(image_lock['image']);args[split:split]=['--env','TQP_ENGINE='+engine]
            start=args.index(image_lock['image'])+1
            if stage=='setup':
                args[start:]=['python','-I','-B','/code/research/tts-quality-pilot-v1/setup_adapter.py','--profile','tts','--stage','setup','--runtime','/runtime']
            else:
                args[start:]=['/runtime/venv/bin/python','-I','-B','/code/research/tts-quality-remaining-v1/worker.py']
                receipt['model_load_started']=True
            receipt[stage]=retained.supervise_container(args,name,output,stage,min(deadline,time.monotonic()+(1200 if stage=='setup' else 4800)))
        result=r.document(output/(engine+'-receipt.json'))
        r.require(result.get('status')=='generated_16_pending_review' and len(result.get('outcomes',[]))==16,
                  'remaining did not complete exact cumulative sixteen')
        r.require(all(row.get('status')=='generated' and row.get('attempts')==1 for row in result['outcomes']),
                  'remaining generation ledger incomplete')
        receipt['status']='remaining_completed_pending_human_review'
    except BaseException as e:
        receipt.update(status='stopped_no_retry',error_type=type(e).__name__);raise
    finally:
        work.mkdir(exist_ok=True)
        retained.atomic(work/'host-receipt.json',receipt)
        state=None
        if (work/'container-tts.json').is_file():state=r.document(work/'container-tts.json').get('container_state')
        output.mkdir(exist_ok=True)
        host_summary(output,container_state=state)
        export(work,output,engine)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(e).__name__}),file=sys.stderr);raise SystemExit(1)
