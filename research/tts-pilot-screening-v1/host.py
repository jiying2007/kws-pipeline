"""New once-only pilot host. No historical release, claim, budget or run history."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time,importlib.util
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from screen import manifest,pack,score,third_subset,require,write,FAMILIES
import boundary as b
ROOT=HERE.parents[1]
DEADLINE=None

def stage(dest):
    lock=json.loads((HERE/'source-lock.json').read_bytes())
    for name,sha in lock['files'].items():
        raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==sha,'adapter source drift')
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    # No target labels, manifest, history, old controls or TTS code mounted.
    for name in ('blind_utils.py','whisper_worker.py'):
        p=dest/'research/tts-pilot-screening-v1'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,p)

def launch(root,code,runtime,output,image,family,input_dir=None,setup=False):
    b.host_capacity()
    output.mkdir();kind='setup' if setup else ('sensevoice' if family=='whisper' else family)
    args=b.container_args('tqp-asr-'+root.name+'-'+output.name,image,code,runtime,output,'asr',kind,(code if setup else input_dir))
    if family=='whisper':
        image_index=args.index(image)
        args=args[:image_index+1]+(['python','-I','-B','/code/research/source-screen32-whisper-v1/setup_runtime.py'] if setup else ['/runtime/venv/bin/python','-I','-B','/code/research/tts-pilot-screening-v1/whisper_worker.py'])
    return b.supervise_container(args,args[2],output,kind,min(DEADLINE or float('inf'),time.monotonic()+(1200 if setup else 4800)))

def primary_terminal(input_dir,out,family):
    # Retained read-only durable receipt validator; no model imports or old controls.
    path=ROOT/'research/source-screen32-v1/asr_worker.py'
    spec=importlib.util.spec_from_file_location('pilot_retained_terminal',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    worker.terminal_outcomes(input_dir,out,family)
    return json.loads((out/'terminal-outcomes.json').read_bytes())

def whisper_terminal(input_dir,out):
    from terminal import whisper_rows
    return whisper_rows(input_dir,out)

def collect(batch,mapping,family,out,input_dir):
    raw=whisper_terminal(input_dir,out) if family=='whisper' else primary_terminal(input_dir,out,family)
    idx={r['opaque_id']:r for r in raw}
    results=[]
    for m in mapping:
        if m['batch']!=batch: continue
        r=idx.get(m['opaque_id'],{})
        require(not r or r.get('wav_sha256')==m['wav_sha256'],'result audio identity')
        results.append({'sample_id':m['sample_id'],'family':family,'attempt':1,'status':r.get('status','not_run'),'raw_text':r.get('raw_text'),'completeness':r.get('completeness','unknown'),'quality_flags':r.get('quality_flags',[])})
    return results

def run(manifest_path,audio_root,root):
    global DEADLINE
    DEADLINE=time.monotonic()+21600
    require(not root.exists(),'fresh output; no reruns')
    root.mkdir();data=manifest(manifest_path);results=[]
    try:
        b.host_capacity();require(shutil.disk_usage(root).free>=25*1024**3,'25 GiB disk reserve')
        code=root/'code';stage(code)
        image_lock=json.loads((ROOT/'research/source-screen32-v1/container-lock.json').read_bytes());b.pull_image(image_lock,root/'image.json');image=image_lock['image']
        primary=root/'primary';mapping=pack(data,audio_root,primary)
        runtime=root/'runtime';runtime.mkdir()
        if mapping:
            launch(root,code,runtime,root/'setup-primary',image,'qwen06',setup=True)
            for family in FAMILIES[:2]:
                for batch in sorted({m['batch'] for m in mapping}):
                    out=root/(family+'-'+batch)
                    try: launch(root,code,runtime,out,image,family,primary/batch)
                    except Exception:
                        results.extend(collect(batch,mapping,family,out,primary/batch));break
                    results.extend(collect(batch,mapping,family,out,primary/batch))
        subset=set(third_subset(data,results))
        if subset:
            third=root/'third';third_map=pack(data,audio_root,third,subset)
            wr=root/'whisper-runtime';wr.mkdir()
            launch(root,code,wr,root/'setup-whisper',image,'whisper',setup=True)
            for batch in sorted({m['batch'] for m in third_map}):
                out=root/('whisper-'+batch)
                try: launch(root,code,wr,out,image,'whisper',third/batch)
                except Exception:
                    results.extend(collect(batch,third_map,'whisper',out,third/batch));break
                results.extend(collect(batch,third_map,'whisper',out,third/batch))
    finally:
        write(root/'results.json',results);write(root/'report.json',score(data,results))

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--audio-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.manifest,a.audio_root,Path(a.output).absolute())
if __name__=='__main__':main()
