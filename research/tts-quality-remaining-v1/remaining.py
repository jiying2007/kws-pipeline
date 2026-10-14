"""Inactive continuation adapter. Binding must contain real reviewed probe evidence.

No imports of inference runtimes; no downloads, model calls, or side effects at import.
"""
from pathlib import Path
import argparse,hashlib,json,math,re,struct,os,io,stat,zipfile
MAX_FILE=8*1024**2
SHA=re.compile(r'[0-9a-f]{64}\Z')
COMMIT=re.compile(r'[0-9a-f]{40}\Z')
ENGINES=('qwen06','qwen17')

def require(ok,message):
    if not ok: raise ValueError(message)

def raw(path,maximum=MAX_FILE):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'regular evidence file required')
    require(path.stat().st_size<=maximum,'evidence file too large')
    return path.read_bytes()

def digest(data): return hashlib.sha256(data).hexdigest()

def document(path):
    def pairs(rows):
        result={}
        for k,v in rows:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(raw(path),object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def filenames(engine,cell):
    return [engine+'-receipt.json',cell+'.claim',cell+'.raw-float.wav',
            'host-receipt.json','worker-status.json','container-tts.json']

def archive_members(archive,expected,incoming):
    allowed=set(expected)|{'container-setup.json','image-transfer.json','runtime-observations.json','progress.json'}
    result={}
    with zipfile.ZipFile(io.BytesIO(raw(archive,32*1024**2))) as z:
        infos=z.infolist()
        require(0<len(infos)<=32,'archive member count')
        require(sum(i.file_size for i in infos)<=32*1024**2,'archive expanded size')
        for info in infos:
            name=info.filename
            require(name in allowed and '/' not in name and '\\' not in name and not info.is_dir(),'unsafe or unexpected archive path')
            require(name not in result,'duplicate archive member')
            mode=info.external_attr>>16
            require(stat.S_IFMT(mode) in (0,stat.S_IFREG),'archive link/special member')
            require(not info.flag_bits&1 and 0<=info.file_size<=MAX_FILE,'encrypted/oversized archive member')
            body=z.read(info)
            require(len(body)==info.file_size,'archive member length')
            result[name]=body
    require(set(expected)<=set(result),'archive missing evidence')
    if incoming is not None:
        for name in expected:
            require(result[name]==raw(Path(incoming)/name),'archive/extracted evidence mismatch')
    return result

def required_sources(source_root,engine):
    root=Path(source_root)
    directories=('tts-quality-pilot-v1','source-screen32-v1','qwen6_tts','qwen6_asr')
    essentials={
      'tts-quality-pilot-v1':('pilot.py','worker.py','run_probe.py','setup_adapter.py','runtime_scope.py',engine+'-model-lock.json'),
      'source-screen32-v1':('hosted_run.py','tts_worker.py','qwen_adapter.py','contract.py','setup_adapter.py','runtime_scope.py','container-lock.json'),
      'qwen6_tts':('setup_locked.py','runtime_gate.py','runtime-lock.json','model-locks.json','setup_diagnostics.py','wheel_identity.py','import_preflight.py','generate_six.py'),
      'qwen6_asr':('setup_locked.py','runtime-lock.json','model-locks.json','setup_diagnostics.py','wheel_identity.py','import_preflight.py'),
    }
    result=set()
    for directory in directories:
        base=root/'research'/directory
        require(base.is_dir() and not base.is_symlink(),'complete source directory missing')
        for name in essentials[directory]: require((base/name).is_file(),'required loader dependency missing')
        for p in base.iterdir():
            if p.is_file() and p.suffix in ('.py','.json','.yml'):
                require(not p.is_symlink(),'source symlink prohibited')
                result.add(str(p.relative_to(root)))
    workflow='.github/workflows/tts-quality-probe-20261014-v1.yml'
    require((root/workflow).is_file(),'probe workflow source missing')
    result.add(workflow)
    return result

def binding_valid(binding,engine):
    require(binding.get('schema')=='tts-remaining-binding-v1','binding schema')
    require(binding.get('state')=='reviewed_real_probe','unfilled template; no execution')
    require(engine in ENGINES and binding.get('engine')==engine,'engine mismatch')
    require(binding.get('repository')=='jiying2007/kws-pipeline','repository mismatch')
    require(type(binding.get('probe_run_id')) is int and binding['probe_run_id']>0,'real run id required')
    require(type(binding.get('artifact_id')) is int and binding['artifact_id']>0,'real artifact id required')
    require(binding.get('probe_run_attempt')==1,'probe rerun prohibited')
    require(binding.get('probe_run_status')=='completed','probe run not terminal')
    require(binding.get('probe_run_conclusion') in ('success','failure'),'unsupported probe run conclusion')
    require(type(binding.get('probe_job_id')) is int and binding['probe_job_id']>0,'real lane job id required')
    require(binding.get('probe_job_name')=='probe ('+engine+')','wrong matrix lane job')
    require(binding.get('probe_job_status')=='completed' and binding.get('probe_job_conclusion')=='success','target lane job failed/missing')

    require(bool(COMMIT.fullmatch(binding.get('probe_source_sha',''))),'real source commit required')
    require(binding.get('workflow_path')=='.github/workflows/tts-quality-probe-20261014-v1.yml','probe workflow mismatch')
    require(binding.get('artifact_name')=='tts-quality-probe-'+engine,'artifact name mismatch')
    require(bool(SHA.fullmatch(binding.get('artifact_archive_sha256',''))),'real downloaded archive hash required')
    require(bool(SHA.fullmatch(binding.get('source_manifest_sha256',''))),'source manifest required')

def verify(binding,engine,incoming,source_root,archive,source_manifest):
    """Verify locked reviewed provenance and every probe byte before model load.

    Source manifest is root's immutable manifest of reviewed probe code; supplied
    archive is the original authenticated GitHub artifact download, not a rezip.
    Binding is reviewed configuration, never accepted from the artifact itself.
    """
    binding_valid(binding,engine)
    require(digest(raw(archive,32*1024**2))==binding['artifact_archive_sha256'],'archive identity mismatch')
    require(digest(raw(source_manifest))==binding['source_manifest_sha256'],'source manifest mismatch')
    manifest=document(source_manifest)
    require(manifest.get('source_sha')==binding['probe_source_sha'],'source commit mismatch')
    sources=manifest.get('files',{})
    require(required_sources(source_root,engine)<=set(sources),'complete staged loader dependency manifest required')
    for name,sha in sources.items():
        require(isinstance(name,str) and (name.startswith('research/') or name=='.github/workflows/tts-quality-probe-20261014-v1.yml') and '..' not in Path(name).parts and not Path(name).is_absolute(),'source path invalid')
        require(bool(SHA.fullmatch(sha)),'source hash invalid')
        require(digest(raw(Path(source_root)/name))==sha,'source bytes changed')
    import importlib.util
    module_path=Path(source_root)/'research/tts-quality-pilot-v1/pilot.py'
    spec=importlib.util.spec_from_file_location('verified_pilot',module_path)
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    cell=pilot.plan(engine)[0]
    expected=filenames(engine,cell['cell_id'])
    require(set(binding.get('files',{}))==set(expected),'exact evidence inventory required')
    incoming=Path(incoming)
    archive_members(archive,expected,incoming)
    bodies={}
    for name in expected:
        sha=binding['files'][name]
        require(bool(SHA.fullmatch(sha)),'evidence hash invalid')
        bodies[name]=raw(incoming/name)
        require(digest(bodies[name])==sha,'evidence hash mismatch')
    receipt=document(incoming/(engine+'-receipt.json'))
    require(receipt.get('status')=='probe_passed','failed or incomplete probe forbidden')
    require(receipt.get('engine')==engine and (receipt.get('model_id'),receipt.get('revision'))==pilot.MODELS[engine],'model identity mismatch')
    outcomes=receipt.get('outcomes',[])
    require(len(outcomes)==1,'probe must have exactly one attempt')
    row=outcomes[0]
    require(all(row.get(k)==v for k,v in cell.items()),'cell zero identity mismatch')
    require(row.get('status')=='generated' and row.get('attempts')==1,'probe not generated exactly once')
    require(row.get('metrics',{}).get('technically_valid') is True,'technical gate failed')
    require(bodies[cell['cell_id']+'.claim']==cell['cell_id'].encode(),'probe durable claim mismatch')
    wav=bodies[cell['cell_id']+'.raw-float.wav']
    require(digest(wav)==row.get('raw_wav_sha256'),'native WAV receipt mismatch')
    verify_native(wav,row.get('native',{}))
    host=document(incoming/'host-receipt.json')
    require(host.get('campaign')=='tts-quality-probe-20261014-v1' and host.get('engine')==engine and host.get('sha')==binding['probe_source_sha'],'host provenance mismatch')
    require(host.get('status')=='probe_completed_pending_human_review','host probe not successful')
    worker=document(incoming/'worker-status.json')
    require(worker.get('status')=='probe_completed','worker probe not successful')
    container=document(incoming/'container-tts.json')
    require(container.get('status')=='complete','container probe not successful')
    state=container.get('container_state',{})
    require(state.get('ExitCode')==0 and state.get('OOMKilled') is False,'container failed/OOM')
    return pilot,cell,bodies

def verify_native(data,native):
    require(data[:4]==b'RIFF' and data[8:12]==b'WAVE' and len(data)>=44,'native WAV header')
    require(struct.unpack('<I',data[4:8])[0]+8==len(data),'native RIFF size')
    pos=12; chunks={}
    while pos+8<=len(data):
        tag=data[pos:pos+4];size=struct.unpack('<I',data[pos+4:pos+8])[0];pos+=8
        require(pos+size<=len(data),'truncated WAV chunk')
        require(tag not in chunks,'duplicate WAV chunk');chunks[tag]=data[pos:pos+size];pos+=size+(size%2)
    require(pos==len(data) and 'FLOAT'==native.get('subtype'),'native subtype/trailing bytes')
    fmt=chunks.get(b'fmt ',b'');pcm=chunks.get(b'data',b'')
    require(len(fmt)>=16,'WAV fmt missing')
    tag,ch,rate,byte_rate,align,bits=struct.unpack('<HHIIHH',fmt[:16])
    require((tag,ch,rate,byte_rate,align,bits)==(3,1,24000,96000,4,32),'native PCM geometry mismatch')
    require(len(pcm)>0 and len(pcm)%4==0 and len(pcm)<=24000*12*4,'native sample count')
    require((native.get('sample_rate_hz'),native.get('channels'),native.get('frames'))==(24000,1,len(pcm)//4),'native receipt geometry mismatch')
    require(digest(pcm)==native.get('source_values_sha256'),'native float-values hash mismatch')
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',pcm)),'nonfinite native audio')

def prepare(binding,engine,incoming,source_root,archive,source_manifest,output):
    _,cell,bodies=verify(binding,engine,incoming,source_root,archive,source_manifest)
    output=Path(output)
    require(not output.exists(),'fresh continuation output required; no resume')
    output.mkdir(parents=True,exist_ok=False)
    # Exclusive continuation claim precedes restoring the already-consumed cell 0.
    with (output/'remaining.claim').open('x') as f:
        f.write(engine);f.flush();os.fsync(f.fileno())
    for name in (engine+'-receipt.json',cell['cell_id']+'.claim',cell['cell_id']+'.raw-float.wav'):
        with (output/name).open('xb') as f:
            f.write(bodies[name]);f.flush();os.fsync(f.fileno())
    (output/'probe-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    return {'engine':engine,'next_cell_index':1,'new_calls_max':15,'previous_calls':1}

def run_verified_loaded(model,torch,np,sf,binding,engine,incoming,source_root,archive,source_manifest,output):
    """Continuation worker hook after prepare and the existing verified model loader.

    Revalidates input/restored probe bytes, then delegates exactly cells 1..15.
    No alternative status, missing claim, or partially attempted output can resume.
    """
    pilot,cell,bodies=verify(binding,engine,incoming,source_root,archive,source_manifest)
    output=Path(output)
    require(raw(output/'remaining.claim')==engine.encode(),'continuation claim missing')
    require(document(output/'probe-binding.json')==binding,'restored binding mismatch')
    for name in (engine+'-receipt.json',cell['cell_id']+'.claim',cell['cell_id']+'.raw-float.wav'):
        require(raw(output/name)==bodies[name],'restored probe drift or lane already attempted')
    allowed_claims={'remaining.claim',cell['cell_id']+'.claim'}
    require({p.name for p in output.glob('*.claim')}==allowed_claims,'remaining cell already claimed')
    # Claim execution before the retained adapter, even if runtime/model checks fail.
    with (output/'execution.claim').open('x') as f:
        f.write(engine);f.flush();os.fsync(f.fileno())
    return pilot.run_loaded(model,torch,np,sf,engine,output,'remaining')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--engine',choices=ENGINES,required=True)
    for name in ('incoming','source-root','archive','source-manifest','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(document(a.binding),a.engine,a.incoming,a.source_root,a.archive,a.source_manifest,a.output)))
