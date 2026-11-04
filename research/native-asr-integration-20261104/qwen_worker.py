"""Concrete zero-forward preparation / strict native Qwen worker entry.

Runs only inside the retained offline supervised runtime. Does not set up
dependencies, register trust roots or publish evidence.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys

def require(ok, why):
    if not ok: raise ValueError(why)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

def source_bodies(code,manifest_path,loaded):
    import inspect
    import importlib.metadata
    manifest=json.loads(Path(manifest_path).read_text())
    require(manifest['schema']=='reviewed-native-source-members-v1','reviewed source manifest')
    result={}
    for row in manifest['files']:
        relative=Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'relative code source')
        path=Path(code)/relative
        require(path.is_file() and not path.is_symlink(),'regular code source')
        raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==row['sha256'],'reviewed code source bytes')
        if raw:result['bindings/code/'+relative.as_posix()]=raw
    thinker=loaded.wrapper.model.thinker
    path=Path(inspect.getsourcefile(inspect.unwrap(thinker.generate)))
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==manifest['transformers_generation_sha256'],
            'reviewed Transformers native source')
    result['bindings/native/transformers/generation/utils.py']=raw
    locks=json.loads((Path(code)/'research/qwen6_asr/model-locks.json').read_text())
    for row in locks['qwen06']['source_lock']:
        relative=Path(row['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'locked native package source path')
        path=Path(importlib.metadata.distribution(row['distribution']).locate_file(str(relative)))
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024**2,
                'bounded actual loaded package source')
        body=path.read_bytes()
        require(hashlib.sha256(body).hexdigest()==row['sha256'],'original model package source pin')
        result['bindings/native/'+relative.as_posix()]=body
    return result

def requests(path,job):
    mapping=json.loads(Path(path).read_text())
    require(set(mapping)=={'schema','clips'} and mapping['schema']=='blind-asr-source-request-map-v1',
            'blind identity map')
    rows=mapping['clips']
    require(len(rows)==len(job['clips']),'exact batch identity map')
    result={}
    for source,row in zip(job['clips'],rows):
        require(set(row)=={'audio_id','source_id','request_id','wav_sha256'} and
                row['audio_id']==source['audio_id'] and row['wav_sha256']==source['wav_sha256'],
                'identity map audio binding')
        require(row['audio_id'] not in result and all(type(row[k]) is str and row[k] for k in ('source_id','request_id')),
                'unique source/request identifiers')
        result[row['audio_id']]={k:row[k] for k in ('source_id','request_id')}
    require(len({r['request_id'] for r in result.values()})==len(rows),'unique requests')
    return result

class PreparationStopped(Exception):
    pass

def execute(args):
    code=Path(args.code)
    sys.path[:0]=[str(code/'tools'),str(code/'research/qwen-native-capture-20261104'),
                 str(code/'research/source-screen32-v1')]
    import capture_worker
    from native_sink import NativeEvidenceSink,read_json,canonical
    from asr_native_contracts import canonical_sha256
    legacy=module(code/'research/source-screen32-v1/asr_worker.py','prospective_legacy_qwen')
    require(platform.python_implementation()=='CPython' and platform.python_version()=='3.12.14'
            and platform.machine()=='x86_64' and sys.platform=='linux' and sys.byteorder=='little',
            'exact retained runtime host')
    job=read_json((Path(args.input)/'job.json').read_bytes())
    identity_map=None
    if args.phase=='infer':
        require(args.policy is not None and args.policy_sha256 is not None and args.identity_map is not None,
                'prefrozen policy and blind identity map required')
        policy_raw=Path(args.policy).read_bytes()
        policy=read_json(policy_raw)
        require(canonical_sha256(policy)==args.policy_sha256,'prefrozen full policy bytes')
        identity_map=requests(args.identity_map,job)
    state={}
    original_loader=legacy.load_qwen_after_approval
    def prepare(loaded):
        bodies=source_bodies(code,args.source_members,loaded)
        prepared=capture_worker.prepare_strict_identity(loaded,Path(args.output)/'qwen06'/'prepared',
            args.executed_head,bodies,recognizer_id='qwen06')
        for member,raw in bodies.items():
            target=Path(args.output)/'qwen06'/'prepared'/member
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        state['prepared']=prepared
        return prepared
    if args.phase=='prepare':
        def loader(*a,**kw):
            loaded=original_loader(*a,**kw)
            prepare(loaded)
            # The retained worker records honest stopped/no-forward diagnostics.
            # No successful recognition receipt is emitted for this phase.
            raise PreparationStopped('zero-forward preparation complete')
        legacy.load_qwen_after_approval=loader
        try:summary=legacy.run('qwen06',args.input,args.output,args.runtime)
        finally:legacy.load_qwen_after_approval=original_loader
        require('prepared' in state and summary['attempted']==0 and summary['success']==0,
                'actual zero-forward prepared identity required')
        return state['prepared']['identity']
    def factory(loaded,root):
        prepared=prepare(loaded)
        rows=[r for r in policy['recognizers'] if r['recognizer_id']=='qwen06']
        require(len(rows)==1 and canonical(rows[0])==canonical(prepared['identity']),
                'reloaded actual Qwen identity differs from frozen policy')
        sink=NativeEvidenceSink(root,policy_raw,prepared['model_raw'],prepared['source_raw'],
            'qwen06',prepared['source_bodies'],prepared['verified_assets'])
        for member,raw in prepared['source_bodies'].items():
            target=root/member;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        return sink
    result=capture_worker.run(legacy,'qwen06',args.input,args.output,args.runtime,
                              native_sink_factory=factory,request_map=identity_map)
    require(result['status']=='complete','strict Qwen inference stopped; no retry')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('prepare','infer'))
    for key in ('code','runtime','input','output','executed-head','source-members'):
        parser.add_argument('--'+key,required=True)
    for key in ('policy','policy-sha256','identity-map'):
        parser.add_argument('--'+key)
    execute(parser.parse_args())

if __name__=='__main__':main()
