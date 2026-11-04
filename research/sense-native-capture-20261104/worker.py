"""Prospective two-phase Sense worker. No action at import; no fallback review.

Only fixed container locations. This is source preparation, not activation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from sense_capture import (BoundSenseCapture, DEPENDENCIES, canonical, prepare_loaded, regular,
                           require, sha, write_once)

class PreparationComplete(BaseException):pass


def read_object(path,maximum=1024**2):
    from frontend_verifier import parse
    value=parse(regular(Path(path),maximum),maximum)
    require(type(value) is dict,'JSON object required')
    return value


def source_closure(review):
    require(set(review)=={'schema','executed_head_sha','dependencies','sources'} and
            review['schema']=='reviewed-sense-runtime-capture-v1','frozen Sense review manifest')
    require(type(review['executed_head_sha']) is str and re.fullmatch('[a-f0-9]{40}',review['executed_head_sha']),
            'reviewed executed source head')
    require(type(review['dependencies']) is dict and set(review['dependencies'])==DEPENDENCIES,
            'all eight pre-reviewed dependency identities required before loading')
    for name,item in review['dependencies'].items():
        require(type(item) is dict and set(item)=={'bytes','sha256'} and type(item['bytes']) is int and
                0<item['bytes']<=(32*1024**2 if name in ('backend_binary','backend_core_library') else 4*1024**2) and
                type(item['sha256']) is str and re.fullmatch('[a-f0-9]{64}',item['sha256']),
                'pre-reviewed dependency identity shape')
    require(type(review['sources']) is list and 1<=len(review['sources'])<=128,'reviewed source closure')
    bodies={}
    for row in review['sources']:
        require(type(row) is dict and set(row)=={'path','member','bytes','sha256'},'source closure row')
        path=Path(row['path']);member=row['member']
        require(path.is_absolute() and (str(path).startswith('/code/') or str(path).startswith('/runtime/')),
                'source closure fixed mounted root')
        require(type(member) is str and member and not Path(member).is_absolute() and
                '..' not in Path(member).parts and member not in bodies,'unique source archive member')
        body=regular(path,32*1024**2)
        require(type(row['bytes']) is int and len(body)==row['bytes'] and sha(body)==row['sha256'],
                'reviewed executed source bytes changed')
        bodies[member]=body
    include_retained_package_sources(review,bodies)
    return bodies


def include_retained_package_sources(review,bodies):
    """Include original pinned execution code without importing those packages."""
    import importlib.metadata
    from frontend_verifier import parse
    lock_path='/code/research/qwen6_asr/model-locks.json'
    locks=[r for r in review['sources'] if r['path']==lock_path]
    require(len(locks)==1 and locks[0]['member'] in bodies,'review-verified original Sense model lock required')
    lock_raw=bodies[locks[0]['member']]
    require(sha(lock_raw)==locks[0]['sha256'] and len(lock_raw)==locks[0]['bytes'],'original model lock identity')
    model=parse(lock_raw)['sensevoice'];rows=model['source_lock']
    expected={'funasr/auto/auto_model.py','funasr/models/sense_voice/model.py',
        'funasr/frontends/wav_frontend.py','funasr/utils/load_utils.py','funasr/utils/fbank.py',
        'funasr/models/ctc/ctc.py','funasr/tokenizer/sentencepiece_tokenizer.py'}
    require(type(rows) is list and len(rows)==len(expected) and
            sha(canonical(rows))==model['source_lock_sha256'],'original package source lock identity')
    seen=set()
    for row in rows:
        require(type(row) is dict and set(row)=={'distribution','relative_path','sha256'} and
                row['distribution']=='funasr' and row['relative_path'] in expected and
                row['relative_path'] not in seen,'exact original seven-member source closure')
        seen.add(row['relative_path'])
        path=Path(importlib.metadata.distribution(row['distribution']).locate_file(row['relative_path'])).absolute()
        require(str(path).startswith('/runtime/venv/'),'original package source under fixed runtime')
        body=regular(path,1024**2)
        require(sha(body)==row['sha256'],'original installed package source bytes changed')
        aliases=[r for r in review['sources'] if r['path']==str(path)]
        require(len(aliases)<=1,'ambiguous reviewed runtime source path')
        if aliases:
            require(bodies[aliases[0]['member']]==body,'reviewed/native package source disagreement')
        else:
            member='runtime-source/'+row['distribution']+'/'+row['relative_path']
            require(member not in bodies,'original package archive member collision')
            bodies[member]=body
    require(seen==expected,'complete original package execution closure')


def identity_map(path,job):
    value=read_object(path)
    require(set(value)=={'schema','clips'} and value['schema']=='blind-asr-source-request-map-v1' and
            type(value['clips']) is list,'frozen blind identity map')
    result={}
    wavs={row['audio_id']:row['wav_sha256'] for row in job['clips']}
    for row in value['clips']:
        require(type(row) is dict and set(row)=={'audio_id','source_id','request_id','wav_sha256'},'blind identity row')
        ident=row['audio_id']
        require(ident in wavs and ident not in result and row['wav_sha256']==wavs[ident],'exact audio identity association')
        require(all(type(row[k]) is str and 0<len(row[k])<=256 for k in ('source_id','request_id')),'explicit source/request IDs')
        result[ident]={k:row[k] for k in ('source_id','request_id')}
    require(set(result)==set(wavs),'full batch identity denominator')
    return result


def strict_identity(loaded,output,review,bodies,parser_path,metadata_groups):
    assets=loaded.receipt['assets']
    require(assets['model_id']=='FunAudioLLM/SenseVoiceSmall' and
            assets['revision']=='3847d57b6bdf2dd8875cb1508d2af43d80a16bf7','pinned Sense model identity')
    inventory=[dict(path=row['filename'],bytes=row['bytes'],sha256=row['sha256']) for row in assets['verified_files']]
    identity=dict(recognizer_id='sensevoice',provider_id='FunAudioLLM',model_id=assets['model_id'],
        model_revision=assets['revision'],model_family='sensevoice',model_sha256=sha(canonical(inventory)))
    model=dict(schema='retained-asr-model-lock-v1',assets=inventory,
        **{k:v for k,v in identity.items() if k!='recognizer_id'})
    source=dict(schema='retained-asr-source-lock-v1',executed_head_sha=review['executed_head_sha'],
        files=[dict(path=path,bytes=len(body),sha256=sha(body)) for path,body in sorted(bodies.items())])
    model_raw=canonical(model);source_raw=canonical(source)
    prepared=prepare_loaded(loaded,output,parser_path,review['dependencies'],source_raw,
        asset_lock_bytes=model_raw,max_audio_frames=192000,metadata_groups=metadata_groups)
    config=prepared['native_config']
    identity.update(native_decode_config=config,native_decode_config_sha256=sha(canonical(config)))
    write_once(Path(output)/'identity.json',identity)
    for name,raw in [('model-lock.json',model_raw),('source-lock.json',source_raw)]:
        with (Path(output)/name).open('xb') as stream:stream.write(raw);stream.flush();__import__('os').fsync(stream.fileno())
    return dict(prepared=prepared,identity=identity,model_raw=model_raw,source_raw=source_raw,
                verified_assets=inventory,source_bodies=bodies)


def persist_source_members(root,bodies):
    """Retain actual private source bytes at the exact locked archive members."""
    root=Path(root)
    require(root.is_dir() and not root.is_symlink(),'fresh strict sink directory')
    for member,body in bodies.items():
        relative=Path(member)
        require(type(member) is str and member and not relative.is_absolute() and
                all(part not in ('','.','..') for part in member.split('/')) and '\\' not in member,
                'safe private source member')
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        require(not any(parent.is_symlink() for parent in target.parents if parent!=root.parent),
                'private source member parent alias')
        with target.open('xb') as stream:
            stream.write(body);stream.flush();__import__('os').fsync(stream.fileno())
        require(sha(target.read_bytes())==sha(body),'persisted private source bytes')


def persist_native_dependencies(root,prepared,review):
    """Preserve original member identities; return assembly metadata, not trust."""
    root=Path(root);refs={}
    model=json.loads(prepared['model_raw']);source=json.loads(prepared['source_raw'])
    deps=prepared['prepared']['dependencies']
    for name,item in deps.items():
        raw=regular(Path(item['path']),32*1024**2)
        require(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'prepared dependency body changed')
        if name=='frontend_observation':
            member='frontend-observation.json'
        elif name in ('tokenizer_file','cmvn_file'):
            matches=[r for r in model['assets'] if r['path']==Path(item['original_runtime_path']).name and
                     r['bytes']==len(raw) and r['sha256']==sha(raw)]
            require(len(matches)==1,'original model dependency member association')
            member=matches[0]['path']
        else:
            matches=[r for r in review['sources'] if r['path']==item['original_runtime_path'] and
                     r['bytes']==len(raw) and r['sha256']==sha(raw)]
            require(len(matches)==1,'original executed dependency member association')
            member=matches[0]['member']
            require(sum(r['path']==member and r['bytes']==len(raw) and r['sha256']==sha(raw)
                        for r in source['files'])==1,'executed dependency inventory association')
        require(member and not Path(member).is_absolute() and '..' not in Path(member).parts,'relative dependency member')
        target=root/member
        if target.exists():require(regular(target,32*1024**2)==raw,'existing original member changed')
        else:persist_source_members(root,{member:raw})
        refs[name]=dict(member=member,sha256=sha(raw),bytes=len(raw),origin=item['origin'])
    manifest=dict(schema='sense-native-dependency-member-map-v1',archive_id=None,
        archive_assignment='pending_authenticated_host_provenance',dependencies=refs,training_admitted=False)
    write_once(root/'native-dependencies.json',manifest)
    return manifest


def execute(phase,legacy,sense_adapter,review,bodies,parser_path,metadata_groups,
            input_root,output_root,runtime_root,policy_raw=None,request_map=None,sink_class=None):
    require(phase in ('prepare','infer'),'fixed phase')
    require(phase=='prepare' or (type(policy_raw) is bytes and type(request_map) is dict and callable(sink_class)),
            'inference requires prefrozen three-ASR policy and source map')
    original_loader=legacy.load_sense_after_approval
    original_infer=sense_adapter.infer_sense_once_after_approval
    state={}
    def loader(*args,**kwargs):
        require(not state,'one loaded runtime only')
        loaded=original_loader(*args,**kwargs)
        directory=Path(output_root)/'sensevoice'
        prepared=strict_identity(loaded,directory/'prepare',review,bodies,parser_path,metadata_groups)
        state['prepared']=prepared
        if phase=='prepare':raise PreparationComplete()
        sink=sink_class(directory/'strict-native',policy_raw,prepared['model_raw'],prepared['source_raw'],
            'sensevoice',prepared['source_bodies'],prepared['verified_assets'])
        persist_source_members(directory/'strict-native',prepared['source_bodies'])
        persist_native_dependencies(directory/'strict-native',prepared,review)
        state['capture']=BoundSenseCapture(loaded,directory/'native-capture',parser_path,review['dependencies'],
            len(loaded.decoder_inputs),prepared['prepared'],sink,request_map)
        return loaded
    def infer(*args,**kwargs):
        require(not kwargs,'retained positional adapter')
        return state['capture'].infer(original_infer,*args)
    legacy.load_sense_after_approval=loader;sense_adapter.infer_sense_once_after_approval=infer
    try:
        try:summary=legacy.run('sensevoice',input_root,output_root,runtime_root)
        except PreparationComplete:
            require(phase=='prepare' and 'prepared' in state,'legitimate zero-forward stop')
            summary=dict(schema='sense-native-phase-v1',phase='prepare',status='prepared_zero_forward',
                model_forward_calls=0,training_admitted=False)
        if phase=='infer':require(summary.get('status')=='complete','Sense inference stopped; no retries')
        require('prepared' in state,'nonempty bounded preparation required')
        write_once(Path(output_root)/'phase-summary.json',summary)
        return summary
    finally:
        legacy.load_sense_after_approval=original_loader
        sense_adapter.infer_sense_once_after_approval=original_infer
        if 'capture' in state:state['capture'].restore()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','infer'),required=True)
    args=parser.parse_args()
    code=Path('/code');policy=code/'runtime-policy'
    review=read_object(policy/'sense-review.json')
    bodies=source_closure(review)  # Must precede any legacy loading.
    sys.path.insert(0,'/code/tools')
    sys.path.insert(0,'/code/research/qwen-native-capture-20261104')
    sys.path.insert(0,'/code/research/source-screen32-v1')
    spec=importlib.util.spec_from_file_location('sense_retained_worker',code/'research/source-screen32-v1/asr_worker.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    # Existing source-verified helper import resolves the exact pinned parser.
    legacy.verify_sources();legacy.import_helpers()
    import sense_adapter
    from sense_binding import _contract
    groups=_contract()['SENSE_GROUPS']
    requests=None;policy_raw=None;sink_class=None
    if args.phase=='infer':
        job=read_object(Path('/input/job.json'))
        requests=identity_map(policy/'identity-map.json',job)
        policy_raw=regular(policy/'recognizer-policy.json',8*1024**2)
        from native_sink import NativeEvidenceSink
        sink_class=NativeEvidenceSink
    return execute(args.phase,legacy,sense_adapter,review,bodies,code/'research/qwen6_asr/sense_parser.py',groups,
                   Path('/input'),Path('/output'),Path('/runtime'),policy_raw,requests,sink_class)

if __name__=='__main__':
    try:main()
    except SystemExit:
        raise
    except BaseException as error:
        print(json.dumps(dict(status='stopped_no_retry',error_type=type(error).__name__)))
        raise SystemExit(1)
