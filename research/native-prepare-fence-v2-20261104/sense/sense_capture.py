"""Prospective SenseVoice observations; no imports of model/numeric packages.

This does not certify native-v2 admission or establish a source trust root.
The caller supplies already-reviewed dependency hashes, never runtime defaults.
"""
from pathlib import Path
import hashlib
import inspect
import json
import math
import os
import re
import struct
import sys

FIELDS = ('fs','window','n_mels','frame_length','frame_shift','filter_length_min',
          'filter_length_max','lfr_m','lfr_n','dither','snip_edges','upsacle_samples')
REQUEST = dict(device='cpu',language='auto',use_itn=False,text_norm='woitn',fs=16000,
               data_type='sound',output_timestamp=False,ban_emo_unk=False)
DEPENDENCIES = {'frontend_source','fbank_source','backend_package_source','backend_binary','cmvn_file',
                'tokenizer_file','parser_source','backend_core_library'}


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def regular(path, maximum):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink(), 'dependency path alias')
    info=path.stat()
    require(path.is_file() and info.st_nlink==1 and 0<info.st_size<=maximum,'bounded regular dependency')
    return path.read_bytes()


def write_once(path, value):
    raw=canonical(value)
    require(len(raw)<=1024**2,'capture JSON budget')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return sha(raw)


def verify_executable_mapping(path, maps_raw):
    """Read-only identity check: this never loads or executes a shared object."""
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'mapped binary path alias')
    st=path.stat()
    require(type(maps_raw) is bytes and 0<len(maps_raw)<=4*1024**2,'bounded actual process mappings')
    expected_device=(os.major(st.st_dev),os.minor(st.st_dev))
    matched=False
    for line in maps_raw.decode('utf-8','strict').splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or fields[5]!=str(path):continue
        require(len(fields[1])==4,'mapping permissions shape')
        if 'x' not in fields[1]:continue
        try:device=tuple(int(part,16) for part in fields[3].split(':'));inode=int(fields[4])
        except ValueError as exc:raise ValueError('mapping identity fields') from exc
        require(device==expected_device and inode==st.st_ino,'mapped file inode/device mismatch')
        matched=True
    require(matched,'exact native binary executable mapping unavailable')
    return True


def mapped_binary(path):
    with Path('/proc/self/maps').open('rb') as stream:raw=stream.read(4*1024**2+1)
    return verify_executable_mapping(path,raw)


def dependency_paths(frontend, tokenizer_path, parser_path, forward_metadata=None):
    # Resolve the actual function globals used by the loaded frontend, not a
    # separately imported module with a similar name.
    forward=frontend.forward
    if forward_metadata is None:
        function=forward.__func__
        source_file=inspect.getsourcefile(function)
        kaldi=function.__globals__['kaldi']
        require(kaldi._HAS_TORCHAUDIO is False and kaldi._knf is not None,'native fbank backend required')
        backend=kaldi._knf
        native_module=sys.modules[backend.OnlineFbank.__module__]
        metadata=dict(source_file=source_file,fbank_source=inspect.getsourcefile(kaldi.fbank),
            backend_package_source=backend.__file__,backend_binary=native_module.__file__)
    else:
        metadata=forward_metadata(forward,('kaldi',))
    require(type(metadata) is dict and set(metadata)=={'source_file','fbank_source',
            'backend_package_source','backend_binary'} and
            all(type(value) is str and value for value in metadata.values()),
            'exact fixed plain-string frontend metadata')
    require(metadata['backend_binary'].endswith(('.so','.pyd')),'loaded native backend binary unavailable')
    return dict(frontend_source=Path(metadata['source_file']).absolute(),
        fbank_source=Path(metadata['fbank_source']).absolute(),
        backend_package_source=Path(metadata['backend_package_source']).absolute(),
        backend_binary=Path(metadata['backend_binary']).absolute(),
        backend_core_library=Path(metadata['backend_package_source']).absolute().parent/'lib/libkaldi-native-fbank-core.so',cmvn_file=Path(frontend.cmvn_file).absolute(),
        tokenizer_file=Path(tokenizer_path).absolute(),parser_source=Path(parser_path).absolute())


def snapshot(frontend, tokenizer, tokenizer_path, parser_path, reviewed, forward_metadata=None):
    require(type(frontend).__name__=='WavFrontend' and
            type(frontend).__module__=='funasr.frontends.wav_frontend','exact frontend class')
    require(set(reviewed)==DEPENDENCIES,'complete reviewed dependencies required')
    require(sys.byteorder=='little','little-endian runtime required')
    values={}
    for name in FIELDS:
        value=getattr(frontend,name)  # Missing observed attributes stay fatal.
        require(type(value) in (str,int,float,bool),'scalar observed frontend field')
        if type(value) is float:require(math.isfinite(value),'finite observed frontend field')
        expected_type = (str,) if name=='window' else (bool,) if name in ('snip_edges','upsacle_samples') else (int,float) if name in ('frame_length','frame_shift','dither') else (int,)
        require(type(value) in expected_type,'observed frontend field type: '+name)
        values[name]=value
    require(type(values['fs']) is int and values['fs']==16000,'observed sample rate')
    require(type(values['snip_edges']) is bool and type(values['upsacle_samples']) is bool,'observed booleans')
    cmvn=frontend.cmvn
    require(str(cmvn.device)=='cpu' and str(cmvn.dtype)=='torch.float32','CPU float32 CMVN')
    shape=list(cmvn.shape)
    require(len(shape)==2 and shape[0]==2 and type(shape[1]) is int and 0<shape[1]<=8192,'bounded CMVN shape')
    rows=cmvn.detach().cpu().tolist()
    require(type(rows) is list and len(rows)==2 and all(type(r) is list and len(r)==shape[1] for r in rows),'CMVN observed values')
    flat=[v for row in rows for v in row]
    require(all(type(v) is float and math.isfinite(v) for v in flat),'finite float32 CMVN')
    tensor_bytes=struct.pack('<%df'%len(flat),*flat)
    paths=dependency_paths(frontend,tokenizer_path,parser_path,forward_metadata)
    mappings={name:mapped_binary(paths[name]) for name in ('backend_binary','backend_core_library')}
    identities={}
    for name,path in paths.items():
        raw=regular(path,32*1024**2 if name in ('backend_binary','backend_core_library') else 4*1024**2)
        expected=reviewed[name]
        require(type(expected) is dict and set(expected)=={'sha256','bytes'},'reviewed dependency shape')
        require(type(expected['bytes']) is int and expected['bytes']==len(raw) and expected['sha256']==sha(raw),
                'reviewed dependency bytes changed: '+name)
        identities[name]=dict(expected)
        if name=='tokenizer_file':
            require(tokenizer.sp.serialized_model_proto()==raw,'loaded tokenizer differs from file')
    return dict(schema='observed-sense-frontend-v1',frontend_class=type(frontend).__module__+'.'+type(frontend).__name__,
        attributes=values,attribute_types={name:type(value).__name__ for name,value in values.items()},
        attribute_source='actual_loaded_WavFrontend_attributes_no_defaults',dependencies=identities,
        native_execution_mappings=mappings,
        cmvn=dict(device='cpu',dtype='float32',shape=shape,encoding='ieee754-binary32-le',
                  values=rows,sha256=sha(tensor_bytes)),
        observation_stage='loaded_object_observation',
        dither_rng_reproducibility='not established',training_admitted=False)


class BoundSenseCapture:
    """One hook per verified runtime. Failure consumes the call and poisons it."""
    def __init__(self, loaded, output, parser_path, reviewed, maximum, prepared=None, strict_sink=None, identity_map=None):
        require(type(maximum) is int and 1<=maximum<=64,'bounded capture count')
        self.loaded=loaded;self.output=Path(output);self.parser_path=Path(parser_path)
        self.reviewed=json.loads(canonical(reviewed));self.maximum=maximum
        self.prepared=json.loads(canonical(prepared)) if prepared is not None else None
        self.strict_sink=strict_sink
        self.identity_map=json.loads(canonical(identity_map)) if identity_map is not None else None
        require(strict_sink is None or (prepared is not None and type(identity_map) is dict),
                'strict sink requires pre-frozen preparation and identities')
        self.output.mkdir(exist_ok=False)
        self.original=loaded.wrapper.model.inference
        self.frontend=loaded.wrapper.kwargs['frontend'];self.tokenizer=loaded.wrapper.kwargs['tokenizer']
        self.tokenizer_path=loaded.tokenizer_file
        self.active=None;self.seen=set();self.dead=False;self.calls=0;self.observation=None
        self.hook=self.inference
        loaded.wrapper.model.inference=self.hook

    def restore(self):
        require(self.loaded.wrapper.model.inference is self.hook,'inference hook replaced unexpectedly')
        self.loaded.wrapper.model.inference=self.original

    def measure(self):
        return snapshot(self.frontend,self.tokenizer,self.tokenizer_path,self.parser_path,self.reviewed)

    def inference(self, *args, **kwargs):
        require(self.active is not None and not self.dead and self.calls==0,'one native call per claimed clip')
        self.calls+=1
        try:
            require(not args and set(kwargs)==set(REQUEST)|{'data_in','key','tokenizer','frontend'},'exact observed inference kwargs')
            require(kwargs['frontend'] is self.frontend,'loaded frontend identity changed')
            # CapturingTokenizer is intentionally supplied by the retained adapter.
            require(getattr(kwargs['tokenizer'],'_tokenizer',None) is self.tokenizer,
                    'retained capturing tokenizer identity')
            require(kwargs['key']==[self.active['opaque_id']],'native opaque association')
            for key,value in REQUEST.items():require(canonical(kwargs[key])==canonical(value),'native request mismatch: '+key)
            pcm=kwargs['data_in']
            require(str(pcm.dtype)=='float32' and pcm.ndim==1 and
                    len(pcm)==self.active['audio_frames'],'actual native PCM geometry')
            raw=pcm.tobytes(order='C')
            require(len(raw)==4*self.active['audio_frames'] and sha(raw)==self.active['float32_pcm_sha256'],
                    'actual native PCM bytes mismatch')
            observed=self.measure()
            if self.prepared is not None:
                require(sha(canonical(observed))==self.prepared['frontend_snapshot_sha256'],
                        'prepared frontend changed before inference')
            observed['request']=dict(REQUEST)
            observed['binding']=self.active
            target=self.output/('frontend-%04d.json'%len(self.seen))
            digest=write_once(target,observed)
            self.observation=dict(snapshot_file=target.name,frontend_snapshot_sha256=digest,
                                  native_call_started=True)
            if self.strict_sink is not None:
                config=self.prepared['native_config']
                require(self.tokenizer.sp.get_piece_size()==config['effective_config']['vocabulary_size'] and
                        self.tokenizer.sp.unk_id()==config['effective_config']['lexical_unknown_id'],
                        'prepared tokenizer semantics changed')
                self.strict_sink.start_native(config)
            result=self.original(*args,**kwargs)
            self.observation['native_inference_returned']=True
            after=self.measure()
            require(canonical(after)==canonical({k:v for k,v in observed.items() if k not in ('request','binding')}),
                    'frontend/dependencies changed during inference')
            require(sha(pcm.tobytes(order='C'))==self.active['float32_pcm_sha256'],'native PCM mutated')
            return result
        except BaseException:
            self.dead=True
            if self.strict_sink is not None:self.strict_sink.fail()
            raise

    def infer(self, original_adapter, runtime, bound, decoder_sha, model_root, state):
        require(runtime is self.loaded and not self.dead and self.active is None,'fresh same runtime')
        require(type(bound.opaque_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,80}',bound.opaque_id), 'bounded opaque identity')
        require(bound.opaque_id not in self.seen and len(self.seen)<self.maximum,'attempt consumed or budget exhausted')
        self.seen.add(bound.opaque_id);self.calls=0;self.observation=None
        try:
            d=bound.descriptor
            self.active=dict(opaque_id=bound.opaque_id,wav_sha256=d['wav']['sha256'],
                pcm_sha256=d['raw_pcm16']['sha256'],float32_pcm_sha256=d['float32_pcm']['sha256'],
                audio_frames=d['float32_pcm']['frame_count'],input_binding_sha256=bound.descriptor_sha256,
                decoder_manifest_sha256=decoder_sha)
            for key,value in self.active.items():
                if key.endswith('sha256'):require(type(value) is str and re.fullmatch('[a-f0-9]{64}',value),'binding SHA256')
            claim_sha=write_once(self.output/('claim-%04d.json'%len(self.seen)),dict(
                schema='prospective-sense-capture-claim-v1',binding=self.active,training_admitted=False))
            if self.strict_sink is not None:
                identity=self.identity_map[bound.opaque_id]
                require(set(identity)=={'source_id','request_id'},'exact frozen identity map')
                self.strict_sink.begin(bound,identity['source_id'],identity['request_id'])
            row,evidence=original_adapter(runtime,bound,decoder_sha,model_root,state)
            require(not self.dead and self.calls==1 and self.observation is not None,'exactly one persisted native capture')
            binding=dict(schema='bound-sense-frontend-observation-v1',binding=self.active,
                claim_sha256=claim_sha,**self.observation,actual_native_inference_calls=1,
                frontend_unchanged_after_inference=True,training_admitted=False)
            if self.strict_sink is not None:
                native=make_native_capture(self.prepared['native_config'],self.active,evidence,
                    self.observation['native_inference_returned'])
            completion=dict(binding,decoder_evidence_without_capture_sha256=sha(canonical(evidence)))
            write_once(self.output/('completed-%04d.json'%len(self.seen)),completion)
            if self.strict_sink is not None:
                self.strict_sink.complete_native(row['raw_text'],native,evidence)
            # The retained worker hashes this complete decoder evidence into
            # its ordinary completion receipt. This is not renamed native-v2.
            return row,dict(evidence,prospective_frontend_capture=binding)
        except BaseException:
            self.dead=True
            if self.strict_sink is not None:self.strict_sink.fail()
            raise
        finally:self.active=None


def run_wrapped(legacy, sense_adapter, input_root, output_root, runtime_root,
                parser_path, reviewed, prepared=None, strict_sink=None, identity_map=None):
    """Wrap the exact supplied retained modules; no changes to their sources.

    Both modules must already be source-verified by the integration caller.
    Reviewed dependency bytes/hashes must be frozen before this run.
    """
    original_loader=legacy.load_sense_after_approval
    original_infer=sense_adapter.infer_sense_once_after_approval
    state={}
    def loader(*args,**kwargs):
        require('capture' not in state,'one loaded Sense runtime per wrapper')
        loaded=original_loader(*args,**kwargs)
        state['capture']=BoundSenseCapture(loaded,Path(output_root)/'sensevoice/native-capture',
            parser_path,reviewed,len(loaded.decoder_inputs),prepared,strict_sink,identity_map)
        return loaded
    def infer(*args,**kwargs):
        require(not kwargs,'exact retained adapter positional call')
        return state['capture'].infer(original_infer,*args)
    legacy.load_sense_after_approval=loader
    sense_adapter.infer_sense_once_after_approval=infer
    try:
        return legacy.run('sensevoice',input_root,output_root,runtime_root)
    finally:
        legacy.load_sense_after_approval=original_loader
        sense_adapter.infer_sense_once_after_approval=original_infer
        if 'capture' in state:state['capture'].restore()


def prepare_loaded(loaded, output, parser_path, reviewed, source_lock_bytes,
                   asset_lock_bytes, max_audio_frames, metadata_groups, forward_metadata=None):
    """Observe an already-loaded runtime without any model/frontend forward.

    Return a proposed native config and private dependencies, not admission.
    Caller must source-verify metadata_groups from the pinned parser contract.
    Runtime-derived frontend bytes remain explicitly distinct from loaded assets.
    """
    require(type(source_lock_bytes) is bytes and type(asset_lock_bytes) is bytes,
            'retained original lock bytes required')
    require(type(max_audio_frames) is int and 1<=max_audio_frames<=16000*60*60,'explicit input policy budget')
    require(type(metadata_groups) in (list,tuple) and len(metadata_groups)==4,'verified parser metadata groups')
    wrapper=loaded.wrapper;frontend=wrapper.kwargs['frontend'];tokenizer=wrapper.kwargs['tokenizer']
    require(type(wrapper.kwargs['batch_size']) is int and wrapper.kwargs['batch_size']==1,'actual loaded batch size')
    parameters=list(wrapper.model.parameters())
    require(parameters and all(str(p.device)=='cpu' and str(p.dtype)=='torch.float32' for p in parameters),
            'observed CPU float32 model parameters')
    observed=snapshot(frontend,tokenizer,loaded.tokenizer_file,parser_path,reviewed,forward_metadata)
    vocabulary=tokenizer.sp.get_piece_size();unknown=tokenizer.sp.unk_id()
    require(type(vocabulary) is int and vocabulary==25055 and type(unknown) is int and unknown==0,
            'observed tokenizer semantics')
    output=Path(output);output.mkdir(exist_ok=False)
    dependencies=output/'dependencies';dependencies.mkdir()
    locations=dependency_paths(frontend,loaded.tokenizer_file,parser_path,forward_metadata)
    refs={}
    for name,path in locations.items():
        raw=regular(path,32*1024**2 if name in ('backend_binary','backend_core_library') else 4*1024**2)
        require(dict(bytes=len(raw),sha256=sha(raw))==reviewed[name],'dependency changed during preparation')
        target=dependencies/(name+'.bin')
        with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        refs[name]=dict(path=str(target),original_runtime_path=str(path),sha256=sha(raw),bytes=len(raw),origin='loaded_dependency_bytes')
    snapshot_path=dependencies/'frontend-observation.json'
    snapshot_sha=write_once(snapshot_path,observed)
    refs['frontend_observation']=dict(path=str(snapshot_path),sha256=snapshot_sha,
        bytes=snapshot_path.stat().st_size,origin='runtime_derived_observation_not_original_asset_or_executed_source')
    execution=dict(device='cpu',dtype='float32',batch_size=1)
    effective=dict(execution=execution,native_request=dict(REQUEST),decoding='full_input_greedy_ctc',
        vad_used=False,rich_postprocess_applied=False,vocabulary_size=vocabulary,lexical_unknown_id=unknown,
        metadata_prefix_tokens=len(metadata_groups),tokenizer_sha256=reviewed['tokenizer_file']['sha256'],
        parser_contract_sha256=reviewed['parser_source']['sha256'],frontend_config_sha256=snapshot_sha)
    config=dict(schema_version=2,evidence_class='speech-asr-native-config-v2',
        adapter_id='sensevoice-full-input-greedy-ctc',adapter_version=1,model_family='sensevoice',
        source_lock_sha256=sha(source_lock_bytes),asset_lock_sha256=sha(asset_lock_bytes),
        conditioning_policy='blind-no-target-or-peer-context-v1',execution=execution,
        input_policy=dict(sample_rate=16000,channels=1,max_audio_frames=max_audio_frames,
                          pcm_format='float32_pcm16_scale_32768',preprocessing='full_input'),
        native_request=dict(REQUEST),effective_config=effective,effective_config_sha256=sha(canonical(effective)))
    write_once(output/'proposed-native-config.json',config)
    result=dict(schema='prepared-sense-native-observation-v1',native_config=config,
        frontend_snapshot_sha256=snapshot_sha,dependencies=refs,forward_calls=0,
        config_semantics='expected_future_direct_inference_flow_with_observed_loaded_frontend',
        strict_resolver_dependency_membership='pending_review_for_runtime_derived_frontend',
        training_admitted=False)
    write_once(output/'preparation.json',result)
    return result


def make_native_capture(config, binding, evidence, returned):
    """Project actual retained token checks; never insert missing observations."""
    token=evidence['token_binding']
    require(returned is True and token['metadata_boundary_verified'] is True and
            token['raw_redecode_matches_capture'] is True and token['lexical_unknown_present'] is False,
            'actual completed bound CTC observations required')
    require(token['token_scope']=='Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits',
            'actual greedy CTC token scope')
    require(evidence['rich_postprocess_applied'] is False and evidence['use_itn'] is False,
            'actual unpostprocessed output')
    inp=evidence['input_binding']
    require(inp['operations']=={'mix':False,'resample':False,'trim':False} and
            inp['float32_pcm']['frame_count']==binding['audio_frames'] and
            inp['float32_pcm']['sha256']==binding['float32_pcm_sha256'], 'native captured input binding')
    require(token['tokenizer']['sha256']==config['effective_config']['tokenizer_sha256'] and
            token['parser_contract_sha256']==config['effective_config']['parser_contract_sha256'],
            'actual tokenizer/parser equals prepared config')
    return dict(schema_version=2,evidence_class='speech-asr-native-capture-v2',
        native_config_sha256=sha(canonical(config)),source_lock_sha256=config['source_lock_sha256'],
        asset_lock_sha256=config['asset_lock_sha256'],effective_config=config['effective_config'],
        effective_config_sha256=config['effective_config_sha256'],
        input=dict(sample_rate=inp['float32_pcm']['sample_rate_hz'],channels=inp['float32_pcm']['channels'],
            audio_frames=binding['audio_frames'],processed_start_frame=0,processed_end_frame=binding['audio_frames'],
            padding_frames=0,trimmed_frames=0,preprocessing='full_input',pcm_format='float32_pcm16_scale_32768'),
        termination=dict(kind='full_input_ctc_return',inference_returned=returned,
            token_scope='ctc_collapsed_nonblank_ids_not_framewise_logits',
            raw_token_ids=token['raw_token_ids'],metadata_token_ids=token['metadata_token_ids'],
            lexical_token_ids=token['lexical_token_ids'],metadata_boundary_verified=token['metadata_boundary_verified'],
            raw_redecode_matches_capture=token['raw_redecode_matches_capture'],truncated=False),
        transcript_postcorrection_applied=False,target_context_received=False,peer_transcript_received=False,
        acoustic_verbatim_completeness='unknown')
