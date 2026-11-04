"""Refuse actual model calls throughout zero-audio native preparation."""
import functools
import inspect
import sys

class ForwardFence:
    def __init__(self,torch,extra=()):
        self.torch=torch;self.extra=extra;self.saved=[];self.attempts=[];self.active=False;self.metadata_entries=[];self.used=False
    def _patch(self,owner,name,label):
        if any(entry[0] is owner and entry[1]==name for entry in self.metadata_entries):return
        raw=owner.__dict__.get(name)
        if raw is None:return
        function=raw.__func__ if isinstance(raw,(classmethod,staticmethod)) else raw
        if not callable(function):return
        @functools.wraps(function)
        def forbidden(*args,**kwargs):
            self.attempts.append(label)
            raise RuntimeError('zero-forward guard rejected '+label)
        value=type(raw)(forbidden) if isinstance(raw,(classmethod,staticmethod)) else forbidden
        self.saved.append((owner,name,raw));setattr(owner,name,value)
        self.metadata_entries.append((owner,name,raw,value))
    def method_metadata(self,bound,required_globals=()):
        """Metadata only from this active fence's exact registered wrapper.

        Never follow arbitrary __wrapped__ attributes or return a callable.
        The original method remains forbidden for execution throughout prepare.
        Source-byte authentication is still performed by the consumer.
        """
        if not self.active or not inspect.ismethod(bound):
            raise ValueError('active fence bound-method metadata required')
        if type(required_globals) is not tuple or required_globals != ('kaldi',):
            raise ValueError('only fixed kaldi metadata is allowed')
        matches=[]
        for owner,name,original,installed in self.metadata_entries:
            function=installed.__func__ if isinstance(installed,(classmethod,staticmethod)) else installed
            if bound.__func__ is function:
                if not isinstance(owner,type) or not isinstance(bound.__self__,owner):
                    raise ValueError('registered method owner identity changed')
                if inspect.getattr_static(owner,name) is not installed:
                    raise ValueError('registered guard method changed')
                matches.append(original)
        if len(matches)!=1:
            raise ValueError('method not uniquely registered by this active fence')
        original=matches[0]
        function=original.__func__ if isinstance(original,(classmethod,staticmethod)) else original
        source=inspect.getsourcefile(function)
        if type(source) is not str or not source:
            raise ValueError('registered native source unavailable')
        kaldi=function.__globals__.get('kaldi')
        if kaldi is None or callable(kaldi):
            raise ValueError('noncallable kaldi metadata owner required')
        backend=getattr(kaldi,'_knf',None)
        if getattr(kaldi,'_HAS_TORCHAUDIO',None) is not False or backend is None or callable(backend):
            raise ValueError('noncallable native kaldi backend required')
        fbank=getattr(kaldi,'fbank',None)
        if not inspect.isfunction(fbank):
            raise ValueError('native fbank function metadata required')
        online=getattr(backend,'OnlineFbank',None)
        module_name=getattr(online,'__module__',None)
        if type(module_name) is not str or not module_name:
            raise ValueError('native backend module identity required')
        native=sys.modules.get(module_name)
        if native is None or callable(native):
            raise ValueError('loaded noncallable native module required')
        metadata={'source_file':source,'fbank_source':inspect.getsourcefile(fbank),
                  'backend_package_source':getattr(backend,'__file__',None),
                  'backend_binary':getattr(native,'__file__',None)}
        if any(type(value) is not str or not value or callable(value) for value in metadata.values()):
            raise ValueError('fixed plain-string metadata paths required')
        if not metadata['backend_binary'].endswith(('.so','.pyd')):
            raise ValueError('native backend binary metadata required')
        return metadata
    def __enter__(self):
        if self.active or self.used:raise ValueError('single-use forward fence')
        self.used=True;self.active=True
        self._patch(self.torch.nn.Module,'_call_impl','torch-module-call')
        pending=[self.torch.nn.Module];seen=set()
        while pending:
            cls=pending.pop()
            if cls in seen:continue
            seen.add(cls);pending.extend(cls.__subclasses__())
            for name in ('forward','generate','inference','infer','decode','transcribe'):
                self._patch(cls,name,cls.__name__+'.'+name)
        for owner,name in self.extra:self._patch(owner,name,name)
        return self
    def assert_zero(self):
        if not self.active or self.attempts:raise RuntimeError('zero-forward proof failed')
    def __exit__(self,kind,value,trace):
        for owner,name,original in reversed(self.saved):setattr(owner,name,original)
        self.active=False

def primary_loader(legacy,family,runtime,prepare_callback):
    """New candidate-zero preload; unchanged scientific loader body.

    The caller must already have verified scope, source, installed runtime and
    installed ForwardFence before calling. No job/audio/descriptor is created.
    """
    from pathlib import Path
    if family not in ('qwen06','sensevoice'):raise ValueError('primary family')
    locks=legacy.decode(legacy.read_regular(legacy.RETAINED/'model-locks.json',65536))
    lock=locks[family]
    declaration=dict(schema='zero-forward-native-preparation-v1',model_id=legacy.MODEL_IDS[family],
        candidate_count=0,audio_inputs=0,inference_allowed=False,
        asset_lock_sha256=lock['asset_lock_sha256'],source_lock_sha256=lock['source_lock_sha256'],
        expected_schema_sha256=lock['expected_schema_sha256'])
    raw=legacy.canonical(declaration)
    original=legacy._preload
    def preload(model_id,root,assets,asset_sha,sources,source_sha,decl,decl_sha,decoder_raw):
        legacy.verify_scope()
        legacy.require(model_id==declaration['model_id'] and root==str(Path(runtime)/'models'/family),
                       'exact zero-input model association')
        legacy.require(legacy.canonical(decl)==raw and decl_sha==legacy.digest(raw) and decoder_raw==b'',
                       'candidate-zero declaration, never a fake decoder/audio input')
        legacy.require(asset_sha==lock['asset_lock_sha256'] and source_sha==lock['source_lock_sha256']
            and legacy.canonical(assets)==legacy.canonical(lock['asset_lock'])
            and legacy.canonical(sources)==legacy.canonical(lock['source_lock']), 'original model/source locks')
        schema=legacy.qwen06_schema() if family=='qwen06' else legacy.sensevoice_schema()
        schema_sha=legacy.canonical_sha(schema)
        legacy.require(schema_sha==lock['expected_schema_sha256'],'independent expected tensor schema')
        asset_receipt=legacy.verify_assets(root,assets,asset_sha,model_id)
        versions=legacy.require_runtime_versions()
        legacy.verify_source_members(sources,source_sha,model_id)
        return schema,schema_sha,decl,asset_receipt,versions,{}
    legacy._preload=preload
    try:
        loader=legacy.load_qwen_after_approval if family=='qwen06' else legacy.load_sense_after_approval
        loaded=loader(str(Path(runtime)/'models'/family),lock['asset_lock'],lock['asset_lock_sha256'],
            lock['source_lock'],lock['source_lock_sha256'],declaration,legacy.digest(raw),b'')
        legacy.require(not loaded.decoder_inputs,'no decoder input may be manufactured')
        return prepare_callback(loaded)
    finally:legacy._preload=original
