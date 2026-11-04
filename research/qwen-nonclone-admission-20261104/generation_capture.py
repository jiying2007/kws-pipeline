"""Prospective observer for actual native Qwen GenerationConfig resolution.

This never reconstructs historical configuration. Install on a newly loaded,
source-verified model before recognition. It preserves the exact returned
objects and writes observed bytes before native generation can continue.
"""
import hashlib,json,os
from pathlib import Path
FIELDS=('num_beams','num_return_sequences','forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','force_words_ids','sequence_bias','stop_strings','do_sample','max_new_tokens','eos_token_id')
EXPECTED={'num_beams':1,'num_return_sequences':1,'forced_bos_token_id':None,'forced_eos_token_id':None,'suppress_tokens':None,'begin_suppress_tokens':None,'bad_words_ids':None,'force_words_ids':None,'sequence_bias':None,'stop_strings':None,'do_sample':False,'max_new_tokens':256,'eos_token_id':[151645,151643]}

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def require(ok,why):
    if not ok:raise ValueError(why)

def snapshot(config):
    value=config.to_dict()
    require(type(value)is dict and all(k in value for k in FIELDS),'actual resolved fields must exist; no defaults')
    for key,wanted in EXPECTED.items():require(type(value[key])is type(wanted) and canonical(value[key])==canonical(wanted),'unexpected effective field: '+key)
    return canonical(value)

class ResolvedConfigRecorder:
    def __init__(self,model,output,max_calls):
        require(type(max_calls)is int and 1<=max_calls<=16,'fixed bounded decoder count')
        require(callable(getattr(model,'_prepare_generation_config',None)),'native resolver required')
        self.model=model;self.original=model._prepare_generation_config;self.output=Path(output);self.calls=0;self.max_calls=max_calls;self.first_sha=None;self.installed=False;self.poisoned=False;self.completed=0
        self.output.mkdir(exist_ok=False)
    def install(self):
        require(not self.installed,'one observer installation only')
        self.model._prepare_generation_config=self.observe;self.installed=True
    def observe(self,*args,**kwargs):
        require(self.installed and not self.poisoned and self.calls<self.max_calls,'no unbounded, failed or repeated recognition')
        self.calls+=1;self.poisoned=True
        # Calling the pinned native resolver is essential: request arguments and
        # model defaults alone are not a resolved effective configuration.
        returned=self.original(*args,**kwargs)
        require(type(returned)is tuple and len(returned)==2,'native resolver tuple')
        raw=snapshot(returned[0]);digest=hashlib.sha256(raw).hexdigest()
        require(self.first_sha is None or digest==self.first_sha,'configuration changed between clips')
        number=self.calls
        path=self.output/('resolved-generation-config-%04d.json'%number)
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        receipt={'schema':'observed-native-generation-config-v1','resolution_ordinal':number,'stage':'native_resolver_return_before_generation_continues','sha256':digest,'bytes':len(raw),'observer_synthesized_fields':False,'historical_reconstruction':False,'training_admitted':False}
        with (self.output/('resolution-%04d.json'%number)).open('xb') as f:f.write(canonical(receipt));f.flush();os.fsync(f.fileno())
        self.completed+=1;self.first_sha=digest;self.poisoned=False
        return returned
    def restore(self):
        require(self.installed,'observer not installed')
        self.model._prepare_generation_config=self.original;self.installed=False
