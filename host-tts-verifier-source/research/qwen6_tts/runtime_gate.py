"""Narrow execution adapter: offline local assets, eager CPU float32 and audited loader calls."""
import contextlib,hashlib,importlib,importlib.util,json,os,pathlib,sys
R=pathlib.Path(__file__).resolve().parent
_active_model_roots=set()
SAFETY_ENV={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_XET':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','NUMBA_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
def require(x,msg):
 if not x:raise RuntimeError(msg)
def cache_paths():
 return {k:R/'runtime'/'buildtmp'/'inference-cache'/k.lower() for k in ['HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','NUMBA_CACHE_DIR','TORCH_HOME','MPLCONFIGDIR','TMPDIR','TORCHINDUCTOR_CACHE_DIR']}
def prepare():
 require(os.environ.get('ORT_DISABLE_TELEMETRY')=='1','ORT process-start telemetry opt-out is required before imports')
 require('onnxruntime' not in sys.modules,'ORT unexpectedly imported before guarded preparation')
 sys.dont_write_bytecode=True
 for k,v in SAFETY_ENV.items():os.environ[k]=v
 for k,p in cache_paths().items():
  p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
 events=[]
 def audit(event,args):
  if event in ['socket.connect','socket.getaddrinfo','urllib.Request']:
   events.append({'event':event,'blocked':True});raise RuntimeError('Network operation prohibited in offline TTS runtime: '+event)
 sys.addaudithook(audit)
 require(importlib.util.find_spec('qwen_asr') is None,'qwen-asr unexpectedly visible')
 require(importlib.util.find_spec('kernels') is None,'hub kernels unexpectedly visible')
 import torch
 import transformers
 from transformers.integrations import hub_kernels
 require(hub_kernels._kernels_available is False,'hub kernel decorator not a no-op')
 def blocked(*a,**k):raise RuntimeError('Excluded unsafe/unneeded execution path invoked')
 torch.load=blocked;torch.jit.load=blocked
 # These are runtime guards for excluded paths, not security patches or an OS sandbox.
 import transformers.dynamic_module_utils as dm
 dm.get_class_from_dynamic_module=blocked;dm.get_cached_module_file=blocked
 from transformers.generation import GenerationMixin
 def no_custom_generate(self,path,*args,**kwargs):
  p=pathlib.Path(path).resolve();require(p in _active_model_roots and p.is_dir(),'custom generation probe outside exact frozen local model roots');require(kwargs.get('trust_remote_code') in [None,False],'custom generation trust requested');require(kwargs.get('local_files_only') is True,'custom generation probe must be local-only');require(not (p/'custom_generate').exists() and not (p/'custom_generate').is_symlink(),'unexpected custom generation assets');raise OSError('Verified local snapshot has no custom generation code; upstream dynamic loading is disabled')
 if hasattr(GenerationMixin,'load_custom_generate'):GenerationMixin.load_custom_generate=no_custom_generate
 hub_kernels.load_and_register_kernel=blocked
 import transformers.modeling_utils as modeling_utils
 modeling_utils.load_and_register_kernel=blocked
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 return torch,transformers,events

def inspect_model_sidecars(root):
 root=pathlib.Path(root).resolve();require(root.is_dir(),'local model directory missing')
 inspected=[]
 def walk(v,path=''):
  if isinstance(v,dict):
   for k,x in v.items():
    if k=='trust_remote_code':require(x is False or x is None,'unsafe trust_remote_code flag')
    if k=='quantization_config':require(x is None,'unexpected quantization configuration')
    if k in ['auto_map','custom_generate','_attn_implementation_internal','_attn_implementation','_flash_attn_2_enabled']:
     require(x in [None,False,'eager'] and k!='auto_map','unsafe configuration key '+path+'.'+k)
    if k=='chat_template' and isinstance(x,dict):require(all('/' not in name and '\\' not in name and '..' not in name for name in x),'unsafe template key')
    walk(x,path+'.'+k)
  elif isinstance(v,list):
   for x in v:walk(x,path)
 for path in root.rglob('*.json'):
  require(path.resolve().is_relative_to(R/'model-blobs') or path.resolve().is_relative_to(root),'sidecar escaped allowlisted asset roots')
  raw=path.read_bytes();require(len(raw)<10*1024**2,'sidecar oversized');walk(json.loads(raw));inspected.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest()})
 require(not list(root.rglob('*.py')),'remote Python assets not allowed')
 return inspected

@contextlib.contextmanager
def local_loader_policy(root,torch,transformers,call_log):
 global _active_model_roots
 root=pathlib.Path(root).resolve();allowed={root,(root/'speech_tokenizer').resolve()};saved=[];previous_roots=_active_model_roots;_active_model_roots=allowed
 for name in ['AutoModel','AutoConfig','AutoProcessor','AutoFeatureExtractor','AutoTokenizer']:
  cls=getattr(transformers,name);original=cls.from_pretrained;descriptor=cls.__dict__.get('from_pretrained');saved.append((cls,descriptor))
  def wrapper(c,path,*args,_name=name,_original=original,**kwargs):
   require(isinstance(path,(str,os.PathLike)),'loader requires local path');p=pathlib.Path(path).resolve();require(p in allowed and p.is_dir(),'model loader path outside exact local allowlist')
   require(not kwargs.get('trust_remote_code',False),'remote code requested')
   kwargs.update(local_files_only=True,trust_remote_code=False,force_download=False)
   if _name=='AutoModel':
    kwargs.update(use_safetensors=True,weights_only=True,attn_implementation='eager',dtype=torch.float32,device_map='cpu')
   safe={k:str(v) for k,v in kwargs.items() if k in ['local_files_only','trust_remote_code','force_download','use_safetensors','weights_only','attn_implementation','dtype','device_map']}
   call_log.append({'loader':_name,'path':str(p),'safety_kwargs':safe})
   result=_original(str(p),*args,**kwargs)
   if _name=='AutoConfig':
    config=result[0] if isinstance(result,tuple) else result
    config._attn_implementation='eager'
    call_log[-1]['completed_config_attention']='eager_recursively'
   return result
  cls.from_pretrained=classmethod(wrapper)
 try:yield
 finally:
  _active_model_roots=previous_roots
  for cls,descriptor in reversed(saved):
   if descriptor is None:delattr(cls,'from_pretrained')
   else:setattr(cls,'from_pretrained',descriptor)
