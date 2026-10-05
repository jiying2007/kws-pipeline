"""Offline imports and 12-sample native I/O only; no model construction or weights."""
import argparse,hashlib,importlib,importlib.metadata,io,json,math,os,pathlib,re,sys,wave,struct
ROOT=pathlib.Path(__file__).resolve().parent
STAGES={'entry','verify_no_model_assets','runtime_prepare','numpy','soundfile','scipy_signal',
        'native_io','qwen_tts','tts_transitive_native','qwen_asr','funasr','native_fbank','complete'}
PACKAGES={'numpy','soundfile','scipy','torch','transformers','qwen-tts','qwen-asr','funasr','onnxruntime','torchaudio','sox','cffi'}
PCM_INPUT=(-1.5,-1.,-.5,-3/65536,-1/65536,0.,1/65536,3/65536,.5,1-1/32768,1.,1.5)
PCM_EXPECTED=(-32768,-32768,-16384,-2,-1,0,0,1,16384,32767,32767,32767)

def require(value,message):
    if not value:raise ValueError(message)

def native_io(np,sf,profile,native_root=None):
    original=np.asarray(PCM_INPUT,dtype='<f4');buffer=io.BytesIO()
    sf.write(buffer,original,24000,format='WAV',subtype='FLOAT');float_wav=buffer.getvalue();buffer.seek(0)
    restored,rate=sf.read(buffer,dtype='float32')
    require(rate==24000 and restored.shape==(12,) and restored.astype('<f4').tobytes()==original.tobytes(),'FLOAT32 fixture mismatch')
    buffer=io.BytesIO();sf.write(buffer,np.asarray(PCM_INPUT,dtype='<f8'),16000,format='WAV',subtype='PCM_16')
    pcm_wav=buffer.getvalue();buffer.seek(0)
    with wave.open(buffer) as f:
        require((f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes())==(16000,1,2,12),'PCM16 fixture geometry')
        pcm=f.readframes(12)
    values=list(struct.unpack('<12h',pcm));require(values==list(PCM_EXPECTED),'PCM16 fixed rounding/saturation fixture mismatch')
    if profile=='tts':
        from generate_six import derivative_signal
        for invalid in ([],[float('nan')]):
            try:derivative_signal(invalid,len(invalid))
            except ValueError:pass
            else:raise ValueError('Expected empty/nonfinite rejection')
    require(hasattr(sf,'_full_path') and not hasattr(sf,'_libname'),'SoundFile used system fallback instead of bundle')
    name=pathlib.Path(os.fsdecode(sf._full_path)).resolve()
    require(name.name=='libsndfile_x86_64.so' and name.is_relative_to(pathlib.Path(native_root or sys.prefix).resolve()),'Bundled libsndfile required')
    return {'frames':12,'float32_values_sha256':hashlib.sha256(original.tobytes()).hexdigest(),
            'float_wav_sha256':hashlib.sha256(float_wav).hexdigest(),'pcm_wav_sha256':hashlib.sha256(pcm_wav).hexdigest(),
            'pcm16_values':values,'bundled_native_module':'_soundfile_data/libsndfile_x86_64.so',
            'bundled_native_sha256':hashlib.sha256(name.read_bytes()).hexdigest(),
            'libsndfile_version':sf.__libsndfile_version__,'empty_nonfinite_rejection_checked':profile=='tts'}

def execute(profile,runtime,target):
    from setup_diagnostics import safe_exception
    require(profile in ('tts','asr'),'Unknown import profile')
    require(not target.exists(),'No preflight retry')
    record={'schema':'native-import-preflight-v1','profile':profile,'stage':'entry','status':'started',
            'weight_loads':0,'model_constructions':0,'forward_calls':0,'raw_text_disclosed':False}
    def save():
        temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n');temporary.replace(target)
    def stage(name):
        require(name in STAGES,'Unknown preflight stage');record['stage']=name;save()
    save()
    try:
        stage('verify_no_model_assets')
        require(not any(p.is_file() for p in (runtime/'models').rglob('*')),'Models must not be downloaded before preflight')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                          HF_HUB_DISABLE_TELEMETRY='1',ORT_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
        for key in ('HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','NUMBA_CACHE_DIR','TORCH_HOME','MPLCONFIGDIR','TMPDIR','TORCHINDUCTOR_CACHE_DIR'):
            path=runtime/'buildtmp/import-cache'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PREFLIGHT_NETWORK_BLOCKED')
        sys.addaudithook(audit)
        stage('runtime_prepare')
        if profile=='tts':
            from runtime_gate import prepare
            torch,transformers,_=prepare()
        else:
            import torch,transformers
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
        def blocked(*args,**kwargs):raise RuntimeError('PREFLIGHT_MODEL_LOADING_PROHIBITED')
        torch.load=blocked;torch.jit.load=blocked
        transformers.PreTrainedModel.from_pretrained=blocked
        for name in ('AutoModel','AutoConfig','AutoProcessor','AutoTokenizer','AutoFeatureExtractor'):
            getattr(transformers,name).from_pretrained=blocked
        import safetensors
        safetensors.safe_open=blocked
        stage('numpy');import numpy as np
        stage('soundfile');import soundfile as sf
        stage('scipy_signal');from scipy.signal import resample_poly
        stage('native_io');record['native_io']=native_io(np,sf,profile);save()
        if profile=='tts':
            stage('qwen_tts');from qwen_tts import Qwen3TTSModel
            stage('tts_transitive_native')
            for name in ('onnxruntime','torchaudio','sox','einops'):importlib.import_module(name)
        else:
            stage('qwen_asr');from qwen_asr import Qwen3ASRModel
            stage('funasr');from funasr import AutoModel
            stage('native_fbank');from funasr.utils import fbank
            require(fbank._HAS_TORCHAUDIO is False and fbank._knf is not None,'Exact native-fbank backend required')
        versions={}
        for name in sorted(PACKAGES):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:pass
        record['versions']=versions;record.update(stage='complete',status='success');save()
    except BaseException as error:
        record.update(status='failed_no_retry',exception_type=safe_exception(error),code='IMPORT_OR_NATIVE_PREFLIGHT_FAILED');save();raise
    return record

def public_receipt(record):
    from setup_diagnostics import EXCEPTIONS
    required={'schema','profile','stage','status','weight_loads','model_constructions','forward_calls','raw_text_disclosed'}
    require(type(record) is dict and required<=set(record)<=required|{'native_io','versions','exception_type','code'},'Preflight receipt shape')
    require(record['schema']=='native-import-preflight-v1' and record['profile'] in ('tts','asr') and record['stage'] in STAGES,'Preflight scope')
    require(record['status'] in ('started','success','failed_no_retry') and record['raw_text_disclosed'] is False,'Preflight status')
    require(all(type(record[k]) is int and record[k]==0 for k in ('weight_loads','model_constructions','forward_calls')),'Preflight counts')
    if 'exception_type' in record:require(record['exception_type'] in EXCEPTIONS|{'OtherException'},'Preflight exception')
    if 'code' in record:require(record['code']=='IMPORT_OR_NATIVE_PREFLIGHT_FAILED','Preflight error code')
    if 'versions' in record:
        require(type(record['versions']) is dict and set(record['versions'])<=PACKAGES,'Preflight versions')
        require(all(type(v) is str and re.fullmatch(r'[0-9A-Za-z.+_-]{1,64}',v) for v in record['versions'].values()),'Preflight version text')
    if 'native_io' in record:
        r=record['native_io'];keys={'frames','float32_values_sha256','float_wav_sha256','pcm_wav_sha256','pcm16_values','bundled_native_module','bundled_native_sha256','libsndfile_version','empty_nonfinite_rejection_checked'}
        require(type(r) is dict and set(r)==keys and type(r['frames']) is int and r['frames']==12,'Native receipt shape')
        require(r['pcm16_values']==list(PCM_EXPECTED) and all(type(x) is int for x in r['pcm16_values']),'Native receipt samples')
        require(r['bundled_native_module']=='_soundfile_data/libsndfile_x86_64.so' and type(r['empty_nonfinite_rejection_checked']) is bool,'Native receipt scope')
        require(type(r['libsndfile_version']) is str and re.fullmatch(r'[0-9.]{1,32}',r['libsndfile_version']),'Native version text')
        require(all(type(r[k]) is str and re.fullmatch(r'[0-9a-f]{64}',r[k]) for k in keys if k.endswith('_sha256')),'Native receipt hashes')
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=('tts','asr'),required=True);parser.add_argument('--runtime',type=pathlib.Path,required=True);args=parser.parse_args()
    execute(args.profile,args.runtime.resolve(),args.runtime.resolve()/'import-preflight.json')
