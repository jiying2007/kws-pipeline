#!/usr/bin/env python3
"""Runner-only dependency/API gate. No model/source assets are opened or downloaded."""
import argparse, hashlib, importlib, importlib.metadata as md, json, math, os, pathlib, platform, subprocess, sys, tempfile, time, traceback
if sys.flags.optimize: raise RuntimeError('Optimized Python bypasses qualification assertions and is forbidden')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    start=time.monotonic();raw=a.lock.read_bytes();lock=json.loads(raw);tests={};attempts=[]
    report={'schema':'cosyvoice3.runtime-qualification.v1','status':'failed','lock_sha256':hashlib.sha256(raw).hexdigest(),'python_executable':sys.executable,'python_version':platform.python_version(),'platform':platform.platform(),'versions':{},'tests':tests,'network_attempts':attempts}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        assert os.environ.get('GITHUB_ACTIONS')=='true','Runner-only qualification; not approved for current host'
        assert sys.prefix!=sys.base_prefix and not (pathlib.Path(sys.prefix)/'pyvenv.cfg').read_text().lower().count('include-system-site-packages = true')
        assert platform.python_version()=='3.12.14' and platform.system()=='Linux' and platform.machine()=='x86_64'
        p=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True,timeout=120)
        report['pip_check']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,p.stdout+p.stderr
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.specifiers import SpecifierSet
        selected={x['name']:x for x in lock['packages']};active=[]
        distributions=list(md.distributions());installed={canonicalize_name(d.metadata['Name']):d for d in distributions};assert len(distributions)==len(installed),'Duplicate distribution identities across sys.path'
        assert all(pathlib.Path(d.locate_file('')).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()) for d in distributions),'Distribution outside the isolated venv'
        assert set(installed)==set(selected),f'Unclosed environment: extra={set(installed)-set(selected)}, missing={set(selected)-set(installed)}'
        for n,x in selected.items():
            dist=installed[n];assert dist.version==x['version'],(n,dist.version,x['version']);report['versions'][n]=dist.version
            rp=dist.metadata.get('Requires-Python');assert not rp or SpecifierSet(rp).contains('3.12.14'),(n,rp)
            if x.get('metadata_sha256'):
                metadata_files=[f for f in dist.files or [] if str(f).endswith('.dist-info/METADATA') and len(pathlib.PurePosixPath(str(f)).parts)==2];assert len(metadata_files)==1,f'{n}: expected exactly one top-level installed METADATA';actual=dist.locate_file(metadata_files[0]).read_bytes();assert hashlib.sha256(actual).hexdigest()==x['metadata_sha256'],f'{n}: installed METADATA differs from locked receipt'
            for raw_req in dist.requires or []:
                req=Requirement(raw_req);contexts=['']+x['selected_extras']
                enabled=not req.marker or any(req.marker.evaluate({**lock['environment'],'extra':e}) for e in contexts)
                if enabled:
                    name=canonicalize_name(req.name);assert name in selected and req.specifier.contains(selected[name]['version']),(n,raw_req)
                    assert set(req.extras)<=set(selected[name]['selected_extras']),(n,raw_req,'missing extras')
                    active.append({'from':n,'requirement':raw_req,'satisfied':True})
        report['installed_active_requirement_checks']=active;tests['exact_closed_distribution_set']=True;tests['installed_metadata_constraints']=True
        os.environ.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLBACKEND':'Agg','NUMBA_NUM_THREADS':'4'})
        def audit(event,args):
            if event in {'socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.sendto'}:
                attempts.append({'event':event,'args':repr(args)[:1000]});raise RuntimeError('Offline runtime qualification forbids networking')
        sys.addaudithook(audit)
        import numpy as np, torch, torchaudio, soundfile as sf, whisper, pyworld
        torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available();tests['cpu_only']=True
        for module in ['conformer','diffusers','einops','gdown','hydra','hyperpyyaml','inflect','librosa','lightning','matplotlib','modelscope','omegaconf','onnxruntime','pyarrow','regex','rich','scipy','tiktoken','tqdm','transformers','wget','x_transformers','pytorch_lightning']:
            importlib.import_module(module)
        tests['all_direct_roots_import']=True
        t=np.arange(16000,dtype=np.float64)/16000;signal=(0.2*np.sin(2*np.pi*220*t)).astype(np.float32);wav=torch.from_numpy(signal).unsqueeze(0)
        resampled=torchaudio.transforms.Resample(16000,24000)(wav);assert resampled.shape==(1,24000) and torch.isfinite(resampled).all();tests['resample_16k_to_24k']=True
        fb=torchaudio.compliance.kaldi.fbank(wav,num_mel_bins=80,dither=0,sample_frequency=16000);assert fb.ndim==2 and fb.shape[1]==80 and torch.isfinite(fb).all();tests['kaldi_fbank_80']=True
        mel=whisper.log_mel_spectrogram(wav,n_mels=128);assert mel.shape==(1,128,100) and torch.isfinite(mel).all();tests['whisper_128mel']={'shape':list(mel.shape),'finite':True}
        f0,ts=pyworld.dio(signal.astype(np.float64),16000);f0=pyworld.stonemask(signal.astype(np.float64),f0,ts,16000);assert f0.shape==ts.shape and f0.size>0 and np.isfinite(f0).all();tests['pyworld_native_cpu']=True
        with tempfile.TemporaryDirectory(prefix='cosyvoice-pcm-') as tmp:
            p=pathlib.Path(tmp)/'test.wav';stereo=np.stack([signal,signal*.5],axis=1);sf.write(p,stereo,16000,subtype='PCM_16');r,sr=sf.read(p,dtype='float32',always_2d=True);assert sr==16000 and r.shape==stereo.shape and np.max(np.abs(r-stereo))<=1/32768+1e-7;tests['soundfile_pcm_roundtrip']=True
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        tiny=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).float().eval()
        with torch.inference_mode():out=tiny(torch.tensor([[1,2,3]],dtype=torch.long)).logits
        assert out.shape==(1,3,32) and torch.isfinite(out).all() and all(p.dtype==torch.float32 for p in tiny.parameters());tests['qwen2_tiny_fp32_forward']=True
        import onnxruntime as ort
        assert 'CPUExecutionProvider' in ort.get_available_providers();tests['ort_cpu_provider_available']=True
        from hyperpyyaml import load_hyperpyyaml
        value=load_hyperpyyaml('v: !new:collections.Counter\n  a: 2\n  b: 1\n')
        assert value['v']['a']==2 and value['v']['b']==1;tests['hyperpyyaml_constructor']=True
        assert not attempts,'Unexpected network attempt was blocked';report['status']='qualified'
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['traceback']=traceback.format_exc()
    finally:
        report['elapsed_seconds']=time.monotonic()-start;a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'output':str(a.output),'tests':list(tests),'error':report.get('error')}))
    return 0 if report['status']=='qualified' else 1
if __name__=='__main__':sys.exit(main())
