"""Freeze bounded TTS evidence; never uploads weights, environments or logs."""
import hashlib,json,pathlib,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parent
def public_setup_summary(setup):
    # A strict public projection; no command argv, absolute paths, host or environment dump.
    result={k:setup[k] for k in ('schema','python','locks','initial_free_bytes','network_configuration',
        'actual_download_bytes','wall_seconds','disk_checkpoints','bootstrap_pip') if k in setup}
    from setup_diagnostics import public_commands,EXCEPTIONS
    packages={r['name']:r for r in json.loads((ROOT/'runtime-lock.json').read_bytes())['files']}
    result['command_results']=public_commands(setup.get('commands',[]),packages)
    phase=setup.get('phase')
    if type(phase) is dict and set(phase)=={'stage','package'}:
        allowed_stages=set(__import__('setup_diagnostics').STAGES)|{'download_package','download_model_asset','extract_source'}
        if phase['stage'] in allowed_stages and phase['package'] in set(packages)|{'qwen_tts','pip',None}:result['active_phase']=phase
    result['status']='error' if 'exception' in setup else ('verified' if 'verification' in setup else 'incomplete')
    if 'exception' in setup: result['exception_type']=setup['exception']['type'] if setup['exception']['type'] in EXCEPTIONS else 'OtherException'
    result['wheels']={name:{k:r[k] for k in ('bytes','sha256','name','version','metadata_sha256','metadata_bytes') if k in r}
                      for name,r in setup.get('wheels',{}).items()}
    verification=setup.get('verification',{})
    result['source_members']=verification.get('source_members',[])
    result['model_assets']=verification.get('model_assets',{})
    return result
MAX_FILE=8*1024**2

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(root,cap):
    if not root.is_dir():raise ValueError('No artifact directory')
    files=[p for p in root.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in root.rglob('*')) or any(p.stat().st_size>MAX_FILE for p in files):raise ValueError('Artifact file/alias bound')
    if sum(p.stat().st_size for p in files)>cap:raise ValueError('Artifact total bound')
    return files

def finalize():
    runtime=ROOT/'runtime';out=runtime/'artifact'
    if out.exists():raise ValueError('No artifact rewrite')
    if not (runtime/'resources.json').is_file():raise ValueError('No TTS stage evidence')
    out.mkdir();shutil.copyfile(runtime/'resources.json',out/'resources.json')
    if (runtime/'setup-receipt.json').is_file():
        summary=public_setup_summary(json.loads((runtime/'setup-receipt.json').read_bytes()))
        (out/'setup-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    source=runtime/'generation'
    allowed={'generation-receipt.json','generation-freeze.json','receipt.tmp'}|{f'qwen6-{i:03d}'+suffix for i in range(1,7) for suffix in ('.wav','.raw-float.wav','.started.json')}
    if source.exists():
        if source.is_symlink() or any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in source.iterdir()):raise ValueError('Unexpected generation artifact member')
        shutil.copytree(source,out/'generation')
    files=checked(out,16*1024**2)
    freeze={'schema':'qwen6-tts-artifact-freeze-v1','files':{p.relative_to(out).as_posix():sha(p) for p in files}}
    (out/'artifact-freeze.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n')
    checked(out,16*1024**2)

def check_blind():
    root=ROOT/'runtime/blind-artifact'
    if {p.name for p in root.iterdir()}!={'blind-inputs.zip','blind-input-freeze.json'}:raise ValueError('Blind artifact membership')
    checked(root,4*1024**2)

if __name__=='__main__':
    try:
        if sys.argv[1:] == ['--blind']:check_blind()
        elif not sys.argv[1:]:finalize()
        else:raise ValueError('Unknown artifact action')
    except Exception as error:
        print(json.dumps({'status':'artifact_blocked','error_class':type(error).__name__,'error_code':'ARTIFACT_BOUND_OR_SCOPE'}),file=sys.stderr)
        raise SystemExit(1)
