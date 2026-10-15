"""Stdlib reproduction and fail-closed CLI checks; supply the fixed --archive."""
import argparse,hashlib,json,pathlib,subprocess,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True);args=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
script=here/'analyze_pcm_portable.py'
def require(ok,message):
 if not ok:raise ValueError(message)
MODE=(['-OO'] if sys.flags.optimize==2 else ['-O'] if sys.flags.optimize==1 else [])
def invoke(archive,*extra):return subprocess.run([sys.executable,*MODE,str(script),'--archive',str(archive),*extra],capture_output=True)
with tempfile.TemporaryDirectory(dir=here) as tmp:
 root=pathlib.Path(tmp);dest=root/'new-output'
 r=invoke(args.archive,'--output',str(dest));require(r.returncode==0,'CLI failed')
 for name in ('summary.json','pcm-energy.json'):require((dest/name).read_bytes()==(here/name).read_bytes(),'frozen bytes differ')
 before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
 r=invoke(args.archive,'--output',str(dest));require(r.returncode!=0,'existing output accepted')
 require(before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()},'existing output changed')
 r=invoke(args.archive);require(r.returncode==0,'stdout CLI failed')
 bundle=json.loads(r.stdout)
 require(set(bundle)=={'summary.json','pcm-energy.json'},'stdout membership differs')
 for name in bundle:require(bundle[name]==json.loads((here/name).read_bytes()),'stdout objects differ')
 bad=root/'bad.zip';bad.write_bytes(b'bad')
 r=invoke(bad,'--output',str(root/'bad-output'));require(r.returncode!=0 and not (root/'bad-output').exists(),'wrong size accepted')
 raw=bytearray(args.archive.read_bytes());raw[0]^=1;bad.write_bytes(raw)
 r=invoke(bad,'--output',str(root/'bad-hash-output'));require(r.returncode!=0 and not (root/'bad-hash-output').exists(),'wrong hash accepted')
print(json.dumps({'status':'PASS','checks':['exact_frozen_bytes','existing_output_not_overwritten','stdout_strict_numeric_objects','wrong_size_rejected','wrong_hash_rejected'],'new_model_calls':0,'new_optimizer_steps':0},sort_keys=True))
