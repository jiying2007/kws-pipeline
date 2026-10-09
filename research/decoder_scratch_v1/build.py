"""Strictly derive scratch-v1; compile full runtime, execute decoder-only tests."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys
from native_elf import require_native, native_expected
from compiler_flags import CFLAGS, STRICT_FP_FLAGS
ROOT=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if sys.flags.optimize:raise RuntimeError("optimized Python disables test assertions; refused")
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,default=ROOT.parent/'native_a20');ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--cc',default='cc');args=ap.parse_args();src=args.source.resolve();out=args.output.resolve()
 if out.exists():ap.error('output must be a new directory')
 if src==out or src in out.parents or ROOT==out or ROOT in out.parents:ap.error('output must be outside source/module')
 deriv=json.loads((ROOT/'DERIVATION.json').read_text())
 if sha(src/'SOURCE_MANIFEST.json')!=deriv['source_manifest_sha256']:
  raise ValueError('pinned source manifest mismatch')
 manifest=json.loads((src/'SOURCE_MANIFEST.json').read_text())
 expected=manifest['files']
 all_paths=list(src.rglob('*'))
 if any(p.is_symlink() for p in all_paths):raise ValueError('source symlinks are not admitted')
 actual={str(p.relative_to(src)) for p in all_paths if p.is_file() and '__pycache__' not in p.parts and p!=src/'SOURCE_MANIFEST.json'}
 if actual!=set(expected):raise ValueError('closed source file inventory mismatch')
 for name,row in expected.items():
  f=src/name
  if f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise ValueError('source manifest mismatch: '+name)
 # Only execute the checker after its bytes and the entire source were verified.
 subprocess.run([sys.executable,'-B',str(src/'tests/check_manifest.py')],check=True,timeout=120)
 if sha(ROOT/'scratch-lifetimes.patch')!=deriv['patch_sha256']:raise ValueError('patch hash mismatch')
 for name,row in deriv['files'].items():
  if sha(src/'baseline/decoder'/name)!=row['source_sha256']:raise ValueError('source hash mismatch: '+name)
 out.mkdir(parents=True);tree=out/'source';tree.mkdir()
 for name in ('baseline','src'):shutil.copytree(src/name,tree/name)
 (tree/'tests').mkdir();shutil.copy2(src/'tests/test_decoder.c',tree/'tests/test_decoder.c');shutil.copy2(src/'resource_sizes.c',tree/'resource_sizes.c')
 for name in ('LICENSE','NOTICE.md'):shutil.copy2(src/name,tree/name)
 commands=[]
 def run(cmd,cwd=tree,env=None):
  p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,env=env,timeout=120);commands.append({'command':list(map(str,cmd)),'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr});p.check_returncode();return p.stdout
 run([sys.executable,'-B',str(ROOT/'test_native_elf.py')])
 run(['patch','--batch','--fuzz=0','-p0','-i',str(ROOT/'scratch-lifetimes.patch')])
 for name,row in deriv['files'].items():
  if sha(tree/'baseline/decoder'/name)!=row['derived_sha256']:raise ValueError('derived hash mismatch: '+name)
 (tree/'DERIVED_PROVENANCE.json').write_text(json.dumps({'schema':'decoder-scratch-lifetime-derived-v1','source_manifest_sha256':sha(src/'SOURCE_MANIFEST.json'),'derivation':deriv,'driver_sha256':{name:sha(ROOT/name) for name in ('build.py','test_decoder.py','compiler_flags.py','native_elf.py')},'historical_baseline_unmodified':True,'native_decoder_layout_changed':True,'board_measured':False},indent=2)+'\n')
 (tree/'DERIVED_MANIFEST.json').write_text(json.dumps({str(p.relative_to(tree)):sha(p) for p in sorted(tree.rglob('*')) if p.is_file()},indent=2)+'\n')
 flags=CFLAGS
 sources=['baseline/precision64/model/a20_fsmn.c','baseline/precision64/model/load.c','baseline/precision64/model/sha256.c','baseline/native/stream/splice.c','baseline/decoder/a20_decoder.c','src/donor_fft64.c','src/pcm_fft64.c','src/a20_stream_fft64.c','src/abi_fft64.c']
 run([args.cc,*flags,'-shared',*sources,'-lm','-o',str(out/'liba20_scratch_v1.so')])
 require_native(out/'liba20_scratch_v1.so',(3,))
 sizes={}
 for name,base in [('original',src),('optimized',tree)]:
  exe=out/('sizes-'+name);run([args.cc,*flags,str(base/'resource_sizes.c'),'-o',str(exe)]);require_native(exe);sizes[name]=json.loads(run([str(exe)]))
 for mode,extra in [('normal',[]),('asan',['-O1','-g','-fsanitize=address','-fno-omit-frame-pointer','-fno-pie','-no-pie']),('ubsan',['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all'])]:
  exe=out/('decoder-'+mode);run([args.cc,*flags,*extra,'tests/test_decoder.c','baseline/decoder/a20_decoder.c','-lm','-o',str(exe)]);env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0');require_native(exe);run([str(exe)],env=env)
 run([sys.executable,str(ROOT/'test_decoder.py'),'--cc',args.cc,'--original',str(src/'baseline/decoder'),'--optimized',str(tree/'baseline/decoder'),'--output',str(out/'differential')])
 differential_path=out/'differential/decoder-test-results.json'
 receipt={'schema':'decoder-scratch-v1-build','pass':True,'compiler':run([args.cc,'--version']).splitlines()[0],'compiler_command':args.cc,'compile_flags':list(CFLAGS),'strict_fp_flags':list(STRICT_FP_FLAGS),'compiler_flags_sha256':sha(ROOT/'compiler_flags.py'),'differential_receipt_sha256':sha(differential_path),'differential':json.loads(differential_path.read_text()),'native_elf':native_expected(),'subprocess_timeout_seconds':120,'sizes':sizes,'commands':commands,'acoustic_model_forwards':0,'audio_frames':0,'runtime_built_not_executed':True,'board_measured':False,'leak_detection':False,'library_sha256':sha(out/'liba20_scratch_v1.so')}
 (out/'RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'pass':True,'receipt':str(out/'RESULTS.json'),'stream_before':sizes['original']['stream_bytes'],'stream_after':sizes['optimized']['stream_bytes']}))
if __name__=='__main__':main()
