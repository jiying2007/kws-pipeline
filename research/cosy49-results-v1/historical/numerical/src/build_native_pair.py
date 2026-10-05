"""Compile original/candidate runtime with identical sources; no model/input calls.

Resource probes use sizeof/offsetof only. Actual stage execution still requires
an explicit export/build release; default invocation is disarmed.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,resource,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1]
from export_candidate import require,sha,read,write,verify_prepared_source,enforce_export_build_limits
FLAGS=['-std=c11','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-frounding-math','-fexcess-precision=standard','-fstack-usage','-fPIC']
SOURCES=['baseline/precision64/model/a20_fsmn.c','baseline/precision64/model/load.c','baseline/precision64/model/sha256.c','baseline/native/stream/splice.c','baseline/decoder/a20_decoder.c','src/donor_fft64.c','src/pcm_fft64.c','src/a20_stream_fft64.c','src/abi_fft64.c']
def resolve(value):
 p=Path(value);return p if p.is_absolute()else WORKSPACE/p

def entry(path):
 p=Path(path).resolve();require(p.is_relative_to(WORKSPACE.resolve()),'bundle workspace scope')
 return dict(path=str(p.relative_to(WORKSPACE)),bytes=p.stat().st_size,sha256=sha(p))

def build_pair(release_path):
 r=read(release_path);require(r.get('schema')=='a20-candidate-native-build-release-v1'and r.get('approved')is True,'native build release')
 verify_prepared_source(r)
 ledger=read(ROOT/'build-guard/work/EXECUTION-LEDGER.json');require(ledger['release_sha256']==sha(release_path)and ledger['status']=='START_RESERVED_NO_RETRY','supervised build reservation')
 write(ROOT/'build-guard/work/worker-claim.json',dict(status='CLAIMED_NO_RETRY',release_sha256=sha(release_path)))
 start=time.monotonic();cpu_start=time.process_time()+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
 exp=resolve(r['candidate_export_directory']);receipt=read(exp/'export-receipt.json')
 require(receipt['status']=='PASS_EXPORT_ONLY'and sha(exp/'export-receipt.json')==r['export_receipt_sha256'],'export receipt barrier')
 require(r['export_saved_output_audit']['status']=='PASS'and r['export_saved_output_audit']['receipt_sha256']==sha(exp/'export-receipt.json'),'independent export audit')
 source=ROOT/'vendor/runtime';srcid={str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()and p.relative_to(source).as_posix()!='baseline/export/a20_identity.h'}
 compiler=Path('/usr/bin/gcc').resolve();identity=dict(compiler_sha256=sha(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0],flags=FLAGS,numerical_source_sha256=srcid,diagnostic_source_sha256=sha(ROOT/'vendor/validation/diagnostic.c'),ABI='verified_same_run_sizeof_offsetof_resource_json')
 require(identity['compiler_sha256']==r['approved_compiler_sha256'],'approved exact compiler')
 for name,h in r['source_bindings'].items():require(sha(ROOT/name)==h,'build source identity')
 out=resolve(r['output_directory']);require(out.resolve()==(ROOT/'build-guard/work/artifact').resolve(),'guarded build output scope');out.mkdir(exist_ok=False);bundles={};resources={}
 for variant in ('original','candidate'):
  folder=out/variant;runtime=folder/'runtime';shutil.copytree(source,runtime);build=folder/'build';build.mkdir()
  source_export=ROOT/'vendor/historical/kws-native-a20-research-v1/export'if variant=='original'else exp
  for n in ('a20.f32','manifest.json','a20_identity.h'):shutil.copyfile(source_export/n,folder/n)
  shutil.copyfile(folder/'a20_identity.h',runtime/'baseline/export/a20_identity.h');commands=[]
  def run(argv):
   remaining=120-(time.monotonic()-start);require(remaining>0,'pair build wall cap')
   x=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=min(90,remaining));commands.append(dict(argv=argv,returncode=x.returncode))
   usage=resource.getrusage(resource.RUSAGE_CHILDREN);require(time.process_time()+usage.ru_utime+usage.ru_stime-cpu_start<=60,'pair build aggregate CPU cap')
   require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())<=32*1024**2,'pair build output cap');return x
  objects=[]
  for i,n in enumerate(SOURCES):
   p=build/f'{i:02d}.o';objects.append(p);run([str(compiler),*FLAGS,'-c',str(runtime/n),'-o',str(p)])
  lib=build/'liba20_fft64.so';run([str(compiler),'-shared',*map(str,objects),'-lm','-o',str(lib)])
  probe=build/'resource_sizes';run([str(compiler),*FLAGS,str(runtime/'resource_sizes.c'),'-o',str(probe)])
  sizes=json.loads(run([str(probe)]).stdout);require(sizes['a20_model_steps']==sizes['actual_audio_frames']==0,'resource size-only probe')
  require(sizes['weights_bytes']==1565280 and sizes['model_cache_bytes']==22528 and sizes['stream_bytes']==333704,'reviewed host structural dimensions')
  write(build/'resource-sizes.json',sizes);resources[variant]=sizes
  diag_src=folder/'validation/diagnostic.c';diag_src.parent.mkdir();shutil.copyfile(ROOT/'vendor/validation/diagnostic.c',diag_src)
  diag=build/'libvalidation_diagnostic.so';run([str(compiler),*FLAGS,'-shared',str(diag_src),'-L'+str(build),'-Wl,-rpath,$ORIGIN','-la20_fft64','-lm','-o',str(diag)])
  manifest=read(folder/'manifest.json');require(sha(folder/'a20.f32')==manifest['payload_sha256'],'export payload consistency')
  b=dict(schema='a20-candidate-native-bundle-v1',variant=variant,checkpoint_sha256=manifest['checkpoint_sha256'],state_sha256=manifest['state_sha256'],payload=entry(folder/'a20.f32'),manifest=entry(folder/'manifest.json'),identity_header=entry(folder/'a20_identity.h'),library=entry(lib),resources=entry(build/'resource-sizes.json'),diagnostic_library=entry(diag),ctypes_api=entry(runtime/'ctypes_api.py'),build_identity=identity,model_calls=0,audio_calls=0,numerical_validation='NOT_RUN')
  write(out/(variant+'-native-bundle.json'),b);bundles[variant]=b;write(folder/'build-receipt.json',dict(commands=commands,source_identity=identity,model_calls=0,acoustic_calls=0))
 require(resources['original']==resources['candidate'],'same structural resource contract')
 require(bundles['original']['build_identity']==bundles['candidate']['build_identity'],'paired exact build identities')
 write(out/'pair-build-receipt.json',dict(status='PASS_BUILD_ONLY',bundles={k:entry(out/(k+'-native-bundle.json'))for k in bundles},resources_equal=True,numerical_validation='NOT_RUN',model_calls=0,acoustic_calls=0))
def supervised_build(release_path):
 r=read(release_path);require(r.get('schema')=='a20-candidate-native-build-release-v1'and r.get('approved')is True,'native build release');verify_prepared_source(r)
 import guard_reuse as guard
 phase_root=ROOT/'build-guard';phase_root.mkdir(exist_ok=True);work=phase_root/'work';work.mkdir(exist_ok=False)
 write(work/'EXECUTION-LEDGER.json',dict(status='START_RESERVED_NO_RETRY',release_sha256=sha(release_path),retries_allowed=False))
 guard.ROOT=phase_root;guard.TOTAL_CPU=60;guard.TOTAL_WALL=120;guard.OUTPUT_CAP=32*1024**2
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 start=time.monotonic();cpu_start=guard.accounted_cpu()
 def deadline(*unused):raise RuntimeError('BUILD_GLOBAL_WALL_LIMIT')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
 record=guard.phase('build',[sys.executable,'-B',str(Path(__file__).resolve()),'--released-worker',str(Path(release_path).resolve())],start=start,cpu_start=cpu_start,max_processes=4,max_threads=4,phase_wall=120,file_cap=20*1024**2)
 signal.alarm(0);metrics=dict(wall_seconds=time.monotonic()-start,aggregate_CPU_seconds=guard.accounted_cpu()-cpu_start)
 passed=record['status']=='PASS'and metrics['wall_seconds']<=120 and metrics['aggregate_CPU_seconds']<=60
 write(work/'supervision.json',dict(status='PASS_BUILD_GUARD'if passed else'FAILED_NO_RETRY',record=record,metrics=metrics,phase_cpu_wall_scope='controller plus one owned worker and compiler descendants through reap; source admission excluded',model_calls=0,audio_calls=0,next_stage_automatic=False))
 return 0 if passed else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path);p.add_argument('--released-worker',type=Path);a=p.parse_args()
 if a.released_worker:
  require(a.release is None,'one build route');build_pair(a.released_worker)
 elif a.release:raise SystemExit(supervised_build(a.release))
 else:raise SystemExit('PREPARATION_ONLY: explicit audited build release required')
