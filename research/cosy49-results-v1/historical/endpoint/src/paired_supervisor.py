#!/usr/bin/env python3
"""Deferred, once-only paired native acquisition. Default is disarmed. No import-time launch."""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,pathlib,resource,selectors,signal,subprocess,sys,time
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
RAW_CAP=14*1024**2; STDERR_CAP=256*1024; RESOURCE_CAP=512*1024; META_CAP=2*1024**2; TOTAL_CAP=32*1024**2
LIMITS={'rss_bytes':268435456,'cpu_seconds':60,'wall_seconds':120,'output_bytes':RAW_CAP+STDERR_CAP,'collector_threads':1,'collector_processes':1}
EXECUTION_WORDS='EXECUTE_EXACT_ORIGINAL_A20_THEN_COSY49_STEP300_NATIVE98_ONCE'
# Read-only reuse of the reviewed observation/cleanup implementation. The original
# module's run()/admission()/__main__ are never imported or made callable here.
_HELPER_NAMES={'MissingProcField','error_record','unknown_io','read_proc','observe_process','guard_reason','apply_hard_limits','cleanup_owned_process'}
def get_helpers():
 path=ROOT/'vendor/supervisor_original.py';tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in _HELPER_NAMES]
 require({n.name for n in nodes}==_HELPER_NAMES,'audited helper closure missing')
 ns={'errno':__import__('errno'),'pathlib':pathlib,'os':os,'resource':resource,'signal':signal,'LIMITS':LIMITS,'PER_FILE_BYTES':RAW_CAP,'RESERVED_RECORD_BYTES':0,'IO_COUNTERS':('rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes')}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns

def require(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def encoded(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def unique(pairs):
 d={}
 for k,v in pairs:require(k not in d,'duplicate JSON key');d[k]=v
 return d
def load(p):return json.loads(pathlib.Path(p).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def regular(base,rel):
 require(type(rel)is str and rel!='','empty path');p=pathlib.PurePosixPath(rel)
 require(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,'nonlocal path')
 q=base/p;require(q.resolve().is_relative_to(base.resolve()),'path escape')
 require(all(not (base/pathlib.Path(*p.parts[:i])).is_symlink() for i in range(1,len(p.parts)+1)),'symlink rejected')
 require(q.is_file(),'regular file required '+str(q));return q

def file_ref(ref,base=WORKSPACE):
 require(set(ref)>={'path','bytes','sha256'},'file ref shape');p=regular(base,ref['path'])
 require(type(ref['bytes'])is int and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'file identity mismatch '+str(p));return p

def write_new(p,x,cap=META_CAP):
 b=x if isinstance(x,bytes) else encoded(x);require(len(b)<=cap,'file budget')
 with open(p,'xb') as f:f.write(b);f.flush();os.fsync(f.fileno())

def verify_freeze():
 require(not list(ROOT.rglob('*.pyc')),'unbound Python bytecode cache forbidden; use the clean frozen source package')
 freeze=load(ROOT/'SOURCE-FREEZE.json');require(freeze['schema']=='a20-endpoint-runtime-source-freeze-v1','freeze schema')
 seen=set()
 for r in freeze['files']:
  require(r['path'] not in seen,'duplicate frozen path');seen.add(r['path']);file_ref(r,ROOT)
 must={'src/paired_supervisor.py','src/collector.c','src/score_endpoint.py','src/compare_endpoint.py','src/compare_saved_scores.py','src/contracts.py','src/geometry.py','src/inputs.h','build/collector','PROTOCOL.json','metadata/manifest.json','metadata/geometry.json','metadata/ENDPOINT-CALL-LIST.json','metadata/ENDPOINT-EVALUATION-PROTOCOL.json','metadata/FROZEN-HISTORICAL-EVIDENCE.json','metadata/OUTPUT-BOUND.json','metadata/numerical-sources.json','metadata/COMPILE-ONLY.json','src/analyze_saved_pair.py','vendor/supervisor_original.py','src/candidate_control.py','metadata/QUALITY-GATE-REUSE.json'}
 require(must<=seen,'incomplete executable source freeze');return freeze

def check_receipt(ref,schema,binding):
 p=file_ref(ref);d=load(p);require(d.get('schema')==schema and d.get('status')=='PASS','receipt not PASS '+schema)
 for k,v in binding.items():require(d.get(k)==v,'receipt binding '+k)
 return d

def verify_bundle(ref,arm):
 p=file_ref(ref);b=load(p)
 require(b.get('schema')=='a20-candidate-native-bundle-v1' and b.get('variant')==('original' if arm=='original_A20' else 'candidate'),'native bundle arm/schema')
 for k in ('payload','manifest','identity_header','library','resources'):file_ref(b[k])
 require(b['payload']['bytes']==1565280,'payload layout bytes')
 require(isinstance(b.get('build_identity'),dict) and bool(b['build_identity']),'native build identity required')
 for k in ('compiler_sha256','compiler_version','flags','ABI','numerical_source_sha256'):
  require(k in b['build_identity'],'missing same-build field '+k)
 expected_sources=load(ROOT/'metadata/numerical-sources.json')
 for name,value in expected_sources.items():require(b['build_identity']['numerical_source_sha256'].get(name)==value,'numerical algorithm source drift '+name)
 compile_record=load(ROOT/'metadata/COMPILE-ONLY.json')
 require(compile_record.get('status')=='PASS_COMPILE_ONLY' and compile_record.get('collector_sha256')==sha(ROOT/'build/collector'),'current exact98 collector compile identity required')
 require(b['build_identity']['compiler_sha256']==compile_record['compiler']['sha256'],'collector/native compiler differs')
 if arm=='original_A20':require(b['payload']['sha256']=='a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d','original payload drift')
 return b

def verify_system(files):
 require(isinstance(files,list) and bool(files),'system runtime closure')
 paths=set()
 for r in files:
  p=pathlib.Path(r['path']);require(p.is_absolute() and p.is_file() and not p.is_symlink(),'resolved system file required')
  require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'system dependency drift');paths.add(str(p))
 require(str(pathlib.Path(sys.executable).resolve()) in paths,'Python runtime not bound')
 require(any('libc.so' in p for p in paths) and any('libm.so' in p for p in paths) and any('ld-linux' in p for p in paths),'incomplete ELF runtime closure')

def verify_inputs():
 m=load(ROOT/'metadata/manifest.json');require(len(m['rows'])==98,'exact98 sources')
 for r in m['rows']:
  p=regular(WORKSPACE,r['local_wav_path']);require(sha(p)==r['wav_sha256'],'WAV drift '+r['recording'])
  x=r['pcm_ingress'];require(x['path']==r['local_wav_path'] and x['bytes']==r['frames']*2,'PCM mapping')
  with open(p,'rb') as f:f.seek(x['byte_offset']);raw=f.read(x['bytes']);tail=f.read(1)
  require(not tail and len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==r['pcm_sha256'],'PCM drift '+r['recording'])
 return m

def admission(release_path):
 # Default gate precedes any lock/ledger/output/process side effect.
 require(pathlib.Path(release_path).stat().st_size<=256*1024,'release size cap')
 release=load(release_path)
 require(release.get('armed') is True and release.get('owner_execution_release')==EXECUTION_WORDS,'DISARMED: explicit separately reviewed endpoint release required')
 require(release.get('schema')=='a20-endpoint98-paired-release-v1','new endpoint release required')
 freeze=verify_freeze();binding={'freeze_sha256':sha(ROOT/'SOURCE-FREEZE.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'call_list_sha256':sha(ROOT/'metadata/ENDPOINT-CALL-LIST.json')}
 for k,v in binding.items():require(release.get(k)==v,'release binding '+k)
 require(release.get('decoder_config_sha256')=='073f57e82dc346d5c71bf7070e21cb8dfaf593d11e9dbfed263efb93b00b2360','frozen decoder contract drift')
 protocol=load(ROOT/'PROTOCOL.json');require(protocol['limits']==LIMITS and protocol['output_caps']=={'raw':RAW_CAP,'stderr':STDERR_CAP,'resource':RESOURCE_CAP,'metadata':META_CAP,'total':TOTAL_CAP},'resource contract drift')
 for key in ('accepted_limited_children_visibility','no_competing_acoustic_run_confirmed','existing98_local_research_scope_confirmed','no_natural_data_transfer','readback_persistence_plan_approved'):
  require(release.get(key) is True,'missing release condition '+key)
 check_receipt(release['preparation_review'],'a20-endpoint98-preparation-review-v1',binding)
 check_receipt(release['private_snapshot_readback'],'a20-endpoint98-source-readback-v1',binding)
 original=verify_bundle(release['original_native_bundle'],'original_A20');candidate=verify_bundle(release['candidate_native_bundle'],'candidate_cosy49_step300')
 require(original['build_identity']==candidate['build_identity'],'paired native build identity differs')
 require(load(file_ref(original['resources']))==load(file_ref(candidate['resources'])),'paired structural resource identity differs')
 require(candidate['payload']['sha256']!=original['payload']['sha256'],'step300 payload must have its own identity')
 checkpoint=file_ref(release['terminal_checkpoint']);state=release['terminal_state_sha256']
 terminal={'checkpoint_sha256':sha(checkpoint),'state_sha256':state,'step':300}
 require(candidate.get('checkpoint_sha256')==terminal['checkpoint_sha256'] and candidate.get('state_sha256')==state,'candidate native export/terminal linkage')
 check_receipt(release['terminal_audit'],'a20-cosy49-terminal-saved-output-audit-v1',terminal)
 native={**terminal,'bundle_sha256':release['candidate_native_bundle']['sha256'],'payload_sha256':candidate['payload']['sha256'],'native_model_steps':230,'reference_model_rows':115,'numerical_source_sha256':release['numerical_source_sha256'],'numerical_protocol_sha256':release['numerical_protocol_sha256']}
 check_receipt(release['candidate_native_audit'],'a20-cosy49-native-saved-output-audit-v1',native)
 verify_system(release['system_files']);require(release['cpu_affinity'] in os.sched_getaffinity(0),'frozen CPU unavailable')
 require(release['environment_identity']==environment_identity(),'same actual environment identity differs')
 verify_inputs();return release,binding,original,candidate

def environment_identity():
 # CPU MHz changes while idle/running; bind stable hardware/topology fields only.
 keys={'processor','vendor_id','cpu family','model','model name','stepping','microcode','physical id','siblings','core id','cpu cores','flags','address sizes'}
 cpu=[line.strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if ':' in line and line.split(':',1)[0].strip() in keys]
 return {'uname':list(os.uname()),'python':sys.version,'python_realpath':str(pathlib.Path(sys.executable).resolve()),'stable_cpu_fields_sha256':hashlib.sha256(('\n'.join(cpu)).encode()).hexdigest(),'boot_id_sha256':sha('/proc/sys/kernel/random/boot_id')}

def write_pair_metadata(out,ledger,name,value):
 b=encoded(value)
 existing=ledger.stat().st_size+sum(p.stat().st_size for p in out.iterdir() if not p.name.endswith(('.raw.jsonl','.stderr.log','.resource.json')))
 require(existing+len(b)<=META_CAP,'combined metadata allocation')
 write_new(out/name,b,META_CAP)


def compact_sample(state,elapsed,errors):
 def errid(value):
  if not value:return None
  key=encoded(value).decode()
  if key not in errors:errors[key]=len(errors)
  return errors[key]
 io=state['io'];names=('rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes')
 return [round(elapsed,6),state.get('returncode'),state.get('VmRSS'),state.get('VmHWM'),state.get('VmSize'),state.get('Threads'),state.get('children_count'),state['observation'],state['critical_monitoring'],state['children_observation'],errid(state.get('critical_errors')),errid(state.get('children_error')),errid(state.get('optional_status_errors')),
         [[io[n]['status'],io[n]['value'],errid(io[n].get('error')),io[n].get('reason')] for n in names]]

def run_arm(arm,bundle,release,binding,out):
 helpers=get_helpers();proc=None;selector=selectors.DefaultSelector();go_r=go_w=None;failure=None
 result={'schema':'a20-endpoint-arm-resource-v1','arm':arm,'status':'FAILED_NO_RETRY','proc_samples_compact':[],'error_dictionary':{},'critical_post_go_samples':0,'lifetime_child_absence':'NOT_PROVEN','continuous_lifetime_compliance_proven':False,'sampling_may_miss_transients':True,'io_terminal_lifetime':'UNKNOWN','initial_environment':environment_identity()}
 errors={};sizes={'raw':0,'stderr':0};caps={'raw':RAW_CAP,'stderr':STDERR_CAP};paths={k:out/(arm+'.'+('raw.jsonl' if k=='raw' else 'stderr.log')) for k in sizes};files={}
 ru0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();last_sample=-1;last_io={};peak=None
 try:
  for k,p in paths.items():files[k]=open(p,'xb')
  go_r,go_w=os.pipe()
  argv=[str(ROOT/'build/collector'),'--execute-after-reviewed-release',binding['protocol_sha256'],sha(ROOT/'metadata/manifest.json'),sha(ROOT/'metadata/geometry.json'),release['decoder_config_sha256'],bundle['payload']['sha256'],bundle['library']['sha256'],str(file_ref(bundle['payload'])),str(file_ref(bundle['library'])),str(go_r),arm]
  result['argv']=argv;result['launch_monotonic_ns']=time.monotonic_ns()
  proc=subprocess.Popen(argv,cwd=WORKSPACE,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,pass_fds=(go_r,),env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'},preexec_fn=lambda:helpers['apply_hard_limits'](release['cpu_affinity'],limits=LIMITS,file_bytes=RAW_CAP))
  os.close(go_r);go_r=None
  state=helpers['observe_process'](proc);reason=helpers['guard_reason'](state,time.monotonic()-start,0,limits=LIMITS,reserve=0)
  require(reason is None and state['observation']=='LIVE' and state['critical_monitoring']=='OBSERVED_RSS_THREADS_ONLY','pre-GO critical monitoring failed: '+str(reason))
  result['pre_go_observation']=state
  for k,stream in [('raw',proc.stdout),('stderr',proc.stderr)]:os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,k)
  os.write(go_w,b'G');os.close(go_w);go_w=None;result['go_monotonic_ns']=time.monotonic_ns()
  while True:
   for key,_ in selector.select(timeout=0.05):
    k=key.data;data=os.read(key.fileobj.fileno(),65536)
    if not data:selector.unregister(key.fileobj);key.fileobj.close();continue
    room=caps[k]-sizes[k];files[k].write(data[:room]);sizes[k]+=min(room,len(data))
    if len(data)>room:failure=k.upper()+'_OUTPUT_LIMIT';result['discarded_over_cap_bytes_observed']=len(data)-room;break
   elapsed=time.monotonic()-start
   if elapsed-last_sample>=0.1 or failure:
    state=helpers['observe_process'](proc);last_sample=elapsed
    if state['observation']=='LIVE' and state['critical_monitoring']=='OBSERVED_RSS_THREADS_ONLY':result['critical_post_go_samples']+=1
    if 'VmRSS' in state:peak=max(peak or 0,state['VmRSS']*1024)
    if 'VmHWM' in state:peak=max(peak or 0,state['VmHWM']*1024)
    for k,v in state['io'].items():
     if v['status']=='AVAILABLE':last_io[k]={**v,'wall_seconds':elapsed}
    result['io_last_successful_observation']=last_io;result['io_final_observation']=state['io'];result['sampled_peak_rss_bytes']=peak
    result['proc_samples_compact'].append(compact_sample(state,elapsed,errors));result['error_dictionary']={v:json.loads(k) for k,v in errors.items()}
    if len(encoded(result))>RESOURCE_CAP-16384:failure=failure or 'RESOURCE_EVIDENCE_LIMIT'
    failure=failure or helpers['guard_reason'](state,elapsed,sum(sizes.values()),limits=LIMITS,reserve=0)
    if failure:break
    if state['returncode'] is not None and not selector.get_map():break
   if elapsed>LIMITS['wall_seconds']:failure='WALL_LIMIT';break
  result['cleanup']=helpers['cleanup_owned_process'](proc,kill=bool(failure));result['returncode']=result['cleanup']['returncode']
  if result['cleanup']['errors']:failure=failure or 'CLEANUP_FAILED'
  if result['returncode']!=0:failure=failure or 'COLLECTOR_NONZERO_EXIT'
  if result['critical_post_go_samples']==0:failure=failure or 'NO_POST_GO_CRITICAL_SAMPLE'
 except BaseException as error:
  result['supervisor_exception']=helpers['error_record'](error,'paired arm supervisor');failure=failure or 'SUPERVISOR_EXCEPTION'
  if proc is not None:result['cleanup']=helpers['cleanup_owned_process'](proc,kill=True);result['returncode']=result['cleanup']['returncode']
 finally:
  for fd in (go_r,go_w):
   if fd is not None:os.close(fd)
  selector.close()
  for f in files.values():f.flush();os.fsync(f.fileno());f.close()
  ru1=resource.getrusage(resource.RUSAGE_CHILDREN)
  result.update(guard_stop_reason=failure,supervisor_launch_to_reap_wall_seconds=time.monotonic()-start,process_children_user_cpu_seconds_delta=ru1.ru_utime-ru0.ru_utime,process_children_system_cpu_seconds_delta=ru1.ru_stime-ru0.ru_stime,process_cpu_scope='one owned collector; includes loader and pipe-barrier wait CPU; no helper process is launched',output_bytes=sizes,final_environment=environment_identity())
  result['status']='COMPLETE_ACQUISITION_PENDING_TRACE_AUDIT' if failure is None else 'FAILED_NO_RETRY'
  if len(encoded(result))>RESOURCE_CAP:raise RuntimeError('resource evidence exceeds hard write allocation; arm is failed and raw files retained')
  write_new(out/(arm+'.resource.json'),result,RESOURCE_CAP)
 return result

def run(release_path):
 observer_start=time.process_time_ns();observer_usage0=resource.getrusage(resource.RUSAGE_SELF)
 release,binding,original,candidate=admission(release_path)
 # Shared lock is an existing fixed resource; no alternative lock path is accepted.
 lock_path=WORKSPACE/'kws-native-a20-recovery-v1/.heavy.lock'
 with open(lock_path,'a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  admission(release_path)
  out=ROOT/'run-once';ledger=ROOT/'EXECUTION-LEDGER.json'
  write_new(ledger,{'status':'RESERVED_NO_RETRY','release_sha256':sha(release_path),**binding,'arms':['original_A20','candidate_cosy49_step300']})
  out.mkdir(mode=0o700,exist_ok=False)
  write_pair_metadata(out,ledger,'release-used.json',release);results=[]
  try:
   initial=environment_identity();before=verify_freeze();verify_inputs();verify_system(release['system_files'])
   for arm,bundle in [('original_A20',original),('candidate_cosy49_step300',candidate)]:
    require(environment_identity()==initial,'environment drift between paired arms');require(sha(ROOT/'SOURCE-FREEZE.json')==binding['freeze_sha256'] and verify_freeze()==before,'source freeze drift before arm');verify_inputs();verify_system(release['system_files']);verify_bundle(release['original_native_bundle'] if arm=='original_A20' else release['candidate_native_bundle'],arm)
    result=run_arm(arm,bundle,release,binding,out);results.append(result)
    verify_bundle(release['original_native_bundle'] if arm=='original_A20' else release['candidate_native_bundle'],arm)
    require(result['status']=='COMPLETE_ACQUISITION_PENDING_TRACE_AUDIT','failed arm stops pair without retry')
    # Completeness must be validated before another arm. This is saved-only analysis.
    from score_endpoint import score
    score(out/(arm+'.raw.jsonl'),ROOT/'metadata/manifest.json',ROOT/'metadata/geometry.json',{'protocol_sha256':binding['protocol_sha256'],'model_sha256':bundle['payload']['sha256'],'library_sha256':bundle['library']['sha256'],'decoder_config_sha256':release['decoder_config_sha256']})
   require(verify_freeze()==before and environment_identity()==initial,'post-pair identity drift');verify_inputs();verify_system(release['system_files'])
   status='PAIR_ACQUIRED_PENDING_PRIVATE_READBACK_AND_INDEPENDENT_SAVED_AUDIT'
  except BaseException as error:
   status='FAILED_NO_RETRY';write_pair_metadata(out,ledger,'failure.json',{'error_type':type(error).__name__,'message':str(error),'completed_or_attempted_arms':len(results)})
  inventory=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
  summary={'schema':'a20-endpoint98-pair-acquisition-v1','status':status,'results':[{'arm':r['arm'],'status':r['status']} for r in results],'files':inventory,'qualification':False,'remote_persistence':'NOT_PERFORMED_BY_THIS_LOCAL_SUPERVISOR','saved_analysis':'Run only after exact local evidence is persisted and read back as separately authorized; no model repeat is needed'}
  require(sum(p.stat().st_size for p in out.iterdir())+len(encoded(summary))+ledger.stat().st_size<=TOTAL_CAP,'total evidence allocation')
  observer_usage1=resource.getrusage(resource.RUSAGE_SELF)
  summary['supervisor_overhead']={'process_cpu_ns':time.process_time_ns()-observer_start,'user_cpu_seconds':observer_usage1.ru_utime-observer_usage0.ru_utime,'system_cpu_seconds':observer_usage1.ru_stime-observer_usage0.ru_stime,'lifetime_peak_rss_bytes':observer_usage1.ru_maxrss*1024,'scope':'before admission to before final summary encoding; parent observer not covered by child hard limits'}
  write_pair_metadata(out,ledger,'pair-summary.json',summary)
  return summary

def main(argv=None):
 if sys.flags.optimize:raise SystemExit('optimized Python forbidden')
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=pathlib.Path,default=ROOT/'EXECUTION-RELEASE.json');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
 if not a.execute:print('DISARMED: preparation only; no child launched');return
 print(json.dumps(run(a.release),sort_keys=True))
if __name__=='__main__':main()
