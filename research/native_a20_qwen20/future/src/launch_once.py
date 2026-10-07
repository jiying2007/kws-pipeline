#!/usr/bin/env python3
"""UNRELEASED future supervisor candidate. run() is deliberately disabled."""
import argparse,fcntl,hashlib,json,os,pathlib,resource,signal,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
LIMITS={'rss_bytes':268435456,'cpu_seconds':60,'wall_seconds':120,'output_bytes':16777216,'collector_threads':1,'collector_processes':1}
IO_COUNTERS=('rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes')
RESERVED_RECORD_BYTES=2097152
PER_FILE_BYTES=6*1024*1024
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def require(v,m):
    if not v:raise RuntimeError(m)
def verify_freeze(root,freeze):
    got={}
    for x in freeze['files']:
        p=root/x['path'];require(p.is_file() and not p.is_symlink(),'missing/symlink identity '+x['path'])
        require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'identity mismatch '+x['path'])
        if x.get('executable'):require(os.access(p,os.X_OK),'not executable '+x['path'])
        got[x['path']]=x['sha256']
    return got

class MissingProcField(ValueError):
    """A required field was absent, including a possible terminal status race."""


def error_record(error, path):
    """Preserve the exact exception text, rather than replacing a read with zero."""
    return {'type':type(error).__name__, 'message':str(error), 'path':str(path),
            'errno':getattr(error,'errno',None), 'filename':getattr(error,'filename',None)}


def unknown_io(reason):
    return {key:{'status':'UNKNOWN','value':None,'reason':reason} for key in IO_COUNTERS}


def read_proc(pid, read_text=None):
    """Collect critical guards separately from optional, per-counter I/O telemetry."""
    read_text = read_text or (lambda path: path.read_text())
    base=pathlib.Path('/proc')/str(pid)
    state={'critical_errors':{}, 'optional_status_errors':{}}
    status_path=base/'status'
    try:
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                for line in read_text(status_path).splitlines() if ':' in line}
    except (OSError,UnicodeError) as error:
        fields=None
        for key in ('VmRSS','Threads'):
            state['critical_errors'][key]=error_record(error,status_path)
    if fields is not None:
        for key in ('VmRSS','Threads','VmHWM','VmSize'):
            try:
                if key not in fields:
                    raise MissingProcField('missing '+key+' in '+str(status_path))
                parts=fields[key].split()
                if key=='Threads':
                    if len(parts)!=1 or not parts[0].isdigit() or int(parts[0])<1:
                        raise ValueError('invalid '+key+' field: '+repr(fields[key]))
                elif len(parts)!=2 or not parts[0].isdigit() or parts[1]!='kB':
                    raise ValueError('invalid '+key+' field: '+repr(fields[key]))
                state[key]=int(parts[0])
            except (ValueError,IndexError) as error:
                target='critical_errors' if key in ('VmRSS','Threads') else 'optional_status_errors'
                state[target][key]=error_record(error,status_path)

    children_path=base/'task'/str(pid)/'children'
    try:
        children=read_text(children_path).strip()
        if any(not token.isdigit() or int(token)<1 for token in children.split()):
            raise ValueError('invalid children field: '+repr(children))
        state['children']=children
    except (OSError,UnicodeError,ValueError) as error:
        state['critical_errors']['children']=error_record(error,children_path)

    io_path=base/'io'
    state['io']={}
    try:
        io_fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                   for line in read_text(io_path).splitlines() if ':' in line}
    except (OSError,UnicodeError) as error:
        for key in IO_COUNTERS:
            state['io'][key]={'status':'UNAVAILABLE','value':None,'error':error_record(error,io_path)}
    else:
        for key in IO_COUNTERS:
            try:
                if key not in io_fields:
                    raise MissingProcField('missing '+key+' in '+str(io_path))
                if not io_fields[key].isdigit():
                    raise ValueError('invalid '+key+' counter: '+repr(io_fields[key]))
                state['io'][key]={'status':'AVAILABLE','value':int(io_fields[key])}
            except ValueError as error:
                state['io'][key]={'status':'UNAVAILABLE','value':None,'error':error_record(error,io_path)}
    return state


def observe_process(proc, read_text=None):
    """Poll first. Only confirmed disappearance races relax missing terminal guards."""
    rc=proc.poll()
    if rc is not None:
        return {'returncode':rc,'observation':'EXITED_BEFORE_READ','critical_errors':{},
                'critical_monitoring':'UNKNOWN_NOT_SAMPLED_AFTER_EXIT',
                'io':unknown_io('process already exited; no post-exit read attempted')}
    state=read_proc(proc.pid,read_text)
    state['returncode']=proc.poll()
    state['observation']='LIVE' if state['returncode'] is None else 'EXITED_DURING_READ'
    errors=state['critical_errors']
    if not errors:
        state['critical_monitoring']='OBSERVED'
    elif state['returncode'] is not None and all(
            error['type'] in ('FileNotFoundError','ProcessLookupError','MissingProcField')
            for error in errors.values()):
        state['critical_monitoring']='UNKNOWN_CONFIRMED_EXIT_RACE'
    else:
        # Permission denial and malformed critical data never become success at exit.
        state['critical_monitoring']='FAILED_UNAVAILABLE'
    return state


def guard_reason(state, elapsed, output_bytes):
    """Enforce known limit crossings even if this observation also sees an exit."""
    if elapsed>LIMITS['wall_seconds']:return 'WALL_LIMIT'
    if 'VmRSS' in state and state['VmRSS']*1024>LIMITS['rss_bytes']:return 'RSS_LIMIT'
    if 'Threads' in state and state['Threads']>LIMITS['collector_threads']:return 'THREAD_LIMIT'
    if state.get('children'):return 'CHILD_PROCESS_FORBIDDEN'
    if output_bytes>LIMITS['output_bytes']-RESERVED_RECORD_BYTES:return 'OUTPUT_LIMIT'
    if state['critical_monitoring']=='FAILED_UNAVAILABLE':return 'CRITICAL_MONITORING_UNAVAILABLE'
    return None


def apply_hard_limits(cpu):
    os.sched_setaffinity(0,{cpu})
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['rss_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(PER_FILE_BYTES,)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    signal.alarm(LIMITS['wall_seconds'])


def monitor_process(proc, start, output_size, *, read_text=None,
                    clock=time.monotonic, sleep=time.sleep, killpg=os.killpg, record=None):
    """Injectable monitor: tests use only invented observations and a fake process."""
    samples=[]
    peak=None
    io_last=unknown_io('no successful sample')
    record={} if record is None else record
    while True:
        state=observe_process(proc,read_text)
        elapsed=clock()-start
        for key in ('VmRSS','VmHWM'):
            if key in state:
                value=state[key]*1024
                peak=value if peak is None else max(peak,value)
        for key, counter in state['io'].items():
            if counter['status']=='AVAILABLE':
                io_last[key]={**counter,'wall_s':elapsed}
        samples.append({'wall_s':elapsed,**state})
        record.update(sampled_peak_rss_bytes=peak,io_last_successful_observation=io_last,
                      io_final_observation=state['io'],proc_samples=samples)
        reason=guard_reason(state,elapsed,output_size())
        if reason is not None or state['returncode'] is not None:
            if reason is not None and proc.poll() is None:
                try:killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass  # Exit between poll and kill; failure remains.
            record.update(returncode=proc.wait(),guard_stop_reason=reason)
            return record
        sleep(0.1)


def encode_resource_record(record, artifact_bytes):
    """Bound the final write too; oversized diagnostics fail rather than overrun."""
    def encode(value):
        return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    def fits(data):
        return len(data)<=PER_FILE_BYTES and artifact_bytes+len(data)<=LIMITS['output_bytes']
    record=dict(record)
    data=encode(record)
    if fits(data):return record,data
    samples=record['proc_samples']
    record.update(status='FAILED_NO_RETRY',
                  guard_stop_reason=record.get('guard_stop_reason') or 'RESOURCE_RECORD_OUTPUT_LIMIT',
                  resource_record_output_limit=True,proc_samples_total=len(samples),
                  proc_samples_omitted=max(0,len(samples)-2),
                  proc_samples=samples if len(samples)<=2 else [samples[0],samples[-1]])
    data=encode(record)
    # Never write beyond the cap, including when even reduced failure evidence cannot fit.
    return record,data if fits(data) else None


def run(release_path):
    raise RuntimeError('FUTURE_DRAFT_NOT_RELEASED: new reviewed protocol, actual launcher freeze and explicit release required')
    # Intentionally unreachable integration candidate below. Do not remove this gate
    # under the v1 release; a future review must bind this file's actual bytes/path.
    release=load(release_path);freeze_path=ROOT/'SOURCE-FREEZE.json';protocol_path=ROOT/'PROTOCOL.json';protocol=load(protocol_path);freeze=load(freeze_path)
    require(release['owner_execution_release']=='EXECUTE_EXACT_QWEN20_ONCE_AFTER_REVIEW_AND_PERSISTENCE','no owner execution release')
    require(release['freeze_sha256']==sha(freeze_path) and release['protocol_sha256']==sha(protocol_path),'release identity mismatch')
    require(release['no_competing_acoustic_run_confirmed'] is True,'resource context not confirmed')
    for kind in ('independent_review','private_snapshot_readback'):
        ref=release[kind];p=pathlib.Path(ref['path']);require(sha(p)==ref['sha256'],kind+' receipt identity')
        receipt=load(p);require(receipt['status']=='PASS' and receipt['freeze_sha256']==sha(freeze_path),kind+' not PASS for freeze')
    require(protocol['limits']==LIMITS,'limits drift')
    identities_before=verify_freeze(ROOT,freeze)
    for p,h in protocol['system_dependencies'].items():require(sha(pathlib.Path(p))==h,'system dependency mismatch '+p)
    shared=ROOT.parent.parent/'kws-native-a20-recovery-v1/.heavy.lock'
    with open(shared,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Never remove either the ledger or output directory to permit a rerun.
        ledger=ROOT/'EXECUTION-LEDGER.json';out=ROOT/'run-once'
        fd=os.open(ledger,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.write(fd,(json.dumps({'status':'START_RESERVED_NO_RETRY','release_sha256':sha(release_path),'freeze_sha256':sha(freeze_path),'utc_epoch_s':time.time()})+'\n').encode());os.fsync(fd);os.close(fd)
        out.mkdir(exist_ok=False)
        save(out/'release-used.json',release);save(out/'identities-before.json',identities_before)
        cpu=protocol['cpu_affinity'];require(cpu in os.sched_getaffinity(0),'frozen CPU unavailable')
        env=dict(os.environ);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
        context={'host_uname':list(os.uname()),'cpu_affinity':[cpu],'supervisor_separate_from_single_collector_process':True,'loadavg':pathlib.Path('/proc/loadavg').read_text().strip(),'parallel_context':release.get('parallel_context'),'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text().split('\n\n')[0],'limits':LIMITS,'clock_monotonic_resolution_s':time.get_clock_info('monotonic').resolution}
        save(out/'resource-context.json',context)
        def guard():
            apply_hard_limits(cpu)
        bindings=load(ROOT/'metadata/run-bindings.json')
        argv=[str(ROOT/'build/collector'),'--run',sha(protocol_path),sha(ROOT/'metadata/verified-inputs.json'),sha(ROOT/'metadata/geometry.json'),sha(ROOT/'metadata/decoder-config.json')]
        start=time.monotonic();reason=None;proc=None
        observation={'sampled_peak_rss_bytes':None,'io_last_successful_observation':unknown_io('no successful sample'),'io_final_observation':unknown_io('no observation'),'proc_samples':[]}
        try:
            with open(out/'raw.jsonl','xb',buffering=0) as raw,open(out/'stderr.log','xb',buffering=0) as err:
                proc=subprocess.Popen(argv,cwd=ROOT,stdout=raw,stderr=err,env=env,start_new_session=True,preexec_fn=guard)
                monitor_process(proc,start,lambda:sum(p.stat().st_size for p in out.iterdir() if p.is_file()),record=observation)
            returncode=observation.pop('returncode');reason=observation.pop('guard_stop_reason')
        except BaseException as ex:
            reason='SUPERVISOR_EXCEPTION:'+type(ex).__name__+':'+str(ex)
            if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            returncode=None if proc is None else proc.returncode
        try:
            after=verify_freeze(ROOT,freeze);identity_ok=after==identities_before
            for p,h in protocol['system_dependencies'].items():require(sha(pathlib.Path(p))==h,'system dependency changed '+p)
        except Exception as ex:after={'error':str(ex)};identity_ok=False
        save(out/'identities-after.json',after)
        status='RAW_COMPLETE_PENDING_AUDIT' if returncode==0 and reason is None and identity_ok else 'FAILED_NO_RETRY'
        artifact_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        record,data=encode_resource_record({'status':status,'returncode':returncode,'guard_stop_reason':reason,'elapsed_wall_s':time.monotonic()-start,**observation,'sampling_interval_s':0.1,'sampling_may_miss_exit_peak':True,'exact_child_rusage_in_raw_run_end_if_complete':True,'identity_after_ok':identity_ok,'raw_sha256':sha(out/'raw.jsonl') if (out/'raw.jsonl').exists() else None,'artifact_bytes_before_resource_record_write':artifact_bytes,'model_runs_attempted':1,'retries_allowed':False},artifact_bytes)
        status=record['status'];reason=record['guard_stop_reason']
        if data is not None:
            with open(out/'resource-record.json','xb') as stream:stream.write(data)
        save(ledger,{'status':status,'release_sha256':sha(release_path),'freeze_sha256':sha(freeze_path),'output_dir':'run-once','returncode':returncode,'guard_stop_reason':reason,'once_only':True})
        return 0 if status=='RAW_COMPLETE_PENDING_AUDIT' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();raise SystemExit(run(a.run))
