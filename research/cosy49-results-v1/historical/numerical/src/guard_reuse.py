"""Supervision with exact-ENOENT owned snapshot recovery; no feature route."""
import os,math,resource,subprocess,sys,time
from pathlib import Path
from common import require,read_json
from supervision_reuse import read_proc,cleanup_owned_process,apply_hard_limits
ROOT=Path(__file__).resolve().parents[1]
TOTAL_CPU=600
TOTAL_WALL=900
AS_RSS=2*1024**3
OUTPUT_CAP=100*1024**2

def clean_env():
    home=ROOT/'work/home';tmp=ROOT/'work/tmp';home.mkdir(mode=0o700,exist_ok=True);tmp.mkdir(mode=0o700,exist_ok=True)
    return {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','HOME':str(home),'TMPDIR':str(tmp),
        'PIP_CONFIG_FILE':'/dev/null','PYTHONHASHSEED':'610104','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TORCH_HOME':str(ROOT/'work/torch-home')}

def cpu_ticks(pid):
    text=Path('/proc')/str(pid)/'stat';parts=text.read_text().rsplit(')',1)[1].split()
    return sum(int(parts[i])for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')

def observe_group(proc,max_processes):
    if proc.poll()is not None:return None
    if getattr(proc,'_a20_inventory_sticky',False):
        from group_inventory import observe_inventory
        return observe_inventory(proc,max_processes,read_proc)
    pending=[proc.pid];seen=set();states=[];cpu=0.
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        require(len(seen)<=max_processes,'PROCESS_LIMIT')
        try:
            require(os.getpgid(pid)==proc.pid,'OWNED_CHILD_GROUP_ESCAPE')
            s=read_proc(pid)
            errors=list(s['critical_errors'].values())
            if s.get('children_observation')!='AVAILABLE_AT_SAMPLE' and s.get('children_error'):
                errors.append(s['children_error'])
            if errors:
                safely_missing=all(e['type']in ('FileNotFoundError','ProcessLookupError','MissingProcField')for e in errors)
                ended=proc.poll()is not None if pid==proc.pid else False
                if pid!=proc.pid:
                    try:os.getpgid(pid)
                    except ProcessLookupError:ended=True
                if safely_missing and ended:
                    if pid==proc.pid:return None
                    continue
            require(not s['critical_errors'],'MANDATORY_PROC_FIELDS_UNAVAILABLE')
            if s['children_observation']=='NOT_AVAILABLE':
                proc._a20_inventory_sticky=True
                from group_inventory import observe_inventory
                return observe_inventory(proc,max_processes,read_proc)
            require(s['children_observation']=='AVAILABLE_AT_SAMPLE','CHILD_VISIBILITY_REQUIRED')
            cpu+=cpu_ticks(pid);states.append(s)
            pending.extend(int(v)for v in s['children'].split())
        except (FileNotFoundError,ProcessLookupError):
            if pid==proc.pid and proc.poll()is not None:return None
            if pid!=proc.pid:
                try:os.getpgid(pid)
                except ProcessLookupError:continue
            raise
    return dict(rss=sum(s['VmRSS']*1024 for s in states),vmsize=sum(s.get('VmSize',0)*1024 for s in states),
        threads=sum(s['Threads']for s in states),processes=len(states),cpu=cpu,
        io={k:sum(s['io'][k]['value']for s in states)if all(s['io'][k]['status']=='AVAILABLE'for s in states)else None
            for k in ('rchar','wchar','read_bytes','write_bytes')})

def bytes_under(path):
    total=0
    for p in path.rglob('*'):
        require(not p.is_symlink(),'unexpected output symlink')
        if p.is_file():total+=p.stat().st_size
    return total

def child_cpu_total():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime

def accounted_cpu():
    return child_cpu_total()+time.process_time()

def phase(name,argv,*,start,cpu_start,max_processes=1,max_threads=1,phase_wall=600,file_cap=4*1024**2):
    remaining_wall=min(phase_wall,TOTAL_WALL-(time.monotonic()-start))
    remaining_cpu=TOTAL_CPU-(accounted_cpu()-cpu_start)
    require(remaining_wall>0 and remaining_cpu>0,'GLOBAL_BUDGET_EXHAUSTED')
    record=dict(name=name,status='STARTED',samples=[],sample_fields=['wall_ms','rss_bytes','virtual_bytes','threads','owned_processes','observed_group_cpu_ms','rchar','wchar','read_bytes','write_bytes'],
                mandatory_live_samples=0,optional_io_null_means='UNKNOWN',continuous_lifetime_absence_or_peak_proven=False)
    limits=dict(rss_bytes=AS_RSS,cpu_seconds=max(1,math.ceil(remaining_cpu)),wall_seconds=max(1,math.ceil(remaining_wall)),output_bytes=OUTPUT_CAP,
                collector_threads=max_threads,collector_processes=max_processes)
    own_cpu=min(os.sched_getaffinity(0));t=time.monotonic();before=child_cpu_total();proc=None
    logs=ROOT/'work/logs';logs.mkdir(exist_ok=True)
    try:
        with (logs/(name+'.stdout')).open('xb')as stdout,(logs/(name+'.stderr')).open('xb')as stderr:
            proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=clean_env(),start_new_session=True,
                preexec_fn=lambda:apply_hard_limits(own_cpu,limits=limits,file_bytes=file_cap))
            while True:
                elapsed=time.monotonic()-t
                require(elapsed<=remaining_wall and time.monotonic()-start<=TOTAL_WALL,'WALL_LIMIT')
                require(bytes_under(logs)<=512*1024,'LOG_OUTPUT_LIMIT')
                require(bytes_under(ROOT/'work/artifact')<=OUTPUT_CAP,'ARTIFACT_WORK_OUTPUT_LIMIT')
                state=observe_group(proc,max_processes)
                if state is not None:
                    require(state['rss']<=AS_RSS,'RSS_LIMIT');require(state['threads']<=max_threads,'THREAD_LIMIT')
                    require(state['cpu']+accounted_cpu()-cpu_start<=TOTAL_CPU,'CPU_LIMIT')
                    if 'observation'in state:
                        from group_inventory import compact_observation
                        record['owned_inventory_metadata']=dict(method='BOUNDED_PROC_STAT_OWNED_PGID_SID_PPID_SNAPSHOT',owned_root=proc.pid,sample_fields=['wall_ms','root_starttime','owned_pids','children_interface_available_1_or_0','owned_disappeared_pids','unrelated_disappeared_count'],snapshot_atomic=False,continuous_lifetime_absence_or_peak_proven=False)
                        record.setdefault('owned_inventory_observations',[]).append(compact_observation(state['observation'],round(elapsed*1000)))
                    record['mandatory_live_samples']+=1
                    record['samples'].append([round(elapsed*1000),state['rss'],state['vmsize'],state['threads'],state['processes'],round(state['cpu']*1000),
                                               *[state['io'][k]for k in ('rchar','wchar','read_bytes','write_bytes')]])
                rc=proc.poll()
                if rc is not None:break
                time.sleep(.1)
            record['returncode']=rc
            cleanup=cleanup_owned_process(proc,kill=False)
            from safe_failure import cleanup_status
            record['cleanup']=cleanup_status(cleanup)
            record.setdefault('cleanup_attempts',[]).append(record['cleanup'])
            require(not cleanup['errors']and cleanup['reaped'],'PROCESS_REAP')
            require(rc==0 and record['mandatory_live_samples']>0,'CHILD_FAILURE_OR_NO_LIVE_GUARD_SAMPLE')
            require(accounted_cpu()-cpu_start<=TOTAL_CPU,'AGGREGATE_CPU_LIMIT')
            record['status']='PASS'
    except BaseException as e:
        if proc is not None:
            cleaned=cleanup_owned_process(proc,kill=True);record['returncode']=cleaned.get('returncode')
            from safe_failure import cleanup_status
            record['cleanup']=cleanup_status(cleaned)
            record.setdefault('cleanup_attempts',[]).append(record['cleanup'])
        from safe_failure import describe
        record.update(status='FAILED_NO_RETRY',safe_supervisor_failure=describe(ROOT,e,name,'OWNED_PROCESS_GUARD'))
        child_receipt=ROOT/'work'/(name+'-safe-error.json')
        if child_receipt.exists():record['safe_child_failure']=read_json(child_receipt,8192)
        else:record['safe_child_failure']=dict(phase=name,stage='CHILD_EXIT',error_code='NO_CHILD_RECEIPT_CAPTURED',exception_type='UNKNOWN')
    record.update(wall_ms=round((time.monotonic()-t)*1000),reaped_child_cpu_ms=round((child_cpu_total()-before)*1000))
    return record
