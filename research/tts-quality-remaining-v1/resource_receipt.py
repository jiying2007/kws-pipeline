"""Read-only Linux worker measurements for remaining only; never reconstruct probes.

No model imports. Call worker_final in its finally, and host_summary from the host
finally after its existing container supervisor. Missing measurements stay unknown.
"""
from pathlib import Path
import json,os,platform,resource,time
CPU_FIELDS=('usage_usec','user_usec','system_usec','nr_periods','nr_throttled','throttled_usec')
STAGES={'scope','verify_runtime','import_runtime','load_model','generation','complete','unknown'}

def observed(value,unit,scope):return {'status':'observed','value':value,'unit':unit,'scope':scope}
def unknown(reason,scope):return {'status':'unknown','reason':reason,'scope':scope}

def bounded_text(path):
    with Path(path).open('r',encoding='ascii') as f:value=f.read(4097)
    if len(value)>4096:raise ValueError('oversized kernel record')
    return value

def sample(*,cgroup_root=Path('/sys/fs/cgroup')):
    result={'schema':'tts-remaining-resource-sample-v1','sample_monotonic_seconds':time.monotonic()}
    try:
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system()!='Linux' or not isinstance(value,(int,float)) or value<0:
            raise ValueError('unsupported RSS units')
        result['process_peak_rss']=observed(int(value)*1024,'bytes','current_worker_process_high_water_not_process_tree')
    except (OSError,ValueError,AttributeError):
        result['process_peak_rss']=unknown('getrusage_unavailable_or_units_unknown','current_worker_process')
    try:
        value=bounded_text(Path(cgroup_root)/'memory.peak').strip()
        if not value.isdecimal():raise ValueError('invalid memory.peak')
        result['cgroup_memory_peak']=observed(int(value),'bytes','current_container_cgroup_process_tree_high_water')
    except (OSError,ValueError,UnicodeError):
        result['cgroup_memory_peak']=unknown('memory_peak_unavailable','current_container_cgroup')
    try:
        stats={}
        for line in bounded_text(Path(cgroup_root)/'cpu.stat').splitlines():
            pieces=line.split()
            if len(pieces)!=2 or not pieces[1].isdecimal():raise ValueError('invalid cpu.stat')
            if pieces[0] in stats:raise ValueError('duplicate cpu.stat')
            stats[pieces[0]]=int(pieces[1])
        result['cgroup_cpu_stat']={key:observed(stats[key],'microseconds' if key.endswith('_usec') else 'count','current_container_cgroup_cumulative')
                                   if key in stats else unknown('field_unavailable','current_container_cgroup') for key in CPU_FIELDS}
    except (OSError,ValueError,UnicodeError):
        result['cgroup_cpu_stat']={key:unknown('cpu_stat_unavailable','current_container_cgroup') for key in CPU_FIELDS}
    return result

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def worker_final(output,*,start_sample=None,stage='unknown',completed=False):
    """Best effort: never masks a model exception or emits raw exception strings."""
    try:
        final=sample()
        receipt={'schema':'tts-remaining-worker-resources-v1','phase':'remaining',
                 'worker_finally_observed':True,'completed':completed is True,
                 'stage':stage if stage in STAGES else 'unknown',
                 'start':start_sample,'final':final,
                 'interpretation':'RSS is one process; cgroup peak includes its container tree. No probe measurements inferred.'}
        atomic(Path(output)/'remaining-worker-resources.json',receipt)
        return True
    except Exception:
        return False

def host_summary(output,*,container_state=None):
    """Invoke even if worker was hard-killed. No receipt means unknown, not zero."""
    output=Path(output);path=output/'remaining-worker-resources.json'
    result={'schema':'tts-remaining-host-resource-status-v1','phase':'remaining',
            'worker_measurements':'unknown','reason':'worker_final_receipt_missing_or_invalid',
            'hard_kill_can_prevent_finally':True}
    try:
        if path.is_symlink() or path.stat().st_size>65536:raise ValueError('invalid resource receipt')
        value=json.loads(path.read_text())
        if value.get('schema')!='tts-remaining-worker-resources-v1' or value.get('worker_finally_observed') is not True:
            raise ValueError('incomplete resource receipt')
        result.update(worker_measurements='receipt_present',reason=None)
    except (OSError,ValueError,TypeError):pass
    if isinstance(container_state,dict):
        result['container_state']={key:container_state[key] for key in ('ExitCode','OOMKilled') if key in container_state and type(container_state[key]) is (bool if key=='OOMKilled' else int)}
    try:
        atomic(output/'remaining-host-resource-status.json',result)
        return True
    except Exception:return False
