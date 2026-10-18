"""Stdlib process-group supervisor: monotonic wall, sampled live RSS, wait4 exit.
wait4 ru_maxrss may contain inherited pre-exec high water; it is recorded
separately and is never described as current tested-process resident memory.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

def rss_tree(parent):
    states={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                    for line in (entry/'status').read_text().splitlines() if ':' in line}
            states[int(entry.name)]=(int(fields['PPid']),int(fields.get('VmRSS','0 kB').split()[0])*1024)
        except (OSError,KeyError,ValueError,ProcessLookupError): pass
    included={parent}
    while True:
        extra={pid for pid,(ppid,_) in states.items() if ppid in included}-included
        if not extra: break
        included.update(extra)
    return sum(states.get(pid,(0,0))[1] for pid in included)

def run_child(args,cwd,stdout_path,stderr_path,wall,rss,output_cap,env=None,outer_guard=None):
    start=time.monotonic()
    row={'command':args,'wall_cap_seconds':wall,'rss_cap_bytes':rss,
         'stdout_stderr_cap_bytes':output_cap,'status':'reserved',
         'sampled_live_process_tree_peak_rss_bytes':0,'live_rss_samples':0,
         'wait4_ru_maxrss_semantics':'Linux KiB, may include inherited pre-exec high-water; not current tested-process RSS'}
    process=None; usage=None; status=None; guard_error=None
    with Path(stdout_path).open('xb') as out, Path(stderr_path).open('xb') as err:
        try:
            process=subprocess.Popen(args,cwd=cwd,stdout=out,stderr=err,env=env,start_new_session=True)
            row['pid']=process.pid; row['status']='running'
            while True:
                pid,status_now,usage_now=os.wait4(process.pid,os.WNOHANG)
                if pid:
                    status=status_now; usage=usage_now
                    process.returncode=os.waitstatus_to_exitcode(status)
                    break
                used=rss_tree(process.pid)
                row['live_rss_samples']+=1
                row['sampled_live_process_tree_peak_rss_bytes']=max(row['sampled_live_process_tree_peak_rss_bytes'],used)
                if used>=rss: raise RuntimeError('live process-tree RSS cap')
                if time.monotonic()-start>=wall: raise RuntimeError('wall cap')
                if Path(stdout_path).stat().st_size+Path(stderr_path).stat().st_size>=output_cap:
                    raise RuntimeError('stdout/stderr output cap')
                if outer_guard is not None: outer_guard()
                time.sleep(0.01)
        except BaseException as exc:
            guard_error=type(exc).__name__+': '+str(exc)
            if process is not None and process.returncode is None:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                try:
                    _,status,usage=os.wait4(process.pid,0)
                    process.returncode=os.waitstatus_to_exitcode(status)
                except ChildProcessError: pass
    row['elapsed_seconds']=time.monotonic()-start
    row['output_bytes']=Path(stdout_path).stat().st_size+Path(stderr_path).stat().st_size
    row['returncode']=None if process is None else process.returncode
    if usage is not None:
        row.update(wait4_ru_maxrss_bytes_including_possible_inherited_peak=usage.ru_maxrss*1024,
                   wait4_user_cpu_seconds=usage.ru_utime,wait4_system_cpu_seconds=usage.ru_stime)
    # Post-completion checks catch short-lived writers that exit between samples.
    if guard_error is None and row['output_bytes']>=output_cap: guard_error='RuntimeError: stdout/stderr output cap'
    if guard_error is None and row['elapsed_seconds']>=wall: guard_error='RuntimeError: wall cap'
    if guard_error is None and row.get('wait4_ru_maxrss_bytes_including_possible_inherited_peak',0)>=rss:
        guard_error='RuntimeError: conservative wait4 high-water cap (may include inherited peak)'
    if guard_error:
        row.update(status='guard_or_launch_failed',error=guard_error)
    else:
        row['status']='passed' if row['returncode']==0 else 'failed_exit'
    return row
