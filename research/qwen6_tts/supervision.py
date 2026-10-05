"""Linux outer process-group supervisor. RSS and disk guards are sampled.

RLIMIT_CPU is kernel enforced per process (inherited by descendants), never a
claim of aggregate hard process-tree CPU or continuous hard memory control.
"""
import os
from contextlib import ExitStack
from pathlib import Path
import resource
import signal
import subprocess
import time

GiB=1024**3

def tree_bytes(root):
    root=Path(root)
    if not root.exists(): return 0
    total=0
    for p in root.rglob('*'):
        if p.is_symlink(): continue  # venv interpreter symlinks; never follow for accounting
        if p.is_file(): total+=p.stat().st_size
    return total

def group_sample(group,proc=Path('/proc')):
    rss=0;cpu=0.;members=[];ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    for p in proc.iterdir():
        if not p.name.isdigit(): continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group: continue
            rss+=int(fields[21])*page;cpu+=(int(fields[11])+int(fields[12]))/ticks;members.append(int(p.name))
        except (OSError,ValueError,IndexError): continue
    return {'rss_bytes':rss,'live_cpu_seconds':cpu,'members':members}

def child_limits(cpu_seconds,cores):
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_seconds,cpu_seconds))
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:cores]))

def supervise(command,*,cwd,env,wall_seconds,cpu_seconds,rss_bytes,cores,workspace,installed,buildtmp,log_path,job_deadline,setup=False,stderr_path=None):
    started=time.monotonic()
    if wall_seconds <= 0 or started >= job_deadline: raise TimeoutError('No wall budget remains; process not started')
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    maximum={'rss_bytes':None,'live_cpu_seconds':None,'workspace_bytes':None,'installed_bytes':None,'buildtmp_bytes':None}
    samples=0;reason=None;next_disk=0
    # Logs go to runtime only; copied into public artifact only through final allowlist.
    with ExitStack() as stack:
        log=stack.enter_context(Path(log_path).open('xb'))
        errors=stack.enter_context(Path(stderr_path).open('xb')) if stderr_path is not None else subprocess.STDOUT
        p=subprocess.Popen(command,cwd=cwd,env=env,start_new_session=True,stdout=log,stderr=errors,
            preexec_fn=lambda:child_limits(cpu_seconds,cores))
        try:
            while p.poll() is None:
                now=time.monotonic();s=group_sample(p.pid);samples+=1
                for k in ('rss_bytes','live_cpu_seconds'): maximum[k]=max(maximum[k] or 0,s[k])
                if now>=next_disk:
                    for k,path in (('workspace_bytes',workspace),('installed_bytes',installed),('buildtmp_bytes',buildtmp)):
                        maximum[k]=max(maximum[k] or 0,tree_bytes(path))
                    next_disk=now+1
                if now-started>=wall_seconds: reason='wall_deadline'
                elif now>=job_deadline: reason='whole_job_deadline'
                elif s['rss_bytes']>rss_bytes: reason='sampled_group_rss'
                elif maximum['workspace_bytes']>10*GiB: reason='sampled_workspace_cap'
                elif maximum['installed_bytes']>3*GiB: reason='sampled_installed_cap'
                elif maximum['buildtmp_bytes']>2*GiB: reason='sampled_buildtmp_cap'
                elif Path(log_path).stat().st_size+(Path(stderr_path).stat().st_size if stderr_path is not None else 0)>8*1024**2: reason='sampled_log_cap'
                if reason:
                    os.killpg(p.pid,signal.SIGKILL);break
                time.sleep(.25)
            returncode=p.wait()
        finally:
            # Also kill/reap on sampling, disk-accounting or supervisor exceptions.
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'returncode':returncode,'termination_reason':reason or ('normal_exit' if returncode==0 else 'process_failure_or_kernel_limit'),
        'wall_seconds':time.monotonic()-started,'reaped_children_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'sample_count':samples,'sample_interval_seconds':.25,'max_sampled':maximum,
        'cpu_limit_seconds_per_process':cpu_seconds,'wall_limit_seconds':wall_seconds,'affinity_core_limit':cores,
        'rss_mode':'sampled_process_group_threshold_not_continuous_hard_limit','rss_threshold_bytes':rss_bytes,
        'cpu_mode':'kernel_RLIMIT_CPU_per_process_not_aggregate_tree_limit','setup':setup}
