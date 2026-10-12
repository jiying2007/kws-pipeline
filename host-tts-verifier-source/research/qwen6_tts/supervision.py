"""Linux outer process-group supervisor. RSS and disk guards are sampled.

RLIMIT_CPU is kernel enforced per process (inherited by descendants), never a
claim of aggregate hard process-tree CPU or continuous hard memory control.
"""
import errno
import os
import stat
from contextlib import ExitStack
from pathlib import Path
import resource
import signal
import subprocess
import time

GiB=1024**3

def require(condition, message):
    if not condition: raise ValueError(message)


def safe_error(error):
    """Fixed metadata only: never paths, messages, traceback locals or payloads."""
    kinds=(TimeoutError,ValueError,TypeError,OSError,RuntimeError,KeyboardInterrupt,SystemExit)
    kind=next((t.__name__ for t in kinds if issubclass(type(error),t)),'OtherError')
    result={'error_type':kind}
    if issubclass(type(error),OSError):
        number=OSError.errno.__get__(error)
        if type(number) is int and number in errno.errorcode:result['errno']=number
    return result


def close_sample_fd(fd, observation, primary):
    """Close once; a secondary close failure cannot replace a primary scan error."""
    try:os.close(fd)
    except BaseException as error:
        observation['fd_close_error_count']=observation.get('fd_close_error_count',0)+1
        if 'fd_close_first_error' not in observation:observation['fd_close_first_error']=safe_error(error)
        return primary if primary is not None else error
    return primary


def tree_bytes(root, observation=None, *, expected_root_identity=None,
        parent_fd=None, deadline=None, limit=None):
    """One non-atomic, nofollow sample; only ENOENT may make it incomplete."""
    seen={} if observation is None else observation
    seen.update(observed_bytes=0,regular_files=0,symlinks=0,directories=0,
        vanished_entries=0,missing_root=False,root_missing_at_start=False,root_observed=False,
        complete=False,coverage='incomplete_lower_bound',stop_reason=None)
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
    if parent_fd is not None:require(type(root) is str and '/' not in root and root!='..','fixed relative sampling root')
    def guard():
        if deadline is not None and time.monotonic()>=deadline:
            seen['stop_reason']='deadline_exhausted'
            raise ValueError('host wall ceiling during filesystem sample')
    def add_bytes(info,kind):
        seen['observed_bytes']+=info.st_size;seen[kind]+=1
        if limit is not None and seen['observed_bytes']>limit:
            seen['stop_reason']='cap_exceeded'
            raise ValueError('filesystem stop threshold exceeded')
    def vanished(error, root=False):
        # A failed filesystem call is skippable only for a genuine ENOENT.
        if seen.get('fd_close_error_count',0) or type(OSError.errno.__get__(error)) is not int or OSError.errno.__get__(error)!=errno.ENOENT:raise error
        seen['vanished_entries']+=1
        if root:seen['missing_root']=True
    def walk(fd):
        try:
            guard()
            with os.scandir(fd) as entries:
                for entry in entries:
                    guard()
                    try:info=os.stat(entry.name,dir_fd=fd,follow_symlinks=False)
                    except OSError as error:
                        vanished(error);continue
                    mode=info.st_mode
                    if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                        add_bytes(info,'regular_files' if stat.S_ISREG(mode) else 'symlinks')
                    elif stat.S_ISDIR(mode):
                        child=None;child_error=None
                        try:
                            try:
                                guard();child=os.open(entry.name,flags,dir_fd=fd)
                                actual=os.fstat(child)
                                require(stat.S_ISDIR(actual.st_mode) and
                                    (actual.st_dev,actual.st_ino)==(info.st_dev,info.st_ino),
                                    'sample directory identity changed')
                                seen['directories']+=1;walk(child)
                            except OSError as error:vanished(error)
                        except BaseException as error:child_error=error;raise
                        finally:
                            if child is not None:
                                failure=close_sample_fd(child,seen,child_error)
                                if child_error is None and failure is not None:raise failure
        except OSError as error:vanished(error)
    fd=None;root_info=None;root_error=None
    try:
        guard()
        try:root_info=os.stat(root,dir_fd=parent_fd,follow_symlinks=False)
        except OSError as error:
            vanished(error,root=True);seen['root_missing_at_start']=True;seen['vanished_entries']-=1
        if root_info is not None:
            seen['root_observed']=True
            if expected_root_identity is not None:
                require((root_info.st_dev,root_info.st_ino)==expected_root_identity,'sample workspace identity changed')
            guard();fd=os.open(root,flags,dir_fd=parent_fd)
            actual=os.fstat(fd)
            require(stat.S_ISDIR(actual.st_mode) and
                (actual.st_dev,actual.st_ino)==(root_info.st_dev,root_info.st_ino),
                'sample root directory identity changed')
            walk(fd)
    except OSError as error:
        try:vanished(error,root=fd is None)
        except BaseException as failure:root_error=failure;raise
    except BaseException as error:root_error=error;raise
    finally:
        if fd is not None:
            failure=close_sample_fd(fd,seen,root_error)
            if root_error is None and failure is not None:raise failure
    seen['complete']=seen['vanished_entries']==0 and not seen['missing_root']
    seen['coverage']='complete_non_atomic_sample' if seen['complete'] else 'incomplete_lower_bound'
    return seen['observed_bytes']


class FilesystemSampler:
    """Three fixed disk scopes; at most one ENOENT-only resample per check."""
    def __init__(self,root,deadline):
        self.root=Path(root);info=os.stat(self.root,follow_symlinks=False)
        require(stat.S_ISDIR(info.st_mode),'owned workspace directory')
        self.root_identity=(info.st_dev,info.st_ino);self.deadline=deadline
        self.receipt={'schema':'qwen6-filesystem-sampling-v1',
            'semantics':'non_atomic_observed_bytes_not_peak_or_quota',
            'maximum_scans_per_check_scope':2,'scopes':{}}

    def sample_bytes(self,root,scope,limit,*,optional_root=False):
        expected={'workspace':self.root,'installed':self.root/'venv','buildtmp':self.root/'buildtmp'}
        require(scope in expected and Path(root)==expected[scope] and type(limit) is int and limit>=0,
            'fixed sampling scope/root/cap')
        row=self.receipt['scopes'].setdefault(scope,{'checks':0,'scans':0,'incomplete_scans':0,
            'vanished_entries':0,'absent_optional_roots':0,'observed_root_once':False,
            'last_samples':[],'selected_sample':None,'status':'not_started'})
        row.update(checks=row['checks']+1,last_samples=[],selected_sample=None,status='started')
        for attempt in range(2):
            if time.monotonic()>=self.deadline:
                row['status']='deadline_exhausted'
                raise ValueError('host wall ceiling before filesystem sample')
            sample={'attempt':attempt+1,'status':'started'};row['last_samples'].append(sample);row['scans']+=1
            try:
                workspace=None;anchor_error=None
                try:
                    workspace=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
                    anchor=os.fstat(workspace)
                    require(stat.S_ISDIR(anchor.st_mode) and
                        (anchor.st_dev,anchor.st_ino)==self.root_identity,'sample workspace identity changed')
                    relative='.' if scope=='workspace' else Path(root).name
                    amount=tree_bytes(relative,sample,parent_fd=workspace,deadline=self.deadline,limit=limit,
                        expected_root_identity=self.root_identity if scope=='workspace' else None)
                except BaseException as error:anchor_error=error;raise
                finally:
                    if workspace is not None:
                        failure=close_sample_fd(workspace,sample,anchor_error)
                        if anchor_error is None and failure is not None:raise failure
                if optional_root and scope!='workspace' and sample['root_missing_at_start'] and not row['observed_root_once']:
                    sample.update(complete=True,coverage='absent_optional_root');row['absent_optional_roots']+=1
                elif sample['missing_root']:
                    row['status']='root_missing'
                    raise ValueError('required or previously observed sample root missing')
                sample['status']='observed'
                if amount>limit:
                    row['status']='cap_exceeded';sample['status']='cap_exceeded'
                    raise ValueError('filesystem stop threshold exceeded')
                if sample['complete']:
                    if time.monotonic()>=self.deadline:
                        sample['stop_reason']='deadline_exhausted'
                        raise ValueError('host wall ceiling after filesystem sample')
                    row.update(selected_sample=attempt+1,status='complete')
                    return amount
            except BaseException as error:
                sample['error']=safe_error(error)
                if sample['status']!='cap_exceeded':sample['status']='failed'
                if row['status']=='started':row['status']=sample.get('stop_reason') or 'failed'
                raise
            finally:
                row['observed_root_once']=row['observed_root_once'] or sample.get('root_observed',False)
                row['vanished_entries']+=sample.get('vanished_entries',0)
                if sample.get('complete') is not True:row['incomplete_scans']+=1
        row['status']='incomplete_after_one_resample'
        raise ValueError('filesystem sample incomplete after one bounded resample')


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
    sampler=FilesystemSampler(workspace,min(started+wall_seconds,job_deadline))
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    maximum={'rss_bytes':None,'live_cpu_seconds':None,'workspace_bytes':None,'installed_bytes':None,'buildtmp_bytes':None}
    samples=0;reason=None;next_disk=0;filesystem_error=None
    # Logs go to runtime only; copied into public artifact only through final allowlist.
    with ExitStack() as stack:
        log=stack.enter_context(Path(log_path).open('xb'))
        errors=stack.enter_context(Path(stderr_path).open('xb')) if stderr_path is not None else subprocess.STDOUT
        if time.monotonic()>=sampler.deadline:raise TimeoutError('No wall budget remains; process not started')
        p=subprocess.Popen(command,cwd=cwd,env=env,start_new_session=True,stdout=log,stderr=errors,
            preexec_fn=lambda:child_limits(cpu_seconds,cores))
        try:
            while p.poll() is None:
                now=time.monotonic();s=group_sample(p.pid);samples+=1
                for k in ('rss_bytes','live_cpu_seconds'): maximum[k]=max(maximum[k] or 0,s[k])
                if now>=sampler.deadline:
                    reason='whole_job_deadline' if job_deadline<=started+wall_seconds else 'wall_deadline'
                elif s['rss_bytes']>rss_bytes: reason='sampled_group_rss'
                if reason is None and now>=next_disk:
                    for k,path,cap in (('workspace_bytes',workspace,10*GiB),('installed_bytes',installed,3*GiB),('buildtmp_bytes',buildtmp,2*GiB)):
                        scope=k.removesuffix('_bytes')
                        try:
                            amount=sampler.sample_bytes(path,scope,cap,optional_root=setup and scope!='workspace')
                        except Exception as error:
                            filesystem_error=safe_error(error)
                            status=sampler.receipt['scopes'].get(scope,{}).get('status')
                            if status=='cap_exceeded':reason='sampled_'+scope+'_cap'
                            elif status=='deadline_exhausted':
                                reason='whole_job_deadline' if job_deadline<=started+wall_seconds else 'wall_deadline'
                            else:reason='sampled_filesystem_error'
                            break
                        # Only selected complete samples enter the successful maximum.
                        maximum[k]=max(maximum[k] or 0,amount)
                    next_disk=now+1
                now=time.monotonic()
                if reason is None:
                    if now>=sampler.deadline:
                        reason='whole_job_deadline' if job_deadline<=started+wall_seconds else 'wall_deadline'
                    elif Path(log_path).stat().st_size+(Path(stderr_path).stat().st_size if stderr_path is not None else 0)>8*1024**2: reason='sampled_log_cap'
                if reason:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    break
                time.sleep(.25)
            returncode=p.wait()
        finally:
            # Also kill/reap on sampling, disk-accounting or supervisor exceptions.
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    # A child may exit 0 during a failed scan; old callers check returncode only.
    return {'returncode':returncode if returncode!=0 or reason is None else 1,
        'process_returncode':returncode,'filesystem_sampling':sampler.receipt,'filesystem_error':filesystem_error,
        'termination_reason':reason or ('normal_exit' if returncode==0 else 'process_failure_or_kernel_limit'),
        'wall_seconds':time.monotonic()-started,'reaped_children_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'sample_count':samples,'sample_interval_seconds':.25,'max_sampled':maximum,
        'cpu_limit_seconds_per_process':cpu_seconds,'wall_limit_seconds':wall_seconds,'affinity_core_limit':cores,
        'rss_mode':'sampled_process_group_threshold_not_continuous_hard_limit','rss_threshold_bytes':rss_bytes,
        'cpu_mode':'kernel_RLIMIT_CPU_per_process_not_aggregate_tree_limit','setup':setup}
