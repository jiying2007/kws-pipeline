"""Narrow retained resource logic; no old campaign execution or decode entrypoint.

Selected source functions preserve the reviewed qwen12 resource semantics.
Only the public diagnostic source-path literal is updated to this actual file.
Legacy resource/gate schema names remain explicit retained-format provenance;
new campaign identity, input authorities and eighteen claims belong to the host.
"""
from pathlib import Path
import errno
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time

from campaign_contract import FAMILIES, LIMITS, MIB, GIB, canonical, read_json, require
from cpu_budget import CpuLedger, parse_cpu_stat

def close_sample_fd(fd, observation, primary):
    """Close once; a secondary close failure cannot replace a primary scan error."""
    try:os.close(fd)
    except BaseException as error:
        observation['fd_close_error_count']=observation.get('fd_close_error_count',0)+1
        if 'fd_close_first_error' not in observation:observation['fd_close_first_error']=safe_error(error)
        return primary if primary is not None else error
    return primary


def tree_bytes(root, observation=None, *, root_json_only=False, expected_root_identity=None,
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
                        if not root_json_only or (stat.S_ISREG(mode) and entry.name.endswith('.json')):
                            add_bytes(info,'regular_files' if stat.S_ISREG(mode) else 'symlinks')
                    elif stat.S_ISDIR(mode) and not root_json_only:
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


FILESYSTEM_SAMPLE_SCOPES=('workspace','public_output','public_frozen','public_whisper_setup',
    'public_policy','public_asr_setup','public_root_json','phase_output')


_HOST_ERROR_ERRNOS=frozenset((1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,
    22,23,24,26,27,28,29,30,31,32,35,36,38,39,40,75,95,110,116,122))


MAX_HOST_ERROR_FRAMES=8


MAX_HOST_ERROR_SCAN=32


MAX_HOST_ERROR_BYTES=2048


def _host_diagnostic_codes(module_code):
    """Snapshot only this executing, frozen source's actual code objects once."""
    pending=[module_code];result={}
    for _ in range(512):
        if not pending:break
        code=pending.pop()
        result[id(code)]=(code,code.co_name)
        pending.extend(value for value in code.co_consts if type(value) is type(module_code))
    return result


def _host_error_frames(error):
    # Built-in traceback slots bypass subclass properties. No frame locals,
    # source lines, exception chains, filenames or external names are inspected.
    frames=[];truncated=False;trace=BaseException.__traceback__.__get__(error)
    for _ in range(MAX_HOST_ERROR_SCAN):
        if trace is None:break
        code=trace.tb_frame.f_code;known=_HOST_DIAGNOSTIC_CODES.get(id(code))
        frame={'file':'<redacted>','function':'<redacted>','line':None}
        if known is not None and known[0] is code:
            line=trace.tb_lineno
            frame={'file':'research/k1-punctuation6-asr18-v1/resource_host.py',
                'function':known[1],'line':line if type(line) is int and 1<=line<=1000000 else None}
        frames.append(frame)
        if len(frames)>MAX_HOST_ERROR_FRAMES:del frames[0];truncated=True
        trace=trace.tb_next
    return {'frames':frames,'frames_truncated':truncated or trace is not None}


def safe_error(error):
    """Bounded fixed public metadata; diagnostic failure never replaces error."""
    kinds=(subprocess.TimeoutExpired,subprocess.CalledProcessError,TimeoutError,
        ValueError,TypeError,KeyError,OSError,RuntimeError,KeyboardInterrupt,SystemExit)
    # issubclass on the actual type cannot consult a hostile error.__class__.
    kind=next((t.__name__ for t in kinds if issubclass(type(error),t)),'OtherError')
    value={'error_type':kind}
    if kind=='OtherError':return value
    try:
        if issubclass(type(error),subprocess.CalledProcessError):
            fields=BaseException.__dict__['__dict__'].__get__(error)
            number=dict.get(fields,'returncode')
            if type(number) is int and -255<=number<=255:value['returncode']=number
        diagnostic={}
        if issubclass(type(error),OSError):
            number=OSError.errno.__get__(error)
            if type(number) is int and number in _HOST_ERROR_ERRNOS:diagnostic['errno']=number
        diagnostic.update(_host_error_frames(error))
        if len(canonical({**value,**diagnostic}))<=MAX_HOST_ERROR_BYTES:value.update(diagnostic)
    except BaseException:pass
    return value


class MeteredDocker:
    """Immutable owned IDs; one bounded cleanup and one final CPU close per scope."""
    def __init__(self,root,deadline):
        self.root=Path(root);root_stat=os.stat(self.root,follow_symlinks=False)
        require(stat.S_ISDIR(root_stat.st_mode),'owned workspace directory')
        self.root_identity=(root_stat.st_dev,root_stat.st_ino)
        self.gates=self.root/'gates';self.gates.mkdir()
        self.deadline=deadline;self.stage='setup';self.names={};self.ledger=CpuLedger();self.cleanup=[]
        self.supervisions=[];self.supervision_failure=None
        self.filesystem_sampling={'schema':'qwen12-filesystem-sampling-v1',
            'semantics':'non_atomic_observed_bytes_not_peak_or_quota','maximum_scans_per_check_scope':2,'scopes':{}}

    def raw(self,*args,timeout=5):
        require(args and args[0] in ('create','inspect','start','rm','pause','kill','ps'),'fixed Docker operation')
        require(0<timeout<=5,'bounded Docker timeout')
        p=subprocess.run(['docker',*args],timeout=timeout,check=True,capture_output=True)
        require(len(p.stdout)+len(p.stderr)<=2*MIB,'bounded Docker response');return p.stdout.decode()

    def gate(self,name):
        require(re.fullmatch('[a-z0-9_.-]{1,120}',name) and name not in self.names,'new container only')
        require(self.stage in LIMITS['cpu_threshold_seconds'],'fixed metered stage')
        gate=self.gates/name;gate.mkdir(mode=0o700)
        self.names[name]={'gate':gate,'stage':self.stage,'create_attempted':False,'created':False,
            'container_id':None,'start_attempted':False,'registered':False,'cpu_closed':False,
            'removed':False,'cleanup_started':False,'cleanup_ok':False}
        return gate

    def remaining_timeout(self,timeout=5):
        remaining=self.deadline-time.monotonic()
        require(remaining>0,'Docker operation deadline exhausted')
        return min(timeout,5,remaining)

    def sample_bytes(self,root,scope,limit,*,root_json_only=False,optional_root=False):
        require(scope in FILESYSTEM_SAMPLE_SCOPES and type(limit) is int and limit>=0,'fixed sampling scope/cap')
        scopes=self.filesystem_sampling['scopes']
        row=scopes.setdefault(scope,{'checks':0,'scans':0,'incomplete_scans':0,'vanished_entries':0,
            'absent_optional_roots':0,'observed_root_once':False,'last_samples':[],'selected_sample':None,'status':'not_started'})
        row.update(checks=row['checks']+1,last_samples=[],selected_sample=None,status='started')
        for attempt in range(2):
            if time.monotonic()>=self.deadline:
                row['status']='deadline_exhausted'
                raise ValueError('host wall ceiling before filesystem sample')
            sample={'attempt':attempt+1,'status':'started'};row['last_samples'].append(sample);row['scans']+=1
            try:
                # Resolve only fixed local scope roots from a verified workspace FD.
                workspace=parent=None;anchor_error=None
                try:
                    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
                    workspace=os.open(self.root,flags)
                    anchor=os.fstat(workspace)
                    require(stat.S_ISDIR(anchor.st_mode) and
                        (anchor.st_dev,anchor.st_ino)==self.root_identity,'sample workspace identity changed')
                    if Path(root)==self.root:
                        relative='.';parent=workspace
                    elif Path(root).parent==self.root:
                        relative=Path(root).name;parent=workspace
                        require(relative in ('output','frozen','whisper-setup','policy','asr-setup'),'fixed public sampling child')
                    else:
                        require(scope=='phase_output' and Path(root).parent==self.root/'output' and
                            Path(root).name in FAMILIES,'fixed family sampling child')
                        info=os.stat('output',dir_fd=workspace,follow_symlinks=False)
                        require(time.monotonic()<self.deadline,'host wall ceiling before output sampling root')
                        parent=os.open('output',flags,dir_fd=workspace);actual=os.fstat(parent)
                        require(stat.S_ISDIR(actual.st_mode) and
                            (actual.st_dev,actual.st_ino)==(info.st_dev,info.st_ino),'sample output parent identity changed')
                        relative=Path(root).name
                    amount=tree_bytes(relative,sample,root_json_only=root_json_only,
                        expected_root_identity=self.root_identity if parent==workspace and relative=='.' else None,
                        parent_fd=parent,deadline=self.deadline,limit=limit)
                except BaseException as error:anchor_error=error;raise
                finally:
                    failure=anchor_error
                    if parent is not None and parent!=workspace:failure=close_sample_fd(parent,sample,failure)
                    if workspace is not None:failure=close_sample_fd(workspace,sample,failure)
                    if anchor_error is None and failure is not None:raise failure
                # Missing optional output roots are explicit absence, not partial subtrees.
                if optional_root and sample['root_missing_at_start'] and not row['observed_root_once']:
                    sample.update(complete=True,coverage='absent_optional_root');row['absent_optional_roots']+=1
                elif sample['missing_root']:
                    row['status']='root_missing'
                    raise ValueError('required or previously observed sample root missing')
                sample['status']='observed'
                if amount>limit:
                    row['status']='cap_exceeded';sample['status']='cap_exceeded'
                    raise ValueError('workspace stop threshold exceeded' if scope=='workspace' else 'output stop threshold exceeded')
                if sample['complete']:
                    require(time.monotonic()<self.deadline,'host wall ceiling after filesystem sample')
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

    def check(self):
        require(time.monotonic()<self.deadline,'host wall ceiling')
        self.sample_bytes(self.root,'workspace',LIMITS['workspace_bytes'])
        public_bytes=0
        for name,scope in (('output','public_output'),('frozen','public_frozen'),
                ('whisper-setup','public_whisper_setup'),('policy','public_policy'),('asr-setup','public_asr_setup')):
            public_bytes+=self.sample_bytes(self.root/name,scope,LIMITS['output_bytes']-public_bytes,optional_root=True)
        public_bytes+=self.sample_bytes(self.root,'public_root_json',LIMITS['output_bytes']-public_bytes,root_json_only=True)
        require(public_bytes<=LIMITS['output_bytes'],'campaign32MiB output stop threshold')
        exceeded=self.ledger.sample();atomic(self.root/'cpu-receipt.json',self.ledger.receipt())
        require(not exceeded,'cumulative CPU threshold exceeded')

    def __call__(self,*args,timeout=5):
        require(args and args[0] in ('create','inspect','start','rm','kill'),'fixed supervised operation')
        if args[0]=='rm':return self.remove(args[-1],timeout=timeout)
        self.check()
        if args[0]=='create':
            name=args[args.index('--name')+1];require(name in self.names,'registered gate')
            row=self.names[name];require(not row['create_attempted'],'single create attempt')
            row['create_attempted']=True
            result=self.raw(*args,timeout=self.remaining_timeout(timeout))
            # A failed/uncertain create cannot prove ownership of a reusable name.
            require(re.fullmatch(r'[0-9a-f]{64}\n?',result) is not None,'full created container ID')
            cid=result.rstrip('\n')
            require(all(r['container_id']!=cid for r in self.names.values()),'unique created container ID')
            row.update(created=True,container_id=cid)
            return result
        name=args[-1];require(name in self.names,'own registered container')
        row=self.names[name];cid=row['container_id']
        require(row['created'] and not row['cleanup_started'] and re.fullmatch('[0-9a-f]{64}',cid),'owned live container ID')
        if args[0]=='start':
            require(not row['start_attempted'],'single start attempt');row['start_attempted']=True
        result=self.raw(*args[:-1],cid,timeout=self.remaining_timeout(timeout))
        if args[0]=='inspect' and '--format' not in args:
            inspected=json.loads(result)
            require(len(inspected)==1 and inspected[0].get('Id')==cid,'inspected owned immutable ID')
        if args[0]=='start':
            inspected=json.loads(self.raw('inspect',cid,timeout=self.remaining_timeout()))
            require(len(inspected)==1 and inspected[0].get('Id')==cid,'started owned immutable ID')
            r=inspected[0];pid=r['State']['Pid']
            require(type(pid) is int and pid>0 and r['State']['Running'],'live gated PID')
            lines=Path('/proc/'+str(pid)+'/cgroup').read_text().splitlines()
            require(len(lines)==1 and lines[0].startswith('0::/'),'host cgroupv2 path')
            path=Path('/sys/fs/cgroup')/lines[0][4:]
            require(path!=Path('/sys/fs/cgroup') and '..' not in path.parts and
                not any(p.is_symlink() for p in (path,*path.parents)),'dedicated cgroup path')
            row['cpu_path']=path/'cpu.stat';row['cgroup_inode']=path.stat().st_ino
            def reader():
                require(path.stat().st_ino==row['cgroup_inode'],'cgroup lifetime identity')
                return parse_cpu_stat(row['cpu_path'].read_text())
            row['reader']=reader;self.ledger.register(name,row['stage'],reader);row['registered']=True
            self.check()
            with (row['gate']/'go').open('x') as f:f.write('one-shot')
        return result

    def gate_result(self,name):
        p=self.names[name]['gate']/'result.json'
        try:
            if not p.exists() and not p.is_symlink():return {'status':'missing'}
            result=read_json(p,4096)
            require(type(result) is dict and set(result)=={'schema','exit_code'} and
                result['schema']=='qwen12-gated-command-v1' and type(result['exit_code']) is int and
                -255<=result['exit_code']<=255,'bounded gate result')
            return {'status':'valid',**result}
        except BaseException as error:return {'status':'invalid',**safe_error(error)}

    def completed(self,name):
        self.check();result=self.gate_result(name)
        if result['status']=='missing':
            row=self.names[name];require(row['created'] and not row['cleanup_started'],'live owned gate')
            state=json.loads(self.raw('inspect','--format','{{json .State}}',row['container_id'],timeout=self.remaining_timeout()))
            require(state['Running'] is True and state['OOMKilled'] is False,'gated process unexpectedly exited')
            return False
        require(result['status']=='valid' and result['exit_code']==0,'child failure; no retry')
        return True

    def remove(self,name,timeout=5):
        require(name in self.names,'only own container cleanup');row=self.names[name]
        if row['cleanup_started']:
            require(row['cleanup_ok'],'previous cleanup unconfirmed or accounting invalid')
            return ''
        row['cleanup_started']=True;started=time.monotonic();deadline=started+15
        cid=row['container_id'];errors=[]
        record={'name':name,'container_id':cid,'status':'removal_unconfirmed','stage':row['stage'],
            'ownership':'verified' if row['created'] else 'unknown','operations':[],
            'cleanup_budget_seconds':15,'host_deadline_expired_at_start':started>=self.deadline,
            'host_deadline_overrun':started>=self.deadline,'pause_acknowledged':False,'errors':errors}
        self.cleanup.append(record)
        def operation(kind,*args):
            before=time.monotonic();item={'operation':kind,'status':'not_started','timeout_seconds':0}
            record['operations'].append(item)
            try:
                # Reserve the last five seconds for stopping, before optional absence proof.
                operation_deadline=deadline-5 if kind in ('pause','inspect_paused') else deadline
                # Optional proof cannot extend the host window. Stopping an already
                # running owned scope keeps the original best-effort cleanup reserve.
                if kind=='verify_absence':operation_deadline=min(operation_deadline,self.deadline)
                remaining=operation_deadline-time.monotonic();require(remaining>0,'cleanup deadline exhausted')
                allowance=min(timeout,1 if kind=='verify_absence' else 5,remaining);item['timeout_seconds']=allowance
                item['status']='started';value=self.raw(*args,timeout=allowance)
                if kind=='verify_absence':require(time.monotonic()<operation_deadline,'absence proof deadline exhausted')
                item.update(status='succeeded',returncode=0);return value
            except BaseException as error:
                item.update(status='failed',**safe_error(error));raise
            finally:item['wall_seconds']=time.monotonic()-before
        try:
            require(row['created'] and re.fullmatch('[0-9a-f]{64}',cid),'container ownership unknown')
            if row['registered'] and not row['cpu_closed']:
                try:
                    operation('pause','pause',cid)
                    state=json.loads(operation('inspect_paused','inspect','--format','{{json .State}}',cid))
                    require(state.get('Paused') is True,'pause acknowledged before final CPU sample')
                    record['pause_acknowledged']=True
                    before=time.monotonic();item={'operation':'final_cpu_close','status':'started'}
                    record['operations'].append(item)
                    try:
                        require(time.monotonic()<deadline-5,'final CPU stop reserve')
                        self.ledger.close(name,row['reader']());row['cpu_closed']=True
                        item['status']='succeeded'
                    except BaseException as error:item.update(status='failed',**safe_error(error));raise
                    finally:item['wall_seconds']=time.monotonic()-before
                except BaseException as error:errors.append({'operation':'final_accounting',**safe_error(error)})
            # Even invalid accounting must not knowingly leave an owned scope running.
            try:
                operation('force_remove','rm','--force',cid);row['removed']=True
                record['removal_proof']='acknowledged_remove'
            except BaseException as error:
                errors.append({'operation':'force_remove',**safe_error(error)})
                try:
                    # The one force-remove may still be completing in the daemon.
                    # Observe only this immutable ID; never retransmit removal or
                    # accept a failed, malformed or foreign-ID response as absence.
                    # Four close/stop operations plus at most ten reads fit the
                    # existing sixteen-operation public receipt bound.
                    for attempt in range(10):
                        listed=operation('verify_absence','ps','--all','--no-trunc','--filter','id='+cid,'--format','{{.ID}}')
                        if listed=='':
                            row['removed']=True;record['removal_proof']='successful_empty_exact_id_query'
                            break
                        require(listed in (cid,cid+'\n'),'unexpected exact owned ID query response')
                        require(attempt<9,'exact owned ID still present after bounded queries')
                        remaining=min(deadline,self.deadline)-time.monotonic()
                        require(remaining>0,'absence proof deadline exhausted')
                        time.sleep(min(1,remaining))
                except BaseException as verification_error:
                    errors.append({'operation':'verify_absence',**safe_error(verification_error)})
        except BaseException as error:errors.append({'operation':'ownership',**safe_error(error)})
        finally:
            accounting_ok=row['cpu_closed'] if row['start_attempted'] or row['registered'] else row['created']
            record['accounting_status']=('final_usage_verified' if row['cpu_closed'] else
                ('not_started' if accounting_ok else 'final_usage_unverified'))
            record['gate_result']=self.gate_result(name)
            record['cleanup_seconds']=time.monotonic()-started
            record['cleanup_overrun_seconds']=max(0,time.monotonic()-deadline)
            record['host_deadline_overrun']=record['host_deadline_overrun'] or time.monotonic()>=self.deadline
            row['cleanup_ok']=row['removed'] and accounting_ok and not self.ledger.failed and time.monotonic()<=deadline and not record['host_deadline_overrun']
            record['status']='removed' if row['removed'] else 'removal_unconfirmed'
            record['cleanup_ok']=row['cleanup_ok']
            # Receipt I/O must not discard the operation outcome or suppress stopping.
            record['receipt_errors']=[]
            try:atomic(self.root/'cpu-receipt.json',self.ledger.receipt())
            except BaseException as error:
                row['cleanup_ok']=record['cleanup_ok']=False
                record['receipt_errors'].append({'operation':'cpu_receipt',**safe_error(error)})
            try:atomic(self.root/'cleanup-receipt.json',{'containers':self.cleanup,
                'all_removed':all(r['removed'] for r in self.names.values() if r['create_attempted']),
                'all_cleanup_verified':all(r['cleanup_ok'] for r in self.names.values() if r['create_attempted'])})
            except BaseException as error:
                row['cleanup_ok']=record['cleanup_ok']=False
                record['receipt_errors'].append({'operation':'cleanup_receipt',**safe_error(error)})
        require(row['cleanup_ok'],'cleanup unconfirmed or final cumulative CPU accounting invalid')
        return ''

    def finish(self):
        failed=False
        for name,row in reversed(list(self.names.items())):
            if row['create_attempted'] and not row['cleanup_started']:
                try:self.remove(name)
                except BaseException:failed=True
            if row['create_attempted'] and not row['cleanup_ok']:failed=True
        require(not failed,'cleanup unconfirmed or accounting invalid')
        require(not self.ledger.failed and self.ledger.receipt()['all_final_usage_verified'],'final cumulative CPU budget/accounting invalid')


def gated_arguments(args,name,docker,gate_script):
    require(args[0]=='create' and args.count('--name')==1,'fixed container creation')
    gate=docker.gate(name);index=args.index('docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a')
    return args[:index]+['--mount',f'type=bind,src={gate},dst=/gate']+args[index:index+1]+[
        'python','-I','-B',gate_script,*args[index+1:]]


def recognition_stages(prepare,freeze_policy,decode,*,setup_seconds=0,clock=time.monotonic,record=None):
    """Fixed three-prepare -> complete-policy freeze -> three-fresh-decode order."""
    spent={family:0.0 for family in FAMILIES};record={} if record is None else record
    record.update(engine_wall_seconds=spent,combined_setup_seconds=setup_seconds)
    for family in FAMILIES:
        remaining=min(600-spent[family],1800-setup_seconds)
        require(remaining>0,'remaining family/setup load budget')
        before=clock()
        try:prepare(family,remaining)
        finally:
            used=clock()-before;spent[family]+=used;setup_seconds+=used
            record['combined_setup_seconds']=setup_seconds
        require(used<=remaining,'prepare-only wall threshold')
    freeze_policy()  # Must persist all3 configs/policy and pass size proof first.
    for family in FAMILIES:
        remaining=600-spent[family];require(remaining>0,'remaining family decode budget')
        before=clock()
        try:decode(family,remaining)
        finally:spent[family]+=clock()-before
        require(spent[family]<=600,'combined family prepare+decode600s')
    return record


# Resource/status snapshots only. Native attempt claims use the stricter new write_once.
def atomic(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    raw = canonical(value); require(len(raw) <= LIMITS['output_bytes'], 'JSON output bound')
    with tmp.open('wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


_HOST_DIAGNOSTIC_CODES = _host_diagnostic_codes(sys._getframe().f_code)
