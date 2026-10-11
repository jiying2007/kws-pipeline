"""Disabled prospective Qwen12×3 hosted host; no old result reuse or gate edits."""
from pathlib import Path
import importlib.util
import json
import os
import re
import random
import tempfile
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from campaign_contract import (BOUNDARY, EXPERIMENT, FAMILIES, LIMITS, MIB, GIB, atomic, canonical,
    denominator,digest,fetch_blind,fresh_run,identity,read_json,regular,require,validate_blind,write_once)
from cpu_budget import CpuLedger,parse_cpu_stat
PREPARED_METADATA=('prepared.json','optional-tool-probes.json','kernel-scope.json','cpu-process-guard.json','blind-job.json','run-identity.json',
    'native-source-lock.json','native-asset-lock.json','native-config.json','native-resolved-generation-config.json',
    'native-frontend-config.json','model-load.json','load-receipt.json','worker-complete.json','progress.json')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def tree_bytes(root):
    total=0
    for p in Path(root).rglob('*'):
        # Count symlink inode bytes without following them (fresh venv uses symlinks).
        if p.is_symlink():total+=p.lstat().st_size
        elif p.is_file():total+=p.stat().st_size
    return total


def safe_error(error):
    """Fixed public metadata only; never exception text, argv, output or paths."""
    kinds=(subprocess.TimeoutExpired,subprocess.CalledProcessError,TimeoutError,
        ValueError,TypeError,KeyError,OSError,RuntimeError,KeyboardInterrupt,SystemExit)
    kind=next((t.__name__ for t in kinds if isinstance(error,t)),'OtherError')
    value={'error_type':kind}
    if isinstance(error,subprocess.CalledProcessError) and type(error.returncode) is int and -255<=error.returncode<=255:
        value['returncode']=error.returncode
    return value


HOST_FAILURE_STAGES=('setup','asr_setup','whisper_setup','prepare','policy_freeze',
    'package_budget_preflight','prepared_retention','decode','unknown')


def host_failure_diagnostic(error,stage):
    """Only trusted size-error fields; unknown failures disclose no payload."""
    value={'schema':'qwen12-host-failure-v1',
        'stage':stage if type(stage) is str and stage in HOST_FAILURE_STAGES else 'unknown',
        'reason_code':'unknown_failure','compressed_bytes':None,'expanded_bytes':None,'member_count':None}
    try:
        from log_transport import PackageLimitError
        if type(error) is PackageLimitError:
            fields=BaseException.__getattribute__(error,'__dict__')
            reason=fields.get('reason_code')
            if type(reason) is str and reason in ('compressed_cap','expanded_cap','member_cap','member_count_cap'):
                value['reason_code']=reason
                for name in ('compressed_bytes','expanded_bytes','member_count'):
                    number=fields.get(name)
                    if type(number) is int and 0<=number<=2**63-1:value[name]=number
    except BaseException:
        # This optional projection must never replace the primary failure.
        value.update(reason_code='diagnostic_unavailable',compressed_bytes=None,expanded_bytes=None,member_count=None)
    return value


class MeteredDocker:
    """Immutable owned IDs; one bounded cleanup and one final CPU close per scope."""
    def __init__(self,root,deadline):
        self.root=Path(root);self.gates=self.root/'gates';self.gates.mkdir()
        self.deadline=deadline;self.stage='setup';self.names={};self.ledger=CpuLedger();self.cleanup=[]
        self.supervisions=[];self.supervision_failure=None

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

    def check(self):
        require(time.monotonic()<self.deadline,'host wall ceiling')
        require(tree_bytes(self.root)<=LIMITS['workspace_bytes'],'task10GiB workspace stop threshold')
        public_bytes=sum(tree_bytes(self.root/n) for n in ('output','frozen','whisper-setup','policy','asr-setup'))
        public_bytes+=sum(p.stat().st_size for p in self.root.glob('*.json') if p.is_file() and not p.is_symlink())
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
                remaining=operation_deadline-time.monotonic();require(remaining>0,'cleanup deadline exhausted')
                allowance=min(timeout,5,remaining);item['timeout_seconds']=allowance
                item['status']='started';value=self.raw(*args,timeout=allowance)
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
                    listed=operation('verify_absence','ps','--all','--no-trunc','--filter','id='+cid,'--format','{{.ID}}')
                    require(listed=='','exact owned ID absence unconfirmed')
                    row['removed']=True;record['removal_proof']='successful_empty_exact_id_query'
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


def stage(source_files,destination):
    destination.mkdir()
    for name,row in source_files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'source roster path')
        path=ROOT/name;require(identity(path)==row,'frozen source identity')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)


def supervise(original,docker,code,runtime,inputs,output,family,deadline,mode=None,frozen=None,policy=None):
    stage_name='setup' if family=='setup' else ('sensevoice' if family=='whisper' else family)
    name=EXPERIMENT+'-'+docker.root.name.split('-')[-1]+'-'+family+('' if mode is None else '-'+mode)
    args=original.container_args(name,read_json(ROOT/'research/source-screen32-v1/container-lock.json')['image'],
        code,runtime,output,'asr',stage_name,inputs)
    image_index=args.index(read_json(ROOT/'research/source-screen32-v1/container-lock.json')['image'])
    command=(['python','-I','-B','/code/research/'+EXPERIMENT+'/setup_adapter.py','--profile','asr','--stage','setup','--runtime','/runtime']
        if family=='setup' else ['/runtime/venv/bin/python','-I','-B','/code/research/'+EXPERIMENT+'/native_worker.py',family,mode])
    extras=[]
    if mode=='decode':
        require(frozen is not None and policy is not None,'frozen ownconfig and wholepolicy mounts')
        extras=['--mount',f'type=bind,src={frozen},dst=/frozen,readonly',
            '--mount',f'type=bind,src={policy},dst=/policy,readonly',
            '--env','KWS_POLICY_SHA256='+identity(policy/'policy.json')['sha256']]
    args=args[:image_index]+extras+['--env','KWS_EXECUTED_HEAD_SHA='+os.environ['GITHUB_SHA'],
        '--env','KWS_RUN_ID='+os.environ['GITHUB_RUN_ID']]+args[image_index:image_index+1]+command
    args=gated_arguments(args,name,docker,'/code/research/'+EXPERIMENT+'/cpu_gate.py')
    began=time.monotonic();phase=None;phase_start=began;active='create';phase_began=began
    primary=cleanup_error=retention_error=None
    record={'name':name,'family':family,'mode':mode,'status':'started','phase_wall_seconds':{},
        'primary_error':None,'cleanup_error':None}
    docker.supervisions.append(record);docker.supervision_failure=None
    def next_phase(value):
        nonlocal active,phase_began
        now=time.monotonic();record['phase_wall_seconds'][active]=now-phase_began
        active=value;phase_began=now
    try:
        docker(*args);next_phase('inspect')
        inspected=json.loads(docker('inspect',name))[0];next_phase('validate')
        # Validate old scope plus the one separately checked gate mount/command.
        mounts=inspected['Mounts'];gate=[m for m in mounts if m['Destination']=='/gate']
        require(len(gate)==1 and gate[0]['Source']==str(docker.names[name]['gate']) and gate[0]['RW'] is True,'exact host gate bind')
        extra_expected={'/gate':(str(docker.names[name]['gate']),True)}
        if mode=='decode':extra_expected.update({'/frozen':(str(frozen),False),'/policy':(str(policy),False)})
        for target,(source,writable) in extra_expected.items():
            observed=[m for m in mounts if m['Destination']==target]
            require(len(observed)==1 and observed[0]['Source']==source and observed[0]['RW'] is writable,'exact frozen/policy gate mount')
        projected={**inspected,'Mounts':[m for m in mounts if m['Destination'] not in extra_expected]}
        original.validate_container_inspect(projected,stage_name)
        require(inspected['Config']['Cmd']==args[args.index(inspected['Config']['Image'])+1:] and
                inspected['Config']['Entrypoint'] in (None,[]),'exact gated child command')
        next_phase('start');docker('start',name);next_phase('wait')
        while not docker.completed(name):
            now=time.monotonic();require(now<deadline,'phase wall ceiling')
            require(tree_bytes(output)<=LIMITS['output_bytes'],'output32MiB stop threshold')
            if family!='setup' and (output/'progress.json').exists():
                p=read_json(output/'progress.json',4096)
                require(set(p)=={'stage','index','started_monotonic'} and p['stage'] in ('load','cell') and
                    type(p['index']) is int and 0<=p['index']<=12,'fixed progress')
                current=(p['stage'],p['index'])
                if current!=phase:
                    require(phase is None or current[1]>phase[1],'no progress restart')
                    phase=current;phase_start=now
                require(phase[0]!='cell' or now-phase_start<120,'outer perclip120s stop')
            time.sleep(.2)
    except BaseException as error:
        primary=error;record['primary_error']={'phase':active,**safe_error(error)}
    finally:
        next_phase('cleanup')
        try:
            if name in docker.names and docker.names[name]['create_attempted']:docker.remove(name)
        except BaseException as error:
            cleanup_error=error;record['cleanup_error']=safe_error(error)
        record['phase_wall_seconds']['cleanup']=time.monotonic()-phase_began
        record['wall_seconds']=time.monotonic()-began
        record['gate_result']=docker.gate_result(name)
        record['status']='failed' if primary is not None or cleanup_error is not None else 'complete'
        try:atomic(docker.root/'supervision-receipt.json',{'schema':'qwen12-supervision-v1','stages':docker.supervisions})
        except BaseException as error:
            retention_error=error;record.update(status='failed',retention_error=safe_error(error))
    # Preserve the original supervision exception, independent of cleanup outcome.
    error=primary if primary is not None else cleanup_error if cleanup_error is not None else retention_error
    if error is not None:
        docker.supervision_failure=(error,record)
        raise error.with_traceback(error.__traceback__)
    return record['wall_seconds']


def whisper_setup(bounded,docker,code,runtime,output,root,run_id,deadline):
    """Reuse the exact reviewed asset transport/phase code with gated supervision."""
    original_supervisor=bounded.Supervisor
    class NewScopeSupervisor(original_supervisor):
        def create(self,role,stage,mounts,command,file_limit):
            name=f'kws-bound-{run_id}-{stage}-{role}'
            gate=docker.gate(name)
            return super().create(role,stage,[*mounts,bounded.mount(gate,'/gate',True)],
                ['python','-I','-B','/opt/cpu_gate.py',*command],file_limit)
        def state(self,name):return not docker.completed(name)
    bounded.Supervisor=NewScopeSupervisor
    try:
        return bounded.acquire_runtime(docker=docker,setup_code=code,runtime=runtime,output=output,root=root,
            job_deadline=deadline,run_id=run_id)
    finally:bounded.Supervisor=original_supervisor


def asr_setup_projection(runtime, destination):
    path=Path(runtime)/'setup-receipt.json'
    if not path.exists():return
    raw=read_json(path,32*MIB)
    value={'schema':'qwen12-asr-setup-public-projection-v1','source_receipt_sha256':identity(path)['sha256'],
        'status':'complete' if 'verification' in raw and 'exception' not in raw else 'failed_or_incomplete',
        'model_calls':0,'retry_count':0}
    for key in ('bootstrap_pip','locks','wheels','builds','disk_checkpoints','verification','import_preflight',
                'actual_download_bytes','wall_seconds'):
        if key in raw:value[key]=raw[key]
    value['commands']=[{k:r[k] for k in ('stage','package','status','returncode','wall_seconds','stderr_diagnostic','exception_type') if k in r}
        for r in raw.get('commands',[])]
    value['downloads']=[{k:r[k] for k in ('filename','bytes','sha256') if k in r} for r in raw.get('downloads',[])]
    if 'exception' in raw:value['error_type']=raw['exception'].get('type')
    value['pre_model_process_compatibility'] = {}
    for name in ('import-process-guard.json', 'import-optional-tool-policy.json',
                 'import-preflight-failure.json'):
        extra = Path(runtime) / name
        if extra.exists():
            value['pre_model_process_compatibility'][name] = {
                **identity(extra), 'record': read_json(extra, 65536)}
    atomic(destination,value)


def collect(root):
    """Explicit public raw/source/native dependency closure, never audio or weights."""
    allowed=['host-receipt.json','cpu-receipt.json','cleanup-receipt.json','supervision-receipt.json','terminal-outcomes.json','acquisition-receipt.json','image-transfer.json','asr-setup-public.json','policy/policy.json','policy/policy-freeze.json','package-budget-preflight.json']
    shared={'prepared.json','run-identity.json','kernel-scope.json','cpu-process-guard.json','blind-job.json','progress.json','contract.json','decoder-inputs.json',
        'model-load.json','native-source-lock.json','native-asset-lock.json','native-config.json','native-resolved-generation-config.json',
        'native-frontend-config.json','optional-tool-probes.json','load-receipt.json','worker-complete.json','worker-failure.json'}
    clip={'started.json','completed.json','raw-output.json','raw-native-output.json','raw-transcript.json','raw-native-row.json','input-capture.json','pre-finalize.json'}
    for family in FAMILIES:
        allowed += ['output/'+family+'/'+n for n in sorted(shared)]
        for i in range(1,13):
            oid=f'clip-{i:06d}';allowed+=['output/'+family+'/'+oid+'/'+n for n in sorted(clip)]
            allowed+=['output/'+family+'/'+oid+suffix for suffix in ('.native-observation.json',
                '.native-generation-return.json','.native-tokenizer-input.json','.native-inference-return.json','.native-returned-raw.json')]
            allowed+=['output/'+family+'/'+oid+'/update-'+str(j).zfill(3)+'.json' for j in range(1,449)]
    allowed+=['frozen/'+family+'/'+name for family in FAMILIES for name in PREPARED_METADATA]
    setup_names=('runtime-lock.json','source-lock.json','asset-lock.json','installed-inventory.json',
        'download-receipt.json','setup-receipt.json','phased-setup-failure.json')
    allowed+=['whisper-setup/'+n for n in setup_names]
    allowed+=['whisper-setup/phase-'+p+'-'+suffix+'.json' for p in ('install','source','model','finalize')
        for suffix in ('progress','complete')]
    # This imported, reviewed helper exposes a fixed positive metadata roster.
    bound=sys.modules.get('qwen12_bounded_host')
    if bound is not None:allowed+=['whisper-setup/bounded-acquisition/'+n for n in bound.public_metadata_names()]
    source_files={}
    for storage,family in ((storage,family) for storage in ('output','frozen') for family in FAMILIES):
        prefix='output/'+family+'/'
        for name,field in (('native-source-lock.json','files'),('native-asset-lock.json','assets')):
            path=root/storage/family/name
            if not path.exists():continue
            lock=read_json(path,8*MIB);rows=list(lock[field])
            if field=='files':rows+=lock.get('empty_source_files',[])
            for row in rows:
                if field=='assets' and row.get('body_retained') is not True:continue
                member=row['path']
                require(type(member) is str and member.startswith(prefix) and '..' not in Path(member).parts and
                    not Path(member).is_absolute(),'same-family public dependency path')
                if field=='files':require((member.startswith(prefix+'retained-source/') and member.endswith('.py')) or
                    (family=='sensevoice' and member==prefix+'retained-source/optional-tool-policy.json' and
                     row.get('role')=='runtime-execution-policy'),'public source body or exact optional policy only')
                else:require(member.startswith(prefix+'retained-assets/') or member==prefix+'native-frontend-config.json','bounded native asset path')
                require(not member.endswith(('.pt','.safetensors','.wav','.whl','.tar.gz')),'no weights/audio/package bodies')
                physical=member if storage=='output' else 'frozen/'+member[len('output/'):]
                require(identity(root/physical)=={'bytes':row['bytes'],'sha256':row['sha256']},'retained source/dependency exact bytes')
                source_files[physical]=row
    allowed+=sorted(source_files)
    result=[]
    for name in sorted(set(allowed)):
        p=root/name
        if p.exists():
            if name not in source_files:read_json(p,8*MIB)
            result.append(name)
    require(sum(identity(root/n)['bytes'] for n in result)<=LIMITS['output_bytes'],'public JSON total bound')
    return sorted(result)


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


def package_budget_preflight(root,transport):
    try:return _package_budget_preflight(root,transport)
    except BaseException as error:
        try:
            write_once(root/'package-budget-preflight.json',{
                'schema':'qwen12-predecode-package-budget-v1',
                'status':'rejected_before_any_audio_forward','new_native_calls':0,
                'diagnostic':host_failure_diagnostic(error,'package_budget_preflight')})
        except BaseException:pass  # Receipt failure cannot mask the original error.
        raise


def _package_budget_preflight(root,transport):
    """Actual complete prepared closure plus bounded synthetic worst-prefix traces.

    This tests retained source/inventory bytes without any recognizer forward.
    Other records reserve32768 CJK characters per transcript and1MiB extra JSON.
    It is a prepared stress fixture; actual completed evidence is checked again,
    and no universal worst-case compression claim is made.
    """
    from log_transport import build_package
    actual=collect(root);rng=random.Random(20261011)
    with tempfile.TemporaryDirectory(prefix='qwen12-size-',dir=root) as temporary:
        trial=Path(temporary)
        for name in actual:
            dst=trial/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dst)
        names=list(actual)
        # Reserve exact separate prepare-only metadata and dependencies, including
        # a fresh-load failure before the new lock/source copy can be emitted.
        for family in FAMILIES:
            prefix='output/'+family+'/'
            prepared=[n for n in actual if n.startswith(prefix) and
                (n[len(prefix):] in PREPARED_METADATA or '/retained-source/' in n or '/retained-assets/' in n)]
            for old_member in prepared:
                member='frozen/'+old_member[len('output/'):];dst=trial/member
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/old_member,dst);names.append(member)
        for family in FAMILIES:
            config=read_json(root/'output'/family/'native-config.json')
            for i in range(1,13):
                prefix=f'output/{family}/clip-{i:06d}/'
                if family=='whisper':
                    ids=[50258,50260,50359,50363]
                    for step in range(1,225):
                        ids.append(rng.randrange(1,50256));name=prefix+f'update-{step:03d}.json'
                        path=trial/name;path.parent.mkdir(parents=True,exist_ok=True)
                        write_once(path,{'step':step,'token_ids':ids,'completed':step==224});names.append(name)
                for suffix in ('started.json','completed.json','raw-native-output.json','raw-transcript.json'):
                    name=prefix+suffix;path=trial/name;path.parent.mkdir(parents=True,exist_ok=True)
                    value={'schema':'synthetic-size-only-not-ASR-evidence','config':config,'token_ids':[rng.randrange(1,151000) for _ in range(256)]}
                    if suffix in ('raw-native-output.json','raw-transcript.json'):
                        value['raw_text']=''.join(chr(0x4e00+rng.randrange(1024)) for _ in range(32768))
                    write_once(path,value);names.append(name)
        name='size-reserve.json';write_once(trial/name,{'schema':'synthetic-size-reserve','padding':'x'*(1024**2)});names.append(name)
        payload,manifest,lines=build_package(trial,sorted(names),transport)
        result={'schema':'qwen12-predecode-package-budget-v1','status':'passed_before_any_audio_forward',
            'actual_prepared_members':len(actual),'worst_fixture_members':len(names),'compressed_bytes':len(payload),
            'expanded_member_bytes':sum(v['bytes'] for v in manifest['members'].values()),
            'log_text_bytes':sum(len(x.encode())+1 for x in lines),'whisper_full_prefix_steps_per_clip':224,
            'transcript_reserve_characters_each_copy':32768,'transcript_copies_per_clip':2,
            'extra_json_reserve_bytes':1024**2,'new_native_calls':0}
        write_once(root/'package-budget-preflight.json',result)
    return result


def execute(env,event,release,freeze):
    run=fresh_run(env,event,release);started=time.monotonic()
    def interrupted(signum,frame):raise TimeoutError('host interrupted or3600s wall ceiling')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,3600)
    old=ROOT/'research/source-screen32-v1';sys.path.insert(0,str(old))
    # Original module is imported for kernel/host/setup constructors only.
    original=load('qwen12_previous_host',old/'hosted_run.py')
    root=Path(env['RUNNER_TEMP'])/(EXPERIMENT+'-'+run['run_id']);root.mkdir(mode=0o700)
    docker=MeteredDocker(root,started+3600)
    receipt={'schema':'qwen12-host-receipt-v1',**run,**BOUNDARY,'status':'started','limits':LIMITS,
        'stages':{s:'not_run' for s in ('asr_setup','qwen06','sensevoice','whisper_setup','whisper')},
        'source_freeze_sha256':identity(HERE/'execution-freeze.json')['sha256'],'storage':'no_artifact_no_cache',
        'workspace_budget_scope':'sampled logical task tree only; excludes Docker image/layers/tmpfs; not a filesystem hard quota',
        'wall_budget_scope':'host supervised thresholds; cleanup and kernel scheduling may overshoot'}
    atomic(root/'host-receipt.json',receipt);inputs=root/'input';output=root/'output';output.mkdir()
    job=None;failed=False;setup_elapsed=0;host_stage='setup'
    try:
        require(shutil.disk_usage(root).free>=12*GIB,'initial12GiB free reserve')
        receipt['capacity']=original.host_capacity();atomic(root/'host-receipt.json',receipt)
        original.pull_image(read_json(old/'container-lock.json'),root/'image-transfer.json');docker.check()
        inputs_receipt=fetch_blind(read_json(HERE/'public-inputs.json'),inputs,urllib.request.build_opener(urllib.request.ProxyHandler({})))
        job=validate_blind(inputs)
        code=root/'code';setup_code=root/'setup-code';stage(freeze['inference_code'],code);stage(freeze['setup_code'],setup_code)
        # New gate is a separately frozen source, not a historical file edit.
        shutil.copyfile(HERE/'cpu_gate.py',setup_code/'research/source-screen32-whisper-v1/attempt-3/cpu_gate.py')
        runtime=root/'asr-runtime';runtime.mkdir();stage_output=root/'asr-setup';stage_output.mkdir()
        host_stage='asr_setup';receipt['stages']['asr_setup']='started';atomic(root/'host-receipt.json',receipt)
        setup_began=time.monotonic()
        try:supervise(original,docker,setup_code,runtime,inputs,stage_output,'setup',min(docker.deadline,time.monotonic()+1800))
        finally:
            setup_elapsed+=time.monotonic()-setup_began;receipt['combined_setup_seconds']=setup_elapsed
        asr_setup_projection(runtime,root/'asr-setup-public.json')
        setup_receipt=read_json(runtime/'setup-receipt.json',32*MIB)
        require(type(setup_receipt['actual_download_bytes']) is int and setup_receipt['actual_download_bytes']<=4*GIB,'ASR acquisition cap')
        acquired={'schema':'qwen12-reserved-acquisition-v1','qwen_sense_body_bytes':setup_receipt['actual_download_bytes'],
            **inputs_receipt,'reserved_total_bytes':LIMITS['acquisition_reserved_bytes'],
            'cap_bytes':6*GIB,'hard_wire_byte_cap':False,
            'byte_scopes':['ASR exact response bodies','Whisper bidirectional tunnel recv','image observed interface reservation','audio exact bodies']}
        atomic(root/'acquisition-receipt.json',acquired);receipt['stages']['asr_setup']='complete'
        asr_runtime=runtime
        runtime=root/'whisper-runtime';runtime.mkdir();out=root/'whisper-setup';out.mkdir();docker.stage='setup'
        remaining=1800-setup_elapsed;require(remaining>0,'combined setup1800s remaining')
        host_stage='whisper_setup';receipt['stages']['whisper_setup']='started';atomic(root/'host-receipt.json',receipt)
        third=ROOT/'research/source-screen32-whisper-v1/attempt-3';sys.path.insert(0,str(third))
        bounded=load('qwen12_bounded_host',third/'bounded_host.py')
        setup_began=time.monotonic()
        try:acquired['whisper']=whisper_setup(bounded,docker,setup_code,runtime,out,root,run['run_id'],min(docker.deadline,time.monotonic()+remaining))
        finally:
            setup_elapsed+=time.monotonic()-setup_began;receipt['combined_setup_seconds']=setup_elapsed
        require(setup_elapsed<=1800,'combined setup1800s total')
        atomic(root/'acquisition-receipt.json',acquired);receipt['stages']['whisper_setup']='complete'
        runtimes={'qwen06':asr_runtime,'sensevoice':asr_runtime,'whisper':runtime}
        policy_dir=root/'policy';policy_dir.mkdir();frozen_root=root/'frozen';frozen_root.mkdir()
        transport={'repository':'jiying2007/kws-pipeline','run_id':int(run['run_id']),'run_attempt':1,
            'job_key':env['GITHUB_JOB'],'executed_head_sha':run['head_sha']}
        def prepare(family,seconds):
            nonlocal host_stage
            host_stage='prepare'
            docker.stage='setup';out=output/family;out.mkdir();receipt['stages'][family+'_prepare']='started';atomic(root/'host-receipt.json',receipt)
            supervise(original,docker,code,runtimes[family],inputs,out,family,min(docker.deadline,time.monotonic()+seconds),mode='prepare')
            require(read_json(out/'worker-complete.json')['calls']==0 and read_json(out/'prepared.json')['audio_forward_calls']==0,'pureload preparation only')
            require(not any(out.glob('clip-*/started.json')),'no consumed attempt during preparation')
            receipt['stages'][family+'_prepare']='complete'
        def freeze_policy():
            nonlocal host_stage
            host_stage='policy_freeze'
            from worker_receipts import build_policy
            policy=build_policy({family:output/family for family in FAMILIES})
            write_once(policy_dir/'policy.json',policy)
            receipt['recognizer_policy_sha256']=identity(policy_dir/'policy.json')['sha256']
            write_once(policy_dir/'policy-freeze.json',{'schema':'qwen12-all3-pre-recognition-freeze-v1',
                'recognizer_policy_sha256':receipt['recognizer_policy_sha256'],'native_calls_started':0,
                'prepared_config_files':{family:identity(output/family/'native-config.json') for family in FAMILIES},
                'frozen_monotonic':time.monotonic(),'run_id':run['run_id'],'executed_head_sha':run['head_sha']})
            host_stage='package_budget_preflight'
            receipt['package_budget']=package_budget_preflight(root,transport)
            host_stage='prepared_retention'
            for family in FAMILIES:(output/family).rename(frozen_root/family)
            receipt['all3_policy_frozen_before_recognition']=True;atomic(root/'host-receipt.json',receipt)
        def decode(family,seconds):
            nonlocal host_stage
            host_stage='decode'
            docker.stage=family;out=output/family;out.mkdir();receipt['stages'][family]='started';atomic(root/'host-receipt.json',receipt)
            supervise(original,docker,code,runtimes[family],inputs,out,family,min(docker.deadline,time.monotonic()+seconds),
                mode='decode',frozen=frozen_root/family,policy=policy_dir)
            require(read_json(out/'worker-complete.json')['calls']==12,'exact12 single native calls')
            receipt['stages'][family]='complete'
        recognition_stages(prepare,freeze_policy,decode,setup_seconds=setup_elapsed,record=receipt)
        setup_elapsed=receipt['combined_setup_seconds'];receipt['status']='all36_native_calls_complete'
    except BaseException as error:
        failed=True;receipt.update(status='stopped_no_retry',primary_error=safe_error(error),**safe_error(error))
        try:receipt['diagnostic']=host_failure_diagnostic(error,host_stage)
        except BaseException:
            receipt['diagnostic']={'schema':'qwen12-host-failure-v1','stage':host_stage,
                'reason_code':'diagnostic_unavailable'}
        if docker.supervision_failure is not None and docker.supervision_failure[0] is error:
            observed=docker.supervision_failure[1]
            receipt.update(primary_error=observed['primary_error'],cleanup_error=observed['cleanup_error'])
        for k,v in receipt['stages'].items():
            if v=='started':receipt['stages'][k]='error'
    finally:
        if (root/'asr-runtime').exists():
            try:asr_setup_projection(root/'asr-runtime',root/'asr-setup-public.json')
            except BaseException:receipt['asr_setup_projection']='invalid_not_published'
        try:docker.finish()
        except BaseException as error:
            failed=True;receipt.update(status='cleanup_unconfirmed',finish_error=safe_error(error))
            if receipt.get('cleanup_error') is None:receipt['cleanup_error']=safe_error(error)
        # Preserve prepared-only metadata/source for never-run families without inventing attempts.
        for family in FAMILIES:
            frozen=root/'frozen'/family
            if frozen.exists() and not (output/family).exists():shutil.copytree(frozen,output/family)
        setup_elapsed=receipt.get('combined_setup_seconds',setup_elapsed)
        if job is None:
            from campaign_contract import blind_job,validate_public
            job=blind_job(validate_public(read_json(HERE/'public-inputs.json')))
        write_once(root/'terminal-outcomes.json',denominator(job,output))
        receipt.update(wall_seconds=time.monotonic()-started,combined_setup_seconds=setup_elapsed,
            supervision=docker.supervisions,cleanup=docker.cleanup)
        atomic(root/'host-receipt.json',receipt)
        if release.get('new_transcripts_publication_approved') is True:
            from log_transport import build_package
            transport={'repository':'jiying2007/kws-pipeline','run_id':int(run['run_id']),'run_attempt':1,
                'job_key':env['GITHUB_JOB'],'executed_head_sha':run['head_sha']}
            from branch_transport import write_package
            payload,_,_=build_package(root,collect(root),transport)
            # The next trusted host step alone receives the publication token.
            # An error/partial native run still retains its complete bounded package.
            write_package(env['RUNNER_TEMP'],payload,transport,'native')
            print(json.dumps({'schema':'qwen12-local-evidence-package-ready-v1',
                'archive_bytes':len(payload),'archive_sha256':digest(payload),
                'publication':'pending_separate_host_step','native_run_failed':failed},sort_keys=True),flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)
    return int(failed)


def main():
    release=read_json(HERE/'execution-release.json');event=read_json(Path(os.environ['GITHUB_EVENT_PATH']))
    fresh_run(os.environ,event,release) # Before Docker, network, runtime imports or setup.
    require(release.get('source_freeze_sha256')==identity(HERE/'execution-freeze.json')['sha256'],'reviewed release/source freeze binding')
    freeze=read_json(HERE/'execution-freeze.json')
    for name,row in freeze['host_code'].items():require(identity(ROOT/name)==row,'host source freeze')
    return execute(dict(os.environ),event,release,freeze)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'blocked_no_retry','error_type':type(error).__name__}));raise SystemExit(2)
