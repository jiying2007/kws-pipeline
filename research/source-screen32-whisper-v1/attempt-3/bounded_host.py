"""Dormant fail-closed host contracts and serial setup supervisor.

No activation entry point. Docker is injected only by the separately admitted
host. Missing broker receipts charge reserved allowances, not invented traffic.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import stat
import time

MIB = 1024**2
CAP = 1536 * MIB
IMAGE = 'docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a'
PROFILES = {
    'broker': {'memory': 256*MIB, 'cpus': 1, 'pids': 16, 'network': 'bridge'},
    'worker': {'memory': 12*1024**3-256*MIB, 'cpus': 3, 'pids': 240, 'network': 'none'},
    'offline': {'memory': 12*1024**3, 'cpus': 4, 'pids': 256, 'network': 'none'},
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def regular(path, maximum=None):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'no path aliases')
    s = path.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'single regular file')
    require(maximum is None or s.st_size <= maximum, 'file bound')
    return s


def write_once(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def read_json(path):
    regular(path, MIB)
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def secure_directory(path, uid=None, gid=None):
    path = Path(path); uid = os.getuid() if uid is None else uid; gid = os.getgid() if gid is None else gid
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'directory alias')
    path.mkdir(mode=0o700)
    s = path.stat()
    require(stat.S_IMODE(s.st_mode) == 0o700 and s.st_uid == uid and s.st_gid == gid, 'private owned directory')
    return path


def check_socket(path, uid, gid):
    s = Path(path).lstat()
    require(stat.S_ISSOCK(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o600 and
            s.st_uid == uid and s.st_gid == gid, 'private owned Unix socket')


def mount(source, destination, writable=False):
    source = Path(source)
    require(source.is_absolute() and ',' not in str(source) and '\n' not in str(source) and
            not any(p.is_symlink() for p in (source, *source.parents)), 'fixed bind source')
    require(re.fullmatch(r'/[A-Za-z0-9_./-]+', destination) and '..' not in Path(destination).parts,
            'fixed bind destination')
    return {'source': str(source), 'destination': destination, 'writable': writable}


def container_args(role, name, code, mounts, command, *, uid, gid, file_limit, image=IMAGE):
    require(role in PROFILES and image == IMAGE, 'fixed role and pinned image')
    require(type(uid) is int and uid > 0 and type(gid) is int and gid >= 0, 'nonroot runner identity')
    require(re.fullmatch('[a-z0-9][a-z0-9_.-]{0,100}', name), 'container name')
    require(type(file_limit) is int and 0 < file_limit <= 4*1024**3, 'file cap')
    p = PROFILES[role]; bindings = [mount(code, '/opt'), *mounts]
    require(len({m['destination'] for m in bindings}) == len(bindings), 'unique mounts')
    require(not any(m['destination'].endswith('docker.sock') or m['source'].endswith('docker.sock') for m in bindings), 'no Docker socket')
    require(role != 'offline' or all(m['destination'] != '/run/kws' for m in bindings), 'offline has no broker mount')
    args = ['create', '--name', name, '--platform', 'linux/amd64', '--pull', 'never',
        '--memory', str(p['memory']), '--memory-swap', str(p['memory']), '--cpus', str(p['cpus']),
        '--pids-limit', str(p['pids']), '--cgroupns', 'private', '--ipc', 'private',
        '--shm-size', str((64 if role=='offline' else 32)*MIB),
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', f'{uid}:{gid}', '--network', p['network'],
        '--ulimit', 'core=0:0', '--ulimit', 'nofile=1024:1024', '--ulimit', f'fsize={file_limit}:{file_limit}',
        '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1', '--log-opt', 'compress=false']
    tmp = '/tmp' if role == 'broker' else '/wheel-tmpfs'; size = (16 if role == 'broker' else 512)*MIB
    args += ['--tmpfs', f'{tmp}:rw,noexec,nosuid,nodev,size={size},uid={uid},gid={gid},mode=0700']
    for m in bindings:
        args += ['--mount', 'type=bind,src='+m['source']+',dst='+m['destination']+('' if m['writable'] else ',readonly')]
    args += ['--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONNOUSERSITE=1', '--env', 'HOME=/nonexistent', '--env', 'TMPDIR='+tmp,
             '--env', 'OMP_NUM_THREADS='+str(p['cpus']), '--env', 'MKL_NUM_THREADS='+str(p['cpus'])]
    return args + [image, *command]


def validate_inspect(record, role, code, mounts, *, uid, gid, file_limit, command):
    p = PROFILES[role]; h = record['HostConfig']; c = record['Config']
    require(h.get('ShmSize')==(64 if role=='offline' else 32)*MIB,'explicit aggregate shared memory bound')
    require(h.get('Memory') == p['memory'] and h.get('MemorySwap') == p['memory'] and
        h.get('NanoCpus') == p['cpus']*10**9 and h.get('PidsLimit') == p['pids'], 'Docker resource proof')
    require(h.get('ReadonlyRootfs') is True and h.get('Privileged') is False and h.get('CgroupnsMode') == 'private' and
        h.get('PidMode') == '' and h.get('IpcMode') == 'private' and h.get('CapAdd') in (None, []) and
        h.get('CapDrop') == ['ALL'] and h.get('SecurityOpt') in (['no-new-privileges'], ['no-new-privileges:true']), 'Docker privilege proof')
    require(h.get('NetworkMode') == p['network'] and not h.get('PortBindings') and not h.get('PublishAllPorts') and
        not h.get('Devices') and not h.get('DeviceRequests') and not h.get('VolumesFrom') and not h.get('ExtraHosts') and
        not h.get('Links') and not h.get('GroupAdd'), 'Docker egress and device proof')
    require(c.get('Image') == IMAGE and c.get('User') == f'{uid}:{gid}' and c.get('Cmd') == command and
        c.get('Entrypoint') in (None, []), 'Docker image/command proof')
    require(h.get('LogConfig') == {'Type':'local','Config':{'max-size':'1m','max-file':'1','compress':'false'}}, 'bounded logs')
    expected = {m['destination']:m for m in [mount(code, '/opt'), *mounts]}
    bindings = [m for m in record['Mounts'] if m['Type'] == 'bind']
    require(len(bindings) == len(expected), 'exact bind count')
    for m in bindings:
        e = expected.get(m['Destination'])
        require(e is not None and m['Source'] == e['source'] and m['RW'] is e['writable'] and
                m.get('Propagation') == 'rprivate', 'exact bind boundary')
    require(all(m['Type'] in ('bind','tmpfs') for m in record['Mounts']), 'no unapproved volume')
    require({r['Name']:(r['Soft'],r['Hard']) for r in h.get('Ulimits',[])} ==
            {'core':(0,0),'nofile':(1024,1024),'fsize':(file_limit,file_limit)}, 'exact process limits')
    tmp = '/tmp' if role == 'broker' else '/wheel-tmpfs'; size = (16 if role == 'broker' else 512)*MIB
    require(h.get('Tmpfs') == {tmp:f'rw,noexec,nosuid,nodev,size={size},uid={uid},gid={gid},mode=0700'}, 'hard storage quota')


def validate_kernel_snapshot(s, role):
    p = PROFILES[role]
    require(s['uid'] > 0 and s['cgroup'] == '0::/', 'private nonroot scope')
    require(s['limits'] == {'memory.max':str(p['memory']),'memory.swap.max':'0',
        'cpu.max':f"{p['cpus']*100000} 100000",'pids.max':str(p['pids'])}, 'kernel resource scope')
    v = s['status']
    require(v.get('NoNewPrivs') == '1' and v.get('Seccomp') == '2' and len(v.get('NSpid','').split()) == 1 and
        all(re.fullmatch('0+',v.get(k,'')) for k in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')), 'kernel privilege scope')
    require(s['readonly_root'] and s['private_readonly_cgroup'], 'readonly scope')
    require(s['interfaces'] == (['eth0','lo'] if role == 'broker' else ['lo']), 'kernel network scope')
    return s


def verify_effective_scope(role):
    """Call inside a container before network or third-party code; no side effects."""
    require(role in PROFILES, 'scope role')
    def read(path):
        with Path(path).open(encoding='ascii') as stream: value=stream.read(65537)
        require(len(value)<=65536,'kernel record bound'); return value.strip()
    rows = [line.split() for line in read('/proc/self/mountinfo').splitlines()]
    roots = [r for r in rows if r[4]=='/']; groups = [r for r in rows if r[4]=='/sys/fs/cgroup' and 'cgroup2' in r]
    status = dict(line.split(':',1) for line in read('/proc/self/status').splitlines() if ':' in line)
    return validate_kernel_snapshot({'schema':'bounded-kernel-scope-v1','role':role,'uid':os.geteuid(),
        'cgroup':read('/proc/self/cgroup'),'limits':{k:' '.join(read('/sys/fs/cgroup/'+k).split())
        for k in ('memory.max','memory.swap.max','cpu.max','pids.max')},'status':{k:v.strip() for k,v in status.items()},
        'readonly_root':len(roots)==1 and 'ro' in roots[0][5].split(','),
        'private_readonly_cgroup':len(groups)==1 and groups[0][3]=='/' and 'ro' in groups[0][5].split(','),
        'interfaces':sorted(p.name for p in Path('/sys/class/net').iterdir())},role)


class Ledger:
    """One new attempt; no restore/reset/retry API."""
    def __init__(self,directory):
        self.directory=Path(directory); self.events=[]; self.total=0; self.active=None; self.failed=False
        write_once(self.directory/'budget-initial.json',{'schema':'bounded-host-budget-v1','cap':CAP})

    @property
    def remaining(self):
        require(not self.failed and self.active is None,'ledger unavailable'); return CAP-self.total

    def begin(self,asset_id,kind,expected_size):
        require(re.fullmatch('[a-z0-9_-]{1,64}',asset_id) and not any(e['asset_id']==asset_id for e in self.events),'one immutable attempt')
        require(kind in ('wheel','source','model') and type(expected_size) is int and 0<expected_size<=1024**3,'asset size')
        allowance=min((expected_size*102+99)//100+MIB,self.remaining) if kind=='wheel' else self.remaining
        require(allowance>expected_size,'remaining encrypted budget')
        self.active={'asset_id':asset_id,'kind':kind,'allowance':allowance,'status':'started'}
        write_once(self.directory/f'budget-{len(self.events):03d}-started.json',self.active); return allowance

    def finish(self,observed_bytes,*,success):
        require(self.active is not None and type(success) is bool,'active ledger charge')
        known=type(observed_bytes) is int and 0<=observed_bytes<=self.active['allowance']
        require(known or observed_bytes is None,'broker receipt')
        charge=observed_bytes if known else self.active['allowance']
        event={**self.active,'observed_ciphertext_bytes':observed_bytes,'budget_charge_bytes':charge,
            'charge_kind':'observed' if known else 'conservative_reserved_allowance','status':'verified' if success else 'failed_no_retry'}
        self.total+=charge; self.failed=not success; self.events.append(event); self.active=None
        require(self.total<=CAP,'aggregate cap'); write_once(self.directory/f'budget-{len(self.events)-1:03d}-final.json',event)

    def summary(self,path):
        require(not self.failed and self.active is None,'complete ledger required')
        value={'schema':'bounded-host-acquisition-v1','status':'complete','aggregate_cap':CAP,'total_read_bytes':self.total,
            'byte_scope':'bidirectional_tunnel_socket_recv_before_forwarding','assets':self.events}
        write_once(path,value); return value


def verify_candidate(directory,row):
    directory=Path(directory); require(not directory.is_symlink(),'candidate alias')
    require(re.fullmatch(r'[A-Za-z0-9_.+%-]+\.whl',row['filename']),'locked wheel name')
    require(sorted(p.name for p in directory.iterdir())==[row['filename']],'exact candidate membership')
    candidate=directory/row['filename']
    require(regular(candidate,row['size_bytes']).st_size==row['size_bytes'] and digest(candidate)==row['sha256'],'host independent wheel identity')
    return candidate


def admit_candidate(directory,row,wheelhouse):
    candidate=verify_candidate(directory,row); target=Path(wheelhouse)/row['filename']
    with candidate.open('rb') as source,target.open('xb') as destination:
        shutil.copyfileobj(source,destination,MIB); destination.flush(); os.fsync(destination.fileno())
    require(regular(target).st_size==row['size_bytes'] and digest(target)==row['sha256'],'admitted wheel identity')
    return target


def validate_broker_receipt(value, plan, allowance):
    require(type(value) is dict and set(value)=={'schema','byte_scope','aggregate_cap','total_read_bytes',
        'prior_charged_bytes','charged_bytes','failure_code','plan_complete','assets','status','asset_id','kind',
        'expected_size','asset_cap'},'exact broker receipt fields')
    require(value.get('schema') == 'bounded-connect-ledger-v1' and value.get('status') == 'complete' and
        value.get('failure_code') is None and value.get('plan_complete') is True and
        value.get('asset_id') == plan['asset_id'] and value.get('kind') == plan['kind'] and
        value.get('expected_size') == plan['expected_size'] and value.get('aggregate_cap') == CAP and
        value.get('prior_charged_bytes') == plan['prior_charged_bytes'] and
        value.get('byte_scope') == 'bidirectional_tunnel_socket_recv_before_forwarding', 'broker final identity')
    expected_cap=(plan['expected_size']*102+99)//100+MIB if plan['kind']=='wheel' else CAP
    require(value.get('asset_cap')==expected_cap,'broker immutable asset cap')
    charged=value.get('charged_bytes')
    require(type(charged) is int and 0<charged<=allowance and
        value.get('total_read_bytes') == plan['prior_charged_bytes']+charged <= CAP, 'broker final budget')
    assets=value.get('assets'); require(type(assets) is dict and set(assets)=={plan['asset_id']}, 'broker asset membership')
    asset=assets[plan['asset_id']]
    fields={'upstream_read_bytes','downstream_read_bytes','total_read_bytes','connections_started','complete'}
    require(type(asset) is dict and set(asset) in (fields,fields|{'transport_closed'}),'exact broker asset fields')
    if 'transport_closed' in asset:
        require(type(asset['transport_closed']) is int and 0<=asset['transport_closed']<=asset.get('connections_started',-1),'bounded transport-close count')
    require(type(asset.get('connections_started')) is int and
        0<asset['connections_started']<=(128 if plan['kind']=='source' else 4),'broker connection bound')
    require(asset.get('complete') is True and asset.get('total_read_bytes')==charged and
        all(type(asset.get(k)) is int and asset[k]>=0 for k in ('upstream_read_bytes','downstream_read_bytes')) and
        asset['upstream_read_bytes']+asset['downstream_read_bytes']==charged, 'broker directional budget')
    return charged


class Supervisor:
    def __init__(self, docker, code, control, evidence, deadline, uid, gid, run_id, clock, sleep):
        self.docker=docker; self.code=Path(code); self.control=Path(control); self.evidence=Path(evidence)
        self.deadline=deadline; self.uid=uid; self.gid=gid; self.run_id=run_id; self.clock=clock; self.sleep=sleep
        self.created=[]; self.failed=False; self.cleanup_count=0; self.ledger=Ledger(evidence)

    def remaining(self):
        # Reserve ten seconds inside the same absolute ceiling for pair teardown.
        value=self.deadline-self.clock()-10
        require(value>=1, 'absolute setup/job deadline'); return value

    def call(self,*args):
        return self.docker(*args,timeout=min(30,self.remaining()))

    def create(self,role,stage,mounts,command,file_limit):
        name=f'kws-bound-{self.run_id}-{stage}-{role}'
        args=container_args(role,name,self.code,mounts,command,uid=self.uid,gid=self.gid,file_limit=file_limit)
        # Record the unique attempted name first: an uncertain create can still create.
        self.created.append(name); self.call(*args)
        records=json.loads(self.call('inspect',name)); require(len(records)==1,'single container inspect')
        validate_inspect(records[0],role,self.code,mounts,uid=self.uid,gid=self.gid,file_limit=file_limit,command=command)
        write_once(self.evidence/(stage+'-'+role+'-scope.json'), {'schema':'bounded-container-inspect-v1',
            'role':role,'status':'verified','limits':PROFILES[role],
            'argv_sha256':hashlib.sha256(canonical(args)).hexdigest(),'socket_mounted':any(m['destination']=='/run/kws' for m in mounts)})
        self.call('start',name); return name

    def state(self,name):
        paths=list(self.evidence.parent.rglob('*'))
        require(not any(p.is_symlink() for p in paths),'public output alias')
        require(sum(regular(p).st_size for p in paths if p.is_file())<=64*MIB,'public output bound')
        state=json.loads(self.call('inspect','--format','{{json .State}}',name))
        require(type(state.get('Running')) is bool and type(state.get('ExitCode')) is int and
            type(state.get('OOMKilled')) is bool,'container state')
        if not state['Running']:
            require(state['ExitCode']==0 and not state['OOMKilled'],'container failed; no retry')
        return state['Running']

    def watch(self,name,broker=None):
        while self.state(name):
            if broker is not None:
                require(self.state(broker),'broker exited before worker')
            self.sleep(min(0.2,self.remaining()))

    def cleanup(self):
        records=[]
        for name in reversed(self.created):
            before=self.clock()
            record={'container':name,'status':'removed','deadline_overrun':before>=self.deadline}
            try:
                # Even after an unexpected overrun, make one bounded removal
                # attempt rather than knowingly leave a live network broker.
                self.docker('rm','--force',name,timeout=min(5,max(0.1,self.deadline-before)))
            except BaseException as error:
                record.update(status='removal_unconfirmed',error_type=type(error).__name__)
            record['deadline_overrun']=record['deadline_overrun'] or self.clock()>self.deadline
            records.append(record)
        self.created=[]
        failed=any(r['status']!='removed' or r['deadline_overrun'] for r in records)
        self.failed=self.failed or failed
        write_once(self.evidence/f'bounded-host-cleanup-{self.cleanup_count:03d}.json',
            {'schema':'bounded-host-cleanup-v1','status':'failed' if failed else 'complete','containers':records})
        self.cleanup_count+=1
        require(not failed,'container cleanup failed or deadline exceeded')

    def pair(self,asset_id,kind,expected_size,worker_mounts,worker_command,file_limit,verify,expected_row=None):
        require(not self.created and not self.failed,'previous pair fully removed')
        allowance=self.ledger.begin(asset_id,kind,expected_size); charged=None; successful=False
        directory=secure_directory(self.control/asset_id,self.uid,self.gid)
        socket_dir=secure_directory(directory/'socket',self.uid,self.gid)
        write_once(socket_dir/'deadline.json',{'deadline_monotonic':self.deadline-10})
        evidence=secure_directory(directory/'evidence',self.uid,self.gid)
        worker_evidence=secure_directory(directory/'worker-evidence',self.uid,self.gid) if kind=='wheel' else None
        if worker_evidence is not None:
            worker_mounts=[*worker_mounts,mount(worker_evidence,'/worker-evidence',True)]
        plan={'schema':'bounded-connect-plan-v1','asset_id':asset_id,'kind':kind,'expected_size':expected_size,
            'prior_charged_bytes':self.ledger.total,'deadline_seconds':min(1200,int(self.remaining()))}
        write_once(directory/'plan.json',plan)
        try:
            broker=self.create('broker',asset_id,[mount(directory/'plan.json','/trusted/plan.json'),
                mount(socket_dir,'/run/kws',True),mount(evidence,'/evidence',True)],
                ['python','-I','-B','/opt/bounded_connect.py','--plan','/trusted/plan.json',
                 '--socket','/run/kws/broker.sock','--evidence','/evidence'],MIB)
            while not (evidence/'ready.json').exists():
                require(self.state(broker),'broker readiness failure'); self.sleep(min(0.1,self.remaining()))
            require(read_json(evidence/'ready.json')=={'schema':'bounded-connect-ready-v1','status':'ready','asset_id':asset_id},'broker readiness')
            check_socket(socket_dir/'broker.sock',self.uid,self.gid)
            retain_kernel_scope(evidence/'kernel-scope.json',self.evidence/(asset_id+'-broker-kernel.json'),'broker',self.uid)
            worker=self.create('worker',asset_id,[*worker_mounts,mount(socket_dir,'/run/kws')],worker_command,file_limit)
            self.watch(worker,broker)
            # Worker is stopped. Verify trusted artifact identity before sealing;
            # promotion occurs only after broker receipt and both removals.
            verify()
            if worker_evidence is not None:
                retain_worker_evidence(worker_evidence,self.evidence,asset_id,self.uid,expected_row,required=True)
            self.call('kill','--signal=USR1',broker); self.watch(broker)
            receipt=read_json(evidence/'receipt.json'); charged=validate_broker_receipt(receipt,plan,allowance)
            write_once(self.evidence/(asset_id+'-broker.json'),receipt)
            self.cleanup(); successful=True
        finally:
            try:
                if self.created: self.cleanup()
            finally:
                if not successful:
                    if worker_evidence is not None:
                        try:
                            retain_worker_evidence(worker_evidence,self.evidence,asset_id,self.uid,expected_row,required=False)
                        except (OSError,ValueError,KeyError,TypeError):
                            pass  # Unvalidated bytes are never public evidence.
                    failure={'schema':'bounded-host-asset-failure-v1','asset_id':asset_id,
                        'status':'failed_no_retry','broker_failure_code':None,'broker_observed_ciphertext_bytes':None,
                        'budget_charge_kind':'conservative_reserved_allowance' if charged is None else 'observed'}
                    try:
                        receipt=read_json(evidence/'receipt.json')
                        if receipt.get('asset_id')==asset_id and receipt.get('prior_charged_bytes')==plan['prior_charged_bytes']:
                            if receipt.get('failure_code') in ('plan','state','request','destination','dns','connection',
                                'connection_limit','deadline','byte_cap','io','poisoned'):
                                failure['broker_failure_code']=receipt['failure_code']
                            seen=receipt.get('charged_bytes')
                            if type(seen) is int and 0<=seen<=allowance:
                                failure['broker_observed_ciphertext_bytes']=seen
                    except (OSError,ValueError,KeyError,TypeError):
                        pass
                    write_once(self.evidence/(asset_id+'-failure.json'),failure)
                self.ledger.finish(charged,success=successful)
        return charged

    def offline(self,phase,mounts,command):
        require(not self.created and not self.failed and not self.ledger.failed,'offline clean boundary')
        try:
            name=self.create('offline',phase,mounts,command,4*1024**3)
            self.watch(name)
        finally:
            if self.created:self.cleanup()


def verify_source(directory,rows):
    directory=Path(directory)
    expected={r['path']:r for r in rows}
    require(not any(p.is_symlink() for p in directory.rglob('*')),'source aliases')
    require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}==set(expected),'source membership')
    for name,row in expected.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'source path')
        path=directory/name
        require(regular(path,row['size_bytes']).st_size==row['size_bytes'] and digest(path)==row['sha256'],'host source identity')


def acquire_runtime(*,docker,setup_code,runtime,output,root,job_deadline,run_id,image=IMAGE,
                    clock=time.monotonic,sleep=time.sleep,uid=None,gid=None):
    """Single absolute 1200s setup within 2400s host, only after outer admission.

    22 serial wheels -> offline install/check/inventory -> official source ->
    checkpoint -> offline finalization. Every failure propagates; caller retains
    denominator16 as not_run and must never invoke inference after an exception.
    """
    require(image==IMAGE and re.fullmatch('[1-9][0-9]{0,19}',str(run_id)),'fixed hosted setup identity')
    uid=os.getuid() if uid is None else uid; gid=os.getgid() if gid is None else gid
    require(uid>0,'nonroot host runner')
    require(type(job_deadline) in (int,float) and math.isfinite(job_deadline) and clock()<job_deadline<=clock()+2400,'host deadline')
    deadline=min(clock()+1200,job_deadline)
    setup_code=Path(setup_code); runtime=Path(runtime); output=Path(output); root=Path(root)
    code=setup_code/'research/source-screen32-whisper-v1/attempt-3'
    base=code.parent; manifest=base/'runtime-lock.json'
    wheels=read_json(manifest)['files']; sources=read_json(base/'source-lock.json')['files']
    require(len(wheels)==22 and len(sources)==22,'fixed locked asset roster')
    control=secure_directory(root/'bounded-setup',uid,gid)
    evidence=secure_directory(output/'bounded-acquisition',uid,gid)
    wheelhouse=secure_directory(runtime/'wheels',uid,gid)
    supervisor=Supervisor(docker,code,control,evidence,deadline,uid,gid,str(run_id),clock,sleep)
    for index,row in enumerate(wheels):
        asset_id=f'wheel-{index+1:02d}'
        candidate=secure_directory(control/(asset_id+'-candidate'),uid,gid)
        seconds=min(300,int(supervisor.remaining()))
        command=['python','-I','-B','/opt/pip_acquire.py','--asset-index',str(index),
            '--manifest','/trusted/runtime-lock.json','--output-root','/wheel-tmpfs',
            '--broker-socket','/run/kws/broker.sock','--candidate-root','/candidate','--seconds',str(seconds)]
        supervisor.pair(asset_id,'wheel',row['size_bytes'],[mount(manifest,'/trusted/runtime-lock.json'),
            mount(candidate,'/candidate',True)],command,row['size_bytes'],lambda:verify_candidate(candidate,row),expected_row=row)
        admit_candidate(candidate,row,wheelhouse)
        shutil.rmtree(candidate)
    phase_script='/code/research/source-screen32-whisper-v1/attempt-3/phased_setup.py'
    mounts=[mount(setup_code,'/code'),mount(runtime,'/runtime',True),mount(output,'/output',True)]
    def command(phase):return ['python','-I','-B',phase_script,phase]
    supervisor.offline('install',mounts,command('install'))
    supervisor.pair('source','source',sum(r['size_bytes'] for r in sources),mounts,command('source'),
        4*1024**3,lambda:verify_source(runtime/'source',sources))
    model_bytes=483617219
    model_sha='9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'
    def verify_model():
        path=runtime/'model/small.pt'
        require(regular(path,model_bytes).st_size==model_bytes and digest(path)==model_sha,'host model identity')
    supervisor.pair('model','model',model_bytes,mounts,command('model'),model_bytes,verify_model)
    acquisition=secure_directory(control/'acquisition',uid,gid)
    summary=supervisor.ledger.summary(acquisition/'receipt.json')
    write_once(evidence/'receipt.json',summary)
    supervisor.offline('finalize',[*mounts,mount(acquisition,'/acquisition')],command('finalize'))
    require(clock()<deadline and not supervisor.created,'setup complete within absolute bound')
    return {'schema':'bounded-host-setup-v1','status':'complete','aggregate_cap':CAP,
            'total_read_bytes':supervisor.ledger.total,'asset_count':len(supervisor.ledger.events),
            'network_teardown_complete':True,'model_calls':0,'setup_deadline_seconds':1200}


def public_metadata_names():
    assets=[f'wheel-{i:02d}' for i in range(1,23)]+['source','model']
    names={'budget-initial.json','receipt.json','install-offline-scope.json','finalize-offline-scope.json'}
    names|={f'budget-{i:03d}-{suffix}.json' for i in range(24) for suffix in ('started','final')}
    names|={asset+suffix for asset in assets for suffix in ('-broker.json','-broker-scope.json','-worker-scope.json','-broker-kernel.json','-worker-result.json','-worker-kernel.json','-failure.json')}
    names|={f'bounded-host-cleanup-{i:03d}.json' for i in range(26)}
    return frozenset(names)


def retain_kernel_scope(source,destination,role,uid):
    snapshot=validate_kernel_snapshot(read_json(source),role)
    require(snapshot.get('schema')=='bounded-kernel-scope-v1' and snapshot.get('role')==role,'kernel record identity')
    require(snapshot['uid']==uid,'kernel runner identity')
    safe={k:snapshot[k] for k in ('uid','cgroup','limits','readonly_root','private_readonly_cgroup','interfaces')}
    safe.update(schema='bounded-kernel-scope-v1',role=role,status={k:snapshot['status'][k] for k in
        ('NoNewPrivs','Seccomp','NSpid','CapInh','CapPrm','CapEff','CapBnd','CapAmb')})
    write_once(destination,safe)


def retain_worker_evidence(directory,destination,asset_id,uid,expected_row,*,required):
    """No log parsing: only fixed, source-validated records cross this boundary."""
    import importlib.util
    directory=Path(directory); destination=Path(destination)
    require(re.fullmatch(r'wheel-(0[1-9]|1[0-9]|2[0-2])',asset_id),'worker asset identity')
    require(not any(p.is_symlink() for p in directory.iterdir()),'worker evidence aliases')
    require({p.name for p in directory.iterdir()}<={'worker-scope.json','worker-result.json'},'worker evidence membership')
    result_path=directory/'worker-result.json'; scope_path=directory/'worker-scope.json'
    if required:
        require(result_path.exists() and scope_path.exists(),'required worker evidence')
    if scope_path.exists():
        regular(scope_path,4096)
        target=destination/(asset_id+'-worker-kernel.json')
        if not target.exists():retain_kernel_scope(scope_path,target,'worker',uid)
    if not result_path.exists():return
    regular(result_path,4096); value=read_json(result_path)
    spec=importlib.util.spec_from_file_location('bounded_worker_evidence_validator',Path(__file__).with_name('pip_acquire.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try:module.validate_result(value)
    except Exception:raise ValueError('worker result schema') from None
    index=int(asset_id[-2:])-1
    require(value['asset_index'] in (None,index),'worker result index')
    if value['status']=='ok':
        require(expected_row is not None and value['asset_index']==index and value['pip_version'] is not None and
            all(value[k]==expected_row[k] for k in ('filename','size_bytes','sha256')),'worker result artifact binding')
    if required:require(value['status']=='ok','worker success evidence')
    target=destination/(asset_id+'-worker-result.json')
    if target.exists():require(read_json(target)==value,'immutable worker evidence')
    else:write_once(target,value)
