"""Offline fixtures only: never calls Docker, pip, DNS or acquisition."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_host_under_test',HERE/'bounded_host.py')
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


def kernel(role):
    p=h.PROFILES[role]
    return {'schema':'bounded-kernel-scope-v1','role':role,'uid':1000,'cgroup':'0::/','limits':{'memory.max':str(p['memory']),'memory.swap.max':'0',
        'cpu.max':f"{p['cpus']*100000} 100000",'pids.max':str(p['pids'])},
        'status':{'NoNewPrivs':'1','Seccomp':'2','NSpid':'1',**{k:'0000000000000000' for k in
        ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')}},'readonly_root':True,'private_readonly_cgroup':True,
        'interfaces':['eth0','lo'] if role=='broker' else ['lo']}


def inspect(role,mounts,command,limit=123):
    p=h.PROFILES[role]; tmp='/tmp' if role=='broker' else '/wheel-tmpfs'; size=(16 if role=='broker' else 512)*h.MIB
    return {'HostConfig':{'Memory':p['memory'],'MemorySwap':p['memory'],'NanoCpus':p['cpus']*10**9,'PidsLimit':p['pids'],
        'ShmSize':(64 if role=='offline' else 32)*h.MIB,'ReadonlyRootfs':True,'Privileged':False,'CgroupnsMode':'private','PidMode':'','IpcMode':'private',
        'CapAdd':None,'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],'NetworkMode':p['network'],
        'LogConfig':{'Type':'local','Config':{'max-size':'1m','max-file':'1','compress':'false'}},
        'Ulimits':[{'Name':k,'Soft':a,'Hard':b} for k,(a,b) in {'core':(0,0),'nofile':(1024,1024),'fsize':(limit,limit)}.items()],
        'Tmpfs':{tmp:f'rw,noexec,nosuid,nodev,size={size},uid=1000,gid=1000,mode=0700'}},
        'Config':{'Image':h.IMAGE,'User':'1000:1000','Cmd':command,'Entrypoint':None},
        'Mounts':[{'Type':'bind','Source':m['source'],'Destination':m['destination'],'RW':m['writable'],'Propagation':'rprivate'}
        for m in [h.mount('/code','/opt'),*mounts]]}


class Contracts(unittest.TestCase):
    def test_sum_resource_envelope(self):
        self.assertEqual(sum(h.PROFILES[r]['memory'] for r in ('broker','worker')),12*1024**3)
        self.assertEqual(sum(h.PROFILES[r]['cpus'] for r in ('broker','worker')),4)
        self.assertEqual(sum(h.PROFILES[r]['pids'] for r in ('broker','worker')),256)

    def test_argv(self):
        mounts=[h.mount('/socket','/run/kws')]
        args=h.container_args('worker','test-1','/code',mounts,['python','-I','-B','/opt/pip_acquire.py'],uid=1000,gid=1000,file_limit=123)
        self.assertEqual(args[args.index('--network')+1],'none')
        self.assertIn('fsize=123:123',args)
        self.assertIn('type=bind,src=/socket,dst=/run/kws,readonly',args)
        self.assertFalse(any(x in args for x in ('--privileged','--publish','--net=host')))
        with self.assertRaises(ValueError):h.container_args('offline','test-1','/code',mounts,['python'],uid=1000,gid=1000,file_limit=123)
        with self.assertRaises(ValueError):h.container_args('worker','test-1','/code',[],['python'],uid=0,gid=0,file_limit=123)

    def test_inspect_and_tampering(self):
        mounts=[h.mount('/socket','/run/kws')]; command=['python','-I','-B','/opt/pip_acquire.py']
        good=inspect('worker',mounts,command)
        def verify(v):h.validate_inspect(v,'worker','/code',mounts,uid=1000,gid=1000,file_limit=123,command=command)
        verify(good)
        for field,value in [('Memory',12*1024**3),('MemorySwap',-1),('NanoCpus',4*10**9),('PidsLimit',256),
            ('ShmSize',64*h.MIB),('Privileged',True),('NetworkMode','bridge'),('PortBindings',{'1/tcp':[{}]}),('CapAdd',['NET_ADMIN']),('Tmpfs',{})]:
            bad=copy.deepcopy(good);bad['HostConfig'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):verify(bad)
        bad=copy.deepcopy(good);bad['Mounts'].append({'Type':'volume'})
        with self.assertRaises(ValueError):verify(bad)

    def test_kernel_proof_mutations(self):
        for role in h.PROFILES:h.validate_kernel_snapshot(kernel(role),role)
        for field,value in [('uid',0),('cgroup','0::/host'),('interfaces',['eth0','lo']),('readonly_root',False)]:
            bad=kernel('worker');bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):h.validate_kernel_snapshot(bad,'worker')
        bad=kernel('worker');bad['status']['CapEff']='1'
        with self.assertRaises(ValueError):h.validate_kernel_snapshot(bad,'worker')

    def test_ledger_no_retry_or_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=h.Ledger(tmp);allowance=ledger.begin('wheel-01','wheel',5)
            ledger.finish(10,success=True);self.assertEqual(ledger.total,10)
            with self.assertRaises(ValueError):ledger.begin('wheel-01','wheel',5)
            reserve=ledger.begin('wheel-02','wheel',5);ledger.finish(None,success=False)
            self.assertEqual(ledger.total,10+reserve)
            self.assertIsNone(ledger.events[-1]['observed_ciphertext_bytes'])
            with self.assertRaises(ValueError):ledger.begin('wheel-03','wheel',5)
            with self.assertRaises(FileExistsError):h.Ledger(tmp)

    def test_candidate_exact_membership_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'candidate';directory.mkdir();wheelhouse=Path(tmp)/'wheels';wheelhouse.mkdir()
            row={'filename':'one.whl','size_bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
            path=directory/'one.whl';path.write_bytes(b'abc')
            h.verify_candidate(directory,row);h.admit_candidate(directory,row,wheelhouse)
            (directory/'extra').write_bytes(b'')
            with self.assertRaises(ValueError):h.verify_candidate(directory,row)
            (directory/'extra').unlink();path.write_bytes(b'abd')
            with self.assertRaises(ValueError):h.verify_candidate(directory,row)
            path.unlink();path.symlink_to(wheelhouse/'one.whl')
            with self.assertRaises(ValueError):h.verify_candidate(directory,row)

    def test_absolute_deadline_and_no_overlapping_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); calls=[]
            s=h.Supervisor(lambda *a,**kw:calls.append(a),directory,directory,directory,100,1000,1000,'1',lambda:90,lambda _:None)
            with self.assertRaises(ValueError):s.remaining()
            self.assertEqual(calls,[])
            s.created=['old-broker']
            with self.assertRaises(ValueError):s.pair('new','wheel',10,[],[],10,lambda:None)

    def test_cleanup_all_even_when_one_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def docker(*args,**kwargs):
                calls.append(args)
                if args[-1]=='worker':raise RuntimeError('fixture')
            s=h.Supervisor(docker,tmp,tmp,tmp,100,1000,1000,'1',lambda:10,lambda _:None)
            s.created=['broker','worker']
            with self.assertRaises(ValueError):s.cleanup()
            self.assertEqual([x[-1] for x in calls],['worker','broker'])
            self.assertEqual(s.created,[])

    def test_bad_receipt_never_charges_as_observed(self):
        plan={'asset_id':'wheel-01','kind':'wheel','expected_size':100,'prior_charged_bytes':10}
        value={'schema':'bounded-connect-ledger-v1','status':'complete','failure_code':None,'plan_complete':True,
            **plan,'aggregate_cap':h.CAP,'byte_scope':'bidirectional_tunnel_socket_recv_before_forwarding',
            'asset_cap':h.MIB+102,'charged_bytes':105,'total_read_bytes':115,
            'assets':{'wheel-01':{'complete':True,'connections_started':1,'total_read_bytes':105,
                'upstream_read_bytes':5,'downstream_read_bytes':100}}}
        self.assertEqual(h.validate_broker_receipt(value,plan,h.MIB+102),105)
        for field,wrong in [('charged_bytes',True),('total_read_bytes',114),('prior_charged_bytes',0),('failure_code','io'),('asset_cap',h.CAP)]:
            bad=copy.deepcopy(value);bad[field]=wrong
            with self.subTest(field=field),self.assertRaises(ValueError):h.validate_broker_receipt(bad,plan,h.MIB+102)

    def test_cleanup_after_deadline_still_removes(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            s=h.Supervisor(lambda *a,**k:calls.append(a),tmp,tmp,tmp,100,1000,1000,'1',lambda:101,lambda _:None)
            s.created=['broker','worker']
            with self.assertRaises(ValueError):s.cleanup()
            self.assertEqual([a[-1] for a in calls],['worker','broker'])
            self.assertTrue(s.failed)
            receipt=h.read_json(Path(tmp)/'bounded-host-cleanup-000.json')
            self.assertTrue(all(r['deadline_overrun'] for r in receipt['containers']))

    def test_uncertain_create_is_registered_for_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def docker(*args,**kwargs):
                calls.append(args)
                if args[0]=='create':raise TimeoutError('fixture uncertain outcome')
            s=h.Supervisor(docker,tmp,tmp,tmp,100,1000,1000,'1',lambda:10,lambda _:None)
            with self.assertRaises(TimeoutError):s.create('broker','wheel-01',[],['python'],123)
            self.assertEqual(s.created,['kws-bound-1-wheel-01-broker'])
            s.cleanup()
            self.assertEqual(calls[-1],('rm','--force','kws-bound-1-wheel-01-broker'))

    def test_pair_failure_removes_both_and_poison_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'control';control.mkdir();evidence=root/'evidence';evidence.mkdir()
            calls=[]
            s=h.Supervisor(lambda *a,**k:calls.append(a),root,control,evidence,100,os.getuid(),os.getgid(),'1',lambda:10,lambda _:None)
            def create(role,stage,mounts,command,file_limit):
                s.created.append(role)
                if role=='broker':
                    h.write_once(control/stage/'evidence/ready.json',{'schema':'bounded-connect-ready-v1','status':'ready','asset_id':stage})
                return role
            s.create=create;s.watch=lambda *a:None
            def invalid():raise ValueError('fixture bad hash')
            with patch.object(h,'check_socket',lambda *a:None),patch.object(h,'retain_kernel_scope',lambda *a:None),patch.object(h,'retain_worker_evidence',lambda *a,**k:None),self.assertRaises(ValueError):
                s.pair('wheel-01','wheel',100,[],['python'],100,invalid)
            self.assertEqual([a[-1] for a in calls],['worker','broker'])
            self.assertTrue(s.ledger.failed)
            self.assertIsNone(s.ledger.events[0]['observed_ciphertext_bytes'])
            self.assertTrue((evidence/'wheel-01-failure.json').exists())
            self.assertEqual(h.read_json(control/'wheel-01/socket/deadline.json'),{'deadline_monotonic':90})

    def test_public_metadata_roster_is_bounded(self):
        names=h.public_metadata_names()
        self.assertIn('wheel-22-failure.json',names)
        self.assertIn('bounded-host-cleanup-025.json',names)
        self.assertNotIn('bounded-host-cleanup-026.json',names)
        self.assertNotIn('small.pt',names)

    def test_serial_pair_success_seals_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'control';control.mkdir();evidence=root/'evidence';evidence.mkdir()
            calls=[];events=[]
            def docker(*args,**kwargs):calls.append(args)
            s=h.Supervisor(docker,root,control,evidence,100,os.getuid(),os.getgid(),'1',lambda:10,lambda _:None)
            def create(role,stage,mounts,command,file_limit):
                s.created.append(role)
                if role=='broker':
                    h.write_once(control/stage/'evidence/ready.json',{'schema':'bounded-connect-ready-v1','status':'ready','asset_id':stage})
                return role
            s.create=create
            def watch(name,broker=None):
                if name=='broker':
                    self.assertEqual(events,['verified'])
                    self.assertEqual(calls[-1],('kill','--signal=USR1','broker'))
                    h.write_once(control/'wheel-01/evidence/receipt.json',{'schema':'bounded-connect-ledger-v1','status':'complete',
                        'failure_code':None,'plan_complete':True,'asset_id':'wheel-01','kind':'wheel','expected_size':100,
                        'prior_charged_bytes':0,'aggregate_cap':h.CAP,'byte_scope':'bidirectional_tunnel_socket_recv_before_forwarding',
                        'asset_cap':h.MIB+102,'charged_bytes':105,'total_read_bytes':105,
                        'assets':{'wheel-01':{'complete':True,'connections_started':1,'total_read_bytes':105,
                            'upstream_read_bytes':5,'downstream_read_bytes':100}}})
            s.watch=watch
            with patch.object(h,'check_socket',lambda *a:None),patch.object(h,'retain_kernel_scope',lambda *a:None),patch.object(h,'retain_worker_evidence',lambda *a,**k:None):
                charged=s.pair('wheel-01','wheel',100,[],['python'],100,lambda:events.append('verified'))
            self.assertEqual(charged,105);self.assertEqual(s.ledger.total,105)
            self.assertFalse(s.ledger.failed);self.assertEqual(s.created,[])
            self.assertEqual([a[-1] for a in calls[-2:]],['worker','broker'])

    def test_full_phase_order_and_install_failure_blocks_model(self):
        actual_code=HERE.parents[2]
        for fail_install in (False,True):
            with self.subTest(fail_install=fail_install),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);runtime=root/'runtime';runtime.mkdir();output=root/'output';output.mkdir();order=[]
                class FakeLedger:
                    total=24
                    def __init__(self):self.events=[]
                    def summary(self,path):
                        value={'assets':self.events};h.write_once(path,value);return value
                class FakeSupervisor:
                    def __init__(self,*args):self.ledger=FakeLedger();self.created=[]
                    def remaining(self):return 100
                    def pair(self,asset_id,*args,**kwargs):order.append(asset_id);self.ledger.events.append(asset_id)
                    def offline(self,phase,*args):
                        order.append(phase)
                        if phase=='install' and fail_install:raise ValueError('fixture dependency failure')
                with patch.object(h,'Supervisor',FakeSupervisor),patch.object(h,'admit_candidate',lambda *a:None),patch.object(h,'secure_directory',lambda p,*a:h_test_mkdir(p)):
                    kwargs=dict(docker=lambda *a,**k:None,setup_code=actual_code,runtime=runtime,output=output,root=root,
                        job_deadline=2000,run_id='1',clock=lambda:1,uid=1000,gid=1000)
                    if fail_install:
                        with self.assertRaises(ValueError):h.acquire_runtime(**kwargs)
                        self.assertEqual(order,[f'wheel-{i:02d}' for i in range(1,23)]+['install'])
                    else:
                        result=h.acquire_runtime(**kwargs)
                        self.assertEqual(result['asset_count'],24)
                        self.assertEqual(order,[f'wheel-{i:02d}' for i in range(1,23)]+['install','source','model','finalize'])

    def test_retained_kernel_scope_is_validated_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.json';target=Path(tmp)/'target.json'
            value=kernel('broker');value['status']['unrelated_private_field']='omitted'
            h.write_once(source,value);h.retain_kernel_scope(source,target,'broker',1000)
            self.assertNotIn('unrelated_private_field',h.read_json(target)['status'])
            with self.assertRaises(ValueError):h.retain_kernel_scope(source,Path(tmp)/'bad.json','broker',999)

    def test_worker_evidence_success_and_failure_are_strict(self):
        for status in ('ok','failed'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'source';source.mkdir();output=root/'output';output.mkdir()
                row={'filename':'one.whl','size_bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
                value={'schema':'bounded-pip-worker-result-v1','status':status,'stage':'complete' if status=='ok' else 'pip_help_probe',
                    'reason':None if status=='ok' else 'pip_capability_probe','asset_index':0,'pip_version':'26.1',
                    **(row if status=='ok' else dict.fromkeys(row))}
                h.write_once(source/'worker-result.json',value);h.write_once(source/'worker-scope.json',kernel('worker'))
                h.retain_worker_evidence(source,output,'wheel-01',1000,row,required=status=='ok')
                self.assertEqual(h.read_json(output/'wheel-01-worker-result.json'),value)
                if status=='failed':
                    with self.assertRaises(ValueError):h.retain_worker_evidence(source,output,'wheel-01',1000,row,required=True)
                value['reason']='private raw error from remote URL'
                (source/'worker-result.json').unlink();h.write_once(source/'worker-result.json',value)
                with self.assertRaises(ValueError):h.retain_worker_evidence(source,output,'wheel-01',1000,row,required=False)

    def test_transport_closed_is_only_a_bounded_counter(self):
        plan={'asset_id':'wheel-01','kind':'wheel','expected_size':100,'prior_charged_bytes':0}
        receipt={'schema':'bounded-connect-ledger-v1','status':'complete','failure_code':None,'plan_complete':True,
            **plan,'aggregate_cap':h.CAP,'byte_scope':'bidirectional_tunnel_socket_recv_before_forwarding',
            'asset_cap':h.MIB+102,'charged_bytes':105,'total_read_bytes':105,
            'assets':{'wheel-01':{'complete':True,'connections_started':1,'total_read_bytes':105,
                'upstream_read_bytes':5,'downstream_read_bytes':100,'transport_closed':1}}}
        self.assertEqual(h.validate_broker_receipt(receipt,plan,h.MIB+102),105)
        for bad_count in (True,-1,2,'1',None):
            value=copy.deepcopy(receipt);value['assets']['wheel-01']['transport_closed']=bad_count
            with self.subTest(value=bad_count),self.assertRaises(ValueError):h.validate_broker_receipt(value,plan,h.MIB+102)
        for reason in ('byte_cap','deadline','destination','connection','io'):
            value=copy.deepcopy(receipt);value['failure_code']=reason
            with self.subTest(reason=reason),self.assertRaises(ValueError):h.validate_broker_receipt(value,plan,h.MIB+102)
        value=copy.deepcopy(receipt);value['assets']['wheel-01']['raw_error']='not permitted'
        with self.assertRaises(ValueError):h.validate_broker_receipt(value,plan,h.MIB+102)

    def test_nonzero_worker_never_verifies_seals_or_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'control';control.mkdir();evidence=root/'evidence';evidence.mkdir()
            calls=[];verified=[]
            s=h.Supervisor(lambda *a,**k:calls.append(a),root,control,evidence,100,os.getuid(),os.getgid(),'1',lambda:10,lambda _:None)
            def create(role,stage,mounts,command,file_limit):
                s.created.append(role)
                if role=='broker':h.write_once(control/stage/'evidence/ready.json',
                    {'schema':'bounded-connect-ready-v1','status':'ready','asset_id':stage})
                return role
            s.create=create
            def stopped_worker(*args):raise ValueError('fixture worker exit1')
            s.watch=stopped_worker
            with patch.object(h,'check_socket',lambda *a:None),patch.object(h,'retain_kernel_scope',lambda *a:None),self.assertRaises(ValueError):
                s.pair('wheel-01','wheel',100,[],['python'],100,lambda:verified.append('would verify valid candidate'))
            self.assertEqual(verified,[])
            self.assertTrue(all(call[0]=='rm' for call in calls))
            self.assertTrue(s.ledger.failed)
            self.assertEqual(s.ledger.events[0]['status'],'failed_no_retry')


def h_test_mkdir(path):
    path=Path(path);path.mkdir(mode=0o700);return path


if __name__=='__main__':unittest.main()
