"""Retained resource regression cases; local fake Docker and temporary bytes only."""
from pathlib import Path
import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import subprocess
import wave
import campaign_contract as c
from cpu_budget import CpuLedger,parse_cpu_stat,inference_filter
import resource_host as host

class OrchestrationTests(unittest.TestCase):
    def test_three_prepare_policy_three_decode(self):
        calls=[];clock=[0]
        def prepare(f,remaining):calls.append(('prepare',f));clock[0]+=10
        def freeze():calls.append(('freeze',))
        def decode(f,remaining):calls.append(('decode',f));self.assertEqual(remaining,590);clock[0]+=20
        result=host.recognition_stages(prepare,freeze,decode,setup_seconds=100,clock=lambda:clock[0])
        self.assertEqual(calls,[('prepare',f) for f in c.FAMILIES]+[('freeze',)]+[('decode',f) for f in c.FAMILIES])
        self.assertEqual(result['combined_setup_seconds'],130)
        self.assertEqual(result['engine_wall_seconds'],dict.fromkeys(c.FAMILIES,30))
    def test_prepare_failure_never_freezes_or_decodes(self):
        calls=[]
        def prepare(f,remaining):calls.append(f);raise ValueError('fixture')
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:self.fail('frozen'),lambda *a:self.fail('decoded'))
        self.assertEqual(calls,['qwen06'])
    def test_budget_failure_never_decodes(self):
        def freeze():raise ValueError('package cap')
        with self.assertRaises(ValueError):host.recognition_stages(lambda *a:None,freeze,lambda *a:self.fail('decoded'))
    def test_each_prepare_failure_blocks_policy_and_all_decodes(self):
        for failed in c.FAMILIES:
            calls=[]
            def prepare(f,remaining):
                calls.append(f)
                if f==failed:raise ValueError('same-run prepare failure')
            with self.subTest(failed=failed), self.assertRaisesRegex(ValueError,'prepare failure'):
                host.recognition_stages(prepare,lambda:self.fail('policy frozen after failure'),
                    lambda *a:self.fail('decode after prepare failure'))
            self.assertEqual(calls,list(c.FAMILIES[:c.FAMILIES.index(failed)+1]))
    def test_missing_identity_or_package_proof_blocks_all_decodes(self):
        for failure in (FileNotFoundError('missing prepare record'),ValueError('run/head identity mismatch'),
                        ValueError('prepared config drift'),ValueError('actual closure package cap')):
            calls=[]
            def freeze():
                calls.append('freeze');raise failure
            with self.subTest(failure=str(failure)), self.assertRaises(type(failure)):
                host.recognition_stages(lambda f,r:calls.append(f),freeze,
                    lambda *a:self.fail('decode before successful full freeze and package'))
            self.assertEqual(calls,[*c.FAMILIES,'freeze'])
    def test_fresh_decode_failure_prevents_later_families(self):
        for failed in c.FAMILIES:
            calls=[]
            def decode(f,remaining):
                calls.append(f)
                if f==failed:raise ValueError('fresh worker scope/guard/config failure')
            with self.subTest(failed=failed), self.assertRaisesRegex(ValueError,'fresh worker'):
                host.recognition_stages(lambda *a:None,lambda:None,decode)
            self.assertEqual(calls,list(c.FAMILIES[:c.FAMILIES.index(failed)+1]))
    def test_combined_family_wall_no_reset(self):
        clock=[0];calls=[]
        def prepare(f,remaining):clock[0]+=599
        def decode(f,remaining):calls.append(f);self.assertEqual(remaining,1);clock[0]+=2
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:None,decode,clock=lambda:clock[0])
        self.assertEqual(calls,['qwen06'])
    def test_prepare_counted_against_setup_budget(self):
        clock=[0]
        def prepare(f,remaining):clock[0]+=51
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:self.fail('frozen'),lambda *a:None,setup_seconds=1750,clock=lambda:clock[0])


class CpuTests(unittest.TestCase):
    def test_stat_parse(self):self.assertEqual(parse_cpu_stat('usage_usec 123\nuser_usec 1\nsystem_usec 122\n'),123)
    def test_bad_stat(self):
        for value in ('usage_usec -1','usage_usec 1\nusage_usec 2','user_usec 1','usage_usec NaN'):
            with self.assertRaises(ValueError):parse_cpu_stat(value)
    def test_cpu_reset_rejected(self):
        value=[4];ledger=CpuLedger();ledger.register('a','setup',lambda:value[0]);value[0]=3
        with self.assertRaises(ValueError):ledger.sample()
    def test_aggregate_threads_and_children_accounted(self):
        value=[0];ledger=CpuLedger();ledger.register('a','qwen06',lambda:value[0]);value[0]=600000001
        self.assertEqual(ledger.sample(),['qwen06']);self.assertTrue(ledger.receipt()['threshold_exceeded'])
    def test_no_reuse_or_false_hardclaim(self):
        ledger=CpuLedger();ledger.register('a','setup',lambda:5);ledger.close('a',10)
        self.assertTrue(ledger.receipt()['all_final_usage_verified']);self.assertFalse(ledger.receipt()['hard_cumulative_kernel_quota'])
        with self.assertRaises(ValueError):ledger.register('a','setup',lambda:0)
    def test_unknown_final_stays_unknown(self):
        ledger=CpuLedger();ledger.register('a','setup',lambda:5)
        self.assertFalse(ledger.receipt()['all_final_usage_verified'])
    def test_final_close_only_exceed_still_removes_then_fails(self):
        with tempfile.TemporaryDirectory() as d:
            docker=host.MeteredDocker(d,999999999999);docker.gate('last-whisper')
            row=docker.names['last-whisper'];row.update(create_attempted=True,created=True,container_id='a'*64,start_attempted=True,registered=True,reader=lambda:1800000001)
            docker.ledger.register('last-whisper','whisper',lambda:1799999999)
            calls=[]
            def raw(*args,**kwargs):
                calls.append(args)
                return '{"Paused":true}' if args[0]=='inspect' else ''
            docker.raw=raw
            with self.assertRaises(ValueError):docker.remove('last-whisper')
            self.assertTrue(row['removed']);self.assertIn(('rm','--force','a'*64),calls)
            with self.assertRaises(ValueError):docker.finish()

    def test_seccomp_syscall_dispatch(self):
        def run(nr,flags=0,arch=0xc000003e):
            code=inference_filter();pc=0;acc=0
            while True:
                op,jt,jf,k=code[pc]
                if op==0x20:acc={0:nr,4:arch,16:flags}[k];pc+=1
                elif op==0x15:pc+=1+(jt if acc==k else jf)
                elif op==0x45:pc+=1+(jt if acc&k else jf)
                elif op==0x06:return k
                else:self.fail('unknown BPF')
        for nr in (57,58,56):self.assertEqual(run(nr)&0xffff0000,0x00050000)
        self.assertEqual(run(56,0x10000),0x7fff0000)
        self.assertEqual(run(435),0x00050000|38)
        self.assertEqual(run(0),0x7fff0000)
        for nr in (0x40000000|56,0x40000000|57,0x40000000|58):self.assertEqual(run(nr),0x80000000)
        self.assertEqual(run(0,arch=0x40000003),0x80000000)


class CleanupAbsenceTests(unittest.TestCase):
    """One uncertain stop, bounded read-only proof, and no real Docker/model work."""
    CID='a'*64
    NAME='cleanup-owned-scope'

    def setUp(self):
        self.now=100.0;self.sleeps=[]
        clock=patch.object(host.time,'monotonic',side_effect=lambda:self.now)
        sleep=patch.object(host.time,'sleep',side_effect=self.sleep)
        subprocess_guard=patch.object(host.subprocess,'run',side_effect=AssertionError('real subprocess forbidden'))
        for replacement in (clock,sleep,subprocess_guard):
            replacement.start();self.addCleanup(replacement.stop)

    def sleep(self,seconds):
        self.assertGreater(seconds,0);self.assertLessEqual(seconds,1)
        self.sleeps.append(seconds);self.now+=seconds

    def scope(self,queries=(),*,host_seconds=100,registered=True,rm_ack=False,
              rm_seconds=5.007,pause_error=None,inspect_result='{"Paused":true}',stage='setup'):
        self.now=100.0;self.sleeps=[];self.calls=[];self.queries=list(queries)
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        docker=host.MeteredDocker(directory.name,self.now+host_seconds);docker.stage=stage;gate=docker.gate(self.NAME)
        (gate/'result.json').write_text('{"schema":"qwen12-gated-command-v1","exit_code":0}')
        row=docker.names[self.NAME]
        row.update(create_attempted=True,created=True,container_id=self.CID,
            start_attempted=registered,registered=registered,reader=lambda:2)
        if registered:docker.ledger.register(self.NAME,stage,lambda:1)
        def raw(*args,timeout=5):
            self.calls.append({'args':args,'timeout':timeout,'started':self.now})
            self.assertGreater(timeout,0);self.assertLessEqual(timeout,5)
            if args[0]=='pause':
                if pause_error is not None:raise pause_error
                return ''
            if args[0]=='inspect':return inspect_result
            if args[0]=='rm':
                self.assertEqual(args,('rm','--force',self.CID));self.now+=rm_seconds
                if not rm_ack:raise subprocess.TimeoutExpired('fixed synthetic rm',timeout)
                return ''
            self.assertEqual(args,('ps','--all','--no-trunc','--filter','id='+self.CID,'--format','{{.ID}}'))
            self.assertLessEqual(timeout,1)
            seconds,value=self.queries.pop(0) if self.queries else (0,self.CID+'\n')
            self.now+=seconds
            if isinstance(value,BaseException):raise value
            return value
        docker.raw=raw
        self.docker=docker;self.row=row
        return docker

    def reads(self):return [row for row in self.calls if row['args'][0]=='ps']

    def assert_receipt(self,*,removed,okay):
        import evidence_transport as transport
        receipt=c.read_json(self.docker.root/'cleanup-receipt.json')
        transport.validate_member_json('cleanup-receipt.json',receipt)
        self.assertEqual(receipt['all_removed'],removed)
        self.assertEqual(receipt['all_cleanup_verified'],okay)
        record=receipt['containers'][0]
        self.assertEqual(record['cleanup_budget_seconds'],15)
        self.assertEqual(record['cleanup_ok'],okay)
        self.assertLessEqual(len(record['operations']),14)
        self.assertLessEqual(len(record['errors']),16)
        self.assertLessEqual(len(record['receipt_errors']),16)
        self.assertEqual(sum(row['args'][0]=='rm' for row in self.calls),1)
        self.assertLessEqual(len(self.reads()),10)
        return record

    def test_acknowledged_remove_keeps_fast_path_without_absence_queries(self):
        docker=self.scope(rm_ack=True,rm_seconds=.01)
        self.assertEqual(docker.remove(self.NAME),'');docker.finish()
        self.assertEqual(self.reads(),[]);self.assertEqual(self.sleeps,[])
        record=self.assert_receipt(removed=True,okay=True)
        self.assertEqual(record['removal_proof'],'acknowledged_remove')
        self.assertEqual([row['operation'] for row in record['operations']],
            ['pause','inspect_paused','final_cpu_close','force_remove'])

    def test_r4_shaped_delayed_deletion_uses_remaining_window(self):
        docker=self.scope([(.019,self.CID+'\n'),(.01,'')])
        self.assertEqual(docker.remove(self.NAME),'');docker.finish()
        record=self.assert_receipt(removed=True,okay=True)
        self.assertEqual(record['removal_proof'],'successful_empty_exact_id_query')
        self.assertTrue(record['pause_acknowledged'])
        self.assertEqual(record['accounting_status'],'final_usage_verified')
        self.assertEqual(record['gate_result']['exit_code'],0)
        self.assertEqual(len(self.reads()),2);self.assertEqual(self.sleeps,[1])
        self.assertAlmostEqual(record['cleanup_seconds'],6.036)
        self.assertEqual([row['operation'] for row in record['errors']],['force_remove'])

    def test_tenth_successful_empty_query_is_within_existing_operation_cap(self):
        docker=self.scope([(0,self.CID)]*9+[(0,'')],rm_seconds=5)
        docker.remove(self.NAME)
        record=self.assert_receipt(removed=True,okay=True)
        self.assertEqual(len(self.reads()),10);self.assertEqual(len(record['operations']),14)
        self.assertEqual(record['cleanup_seconds'],14);self.assertEqual(self.sleeps,[1]*9)

    def test_persistent_exact_id_stops_after_ten_queries(self):
        docker=self.scope()
        with self.assertRaisesRegex(ValueError,'cleanup unconfirmed'):docker.remove(self.NAME)
        record=self.assert_receipt(removed=False,okay=False)
        self.assertEqual(len(self.reads()),10);self.assertEqual(len(record['operations']),14)
        self.assertEqual([row['operation'] for row in record['errors']],['force_remove','verify_absence'])
        self.assertNotIn('removal_proof',record)
        with self.assertRaises(ValueError):docker.finish()

    def test_all_metered_stages_keep_same_success_and_failure_rules(self):
        for stage in ('setup',*c.FAMILIES):
            for gone in (False,True):
                with self.subTest(stage=stage,gone=gone):
                    docker=self.scope([(0,self.CID),(0,'')] if gone else (),stage=stage)
                    if gone:docker.remove(self.NAME);docker.finish()
                    else:
                        with self.assertRaises(ValueError):docker.remove(self.NAME)
                        with self.assertRaises(ValueError):docker.finish()
                    record=self.assert_receipt(removed=gone,okay=gone)
                    self.assertEqual(record['stage'],stage)
                    self.assertEqual(record['accounting_status'],'final_usage_verified')

    def test_query_error_or_timeout_stops_without_retrying_that_read(self):
        failures=(subprocess.CalledProcessError(1,['private'],output=b'',stderr=b'PRIVATE'),
            subprocess.TimeoutExpired('private',1,output=b''),OSError('PRIVATE'),UnicodeError('PRIVATE'))
        for failure in failures:
            with self.subTest(kind=type(failure).__name__):
                docker=self.scope([(0,self.CID+'\n'),(1,failure),(0,'')])
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=False,okay=False)
                self.assertEqual(len(self.reads()),2);self.assertEqual(self.sleeps,[1])
                self.assertEqual(self.queries,[(0,'')])
                self.assertNotIn('PRIVATE',c.canonical(record).decode())

    def test_malformed_or_foreign_query_result_never_becomes_absence(self):
        for result in ('\n',' ','b'*64+'\n',self.CID[:12]+'\n',self.CID+'\n'+self.CID+'\n',
                       self.CID+'\r\n',' '+self.CID,self.CID.upper()+'\n',b''):
            with self.subTest(result=repr(result)):
                docker=self.scope([(0,result),(0,'')])
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                self.assert_receipt(removed=False,okay=False)
                self.assertEqual(len(self.reads()),1);self.assertEqual(self.sleeps,[])

    def test_unknown_or_malformed_owned_id_never_sends_docker_commands(self):
        for cid,created in ((None,False),(self.CID,False),('a'*12,True),('A'*64,True),('a'*64+'\n',True)):
            with self.subTest(cid=cid,created=created):
                docker=self.scope();self.row.update(container_id=cid,created=created)
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                self.assertEqual(self.calls,[]);self.assertFalse(self.row['cleanup_ok'])

    def test_absence_cannot_repair_pause_failure_or_missing_pause_ack(self):
        for options in ({'pause_error':subprocess.TimeoutExpired('pause',5)},
                        {'inspect_result':'{"Paused":false}'},{'inspect_result':'not-json'}):
            with self.subTest(options=options):
                docker=self.scope([(0,self.CID),(0,'')],**options)
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=True,okay=False)
                self.assertFalse(record['pause_acknowledged']);self.assertFalse(self.row['cpu_closed'])
                self.assertEqual(record['accounting_status'],'final_usage_unverified')
                self.assertIn('final_accounting',[row['operation'] for row in record['errors']])
                with self.assertRaises(ValueError):docker.finish()

    def test_absence_cannot_repair_missing_or_reset_final_cpu_usage(self):
        for reader in (lambda:0,lambda:(_ for _ in ()).throw(OSError('PRIVATE final CPU'))):
            with self.subTest(reader=reader):
                docker=self.scope([(0,'')]);self.row['reader']=reader
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=True,okay=False)
                self.assertTrue(record['pause_acknowledged']);self.assertFalse(self.row['cpu_closed'])
                self.assertEqual(record['accounting_status'],'final_usage_unverified')
                with self.assertRaises(ValueError):docker.finish()

    def test_absence_cannot_repair_unregistered_started_cpu_scope(self):
        docker=self.scope([(0,'')],registered=False);self.row['start_attempted']=True
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=True,okay=False)
        self.assertEqual(record['accounting_status'],'final_usage_unverified')

    def test_absence_keeps_never_started_owned_scope_cleanup_semantics(self):
        docker=self.scope([(0,'')],registered=False);docker.remove(self.NAME);docker.finish()
        record=self.assert_receipt(removed=True,okay=True)
        self.assertEqual(record['accounting_status'],'not_started');self.assertFalse(record['pause_acknowledged'])

    def test_absence_cannot_hide_final_or_preexisting_cpu_threshold_failure(self):
        for prior in (False,True):
            with self.subTest(prior=prior):
                docker=self.scope([(0,self.CID),(0,'')])
                if prior:docker.ledger.failed=True
                else:self.row['reader']=lambda:c.LIMITS['cpu_threshold_seconds']['setup']*1000000
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=True,okay=False)
                self.assertEqual(record['accounting_status'],'final_usage_verified')
                self.assertTrue(docker.ledger.failed)
                with self.assertRaises(ValueError):docker.finish()

    def test_host_deadline_limits_each_query_and_cumulative_wait(self):
        docker=self.scope([(0,self.CID),(.25,subprocess.TimeoutExpired('ps',.25)),(0,'')],
            host_seconds=6.25,rm_seconds=5)
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=False,okay=False)
        self.assertEqual([row['timeout'] for row in self.reads()],[1,.25])
        self.assertEqual(self.sleeps,[1]);self.assertEqual(record['cleanup_seconds'],6.25)
        self.assertTrue(record['host_deadline_overrun'])

    def test_sleep_uses_only_remaining_host_window_and_no_new_query(self):
        docker=self.scope([(0,self.CID),(0,'')],host_seconds=5.25,rm_seconds=5)
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=False,okay=False)
        self.assertEqual(self.sleeps,[.25]);self.assertEqual(len(self.reads()),1)
        self.assertEqual(record['cleanup_seconds'],5.25)

    def test_cleanup_deadline_is_cumulative_across_queries_and_sleeps(self):
        docker=self.scope([(1,self.CID)]*10,rm_seconds=5)
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=False,okay=False)
        self.assertEqual(len(self.reads()),5);self.assertEqual(self.sleeps,[1]*5)
        self.assertEqual(record['cleanup_seconds'],15);self.assertEqual(record['cleanup_overrun_seconds'],0)
        self.assertTrue(all(row['started']+row['timeout']<=115 for row in self.reads()))

    def test_late_empty_result_at_either_deadline_is_not_removal_proof(self):
        for host_seconds,query_seconds in ((6,1),(100,10)):
            with self.subTest(host_seconds=host_seconds):
                docker=self.scope([(query_seconds,'')],host_seconds=host_seconds,rm_seconds=5)
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=False,okay=False)
                self.assertNotIn('removal_proof',record);self.assertEqual(len(self.reads()),1)

    def test_expired_host_still_attempts_owned_stop_but_skips_optional_queries(self):
        for host_seconds,rm_ack in ((0,False),(-1,True),(2,False),(2,True)):
            with self.subTest(host_seconds=host_seconds,rm_ack=rm_ack):
                docker=self.scope(host_seconds=host_seconds,rm_ack=rm_ack)
                with self.assertRaises(ValueError):docker.remove(self.NAME)
                record=self.assert_receipt(removed=rm_ack,okay=False)
                self.assertEqual(self.reads(),[]);self.assertEqual(self.sleeps,[])
                self.assertTrue(record['host_deadline_overrun'])
                self.assertEqual(record['host_deadline_expired_at_start'],host_seconds<=0)
                self.assertEqual(record['accounting_status'],'final_usage_verified')

    def test_original_pause_and_final_close_reserve_still_leaves_one_stop(self):
        docker=self.scope([(0,'')],rm_seconds=5)
        original=docker.raw
        def delayed(*args,**kwargs):
            if args[0] in ('pause','inspect'):self.now+=5
            return original(*args,**kwargs)
        docker.raw=delayed
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=False,okay=False)
        self.assertFalse(self.row['cpu_closed']);self.assertEqual(self.reads(),[])
        self.assertEqual(record['cleanup_seconds'],15)
        self.assertEqual(next(row['timeout'] for row in self.calls if row['args'][0]=='rm'),5)

    def test_expired_final_cpu_read_still_stops_and_stays_failed(self):
        docker=self.scope([(0,'')],host_seconds=1,rm_ack=True,rm_seconds=.1)
        def reader():self.now+=2;return 2
        self.row['reader']=reader
        with self.assertRaises(ValueError):docker.remove(self.NAME)
        record=self.assert_receipt(removed=True,okay=False)
        self.assertTrue(record['host_deadline_overrun']);self.assertEqual(self.reads(),[])

    def test_caller_timeout_never_expands_remove_or_read_allowance(self):
        for timeout in (.25,50):
            with self.subTest(timeout=timeout):
                docker=self.scope([(0,self.CID),(0,'')],rm_seconds=.2)
                docker.remove(self.NAME,timeout=timeout)
                self.assert_receipt(removed=True,okay=True)
                self.assertEqual(next(row['timeout'] for row in self.calls if row['args'][0]=='rm'),min(timeout,5))
                self.assertTrue(all(row['timeout']<=min(timeout,1) for row in self.reads()))

    def test_cleanup_called_again_never_retries_or_takes_over_scope(self):
        for confirmed in (False,True):
            with self.subTest(confirmed=confirmed):
                docker=self.scope([(0,'')] if confirmed else ())
                if confirmed:docker.remove(self.NAME)
                else:
                    with self.assertRaises(ValueError):docker.remove(self.NAME)
                previous=copy.deepcopy(self.calls);sleep_count=len(self.sleeps)
                self.queries=[(0,'')]
                if confirmed:docker.remove(self.NAME);docker.finish()
                else:
                    with self.assertRaises(ValueError):docker.remove(self.NAME)
                    with self.assertRaises(ValueError):docker.finish()
                self.assertEqual(self.calls,previous);self.assertEqual(len(self.sleeps),sleep_count)
                self.assertEqual(len(docker.cleanup),1)
                with self.assertRaisesRegex(ValueError,'new container only'):docker.gate(self.NAME)

    def test_supervisor_cleanup_failure_prevents_policy_and_decode(self):
        import runtime_session as runtime
        docker=self.scope();events=[]
        def command(_docker,*args,**kwargs):
            events.append(args[0])
            return '[{}]' if args[0]=='inspect' else self.CID
        def prepare(family,remaining):
            runtime.supervise(docker,['create','--name',self.NAME],docker.root,
                family=family,mode='prepare',deadline=docker.deadline)
        with patch.object(runtime,'gated_arguments',side_effect=lambda args,*rest:args), \
             patch.object(runtime,'validate_launch'),patch.object(host.MeteredDocker,'__call__',new=command), \
             patch.object(docker,'check'),self.assertRaisesRegex(ValueError,'cleanup unconfirmed'):
            host.recognition_stages(prepare,lambda:self.fail('policy frozen after unresolved cleanup'),
                lambda *args:self.fail('decode after unresolved cleanup'),clock=lambda:self.now)
        self.assertEqual(events,['create','inspect','start'])
        self.assert_receipt(removed=False,okay=False)
        self.assertEqual(len(docker.supervisions),1)
        supervision=docker.supervisions[0]
        self.assertTrue(supervision['worker_exit_success']);self.assertIsNone(supervision['primary_error'])
        self.assertEqual(supervision['status'],'failed');self.assertIsNotNone(supervision['cleanup_error'])
        with self.assertRaises(ValueError):docker.finish()

    def test_supervisor_unresolved_decode_cleanup_prevents_later_families(self):
        import runtime_session as runtime
        docker=self.scope(stage='qwen06');events=[]
        def command(_docker,*args,**kwargs):return '[{}]' if args[0]=='inspect' else self.CID
        def decode(family,remaining):
            events.append(('decode',family))
            runtime.supervise(docker,['create','--name',self.NAME],docker.root,
                family=family,mode='decode',deadline=docker.deadline)
        with patch.object(runtime,'gated_arguments',side_effect=lambda args,*rest:args), \
             patch.object(runtime,'validate_launch'),patch.object(host.MeteredDocker,'__call__',new=command), \
             patch.object(docker,'check'),self.assertRaisesRegex(ValueError,'cleanup unconfirmed'):
            host.recognition_stages(lambda family,_:events.append(('prepare',family)),
                lambda:events.append(('freeze',)),decode,clock=lambda:self.now)
        self.assertEqual(events,[*[('prepare',family) for family in c.FAMILIES],('freeze',),('decode','qwen06')])
        self.assert_receipt(removed=False,okay=False)
        self.assertEqual(docker.supervisions[0]['status'],'failed')

    def test_supervisor_worker_error_is_not_hidden_by_eventual_absence(self):
        import runtime_session as runtime
        for family,mode in (('setup',None),('qwen06','prepare'),('qwen06','decode')):
            with self.subTest(family=family,mode=mode):
                docker=self.scope([(0,self.CID),(0,'')]);failure=ValueError('synthetic worker failure')
                def command(_docker,*args,**kwargs):return '[{}]' if args[0]=='inspect' else self.CID
                with patch.object(runtime,'gated_arguments',side_effect=lambda args,*rest:args), \
                     patch.object(runtime,'validate_launch'),patch.object(host.MeteredDocker,'__call__',new=command), \
                     patch.object(docker,'completed',side_effect=failure),self.assertRaises(ValueError) as caught:
                    runtime.supervise(docker,['create','--name',self.NAME],docker.root,
                        family=family,mode=mode,deadline=docker.deadline)
                self.assertIs(caught.exception,failure);self.assert_receipt(removed=True,okay=True)
                record=docker.supervisions[0]
                self.assertEqual(record['status'],'failed');self.assertFalse(record['worker_exit_success'])
                self.assertIsNotNone(record['primary_error']);self.assertIsNone(record['cleanup_error'])


class FilesystemSamplingTests(unittest.TestCase):
    def docker(self,root):return host.MeteredDocker(root,host.time.monotonic()+30)
    def partial(self,n=3):return dict(observed_bytes=n,vanished_entries=1,complete=False,missing_root=False,root_missing_at_start=False,coverage='incomplete_lower_bound')
    def test_static_bytes_and_root_json_keep_nofollow_rules(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();(r/'a.json').write_bytes(b'abc');(r/'b').write_bytes(b'12345');o=Path(td)/'outside';o.write_bytes(b'x'*4096);(r/'link.json').symlink_to(o);(r/'sub').mkdir();(r/'sub/c.json').write_bytes(b'zz')
            s={};self.assertEqual(host.tree_bytes(r,s),10+len(str(o)));self.assertTrue(s['complete']);self.assertEqual(s['symlinks'],1);self.assertEqual(host.tree_bytes(r,{},root_json_only=True),3)
    def test_one_nofollow_stat_per_entry(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat;calls=[]
            def observe(p,*a,**kw):
                if p=='item':calls.append(kw.copy())
                return real(p,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(host.tree_bytes(r),1)
            self.assertEqual(len(calls),1);self.assertIs(calls[0]['follow_symlinks'],False);self.assertIsInstance(calls[0]['dir_fd'],int)
    def test_real_unlink_then_stable_resample_returns_only_second(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);p=r/'vanish';p.write_bytes(b'1234');(r/'stable').write_bytes(b'abc');real=host.os.stat;done=False
            def observe(name,*a,**kw):
                nonlocal done
                if name=='vanish' and not done:done=True;p.unlink()
                return real(name,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(d.sample_bytes(r,'workspace',100),3)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual((row['scans'],row['selected_sample']),(2,2));self.assertEqual(row['last_samples'][0]['vanished_entries'],1);self.assertFalse(row['last_samples'][0]['complete']);self.assertEqual(row['last_samples'][1]['observed_bytes'],3)
    def test_two_partial_samples_fail_without_third(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial());return 3
            with patch.object(host,'tree_bytes',sample),self.assertRaisesRegex(ValueError,'one bounded resample'):d.sample_bytes(td,'workspace',10)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual(len(calls),2);self.assertIsNone(row['selected_sample']);self.assertEqual(row['vanished_entries'],2)
    def test_partial_over_cap_stops_before_resampling(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial(11));return 11
            with patch.object(host,'tree_bytes',sample),self.assertRaises(ValueError):d.sample_bytes(td,'workspace',10)
            self.assertEqual(len(calls),1);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['status'],'cap_exceeded')
    def test_complete_real_over_cap_stops(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);(Path(td)/'large').write_bytes(b'x'*11)
            with self.assertRaises(ValueError):d.sample_bytes(td,'workspace',10)
            self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1)
    def test_deadline_exhausted_does_not_start_resample(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial());return 3
            with patch.object(host,'tree_bytes',sample),patch.object(host.time,'monotonic',side_effect=[d.deadline-1,d.deadline]),self.assertRaisesRegex(ValueError,'wall ceiling'):d.sample_bytes(td,'workspace',10)
            self.assertEqual(len(calls),1);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['status'],'deadline_exhausted')
    def test_non_enoent_fail_closed_and_keep_original_exception(self):
        import errno
        for number in (errno.EACCES,errno.EIO,errno.ELOOP,errno.ENOTDIR,None,9999):
            with self.subTest(errno=number),tempfile.TemporaryDirectory() as td:
                d=self.docker(td);(Path(td)/'item').write_bytes(b'x');real=host.os.stat;error=OSError(number,'PRIVATE','/PRIVATE')
                def stat(p,*a,**kw):
                    if p=='item':raise error
                    return real(p,*a,**kw)
                with patch.object(host.os,'stat',stat),self.assertRaises(OSError) as caught:d.sample_bytes(td,'workspace',100)
                self.assertIs(caught.exception,error);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1);self.assertNotIn('PRIVATE',c.canonical(d.filesystem_sampling).decode())
    def test_partial_before_eio_preserves_counters_and_lower_bound(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td)
            def sample(root,obs,**kw):obs.update(self.partial(7));raise OSError(errno.EIO,'PRIVATE')
            with patch.object(host,'tree_bytes',sample),self.assertRaises(OSError):d.sample_bytes(td,'workspace',100)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual((row['vanished_entries'],row['incomplete_scans']),(1,1));self.assertEqual(row['last_samples'][0]['observed_bytes'],7)
    def test_optional_absence_and_required_workspace_missing_differ(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);self.assertEqual(d.sample_bytes(r/'output','public_output',100,optional_root=True),0)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertEqual(row['scans'],1);self.assertEqual(row['last_samples'][0]['coverage'],'absent_optional_root')
            r.rename(Path(td)/'moved')
            with self.assertRaises(OSError):d.sample_bytes(r,'workspace',100)
            self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1)
    def test_owned_workspace_replaced_between_samples_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);r.rename(Path(td)/'old');r.mkdir()
            with self.assertRaisesRegex(ValueError,'workspace identity'):d.sample_bytes(r,'workspace',100)
    def test_root_lstat_open_replacement_rejects_and_closes(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();real=host.os.open;opened=[]
            def change(p,*a,**kw):
                if Path(p)==r:r.rename(Path(td)/'old');r.mkdir()
                fd=real(p,*a,**kw);opened.append(fd);return fd
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root directory identity'):host.tree_bytes(r)
            for fd in opened:
                with self.assertRaises(OSError):os.fstat(fd)
    def test_optional_root_seen_then_deleted_is_not_initial_absence(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();real=host.os.open
            def change(p,*a,**kw):
                if p=='output':out.rmdir()
                return real(p,*a,**kw)
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(out,'public_output',100,optional_root=True)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertFalse(row['last_samples'][0]['root_missing_at_start']);self.assertEqual(row['absent_optional_roots'],0)
    def test_child_directory_replaced_by_symlink_or_other_directory_rejects(self):
        for symlink in (True,False):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as td:
                r=Path(td)/'task';r.mkdir();sub=r/'sub';sub.mkdir();out=Path(td)/'outside';out.mkdir();(out/'private').write_bytes(b'x'*10000);real=host.os.open
                def change(p,*a,**kw):
                    if p=='sub':
                        sub.rename(r/'old')
                        if symlink:sub.symlink_to(out,target_is_directory=True)
                        else:sub.mkdir()
                    return real(p,*a,**kw)
                with patch.object(host.os,'open',change),self.assertRaises((OSError,ValueError)):host.tree_bytes(r)
    def test_file_to_symlink_after_stat_counts_observed_file_only(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();p=r/'item';p.write_bytes(b'1234567');out=Path(td)/'outside';out.write_bytes(b'x'*10000);real=host.os.stat
            def change(name,*a,**kw):
                info=real(name,*a,**kw)
                if name=='item':p.unlink();p.symlink_to(out)
                return info
            with patch.object(host.os,'stat',change):self.assertEqual(host.tree_bytes(r),7)
            self.assertEqual(host.tree_bytes(r),len(str(out)))
    def test_all_descriptors_close_on_traversal_success_or_eio(self):
        import errno
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as td:
                r=Path(td);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x');opened=[];realopen=host.os.open;realstat=host.os.stat
                def observe(*a,**kw):fd=realopen(*a,**kw);opened.append(fd);return fd
                def stat(p,*a,**kw):
                    if fail and p=='item':raise OSError(errno.EIO,'PRIVATE')
                    return realstat(p,*a,**kw)
                with patch.object(host.os,'open',observe),patch.object(host.os,'stat',stat):
                    if fail:
                        with self.assertRaises(OSError):host.tree_bytes(r)
                    else:self.assertEqual(host.tree_bytes(r),1)
                self.assertGreaterEqual(len(opened),2)
                for fd in opened:
                    with self.assertRaises(OSError):os.fstat(fd)
    def test_real_concurrent_mutation_retains_non_atomic_sampling_semantics(self):
        import threading
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);paths=[r/str(i) for i in range(16)]
            for p in paths:p.write_bytes(b'x'*16)
            stop=threading.Event();errors=[]
            def churn():
                try:
                    while not stop.is_set():
                        for p in paths:
                            if stop.is_set():break
                            p.unlink();p.write_bytes(b'x'*16)
                except BaseException as error:errors.append(type(error).__name__)
            t=threading.Thread(target=churn);t.start()
            try:
                for _ in range(20):
                    obs={};amount=host.tree_bytes(r,obs);self.assertEqual(amount,obs['observed_bytes']);self.assertLessEqual(amount,256);self.assertEqual(obs['complete'],obs['vanished_entries']==0)
            finally:stop.set();t.join(2)
            self.assertFalse(t.is_alive());self.assertEqual(errors,[])
    def test_source_input_integrity_still_strict_for_missing_or_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'link').symlink_to('missing')
            for p in (r/'missing',r/'link'):
                with self.assertRaises((OSError,ValueError)):c.identity(p)
                with self.assertRaises((OSError,ValueError)):c.read_json(p)

    def test_early_cap_never_stats_the_next_entry(self):
        from contextlib import contextmanager
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'a_big').write_bytes(b'x'*11);(r/'z_after').write_bytes(b'x');scan=host.os.scandir;real=host.os.stat
            @contextmanager
            def ordered(fd):
                with scan(fd) as entries:yield iter(sorted(entries,key=lambda e:e.name))
            def stat(p,*a,**kw):
                if p=='z_after':raise AssertionError('read after known cap')
                return real(p,*a,**kw)
            obs={}
            with patch.object(host.os,'scandir',ordered),patch.object(host.os,'stat',stat),self.assertRaisesRegex(ValueError,'stop threshold'):host.tree_bytes(r,obs,limit=10)
            self.assertEqual(obs['observed_bytes'],11);self.assertEqual(obs['stop_reason'],'cap_exceeded')
    def test_entry_deadline_prevents_entry_metadata_access(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat
            def stat(p,*a,**kw):
                if p=='item':raise AssertionError('read after deadline')
                return real(p,*a,**kw)
            obs={}
            with patch.object(host.time,'monotonic',side_effect=[0,0,0,1]),patch.object(host.os,'stat',stat),self.assertRaisesRegex(ValueError,'wall ceiling'):host.tree_bytes(r,obs,deadline=1)
            self.assertEqual(obs['stop_reason'],'deadline_exhausted')
    def test_seen_optional_root_cannot_later_be_absent(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),0);out.rmdir()
            with self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(out,'public_output',10,optional_root=True)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertTrue(row['observed_root_once']);self.assertEqual(row['absent_optional_roots'],0)
    def test_legal_output_to_frozen_and_new_output_inode_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();(out/'old').write_bytes(b'123');self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),3)
            out.rename(r/'frozen');out.mkdir();(out/'new').write_bytes(b'45');self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),2)
            self.assertEqual(d.sample_bytes(r/'frozen','public_frozen',10,optional_root=True),3)
    def test_public_child_uses_open_workspace_fd_after_ancestor_rename(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);(r/'output').mkdir();(r/'output/old').write_bytes(b'123');outside=Path(td)/'outside';outside.mkdir();(outside/'output').mkdir();(outside/'output/private').write_bytes(b'x'*10000)
            real=host.os.open;changed=False
            def move(p,*a,**kw):
                nonlocal changed
                fd=real(p,*a,**kw)
                if Path(p)==r and not changed:changed=True;r.rename(Path(td)/'oldtask');r.symlink_to(outside,target_is_directory=True)
                return fd
            with patch.object(host.os,'open',move):self.assertEqual(d.sample_bytes(r/'output','public_output',100,optional_root=True),3)
            with self.assertRaises(OSError):d.sample_bytes(r/'output','public_output',100,optional_root=True)
    def test_public_aggregate_cap_does_not_reset_each_scope(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'output').mkdir();(r/'output/a').write_bytes(b'x'*6);(r/'whisper-setup').mkdir();(r/'whisper-setup/b').write_bytes(b'x'*5)
            with patch.dict(host.LIMITS,{'output_bytes':10}),self.assertRaisesRegex(ValueError,'stop threshold'):d.check()
            row=d.filesystem_sampling['scopes']['public_whisper_setup'];self.assertEqual(row['status'],'cap_exceeded');self.assertEqual(row['last_samples'][0]['observed_bytes'],5)

    def test_primary_scan_error_is_not_replaced_by_fd_cleanup_error(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x');realstat=host.os.stat;realclose=host.os.close;primary=OSError(errno.EACCES,'PRIVATE_PRIMARY');obs={}
            def stat(p,*a,**kw):
                if p=='item':raise primary
                return realstat(p,*a,**kw)
            def close(fd):realclose(fd);raise OSError(errno.EIO,'PRIVATE_CLOSE')
            with patch.object(host.os,'stat',stat),patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:host.tree_bytes(r,obs)
            self.assertIs(caught.exception,primary);self.assertEqual(obs['fd_close_error_count'],2);self.assertEqual(obs['fd_close_first_error']['errno'],errno.EIO);self.assertNotIn('PRIVATE',c.canonical(obs).decode())
    def test_fd_cleanup_error_without_primary_is_fatal(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            realclose=host.os.close;error=OSError(errno.EIO,'PRIVATE_CLOSE');obs={}
            def close(fd):realclose(fd);raise error
            with patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:host.tree_bytes(Path(td),obs)
            self.assertIs(caught.exception,error);self.assertEqual(obs['fd_close_error_count'],1)

    def test_close_enoent_is_never_treated_as_a_vanished_entry(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'sub').mkdir();realclose=host.os.close
            def close(fd):realclose(fd);raise OSError(errno.ENOENT,'PRIVATE_CLOSE')
            with patch.object(host.os,'close',close),self.assertRaises(OSError):d.sample_bytes(r,'workspace',100)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual(row['scans'],1);self.assertEqual(row['vanished_entries'],0);self.assertGreater(row['last_samples'][0]['fd_close_error_count'],0)
    def test_parent_close_failure_still_closes_workspace_anchor(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'output/qwen06').mkdir(parents=True);realopen=host.os.open;realclose=host.os.close;opened=[];closed=[];error=OSError(errno.EIO,'PRIVATE_PARENT_CLOSE')
            def openfd(*a,**kw):fd=realopen(*a,**kw);opened.append(fd);return fd
            def close(fd):
                realclose(fd);closed.append(fd)
                if fd==opened[1]:raise error
            with patch.object(host.os,'open',openfd),patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:d.sample_bytes(r/'output/qwen06','phase_output',100)
            self.assertIs(caught.exception,error);self.assertEqual(set(opened),set(closed));self.assertEqual(len(opened),len(closed))
            for fd in opened:
                with self.assertRaises(OSError):os.fstat(fd)

    def test_handled_ambient_exception_does_not_hide_close_only_failure(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);realclose=host.os.close;error=OSError(errno.EIO,'PRIVATE_CLOSE')
            def close(fd):realclose(fd);raise error
            try:raise ValueError('PRIVATE_HANDLED')
            except ValueError:
                with patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:d.sample_bytes(Path(td),'workspace',100)
            self.assertIs(caught.exception,error);self.assertIsNone(d.filesystem_sampling['scopes']['workspace']['selected_sample'])


if __name__=="__main__": unittest.main()
