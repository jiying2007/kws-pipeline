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
