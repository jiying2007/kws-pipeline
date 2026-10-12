"""Pure-stdlib regressions: temporary files and synthetic children, no models."""
import errno
import inspect
import json
import os
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import supervision as host


class FilesystemSamplingTests(unittest.TestCase):
    def sampler(self,root):return host.FilesystemSampler(root,time.monotonic()+30)
    def row(self,sampler,scope='workspace'):return sampler.receipt['scopes'][scope]

    def test_static_bytes_count_symlink_itself_never_target(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();(r/'a').write_bytes(b'abc');(r/'sub').mkdir();(r/'sub/b').write_bytes(b'12345')
            out=Path(td)/'outside';out.mkdir();(out/'private').write_bytes(b'x'*10000)
            (r/'link').symlink_to(out,target_is_directory=True);(r/'broken').symlink_to('absent')
            obs={};self.assertEqual(host.tree_bytes(r,obs),8+len(str(out))+len('absent'))
            self.assertTrue(obs['complete']);self.assertEqual(obs['symlinks'],2)
            self.assertEqual(obs['directories'],1);self.assertEqual(obs['regular_files'],2)

    def test_one_nofollow_stat_per_entry(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat;calls=[]
            def observe(p,*a,**kw):
                if p=='item':calls.append(kw.copy())
                return real(p,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(host.tree_bytes(r),1)
            self.assertEqual(len(calls),1);self.assertIs(calls[0]['follow_symlinks'],False)
            self.assertIsInstance(calls[0]['dir_fd'],int)

    def test_real_unlink_then_stable_resample_returns_only_second(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);p=r/'vanish';p.write_bytes(b'1234');(r/'stable').write_bytes(b'abc')
            real=host.os.stat;done=False
            def observe(name,*a,**kw):
                nonlocal done
                if name=='vanish' and not done:done=True;p.unlink()
                return real(name,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(d.sample_bytes(r,'workspace',100),3)
            row=self.row(d);self.assertEqual((row['scans'],row['selected_sample']),(2,2))
            self.assertEqual(row['last_samples'][0]['vanished_entries'],1)
            self.assertFalse(row['last_samples'][0]['complete']);self.assertEqual(row['last_samples'][1]['observed_bytes'],3)

    def test_two_genuine_unlinks_fail_without_third_scan(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);(r/'stable').write_bytes(b'abc');realstat=host.os.stat;realscan=host.os.scandir
            @contextmanager
            def mutate(fd):
                (r/'vanish').write_bytes(b'xx')
                with realscan(fd) as entries:yield entries
            def observe(name,*a,**kw):
                if name=='vanish':(r/'vanish').unlink()
                return realstat(name,*a,**kw)
            with patch.object(host.os,'scandir',mutate),patch.object(host.os,'stat',observe),self.assertRaisesRegex(ValueError,'one bounded resample'):
                d.sample_bytes(r,'workspace',100)
            row=self.row(d);self.assertEqual(row['scans'],2);self.assertEqual(row['vanished_entries'],2)
            self.assertEqual(row['incomplete_scans'],2);self.assertIsNone(row['selected_sample'])
            self.assertEqual([s['observed_bytes'] for s in row['last_samples']],[3,3])
            self.assertEqual(row['status'],'incomplete_after_one_resample')

    def test_partial_cap_stops_before_next_entry_and_no_resample(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r)
            for name,size in (('a_vanish',1),('b_big',11),('z_after',1)):(r/name).write_bytes(b'x'*size)
            realscan=host.os.scandir;realstat=host.os.stat
            @contextmanager
            def ordered(fd):
                with realscan(fd) as entries:yield iter(sorted(entries,key=lambda e:e.name))
            def observe(name,*a,**kw):
                if name=='a_vanish':(r/name).unlink()
                if name=='z_after':raise AssertionError('metadata read after known cap')
                return realstat(name,*a,**kw)
            with patch.object(host.os,'scandir',ordered),patch.object(host.os,'stat',observe),self.assertRaisesRegex(ValueError,'stop threshold'):
                d.sample_bytes(r,'workspace',10)
            row=self.row(d);self.assertEqual(row['scans'],1);self.assertEqual(row['status'],'cap_exceeded')
            self.assertEqual(row['last_samples'][0]['observed_bytes'],11);self.assertEqual(row['vanished_entries'],1)

    def test_complete_real_cap_stops_immediately(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);(r/'big').write_bytes(b'x'*11)
            with self.assertRaises(ValueError):d.sample_bytes(r,'workspace',10)
            self.assertEqual(self.row(d)['scans'],1);self.assertEqual(self.row(d)['status'],'cap_exceeded')

    def test_non_enoent_fails_closed_and_preserves_original_exception(self):
        for number in (errno.EACCES,errno.EIO,errno.ELOOP,errno.ENOTDIR,errno.EINTR,errno.ESTALE,None,9999):
            with self.subTest(errno=number),tempfile.TemporaryDirectory() as td:
                r=Path(td);d=self.sampler(r);(r/'item').write_bytes(b'x');real=host.os.stat;error=OSError(number,'PRIVATE','/PRIVATE')
                def observe(name,*a,**kw):
                    if name=='item':raise error
                    return real(name,*a,**kw)
                with patch.object(host.os,'stat',observe),self.assertRaises(OSError) as caught:d.sample_bytes(r,'workspace',100)
                self.assertIs(caught.exception,error);self.assertEqual(self.row(d)['scans'],1)
                self.assertIsNone(self.row(d)['selected_sample']);self.assertNotIn('PRIVATE',json.dumps(d.receipt))

    def test_runtime_error_is_not_a_retry(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.sampler(td);error=RuntimeError('PRIVATE')
            with patch.object(host,'tree_bytes',side_effect=error) as scan,self.assertRaises(RuntimeError) as caught:d.sample_bytes(td,'workspace',100)
            self.assertIs(caught.exception,error);self.assertEqual(scan.call_count,1)

    def test_deadline_prevents_entry_metadata_access(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat;obs={}
            def observe(name,*a,**kw):
                if name=='item':raise AssertionError('read after deadline')
                return real(name,*a,**kw)
            with patch.object(host.time,'monotonic',side_effect=[0,0,0,1]),patch.object(host.os,'stat',observe),self.assertRaisesRegex(ValueError,'wall ceiling'):
                host.tree_bytes(r,obs,deadline=1)
            self.assertEqual(obs['stop_reason'],'deadline_exhausted')

    def test_same_deadline_for_both_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.sampler(td);real=host.tree_bytes;seen=[]
            def sample(root,obs,**kw):
                seen.append(kw['deadline']);amount=real(root,obs,**kw)
                if len(seen)==1:obs.update(complete=False,vanished_entries=1,coverage='incomplete_lower_bound')
                return amount
            with patch.object(host,'tree_bytes',sample):self.assertEqual(d.sample_bytes(td,'workspace',10),0)
            self.assertEqual(seen,[d.deadline,d.deadline])

    def test_exhausted_deadline_blocks_resample(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.sampler(td)
            def sample(root,obs,**kw):
                obs.update(observed_bytes=3,vanished_entries=1,complete=False,missing_root=False,root_missing_at_start=False)
                return 3
            with patch.object(host,'tree_bytes',side_effect=sample) as scan,patch.object(host.time,'monotonic',side_effect=[d.deadline-1,d.deadline]),self.assertRaisesRegex(ValueError,'wall ceiling'):
                d.sample_bytes(td,'workspace',10)
            self.assertEqual(scan.call_count,1);self.assertEqual(self.row(d)['status'],'deadline_exhausted')
            self.assertIsNone(self.row(d)['selected_sample'])

    def test_deadline_after_scan_does_not_select_sample(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.sampler(td)
            def sample(root,obs,**kw):
                obs.update(observed_bytes=3,vanished_entries=0,complete=True,missing_root=False,root_missing_at_start=False)
                return 3
            with patch.object(host,'tree_bytes',sample),patch.object(host.time,'monotonic',side_effect=[d.deadline-1,d.deadline]),self.assertRaisesRegex(ValueError,'wall ceiling'):
                d.sample_bytes(td,'workspace',10)
            self.assertEqual(self.row(d)['status'],'deadline_exhausted');self.assertIsNone(self.row(d)['selected_sample'])

    def test_optional_initial_absence_is_explicit_and_required_missing_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r)
            self.assertEqual(d.sample_bytes(r/'venv','installed',10,optional_root=True),0)
            row=self.row(d,'installed');self.assertEqual(row['last_samples'][0]['coverage'],'absent_optional_root')
            self.assertEqual(row['absent_optional_roots'],1)
            with self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(r/'buildtmp','buildtmp',10)
            self.assertEqual(self.row(d,'buildtmp')['scans'],1)

    def test_seen_optional_root_cannot_later_be_absent(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);out=r/'venv';out.mkdir()
            self.assertEqual(d.sample_bytes(out,'installed',10,optional_root=True),0);out.rmdir()
            with self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(out,'installed',10,optional_root=True)
            self.assertTrue(self.row(d,'installed')['observed_root_once']);self.assertEqual(self.row(d,'installed')['absent_optional_roots'],0)

    def test_optional_root_disappears_after_lstat_not_initial_absence(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);out=r/'venv';out.mkdir();real=host.os.open
            def change(p,*a,**kw):
                if p=='venv':out.rmdir()
                return real(p,*a,**kw)
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root missing'):
                d.sample_bytes(out,'installed',100,optional_root=True)
            self.assertFalse(self.row(d,'installed')['last_samples'][0]['root_missing_at_start'])

    def test_workspace_missing_or_replaced_is_fatal(self):
        for replacement in (False,True):
            with self.subTest(replacement=replacement),tempfile.TemporaryDirectory() as td:
                r=Path(td)/'task';r.mkdir();d=self.sampler(r);r.rename(Path(td)/'old')
                if replacement:r.mkdir()
                with self.assertRaises((OSError,ValueError)):d.sample_bytes(r,'workspace',10)
                self.assertEqual(self.row(d)['scans'],1)

    def test_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'link').symlink_to(r,target_is_directory=True)
            with self.assertRaises(OSError):host.tree_bytes(r/'link')
            with self.assertRaises(ValueError):self.sampler(r/'link')

    def test_root_lstat_open_replacement_rejects_and_closes(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();real=host.os.open;opened=[]
            def change(p,*a,**kw):
                if Path(p)==r:r.rename(Path(td)/'old');r.mkdir()
                fd=real(p,*a,**kw);opened.append(fd);return fd
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root directory identity'):host.tree_bytes(r)
            for fd in opened:
                with self.assertRaises(OSError):os.fstat(fd)

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

    def test_file_to_symlink_after_stat_uses_observed_file_only(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();p=r/'item';p.write_bytes(b'1234567');out=Path(td)/'outside';out.write_bytes(b'x'*10000);real=host.os.stat
            def change(name,*a,**kw):
                info=real(name,*a,**kw)
                if name=='item':p.unlink();p.symlink_to(out)
                return info
            with patch.object(host.os,'stat',change):self.assertEqual(host.tree_bytes(r),7)
            self.assertEqual(host.tree_bytes(r),len(str(out)))

    def test_fixed_child_uses_workspace_fd_after_ancestor_rename(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.sampler(r);(r/'venv').mkdir();(r/'venv/a').write_bytes(b'123')
            out=Path(td)/'outside';out.mkdir();(out/'venv').mkdir();(out/'venv/private').write_bytes(b'x'*10000)
            real=host.os.open;changed=False
            def move(p,*a,**kw):
                nonlocal changed
                fd=real(p,*a,**kw)
                if Path(p)==r and not changed:changed=True;r.rename(Path(td)/'oldtask');r.symlink_to(out,target_is_directory=True)
                return fd
            with patch.object(host.os,'open',move):self.assertEqual(d.sample_bytes(r/'venv','installed',100),3)
            with self.assertRaises(OSError):d.sample_bytes(r/'venv','installed',100)

    def test_all_fds_close_on_success_or_scan_error_and_no_growth(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as td:
                r=Path(td);d=self.sampler(r);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x')
                opened=[];realopen=host.os.open;realstat=host.os.stat;before=len(os.listdir('/proc/self/fd'))
                def observe(*a,**kw):fd=realopen(*a,**kw);opened.append(fd);return fd
                def observe_stat(p,*a,**kw):
                    if fail and p=='item':raise OSError(errno.EIO,'PRIVATE')
                    return realstat(p,*a,**kw)
                with patch.object(host.os,'open',observe),patch.object(host.os,'stat',observe_stat):
                    for _ in range(30):
                        if fail:
                            with self.assertRaises(OSError):d.sample_bytes(r,'workspace',100)
                        else:self.assertEqual(d.sample_bytes(r,'workspace',100),1)
                self.assertGreaterEqual(len(opened),90);self.assertEqual(len(os.listdir('/proc/self/fd')),before)
                for fd in opened:
                    with self.assertRaises(OSError):os.fstat(fd)

    def test_primary_scan_error_survives_all_close_failures(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x')
            realstat=host.os.stat;realclose=host.os.close;primary=OSError(errno.EACCES,'PRIVATE_PRIMARY');closed=[]
            def observe(name,*a,**kw):
                if name=='item':raise primary
                return realstat(name,*a,**kw)
            def close(fd):realclose(fd);closed.append(fd);raise OSError(errno.EIO,'PRIVATE_CLOSE')
            with patch.object(host.os,'stat',observe),patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:
                d.sample_bytes(r,'workspace',100)
            self.assertIs(caught.exception,primary);sample=self.row(d)['last_samples'][0]
            self.assertEqual(sample['fd_close_error_count'],3);self.assertEqual(sample['fd_close_first_error']['errno'],errno.EIO)
            self.assertNotIn('PRIVATE',json.dumps(d.receipt));self.assertEqual(len(set(closed)),3)
            for fd in closed:
                with self.assertRaises(OSError):os.fstat(fd)

    def test_cap_failure_survives_fd_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.sampler(r);(r/'item').write_bytes(b'xx');realclose=host.os.close
            def close(fd):realclose(fd);raise OSError(errno.EIO,'PRIVATE_CLOSE')
            with patch.object(host.os,'close',close),self.assertRaisesRegex(ValueError,'stop threshold'):d.sample_bytes(r,'workspace',1)
            self.assertEqual(self.row(d)['status'],'cap_exceeded');self.assertEqual(self.row(d)['scans'],1)

    def test_close_only_error_is_fatal_including_enoent_and_ambient_exception(self):
        for number in (errno.EIO,errno.ENOENT):
            with self.subTest(errno=number),tempfile.TemporaryDirectory() as td:
                r=Path(td);d=self.sampler(r);(r/'sub').mkdir();realclose=host.os.close;error=OSError(number,'PRIVATE_CLOSE')
                def close(fd):realclose(fd);raise error
                try:raise ValueError('PRIVATE_HANDLED')
                except ValueError:
                    with patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:d.sample_bytes(r,'workspace',100)
                self.assertIs(caught.exception,error);row=self.row(d);self.assertEqual(row['scans'],1)
                self.assertEqual(row['vanished_entries'],0);self.assertGreater(row['last_samples'][0]['fd_close_error_count'],0)
                self.assertIsNone(row['selected_sample'])

    def test_receipt_size_is_bounded_by_last_two_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.sampler(td)
            for _ in range(100):d.sample_bytes(td,'workspace',10)
            row=self.row(d);self.assertEqual(row['checks'],100);self.assertEqual(row['scans'],100)
            self.assertEqual(len(row['last_samples']),1);self.assertLess(len(json.dumps(d.receipt)),3000)


class SupervisorTests(unittest.TestCase):
    def run_child(self,r,code='import time; time.sleep(.05)',**changes):
        kwargs=dict(cwd=r,env=dict(os.environ),wall_seconds=1200,cpu_seconds=2400,rss_bytes=4*host.GiB,
            cores=4,workspace=r,installed=r/'venv',buildtmp=r/'buildtmp',log_path=r/'run.log',
            job_deadline=time.monotonic()+2400,setup=True,stderr_path=r/'run.stderr')
        kwargs.update(changes)
        return host.supervise([sys.executable,'-I','-B','-c',code],**kwargs)

    def test_original_signature_stays_compatible(self):
        self.assertEqual(list(inspect.signature(host.supervise).parameters),['command','cwd','env','wall_seconds','cpu_seconds',
            'rss_bytes','cores','workspace','installed','buildtmp','log_path','job_deadline','setup','stderr_path'])

    def test_real_child_success_and_preserved_setup_limits(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);result=self.run_child(r)
            self.assertEqual(result['returncode'],0);self.assertEqual(result['process_returncode'],0)
            self.assertEqual(result['termination_reason'],'normal_exit');self.assertIsNone(result['filesystem_error'])
            self.assertEqual(result['cpu_limit_seconds_per_process'],2400);self.assertEqual(result['wall_limit_seconds'],1200)
            self.assertEqual(result['rss_threshold_bytes'],4*host.GiB);self.assertEqual(result['affinity_core_limit'],4)
            self.assertEqual(result['sample_interval_seconds'],.25)
            for scope in ('installed','buildtmp'):
                self.assertEqual(result['filesystem_sampling']['scopes'][scope]['last_samples'][0]['coverage'],'absent_optional_root')

    def test_real_child_generation_and_preserved_limits(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'venv').mkdir();(r/'buildtmp').mkdir()
            result=self.run_child(r,setup=False,cpu_seconds=3600,rss_bytes=7*host.GiB)
            self.assertEqual(result['returncode'],0);self.assertEqual(result['cpu_limit_seconds_per_process'],3600)
            self.assertEqual(result['wall_limit_seconds'],1200);self.assertEqual(result['rss_threshold_bytes'],7*host.GiB)

    def test_real_child_transient_unlink_succeeds_with_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'vanish').write_bytes(b'123');real=host.os.stat;done=False
            def observe(name,*a,**kw):
                nonlocal done
                if name=='vanish' and not done:done=True;(r/'vanish').unlink()
                return real(name,*a,**kw)
            with patch.object(host.os,'stat',observe):result=self.run_child(r)
            row=result['filesystem_sampling']['scopes']['workspace'];self.assertEqual(result['returncode'],0)
            self.assertEqual(row['selected_sample'],2);self.assertEqual(row['incomplete_scans'],1)

    def test_real_child_repeated_enoent_fails_with_no_incomplete_zero_maximum(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);realscan=host.os.scandir;realstat=host.os.stat
            @contextmanager
            def mutate(fd):
                # Only the sampler uses an integer FD; /proc enumeration stays untouched.
                if isinstance(fd,int):(r/'vanish').write_bytes(b'xx')
                with realscan(fd) as entries:yield entries
            def observe(name,*a,**kw):
                if name=='vanish':(r/'vanish').unlink()
                return realstat(name,*a,**kw)
            with patch.object(host.os,'scandir',mutate),patch.object(host.os,'stat',observe):
                result=self.run_child(r,'import time; time.sleep(10)')
            row=result['filesystem_sampling']['scopes']['workspace'];self.assertNotEqual(result['returncode'],0)
            self.assertEqual(result['termination_reason'],'sampled_filesystem_error');self.assertEqual(row['scans'],2)
            self.assertIsNone(result['max_sampled']['workspace_bytes']);self.assertIsNone(row['selected_sample'])

    def test_real_child_permission_error_kills_reaps_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat
            def observe(name,*a,**kw):
                if name=='item':raise PermissionError(errno.EACCES,'PRIVATE','/PRIVATE')
                return real(name,*a,**kw)
            with patch.object(host.os,'stat',observe):result=self.run_child(r,'import time; time.sleep(10)')
            self.assertEqual(result['process_returncode'],-signal.SIGKILL);self.assertNotEqual(result['returncode'],0)
            self.assertEqual(result['filesystem_error']['errno'],errno.EACCES);self.assertNotIn('PRIVATE',json.dumps(result))
            self.assertIsNone(result['max_sampled']['workspace_bytes'])

    def test_actual_disk_thresholds_unchanged_with_sparse_tempfiles(self):
        for scope,cap in (('workspace',10*host.GiB),('installed',3*host.GiB),('buildtmp',2*host.GiB)):
            with self.subTest(scope=scope),tempfile.TemporaryDirectory() as td:
                r=Path(td);target=r if scope=='workspace' else r/('venv' if scope=='installed' else 'buildtmp')
                target.mkdir(exist_ok=True)
                with (target/'sparse').open('wb') as stream:stream.truncate(cap+1)
                result=self.run_child(r,'import time; time.sleep(10)')
                self.assertEqual(result['termination_reason'],'sampled_'+scope+'_cap');self.assertNotEqual(result['returncode'],0)
                row=result['filesystem_sampling']['scopes'][scope];self.assertEqual(row['scans'],1)
                self.assertEqual(row['last_samples'][0]['observed_bytes'],cap+1)

    def test_wall_deadline_kills_real_child(self):
        with tempfile.TemporaryDirectory() as td:
            result=self.run_child(Path(td),'import time; time.sleep(10)',wall_seconds=.05)
            self.assertEqual(result['termination_reason'],'wall_deadline');self.assertEqual(result['process_returncode'],-signal.SIGKILL)

    def test_scan_deadline_uses_earlier_stage_or_job_deadline(self):
        for job_first in (False,True):
            with self.subTest(job_first=job_first),tempfile.TemporaryDirectory() as td:
                r=Path(td);captured=[];real=host.FilesystemSampler.sample_bytes
                def sample(obj,*a,**kw):captured.append(obj.deadline);return real(obj,*a,**kw)
                deadline=time.monotonic()+(.08 if job_first else 2400)
                with patch.object(host.FilesystemSampler,'sample_bytes',sample):
                    result=self.run_child(r,'import time; time.sleep(10)',wall_seconds=.1,job_deadline=deadline)
                self.assertTrue(captured);self.assertEqual(len(set(captured)),1)
                self.assertLessEqual(captured[0],deadline)
                self.assertEqual(result['termination_reason'],'whole_job_deadline' if job_first else 'wall_deadline')

    def test_zero_exit_race_cannot_hide_supervisor_failure(self):
        class ExitedChild:
            pid=424242
            def poll(self):return None
            def wait(self):return 0
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            with patch.object(host.subprocess,'Popen',return_value=ExitedChild()),patch.object(host.os,'killpg',side_effect=ProcessLookupError),patch.object(host.FilesystemSampler,'sample_bytes',side_effect=OSError(errno.EIO,'PRIVATE')):
                result=self.run_child(r)
            self.assertEqual(result['process_returncode'],0);self.assertEqual(result['returncode'],1)
            self.assertEqual(result['termination_reason'],'sampled_filesystem_error')

    def test_no_wall_budget_does_not_start_child(self):
        with tempfile.TemporaryDirectory() as td,patch.object(host.subprocess,'Popen') as popen:
            with self.assertRaises(TimeoutError):self.run_child(Path(td),wall_seconds=0)
            popen.assert_not_called()

    def test_budget_exhausted_during_prelaunch_does_not_start_child(self):
        with tempfile.TemporaryDirectory() as td,patch.object(host.subprocess,'Popen') as popen:
            # run_child default job deadline, supervisor start, prelaunch guard.
            with patch.object(host.time,'monotonic',side_effect=[0,0,2]),self.assertRaises(TimeoutError):
                self.run_child(Path(td),wall_seconds=1,job_deadline=10)
            popen.assert_not_called()

    def test_child_kernel_cpu_and_affinity_forwarded_exactly(self):
        for cpu in (2400,3600):
            with patch.object(host.resource,'setrlimit') as limits,patch.object(host.os,'sched_getaffinity',return_value={1,2,3,4,5}),patch.object(host.os,'sched_setaffinity') as affinity:
                host.child_limits(cpu,4)
            limits.assert_called_once_with(host.resource.RLIMIT_CPU,(cpu,cpu));affinity.assert_called_once_with(0,{1,2,3,4})


if __name__=='__main__':unittest.main()
