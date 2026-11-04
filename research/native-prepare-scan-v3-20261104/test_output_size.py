"""Deterministic local-filesystem fixtures; no container/model/network calls."""
from pathlib import Path
import ast
import errno
import importlib.util
import json
import types
from contextlib import contextmanager
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import output_size as scanner

class OutputSizeTests(unittest.TestCase):
    def test_old_is_file_then_stat_reproduces_atomic_replace_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);temporary=root/'setup-progress.json.tmp';target=root/'setup-progress.json'
            temporary.write_bytes(b'new');target.write_bytes(b'old')
            original=Path.is_file
            def racing(path):
                result=original(path)
                if path==temporary and result:os.replace(temporary,target)
                return result
            with patch.object(Path,'is_file',racing):
                with self.assertRaises(FileNotFoundError):
                    sum(path.stat().st_size for path in [temporary] if path.is_file())
            self.assertEqual(target.read_bytes(),b'new')
    def test_exact_regular_total_nested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'a'*11);(root/'nested').mkdir();(root/'nested/b').write_bytes(b'b'*7)
            self.assertEqual(scanner.observed_output_size(root),18)
    def test_atomic_replace_during_entry_stat_rescans_whole_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);temporary=root/'setup-progress.json.tmp';target=root/'setup-progress.json'
            temporary.write_bytes(b'new-body');target.write_bytes(b'old')
            original=os.stat;triggered=[]
            def racing(name,*args,**kwargs):
                if name==temporary.name and kwargs.get('dir_fd') is not None and not triggered:
                    triggered.append(True);os.replace(temporary,target)
                return original(name,*args,**kwargs)
            with patch.object(scanner.os,'stat',side_effect=racing):
                self.assertEqual(scanner.observed_output_size(root),len(b'new-body'))
            self.assertEqual(triggered,[True])
    def test_disappearance_is_not_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(scanner,'_scan',side_effect=[FileNotFoundError(errno.ENOENT,'rename'),42]) as scan:
                self.assertEqual(scanner.observed_output_size(root),42);self.assertEqual(scan.call_count,2)
    def test_persistent_enoent_stops_with_bounded_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(scanner,'_scan',side_effect=FileNotFoundError(errno.ENOENT,'persistent')) as scan:
                with self.assertRaises(scanner.UnstableOutputError):scanner.observed_output_size(Path(tmp))
                self.assertEqual(scan.call_count,scanner.MAX_ATTEMPTS)
    def test_nontransient_io_errors_never_retry(self):
        for number in (errno.EIO,errno.EACCES,errno.ELOOP,errno.ENOTDIR):
            with tempfile.TemporaryDirectory() as tmp:
                error=OSError(number,'private diagnostic')
                with patch.object(scanner,'_scan',side_effect=error) as scan:
                    with self.assertRaises(OSError) as caught:scanner.observed_output_size(Path(tmp))
                    self.assertIs(caught.exception,error);self.assertEqual(scan.call_count,1)
    def test_exact_64mib_allowed_one_more_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'payload'
            with path.open('wb') as stream:stream.truncate(scanner.LIMIT)
            self.assertEqual(scanner.observed_output_size(Path(tmp)),scanner.LIMIT)
            with path.open('ab') as stream:stream.write(b'x')
            with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(Path(tmp))
    def test_limit_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for limit in (scanner.LIMIT+1,True,float(scanner.LIMIT)):
                with self.assertRaises(ValueError):scanner.observed_output_size(Path(tmp),limit)
    def test_file_and_directory_symlinks_rejected(self):
        for is_dir in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'output';root.mkdir();outside=Path(tmp)/'outside'
                if is_dir:outside.mkdir()
                else:outside.write_bytes(b'not readable through output')
                (root/'alias').symlink_to(outside,target_is_directory=is_dir)
                with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
    def test_root_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();alias=Path(tmp)/'alias';alias.symlink_to(root,target_is_directory=True)
            with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(alias)
    def test_hardlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'output';root.mkdir();outside=Path(tmp)/'outside';outside.write_bytes(b'x')
            os.link(outside,root/'alias')
            with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
    def test_fifo_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);os.mkfifo(str(root/'pipe'))
            with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
    def test_in_place_growth_invalidates_sum_and_recounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'payload';path.write_bytes(b'x')
            original=scanner._open;opens=[]
            def racing(name,parent,directory=False):
                if name=='payload':
                    opens.append(name)
                    if len(opens)==2:path.write_bytes(b'new grown bytes')
                return original(name,parent,directory)
            with patch.object(scanner,'_open',side_effect=racing):
                self.assertEqual(scanner.observed_output_size(root),len(b'new grown bytes'))
            self.assertGreaterEqual(len(opens),4)
    def test_directory_mutation_restarts_without_omitting_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'first').write_bytes(b'1');original=os.scandir;called=[]
            @contextmanager
            def racing(fd):
                with original(fd) as iterator:entries=list(iterator)
                if not called:called.append(True);(root/'second').write_bytes(b'22')
                yield iter(entries)
            with patch.object(scanner.os,'scandir',side_effect=racing):
                self.assertEqual(scanner.observed_output_size(root),3)
    def test_depth_and_entry_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'1');(root/'b').write_bytes(b'2')
            with patch.object(scanner,'MAX_ENTRIES',1):
                with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
            (root/'sub').mkdir()
            with patch.object(scanner,'MAX_DEPTH',0):
                with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
    def test_streaming_iteration_stops_at_entry_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('a','b','c','d'):(root/name).write_bytes(b'')
            original=os.scandir;seen=[];closed=[]
            @contextmanager
            def counted(fd):
                with original(fd) as iterator:
                    def entries():
                        for entry in iterator:
                            seen.append(entry.name);yield entry
                    try:yield entries()
                    finally:closed.append(True)
            with patch.object(scanner,'MAX_ENTRIES',2),patch.object(scanner.os,'scandir',side_effect=counted):
                with self.assertRaises(scanner.OutputBoundError):scanner.observed_output_size(root)
            self.assertEqual(len(seen),3);self.assertEqual(closed,[True])
    def test_expired_observation_stops(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(scanner.time,'monotonic',side_effect=[0,1]):
            with self.assertRaises(scanner.UnstableOutputError):scanner.observed_output_size(Path(tmp))
    def test_late_final_root_snapshot_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(scanner,'_scan',return_value=0),\
             patch.object(scanner.time,'monotonic',side_effect=[0,0,1]):
            with self.assertRaises(scanner.UnstableOutputError):scanner.observed_output_size(Path(tmp))
    def supervisor(self,scan_results):
        screen=types.ModuleType('screen')
        def require(ok,message):
            if not ok:raise ValueError(message)
        screen.require=require;screen.canonical=lambda value:json.dumps(value).encode()
        with patch.dict(sys.modules,screen=screen):
            spec=importlib.util.spec_from_file_location('test_boundary_overlay',HERE/'boundary.py')
            boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)
        calls=[]
        def docker(*args,**kwargs):
            calls.append(args)
            if args[:2]==('inspect','--format'):return json.dumps(dict(Running=False,ExitCode=0,OOMKilled=False))
            if args[0]=='inspect':return '[{}]'
            return ''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'output';root.mkdir()
            with patch.object(boundary,'docker',side_effect=docker),patch.object(boundary,'validate_container_inspect'),\
                 patch.object(boundary,'observed_output_size',side_effect=scan_results) as observation,\
                 patch.object(boundary.time,'monotonic',return_value=0):
                try:result=boundary.supervise_container(['create'],'dedicated',root,'setup',100)
                except ValueError as error:result=error
            self.assertIn(('rm','--force','dedicated'),calls)
            self.assertEqual(calls.count(('create',)),1);self.assertEqual(calls.count(('start','dedicated')),1)
            return result,observation.call_count,json.loads((Path(tmp)/'container-setup.json').read_text())
    def test_supervisor_stops_and_cleans_after_unstable_scan(self):
        result,count,receipt=self.supervisor([scanner.UnstableOutputError('unstable')])
        self.assertIsInstance(result,scanner.UnstableOutputError);self.assertEqual(count,1)
        self.assertEqual(receipt['status'],'failed_no_retry');self.assertEqual(receipt['operations']['cleanup']['status'],'complete')
    def test_final_stopped_container_check_catches_late_overbudget(self):
        result,count,receipt=self.supervisor([0,scanner.OutputBoundError('over limit')])
        self.assertIsInstance(result,scanner.OutputBoundError);self.assertEqual(count,2)
        self.assertEqual(receipt['status'],'failed_no_retry')
    def test_success_requires_both_live_and_post_exit_accounting(self):
        result,count,receipt=self.supervisor([0,0])
        self.assertEqual(result['status'],'complete');self.assertEqual(count,2)
        self.assertEqual(receipt['status'],'complete')
    def test_boundary_overlay_changes_only_accounting_and_final_check(self):
        tree=ast.parse((HERE/'boundary.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise_container')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='observed_output_size']
        self.assertEqual(len(calls),2)
        self.assertIn("docker('rm', '--force'",(HERE/'boundary.py').read_text())

if __name__=='__main__':unittest.main()
