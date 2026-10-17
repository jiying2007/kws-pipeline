import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import argparse
import os
import time
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.direct_fit import result_delivery as d


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_fixture(self, extra='', files=None, cap=262144):
        source = "import os,sys,json,hashlib; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); (p/'checkpoint').write_bytes(b'artificial optimizer bytes'); (p/'run.json').write_text(json.dumps({'status':'completed','trace_status':'completed','optimizer_steps':{'ctc':30,'direct':30},'checkpoints':{'checkpoint':hashlib.sha256((p/'checkpoint').read_bytes()).hexdigest()}})); " + extra
        return d.execute([sys.executable, '-c', source, '{output}'], self.root/'output', files or ['run.json','checkpoint'], cap)

    def test_restore_after_output_disappears(self):
        frame, manifest = self.run_fixture()
        shutil.rmtree(self.root/'output')
        result = d.recover(frame, self.root/'recovered')
        self.assertTrue(result['complete_selected_recovery'])
        self.assertEqual((self.root/'recovered/checkpoint').read_bytes(), b'artificial optimizer bytes')
        self.assertEqual(manifest['summary']['optimizer_steps'], {'ctc':30,'direct':30})
        self.assertEqual(result['delivery_status'], 'unconfirmed')

    def test_nonzero_is_preserved_with_result(self):
        frame, manifest = self.run_fixture('sys.exit(7)')
        self.assertEqual(manifest['terminal']['returncode'],7)
        self.assertEqual(d.recover(frame,self.root/'restore')['terminal']['returncode'],7)

    def test_inherited_pipe_does_not_block_forever(self):
        import time
        start=time.monotonic()
        _,manifest=self.run_fixture("import subprocess; subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)'])")
        self.assertLess(time.monotonic()-start,2.8)
        self.assertEqual(manifest['failures']['process_streams'],'incomplete_after_child_exit')
        self.assertEqual(manifest['terminal']['returncode'],0)

    def test_signal_is_preserved(self):
        _, manifest = self.run_fixture('os.kill(os.getpid(),9)')
        self.assertEqual(manifest['terminal']['returncode'],-9)
        self.assertEqual(manifest['terminal']['signal'],9)

    def test_missing_and_hash_binding(self):
        frame, manifest = self.run_fixture("(p/'checkpoint').unlink()")
        self.assertIn('checkpoint',manifest['failures'])
        self.assertIn('run_binding:checkpoint',manifest['failures'])
        self.assertFalse(d.recover(frame,self.root/'restore')['complete_selected_recovery'])

    def test_changed_checkpoint_rejected(self):
        _, manifest = self.run_fixture("(p/'checkpoint').write_bytes(b'changed')")
        self.assertIn('run_binding:checkpoint',manifest['failures'])

    def test_oversize_keeps_only_evidence(self):
        frame, manifest = self.run_fixture("(p/'noise').write_bytes(os.urandom(10000))", ['run.json','checkpoint','noise'],2048)
        self.assertEqual(manifest['recovery_scope'],'manifest_only')
        self.assertLessEqual(json.loads(frame[len(d.PREFIX):])['archive_bytes'],2048)
        self.assertFalse(d.recover(frame,self.root/'restore')['complete_selected_recovery'])

    def test_manifest_itself_over_limit(self):
        names=['item%03d' % i for i in range(100)]
        command=[sys.executable,'-c',
                 "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); [(p/('item%03d'%i)).write_bytes(os.urandom(100)) for i in range(100)]",'{output}']
        frame,manifest=d.execute(command,self.root/'output',names,2048)
        self.assertEqual(manifest['failures']['package'],'manifest_size_limit')
        self.assertLessEqual(json.loads(frame[len(d.PREFIX):])['archive_bytes'],2048)
        self.assertFalse(d.recover(frame,self.root/'restore')['complete_selected_recovery'])

    def test_truncation_and_tampering(self):
        frame, _ = self.run_fixture()
        for n in (1,20,len(frame)//2):
            with self.subTest(n=n), self.assertRaises((ValueError,KeyError)):
                d.recover(frame[:-n],self.root/'bad')
        value=json.loads(frame[len(d.PREFIX):]); value['archive_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archive_integrity'):
            d.recover(d.PREFIX.encode()+d.encoded(value)+b'\n',self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_symlinks_and_fifo_are_not_collected(self):
        _, manifest=self.run_fixture("(p/'secret').symlink_to('/etc/passwd'); os.mkfifo(p/'fifo'); (p/'dir').symlink_to('/etc')", ['run.json','checkpoint','secret','fifo','dir/passwd'])
        self.assertEqual(set(manifest['failures']),{'secret','fifo','dir/passwd'})

    def test_unsafe_paths(self):
        for name in ('../escape','/absolute','a/../b','a//b','a\\b','handoff-manifest.json','handoff.zip','handoff.zip/child'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_fixture(files=[name])

    def test_file_directory_collision(self):
        with self.assertRaisesRegex(ValueError, 'file_directory_collision'):
            self.run_fixture(files=['a','a/b'])

    def test_stale_directory_rejected_before_launch(self):
        (self.root/'output').mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.run_fixture()

    def test_launch_failure_is_terminal_unknown(self):
        frame,manifest=d.execute(['/nonexistent/artificial-command'],self.root/'out',['run.json'])
        self.assertIsNone(manifest['terminal']['returncode'])
        self.assertEqual(manifest['terminal']['launch_error'],'FileNotFoundError')
        self.assertFalse(d.recover(frame,self.root/'restore')['complete_selected_recovery'])

    def test_logs_are_hashed_not_disclosed(self):
        frame,manifest=self.run_fixture("print('fixture-secret-do-not-display'); print('error-secret',file=sys.stderr)")
        self.assertNotIn(b'fixture-secret-do-not-display',frame)
        self.assertEqual(manifest['streams']['stdout']['sha256'],d.sha(b'fixture-secret-do-not-display\n'))

    def test_forged_delivered_claim_rejected(self):
        frame,_=self.run_fixture()
        value=json.loads(frame[len(d.PREFIX):]); value['manifest']['delivery_status']='delivered'
        value['manifest_sha256']=d.sha(d.encoded(value['manifest']))
        with self.assertRaisesRegex(ValueError,'invalid_persistence_claim'):
            d.recover(d.PREFIX.encode()+d.encoded(value)+b'\n',self.root/'bad')

    def test_destination_must_be_fresh(self):
        frame,_=self.run_fixture()
        (self.root/'restore').mkdir()
        with self.assertRaises(FileExistsError):
            d.recover(frame,self.root/'restore')


class ReviewTest(unittest.TestCase):
    def test_inherited_pipe_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = "import subprocess,sys; from pathlib import Path; p=Path(sys.argv[1]);p.mkdir();(p/'r').write_bytes(b'x'); subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)'])"
            start = time.monotonic()
            frame, m = d.execute([sys.executable, '-c', source, '{output}'], root/'out', ['r'])
            self.assertLess(time.monotonic()-start, 2.9)
            self.assertEqual(m['terminal']['returncode'], 0)
            self.assertEqual(m['failures']['process_streams'], 'incomplete_after_child_exit')
            self.assertFalse(d.recover(frame, root/'restored')['complete_selected_recovery'])

    def test_leaf_insertion_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source="import sys; from pathlib import Path; p=Path(sys.argv[1]);p.mkdir();(p/'r').write_bytes(b'new')"
            frame, _=d.execute([sys.executable,'-c',source,'{output}'],root/'out',['r'])
            victim=root/'victim'; victim.write_bytes(b'old')
            original=os.open
            def raced(path,flags,*args,**kwargs):
                if path=='r' and flags & os.O_WRONLY:
                    os.symlink(str(victim),path,dir_fd=kwargs['dir_fd'])
                return original(path,flags,*args,**kwargs)
            with patch.object(os,'open',raced), self.assertRaises(FileExistsError):
                d.recover(frame,root/'restored')
            self.assertEqual(victim.read_bytes(),b'old')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    successful = result.wasSuccessful() and not result.skipped and result.testsRun == 19
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        hashes = {path: d.sha((ROOT / path).read_bytes()) for path in (
            'research/direct_fit/result_delivery.py', 'tests/test_direct_fit_delivery.py')}
        args.report.write_text(json.dumps({'schema': 'direct-fit-delivery-software-fixture-v1',
            'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'successful': successful, 'source_sha256': hashes,
            'real_audio_training_calls': 0, 'tts_calls': 0, 'asr_calls': 0,
            'external_service_calls': 0, 'persistence_proven': False,
            'scope': 'artificial bytes, subprocess terminal states and local restoration only'}, indent=2) + '\n')
    raise SystemExit(0 if successful else 1)
