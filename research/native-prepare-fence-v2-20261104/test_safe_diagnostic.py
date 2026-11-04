"""Invented stdlib fixtures only; no model or remote execution."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import safe_diagnostic as d

class DiagnosticTests(unittest.TestCase):
    def successful(self,root):
        with d.StageTracker(root,'sensevoice') as tracker:
            for stage in d.STAGES[1:-1]:tracker.enter(stage)
            tracker.finish()
        return root/d.FILENAME
    def test_success_requires_explicit_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.successful(Path(tmp));summary=d.read_summary(path,'sensevoice')
            self.assertEqual(summary,dict(schema='native-prepare-safe-summary-v1',family='sensevoice',
                status='complete',stage='complete',exception_class=None))
    def test_fixed_stage_and_error_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);error=RuntimeError('/secret/path token=PRIVATE')
            with self.assertRaises(RuntimeError) as caught:
                with d.StageTracker(root,'sensevoice') as tracker:
                    tracker.enter('verify_sources');raise error
            self.assertIs(caught.exception,error)
            raw=(root/d.FILENAME).read_text();self.assertNotIn('PRIVATE',raw);self.assertNotIn('/secret',raw)
            self.assertEqual(d.read_summary(root/d.FILENAME,'sensevoice')['exception_class'],'RuntimeError')
    def test_exception_text_never_accessed(self):
        class SecretError(Exception):
            def __str__(self):raise AssertionError('must not inspect error')
            def __repr__(self):raise AssertionError('must not inspect error')
        self.assertEqual(d.exception_class(SecretError('secret')),'OtherException')
    def test_fake_builtin_name_is_other(self):
        fake=type('RuntimeError',(Exception,),{})
        self.assertEqual(d.exception_class(fake()),'OtherException')
    def test_all_exception_types_have_fixed_names(self):
        for cls in d.ERROR_TYPES:self.assertEqual(d.exception_class(cls()),cls.__name__)
    def test_keyboard_interrupt_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                with d.StageTracker(Path(tmp),'qwen06'):raise KeyboardInterrupt('secret')
            self.assertEqual(d.read_summary(Path(tmp)/d.FILENAME,'qwen06')['exception_class'],'KeyboardInterrupt')
    def test_write_failure_never_masks_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            error=RuntimeError('original')
            original=d._write;calls=[]
            def write(path,value):
                calls.append(value['status'])
                if value['status']=='failed':raise OSError('private')
                return original(path,value)
            with patch.object(d,'_write',side_effect=write):
                with self.assertRaises(RuntimeError) as caught:
                    with d.StageTracker(Path(tmp),'qwen06'):raise error
            self.assertIs(caught.exception,error);self.assertEqual(calls,['running','failed'])
    def test_failure_after_finish_request_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):
                with d.StageTracker(root,'qwen06') as tracker:
                    for stage in d.STAGES[1:-1]:tracker.enter(stage)
                    tracker.finish();raise RuntimeError('late failure')
            self.assertEqual(d.read_summary(root/d.FILENAME,'qwen06')['status'],'failed')
    def test_missing_finish_cannot_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                with d.StageTracker(root,'qwen06'):pass
            self.assertEqual(d.read_summary(root/d.FILENAME,'qwen06')['status'],'failed')
    def test_no_restart_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.successful(root);before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                with d.StageTracker(root,'sensevoice'):pass
            self.assertEqual(path.read_bytes(),before)
    def test_unordered_or_unknown_stage_refused(self):
        for bad in ('entry','/private/path'):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    with d.StageTracker(Path(tmp),'qwen06') as tracker:tracker.enter(bad)
    def test_early_finish_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with d.StageTracker(Path(tmp),'qwen06') as tracker:tracker.finish()
    def test_partial_running_record_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with d.StageTracker(root,'qwen06') as tracker:
                self.assertEqual(d.read_summary(root/d.FILENAME,'qwen06')['reason'],'not_terminal')
                for stage in d.STAGES[1:-1]:tracker.enter(stage)
                tracker.finish()
    def test_missing_file_fixed_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(d.read_summary(Path(tmp)/'missing','qwen06')['reason'],'missing')
    def test_alias_oversize_and_empty_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=self.successful(root);alias=root/'alias';alias.symlink_to(original)
            self.assertEqual(d.read_summary(alias,'sensevoice')['reason'],'unsafe_file')
            alias.unlink();os.link(original,alias)
            self.assertEqual(d.read_summary(original,'sensevoice')['reason'],'unsafe_file')
            alias.unlink()
            for raw in (b'',b'x'*(d.MAX_BYTES+1)):
                original.write_bytes(raw)
                self.assertEqual(d.read_summary(original,'sensevoice')['reason'],'unsafe_file')
    def test_strict_json_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.successful(root);valid=json.loads(path.read_bytes())
            malformed=[b'{',b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b'\xff',b'[]']
            for change in [dict(extra='SECRET'),dict(family='qwen06'),dict(exception_class='secret'),
                           dict(status='success'),dict(stage='/path'),dict(started_stages=['complete'])]:
                value=dict(valid);value.update(change);malformed.append(json.dumps(value).encode())
            for raw in malformed:
                path.write_bytes(raw);summary=d.read_summary(path,'sensevoice')
                self.assertEqual(summary['reason'],'invalid');self.assertNotIn('SECRET',str(summary))
    def test_finish_requires_all_core_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with d.StageTracker(Path(tmp),'qwen06') as tracker:
                    tracker.enter('write_preparation_proof');tracker.finish()

if __name__=='__main__':unittest.main()
