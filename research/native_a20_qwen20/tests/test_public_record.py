"""Invented mutations and saved-JSON checks only; no model/audio/process execution."""
import gzip
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('A20_QWEN20_DATA_ROOT',str(ROOT)))
SPEC=importlib.util.spec_from_file_location('public_validate',ROOT/'validate.py')
v=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v)
class PublicRecordTests(unittest.TestCase):
    def test_full_offline_saved_evidence(self):
        with mock.patch('subprocess.Popen',side_effect=AssertionError('no process')):
            result=v.validate(ROOT,DATA)
        self.assertEqual(result['status'],'PASS_SAVED_EVIDENCE_ONLY')
        self.assertEqual(result['original_run_status'],'FAILED_NO_RETRY')
    def test_relocation(self):
        with tempfile.TemporaryDirectory() as d:
            code=Path(d)/'code';data=Path(d)/'data';shutil.copytree(ROOT,code);shutil.copytree(DATA,data)
            self.assertEqual(v.validate(code,data)['events'],9)
    def test_missing_changed_extra_and_symlink(self):
        for kind in ('missing','changed','extra','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                data=Path(d)/'copy';shutil.copytree(DATA,data);p=data/'evidence/resource-record.json'
                if kind=='missing':p.unlink()
                if kind=='changed':p.write_bytes(p.read_bytes()+b' ')
                if kind=='extra':(data/'extra').write_text('x')
                if kind=='symlink':(data/'extra').symlink_to(p)
                with self.assertRaises(v.Invalid):v.validate(ROOT,data)
    def test_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('','/tmp/x','../x','a/../x','a//b','./x','a\\b'):
                with self.subTest(name=name),self.assertRaises(v.Invalid):v.safe_path(Path(d),name)
    def test_duplicate_json_key(self):
        with self.assertRaises(v.Invalid):v.parse(b'{"a":1,"a":2}')
    def test_nonfinite_json(self):
        for value in ('NaN','Infinity','-Infinity'):
            with self.subTest(value=value),self.assertRaises(v.Invalid):v.parse(value)
    def test_truncated_gzip(self):
        blob=(DATA/'evidence/raw.jsonl.gz').read_bytes()
        with self.assertRaises(v.Invalid):v.raw_bytes(blob[:-1])
    def test_trailing_gzip_data(self):
        blob=(DATA/'evidence/raw.jsonl.gz').read_bytes()
        with self.assertRaises(v.Invalid):v.raw_bytes(blob+b'extra')
    def test_concatenated_gzip_members(self):
        blob=(DATA/'evidence/raw.jsonl.gz').read_bytes()
        with self.assertRaises(v.Invalid):v.raw_bytes(blob+gzip.compress(b'x',mtime=0))
    def test_bounded_decompression(self):
        with self.assertRaises(v.Invalid):v.raw_bytes(gzip.compress(b'x'*(v.RAW_BYTES+1),mtime=0))
    def test_same_size_wrong_raw(self):
        raw=gzip.decompress((DATA/'evidence/raw.jsonl.gz').read_bytes());wrong=raw.replace(b'"pid":7',b'"pid":8',1)
        self.assertNotEqual(raw,wrong);self.assertEqual(len(raw),len(wrong))
        with self.assertRaises(v.Invalid):v.raw_bytes(gzip.compress(wrong,mtime=0))
    def test_unbound_annotations_rejected(self):
        manifest=v.parse((DATA/'metadata/inputs.public.json').read_bytes())
        with self.assertRaises(v.Invalid):v.reconstruct_original_manifest(manifest,b'{}\n')
    def test_greedy_blank_order(self):
        mod=v.load_module(ROOT/'src/derive_saved.py','toy_derive')
        self.assertEqual(mod.collapse([3,3,0,3,3,4]),[3,3,4]);self.assertEqual(mod.collapse([0,0]),[])
    def archived_launcher(self):
        source=(ROOT/'archive/launch_once.py.txt').read_text();m=types.ModuleType('archival_launcher');m.__file__=str(ROOT/'archive/launch_once.py.txt')
        exec(compile(source,'archival_launcher','exec'),m.__dict__);return m
    def test_original_supervisor_failure_with_mocks(self):
        m=self.archived_launcher()
        with mock.patch.object(Path,'read_text',side_effect=['VmRSS: 1 kB\nThreads: 1\n',PermissionError(13,'invented denied','/proc/7/io')]):
            with self.assertRaises(PermissionError):m.read_proc(7)
    def test_original_missing_children_stays_unknown(self):
        m=self.archived_launcher()
        with mock.patch.object(Path,'read_text',side_effect=['VmRSS: 1 kB\nThreads: 1\n','rchar: 0\n',FileNotFoundError('invented')]):observed=m.read_proc(7)
        self.assertNotIn('children',observed)
    def test_projection_does_not_mutate_raw(self):
        before=(DATA/'evidence/raw.jsonl.gz').read_bytes();v.validate(ROOT,DATA)
        self.assertEqual((DATA/'evidence/raw.jsonl.gz').read_bytes(),before)
    def test_projection_never_claims_acoustic_pass(self):
        p=v.parse((DATA/'metadata/protocol.public.json').read_bytes())
        self.assertEqual(p['status'],'PUBLIC_TECHNICAL_PROJECTION_OF_FROZEN_PROTOCOL_NOT_EXECUTED')
        self.assertFalse(p['numerical_qualification']);self.assertFalse(p['independent_qualified_metrics_allowed'])
if __name__=='__main__':unittest.main()
