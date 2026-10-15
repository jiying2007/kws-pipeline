"""Offline tamper tests. These do not establish model truth or audio accuracy."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verifier',ROOT/'verify_observation.py')
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class TamperTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        self.root=self.repo/'research/kw1-preset-supplement-observation-v1'
        shutil.copytree(ROOT,self.root)
        workflow=self.repo/'.github/workflows/kw1-supplement-observation.yml'
        workflow.parent.mkdir(parents=True)
        shutil.copyfile(ROOT.parents[1]/'.github/workflows/kw1-supplement-observation.yml',workflow)
    def rewrite(self,name,func,rehash=True):
        p=self.root/name
        d=json.loads(p.read_text());func(d)
        p.write_text(json.dumps(d,ensure_ascii=False))
        if rehash:
            m=self.root/'source-retention.json';r=json.loads(m.read_text())
            r['files'][name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            m.write_text(json.dumps(r))
    def test_pristine(self):
        self.assertTrue(v.verify(self.root)['verified'])
    def test_changed_summary_rehashed(self):
        self.rewrite('observation.json',lambda d:d['results'].update(candidate_ids=[]))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_changed_raw_report_rehashed(self):
        self.rewrite('report.json',lambda d:d.update(training_admitted=True))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_wrong_terminal_state_rehashed(self):
        self.rewrite('observation.json',lambda d:d['run'].update(status='in_progress',conclusion=None))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_wrong_source_rehashed(self):
        self.rewrite('observation.json',lambda d:d['source'].update(commit='0'*40))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_wrong_config_rehashed(self):
        self.rewrite('observation.json',lambda d:d['source'].update(config_sha256='0'*64))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_wrong_artifact_rehashed(self):
        self.rewrite('observation.json',lambda d:d['artifact'].update(id=1))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_missing_raw_file(self):
        (self.root/'ledger.json').unlink()
        with self.assertRaises(OSError):v.verify(self.root)
    def test_duplicate_json_key(self):
        (self.root/'source-retention.json').write_text('{"files":{},"files":{}}')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_changed_workflow(self):
        (self.repo/'.github/workflows/kw1-supplement-observation.yml').write_text('changed')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_bad_zip(self):
        p=self.repo/'bad.zip';p.write_bytes(b'bad')
        with self.assertRaises(ValueError):v.verify(self.root,p)

if __name__=='__main__':unittest.main()
