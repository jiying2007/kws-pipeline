#!/usr/bin/env python3
"""Invented source-projection fixtures. No network, acquisition or model calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
# source_tests -> consolidation -> research -> repository
from verify_research_sources import ARM, DISABLED_ARM, check_file, relative, verify
from run_research_source_checks import selected_checks

class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        def file(name, raw):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            return dict(path=name, mode='100644', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        self.file = file
        archived = file('research/consolidation/archive/workflows/example.yml', b'name: historical\n')
        baseline = file('.github/workflows/base.yml', b'name: unchanged\n')
        file('.github/workflows/research-source-consolidation.yml', b'name: offline\n')
        file(ARM, json.dumps(DISABLED_ARM).encode())
        provenance = dict(archived, source_commit='a'*40, original_path='.github/workflows/old.yml')
        provenance['source_url'] = 'https://github.com/jiying2007/kws-pipeline/blob/'+'a'*40+'/.github/workflows/old.yml'
        self.manifest = dict(schema='research-source-retention-v1', repository='jiying2007/kws-pipeline',
            base_commit='b'*40, files=[archived], provenance=[provenance],
            source_projections=[dict(original_path='.github/workflows/old.yml', archive_path=archived['path'],
                source_commit='a'*40, **{k:archived[k] for k in ('bytes','sha256','git_blob_sha1')})],
            baseline_active_workflows=[baseline], pointer_only=[],
            allowed_new_active_workflows=['.github/workflows/research-source-consolidation.yml'])
    def test_valid_retention(self):
        self.assertEqual(verify(self.root,self.manifest),1)
    def test_changed_bytes(self):
        (self.root/self.manifest['files'][0]['path']).write_text('changed')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_changed_mode(self):
        (self.root/self.manifest['files'][0]['path']).chmod(0o755)
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_duplicate_file(self):
        self.manifest['files']*=2
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_missing_provenance(self):
        self.manifest['provenance']=[]
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_bad_source_binding(self):
        self.manifest['provenance'][0]['source_url']='https://example.invalid'
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_bad_source_hash(self):
        self.manifest['provenance'][0]['git_blob_sha1']='0'*40
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_duplicate_provenance(self):
        self.manifest['provenance']*=2
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_changed_projection_identity(self):
        self.manifest['source_projections'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_projection_escape(self):
        self.manifest['source_projections'][0]['original_path']='../old.yml'
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_arbitrary_projection(self):
        self.manifest['source_projections'][0]['original_path']='src/main.c'
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_activated_old_workflow(self):
        self.file('.github/workflows/old.yml', b'name: historical\n')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_unexpected_active_workflow(self):
        self.file('.github/workflows/new-live.yml', b'new')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_changed_baseline_workflow(self):
        (self.root/'.github/workflows/base.yml').write_text('changed')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_rearmed_cosy30(self):
        arm=dict(DISABLED_ARM,mode='run-fixed30-once')
        (self.root/ARM).write_text(json.dumps(arm))
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_pointer_only_audio_not_imported(self):
        self.manifest['pointer_only']=[dict(path=None,kind='source-pointer-only',original_path='research/x/audio.zip')]
        self.assertEqual(verify(self.root,self.manifest),1)
        self.file('research/x/audio.zip',b'not real audio')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_symlink_rejected(self):
        p=self.root/self.manifest['files'][0]['path'];p.unlink();p.symlink_to(self.root/'.github/workflows/base.yml')
        with self.assertRaises(ValueError):verify(self.root,self.manifest)
    def test_noncanonical_paths(self):
        for name in ['../x','/x','a/../x','a//x','a\\x','',None]:
            with self.subTest(name=name),self.assertRaises(ValueError):relative(name)

class AllowlistTests(unittest.TestCase):
    def row(self, **kw):
        return dict(id='invented',cwd='.',argv=['python3','-B','research/example/test.py'],requirements=[],scope='offline_synthetic_or_static',**kw)
    def test_one_exact_command(self):
        self.assertEqual(len(selected_checks([self.row()], 'synthetic')),1)
    def test_unknown_id(self):
        with self.assertRaises(ValueError):selected_checks([self.row()],'synthetic',['absent'])
    def test_duplicate_id(self):
        with self.assertRaises(ValueError):selected_checks([self.row(),self.row()],'synthetic')
    def test_saved_requires_explicit_suite(self):
        r=self.row();r['requirements']=['PREPARED_FIXTURE: invented numeric values']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
        self.assertEqual(selected_checks([r],'saved'),[r])
    def test_no_generic_test_discovery(self):
        r=self.row();r['argv']=['python3','-m','unittest','discover','-s','research']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_wildcard_discovery(self):
        r=self.row();r['argv']=['python3','-m','unittest','discover','-s','research','-p','test_*.py']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_execution_flag(self):
        r=self.row();r['argv']+=['--execute-release','release.json']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_shell(self):
        r=self.row();r['argv']=['sh','-c','true']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_env_shell_escape(self):
        r=self.row();r['argv']=['env','bash','-c','true']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_env_path_injection(self):
        r=self.row();r['argv']=['env','PATH=../offline-fixtures/bin','python3','-V']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_no_fixture_env_traversal(self):
        r=self.row();r['argv']=['env','A20_N0_DATA_ROOT=../offline-fixtures/../../outside','python3','-V']
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_fixture_env_python(self):
        r=self.row();r['argv']=['env','A20_N0_DATA_ROOT=../offline-fixtures/n0','python3','-B','research/example/test.py']
        self.assertEqual(selected_checks([r],'synthetic'),[r])
    def test_no_report_filename_escape(self):
        for name in ['../bad','/bad','.hidden','a/b','a\\b','name with spaces','']:
            r=self.row();r['id']=name
            with self.subTest(name=name),self.assertRaises(ValueError):selected_checks([r],'synthetic')
    def test_cwd_scope(self):
        r=self.row();r['cwd']='../outside'
        with self.assertRaises(ValueError):selected_checks([r],'synthetic')

if __name__=='__main__':unittest.main(verbosity=2)
