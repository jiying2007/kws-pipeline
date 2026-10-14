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
        added = file('.github/workflows/research-source-consolidation.yml', b'name: offline\n')
        admission = file('.github/workflows/qwen12-native-admission-v1.yml', b'name: disabled native; pure source CI\n')
        asr18 = file('.github/workflows/k1-punctuation6-asr18-v1.yml', b'name: disabled asr18; pure source CI\n')
        asr18_r4 = file('.github/workflows/k1-punctuation6-asr18-r4.yml', b'name: synthetic current asr18 r4\n')
        asr18_r5 = file('.github/workflows/k1-punctuation6-asr18-r5.yml', b'name: synthetic current asr18 r5\n')
        prepare = file('.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml', b'name: disabled prepare diagnostic; pure source CI\n')
        transfer = file('.github/workflows/k1-evidence-transfer-qualification-v1.yml', b'name: zero-model transfer qualification\n')
        atomic_claim = file('.github/workflows/k1-atomic-claim-qualification-v1.yml', b'name: zero-model atomic claim qualification\n')
        file(ARM, json.dumps(DISABLED_ARM).encode())
        provenance = dict(archived, source_commit='a'*40, original_path='.github/workflows/old.yml')
        provenance['source_url'] = 'https://github.com/jiying2007/kws-pipeline/blob/'+'a'*40+'/.github/workflows/old.yml'
        self.manifest = dict(schema='research-source-retention-v1', repository='jiying2007/kws-pipeline',
            base_commit='b'*40, files=[archived], provenance=[provenance],
            source_projections=[dict(original_path='.github/workflows/old.yml', archive_path=archived['path'],
                source_commit='a'*40, **{k:archived[k] for k in ('bytes','sha256','git_blob_sha1')})],
            baseline_active_workflows=[baseline], current_added_active_workflows=[added, admission, asr18, prepare, transfer, atomic_claim, asr18_r4, asr18_r5], pointer_only=[],
            allowed_new_active_workflows=['.github/workflows/research-source-consolidation.yml',
                '.github/workflows/qwen12-native-admission-v1.yml',
                '.github/workflows/k1-punctuation6-asr18-v1.yml',
                '.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml',
                '.github/workflows/k1-evidence-transfer-qualification-v1.yml',
                '.github/workflows/k1-atomic-claim-qualification-v1.yml',
                '.github/workflows/k1-punctuation6-asr18-r4.yml',
                '.github/workflows/k1-punctuation6-asr18-r5.yml'])
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
    def test_missing_allowed_workflow_rejected(self):
        (self.root / '.github/workflows/research-source-consolidation.yml').unlink()
        with self.assertRaises(ValueError): verify(self.root, self.manifest)
    def test_current_added_workflow_same_size_byte_mutation_rejected(self):
        path = self.root / '.github/workflows/research-source-consolidation.yml'
        path.write_bytes(b'name: OFFLINE\n')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            verify(self.root, self.manifest)
    def test_current_added_workflow_size_mutation_rejected(self):
        path = self.root / '.github/workflows/research-source-consolidation.yml'
        path.write_bytes(path.read_bytes() + b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'byte count changed'):
            verify(self.root, self.manifest)
    @unittest.skipUnless(__import__('os').name == 'posix', 'POSIX mode contract')
    def test_current_added_workflow_mode_mutation_rejected(self):
        path = self.root / '.github/workflows/research-source-consolidation.yml'
        path.chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'mode changed'):
            verify(self.root, self.manifest)
    def test_current_added_workflow_identity_is_required_and_closed(self):
        original = copy.deepcopy(self.manifest['current_added_active_workflows'])
        for bad in (None, [], original + original,
                    [dict(original[0], path='.github/workflows/base.yml')] + original[1:],
                    [original[0], original[0], original[2]], ['wrong'] + original[1:]):
            with self.subTest(bad=bad):
                self.manifest['current_added_active_workflows'] = bad
                with self.assertRaises(ValueError):
                    verify(self.root, self.manifest)
        del self.manifest['current_added_active_workflows']
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest)
    def test_current_added_workflow_identity_fields_all_bind(self):
        original = copy.deepcopy(self.manifest['current_added_active_workflows'])
        for index, row in enumerate(original):
            for key, value in (('bytes', True), ('bytes', row['bytes'] + 1),
                               ('mode', '100755'), ('mode', '120000'),
                               ('sha256', '0' * 64), ('git_blob_sha1', '0' * 40)):
                with self.subTest(path=row['path'], key=key, value=value):
                    self.manifest['current_added_active_workflows'] = copy.deepcopy(original)
                    self.manifest['current_added_active_workflows'][index][key] = value
                    with self.assertRaisesRegex(ValueError, 'byte count|mode|identity changed'):
                        verify(self.root, self.manifest)
    def test_current_added_workflow_symlink_is_not_an_identity(self):
        path = self.root / '.github/workflows/research-source-consolidation.yml'
        archived = self.root / 'research/consolidation/archive/workflows/current.yml'
        archived.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(archived)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify(self.root, self.manifest)
    def test_historical_workflow_receipts_do_not_replace_current_identity(self):
        current = copy.deepcopy(self.manifest['current_added_active_workflows'][0])
        old = self.file('research/consolidation/archive/workflows/previous.yml', b'name: previous\n')
        self.manifest['active_workflow_maintenance'] = [{'files': [{'path': current['path'], 'before': old, 'after': current}]}]
        self.assertEqual(verify(self.root, self.manifest), 1)
        self.manifest['current_added_active_workflows'][0] = dict(old, path=current['path'])
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest)
    def test_reviewed_current_workflow_update_does_not_rewrite_history(self):
        previous = copy.deepcopy(self.manifest['current_added_active_workflows'][0])
        updated = self.file(previous['path'], b'name: reviewed current source\n')
        self.manifest['current_added_active_workflows'][0] = updated
        self.manifest['active_workflow_maintenance'] = [{'files': [{'path': previous['path'], 'before': previous, 'after': updated}]}]
        self.assertEqual(verify(self.root, self.manifest), 1)
        self.assertNotEqual(previous['sha256'], updated['sha256'])
    def test_non_exact_allowance_rejected(self):
        original = self.manifest['allowed_new_active_workflows']
        for bad in ([], original + original,
                    original + ['.github/workflows/archive-branches-once-20261007.yml'],
                    original + ['.github/workflows/arbitrary.yml']):
            self.manifest['allowed_new_active_workflows'] = bad
            with self.assertRaises(ValueError): verify(self.root, self.manifest)
        self.manifest['allowed_new_active_workflows'] = original
    def test_asr18_workflow_requires_exact_name_and_current_identity(self):
        for index in (2, 6, 7):
            with self.subTest(path=self.manifest['allowed_new_active_workflows'][index]):
                self.check_asr18_workflow_identity(index)

    def check_asr18_workflow_identity(self, index):
        approved = self.manifest['allowed_new_active_workflows'][index]
        path = self.root / approved
        current = copy.deepcopy(self.manifest['current_added_active_workflows'][index])
        raw = path.read_bytes()
        self.assertEqual(verify(self.root, self.manifest), 1)
        for lookalike in ('.github/workflows/k1-punctuation6-asr18-v10.yml',
                          '.github/workflows/k1-punctuation6-asr18-v1-extra.yml',
                          '.github/workflows/k1-punctuation6-asr18-v2.yml',
                          '.github/workflows/k1-punctuation6-asr18-r6.yml'):
            with self.subTest(path=lookalike):
                replacement = self.file(lookalike, raw)
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    verify(self.root, self.manifest)
                path.unlink()
                self.manifest['allowed_new_active_workflows'][index] = lookalike
                self.manifest['current_added_active_workflows'][index] = replacement
                with self.assertRaisesRegex(ValueError, 'unexpected active workflow addition'):
                    verify(self.root, self.manifest)
                (self.root / lookalike).unlink()
                path.write_bytes(raw)
                self.manifest['allowed_new_active_workflows'][index] = approved
                self.manifest['current_added_active_workflows'][index] = current
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing retained file'):
            verify(self.root, self.manifest)
        path.write_bytes(raw)
    def test_prepare_workflow_requires_exact_name_and_current_identity(self):
        approved = '.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml'
        path = self.root / approved
        current = copy.deepcopy(self.manifest['current_added_active_workflows'][3])
        raw = path.read_bytes()
        self.assertEqual(verify(self.root, self.manifest), 1)
        for lookalike in ('.github/workflows/k1-punctuation6-prepare-diagnostic-v10.yml',
                          '.github/workflows/k1-punctuation6-prepare-diagnostic-v1-extra.yml',
                          '.github/workflows/k1-punctuation6-prepare-diagnostic-v2.yml'):
            with self.subTest(path=lookalike):
                replacement = self.file(lookalike, raw)
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    verify(self.root, self.manifest)
                path.unlink()
                self.manifest['allowed_new_active_workflows'][3] = lookalike
                self.manifest['current_added_active_workflows'][3] = replacement
                with self.assertRaisesRegex(ValueError, 'unexpected active workflow addition'):
                    verify(self.root, self.manifest)
                (self.root / lookalike).unlink()
                path.write_bytes(raw)
                self.manifest['allowed_new_active_workflows'][3] = approved
                self.manifest['current_added_active_workflows'][3] = current
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing retained file'):
            verify(self.root, self.manifest)
    def test_transfer_workflow_requires_exact_name_and_current_identity(self):
        approved = '.github/workflows/k1-evidence-transfer-qualification-v1.yml'
        path = self.root / approved
        current = copy.deepcopy(self.manifest['current_added_active_workflows'][4])
        raw = path.read_bytes()
        self.assertEqual(verify(self.root, self.manifest), 1)
        for lookalike in ('.github/workflows/k1-evidence-transfer-qualification-v10.yml',
                          '.github/workflows/k1-evidence-transfer-qualification-v1-extra.yml',
                          '.github/workflows/k1-evidence-transfer-qualification-v2.yml'):
            with self.subTest(path=lookalike):
                replacement = self.file(lookalike, raw)
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    verify(self.root, self.manifest)
                path.unlink()
                self.manifest['allowed_new_active_workflows'][4] = lookalike
                self.manifest['current_added_active_workflows'][4] = replacement
                with self.assertRaisesRegex(ValueError, 'unexpected active workflow addition'):
                    verify(self.root, self.manifest)
                (self.root / lookalike).unlink()
                path.write_bytes(raw)
                self.manifest['allowed_new_active_workflows'][4] = approved
                self.manifest['current_added_active_workflows'][4] = current
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing retained file'):
            verify(self.root, self.manifest)
    def test_atomic_claim_workflow_requires_exact_name_and_current_identity(self):
        approved = '.github/workflows/k1-atomic-claim-qualification-v1.yml'
        path = self.root / approved
        current = copy.deepcopy(self.manifest['current_added_active_workflows'][5])
        raw = path.read_bytes()
        self.assertEqual(verify(self.root, self.manifest), 1)
        for lookalike in ('.github/workflows/k1-atomic-claim-qualification-v10.yml',
                          '.github/workflows/k1-atomic-claim-qualification-v1-extra.yml',
                          '.github/workflows/k1-atomic-claim-qualification-v2.yml'):
            with self.subTest(path=lookalike):
                replacement = self.file(lookalike, raw)
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    verify(self.root, self.manifest)
                path.unlink()
                self.manifest['allowed_new_active_workflows'][5] = lookalike
                self.manifest['current_added_active_workflows'][5] = replacement
                with self.assertRaisesRegex(ValueError, 'unexpected active workflow addition'):
                    verify(self.root, self.manifest)
                (self.root / lookalike).unlink()
                path.write_bytes(raw)
                self.manifest['allowed_new_active_workflows'][5] = approved
                self.manifest['current_added_active_workflows'][5] = current
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing retained file'):
            verify(self.root, self.manifest)
    def test_host_verifier_sources_do_not_expand_retained_research_scope(self):
        row = self.file('host-tts-verifier-source/synthetic.py', b'# synthetic\n')
        self.manifest['files'].append(row)
        with self.assertRaisesRegex(ValueError, 'outside isolated research tree'):
            verify(self.root, self.manifest)
    def test_retired_workflow_cannot_be_reactivated(self):
        self.file('.github/workflows/archive-branches-once-20261007.yml', b'name: retired\n')
        with self.assertRaises(ValueError): verify(self.root, self.manifest)
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



# Narrow current-revision maintenance tests; original retention suite above stays intact.
import os
import verify_research_sources as HOTPATH_RETENTION
from run_research_source_checks import make_projection as hotpath_make_projection
HOTPATH_ROOT = Path(__file__).resolve().parents[3]


def hotpath_identity(path, raw, mode='100644'):
    return {'path': path, 'mode': mode, 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


class SourceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='retention-revision-offline-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = json.loads((HOTPATH_ROOT / HOTPATH_RETENTION.MANIFEST).read_text())
        self.manifest.pop('current_source_revision', None)
        old_path = HOTPATH_ROOT / HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE
        if not old_path.is_file():
            old_path = HOTPATH_ROOT / HOTPATH_RETENTION.CURRENT_REVISION_PATH
        self.old_raw = old_path.read_bytes()
        self.assertEqual(hotpath_identity(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.old_raw), HOTPATH_RETENTION.ORIGINAL_SOURCE)
        # This is explicitly synthetic changed text; no scientific behavior or
        # production identity is inferred from it.
        self.current_raw = self.old_raw + b'\n# SYNTHETIC TEST REVISION ONLY\n'
        self.revision = {'revision_id': HOTPATH_RETENTION.CURRENT_REVISION_ID,
            'path': HOTPATH_RETENTION.CURRENT_REVISION_PATH,
            'archive_path': HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE,
            'supersedes_commit': HOTPATH_RETENTION.CURRENT_REVISION_SUPERSEDES,
            'original': copy.deepcopy(HOTPATH_RETENTION.ORIGINAL_SOURCE),
            'original_provenance': [copy.deepcopy(row) for row in self.manifest['provenance']
                if row['path'] == HOTPATH_RETENTION.CURRENT_REVISION_PATH],
            'current': hotpath_identity(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.current_raw)}
        self.manifest['current_source_revision'] = self.revision
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE, self.old_raw)
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.current_raw)

    def write(self, name, raw, mode='100644'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o755 if mode == '100755' else 0o644)
        return path

    def check(self):
        return HOTPATH_RETENTION.validate_current_source_revision(self.root, self.manifest)

    def populate_complete_fixture(self):
        for group in ('files', 'baseline_active_workflows', 'current_added_active_workflows'):
            for row in self.manifest[group]:
                if row['path'] == HOTPATH_RETENTION.CURRENT_REVISION_PATH:
                    continue
                self.write(row['path'], (HOTPATH_ROOT / row['path']).read_bytes(), row['mode'])
        self.write(HOTPATH_RETENTION.ARM, (HOTPATH_ROOT / HOTPATH_RETENTION.ARM).read_bytes())

    def test_current_and_archive_are_independently_exact(self):
        self.assertEqual(self.check(), self.revision)
        self.assertEqual(HOTPATH_RETENTION.retained_source_path(self.manifest, HOTPATH_RETENTION.CURRENT_REVISION_PATH),
            HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE)
        other = 'research/fixed30_asr/core/vendor/qwen_loading_gate_v2.py'
        self.assertEqual(HOTPATH_RETENTION.retained_source_path(self.manifest, other), other)

    def test_complete_inventory_stays_718_and_historical_projection_path_keeps_original_bytes(self):
        self.populate_complete_fixture()
        self.assertEqual(HOTPATH_RETENTION.verify(self.root, self.manifest), 718)
        projected_source = self.root / HOTPATH_RETENTION.retained_source_path(self.manifest, HOTPATH_RETENTION.CURRENT_REVISION_PATH)
        self.assertEqual(projected_source.read_bytes(), self.old_raw)
        self.assertNotEqual(projected_source.read_bytes(), (self.root / HOTPATH_RETENTION.CURRENT_REVISION_PATH).read_bytes())

    def test_original_without_revision_retains_existing_strict_behavior(self):
        self.populate_complete_fixture()
        del self.manifest['current_source_revision']
        with self.assertRaises(ValueError):
            HOTPATH_RETENTION.verify(self.root, self.manifest)
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.old_raw)
        self.assertEqual(HOTPATH_RETENTION.verify(self.root, self.manifest), 718)

    def test_no_generic_path_or_archive_override(self):
        for key in ('path', 'archive_path'):
            for value in ('research/other.py', '../escape.py', '/absolute.py', './noncanonical.py',
                          'research//doubled.py', 'research\\backslash.py'):
                with self.subTest(key=key, value=value):
                    original = self.revision[key]
                    self.revision[key] = value
                    with self.assertRaisesRegex(ValueError, 'fixed scope'):
                        self.check()
                    self.revision[key] = original

    def test_revision_identity_and_superseded_commit_are_fixed(self):
        for key in ('revision_id', 'supersedes_commit'):
            old = self.revision[key]
            self.revision[key] = 'a' * 40
            with self.assertRaisesRegex(ValueError, 'fixed scope'):
                self.check()
            self.revision[key] = old

    def test_missing_or_additional_revision_fields(self):
        saved = copy.deepcopy(self.revision)
        for key in saved:
            self.revision.pop(key)
            with self.assertRaisesRegex(ValueError, 'revision fields'):
                self.check()
            self.revision[key] = saved[key]
        self.revision['source_commit'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'revision fields'):
            self.check()

    def test_no_null_list_or_multiple_revisions(self):
        for value in (None, [], [self.revision], [self.revision, self.revision], 'revision'):
            self.manifest['current_source_revision'] = value
            with self.assertRaises(ValueError):
                self.check()

    def test_missing_or_changed_original_archive_rejected(self):
        path = self.root / HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing retained file'):
            self.check()
        path.write_bytes(self.old_raw + b'changed')
        with self.assertRaises(ValueError):
            self.check()
        path.write_bytes(bytes([self.old_raw[0] ^ 1]) + self.old_raw[1:])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.check()

    def test_current_tamper_is_not_hidden_by_valid_archive(self):
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.current_raw + b'changed')
        with self.assertRaisesRegex(ValueError, 'byte count'):
            self.check()

    def test_current_sha256_and_git_blob_are_both_checked(self):
        for key, fake in (('sha256', '0' * 64), ('git_blob_sha1', '0' * 40)):
            old = self.revision['current'][key]
            self.revision['current'][key] = fake
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.check()
            self.revision['current'][key] = old

    def test_current_identity_shape_types_and_mode_are_exact(self):
        for key, bad in (('bytes', True), ('bytes', -1), ('sha256', None), ('sha256', 'x' * 64),
                         ('git_blob_sha1', 'main'), ('mode', '100755'), ('path', 'research/other.py')):
            old = self.revision['current'][key]
            self.revision['current'][key] = bad
            with self.assertRaisesRegex(ValueError, 'exact identity fields'):
                self.check()
            self.revision['current'][key] = old
        self.revision['current']['source_commit'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'exact identity fields'):
            self.check()

    def test_old_bytes_cannot_be_claimed_as_current_revision(self):
        self.revision['current'] = dict(HOTPATH_RETENTION.ORIGINAL_SOURCE)
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_PATH, self.old_raw)
        with self.assertRaisesRegex(ValueError, 'distinct derived bytes'):
            self.check()

    def test_complete_original_row_cannot_be_rewritten(self):
        for key in self.revision['original']:
            old = self.revision['original'][key]
            self.revision['original'][key] = None
            with self.assertRaisesRegex(ValueError, 'original identity changed'):
                self.check()
            self.revision['original'][key] = old

    def test_all_original_provenance_relations_are_retained(self):
        self.revision['original_provenance'].pop()
        with self.assertRaisesRegex(ValueError, 'original provenance changed'):
            self.check()

    def test_no_change_to_any_of_718_rows_or_provenance(self):
        for group in ('files', 'provenance'):
            original = copy.deepcopy(self.manifest[group])
            for mutation in ('hash', 'remove', 'duplicate', 'order'):
                with self.subTest(group=group, mutation=mutation):
                    rows = self.manifest[group]
                    if mutation == 'hash': rows[-1]['sha256'] = '0' * 64
                    if mutation == 'remove': rows.pop()
                    if mutation == 'duplicate': rows.append(copy.deepcopy(rows[-1]))
                    if mutation == 'order': rows.reverse()
                    with self.assertRaisesRegex(ValueError, 'original 718 inventory/provenance changed'):
                        self.check()
                    self.manifest[group] = copy.deepcopy(original)

    def test_symlink_is_not_archival_retention_or_current_identity(self):
        for name in (HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE, HOTPATH_RETENTION.CURRENT_REVISION_PATH):
            path = self.root / name
            raw = path.read_bytes()
            target = self.root / 'symlink-target'
            target.write_bytes(raw)
            path.unlink()
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.check()
            path.unlink()
            path.write_bytes(raw)

    @unittest.skipUnless(os.name == 'posix', 'POSIX mode check')
    def test_archive_or_current_mode_change_fails(self):
        for name in (HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE, HOTPATH_RETENTION.CURRENT_REVISION_PATH):
            path = self.root / name
            path.chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'mode changed'):
                self.check()
            path.chmod(0o644)


    def test_actual_projection_reads_original_archive_instead_of_current_vendor(self):
        self.populate_complete_fixture()
        core_name = 'research/consolidation/core-2026-10-07.json'
        core = json.loads((HOTPATH_ROOT / core_name).read_text())
        self.write(core_name, (HOTPATH_ROOT / core_name).read_bytes())
        support = ('research/consolidation/source_tests/test_n1_companion_offline.py',
            'research/consolidation/prepare_n1_saved_fixtures.py',
            'research/consolidation/source_fixtures/n1-saved-inputs.json')
        for row in core['files']:
            self.write(row['path'], (HOTPATH_ROOT / row['path']).read_bytes(), row['mode'])
        for name in support:
            self.write(name, (HOTPATH_ROOT / name).read_bytes())
        self.assertEqual(HOTPATH_RETENTION.verify(self.root, self.manifest), 718)
        projected = self.root / 'projected'
        projected.mkdir()
        hotpath_make_projection(self.root, projected, self.manifest)
        actual = (projected / HOTPATH_RETENTION.CURRENT_REVISION_PATH).read_bytes()
        self.assertEqual(actual, self.old_raw)
        self.assertNotEqual(actual, self.current_raw)
        self.assertEqual(hotpath_identity(HOTPATH_RETENTION.CURRENT_REVISION_PATH, actual),
            HOTPATH_RETENTION.ORIGINAL_SOURCE)

    def test_wrong_archive_cannot_reach_historical_projection(self):
        self.populate_complete_fixture()
        self.write(HOTPATH_RETENTION.CURRENT_REVISION_ARCHIVE, self.current_raw)
        projected = self.root / 'projected'
        projected.mkdir()
        # The runner's actual precondition is full verify() before make_projection().
        with self.assertRaises(ValueError):
            HOTPATH_RETENTION.verify(self.root, self.manifest)
            hotpath_make_projection(self.root, projected, self.manifest)
        self.assertFalse((projected / HOTPATH_RETENTION.CURRENT_REVISION_PATH).exists())


if __name__=='__main__':unittest.main(verbosity=2)
