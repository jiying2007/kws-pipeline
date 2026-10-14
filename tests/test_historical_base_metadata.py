import unittest
import tempfile
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_historical_base_metadata import check, FIELDS, audit_corpus


def row(split):
    return {'split': split, 'speech_like_provenance': {f: split+'-'+f for f in FIELDS}}


class Checks(unittest.TestCase):
    def test_voice_never_fills_missing_speaker(self):
        a,b=row('train'),row('calibration')
        for r in (a,b):
            del r['speech_like_provenance']['speaker_id']
            del r['speech_like_provenance']['session_id']
        result=check({'train':[a],'calibration':[b]})
        self.assertFalse(result['identity_metadata_complete_and_disjoint'])
        self.assertEqual(result['coverage']['train']['missing']['speaker_id'],1)
        self.assertFalse(result['runner_ready'])

    def test_overlap_detected(self):
        a,b=row('train'),row('calibration')
        b['speech_like_provenance']['source_family_id']=a['speech_like_provenance']['source_family_id']
        result=check({'train':[a],'calibration':[b]})
        self.assertFalse(result['identity_metadata_complete_and_disjoint'])

    def test_holdouts_and_mismatched_splits_rejected(self):
        with self.assertRaises(ValueError):check({'train':[], 'test':[]})
        with self.assertRaises(ValueError):check({'train':[row('qualification')],'calibration':[]})

    def test_empty_or_placeholder_metadata_never_complete(self):
        self.assertFalse(check({'train': [], 'calibration': []})['identity_metadata_complete_and_disjoint'])
        a,b=row('train'),row('calibration')
        a['speech_like_provenance']['speaker_id']='unknown-voice0'
        self.assertFalse(check({'train':[a],'calibration':[b]})['identity_metadata_complete_and_disjoint'])

    def test_complete_metadata_still_not_runner_admission(self):
        result=check({'train':[row('train')],'calibration':[row('calibration')]})
        self.assertTrue(result['identity_metadata_complete_and_disjoint'])
        self.assertFalse(result['runner_ready'])


class BoundaryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corpus'
        for split in ('train', 'calibration'):
            directory = self.root / 'bundle' / split
            directory.mkdir(parents=True)
            (directory / 'dataset-index.jsonl').write_text(json.dumps(row(split)) + '\n')
            (directory / 'dataset-summary.json').write_text(json.dumps({'recordings': 1}))
        self.target = Path(self.temp.name) / 'fixture-metadata'
        self.target.write_text('MUST NOT PARSE THIS FIXTURE')

    def assert_boundary_rejected(self, root=None):
        with self.assertRaisesRegex(ValueError, 'metadata boundary:'):
            audit_corpus(root or self.root)

    def test_regular_selected_files_only(self):
        result = audit_corpus(self.root)
        self.assertEqual(result['protected_splits_opened'], [])
        self.assertFalse(result['expected_counts_match'])

    def test_index_symlink_rejected_before_target_read(self):
        path = self.root / 'bundle/train/dataset-index.jsonl'
        path.unlink(); path.symlink_to(self.target)
        self.assert_boundary_rejected()

    def test_summary_symlink_rejected_before_target_read(self):
        path = self.root / 'bundle/train/dataset-summary.json'
        path.unlink(); path.symlink_to(self.target)
        self.assert_boundary_rejected()

    def test_split_parent_symlink_rejected(self):
        path = self.root / 'bundle/train'
        moved = self.root / 'bundle/fixture-directory'
        path.rename(moved); path.symlink_to(moved, target_is_directory=True)
        self.assert_boundary_rejected()

    def test_bundle_parent_symlink_rejected(self):
        path = self.root / 'bundle'
        moved = self.root / 'fixture-bundle'
        path.rename(moved); path.symlink_to(moved, target_is_directory=True)
        self.assert_boundary_rejected()

    def test_corpus_parent_symlink_rejected(self):
        alias = Path(self.temp.name) / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        self.assert_boundary_rejected(alias)

if __name__=='__main__':unittest.main()
