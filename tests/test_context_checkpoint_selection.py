#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'training'), str(ROOT/'tests')]
import select_context_checkpoint as selection
from frozen_speech_ablation import materialize_pool, sha, write, POLICY, verify_pool
from test_frozen_speech_ablation import fixture, KEYWORDS
from startup_context_training import train
import torch


def metric(matches=(8, 8), fa=0):
    return {'expected':16, 'matched':sum(matches), 'false_rejects':16-sum(matches),'false_accepts':fa,
        'per_keyword':{str(i+1):{'expected':8,'matched':n,'false_rejects':8-n} for i,n in enumerate(matches)}}


def candidate(epoch=100, matches=(8,8), fa=0):
    return {'epoch':epoch,'metrics':{'train':{a:metric() for a in selection.ARMS},
        'calibration':{a:metric(matches,fa) for a in selection.ARMS}}}


class SelectionPolicyTests(unittest.TestCase):
    def test_false_accepts_rank_before_loss_or_latest_epoch(self):
        early=candidate(200,(7,7),0);final=candidate(600,(8,8),3)
        self.assertIs(selection.select([final,early]),early)

    def test_failed_train_fit_cannot_win_with_silent_calibration(self):
        bad=candidate();bad['metrics']['train']['continuous']=metric((0,0),0)
        good=candidate(200,(7,7),1)
        self.assertIs(selection.select([bad,good]),good)

    def test_per_keyword_recall_floor_not_aggregate(self):
        bad=candidate(matches=(8,5));good=candidate(200,matches=(6,6))
        self.assertIs(selection.select([bad,good]),good)

    def test_no_eligible_candidate_is_explicit_none(self):
        self.assertIsNone(selection.select([candidate(matches=(0,0))]))

    def test_prior_nonwake_training_errors_are_not_ignored(self):
        row=candidate();row['metrics']['train']['prior-nonwake']['false_accepts']=1
        self.assertFalse(selection.eligibility(row['metrics'])[0])

    def test_ties_use_earlier_epoch_deterministically(self):
        a,b=candidate(200),candidate(600)
        self.assertIs(selection.select([b,a]),a)

    def test_missing_arms_or_test_metrics_rejected(self):
        for mut in (lambda r:r['metrics'].update(test={}),
                    lambda r:r['metrics']['calibration'].pop('continuous')):
            r=candidate();mut(r)
            with self.assertRaises(ValueError):selection.select([r])

    def test_bad_counts_and_aggregate_tampering_rejected(self):
        for value in (True,0.5,'0',-1):
            r=candidate();r['metrics']['calibration']['silence-1s']['false_accepts']=value
            with self.subTest(value=value),self.assertRaises(ValueError):selection.select([r])
        r=candidate();r['metrics']['calibration']['continuous']['per_keyword']['1']['matched']=7
        with self.assertRaises(ValueError):selection.select([r])

    def test_duplicate_candidate_epoch_rejected(self):
        with self.assertRaises(ValueError):selection.select([candidate(),candidate()])

    def test_selection_view_rejects_test_before_any_audio_is_loaded(self):
        with mock.patch.object(selection,'pcm',side_effect=AssertionError('audio read')):
            with self.assertRaisesRegex(ValueError,'train/calibration only'):
                selection.score_view(ROOT,[{'split':'test'}],ROOT,ROOT,ROOT,ROOT/'never-created')

    def test_confined_paths_reject_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);(root/'data').write_text('ok');(root/'alias').symlink_to(root/'data')
            self.assertEqual(selection.confined(root,'data'),root/'data')
            for path in ('../data','/data','alias','a//data'):
                with self.subTest(path=path),self.assertRaises(ValueError):selection.confined(root,path)


class RetentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='context-selection-');cls.root=pathlib.Path(cls.tmp.name)
        cls.pool=cls.root/'pool';cls.pool.mkdir()
        kept=materialize_pool([r for r in fixture(cls.root) if r['target_ids']],cls.pool,KEYWORDS)
        for name in ('tokens.example.txt','zh_cn_example.tsv','zh_cn_example.tsv.margin.json'):
            shutil.copyfile(ROOT/'keywords'/name,cls.pool/name)
        files={str(p.relative_to(cls.pool)):sha(p) for p in cls.pool.rglob('*') if p.is_file()}
        write(cls.pool/'pool.json',{'policy':POLICY,'development_only':True,'release_authority':False,
                                  'files':files,'rows':kept})
        cls.pool_sha=sha(cls.pool/'pool.json');cls.rows=kept
        cls.plain=cls.root/'plain';cls.retained=cls.root/'retained'
        train(cls.pool,cls.pool_sha,cls.plain,'grounded-ctc','all',2,1337)
        train(cls.pool,cls.pool_sha,cls.retained,'grounded-ctc','all',2,1337,retain_milestones=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def clone(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        out=pathlib.Path(tmp.name)/'retained';shutil.copytree(self.retained,out);return out

    def test_retention_is_numerical_noop_on_actual_trainer(self):
        a=torch.load(self.plain/'model.pt',weights_only=True);b=torch.load(self.retained/'model.pt',weights_only=True)
        self.assertEqual(a['float_state_identity'],b['float_state_identity'])
        self.assertEqual(a['epoch_history'],b['epoch_history'])
        self.assertEqual(a['batch_order_sha256'],b['batch_order_sha256'])
        self.assertEqual(a['context_order_sha256'],b['context_order_sha256'])
        self.assertEqual(sha(self.plain/'model.kwm'),sha(self.retained/'model.kwm'))
        self.assertFalse((self.plain/'milestones.json').exists())

    def test_real_exported_milestone_reads_back(self):
        index,records=selection.load_milestones(self.retained,self.pool_sha,self.rows)
        self.assertIs(index['completed'],True);self.assertEqual([r['epoch'] for r in records],[2])
        self.assertEqual(records[0]['files']['model.kwm'],sha(self.retained/'model.kwm'))

    def test_partial_snapshot_index_not_accepted(self):
        for bad in (False,'true',1,None):
            root=self.clone();data=json.loads((root/'milestones.json').read_text());data['completed']=bad
            write(root/'milestones.json',data)
            with self.subTest(bad=bad),self.assertRaises(ValueError):selection.load_milestones(root,self.pool_sha,self.rows)

    def test_missing_declared_snapshot_rejected(self):
        root=self.clone();data=json.loads((root/'milestones.json').read_text());data['records']=[]
        write(root/'milestones.json',data)
        with self.assertRaisesRegex(ValueError,'milestone epoch'):selection.load_milestones(root,self.pool_sha,self.rows)

    def test_tampered_weight_or_export_rejected(self):
        root=self.clone();p=root/'milestones/epoch-0002/model.kwm';p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'bytes changed'):selection.load_milestones(root,self.pool_sha,self.rows)

    def test_other_pool_or_training_corpus_rejected(self):
        with self.assertRaises(ValueError):selection.load_milestones(self.retained,'0'*64,self.rows)
        rows=copy.deepcopy(self.rows);next(r for r in rows if r['split']=='train')['target_ids']=[1]
        with self.assertRaisesRegex(ValueError,'source or history'):selection.load_milestones(self.retained,self.pool_sha,rows)

    def test_retention_must_be_bool_and_cannot_overwrite(self):
        with self.assertRaises(ValueError):train(self.pool,self.pool_sha,self.root/'bad','grounded-ctc','all',2,1,retain_milestones='true')
        with self.assertRaises(FileExistsError):train(self.pool,self.pool_sha,self.retained,'grounded-ctc','all',2,1337,retain_milestones=True)


if __name__=='__main__':unittest.main()
