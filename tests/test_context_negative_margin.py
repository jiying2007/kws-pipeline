#!/usr/bin/env python3
from __future__ import annotations
import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'training'), str(ROOT/'tests')]
from model import TinyStreamingRNN
import torch
from context_negative_margin import negative_wake_margin, KEYWORD_SEQUENCES, CEILING, POLICY
from startup_context_training import train, validate_recipe
from sequence_margin import _decoder_sequence_log_confidence
from frozen_speech_ablation import materialize_pool, sha, write, POLICY as POOL_POLICY
from test_frozen_speech_ablation import fixture, KEYWORDS
from select_context_checkpoint import load_milestones


def logits_for(sequence, prefix=0, tail=0):
    logits = torch.full((len(sequence)+prefix+tail,1,5),-12.0)
    logits[:,:,0] = 0
    for i,t in enumerate(sequence):
        logits[i+prefix,0,0] = -12
        logits[i+prefix,0,t] = 0
    return logits.log_softmax(-1)


class MarginTests(unittest.TestCase):
    def test_complete_target_does_not_penalize_its_own_keyword(self):
        for word in KEYWORD_SEQUENCES:
            x=logits_for(word)
            self.assertEqual(float(negative_wake_margin(x,[torch.tensor(word)],[4],[(0,4)])),0)

    def test_incomplete_label_penalizes_false_full_word(self):
        x=logits_for((1,2,3,4)).detach().requires_grad_()
        loss=negative_wake_margin(x,[torch.tensor([1,2,3])],[4],[(0,4)])
        self.assertGreater(float(loss.detach()),0.6);loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[3,0,4]),0)

    def test_other_keyword_is_negative_on_a_positive_row(self):
        x=logits_for((3,4,3,4))
        self.assertGreater(float(negative_wake_margin(x,[torch.tensor([1,2,3,4])],[4],[(0,4)])),0.6)

    def test_own_incomplete_path_below_ceiling_is_not_penalized(self):
        x=logits_for((1,2,3,0,0))
        self.assertEqual(float(negative_wake_margin(x,[torch.tensor([1,2,3])],[5],[(0,5)])),0)

    def test_startup_and_padding_cannot_supply_false_word(self):
        x=logits_for((1,2,3,4),tail=8).detach().requires_grad_()
        loss=negative_wake_margin(x,[torch.tensor([1,2,3])],[10],[(4,8)])
        self.assertEqual(float(loss.detach()),0);loss.backward()
        self.assertEqual(float(x.grad[:4].abs().sum()),0)
        self.assertEqual(float(x.grad[8:].abs().sum()),0)

    def test_vectorized_scores_match_existing_chronological_surrogate(self):
        torch.manual_seed(123)
        x=(4*torch.randn(30,3,5)).log_softmax(-1)
        ys=[torch.tensor([1,2]),torch.tensor([3,4]),torch.tensor([1,2,3,4])]
        spans=[(2,25),(4,27),(1,23)]; expected=[]
        for i,((a,b),y) in enumerate(zip(spans,ys)):
            hinges=[torch.relu(_decoder_sequence_log_confidence(x[a:b,i],w)-math.log(CEILING))
                    for w in KEYWORD_SEQUENCES if tuple(y.tolist())!=w]
            expected.append(torch.stack(hinges).max())
        self.assertTrue(torch.allclose(negative_wake_margin(x,ys,[30]*3,spans),torch.stack(expected).mean()))

    def test_no_chronological_path_has_finite_zero_gradient(self):
        x=logits_for((1,2)).detach().requires_grad_()
        loss=negative_wake_margin(x,[torch.tensor([1,2])],[2],[(0,2)]);loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertTrue(torch.isfinite(x.grad).all())

    def test_batch_permutation_does_not_change_loss(self):
        x=torch.cat([logits_for((1,2,3,4)),logits_for((3,4,3,4))],dim=1)
        ys=[torch.tensor([1,2]),torch.tensor([3,4])]
        a=negative_wake_margin(x,ys,[4,4],[(0,4)]*2)
        b=negative_wake_margin(x[:,[1,0]],ys[::-1],[4,4],[(0,4)]*2)
        self.assertEqual(float(a),float(b))

    def test_invalid_inputs_rejected(self):
        x=logits_for((1,2,3,4))
        for y,n,span in (([0],4,(0,4)),([5],4,(0,4)),([],4,(0,4)),([1],True,(0,4)),([1],4,(0,5)),([1],4,(False,4))):
            with self.subTest(y=y,n=n,span=span),self.assertRaises(ValueError):
                negative_wake_margin(x,[torch.tensor(y,dtype=torch.long)],[n],[span])
        for val in (float('nan'),float('inf'),-1e8,1.0):
            bad=x.clone();bad[0,0,0]=val
            with self.assertRaises(ValueError):negative_wake_margin(bad,[torch.tensor([1])],[4],[(0,4)])

    def test_disabled_trainer_does_not_call_margin(self):
        with mock.patch('context_negative_margin.negative_wake_margin',side_effect=AssertionError('called')):
            # Full disabled execution is covered by IntegrationTests; argument errors
            # also occur before pool access, so malformed settings cannot be ignored.
            for weight in (True,'0',-1,2,float('nan')):
                with self.assertRaises(ValueError):train(ROOT,'',ROOT/'unused','grounded-ctc','all',2,1337,negative_margin_weight=weight)
            with self.assertRaises(ValueError):train(ROOT,'',ROOT/'unused','plain-ctc','all',2,1337,negative_margin_weight=.1)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name);cls.pool=cls.root/'pool';cls.pool.mkdir()
        rows=materialize_pool([r for r in fixture(cls.root) if r['target_ids']],cls.pool,KEYWORDS)
        for name in ('tokens.example.txt','zh_cn_example.tsv','zh_cn_example.tsv.margin.json'):
            shutil.copyfile(ROOT/'keywords'/name,cls.pool/name)
        files={str(p.relative_to(cls.pool)):sha(p) for p in cls.pool.rglob('*') if p.is_file()}
        write(cls.pool/'pool.json',{'policy':POOL_POLICY,'development_only':True,'release_authority':False,'files':files,'rows':rows})
        cls.ps=sha(cls.pool/'pool.json');cls.rows=rows
        with mock.patch('context_negative_margin.negative_wake_margin',side_effect=AssertionError('disabled margin executed')):
            train(cls.pool,cls.ps,cls.root/'baseline','grounded-ctc','all',2,1337)
        train(cls.pool,cls.ps,cls.root/'treatment','grounded-ctc','all',2,1337,negative_margin_weight=.1,retain_milestones=True)
        cls.a=torch.load(cls.root/'baseline/model.pt',weights_only=True)
        cls.b=torch.load(cls.root/'treatment/model.pt',weights_only=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_recipe_is_explicit_and_default_is_unchanged(self):
        self.assertNotIn('negative_wake_margin',self.a['development_recipe'])
        self.assertNotIn('negative_wake_margin',self.a['epoch_history'][0])
        spec=self.b['development_recipe']['negative_wake_margin']
        self.assertEqual(spec['policy'],POLICY);self.assertEqual(spec['weight'],.1)
        self.assertFalse(spec['exact_runtime_loss']);self.assertFalse(self.b['development_recipe']['release_authority'])

    def test_paired_initial_state_and_schedules_match(self):
        for key in ('initial_float_state_identity','batch_order_sha256','context_order_sha256','training_corpus_identity'):
            self.assertEqual(self.a[key],self.b[key])

    def test_loss_is_executed_and_recorded_in_export(self):
        prov=json.loads((self.root/'treatment/model.kwm.provenance.json').read_text())
        self.assertEqual(prov['training']['development_recipe'],self.b['development_recipe'])
        self.assertEqual(prov['training']['epoch_history'][0]['negative_wake_margin'],self.b['epoch_history'][0]['negative_wake_margin'])
        self.assertIn('training/context_negative_margin.py',self.b['training_environment']['training_code_sha256'])
        load_milestones(self.root/'treatment',self.ps,self.rows)


if __name__=='__main__':unittest.main()
