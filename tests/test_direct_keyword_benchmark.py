#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys,tempfile,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training'))
from direct_keyword_benchmark import POLICY,direct_class,frame_loss,same_voice_pairs,write_vocab

class DirectKeywordBenchmarkTests(unittest.TestCase):
    def test_class_mapping(self):
        self.assertEqual(direct_class({'expected':[]}),0)
        self.assertEqual(direct_class({'expected':[{'keyword_id':1}]}),1)
        self.assertEqual(direct_class({'expected':[{'keyword_id':2}]}),2)
        with self.assertRaises(ValueError): direct_class({'expected':[{'keyword_id':1},{'keyword_id':2}]})

    def test_loss_is_finite_and_has_gradient(self):
        logits=torch.randn(12,3,3,requires_grad=True);lp=logits.log_softmax(-1)
        loss=frame_loss(lp,[12,10,9],[(2,9),(1,8),(1,7)],[1,2,0])
        self.assertTrue(torch.isfinite(loss));loss.backward();self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_positive_requires_valid_class_and_span(self):
        lp=torch.randn(5,1,3).log_softmax(-1)
        for bad in ([3],[-1]):
            with self.assertRaises(ValueError): frame_loss(lp,[5],[(1,4)],bad)
        with self.assertRaises(ValueError): frame_loss(lp,[5],[(4,4)],[1])

    def test_same_voice_pairs_preserve_full_transcript(self):
        def row(tokens,voice): return {'target_ids':tokens,'source_provenance':{'voice_id':voice}}
        rows=[row([1,2],'a'),row([3,4],'a'),row([1,2,3,4],'a'),row([1,2],'b'),row([3,4],'b'),row([1,2,3,4],'b')]
        self.assertEqual(same_voice_pairs(rows),{2:(0,1),5:(3,4)})

    def test_missing_or_cross_voice_components_do_not_pair(self):
        rows=[{'target_ids':[1,2],'source_provenance':{'voice_id':'a'}},{'target_ids':[3,4],'source_provenance':{'voice_id':'b'}},{'target_ids':[1,2,3,4],'source_provenance':{'voice_id':'a'}}]
        self.assertEqual(same_voice_pairs(rows),{})

    def test_vocab_is_two_single_token_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);tokens,keywords,pack=write_vocab(root)
            self.assertIn('<blk> 0',tokens.read_text());text=keywords.read_text()
            self.assertIn('wake1',text);self.assertIn('wake2',text);self.assertTrue(pack.is_file())

if __name__=='__main__':unittest.main()
