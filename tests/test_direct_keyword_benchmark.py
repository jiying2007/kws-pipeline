#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys,tempfile,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training'))
from direct_keyword_benchmark import POLICY,ROOT_ADMISSION_MARGIN,ctc_sequence_loss,direct_class,frame_loss,grounded_ctc_loss,root_admission_loss,same_voice_pairs,write_vocab

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

    def test_end_window_is_explicit_and_bounded(self):
        self.assertEqual(POLICY,'direct-whole-keyword-root-admission-paired-v5')
        logits=torch.randn(16,1,3).log_softmax(-1)
        short=frame_loss(logits,[16],[(2,14)],[1],4)
        long=frame_loss(logits,[16],[(2,14)],[1],12)
        self.assertNotEqual(float(short),float(long))
        for bad in (0,33,True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                frame_loss(logits,[16],[(2,14)],[1],bad)

    def test_one_token_ctc_is_finite_normalized_and_has_gradient(self):
        logits=torch.randn(12,3,3,requires_grad=True);lp=logits.log_softmax(-1)
        lengths=[12,10,9];classes=[1,2,0]
        loss=ctc_sequence_loss(lp,lengths,classes)
        raw=torch.nn.functional.ctc_loss(
            lp,torch.tensor([1,2]),torch.tensor(lengths),torch.tensor([1,1,0]),
            blank=0,reduction='none',zero_infinity=True)
        expected=(raw/torch.tensor(lengths,dtype=raw.dtype)).mean()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.allclose(loss,expected))
        loss.backward();self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_one_token_ctc_accepts_all_negative_batch_and_rejects_bad_geometry(self):
        logits=torch.randn(6,2,3,requires_grad=True);lp=logits.log_softmax(-1)
        loss=ctc_sequence_loss(lp,[6,5],[0,0])
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for lengths,classes in (([0,5],[0,0]),([7,5],[0,0]),([6,5],[3,0]),([6,5],[True,0])):
            with self.subTest(lengths=lengths,classes=classes), self.assertRaises(ValueError):
                ctc_sequence_loss(lp.detach(),lengths,classes)

    def test_grounded_ctc_rejects_startup_only_shortcut(self):
        logits=torch.full((8,1,3),-8.0);logits[:,:,0]=8.0
        logits[0,0,0]=-8.0;logits[0,0,1]=8.0
        lp=logits.log_softmax(-1)
        ungrounded=ctc_sequence_loss(lp,[8],[1])
        total,ctc,blank=grounded_ctc_loss(lp,[8],[(3,7)],[1])
        self.assertGreater(float(ctc),float(ungrounded))
        self.assertGreater(float(blank),0.0);self.assertGreater(float(total),float(ctc))

    def test_grounded_ctc_keeps_negative_full_sequence_blank(self):
        blank_logits=torch.full((6,1,3),-8.0);blank_logits[:,:,0]=8.0
        wake_logits=blank_logits.clone();wake_logits[3,0,0]=-8.0;wake_logits[3,0,1]=8.0
        clean=grounded_ctc_loss(blank_logits.log_softmax(-1),[6],[(1,5)],[0])[0]
        wrong=grounded_ctc_loss(wake_logits.log_softmax(-1),[6],[(1,5)],[0])[0]
        self.assertGreater(float(wrong),float(clean))

    def test_grounded_ctc_geometry_fails_closed(self):
        lp=torch.randn(6,1,3).log_softmax(-1)
        for lengths,spans,classes in (([6],[(4,4)],[1]),([7],[(1,5)],[1]),([6],[(1,5)],[True]),([6],[(1,7)],[1])):
            with self.subTest(lengths=lengths,spans=spans,classes=classes), self.assertRaises(ValueError):
                grounded_ctc_loss(lp,lengths,spans,classes)

    def test_root_admission_loss_matches_runtime_margin_and_ignores_negatives(self):
        logits=torch.full((5,2,3),-4.0)
        logits[:,:,0]=0.0
        logits[2,0,1]=-0.25
        lp=logits.log_softmax(-1)
        self.assertEqual(ROOT_ADMISSION_MARGIN,0.5)
        self.assertAlmostEqual(float(root_admission_loss(lp,[5,5],[(1,4),(1,4)],[1,0])),0.0,places=6)
        logits[2,0,1]=-1.25;lp=logits.log_softmax(-1)
        self.assertAlmostEqual(float(root_admission_loss(lp,[5,5],[(1,4),(1,4)],[1,0])),0.75,places=5)

    def test_root_admission_geometry_fails_closed(self):
        lp=torch.randn(6,1,3).log_softmax(-1)
        for lengths,spans,classes in (([6],[(4,4)],[1]),([7],[(1,5)],[1]),([6],[(1,5)],[True]),([6],[(1,7)],[1])):
            with self.subTest(lengths=lengths,spans=spans,classes=classes), self.assertRaises(ValueError):
                root_admission_loss(lp,lengths,spans,classes)

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
