#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
import pathlib
import struct
import sys
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'tools')]
from model import TinyStreamingRNN
import torch
from startup_context_training import active_span, loss_components, negative_runtime_margin, respeed_pcm, select_rows
from startup_context_evaluation import blank_prefix, shifted, measure
from frontend import features
from startup_context_training import validate_recipe


class StartupContextTests(unittest.TestCase):
    def test_scope_is_train_only(self):
        rows=[{'split':s,'expected':[{'keyword_id':k}],'target_ids':[1]} for s in ('train','test','qualification') for k in (1,2)]
        selected=select_rows({'rows':rows},'all')
        self.assertEqual(len(selected),2)
        self.assertTrue(all(r['split']=='train' for r in selected))

    def test_positive_control_excludes_nonwake(self):
        rows=[{'split':'train','expected':[{'keyword_id':k}]} for k in (1,2)]
        rows.append({'split':'train','expected':[]})
        self.assertEqual(len(select_rows({'rows':rows},'positive-only')),2)
        self.assertEqual(len(select_rows({'rows':rows},'all')),3)

    def test_missing_keyword_and_bad_scope_fail(self):
        with self.assertRaises(ValueError):select_rows({'rows':[]},'all')
        with self.assertRaises(ValueError):select_rows({'rows':[]},'test')

    def test_shift_moves_event_without_mutating_source(self):
        original=[{'keyword_id':1,'start_s':.1,'end_s':1.1}]
        shifted_events=shifted(original,16000)
        self.assertEqual(shifted_events[0]['start_s'],1.1)
        self.assertEqual(shifted_events[0]['end_s'],2.1)
        self.assertEqual(original[0]['start_s'],.1)

    def test_shift_rejects_non_hop_and_invalid_inputs(self):
        for n in (-320,1,320.0,True):
            with self.subTest(n=n),self.assertRaises(ValueError):shifted([],n)

    def test_activity_detector_no_model_alignment_or_zero_signal(self):
        samples=torch.zeros(3200,dtype=torch.int16);samples[1600:2400]=1000
        self.assertEqual(active_span(samples,9),(4,8))
        with self.assertRaises(ValueError):active_span(torch.zeros(1000,dtype=torch.int16),2)

    def test_hop_aligned_pcm_context_preserves_voice_features(self):
        t=torch.arange(24000)
        voice=(torch.sin(t*.13)*.1).float()
        bare=features(voice,32)
        shifted_x=features(torch.cat([torch.zeros(16000),voice,torch.zeros(8000)]),32)
        self.assertTrue(torch.equal(bare,shifted_x[50:50+len(bare)]))

    def test_speed_augmentation_is_deterministic_and_preserves_source(self):
        source=(torch.arange(16000,dtype=torch.int32)%2000-1000).to(torch.int16)
        original=source.clone()
        self.assertTrue(torch.equal(respeed_pcm(source,1.0),source))
        fast=respeed_pcm(source,1.15)
        slow=respeed_pcm(source,0.85)
        self.assertLess(len(fast),len(source))
        self.assertGreater(len(slow),len(source))
        self.assertTrue(torch.equal(fast,respeed_pcm(source,1.15)))
        self.assertTrue(torch.equal(source,original))
        with self.assertRaisesRegex(ValueError,'unsupported speed'):
            respeed_pcm(source,1.5)
        with self.assertRaisesRegex(ValueError,'unsupported speed'):
            respeed_pcm(source,True)

    def test_losses_finite_and_differentiable(self):
        for variant in ('plain-ctc','context-ctc','span-ctc','blank-ctc','grounded-ctc'):
            z=torch.randn(20,2,5,requires_grad=True)
            c,b=loss_components(z.log_softmax(-1),[torch.tensor([1,2]),torch.tensor([3,4])],[20,18],[(5,15),(4,14)],variant)
            (c+.3*b).backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            if variant not in ('blank-ctc','grounded-ctc'):self.assertEqual(float(b.detach()),0.0)

    def test_ctc_cannot_use_startup_tokens_in_grounded_mode(self):
        # Target 1,2 appears only before the activity span. It must not lower
        # the grounded transcription loss, even though full-clip CTC accepts it.
        lp=torch.full((20,1,3),-12.0);lp[:,0,0]=0
        lp[0,0]=torch.tensor([-12.,0.,-12.]);lp[1,0]=torch.tensor([-12.,-12.,0.])
        c_plain,_=loss_components(lp,[torch.tensor([1,2])],[20],[(8,16)],'plain-ctc')
        c_ground,_=loss_components(lp,[torch.tensor([1,2])],[20],[(8,16)],'grounded-ctc')
        self.assertGreater(float(c_ground),float(c_plain)+5)

    def test_factorial_controls_isolate_span_and_blank_supervision(self):
        lp=torch.randn(20,1,5).log_softmax(-1).detach().requires_grad_()
        args=([torch.tensor([1,2])],[20],[(5,15)])
        context_ctc,context_blank=loss_components(lp,*args,'context-ctc')
        span_ctc,span_blank=loss_components(lp,*args,'span-ctc')
        blank_ctc,blank_blank=loss_components(lp,*args,'blank-ctc')
        grounded_ctc,grounded_blank=loss_components(lp,*args,'grounded-ctc')
        self.assertTrue(torch.equal(context_ctc,blank_ctc))
        self.assertTrue(torch.equal(span_ctc,grounded_ctc))
        self.assertTrue(torch.equal(blank_blank,grounded_blank))
        self.assertEqual(float(context_blank.detach()),0.0)
        self.assertEqual(float(span_blank.detach()),0.0)
        self.assertGreater(float(blank_blank.detach()),0.0)
        span_ctc.backward(retain_graph=True)
        self.assertEqual(float(lp.grad[:5].abs().sum()),0.0)
        self.assertGreater(float(lp.grad[5:15].abs().sum()),0.0)
        lp.grad.zero_()
        blank_blank.backward()
        self.assertGreater(float(lp.grad[:3].abs().sum()),0.0)
        self.assertEqual(float(lp.grad[3:17].abs().sum()),0.0)

    def test_negative_runtime_margin_targets_executable_nonwake_path(self):
        z=torch.full((12,2,5),-7.0)
        z[:,:,0]=5.0
        for frame,token in enumerate((1,2,3,4),1):
            z[frame,0,token]=9.0
        z=z.detach().requires_grad_()
        value=negative_runtime_margin(z.log_softmax(-1),
            [torch.tensor([1,2,3]),torch.tensor([1,2,3,4])],
            [12,12],[[1,2,3,4],[3,4,3,4]])
        self.assertGreater(float(value.detach()),0.0)
        value.backward()
        self.assertGreater(float(z.grad[:,0].abs().sum()),0.0)
        self.assertEqual(float(z.grad[:,1].abs().sum()),0.0)
        with self.assertRaisesRegex(ValueError,'batch shape mismatch'):
            negative_runtime_margin(z.log_softmax(-1),[],[],[[1,2,3,4]])
        with self.assertRaisesRegex(ValueError,'path policy is invalid'):
            negative_runtime_margin(z[:,:1].log_softmax(-1),[torch.tensor([1,2,3])],
                                    [12],[[1,2,3,4]],negative_path_policy='unknown')

    def test_ctc_gradient_cannot_reach_excluded_prefix(self):
        lp=torch.randn(20,1,5).log_softmax(-1).detach().requires_grad_()
        c,_=loss_components(lp,[torch.tensor([1,2])],[20],[(8,16)],'grounded-ctc')
        c.backward()
        self.assertEqual(float(lp.grad[:8].abs().sum()),0)
        self.assertGreater(float(lp.grad[8:16].abs().sum()),0)

    def test_blank_supervision_excludes_boundary_guard_and_batch_padding(self):
        lp=torch.randn(30,1,5).log_softmax(-1).detach().requires_grad_()
        _,b=loss_components(lp,[torch.tensor([1,2])],[20],[(5,15)],'grounded-ctc');b.backward()
        self.assertEqual(float(lp.grad[3:17].abs().sum()),0)
        self.assertEqual(float(lp.grad[20:].abs().sum()),0)
        self.assertGreater(float(lp.grad[:3,0,0].abs().sum()),0)

    def test_repeated_token_alignment_requires_extra_frame(self):
        with self.assertRaisesRegex(ValueError,'infeasible'):
            loss_components(torch.zeros(2,1,5),[torch.tensor([1,1])],[2],[(0,2)],'grounded-ctc')

    def test_invalid_targets_and_spans_rejected(self):
        for y,span in (([],(2,8)),([0],(2,8)),([5],(2,8)),([1],(8,2)),([1],(0,11))):
            with self.subTest(y=y,span=span),self.assertRaises(ValueError):
                loss_components(torch.zeros(10,1,5),[torch.tensor(y,dtype=torch.long)],[10],[span],'grounded-ctc')

    def test_nonfinite_ctc_fails_instead_of_zero_infinity(self):
        with self.assertRaisesRegex(ValueError,'non-finite'):
            loss_components(torch.full((10,1,5),float('nan')),[torch.tensor([1])],[10],[(2,8)],'grounded-ctc')

    def test_counterfactual_only_edits_added_silence_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);p=root/'in.kwtr';out=root/'edit.kwtr'
            raw=bytearray(120);raw[:8]=b'KWTRACE1';struct.pack_into('<H',raw,12,5);struct.pack_into('<Q',raw,40,12)
            for i in range(12):raw+=struct.pack('<Q8B5f',400+i*320,1,0,0,0,0,0,0,0,1.,2.,3.,4.,5.)
            p.write_bytes(raw);self.assertEqual(blank_prefix(p,out,3200),9)
            got=out.read_bytes();self.assertEqual(got[:120],raw[:120]);self.assertEqual(got[120+9*36:],raw[120+9*36:])
            for i in range(9):self.assertEqual(got[120+i*36:136+i*36],raw[120+i*36:136+i*36])
            receipt=json.loads(out.with_suffix('.counterfactual.json').read_text())
            self.assertTrue(receipt['counterfactual']);self.assertFalse(receipt['authentic_model_output'])

    def test_matching_metrics_do_not_hide_false_accepts(self):
        ref=[{'recording':'r','duration_s':3.0,'expected':[{'keyword_id':1,'start_s':1.,'end_s':2.}]}]
        det=[{'recording':'r','keyword_id':1,'time_s':1.5,'confidence':.9},
             {'recording':'r','keyword_id':2,'time_s':1.5,'confidence':.9}]
        score=measure(ref,det);self.assertEqual(score['matched'],1);self.assertEqual(score['false_accepts'],1)

    def test_no_decoder_or_shipping_gate_override_in_experiment(self):
        source=(ROOT/'training/startup_context_evaluation.py').read_text()
        self.assertNotIn('--state-retention',source)
        self.assertNotIn('--decoder-fuzzy',source)
        trainer=(ROOT/'training/startup_context_training.py').read_text()
        self.assertNotIn("r['split'] == 'test'",trainer)


class ExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        sys.path.insert(0,str(ROOT/'tests'))
        from test_frozen_speech_ablation import fixture, KEYWORDS
        from frozen_speech_ablation import materialize_pool, sha, write, POLICY
        from startup_context_training import train
        cls.tmp=tempfile.TemporaryDirectory();cls.root=pathlib.Path(cls.tmp.name)
        pool=cls.root/'pool';pool.mkdir()
        rows=[r for r in fixture(cls.root) if r['target_ids']]
        kept=materialize_pool(rows,pool,KEYWORDS)
        for name in ('tokens.example.txt','zh_cn_example.tsv','zh_cn_example.tsv.margin.json'):
            shutil.copyfile(ROOT/'keywords'/name,pool/name)
        files={str(p.relative_to(pool)):sha(p) for p in pool.rglob('*') if p.is_file()}
        write(pool/'pool.json',{'policy':POLICY,'development_only':True,'release_authority':False,'files':files,'rows':kept})
        cls.pool=pool;cls.pool_sha=sha(pool/'pool.json');cls.out=cls.root/'trial'
        train(pool,cls.pool_sha,cls.out,'grounded-ctc','all',2,1337)
        cls.checkpoint=torch.load(cls.out/'model.pt',map_location='cpu',weights_only=True)
        cls.provenance=json.loads((cls.out/'model.kwm.provenance.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_actual_train_export_preserves_recipe_and_state(self):
        from training_state import state_identity
        self.assertEqual(self.checkpoint['float_state_identity'],state_identity(self.checkpoint['state_dict']))
        self.assertEqual(self.provenance['training']['development_recipe'],self.checkpoint['development_recipe'])
        self.assertFalse(self.provenance['training']['development_recipe']['release_authority'])
        self.assertEqual(self.checkpoint['epochs'],2)
        self.assertIn('non_speech_blank',self.provenance['training']['epoch_history'][-1])
        self.assertEqual(len(self.checkpoint['batch_order_sha256']),64)

    def test_factorial_controls_export_distinct_recipes_with_paired_inputs(self):
        from startup_context_training import train
        from training_state import state_identity
        for variant,blank_weight in (('span-ctc',0.0),('blank-ctc',0.3)):
            with self.subTest(variant=variant):
                out=self.root/variant
                train(self.pool,self.pool_sha,out,variant,'all',2,1337)
                cp=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
                provenance=json.loads((out/'model.kwm.provenance.json').read_text())
                recipe=cp['development_recipe']
                self.assertEqual(recipe['variant'],variant)
                self.assertEqual(recipe['blank_weight'],blank_weight)
                self.assertEqual(recipe['prefix_hops'],[0,10,25,50])
                self.assertEqual(cp['initial_float_state_identity'],self.checkpoint['initial_float_state_identity'])
                self.assertEqual(cp['batch_order_sha256'],self.checkpoint['batch_order_sha256'])
                self.assertEqual(cp['context_order_sha256'],self.checkpoint['context_order_sha256'])
                self.assertEqual(cp['float_state_identity'],state_identity(cp['state_dict']))
                self.assertEqual(provenance['training']['development_recipe'],recipe)

    def test_speed_training_keeps_batch_and_context_pairing(self):
        from startup_context_training import train
        out=self.root/'speed'
        train(self.pool,self.pool_sha,out,'grounded-speed','all',2,1337)
        cp=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
        recipe=cp['development_recipe']
        self.assertEqual(recipe['speed_factors'],[0.85,1.0,1.15])
        self.assertEqual(cp['initial_float_state_identity'],self.checkpoint['initial_float_state_identity'])
        self.assertEqual(cp['batch_order_sha256'],self.checkpoint['batch_order_sha256'])
        self.assertEqual(cp['context_order_sha256'],self.checkpoint['context_order_sha256'])
        self.assertEqual(len(json.loads((out/'schedule.json').read_text())['speeds_sha256']),64)

    def test_paired_warm_start_binds_source_and_resets_optimizer(self):
        from startup_context_training import train
        source=self.out/'model.pt'
        rows=[]
        for variant in ('grounded-ctc','grounded-negative-margin','grounded-sparse-margin'):
            out=self.root/('warm-'+variant)
            train(self.pool,self.pool_sha,out,variant,'all',1,2346,
                  initial_checkpoint=source)
            cp=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            receipt=cp['development_recipe']['warm_start']
            self.assertEqual(receipt['checkpoint_sha256'],__import__('hashlib').sha256(source.read_bytes()).hexdigest())
            self.assertEqual(cp['initial_float_state_identity'],self.checkpoint['float_state_identity'])
            self.assertFalse(receipt['optimizer_state_restored'])
            rows.append(cp)
        self.assertEqual(rows[0]['batch_order_sha256'],rows[1]['batch_order_sha256'])
        self.assertEqual(rows[0]['context_order_sha256'],rows[1]['context_order_sha256'])
        self.assertEqual(rows[0]['batch_order_sha256'],rows[2]['batch_order_sha256'])
        self.assertEqual(rows[0]['context_order_sha256'],rows[2]['context_order_sha256'])
        self.assertEqual(rows[1]['development_recipe']['negative_runtime_margin_weight'],0.1)
        self.assertEqual(rows[2]['development_recipe']['negative_path_policy'],'sparse-chronological-v1')

    def test_warm_start_rejects_wrong_pool_identity(self):
        from startup_context_training import train
        source=self.root/'wrong-source.pt'
        cp=copy.deepcopy(self.checkpoint)
        cp['development_recipe']['pool_sha256']='0'*64
        torch.save(cp,source)
        with self.assertRaisesRegex(ValueError,'does not match frozen grounded source'):
            train(self.pool,self.pool_sha,self.root/'wrong-warm','grounded-negative-margin',
                  'all',1,2346,initial_checkpoint=source)

    def test_export_rejects_laundered_experimental_authority(self):
        for field,bad in (('release_authority',True),('development_only','true'),('policy','shipping')):
            cp=copy.deepcopy(self.checkpoint);cp['development_recipe'][field]=bad
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'recipe authority'):validate_recipe(cp['development_recipe'])

    def test_trial_cannot_overwrite_evidence(self):
        from startup_context_training import train
        with self.assertRaises(FileExistsError):train(self.pool,self.pool_sha,self.out,'grounded-ctc','all',2,1337)

    def test_trial_rejects_tampered_pool_before_training(self):
        from startup_context_training import train
        with self.assertRaisesRegex(ValueError,'receipt hash'):train(self.pool,'0'*64,self.root/'bad','grounded-ctc','all',2,1337)


if __name__=='__main__':unittest.main()
