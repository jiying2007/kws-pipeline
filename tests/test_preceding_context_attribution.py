#!/usr/bin/env python3
from __future__ import annotations
from contextlib import ExitStack
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
import preceding_context_attribution as attr
import preceding_context_evaluation as ctx

WORDS = {1: (1,2,3,4), 2: (3,4,3,4)}


def event(kid=1, time=.02):
    return {'recording':'r', 'keyword_id':kid, 'time_s':time, 'confidence':.8}


class AttributionTests(unittest.TestCase):
    def test_canonical_match_not_naive_inside_window(self):
        ref = {'recording':'r','duration_s':2,'expected':[{'keyword_id':1,'start_s':.5,'end_s':1}]}
        summary, fa = attr.scored(ref, [event(2,.9),event(1,1),event(1,1.1)])
        self.assertEqual(summary['matched'],1)
        self.assertEqual([(r['keyword_id'],r['time_s']) for r in fa],[(2,.9),(1,1.1)])

    def test_interval_boundaries_are_half_open(self):
        self.assertEqual(attr.emission_region(.039,640,3840),'prior_clip')
        self.assertEqual(attr.emission_region(.04,640,3840),'gap')
        self.assertEqual(attr.emission_region(.24,640,3840),'target_clip')

    def test_invalid_geometry_or_time_rejected(self):
        for a,b in ((0,3200),(640,641),(True,3840),(640,320)):
            with self.subTest(a=a,b=b), self.assertRaises(ValueError):
                attr.emission_region(.1,a,b)
        for t in (float('nan'),float('inf'),-.1,True,'1'):
            with self.subTest(t=t), self.assertRaises(ValueError):
                attr.emission_region(t,640,3840)

    def test_all_four_control_relationships(self):
        a,b,c,d = [event(k,.3) for k in (1,2,3,4)]
        rows = attr.attribute_events([a,b,c,d],[a,b],[a,c],640,3840)
        self.assertEqual([r['control_relation'] for r in rows],
                         ['both_controls','silence_target_reproduced','isolated_prior_reproduced','combined_only'])
        self.assertTrue(all(r['survives_decoder_prefix_blanking_at_same_time'] is None for r in rows))

    def test_duplicate_events_consume_control_capacity_once(self):
        rows=attr.attribute_events([event(),event()],[event()],[],640,3840)
        self.assertEqual([r['control_relation'] for r in rows],['silence_target_reproduced','combined_only'])

    def test_confidence_is_not_part_of_exact_event_key(self):
        low={**event(), 'confidence':.6}
        rows=attr.attribute_events([event()],[],[low],640,3840)
        self.assertEqual(rows[0]['control_relation'],'isolated_prior_reproduced')

    def test_time_change_does_not_claim_exact_reproduction(self):
        rows=attr.attribute_events([event(time=.3)],[event(time=.32)],[],640,3840)
        self.assertEqual(rows[0]['control_relation'],'combined_only')

    def test_counterfactual_same_signature_and_new_wrong_keyword(self):
        old={'false_accepts':1,'matched':0};new={'false_accepts':1,'matched':0}
        effect=attr.counterfactual_effect(old,new)
        self.assertFalse(effect['all_false_accepts_removed'])
        self.assertEqual(effect['false_accept_count_delta'],0)
        row=attr.attribute_events([event(1,.3)],[],[],640,3840,[event(2,.5)])[0]
        self.assertFalse(row['survives_decoder_prefix_blanking_at_same_time'])

    def test_silent_model_does_not_pass_a_counterfactual_fix(self):
        self.assertFalse(attr.counterfactual_effect({'false_accepts':0,'matched':0},
                                                   {'false_accepts':0,'matched':0})['all_false_accepts_removed'])

    def test_reference_validation_before_interpretation(self):
        ref={'recording':'r','duration_s':1,'expected':[]}
        for bad in ([event(time=2)],[event(time=float('nan'))],[{**event(),'recording':'other'}]):
            with self.assertRaises(ValueError):attr.scored(ref,bad)


class IntegrationTests(unittest.TestCase):
    def run_case(self, root, mutate=None):
        model,runner,dump,replay = [root/n for n in ('model','runner','dump','replay')]
        for path in (model,runner,dump,replay):path.write_bytes(path.name.encode())
        pool={'rows':[]}
        for split in attr.SPLITS:
            for kid, seq in ((1,[1,2,3,4]),(2,[3,4,3,4]),(None,[1,2,4])):
                pool['rows'].append({'split':split,'path':'fixture.wav','wav_sha256':'a'*64,
                    'target_ids':seq,'expected':[] if kid is None else [{'keyword_id':kid,'start_s':.01,'end_s':.03}]})
        source=root/'source';out=root/'output'
        with ExitStack() as stack:
            for module in (ctx,attr):
                stack.enter_context(mock.patch.object(module,'verify_pool',return_value=pool))
                stack.enter_context(mock.patch.object(module,'keyword_target_sequences',return_value=WORDS))
                stack.enter_context(mock.patch.object(module,'resolve',return_value=root/'fixture.wav'))
                stack.enter_context(mock.patch.object(module,'pcm',return_value=b'\1\0'*640))
                stack.enter_context(mock.patch.object(module,'collect',return_value=[]))
            def compile_pack(args,**kwargs):
                pathlib.Path(args[args.index('--out-pack')+1]).write_bytes(b'pack')
            stack.enter_context(mock.patch.object(attr.subprocess,'run',side_effect=compile_pack))
            ctx.evaluate(root,'p'*64,model,runner,source)
            if mutate:mutate(source,out,runner)
            return attr.evaluate(root,'p'*64,model,runner,dump,replay,source,out)

    def test_all_real_control_flow_and_false_verdicts_remain_report_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);result=self.run_case(root)
            self.assertTrue(result['completed']);self.assertFalse(result['release_authority'])
            self.assertTrue(result['report_only']);self.assertEqual(set(result['splits']),set(attr.SPLITS))
            for r in result['splits'].values():
                self.assertEqual(r['context']['matched'],0);self.assertEqual(r['context']['expected'],2)
                self.assertEqual(r['false_accept_emission_regions'],{})
            self.assertFalse((root/'output/scratch.wav').exists())

    def test_authority_and_identity_fail_closed(self):
        valid={'policy':attr.SOURCE_POLICY,'completed':True,'development_only':True,'report_only':True,
               'release_authority':False,'pool_sha256':'p','model_sha256':'m'}
        for k,v in [('completed','true'),('completed',False),('release_authority',True),
                    ('policy','unknown'),('pool_sha256','q'),('model_sha256','n')]:
            with self.subTest(k=k),self.assertRaises(ValueError):attr.validate_source({**valid,k:v},'p','m')

    def test_source_pair_tampering_keeps_incomplete_report(self):
        def mutate(source,out,runner):
            path=source/'train-prior-nonwake.json';value=json.loads(path.read_text())
            value['records'][0]['prefix_samples']+=320;path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            with self.assertRaisesRegex(ValueError,'deterministic frozen'):self.run_case(root,mutate)
            self.assertFalse(json.loads((root/'output/summary.json').read_text())['completed'])

    def test_missing_records_do_not_become_no_errors(self):
        def mutate(source,out,runner):
            path=source/'train-prior-nonwake.json';value=json.loads(path.read_text());value['records'].pop()
            path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'coverage'):self.run_case(pathlib.Path(tmp),mutate)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                self.run_case(pathlib.Path(tmp),lambda s,o,r:o.mkdir())

    def test_frozen_threshold_pack_is_required(self):
        def mutate(source,out,runner):
            (source/'keywords.kwk').write_bytes(b'changed')
            p=source/'summary.json';v=json.loads(p.read_text());v['pack_sha256']=attr.sha(source/'keywords.kwk')
            p.write_text(json.dumps(v))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'configured thresholds'):self.run_case(pathlib.Path(tmp),mutate)


if __name__=='__main__':unittest.main()
