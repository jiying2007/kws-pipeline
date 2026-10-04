"""Fictional metadata only. These tests never load or execute a model."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quality as q


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.m = q.load(ROOT / 'examples/toy-manifest.json')
        self.p = q.load(ROOT / 'examples/toy-predictions.json')
        self.digest = q.sha(ROOT / 'examples/toy-manifest.json')

    def score(self):
        return q.score(self.m, self.p, self.digest)

    def assert_bad(self, message=None):
        with self.assertRaises(q.Invalid, msg=message):
            self.score()

    def test_exact_toy_metrics(self):
        r = self.score()
        d = r['partitions']['dev']
        k = d['per_keyword'][self.m['keywords'][0]]
        self.assertEqual((k['eligible_positive_events'], k['hits'], k['misses']), (3, 2, 1))
        self.assertEqual(k['frr'], 1/3)
        self.assertEqual(k['recall'], 2/3)
        self.assertEqual(k['duplicate_decisions'], 1)
        self.assertEqual(k['wrong_keyword_decisions_in_positive_windows'], 1)
        self.assertEqual((k['negative_hours'], k['false_alarm_events'], k['observed_fa_per_hour']), (2, 1, .5))
        self.assertEqual(d['all_keyword_false_alarm_events'], 2)
        self.assertEqual(d['all_keyword_observed_fa_per_hour'], 1)
        self.assertAlmostEqual(k['fa_per_hour_poisson_upper'], 4.743864518390577/2, places=11)
        self.assertAlmostEqual(k['word_tail_latency']['p50_s'], -.1, places=14)
        self.assertEqual(k['word_tail_latency']['count'], 1)
        self.assertEqual(k['missing_word_tail_hits'], 1)
        self.assertEqual(d['decisions_outside_verified_truth'], 1)
        self.assertEqual([a['asset_id'] for a in r['exploratory']], ['unknown', 'exposed'])
        self.assertEqual(r['qualified_heldout_assets'], 1)
        self.assertIsNone(r['thresholds'])
        self.assertFalse(r['shipping_approved'])

    def test_strata_retain_denominators(self):
        d = self.score()['partitions']['dev']
        self.assertEqual(d['strata']['dataset']['toy-positive']['negative_hours'], 0)
        self.assertEqual(d['strata']['dataset']['toy-continuous-negative']['negative_hours'], 2)
        self.assertEqual(d['strata']['far_field']['unknown']['negative_hours'], 2)

    def test_zero_fa_bound_not_zero_far(self):
        self.p['assets'][1]['decisions'] = []
        d = self.score()['partitions']['dev']
        self.assertEqual(d['all_keyword_observed_fa_per_hour'], 0)
        self.assertAlmostEqual(d['all_keyword_fa_per_hour_poisson_upper'], -math.log(.05)/2)
        self.assertIn('does not establish', self.score()['assumptions']['zero_events'])

    def test_no_negative_hours_never_clip_fpr(self):
        self.m['assets'][1]['negative_intervals'] = []
        d = self.score()['partitions']['dev']
        self.assertEqual(d['negative_hours'], 0)
        self.assertIsNone(d['all_keyword_observed_fa_per_hour'])
        self.assertIsNone(d['all_keyword_fa_per_hour_poisson_upper'])

    def test_assumptions_not_asserted(self):
        self.m['protocol']['iid_binomial_assumption'] = False
        self.m['protocol']['stationary_poisson_assumption'] = False
        k = self.score()['partitions']['dev']['per_keyword'][self.m['keywords'][0]]
        self.assertIsNone(k['frr_wilson_interval'])
        self.assertIsNone(k['fa_per_hour_poisson_upper'])
        self.assertEqual(k['observed_fa_per_hour'], .5)

    def test_missing_true_tail_never_clip_end_fallback(self):
        for e in self.m['assets'][0]['positive_events']:
            e['word_tail_s'] = e['word_tail_evidence_ref'] = None
        k = self.score()['partitions']['dev']['per_keyword'][self.m['keywords'][0]]
        self.assertEqual(k['word_tail_latency']['status'], 'NOT_QUALIFIED')
        self.assertEqual(k['missing_word_tail_hits'], 2)

    def test_window_does_not_rescue_other_event(self):
        self.p['assets'][0]['decisions'].append({'id':'another-duplicate','keyword':self.m['keywords'][0],'decision_time_s':1.9})
        k = self.score()['partitions']['dev']['per_keyword'][self.m['keywords'][0]]
        self.assertEqual(k['misses'], 1)
        self.assertEqual(k['duplicate_decisions'], 2)

    def test_eof_callback_is_counted(self):
        self.p['assets'][1]['decisions'][0]['decision_time_s'] = 7200
        self.assertEqual(self.score()['partitions']['dev']['all_keyword_false_alarm_events'], 2)
        self.m['assets'][4]['positive_events'][0]['end_s'] = 3
        self.p['assets'][4]['decisions'][0]['decision_time_s'] = 3
        k = self.score()['partitions']['heldout']['per_keyword'][self.m['keywords'][1]]
        self.assertEqual(k['hits'], 1)
        self.assertEqual(k['word_tail_latency']['max_s'], 1.5)

    def test_true_tail_at_recording_eof(self):
        event=self.m['assets'][4]['positive_events'][0]
        event.update(end_s=3,word_tail_s=3)
        self.p['assets'][4]['decisions'][0]['decision_time_s']=3
        k=self.score()['partitions']['heldout']['per_keyword'][self.m['keywords'][1]]
        self.assertEqual(k['hits'],1)
        self.assertEqual(k['word_tail_latency']['count'],1)
        self.assertEqual(k['word_tail_latency']['max_s'],0)

    def test_touching_intervals_half_open(self):
        n = self.m['assets'][1]['negative_intervals'][0]
        n['end_s'] = 3600
        other = copy.deepcopy(n);other['start_s'] = 3600;other['end_s'] = 7200
        self.m['assets'][1]['negative_intervals'].append(other)
        self.p['assets'][1]['decisions'][0]['decision_time_s'] = 3600
        self.assertEqual(self.score()['partitions']['dev']['all_keyword_false_alarm_events'], 2)

    def test_wilson_known_value_and_boundaries(self):
        self.assertEqual(q.wilson(0, 0, .95), None)
        self.assertAlmostEqual(q.wilson(1, 3, .95)[0], .06149194472039621)
        self.assertAlmostEqual(q.wilson(1, 3, .95)[1], .7923403991979522)
        for total in (1, 2, 3, 10, 100):
            self.assertEqual(q.wilson(0, total, .95)[0], 0)
            self.assertEqual(q.wilson(total, total, .95)[1], 1)

    def test_poisson_known_values(self):
        self.assertAlmostEqual(q.poisson_upper_count(0, .95), 2.99573227355399)
        self.assertAlmostEqual(q.poisson_upper_count(1, .95), 4.743864518390577, places=11)
        self.assertAlmostEqual(q.poisson_upper_count(2, .95), 6.295793621871989, places=11)
        self.assertAlmostEqual(q.poisson_upper_count(10, .95), 16.9622192357219, places=10)

    def test_malformed_numbers(self):
        for value in (True, float('nan'), float('inf'), -1):
            with self.subTest(value=value):
                original = self.p['assets'][0]['decisions'][0]['decision_time_s']
                self.p['assets'][0]['decisions'][0]['decision_time_s'] = value
                self.assert_bad()
                self.p['assets'][0]['decisions'][0]['decision_time_s'] = original

    def test_semantic_rejections(self):
        cases = [
            ('exposed heldout', lambda: self.m['assets'][4]['exposures'].append('results_observed')),
            ('unknown gold', lambda: self.m['assets'][0].update(label_strength='unknown')),
            ('weak gold', lambda: self.m['assets'][0].update(label_strength='weak')),
            ('unbound gold', lambda: self.m['assets'][0].update(label_binding='not_publicly_bound')),
            ('unreviewed license', lambda: self.m['assets'][0]['license'].update(review_status='unknown')),
            ('role permission', lambda: self.m['assets'][0]['license'].update(approved_roles=[])),
            ('exposure audit', lambda: self.m['assets'][0].update(exposure_review_complete=False)),
            ('freeze', lambda: self.m['protocol'].update(split_frozen_before_predictions=False)),
            ('heldout authorization', lambda: self.m['protocol'].update(heldout_open_authorized=False)),
            ('missing speaker', lambda: self.m['assets'][0]['source'].update(speaker=None)),
            ('missing session', lambda: self.m['assets'][0]['source'].update(session=None)),
            ('noncontinuous negatives', lambda: self.m['assets'][1].update(continuous=False)),
            ('looped negatives', lambda: self.m['assets'][1].update(replayed_or_looped=True)),
            ('duration', lambda: self.m['assets'][1].update(duration_s=7199)),
            ('negative outside', lambda: self.m['assets'][1]['negative_intervals'][0].update(end_s=7201)),
            ('partial keyword absence', lambda: self.m['assets'][1]['negative_intervals'][0].update(verified_absent_keywords=[self.m['keywords'][0]])),
            ('tail outside', lambda: self.m['assets'][0]['positive_events'][0].update(word_tail_s=9)),
            ('tail evidence missing', lambda: self.m['assets'][0]['positive_events'][0].update(word_tail_evidence_ref=None)),
            ('event duplicate', lambda: self.m['assets'][0]['positive_events'].append(copy.deepcopy(self.m['assets'][0]['positive_events'][0]))),
            ('negative duplicate', lambda: self.m['assets'][1]['negative_intervals'].append(copy.deepcopy(self.m['assets'][1]['negative_intervals'][0]))),
            ('positive overlap', lambda: self.m['assets'][0]['positive_events'][1].update(start_s=1.9)),
            ('missing coverage', lambda: self.p['assets'].pop()),
            ('partial processing', lambda: self.p['assets'][1].update(complete=False)),
            ('wrong processed duration', lambda: self.p['assets'][1].update(processed_duration_s=7000)),
            ('unknown prediction asset', lambda: self.p['assets'][0].update(asset_id='other')),
            ('extra schema field', lambda: self.p.update(allow_missing=True)),
            ('duplicate decision', lambda: self.p['assets'][0]['decisions'].append(copy.deepcopy(self.p['assets'][0]['decisions'][0]))),
            ('unknown decision keyword', lambda: self.p['assets'][0]['decisions'][0].update(keyword='other')),
            ('wrong time semantics', lambda: self.p.update(decision_time_semantics='decoder_keyword_end')),
        ]
        for name, change in cases:
            with self.subTest(case=name):
                self.setUp();change();self.assert_bad(name)

    def test_identity_rejections(self):
        for key in ('manifest_sha256','protocol_sha256','model_sha256','source_sha256','decoder_config_sha256'):
            with self.subTest(key=key):
                self.setUp();self.p[key]='0'*64;self.assert_bad()
        self.setUp();self.p['assets'][0]['pcm_sha256']='0'*64;self.assert_bad()

    def test_cross_split_family_speaker_and_pcm(self):
        for key in ('family', 'speaker'):
            with self.subTest(key=key):
                self.setUp();self.m['assets'][4]['source'][key]=self.m['assets'][0]['source'][key];self.assert_bad()
        for key in ('leakage_group','derivation_family','pcm_sha256','wav_sha256'):
            with self.subTest(key=key):
                self.setUp();self.m['assets'][4][key]=self.m['assets'][0][key];self.assert_bad()

    def test_inventory_exposure_leaks_to_heldout(self):
        self.m['assets'][3]['role']='inventory'
        self.m['assets'][3]['source']['speaker']=self.m['assets'][4]['source']['speaker']
        self.assert_bad()

    def test_same_split_duplicate_audio(self):
        self.m['assets'][2]['pcm_sha256']=self.m['assets'][1]['pcm_sha256']
        self.assert_bad()

    def test_negative_crops_cannot_multiply_exposure(self):
        for key in ('derivation_family','session'):
            with self.subTest(key=key):
                self.setUp()
                a=copy.deepcopy(self.m['assets'][1]);a['id']='crop';a['wav_sha256']='1'*64;a['pcm_sha256']='2'*64
                a['derivation_family']='new-origin';a['source']['session']='new-session'
                if key=='session':a['source']['session']=self.m['assets'][1]['source']['session']
                else:a['derivation_family']=self.m['assets'][1]['derivation_family']
                self.m['assets'].append(a);self.m['protocol']['input_order'].append('crop')
                run=copy.deepcopy(self.p['assets'][1]);run.update(asset_id='crop',wav_sha256=a['wav_sha256'],pcm_sha256=a['pcm_sha256'])
                self.p['assets'].append(run);self.assert_bad()

    def test_frozen_input_order_and_state_policy(self):
        self.p['assets'].reverse();self.assert_bad()
        self.setUp();self.m['assets'].reverse();self.assert_bad()
        self.setUp();self.p['state_policy']='reset_every_chunk';self.assert_bad()

    def test_extreme_confidence_is_finite(self):
        self.m['protocol']['confidence']=.9999999999999999
        r=self.score()['partitions']['dev']['per_keyword'][self.m['keywords'][0]]
        self.assertTrue(all(math.isfinite(x) for x in r['frr_wilson_interval']))
        self.assertTrue(math.isfinite(r['fa_per_hour_poisson_upper']))

    def test_generator_group_conflicts(self):
        for field in ('voice','prompt_family'):
            with self.subTest(field=field):
                self.setUp()
                for i in (0,4):
                    self.m['assets'][i]['source']['tier']='synthetic'
                    self.m['assets'][i]['source']['generator']=dict(model='toy-generator',revision='toy-pin',voice='voice-'+str(i),seed=i,prompt_family='prompt-'+str(i),evidence_ref='fictional source',evidence_sha256='1'*64)
                self.m['assets'][4]['source']['generator'][field]=self.m['assets'][0]['source']['generator'][field]
                self.assert_bad()

    def test_recorded_resource_not_inferred(self):
        self.assertEqual(self.score()['recorded_resources'], [])
        r=dict(name='peak_rss_bytes',value=1024,unit='bytes',phase='whole_run',evidence_ref='fictional measurement',evidence_sha256='1'*64,environment='toy host',sample_count=1,correlation_note='one fictional sample')
        self.p['recorded_resources']=[r]
        self.assertEqual(self.score()['recorded_resources'],[r])
        r['unit']='seconds';self.assert_bad()
        r['unit']='bytes';r['value']=-1;self.assert_bad()

    def test_existing_inventory_denominators_empty(self):
        m=q.load(ROOT/'inventory/existing-50.manifest.json')
        r=q.admission(m)
        self.assertEqual(r['assets'],50)
        self.assertEqual(r['scoring_eligible_assets'],0)
        self.assertEqual(r['qualified_heldout_assets'],0)
        self.assertEqual(sum(a['duration_s'] for a in m['assets']),73.8)
        self.assertEqual(sum(len(a['negative_intervals'])+len(a['positive_events']) for a in m['assets']),0)
        fixed=[a for a in m['assets'] if a['source']['dataset']=='fixed12']
        self.assertEqual(len(fixed),12)
        self.assertTrue(all(a['label_binding']=='not_publicly_bound' for a in fixed))

    def test_json_duplicate_nonfinite_and_bad_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input.json'
            for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
                p.write_text(value)
                if '1e999' in value:
                    with self.assertRaises(q.Invalid):q.schema_check(q.load(p),{'type':'object','properties':{'a':{'type':'number'}}})
                else:
                    with self.assertRaises(q.Invalid):q.load(p)

    def test_cli_atomic_input_and_existing_output_protection(self):
        command=[sys.executable,str(ROOT/'quality.py'),'score','--manifest',str(ROOT/'examples/toy-manifest.json'),'--predictions',str(ROOT/'examples/toy-predictions.json')]
        r=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['assets'],5)
        r=subprocess.run(command+['--output',str(ROOT/'examples/toy-manifest.json')],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'result.json';out.write_text('keep')
            r=subprocess.run(command+['--output',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertEqual(out.read_text(),'keep')


if __name__=='__main__':
    unittest.main()
