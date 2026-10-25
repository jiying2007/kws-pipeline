"""Stdlib-only contracts. Fixtures are invented, never model measurements."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/model_comparison'))
from compare import compare, validate_split
from runner import (GreedyKeywordDecoder, projected_identity, SOURCE_PATHS, source_hashes,
                    verify_source_hashes, sample_order, bounded_integer, MAX_CHUNK_FRAMES, check_wav_bound)


def fixture():
    rows = [dict(split=s, speaker_id=s, session_id=s, source_id=s, pcm_sha256=str(i)*64)
            for i, s in enumerate(('train','calibration','selection'))]
    plan = dict(development_only=True, protected_evidence_used=False,
                split_inventory=rows, seed=7, max_far_per_hour=0.5,
                budget=dict(max_training_seconds=100, max_peak_rss_bytes=1000,
                            max_model_bytes=1000, max_macs_per_frame=1000, sample_exposures=10))
    control = dict(arm='control', status='MEASURED', split_sha256=validate_split(rows), seed=7,
                   sample_exposures=10, threshold_source='calibration-only',
                   selection_used_for_tuning=False, thresholds_sha256='a'*64,
                   negative_hours=8, false_accept_events=2, training_seconds=10,
                   peak_rss_bytes=100, model_bytes=100, macs_per_frame=100,
                   per_keyword={w: dict(positives=10,false_rejects=2) for w in ('你好小窝','小窝小窝')})
    for k in ('source_sha','runner_id','frontend_sha256','tokens_sha256','optimizer_sha256',
              'decoder_sha256','selection_stream_sha256','calibration_stream_sha256'):
        control[k] = 'fixture'
    treatment = copy.deepcopy(control)
    treatment['arm'] = 'treatment'
    treatment['per_keyword']['你好小窝']['false_rejects'] = 1
    return plan, [control, treatment]


class ComparisonTests(unittest.TestCase):
    def test_fixture_arithmetic(self):
        p, r = fixture()
        result = compare(p,r)
        self.assertTrue(result['treatment_per_word_dominates_at_shared_gate'])
        self.assertFalse(result['shipping_approved'])
    def test_each_leakage_dimension(self):
        for key in ('speaker_id','session_id','source_id','pcm_sha256'):
            p,_=fixture()
            p['split_inventory'][1][key]=p['split_inventory'][0][key]
            with self.assertRaisesRegex(ValueError,'leakage'): validate_split(p['split_inventory'])
    def test_protected_split(self):
        p,_=fixture();p['split_inventory'][0]['split']='qualification'
        with self.assertRaises(ValueError):validate_split(p['split_inventory'])
    def test_reject_unmatched_and_tuned(self):
        for key,value in [('seed',8),('sample_exposures',9),('source_sha','different'),
                          ('selection_used_for_tuning',True),('status','NOT_RUN'),('negative_hours',4)]:
            p,r=fixture();r[1][key]=value
            with self.assertRaises(ValueError):compare(p,r)
    def test_far_and_resource_failure_block_winner(self):
        for key,value in [('false_accept_events',5),('training_seconds',101)]:
            p,r=fixture();r[1][key]=value
            self.assertFalse(compare(p,r)['treatment_per_word_dominates_at_shared_gate'])
    def test_one_word_regression_blocks_winner(self):
        p,r=fixture();r[1]['per_keyword']['小窝小窝']['false_rejects']=3
        self.assertFalse(compare(p,r)['treatment_per_word_dominates_at_shared_gate'])
    def test_nan_and_zero_denominator(self):
        p,r=fixture();r[1]['negative_hours']=float('nan')
        with self.assertRaises(ValueError):compare(p,r)
        p,r=fixture();r[1]['per_keyword']['你好小窝']['positives']=0
        with self.assertRaises(ValueError):compare(p,r)


class DecoderTests(unittest.TestCase):
    def decoder(self, **kw):
        return GreedyKeywordDecoder([dict(id=2,tokens=[3,4,3,4],threshold=0.5)], **kw)
    def test_repeat_and_blank(self):
        d=self.decoder();hits=[]
        for x in [3,3,0,4,4,0,3,0,4]:hits.extend(d.accept(x,.8))
        self.assertEqual(len(hits),1)
        self.assertAlmostEqual(hits[0]['time_s'],(400+8*320)/16000)
    def test_no_stale_prefix(self):
        d=self.decoder(max_frames=4)
        hits=[]
        for x in [3,4,0,0,0,3,4]:hits.extend(d.accept(x,.8))
        self.assertEqual(hits,[])
    def test_threshold_and_reset(self):
        d=self.decoder()
        self.assertEqual([h for x in [3,4,3,4] for h in d.accept(x,.4)],[])
        self.assertEqual(self.decoder().history,[])
    def test_cooldown(self):
        d=self.decoder(cooldown_frames=50)
        hits=[h for x in [3,4,3,4,3,4] for h in d.accept(x,.8)]
        self.assertEqual(len(hits),1)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):self.decoder().accept(1,float('nan'))


class HardeningTests(unittest.TestCase):
    def test_preset_voice_leakage_survives_projection(self):
        for location in ('top', 'provenance', 'links'):
            p,_=fixture(); rows=[]
            for i,row in enumerate(p['split_inventory']):
                voice={'preset_voice_id':'shared-voice' if i < 2 else 'other-voice'}
                raw=(voice if location == 'top' else
                     {'speech_like_provenance':voice} if location == 'provenance' else
                     {'lineage_links':[voice]})
                rows.append(projected_identity(row,raw,row['split']))
            with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_direct_preset_voice_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]['preset_voice_id']='shared-voice'
        rows[1]['preset_voice_id']='shared-voice'
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_preset_voice_same_split_and_distinct_voices_allowed(self):
        p,_=fixture();rows=p['split_inventory']
        for row in rows:row['preset_voice_id']='voice-'+row['split']
        rows.append(dict(rows[0],source_id='another-train-source',pcm_sha256='f'*64))
        validate_split(rows)
    def test_preset_voice_placeholder_rejected(self):
        for value in ('unknown',' pending_001 ',None,''):
            p,_=fixture();rows=p['split_inventory']
            rows[0]=projected_identity(rows[0],{'preset_voice_id':value},'train')
            with self.assertRaises(ValueError):validate_split(rows)
    def test_preset_voice_namespace_not_source_identity(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]['preset_voice_id']=rows[1]['source_id']
        validate_split(rows)
    def test_unknown_identities_rejected(self):
        for key in ('speaker_id','session_id','source_id'):
            for value in ('unknown',' NONE ','pending','null','n/a','Unknown-speaker','pending_001'):
                p,_=fixture(); p['split_inventory'][0][key]=value
                with self.assertRaises(ValueError):validate_split(p['split_inventory'])
    def test_augmented_source_family_leakage(self):
        for key in ('source_family','source_family_id','family_id','derivation_family','derivation_family_id',
                    'source_pcm','source_pcm_sha256','original_pcm_sha256','reference_audio_sha256'):
            p,_=fixture(); rows=p['split_inventory']
            shared='f'*64 if 'pcm' in key or key.endswith('sha256') else 'original-family'
            projected=[]
            for i,row in enumerate(rows):
                raw={key:shared} if i < 2 else {}
                projected.append(projected_identity(row,raw,row['split']))
            with self.assertRaisesRegex(ValueError,'leakage'):validate_split(projected)
    def test_file_and_pcm_hash_namespaces_are_distinct(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'reference_audio_sha256':rows[0]['pcm_sha256']},'calibration')
        validate_split(rows)
    def test_family_aliases_cross_connect(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]=projected_identity(rows[0],{'source_family_id':'shared'},'train')
        rows[1]=projected_identity(rows[1],{'derivation_family':'shared'},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_source_id_to_derivation_family_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'derivation_family':rows[0]['source_id']},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_original_to_augmented_pcm_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'source_pcm_sha256':rows[0]['pcm_sha256']},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_transitive_family_alias_and_nested_provenance(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]=projected_identity(rows[0],{'speech_like_provenance':{'source_family':'family-A'}},'train')
        bridge=dict(rows[0],source_id='bridge-source',pcm_sha256='e'*64,
                    lineage_links=[{'source_family':'family-A','derivation_family':'family-B'}])
        rows.append(bridge)
        rows[1]=projected_identity(rows[1],{'derivation_family':'family-B'},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_pcm_projection_uses_measured_and_checks_claim(self):
        p,_=fixture();row=p['split_inventory'][0]
        with self.assertRaises(ValueError):projected_identity(row,{'pcm_sha256':'a'*64},'train')
        self.assertEqual(projected_identity(row,{},'train')['pcm_sha256'],row['pcm_sha256'])
    def test_exact_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in SOURCE_PATHS:
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(path)
            valid=source_hashes(root)
            verify_source_hashes(valid,root)
            missing=dict(valid);missing.pop(SOURCE_PATHS[0])
            extra=dict(valid,unexpected='a'*64)
            bad=dict(valid);bad[SOURCE_PATHS[0]]='not-a-digest'
            upper=dict(valid);upper[SOURCE_PATHS[0]]='A'*64
            wrong=dict(valid);wrong[SOURCE_PATHS[0]]='0'*64
            for invalid in ({},None,missing,extra,bad,upper,wrong):
                with self.assertRaises(ValueError):verify_source_hashes(invalid,root)
    def test_schedule_bounded_lazy_and_repeatable(self):
        a=sample_order(5,3,7)
        self.assertIs(iter(a),a)
        self.assertEqual(list(a),list(sample_order(5,3,7)))
        for size,epochs in ((1,101),(10001,1),(10000,11)):
            with self.assertRaises(ValueError):list(sample_order(size,epochs,1))
    def test_audio_duration_bound(self):
        import wave
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'too-long.wav'
            with wave.open(str(path),'wb') as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(16000)
                wf.writeframes(b'\0\0'*16001)
            with self.assertRaises(ValueError):check_wav_bound(path,1)
    def test_chunk_bounds(self):
        for value in (0,-1,MAX_CHUNK_FRAMES+1,True,1.5):
            with self.assertRaises(ValueError):bounded_integer(value,'chunk',1,MAX_CHUNK_FRAMES)

if __name__=='__main__':unittest.main()
