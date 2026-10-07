from copy import deepcopy
import unittest
from compare_melo6 import compare_verified_saved, TEXTS, MODELS


class ComparisonProof(unittest.TestCase):
    def setUp(self):
        self.intent = [{'intended_text':t} for t in TEXTS]
        self.clips = [{'audio_id':f'clip-{i:06d}','wav_sha256':f'{i:064x}'} for i in range(1,7)]
        rows = [{'opaque_id':c['audio_id'],'wav_sha256':c['wav_sha256'],'status':'success',
                 'completeness':'complete','raw_text':t,'quality_flags':[]} for c,t in zip(self.clips,TEXTS)]
        self.rows = {m:deepcopy(rows) for m in MODELS}
        self.binding = {k:'a'*64 for k in ('plan_sha256','tts_generation_freeze_sha256','blind_job_sha256',
                                          'sensevoice_raw_freeze_sha256','qwen06_raw_freeze_sha256')}

    def test_two_asr_agreement_does_not_create_acoustic_human_or_training_truth(self):
        out = compare_verified_saved(self.intent,self.clips,self.rows,self.binding)
        for row in out['clips']:
            self.assertTrue(row['machine_text_agreement_only'])
            self.assertEqual(row['acoustic_tail_completeness'],'UNKNOWN')
            self.assertEqual(row['human_pronunciation'],'UNKNOWN')
            self.assertIsNone(row['gold_label'])
            self.assertIsNone(row['ctc_target'])
            self.assertFalse(row['training_admission'])
        self.assertFalse(out['speaker_isolated_split_possible'])
        self.assertFalse(out['kws_improvement_measured'])
        self.assertTrue(out['all_six_weak_lexical_gate_passed'])
        self.assertFalse(out['automatic_source_advancement'])
        self.assertFalse(out['automatic_retry'])

    def test_homophone_or_repeat_loss_is_not_repaired(self):
        self.rows['sensevoice'][0]['raw_text']='你好小屋'
        self.rows['qwen06'][1]['raw_text']='小窝'
        out=compare_verified_saved(self.intent,self.clips,self.rows,self.binding)
        self.assertFalse(out['clips'][0]['machine_text_agreement_only'])
        self.assertFalse(out['clips'][1]['machine_text_agreement_only'])
        self.assertEqual(out['overall_result'],'failed_or_inconclusive_quarantine')
        self.assertEqual(out['clips'][0]['assessment'],'quarantine')

    def test_incomplete_matching_text_is_not_machine_agreement(self):
        self.rows['qwen06'][0]['completeness']='unknown'
        out=compare_verified_saved(self.intent,self.clips,self.rows,self.binding)
        self.assertFalse(out['clips'][0]['machine_text_agreement_only'])

    def test_hash_drift_or_missing_denominator_fails(self):
        self.rows['qwen06'][0]['wav_sha256']='b'*64
        with self.assertRaises(ValueError):
            compare_verified_saved(self.intent,self.clips,self.rows,self.binding)

    def test_any_of_six_required_controls_failing_prevents_lexical_gate(self):
        self.rows['sensevoice'][5]['status']='not_run'
        self.rows['sensevoice'][5]['raw_text']=None
        out=compare_verified_saved(self.intent,self.clips,self.rows,self.binding)
        self.assertFalse(out['all_six_weak_lexical_gate_passed'])
        self.assertFalse(out['automatic_source_advancement'])
        self.assertFalse(out['automatic_retry'])
        self.assertEqual(out['clips'][5]['acoustic_tail_completeness'],'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
