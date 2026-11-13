"""No audio, no numpy, no torch: hostile source-only full-chain receipt cases."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

import integrated_pcm_encoder as chain

HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invented_receipt():
    return {
        'schema':chain.POLICY,
        'status':chain.STATUS,
        'source_tts_zip_sha256':chain.pcm.TTS_SHA256,
        'source_review_json_sha256':chain.pcm.REVIEW_SHA256,
        'runner_sha256':digest(HERE/'integrated_pcm_encoder.py'),
        'saved_training_script_sha256':digest(HERE/'english63_weak_train.py'),
        'saved_streaming_encoder_script_sha256':digest(HERE/'streaming_parity.py'),
        'saved_frontend_script_sha256':digest(HERE/'pcm_frontend_parity.py'),
        'source_audio_clips':63,
        'weak_train_role':30,
        'exposed_development_role':12,
        'nearphone_quarantined':21,
        'optimizer_updates':0,
        'new_audio_or_asr_generated':False,
        'models':[{
            'kind':kind,
            'weights_sha256':'a'*64,
            'clip_count':63,
            'streamed_clip_comparisons':315,
            'chunk_samples':list(chain.CHUNKS),
            'max_abs_logit_diff_by_chunk':{str(n):0.00001 for n in chain.CHUNKS},
            'changed_final_classes':0,
            'final_clip_only':True,
            'continuous_event_or_far':False,
            'model_promotion_allowed':False,
        } for kind in chain.KINDS],
        'human_gold':False,
        'formal_training_admitted':False,
        'threshold_selection':False,
        'continuous_wake_event_tested':False,
        'independent_english_frr':'NOT_RUN',
        'continuous_negative_far_per_hour':'NOT_RUN',
        'ssc305_board_qualification':'NOT_RUN',
        'model_selection_made':False,
        'model_promotion_allowed':False,
    }


class EnglishFullChainSourceCases(unittest.TestCase):
    def test_invented_nonpromotable_metadata_passes_source_only(self):
        self.assertTrue(chain.verify_result_metadata(invented_receipt()))

    def test_forged_source_or_model_promotion_fails_closed(self):
        base=invented_receipt()
        changes=(
            lambda d:d.__setitem__('source_tts_zip_sha256','0'*64),
            lambda d:d.__setitem__('source_review_json_sha256','0'*64),
            lambda d:d.__setitem__('runner_sha256','0'*64),
            lambda d:d.__setitem__('saved_training_script_sha256','0'*64),
            lambda d:d.__setitem__('saved_frontend_script_sha256','0'*64),
            lambda d:d.__setitem__('model_promotion_allowed',True),
            lambda d:d.__setitem__('model_selection_made',True),
            lambda d:d.__setitem__('human_gold',True),
            lambda d:d.__setitem__('formal_training_admitted',True),
            lambda d:d.__setitem__('continuous_wake_event_tested',True),
            lambda d:d.__setitem__('threshold_selection',True),
            lambda d:d.__setitem__('continuous_negative_far_per_hour',0),
            lambda d:d.__setitem__('weak_train_role',29),
            lambda d:d['models'][0].__setitem__('model_promotion_allowed',True),
            lambda d:d['models'][0].__setitem__('changed_final_classes',1),
            lambda d:d['models'][0].__setitem__('streamed_clip_comparisons',314),
            lambda d:d['models'][0]['max_abs_logit_diff_by_chunk'].__setitem__('161',0.01),
            lambda d:d['models'][0].__setitem__('kind','fsmn'),
        )
        for alter in changes:
            bad=copy.deepcopy(base)
            alter(bad)
            with self.assertRaises(ValueError):
                chain.verify_result_metadata(bad)

    def test_real_report_if_retained_must_still_be_research_only(self):
        path=HERE/'INTEGRATED_PCM_ENCODER_PARITY.json'
        if path.exists():
            self.assertTrue(chain.verify_result_metadata(json.loads(path.read_text())))
            self.assertFalse(json.loads(path.read_text())['model_promotion_allowed'])


if __name__=='__main__':
    unittest.main()
