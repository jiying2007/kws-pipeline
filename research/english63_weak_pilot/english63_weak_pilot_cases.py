"""Source-only, no-Torch/no-real-audio checks for English weak pilot receipts."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0,str(HERE))
import english63_weak_train as pilot


def digest(raw: bytes):
    return hashlib.sha256(raw).hexdigest()


def verify_metadata_receipts(report_override=None):
    report = (json.loads((ROOT/'RESULT.json').read_text()) if report_override is None
              else report_override)
    pilot.require(report['schema']==pilot.POLICY and
                  report['exact_input_sha256']==pilot.INPUT_SHA256 and
                  report['training_script_sha256']==digest((ROOT/'english63_weak_train.py').read_bytes()) and
                  report['source_tts_sha256']==pilot.SOURCE_TTS_SHA256 and
                  report['source_three_evidence_join_sha256']==pilot.SOURCE_REVIEW_SHA256 and
                  report['pretrained_checkpoint_used'] is False and
                  report['training_admitted'] is False and
                  report['human_gold'] is False and
                  report['formal_promotion_allowed'] is False and
                  report['model_selection_made'] is False and
                  report['weak_train_count']==30 and
                  report['exposed_development_count']==12 and
                  report['nearphone_quarantine_count']==21 and
                  report['new_tts_calls']==report['new_asr_calls']==report['new_ipa_calls']==0,
                  'EN_PILOT_PUBLIC_RECEIPT_POLICY_DRIFT')
    pilot.require(type(report['models']) is list and
                  {m['architecture'] for m in report['models']}=={'gru','dsc_tcn'},
                  'EN_PILOT_PUBLIC_MODEL_ROSTER_DRIFT')
    result=[]
    for model in report['models']:
        kind=model['architecture']
        base=kind+'-weights.npz'
        raw=base64.b64decode((ROOT/(base+'.b64')).read_text().strip(),validate=True)
        pilot.require(digest(raw)==model['weights_sha256'] and
                      model['weights_file']==base and
                      0 < len(raw) < 500000,
                      'EN_PILOT_PUBLIC_WEIGHT_HASH_DRIFT')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=z.namelist()
            pilot.require(len(names)==len(set(names)) and
                          len(names)==len(model['tensor_shapes']) and
                          all(name.endswith('.npy') for name in names) and
                          set(name[:-4] for name in names)==set(model['tensor_shapes']) and
                          sum(info.file_size for info in z.infolist()) < 500000,
                          'EN_PILOT_PUBLIC_WEIGHTS_UNSAFE')
        rows=model['per_record']
        pilot.require(type(rows) is list and len(rows)==63 and
                      len({x['cell_id'] for x in rows})==63 and
                      model['updates']==pilot.STEPS and
                      model['class_roster']==pilot.CLASSES and
                      model['human_gold'] is False and
                      model['training_admitted'] is False and
                      model['model_promotion_allowed'] is False and
                      model['continuous_FAR_per_hour']=='NOT_RUN' and
                      model['board_SSC305']=='NOT_RUN',
                      'EN_PILOT_PUBLIC_MODEL_POLICY_DRIFT')
        counts={k:0 for k in pilot.ROLES}
        for r in rows:
            a=r['action']
            pilot.require(a in counts and
                          type(r['cell_id']) is str and pilot.CELL.fullmatch(r['cell_id']) and
                          r['source_family_id'] in {'qwen3-preset-'+v for v in pilot.TRAIN_VOICES | pilot.DEV_VOICES} and
                          r['human_gold'] is False and
                          r['training_admitted'] is False and
                          r['trusted_negative'] is False and
                          r['prediction_is_acoustic_truth'] is False and
                          r['window_or_streaming_event'] is False and
                          r['predicted_class'] in (0,1,2) and
                          len(r['raw_clip_probabilities'])==3 and
                          abs(sum(r['raw_clip_probabilities'])-1)<0.0001 and
                          ((r['requested_weak_intent_id'] is None) if
                           a=='QUARANTINE_AMBIGUOUS_NEARPHONE' else
                           r['requested_weak_intent_id'] in (0,1,2)),
                          'EN_PILOT_PUBLIC_ROW_TRUTH_DRIFT')
            counts[a]+=1
        pilot.require(counts==pilot.EXPECTED_COUNTS,
                      'EN_PILOT_PUBLIC_SPLIT_DRIFT')
        result.append((kind,digest(raw),len(rows)))
    return result


class EnglishWeakPilotSourceCases(unittest.TestCase):
    def test_saved_source_and_weights_are_hash_bound_and_never_promoted(self):
        self.assertEqual(len(verify_metadata_receipts()),2)

    def test_wrong_or_corrupt_audio_archive_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            wrong=Path(temp)/'source.zip'
            wrong.write_bytes(b'invented audio archive')
            with self.assertRaisesRegex(ValueError,'EN_PILOT_ZIP_SHA_CHANGED'):
                pilot.load_strict_inputs(wrong)

    def test_missing_input_zip_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            absent=Path(temp)/'never-created.zip'
            with self.assertRaisesRegex(ValueError,'EN_PILOT_SOURCE_INVALID'):
                pilot.load_strict_inputs(absent)

    def test_changing_promotion_or_source_id_is_detectable(self):
        original=json.loads((ROOT/'RESULT.json').read_text())
        for key,new in [('training_admitted',True),('formal_promotion_allowed',True),
                        ('exact_input_sha256','0'*64)]:
            forged=copy.deepcopy(original)
            forged[key]=new
            with self.assertRaisesRegex(ValueError,'EN_PILOT_PUBLIC_RECEIPT_POLICY_DRIFT'):
                verify_metadata_receipts(forged)

    def test_nearphone_admission_or_weight_byte_drift_is_detected(self):
        original=json.loads((ROOT/'RESULT.json').read_text())
        forged=copy.deepcopy(original)
        target=next(r for r in forged['models'][0]['per_record']
                    if r['action']=='QUARANTINE_AMBIGUOUS_NEARPHONE')
        target['trusted_negative']=True
        with self.assertRaisesRegex(ValueError,'EN_PILOT_PUBLIC_ROW_TRUTH_DRIFT'):
            verify_metadata_receipts(forged)
        forged=copy.deepcopy(original)
        forged['models'][0]['weights_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'EN_PILOT_PUBLIC_WEIGHT_HASH_DRIFT'):
            verify_metadata_receipts(forged)


if __name__=='__main__':
    unittest.main()
