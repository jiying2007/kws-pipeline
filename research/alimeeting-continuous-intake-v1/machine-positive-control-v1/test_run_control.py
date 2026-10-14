"""Offline synthetic-only contracts; never call a model or open real corpora."""
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import wave

spec=importlib.util.spec_from_file_location('control',Path(__file__).with_name('run_control.py'))
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def fixture():
    slots=[];generation=[];results=[]
    for i in range(14):
        sid=f'tqp-qwen06-{i:02d}';target='你好小窝' if i<2 else '小窝小窝'
        slots.append({'sample_id':sid,'three_model_training_candidate':i<13,'human_gold':False,
                      'training_admitted':False,'target_vote_count':3,'audio_quality_pass':True})
        generation.append({'sample_id':sid,'target_text':target,'status':'success','engine':'qwen06',
                           'wav_sha256':'1'*64,'native_wav_sha256':'2'*64})
        for family in sorted(control.FAMILIES):
            results.append({'sample_id':sid,'family':family,'status':'success','completeness':'complete',
                            'quality_flags':[],'attempt':1,'raw_text':target+'。'})
    return {'slots':slots},generation,results


class ControlTests(unittest.TestCase):
    def test_exact_selection_and_denominators(self):
        rows=control.select(*fixture())
        self.assertEqual(len(rows),13)
        self.assertEqual(sum(r['keyword_id']==1 for r in rows),2)
        self.assertTrue(all(r['development_only'] and not r['human_gold'] for r in rows))
    def test_duplicate_slots_rejected(self):
        a,b,c=fixture();a['slots'].append(copy.deepcopy(a['slots'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):control.select(a,b,c)
    def test_duplicate_asr_rejected(self):
        a,b,c=fixture();c.append(copy.deepcopy(c[0]))
        with self.assertRaisesRegex(ValueError,'duplicate ASR'):control.select(a,b,c)
    def test_homophone_not_normalized(self):
        a,b,c=fixture();c[0]['raw_text']='你好小屋'
        with self.assertRaisesRegex(ValueError,'ASR target'):control.select(a,b,c)
    def test_incomplete_and_flagged_rejected(self):
        for key,value in [('completeness','partial'),('quality_flags',['truncated']),('attempt',2)]:
            a,b,c=fixture();c[0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'incomplete'):control.select(a,b,c)
    def test_not_reselect_after_exclusion(self):
        a,b,c=fixture();a['slots'][0]['three_model_training_candidate']=False
        with self.assertRaisesRegex(ValueError,'exactly 13'):control.select(a,b,c)
    def test_wrong_keyword_mix(self):
        rows=[{'target_text':'你好小窝','matching_event_count':1,'event_count':2,'wrong_keyword_event_count':1},
              {'target_text':'你好小窝','matching_event_count':0,'event_count':0,'wrong_keyword_event_count':0},
              {'target_text':'小窝小窝','matching_event_count':0,'event_count':1,'wrong_keyword_event_count':1}]
        result=control.coverage(rows)
        self.assertEqual(result['你好小窝']['trigger_coverage'],0.5)
        self.assertEqual(result['小窝小窝']['clips_with_wrong_keyword_trigger'],1)
        self.assertEqual(result['小窝小窝']['trigger_coverage'],0)
    def test_pcm_contract(self):
        body=io.BytesIO()
        with wave.open(body,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\x00\x00'*1600)
        count,digest=control.inspect_wav(body.getvalue())
        self.assertEqual(count,1600);self.assertEqual(len(digest),64)
    def test_stereo_rejected(self):
        body=io.BytesIO()
        with wave.open(body,'wb') as wav:
            wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\x00'*6400)
        with self.assertRaisesRegex(ValueError,'mono PCM16'):control.inspect_wav(body.getvalue())
    def test_requires_execute_and_exclusive_output(self):
        with mock.patch.object(control,'prepare') as prepare, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'--execute'):control.run('absent','absent',Path(tmp)/'fresh')
            with self.assertRaisesRegex(ValueError,'output exists'):control.run('absent','absent',tmp,True)
            prepare.assert_not_called()
    def test_plain_stream_frame_validator(self):
        samples=1600;stats={'processed_samples':samples,'processed_frames':4,'detections':0,
                           'discontinuities':0,'lost_samples':0,'external_vad_frames':0}
        control.diagnostic.validate_outputs('control',samples,[],stats)
        stats['processed_frames']=0
        with self.assertRaisesRegex(ValueError,'processed_frames'):control.diagnostic.validate_outputs('control',samples,[],stats)

if __name__=='__main__':unittest.main()
