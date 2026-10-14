import copy,json,unittest
from pathlib import Path
from ab_data_admission import assess
HERE=Path(__file__).resolve().parent

def fixture(voices):
    report={'slots':[]};generation=[];plan={'qwen06':[]};native={'outcomes':[]}
    for i,voice in enumerate(voices):
        sid='sample-'+str(i);word='你好小窝'
        report['slots'].append(dict(sample_id=sid,engine='qwen06',three_model_training_candidate=True,human_gold=False,training_admitted=False))
        generation.append(dict(sample_id=sid,engine='qwen06',status='success',target_text=word,native_wav_sha256=str(i),wav_sha256='derived'+str(i)))
        plan['qwen06'].append(dict(cell_id=sid,engine='qwen06',speaker=voice,text=word))
        native['outcomes'].append(dict(cell_id=sid,engine='qwen06',speaker=voice,text=word,status='generated',attempts=1,raw_wav_sha256=str(i)))
    return report,generation,plan,[native]

class AdmissionTests(unittest.TestCase):
    def test_two_voices_cannot_fill_three_disjoint_roles(self):
        r=assess(*fixture(['Serena','Vivian']))
        self.assertEqual(r['status'],'BLOCKED_INSUFFICIENT_KW1_GROUPS')
        self.assertFalse(r['split_assignment_performed']);self.assertFalse(r['heldout_reuse_allowed'])
    def test_repeats_do_not_increase_independent_groups(self):
        r=assess(*fixture(['Serena','Serena','Serena']))
        self.assertEqual(r['coverage']['你好小窝']['candidate_count'],3)
        self.assertEqual(len(r['coverage']['你好小窝']['independent_voice_groups']),1)
        self.assertEqual(r['status'],'BLOCKED_INSUFFICIENT_KW1_GROUPS')
    def test_native_speaker_or_hash_mismatch_rejected(self):
        for field,value in [('speaker','Other'),('raw_wav_sha256','wrong')]:
            args=fixture(['Serena']);args[3][0]['outcomes'][0][field]=value
            with self.assertRaises(ValueError):assess(*args)
    def test_retained_real_data_boundary_and_numeric_status(self):
        r=json.loads((HERE/'ab-data-admission.json').read_bytes())
        self.assertEqual(r['status'],'BLOCKED_INSUFFICIENT_KW1_GROUPS')
        self.assertEqual({(x['sample_id'],x['speaker']) for x in r['candidates'] if x['target_text']=='你好小窝'},{('tqp-qwen06-05','Serena'),('tqp-qwen17-01','Vivian')})
        self.assertEqual(r['matched_quality_ab_status'],'NOT_RUN');self.assertEqual(r['numeric_fixture_status'],'PASS')
        self.assertFalse(r['labels_changed']);self.assertFalse(r['training_admitted'])
if __name__=='__main__':unittest.main()
