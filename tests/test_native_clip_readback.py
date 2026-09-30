#!/usr/bin/env python3
"""Transport/readback fixtures only; data catalog semantic validation stays in kws-data."""
import copy
import json
import pathlib
import sys
import subprocess
import tempfile
import unittest
import wave
from unittest import mock

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
import readback_native_clips as consumer


def row(name='p', kind='positive', keyword=1):
    return dict(recording=name,kind=kind,keyword_id=keyword,duration_s=1.0,
                dataset_id='fixture',split='development_a',review_method='human',
                review_evidence_class='speech-like-audio-review-v1',observed_development=True)


def event(name='p', keyword=1, time=.4):
    return dict(recording=name,keyword_id=keyword,time_s=time,confidence=.9)


class ClipReadbackTests(unittest.TestCase):
    def test_zero_events_are_misses_not_unprocessed(self):
        report=consumer.summarize([row(),row('n','confusable',None)],[])
        self.assertTrue(report['recordings'][0]['target_miss'])
        self.assertEqual(report['recordings'][1]['confusable_event_count'],0)
        self.assertFalse(report['qualification_allowed'])
        self.assertNotIn('far_per_hour', report)
        self.assertNotIn('endpoint_latency',report)

    def test_wrong_and_additional_target_events(self):
        report=consumer.summarize([row(),row('n','confusable',None)],
                                  [event(),event(time=.5),event(keyword=2),event('n',2)])
        p,n=report['recordings']
        self.assertTrue(p['target_hit'])
        self.assertEqual((p['target_event_count'],p['additional_target_events'],p['wrong_keyword_events']), (2,1,1))
        self.assertEqual(n['confusable_event_count'],1)
        report=consumer.summarize([row()],[event(keyword=2)])
        self.assertTrue(report['recordings'][0]['target_miss'])

    def test_bad_detections_fail_closed(self):
        for bad in [event('unknown'),event(keyword=True),event(keyword=1.0),event(keyword=2**32),event(time=float('nan')),
                    event(time=-.1),event(time=1.1),dict(event(),confidence=2),dict(event(),confidence=True)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                consumer.summarize([row()],[bad])

    def test_native_roles_and_review_methods_preserved(self):
        rows=[row(),dict(row('asr'),review_method='asr',review_evidence_class='speech-like-asr-review-v1',split='development_b')]
        report=consumer.summarize(rows,[])
        self.assertEqual(len(report['groups']),2)
        self.assertEqual([r['split'] for r in report['recordings']],['development_a','development_b'])
        self.assertEqual([r['review_method'] for r in report['recordings']],['human','asr'])

    def make_export(self,root):
        (root/'catalog.json').write_text('{}\n')
        with wave.open(str(root/'fixture.wav'),'wb') as f:
            f.setparams((1,2,16000,0,'NONE','not compressed')); f.writeframes(b'\x01\x00'*16000)
        r={**row(),**consumer.inspect_pcm16_wav(root/'fixture.wav'), 'path':'fixture.wav'}
        identity=dict(dataset_id='fixture',manifest_sha256='1'*64,review_sha256='2'*64,splits_sha256='3'*64)
        bundle=dict(dataset_id='fixture',identity=identity,content_id='sha256:'+consumer.digest(identity),
                    qualification_allowed=False,recordings_sha256=consumer.digest([r]),recordings=[r])
        return dict(schema_version=1,evidence_class='kws-data-native-consumer-receipt-v1',
                    data_repository_commit='a'*40,catalog_sha256=consumer.sha(root/'catalog.json'),
                    qualification_allowed=False,labels='native-text-no-token-or-event-alignment',datasets=[bundle])

    def test_export_pins_and_consumed_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); receipt=self.make_export(root); path=root/'receipt.json'
            def load(expected_sha=None):
                path.write_text(json.dumps(receipt))
                with mock.patch.object(consumer.subprocess,'check_output',side_effect=['a'*40+'\n','']):
                    return consumer.load_export(path,root,expected_receipt_sha256=expected_sha or consumer.sha(path),
                      expected_commit='a'*40,expected_catalog_sha256=consumer.sha(root/'catalog.json'),expected_datasets=['fixture'])
            _,rows=load()
            self.assertEqual(rows[0]['path'],'fixture.wav')
            self.assertFalse(rows[0]['event_annotations_available'])
            self.assertNotIn('expected',rows[0])
            self.assertNotIn('tokens',rows[0])
            with self.assertRaisesRegex(ValueError,'receipt SHA'): load('0'*64)
            receipt['data_repository_commit']='b'*40
            with self.assertRaisesRegex(ValueError,'data identity'):load()
            receipt['data_repository_commit']='a'*40
            receipt['datasets'][0]['recordings'][0]['frames']=1
            with self.assertRaisesRegex(ValueError,'row digest'):load()
            receipt['datasets'][0]['recordings_sha256']=consumer.digest(receipt['datasets'][0]['recordings'])
            with self.assertRaisesRegex(ValueError,'WAV identity'):load()

    def test_reject_token_and_event_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); original=self.make_export(root); path=root/'receipt.json'
            for field in ('expected','tokens','token_ids','start_s','end_s','match_not_before_s','start_sample','end_sample'):
                with self.subTest(field=field):
                    receipt=copy.deepcopy(original)
                    receipt['datasets'][0]['recordings'][0][field]=[]
                    receipt['datasets'][0]['recordings_sha256']=consumer.digest(receipt['datasets'][0]['recordings'])
                    path.write_text(json.dumps(receipt))
                    with mock.patch.object(consumer.subprocess,'check_output',side_effect=['a'*40+'\n','']):
                        with self.assertRaisesRegex(ValueError,'annotations are not supported'):
                            consumer.load_export(path,root,expected_receipt_sha256=consumer.sha(path),expected_commit='a'*40,
                                expected_catalog_sha256=consumer.sha(root/'catalog.json'),expected_datasets=['fixture'])

    def test_cli_end_to_end_execution_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); data=root/'data'; data.mkdir()
            receipt=self.make_export(data)
            subprocess.run(['git','init','-q',str(data)],check=True)
            subprocess.run(['git','-C',str(data),'add','.'],check=True)
            subprocess.run(['git','-C',str(data),'-c','user.name=fixture','-c','user.email=fixture@example.invalid',
                            'commit','-qm','synthetic transport fixture'],check=True)
            commit=subprocess.check_output(['git','-C',str(data),'rev-parse','HEAD'],text=True).strip()
            receipt['data_repository_commit']=commit
            receipt_path=root/'receipt.json';receipt_path.write_text(json.dumps(receipt))
            runner=root/'fixture-runner'
            runner.write_text('#!'+sys.executable+'\nimport json,sys\nprint(json.dumps(dict(recording=sys.argv[4],keyword_id=2,time_s=.4,confidence=.8)))\n')
            runner.chmod(0o755)
            model=root/'fixture-model';model.write_bytes(b'model-fixture')
            keywords=root/'fixture-keywords';keywords.write_bytes(b'keywords-fixture')
            build=root/'fixture-build-receipt.json';build.write_text('{"fixture":true}\n')
            output=root/'output'
            command=[sys.executable,str(ROOT/'eval/readback_native_clips.py'),
                '--receipt',str(receipt_path),'--data-root',str(data),
                '--expected-receipt-sha256',consumer.sha(receipt_path),'--expected-data-commit',commit,
                '--expected-catalog-sha256',consumer.sha(data/'catalog.json'),'--dataset','fixture',
                '--runner',str(runner),'--expected-runner-sha256',consumer.sha(runner),
                '--runner-build-receipt',str(build),'--model',str(model),'--expected-model-sha256',consumer.sha(model),
                '--keywords',str(keywords),'--expected-keywords-sha256',consumer.sha(keywords),'--output-root',str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/'clip-readback.json').read_text())
            self.assertTrue(report['recordings'][0]['target_miss'])
            self.assertEqual(report['recordings'][0]['wrong_keyword_events'],1)
            inputs=json.loads((output/'execution-inputs.jsonl').read_text())
            self.assertNotIn('expected',inputs)
            self.assertFalse(inputs['event_annotations_available'])
            self.assertEqual(report['data_export_identity']['data_repository_commit'],commit)
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)
            self.assertFalse(subprocess.check_output(['git','-C',str(data),'status','--porcelain'],text=True))

    def test_path_escape_and_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);receipt=self.make_export(root);path=root/'receipt.json'
            for unsafe in ('../fixture.wav','/tmp/fixture.wav'):
                receipt['datasets'][0]['recordings'][0]['path']=unsafe
                receipt['datasets'][0]['recordings_sha256']=consumer.digest(receipt['datasets'][0]['recordings'])
                path.write_text(json.dumps(receipt))
                with mock.patch.object(consumer.subprocess,'check_output',side_effect=['a'*40+'\n','']):
                    with self.assertRaisesRegex(ValueError,'unsafe export path'):
                        consumer.load_export(path,root,expected_receipt_sha256=consumer.sha(path),expected_commit='a'*40,
                            expected_catalog_sha256=consumer.sha(root/'catalog.json'),expected_datasets=['fixture'])
            with mock.patch.object(consumer.subprocess,'check_output',side_effect=['a'*40+'\n',' M catalog.json']):
                with self.assertRaisesRegex(ValueError,'clean data checkout'):
                    consumer.load_export(path,root,expected_receipt_sha256=consumer.sha(path),expected_commit='a'*40,
                        expected_catalog_sha256=consumer.sha(root/'catalog.json'),expected_datasets=['fixture'])


if __name__=='__main__': unittest.main()
