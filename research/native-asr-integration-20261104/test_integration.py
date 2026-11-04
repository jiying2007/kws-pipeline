"""Source-only orchestration contracts. No Docker/package/model calls."""
from pathlib import Path
import ast
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'tools'))
import host
import qwen_worker
from balanced_order import ORDINALS,validate_ordinals

class IntegrationTests(unittest.TestCase):
    def test_precommitted_balanced_order(self):
        self.assertEqual(len(ORDINALS),64)
        self.assertEqual(set(ORDINALS),set(range(64)))
        self.assertEqual(list(ORDINALS[:16]),[0,2,4,6,16,18,20,22,32,34,36,38,48,50,52,54])
        for batch in range(4):
            values=ORDINALS[batch*16:(batch+1)*16]
            self.assertEqual([sum(v//16==voice for v in values) for voice in range(4)],[4,4,4,4])
        self.assertTrue(validate_ordinals(list(ORDINALS)))
        with self.assertRaises(ValueError):validate_ordinals(list(range(64)))
        with self.assertRaises(ValueError):validate_ordinals(list(ORDINALS[:-1]))
    def test_batch_local_sort_does_not_mix_global_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';work.mkdir();clips=[]
            for index in range(64):
                digest='%064x'%(64-index)
                path=root/('source%d.wav'%index);path.write_bytes(b'invented')
                clips.append(dict(audio_id='clip-%06d'%(index+1),path=path.name,wav_sha256=digest,
                    source_id='source%d'%index,request_id='request%d'%index,producer_ordinal=ORDINALS[index]))
            batches=host.batch_inputs(root,work,dict(clips=clips))
            for batch,(directory,mapping,rows) in enumerate(batches):
                self.assertEqual([r['audio_id'] for r in rows],['clip-%06d'%i for i in range(1,17)])
                self.assertEqual([r['wav_sha256'] for r in rows],sorted(r['wav_sha256'] for r in clips[batch*16:(batch+1)*16]))
                self.assertEqual({r['global_audio_id'] for r in rows},{r['audio_id'] for r in clips[batch*16:(batch+1)*16]})
                self.assertEqual({r['source_id'] for r in mapping['clips']},{r['source_id'] for r in clips[batch*16:(batch+1)*16]})
                for row in rows:
                    result=host.collect('qwen06',work/'missing',[row],{'recognizers':[{'model_family':'qwen3-asr'}]})
                    self.assertEqual(result[0]['audio_id'],row['global_audio_id'])
    def test_source_pending_input_prevents_launch(self):
        lock=json.loads((HERE/'source-lock.json').read_text());lock['input_lock_sha256']=None
        with self.assertRaisesRegex(ValueError,'input lock'):
            host.verify_lock(host.ROOT,lock)
    def test_partial_completion_keeps_consumed_slot(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);cell=output/'qwen06/strict-native/clip-000001';cell.mkdir(parents=True)
            (cell/'completed.json').write_text('{')
            rows=host.collect('qwen06',output,[dict(audio_id='clip-000001',global_audio_id='clip-000064')],
                {'recognizers':[{'model_family':'qwen3-asr'}]})
            self.assertEqual(rows,[dict(audio_id='clip-000064',status='failed_no_retry')])
    def test_whisper_job_hash_required(self):
        with self.assertRaises(ValueError):host.worker_command('whisper','infer','a'*40,'b'*64)
    def test_actual_worker_commands(self):
        q=host.worker_command('qwen06','infer','a'*40,'b'*64,2,'c'*64)
        self.assertIn('/code/runtime-policy/identity-map-2.json',q)
        self.assertIn('--source-members',q)
        w=host.worker_command('whisper','infer','a'*40,'b'*64,2,'c'*64)
        self.assertEqual(w[w.index('--job-sha256')+1],'c'*64)
        s=host.worker_command('sensevoice','prepare','a'*40)
        self.assertEqual(s[-2:],['--phase','prepare'])
    def test_expired_reserve_zero_docker(self):
        class Boundary:
            def host_capacity(self):raise AssertionError('must not call Docker')
        with patch.object(host.time,'monotonic',return_value=100):
            with self.assertRaises(ValueError):
                host.launch(Boundary(),None,None,None,None,None,None,None,219,None)
    def test_slow_capacity_zero_container(self):
        class Boundary:
            def host_capacity(self):pass
            def container_args(self,*args):raise AssertionError('must not build container')
        with patch.object(host.time,'monotonic',side_effect=[100,201]):
            with self.assertRaises(ValueError):
                host.launch(Boundary(),None,None,None,None,None,None,None,320,None)
    def test_public_aggregate_drops_arbitrary_raw_values(self):
        states={f:[dict(status='not_run',raw_text='SECRET-INVENTED-TEST',native_capture={'a':'NEVER-PUBLISH'})
                   for _ in range(64)] for f in host.FAMILIES}
        with tempfile.TemporaryDirectory() as d:
            host.export_aggregate(Path(d),states,0,'stopped_no_retry')
            raw=(Path(d)/'aggregate-only/status.json').read_text()
            self.assertNotIn('SECRET',raw);self.assertNotIn('NEVER',raw)
            self.assertNotIn('raw_text',raw);self.assertNotIn('native_capture',raw)
            self.assertEqual(len(list((Path(d)/'aggregate-only').iterdir())),1)
            self.assertEqual(json.loads(raw)['planned_asr_slots'],192)
    def test_no_policy_from_missing_real_preparations(self):
        with tempfile.TemporaryDirectory() as d:
            sys.path.insert(0,str(host.ROOT/'tools'))
            with self.assertRaises(FileNotFoundError):
                host.freeze_policy([Path(d)/'missing.json'],Path(d)/'policy.json')
            self.assertFalse((Path(d)/'policy.json').exists())
    def test_blind_identity_map_rejects_target_field(self):
        job=dict(clips=[dict(audio_id='clip-000001',wav_sha256='a'*64)])
        mapping=dict(schema='blind-asr-source-request-map-v1',clips=[dict(audio_id='clip-000001',
            wav_sha256='a'*64,source_id='source',request_id='request',target='not allowed')])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'map.json';p.write_text(json.dumps(mapping))
            with self.assertRaises(ValueError):qwen_worker.requests(p,job)
    def test_incomplete_strict_capture_not_success(self):
        # Absent actual strict completion is not replaced by a reported success.
        policy={'recognizers':[dict(model_family='qwen3-asr')]}
        with tempfile.TemporaryDirectory() as d:
            row=dict(audio_id='clip-000001')
            result=host.collect('qwen06',Path(d),[row],policy)
            self.assertEqual(result,[dict(audio_id='clip-000001',status='not_run')])
            claim=Path(d)/'qwen06/clip-000001.started.json';claim.parent.mkdir();claim.write_text('{}')
            result=host.collect('qwen06',Path(d),[row],policy)
            self.assertEqual(result,[dict(audio_id='clip-000001',status='failed_no_retry')])
    def test_python38_syntax(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text(),feature_version=(3,8))

    def test_concrete_qwen_prepare_stops_before_forward(self):
        sys.path.insert(0,str(host.ROOT/'research/qwen-native-capture-20261104'))
        import capture_worker
        events=[]
        legacy=NS(load_qwen_after_approval=lambda:events.append('load') or NS())
        def retained_run(*args):
            try:legacy.load_qwen_after_approval()
            except qwen_worker.PreparationStopped:
                return dict(attempted=0,success=0,status='stopped')
            raise AssertionError('prepare reached inference loop')
        legacy.run=retained_run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs=root/'input';inputs.mkdir();(inputs/'job.json').write_text('{"clips":[]}')
            args=NS(code=str(host.ROOT),phase='prepare',input=str(inputs),output=str(root/'output'),
                    runtime=str(root/'runtime'),source_members='invented',executed_head='a'*40)
            with patch.object(qwen_worker,'module',return_value=legacy), \
                 patch.object(qwen_worker,'source_bodies',return_value={}), \
                 patch.object(qwen_worker.platform,'python_version',return_value='3.12.14'), \
                 patch.object(capture_worker,'prepare_strict_identity',return_value={'identity':{'invented':True}}):
                self.assertEqual(qwen_worker.execute(args),{'invented':True})
        self.assertEqual(events,['load'])

if __name__=='__main__':unittest.main()
