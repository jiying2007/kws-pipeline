from pathlib import Path
import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
import wave
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import campaign_contract as c
from cpu_budget import CpuLedger,parse_cpu_stat,inference_filter
import hosted_run as host


class ContractTests(unittest.TestCase):
    def setUp(self):self.public=c.read_json(HERE/'public-inputs.json')
    def test_exact_public_projection(self):self.assertEqual(len(c.validate_public(self.public)),12)
    def test_private_label_injection_rejected(self):
        for key in ('local_wav_path','intended_text','voice','historical_keyword_id'):
            bad=copy.deepcopy(self.public);bad['records'][0][key]='x'
            with self.assertRaises(ValueError):c.validate_public(bad)
    def test_remote_commit_drift(self):
        self.public['records'][0]['commit']='0'*40
        with self.assertRaises(ValueError):c.validate_public(self.public)
    def test_duplicate_audio(self):
        self.public['records'][1]=self.public['records'][0]
        with self.assertRaises(ValueError):c.validate_public(self.public)
    def test_reorder_rejected(self):
        self.public['records'].reverse()
        with self.assertRaises(ValueError):c.validate_public(self.public)
    def test_qwen16_cannot_pass(self):
        self.public['count']=16
        with self.assertRaises(ValueError):c.validate_public(self.public)
    def test_anonymous_only(self):
        job=c.blind_job(c.validate_public(self.public))
        self.assertEqual(set(job),{'schema','clips'})
        for r in job['clips']:self.assertEqual(set(r),{'audio_id','audio_path','wav_sha256'})
    def test_full_frame_hash(self):
        b=io.BytesIO()
        with wave.open(b,'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(b'\1\0'*100)
        raw=b.getvalue();r={'bytes':len(raw),'sha256':c.digest(raw),'git_blob_sha1':c.hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'frames':100,'pcm_sha256':c.digest(b'\1\0'*100)}
        self.assertEqual(len(c.validate_wav(raw,r)),200)
        r['pcm_sha256']='0'*64
        with self.assertRaises(ValueError):c.validate_wav(raw,r)
    def test_release_disabled_before_any_effect(self):
        with self.assertRaises(ValueError):c.fresh_run({}, {}, c.read_json(HERE/'execution-release.json'))
    def test_real_run_number_preserved(self):
        env={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/'+c.BRANCH,'GITHUB_EVENT_NAME':'push','GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_NUMBER':'7','GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40}
        event={'created':True,'repository':{'private':False},'after':'a'*40}
        release={k:True for k in ('enabled','new_transcripts_publication_approved','budget_review_approved','source_review_approved')}
        self.assertEqual(c.fresh_run(env,event,release)['run_number'],7)
        env['GITHUB_RUN_ATTEMPT']='2'
        with self.assertRaises(ValueError):c.fresh_run(env,event,release)
    def test_all36_missing_remain_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            rows=c.denominator(c.blind_job(self.public['records']),d)
            self.assertEqual(len(rows),36);self.assertTrue(all(r['status']=='not_run' for r in rows))
    def test_consumed_attempt_stays_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'qwen06'/'clip-000001';p.mkdir(parents=True);c.write_once(p/'started.json',{})
            rows=c.denominator(c.blind_job(self.public['records']),d)
            self.assertEqual(rows[0]['attempts'],1);self.assertEqual(rows[0]['status'],'error')
    def test_budget_reservations_less_than6gib(self):self.assertLess(c.LIMITS['acquisition_reserved_bytes'],6*c.GIB)
    def test_json_duplicate_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for text in ('{"x":1,"x":2}','{"x":NaN}'):
                p.write_text(text)
                with self.assertRaises(ValueError):c.read_json(p)
    def test_file_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_text('{}');(p/'alias').symlink_to('x')
            with self.assertRaises(ValueError):c.read_json(p/'alias')
    def test_no_model_import_on_contract(self):
        self.assertFalse(any(x in sys.modules for x in ('torch','transformers','funasr','whisper')))
    def test_dormant_workflow_no_artifact_action(self):
        text=(HERE/'workflow.yml').read_text()
        self.assertIn('false &&',text);self.assertIn('runs-on: ubuntu-24.04',text)
        self.assertNotIn('uses: actions/upload-artifact',text);self.assertNotIn('uses: actions/cache',text)
    def test_gate_unique_and_no_path_injection(self):
        with tempfile.TemporaryDirectory() as d:
            docker=host.MeteredDocker(d,999999999999)
            docker.gate('once')
            with self.assertRaises(ValueError):docker.gate('once')
            with self.assertRaises(ValueError):docker.gate('../escape')
    def test_unapproved_output_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weight.pt').write_bytes(b'secret');(root/'audio.wav').write_bytes(b'private')
            c.write_once(root/'host-receipt.json',{'schema':'x'})
            self.assertEqual(host.collect(root),['host-receipt.json'])

    def test_whisper_setup_global_output_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);docker=host.MeteredDocker(root,999999999999)
            (root/'whisper-setup').mkdir()
            with (root/'whisper-setup'/'large.json').open('wb') as f:f.truncate(c.LIMITS['output_bytes']+1)
            with self.assertRaisesRegex(ValueError,'output stop threshold'):docker.check()
    def test_fresh_load_failure_retains_prepared_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'frozen/qwen06';f.mkdir(parents=True)
            (root/'output/qwen06').mkdir(parents=True) # load failed before new locks
            body=f/'retained-source/code/example.py';body.parent.mkdir(parents=True);body.write_bytes(b'x=1')
            c.write_once(f/'native-source-lock.json',{'files':[{'path':'output/qwen06/retained-source/code/example.py',**c.identity(body)}]})
            result=host.collect(root)
            self.assertIn('frozen/qwen06/retained-source/code/example.py',result)
            self.assertIn('frozen/qwen06/native-source-lock.json',result)
            self.assertFalse(any(n.endswith('started.json') for n in result))


class OrchestrationTests(unittest.TestCase):
    def test_three_prepare_policy_three_decode(self):
        calls=[];clock=[0]
        def prepare(f,remaining):calls.append(('prepare',f));clock[0]+=10
        def freeze():calls.append(('freeze',))
        def decode(f,remaining):calls.append(('decode',f));self.assertEqual(remaining,590);clock[0]+=20
        result=host.recognition_stages(prepare,freeze,decode,setup_seconds=100,clock=lambda:clock[0])
        self.assertEqual(calls,[('prepare',f) for f in c.FAMILIES]+[('freeze',)]+[('decode',f) for f in c.FAMILIES])
        self.assertEqual(result['combined_setup_seconds'],130)
        self.assertEqual(result['engine_wall_seconds'],dict.fromkeys(c.FAMILIES,30))
    def test_prepare_failure_never_freezes_or_decodes(self):
        calls=[]
        def prepare(f,remaining):calls.append(f);raise ValueError('fixture')
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:self.fail('frozen'),lambda *a:self.fail('decoded'))
        self.assertEqual(calls,['qwen06'])
    def test_budget_failure_never_decodes(self):
        def freeze():raise ValueError('package cap')
        with self.assertRaises(ValueError):host.recognition_stages(lambda *a:None,freeze,lambda *a:self.fail('decoded'))
    def test_combined_family_wall_no_reset(self):
        clock=[0];calls=[]
        def prepare(f,remaining):clock[0]+=599
        def decode(f,remaining):calls.append(f);self.assertEqual(remaining,1);clock[0]+=2
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:None,decode,clock=lambda:clock[0])
        self.assertEqual(calls,['qwen06'])
    def test_prepare_counted_against_setup_budget(self):
        clock=[0]
        def prepare(f,remaining):clock[0]+=51
        with self.assertRaises(ValueError):host.recognition_stages(prepare,lambda:self.fail('frozen'),lambda *a:None,setup_seconds=1750,clock=lambda:clock[0])


class CpuTests(unittest.TestCase):
    def test_stat_parse(self):self.assertEqual(parse_cpu_stat('usage_usec 123\nuser_usec 1\nsystem_usec 122\n'),123)
    def test_bad_stat(self):
        for value in ('usage_usec -1','usage_usec 1\nusage_usec 2','user_usec 1','usage_usec NaN'):
            with self.assertRaises(ValueError):parse_cpu_stat(value)
    def test_cpu_reset_rejected(self):
        value=[4];ledger=CpuLedger();ledger.register('a','setup',lambda:value[0]);value[0]=3
        with self.assertRaises(ValueError):ledger.sample()
    def test_aggregate_threads_and_children_accounted(self):
        value=[0];ledger=CpuLedger();ledger.register('a','qwen06',lambda:value[0]);value[0]=600000001
        self.assertEqual(ledger.sample(),['qwen06']);self.assertTrue(ledger.receipt()['threshold_exceeded'])
    def test_no_reuse_or_false_hardclaim(self):
        ledger=CpuLedger();ledger.register('a','setup',lambda:5);ledger.close('a',10)
        self.assertTrue(ledger.receipt()['all_final_usage_verified']);self.assertFalse(ledger.receipt()['hard_cumulative_kernel_quota'])
        with self.assertRaises(ValueError):ledger.register('a','setup',lambda:0)
    def test_unknown_final_stays_unknown(self):
        ledger=CpuLedger();ledger.register('a','setup',lambda:5)
        self.assertFalse(ledger.receipt()['all_final_usage_verified'])
    def test_final_close_only_exceed_still_removes_then_fails(self):
        with tempfile.TemporaryDirectory() as d:
            docker=host.MeteredDocker(d,999999999999);docker.gate('last-whisper')
            row=docker.names['last-whisper'];row.update(created=True,registered=True,reader=lambda:1800000001)
            docker.ledger.register('last-whisper','whisper',lambda:1799999999)
            calls=[]
            def raw(*args,**kwargs):
                calls.append(args)
                return '{"Paused":true}' if args[0]=='inspect' else ''
            docker.raw=raw
            with self.assertRaises(ValueError):docker.remove('last-whisper')
            self.assertTrue(row['removed']);self.assertIn(('rm','--force','last-whisper'),calls)
            with self.assertRaises(ValueError):docker.finish()

    def test_seccomp_syscall_dispatch(self):
        def run(nr,flags=0,arch=0xc000003e):
            code=inference_filter();pc=0;acc=0
            while True:
                op,jt,jf,k=code[pc]
                if op==0x20:acc={0:nr,4:arch,16:flags}[k];pc+=1
                elif op==0x15:pc+=1+(jt if acc==k else jf)
                elif op==0x45:pc+=1+(jt if acc&k else jf)
                elif op==0x06:return k
                else:self.fail('unknown BPF')
        for nr in (57,58,56):self.assertEqual(run(nr)&0xffff0000,0x00050000)
        self.assertEqual(run(56,0x10000),0x7fff0000)
        self.assertEqual(run(435),0x00050000|38)
        self.assertEqual(run(0),0x7fff0000)
        for nr in (0x40000000|56,0x40000000|57,0x40000000|58):self.assertEqual(run(nr),0x80000000)
        self.assertEqual(run(0,arch=0x40000003),0x80000000)


if __name__=='__main__':unittest.main()
