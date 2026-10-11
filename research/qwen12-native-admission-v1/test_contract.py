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
    def test_pre_model_guard_and_failure_records_retained_with_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime';runtime.mkdir()
            c.write_once(runtime/'setup-receipt.json',{'commands':[]})
            names=('import-process-guard.json','import-optional-tool-policy.json','import-preflight-failure.json')
            for name in names:c.write_once(runtime/name,{'schema':'synthetic-'+name})
            host.asr_setup_projection(runtime,root/'public.json')
            value=c.read_json(root/'public.json')['pre_model_process_compatibility']
            self.assertEqual(set(value),set(names))
            for name in names:
                self.assertEqual({k:value[name][k] for k in ('bytes','sha256')},c.identity(runtime/name))
                self.assertEqual(value[name]['record'],c.read_json(runtime/name))

    def test_optional_policy_closure_is_exact_and_dynamic_receipt_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'output/sensevoice';(out/'retained-source').mkdir(parents=True)
            member='output/sensevoice/retained-source/optional-tool-policy.json'
            c.write_once(root/member,{'schema':'synthetic-stable-policy'})
            row={'path':member,**c.identity(root/member),'role':'runtime-execution-policy'}
            c.write_once(out/'native-source-lock.json',{'files':[row]})
            c.write_once(out/'optional-tool-probes.json',{'schema':'synthetic-dynamic-probes'})
            self.assertEqual(set(host.collect(root)),{member,'output/sensevoice/native-source-lock.json',
                'output/sensevoice/optional-tool-probes.json'})
            row['role']='unapproved'
            (out/'native-source-lock.json').write_bytes(c.canonical({'files':[row]}))
            with self.assertRaises(ValueError):host.collect(root)

    def test_release_disabled_before_any_effect(self):
        _,_,release=self.dispatch_fixture();release['enabled']=False
        with self.assertRaisesRegex(ValueError,'dispatch approval'):c.fresh_run({}, {}, release)
    def test_real_run_number_preserved(self):
        env={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/'+c.BRANCH,'GITHUB_EVENT_NAME':'push','GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_NUMBER':'7','GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40}
        event={'created':True,'repository':{'private':False},'after':'a'*40}
        release={k:True for k in c.DISPATCH_APPROVALS}
        release['predecessor_zero_recognition']=copy.deepcopy(c.PREDECESSOR_ZERO_RECOGNITION)
        release['required_runtime_preconditions']=list(c.REQUIRED_RUNTIME_PRECONDITIONS)
        release['evidence_transport']='github-repository-zip-chunks-v1'
        self.assertEqual(c.fresh_run(env,event,release)['run_number'],7)
        env['GITHUB_RUN_ATTEMPT']='2'
        with self.assertRaises(ValueError):c.fresh_run(env,event,release)
    def dispatch_fixture(self):
        env={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/'+c.BRANCH,
            'GITHUB_EVENT_NAME':'push','GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_NUMBER':'7',
            'GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40}
        event={'created':True,'repository':{'private':False},'after':'a'*40}
        release={k:True for k in c.DISPATCH_APPROVALS}
        release['predecessor_zero_recognition']=copy.deepcopy(c.PREDECESSOR_ZERO_RECOGNITION)
        release['required_runtime_preconditions']=list(c.REQUIRED_RUNTIME_PRECONDITIONS)
        release['evidence_transport']='github-repository-zip-chunks-v1'
        return env,event,release
    def test_recovery_predecessor_is_fixed_and_original_ref_rejected(self):
        for value in (None, {}, True, dict(c.PREDECESSOR_ZERO_RECOGNITION, started=True),
                      dict(c.PREDECESSOR_ZERO_RECOGNITION, started=1),
                      dict(c.PREDECESSOR_ZERO_RECOGNITION, cross_execution_recognition_limit=72)):
            env,event,release=self.dispatch_fixture();release['predecessor_zero_recognition']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture()
        env['GITHUB_REF']='refs/heads/research/qwen12-native-admission-v1'
        with self.assertRaises(ValueError):c.fresh_run(env,event,release)

    def test_all_dispatch_approvals_require_literal_true(self):
        for key in c.DISPATCH_APPROVALS:
            for value in (False,0,1,None,'true',[],{}):
                env,event,release=self.dispatch_fixture();release[key]=value
                with self.subTest(key=key,value=value), self.assertRaisesRegex(ValueError,'dispatch approval'):
                    c.fresh_run(env,event,release)
            env,event,release=self.dispatch_fixture();del release[key]
            with self.subTest(key=key,missing=True), self.assertRaisesRegex(ValueError,'dispatch approval'):
                c.fresh_run(env,event,release)
    def test_runtime_preconditions_are_exact_obligations_not_pass_flags(self):
        expected=list(c.REQUIRED_RUNTIME_PRECONDITIONS)
        for value in (None,True,False,{},tuple(expected),'passed',expected[:-1],expected+['passed'],
                      expected[::-1],expected[:-1]+[True],dict.fromkeys(expected,True)):
            env,event,release=self.dispatch_fixture();release['required_runtime_preconditions']=value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'runtime preconditions'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture();del release['required_runtime_preconditions']
        with self.assertRaisesRegex(ValueError,'runtime preconditions'):c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture();release['activation_blockers']=[]
        with self.assertRaisesRegex(ValueError,'runtime preconditions'):c.fresh_run(env,event,release)
    @staticmethod
    def native_gate(enabled):
        # Only the separately reviewed literal false differs between source states.
        return ("    if: ${{ "+('' if enabled else 'false && ')+
            "github.repository == 'jiying2007/kws-pipeline' && github.event_name == 'push' && "+
            "github.ref == 'refs/heads/"+c.BRANCH+"' && github.event.created == true && "+
            "github.event.repository.private == false && github.run_attempt == 1 }}")
    def assert_source_activation_state(self,release,text):
        # Test-only source consistency, never approval or proof of live runtime checks.
        self.assertIs(type(release),dict);self.assertIs(type(text),str)
        for key in c.DISPATCH_APPROVALS:self.assertIs(type(release.get(key)),bool,key)
        self.assertIs(release['new_transcripts_publication_approved'],True)
        if release['enabled']:
            for key in c.DISPATCH_APPROVALS:self.assertIs(release[key],True,key)
        self.assertEqual(text.count('  qwen12:\n'),1)
        self.assertEqual(text.count('  contracts:\n'),1)
        native=text.split('  qwen12:\n',1)[1].split('  contracts:\n',1)[0]
        for field,expected in (('if',self.native_gate(release['enabled'])),
                ('needs','    needs: contracts'),('runs-on','    runs-on: ubuntu-24.04'),
                ('timeout-minutes','    timeout-minutes: 70')):
            self.assertEqual([line for line in native.splitlines() if line.startswith('    '+field+':')],
                             [expected])
        self.assertEqual(release['required_runtime_preconditions'],list(c.REQUIRED_RUNTIME_PRECONDITIONS))
        self.assertEqual(release['evidence_transport'],'github-repository-zip-chunks-v1')
        self.assertNotIn('activation_blockers',release)
        self.assertNotIn('maximum_log_api_readback_approved',release)
    def source_state_fixture(self,enabled):
        # Construct both states only in memory, independent of the committed state.
        release=c.read_json(HERE/'execution-release.json')
        release.update(dict.fromkeys(c.DISPATCH_APPROVALS,enabled))
        release['new_transcripts_publication_approved']=True
        text=(HERE/'workflow.yml').read_text()
        text=text.replace(self.native_gate(not enabled),self.native_gate(enabled))
        return release,text
    def test_committed_release_and_workflow_have_consistent_reviewed_state(self):
        self.assert_source_activation_state(c.read_json(HERE/'execution-release.json'),
                                            (HERE/'workflow.yml').read_text())
    def test_disabled_and_activated_source_fixtures(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled):self.assert_source_activation_state(*self.source_state_fixture(enabled))
    def test_source_approvals_must_be_present_and_strict_booleans(self):
        for enabled in (False,True):
            for key in c.DISPATCH_APPROVALS:
                for value in (None,0,1,'false','true',[],{}):
                    release,text=self.source_state_fixture(enabled);release[key]=value
                    with self.subTest(enabled=enabled,key=key,value=value),self.assertRaises(AssertionError):
                        self.assert_source_activation_state(release,text)
                release,text=self.source_state_fixture(enabled);del release[key]
                with self.subTest(enabled=enabled,key=key,missing=True),self.assertRaises(AssertionError):
                    self.assert_source_activation_state(release,text)
        for value in (None,False,1,'approved',[]):
            with self.subTest(release=value),self.assertRaises(AssertionError):
                self.assert_source_activation_state(value,self.source_state_fixture(False)[1])
    def test_activated_source_requires_every_approval(self):
        for key in c.DISPATCH_APPROVALS:
            release,text=self.source_state_fixture(True);release[key]=False
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.assert_source_activation_state(release,text)
        release,text=self.source_state_fixture(False);release['new_transcripts_publication_approved']=False
        with self.assertRaises(AssertionError):self.assert_source_activation_state(release,text)
    def test_source_rejects_mixed_activation_gates(self):
        for enabled in (False,True):
            release,_=self.source_state_fixture(enabled);_,text=self.source_state_fixture(not enabled)
            with self.subTest(enabled=enabled),self.assertRaises(AssertionError):
                self.assert_source_activation_state(release,text)
    def test_source_rejects_missing_or_weakened_native_workflow_guards(self):
        for enabled in (False,True):
            release,text=self.source_state_fixture(enabled);gate=self.native_gate(enabled)
            mutations=[(gate,''),(gate,gate+'\n'+gate),(gate,gate.replace(' }}',' || true }}')),
                ("github.repository == 'jiying2007/kws-pipeline'","github.repository == 'other/repository'"),
                ("github.event_name == 'push'","github.event_name == 'workflow_dispatch'"),
                ("refs/heads/"+c.BRANCH,"refs/heads/main"),('github.event.created == true','github.event.created == 1'),
                ('github.event.repository.private == false','github.event.repository.private == 0'),
                ('github.run_attempt == 1','github.run_attempt == 2'),('    needs: contracts','    needs: []'),
                ('    runs-on: ubuntu-24.04','    runs-on: self-hosted'),
                ('    timeout-minutes: 70','    timeout-minutes: 71')]
            for before,after in mutations:
                with self.subTest(enabled=enabled,guard=before),self.assertRaises(AssertionError):
                    self.assert_source_activation_state(release,text.replace(before,after,1))
    def test_source_freeze_binds_current_files(self):
        import prepare_freeze
        freeze=c.read_json(HERE/'execution-freeze.json')
        self.assertEqual(freeze,prepare_freeze.build())
        self.assertEqual(c.read_json(HERE/'execution-release.json')['source_freeze_sha256'],
                         c.identity(HERE/'execution-freeze.json')['sha256'])
    def test_selected_repository_transport_rejects_log_claims(self):
        for value in (None,False,True,1,{},'github-job-log-chunks-v1'):
            env,event,release=self.dispatch_fixture();release['evidence_transport']=value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'runtime preconditions'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture();release['maximum_log_api_readback_approved']=True
        with self.assertRaisesRegex(ValueError,'runtime preconditions'):c.fresh_run(env,event,release)
    def test_publication_jobs_are_fixed_push_only_and_depend_on_contracts(self):
        text=(HERE/'workflow.yml').read_text()
        self.assertEqual(text,(HERE.parents[1]/'.github/workflows/qwen12-native-admission-v1.yml').read_text())
        native=text.split('  qwen12:\n',1)[1].split('  contracts:\n',1)[0]
        fixture=text.split('  transport-fixture:\n',1)[1]
        for block,branch,purpose in ((native,c.BRANCH,'native'),
                (fixture,'research/qwen12-native-transport-preflight-v1','synthetic')):
            self.assertIn("github.event_name == 'push'",block)
            self.assertIn("github.ref == 'refs/heads/"+branch+"'",block)
            self.assertIn('github.event.created == true',block)
            self.assertIn('github.event.repository.private == false',block)
            self.assertIn('github.run_attempt == 1',block)
            self.assertIn('needs: contracts',block)
            self.assertEqual(block.count('contents: write'),1)
            self.assertEqual(block.count('actions: read'),1)
            self.assertEqual(block.count('KWS_EVIDENCE_TOKEN:'),1)
            self.assertIn('if: ${{ always() }}',block)
            self.assertIn('evidence_publisher.py '+purpose,block)
        capture=native.split('- name: Run reviewed',1)[1].split('- name: Publish complete',1)[0]
        self.assertNotIn('TOKEN',capture);self.assertNotIn('env:',capture)
        contracts=text.split('  contracts:\n',1)[1].split('  transport-fixture:\n',1)[0]
        self.assertNotIn('contents: write',contracts);self.assertNotIn('TOKEN',contracts)
        self.assertNotIn('pull_request_target:',text)
        self.assertNotIn('emit_synthetic_log_fixture.py',text)
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
    def test_workflow_no_artifact_action(self):
        text=(HERE/'workflow.yml').read_text()
        self.assertIn('runs-on: ubuntu-24.04',text)
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
    def test_each_prepare_failure_blocks_policy_and_all_decodes(self):
        for failed in c.FAMILIES:
            calls=[]
            def prepare(f,remaining):
                calls.append(f)
                if f==failed:raise ValueError('same-run prepare failure')
            with self.subTest(failed=failed), self.assertRaisesRegex(ValueError,'prepare failure'):
                host.recognition_stages(prepare,lambda:self.fail('policy frozen after failure'),
                    lambda *a:self.fail('decode after prepare failure'))
            self.assertEqual(calls,list(c.FAMILIES[:c.FAMILIES.index(failed)+1]))
    def test_missing_identity_or_package_proof_blocks_all_decodes(self):
        for failure in (FileNotFoundError('missing prepare record'),ValueError('run/head identity mismatch'),
                        ValueError('prepared config drift'),ValueError('actual closure package cap')):
            calls=[]
            def freeze():
                calls.append('freeze');raise failure
            with self.subTest(failure=str(failure)), self.assertRaises(type(failure)):
                host.recognition_stages(lambda f,r:calls.append(f),freeze,
                    lambda *a:self.fail('decode before successful full freeze and package'))
            self.assertEqual(calls,[*c.FAMILIES,'freeze'])
    def test_fresh_decode_failure_prevents_later_families(self):
        for failed in c.FAMILIES:
            calls=[]
            def decode(f,remaining):
                calls.append(f)
                if f==failed:raise ValueError('fresh worker scope/guard/config failure')
            with self.subTest(failed=failed), self.assertRaisesRegex(ValueError,'fresh worker'):
                host.recognition_stages(lambda *a:None,lambda:None,decode)
            self.assertEqual(calls,list(c.FAMILIES[:c.FAMILIES.index(failed)+1]))
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


class FailureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import native_worker
        self.worker=native_worker
        self.context=native_worker._FAILURE_CONTEXT
        self.probe=native_worker._DIAGNOSTIC_PROBE_CODE
        native_worker._DIAGNOSTIC_PROBE_CODE=None
        native_worker._diagnostic_stage('load_model')
    def tearDown(self):
        self.worker._FAILURE_CONTEXT=self.context
        self.worker._DIAGNOSTIC_PROBE_CODE=self.probe
    def diagnostic(self,error):return self.worker._failure_diagnostic(error)
    def encoded(self,value):return c.canonical(value).decode()
    @staticmethod
    def raised_error(filename,function='probe',message='private payload'):
        # Synthetic code metadata and messages only; no model or external call.
        scope={'failure':PermissionError(13,message)}
        exec(compile('def '+function+'():\n    raise failure\n',filename,'exec'),scope)
        try:scope[function]()
        except PermissionError as error:return error
    def test_missing_traceback_errno_and_context_are_safe(self):
        self.worker._FAILURE_CONTEXT=None
        value=self.diagnostic(PermissionError())
        self.assertEqual((value['stage'],value['index'],value['frames']),('unknown',None,[]))
        self.assertNotIn('errno',value)
        self.assertFalse(value['frames_truncated'])
    def test_stage_index_are_fixed_bounded_values(self):
        for context in ((),('https://private.invalid/token',1),('infer',True),('infer',-1),
                        ('infer',13),('infer','/home/private'),('infer',2**1000),(['secret'],[])):
            self.worker._FAILURE_CONTEXT=context
            value=self.diagnostic(ValueError('/home/private/audio.wav'))
            self.assertNotIn('private',self.encoded(value))
            self.assertNotIn('secret',self.encoded(value))
            if type(context) is tuple and len(context)==2 and context[0]=='infer':
                self.assertEqual(value['stage'],'infer');self.assertIsNone(value['index'])
        self.worker._diagnostic_stage('infer',12)
        value=self.diagnostic(ValueError())
        self.assertEqual((value['stage'],value['index']),('infer',12))
    def test_builtin_errno_without_message_or_filename(self):
        error=PermissionError(13,'token=secret transcript','/home/private/audio.wav')
        value=self.diagnostic(error)
        self.assertEqual(value['errno'],13)
        self.assertEqual(self.worker._safe_error_type(error),'PermissionError')
        for text in ('secret','transcript','private','audio.wav'):self.assertNotIn(text,self.encoded(value))
    def test_untrusted_errno_values_are_omitted(self):
        for number in (None,False,True,-1,0,4096,2**1000,'https://private.invalid/token'):
            error=OSError();error.errno=number
            self.assertNotIn('errno',self.diagnostic(error))
    def test_custom_exception_hooks_names_and_chains_are_not_read(self):
        class Hostile(PermissionError):
            def __str__(self):raise RuntimeError('do not stringify')
            def __getattribute__(self,name):
                if name in ('__traceback__','__cause__','__context__','errno','args','filename'):
                    raise RuntimeError('do not inspect payload')
                return super().__getattribute__(name)
        Hostile.__name__='token_SECRET_https://private.invalid/transcript'
        error=Hostile(13,'private payload','/home/private/audio.wav')
        self.assertEqual(self.worker._safe_error_type(error),'OtherException')
        self.assertEqual(self.diagnostic(error)['errno'],13)
        self.assertNotIn('private',self.encoded(self.diagnostic(error)))
    def test_trusted_code_identity_emits_only_relative_metadata(self):
        from unittest.mock import patch
        import linecache
        with patch.object(linecache,'getline',side_effect=AssertionError('source line read')):
            try:self.worker.require(False,'token_SECRET /home/private/audio.wav transcript')
            except ValueError as error:value=self.diagnostic(error)
        frame=value['frames'][-1]
        self.assertEqual(frame['file'],'research/qwen12-native-admission-v1/campaign_contract.py')
        self.assertEqual(frame['function'],'require');self.assertGreater(frame['line'],0)
        for text in ('token_SECRET','/home/private','audio.wav','transcript',str(HERE.parents[1])):
            self.assertNotIn(text,self.encoded(value))
        for key in ('message_disclosed','source_lines_disclosed','locals_disclosed','chain_followed'):
            self.assertIs(value[key],False)
    def test_unknown_and_forged_paths_and_function_names_are_redacted(self):
        filenames=('/home/private/audio.wav','https://private.invalid/token_SECRET',
            str(HERE/'native_worker.py'),str(HERE/'../qwen12-native-admission-v1/native_worker.py'),
            '/runtime/venv/lib/python3.12/site-packages/funasr/utils/../utils/load_utils.py')
        for filename in filenames:
            for function in ('token_SECRET_transcript','main'):
                with self.subTest(filename=filename,function=function):
                    value=self.diagnostic(self.raised_error(filename,function))
                    self.assertTrue(all(row=={'file':'<redacted>','function':'<redacted>','line':None}
                                        for row in value['frames']))
                    for text in ('token_SECRET','transcript','private',str(HERE)):
                        self.assertNotIn(text,self.encoded(value))
    def test_changed_function_cannot_join_registered_source(self):
        from unittest.mock import patch
        import types
        error=self.raised_error(str(HERE/'native_worker.py'),'token_SECRET_transcript')
        trace=error.__traceback__
        while trace.tb_next:trace=trace.tb_next
        function=types.FunctionType(trace.tb_frame.f_code,{'failure':error})
        with patch.object(self.worker,'alarm',function):self.worker._register_diagnostic_codes()
        self.assertNotIn(id(function.__code__),self.worker._DIAGNOSTIC_CODES)
        self.assertEqual(self.diagnostic(error)['frames'][-1]['file'],'<redacted>')
    def test_fixed_locked_runtime_and_subprocess_locations(self):
        import subprocess
        from unittest.mock import patch
        self.worker._register_diagnostic_codes()
        candidates=[getattr(subprocess,name) for name in ('run','call','check_call','check_output')]
        candidates.extend(getattr(subprocess.Popen,name) for name in ('__init__','_execute_child','_posix_spawn'))
        for function in candidates:
            code=function.__code__;known=self.worker._DIAGNOSTIC_CODES.get(id(code))
            self.assertIsNotNone(known);self.assertIs(known[0],code)
            self.assertEqual(known[1:],('stdlib/subprocess.py',code.co_name))
        # Real stdlib code rejects stdout before Popen. Never start a process.
        with patch.object(subprocess,'Popen') as popen:
            try:subprocess.check_output(['unused'],stdout='token_SECRET')
            except ValueError as error:value=self.diagnostic(error)
            else:self.fail('check_output did not reject stdout')
            popen.assert_not_called()
        self.assertEqual(value['frames'][-1]['file'],'stdlib/subprocess.py')
        self.assertEqual(value['frames'][-1]['function'],'check_output')
        self.assertGreater(value['frames'][-1]['line'],0)
        for text in ('token_SECRET',str(Path(subprocess.__file__).parent)):self.assertNotIn(text,self.encoded(value))
        for filename in (subprocess.__file__,'/usr/local/lib/python3.12/subprocess.py',
                         '/home/private/subprocess.py'):
            for name in ('check_output','__init__','_execute_child','_posix_spawn','token_SECRET_transcript'):
                bad=self.diagnostic(self.raised_error(filename,name))
                self.assertEqual(bad['frames'][-1],{'file':'<redacted>','function':'<redacted>','line':None})
    def test_subprocess_snapshot_does_not_register_replacement_function(self):
        import subprocess
        import types
        from unittest.mock import patch
        self.worker._register_diagnostic_codes()
        error=self.raised_error(subprocess.__file__,'check_output')
        trace=error.__traceback__
        while trace.tb_next:trace=trace.tb_next
        function=types.FunctionType(trace.tb_frame.f_code,{'failure':error,'__name__':'subprocess'})
        with patch.object(subprocess,'check_output',function):self.worker._register_diagnostic_codes()
        self.assertNotIn(id(function.__code__),self.worker._DIAGNOSTIC_CODES)
        self.assertEqual(self.diagnostic(error)['frames'][-1]['file'],'<redacted>')
    def test_locked_probe_code_survives_missing_module_and_rejects_forgery(self):
        import types
        filename='/runtime/venv/lib/python3.12/site-packages/funasr/utils/load_utils.py'
        source='def is_ffmpeg_installed():\n    raise failure\n'
        expected=next(code for code in compile(source,filename,'exec').co_consts if type(code) is types.CodeType)
        error=self.raised_error(filename,'is_ffmpeg_installed')
        self.assertNotIn('funasr.utils.load_utils',sys.modules)
        self.assertEqual(self.diagnostic(error)['frames'][-1]['file'],'<redacted>')
        self.worker.register_optional_probe_code(expected)
        value=self.diagnostic(error)
        self.assertEqual(value['frames'][-1],{'file':'runtime/site-packages/funasr/utils/load_utils.py',
            'function':'is_ffmpeg_installed','line':2})
        other=self.raised_error('/home/private/load_utils.py','is_ffmpeg_installed')
        self.assertEqual(self.diagnostic(other)['frames'][-1]['file'],'<redacted>')
        scope={'failure':PermissionError(13,'token_SECRET')}
        exec(compile(source.replace('raise failure','raise failure from None'),filename,'exec'),scope)
        try:scope['is_ffmpeg_installed']()
        except PermissionError as changed:self.assertEqual(self.diagnostic(changed)['frames'][-1]['file'],'<redacted>')
    def test_registered_optional_contextmanager_includes_wrapped_code(self):
        import optional_tools
        self.worker._register_diagnostic_codes()
        code=optional_tools.unavailable_ffmpeg.__wrapped__.__code__
        row=self.worker._DIAGNOSTIC_CODES.get(id(code))
        self.assertIsNotNone(row);self.assertIs(row[0],code)
        self.assertEqual(row[1],'research/qwen12-native-admission-v1/optional_tools.py')
    def test_long_traceback_has_scan_frame_and_byte_bounds(self):
        def recurse(count):
            if count:recurse(count-1)
            else:raise PermissionError(1,'token_SECRET')
        try:recurse(200)
        except PermissionError as error:value=self.diagnostic(error)
        self.assertEqual(len(value['frames']),self.worker.MAX_DIAGNOSTIC_FRAMES)
        self.assertIs(value['frames_truncated'],True)
        self.assertLessEqual(len(c.canonical(value)),self.worker.MAX_DIAGNOSTIC_BYTES)
        self.assertNotIn('token_SECRET',self.encoded(value))
    def test_long_and_cyclic_exception_chains_are_not_traversed(self):
        error=PermissionError(13,'outer private payload');previous=error
        for _ in range(2000):
            following=RuntimeError('inner token_SECRET');previous.__cause__=following;previous=following
        previous.__cause__=error;error.__context__=previous
        value=self.diagnostic(error)
        self.assertFalse(value['chain_followed']);self.assertEqual(value['frames'],[])
        self.assertNotIn('token_SECRET',self.encoded(value))
    def test_diagnostic_failure_preserves_original_terminal_record(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            with patch.object(self.worker,'_failure_diagnostic',side_effect=MemoryError('private payload')):
                self.worker.write_failure(PermissionError(13,'secret'),Path(d))
            value=c.read_json(Path(d)/'worker-failure.json')
        self.assertEqual(value['status'],'failed_no_retry');self.assertEqual(value['error_type'],'PermissionError')
        self.assertEqual(value['diagnostic'],{'schema':'qwen12-bounded-failure-v1','status':'unavailable'})
        self.assertFalse(value['training_admitted']);self.assertNotIn('private',self.encoded(value))
    def test_failure_is_write_once_and_preserves_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'worker-failure.json'
            self.worker.write_failure(PermissionError(13,'secret'),Path(d));original=path.read_bytes()
            self.worker.write_failure(RuntimeError('different private payload'),Path(d))
            self.assertEqual(path.read_bytes(),original)
            value=c.read_json(path)
        self.assertEqual(value['error_type'],'PermissionError')
        self.assertEqual(value['diagnostic']['stage'],'load_model')
        self.assertEqual(value['diagnostic']['index'],0)
        self.assertLessEqual(len(original),self.worker.MAX_DIAGNOSTIC_BYTES)
        self.assertEqual({key:value[key] for key in c.BOUNDARY},c.BOUNDARY)
    def test_timer_and_diagnostic_write_failure_still_exit_one_quietly(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout,redirect_stderr
        output=io.StringIO()
        with patch.object(self.worker,'main',side_effect=PermissionError(13,'token_SECRET')), \
             patch.object(self.worker.signal,'setitimer',side_effect=RuntimeError('private cleanup')), \
             patch.object(self.worker,'write_once',side_effect=OSError('private output')), \
             redirect_stdout(output),redirect_stderr(output):
            result=self.worker.run()
        self.assertEqual(result,1);self.assertEqual(output.getvalue(),'')
    def test_failure_handler_crash_still_exits_one(self):
        from unittest.mock import patch
        with patch.object(self.worker,'main',side_effect=KeyboardInterrupt('secret')), \
             patch.object(self.worker,'write_failure',side_effect=SystemExit(0)), \
             patch.object(self.worker.signal,'setitimer'):
            self.assertEqual(self.worker.run(),1)
    def test_success_does_not_write_failure(self):
        from unittest.mock import patch
        with patch.object(self.worker,'main'),patch.object(self.worker,'write_failure') as failure:
            self.assertEqual(self.worker.run(),0);failure.assert_not_called()
    def test_native_worker_import_does_not_load_models(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers','funasr','whisper')))


if __name__=='__main__':unittest.main()
