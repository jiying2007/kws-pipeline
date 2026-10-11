from pathlib import Path
import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import subprocess
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
    def test_all_three_consumed_predecessors_match_terminal_evidence(self):
        proof=c.PREDECESSOR_ZERO_RECOGNITION
        self.assertEqual(set(proof),{'executions','cross_execution_recognition_limit'})
        self.assertEqual(proof['cross_execution_recognition_limit'],36)
        self.assertEqual([row['run_id'] for row in proof['executions']],
                         [38154869005,38160295526,38167052146])
        for row in proof['executions']:
            terminal=c.read_json(HERE/'evidence'/('run-'+str(row['run_id']))/'terminal-status.json')
            self.assertEqual(row,{
                'run_id':terminal['run']['id'],'run_attempt':terminal['run']['attempt'],
                'executed_head_sha':terminal['source']['commit'],
                'evidence_commit':terminal['evidence']['commit'],
                'archive_sha256':terminal['evidence']['archive_sha256'],
                **{key:terminal['recognition_slots'][key] for key in ('started','completed','not_run')}})
            self.assertEqual((row['started'],row['completed'],row['not_run']),(0,0,36))
        self.assertEqual(sum(row['started'] for row in proof['executions'])+c.LIMITS['attempts'],
                         proof['cross_execution_recognition_limit'])
    def test_recovery_predecessor_roster_is_exact(self):
        proof=c.PREDECESSOR_ZERO_RECOGNITION
        for value in (None, {}, True, proof['executions'][0],
                      dict(proof,executions=proof['executions'][:1]),
                      dict(proof,executions=proof['executions'][1:]),
                      dict(proof,executions=proof['executions'][::-1]),
                      dict(proof,executions=proof['executions']+[proof['executions'][0]]),
                      dict(proof,cross_execution_recognition_limit=72),
                      dict(proof,cross_execution_recognition_limit=True)):
            env,event,release=self.dispatch_fixture();release['predecessor_zero_recognition']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture()
        del release['predecessor_zero_recognition']
        with self.assertRaisesRegex(ValueError,'predecessor'):c.fresh_run(env,event,release)
    def test_each_predecessor_changed_nonzero_or_omitted_is_rejected(self):
        changes={'run_id':38160295527,'run_attempt':2,'executed_head_sha':'a'*40,
                 'evidence_commit':'b'*40,'archive_sha256':'c'*64,
                 'started':1,'completed':1,'not_run':35}
        for index in range(3):
            env,event,release=self.dispatch_fixture()
            del release['predecessor_zero_recognition']['executions'][index]
            with self.subTest(predecessor=index,missing=True),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
            for key,value in changes.items():
                for mode in ('changed','omitted'):
                    env,event,release=self.dispatch_fixture()
                    row=release['predecessor_zero_recognition']['executions'][index]
                    if mode=='omitted':del row[key]
                    else:row[key]=value
                    with self.subTest(predecessor=index,key=key,mode=mode),self.assertRaisesRegex(ValueError,'predecessor'):
                        c.fresh_run(env,event,release)
            for key in ('run_id','run_attempt','started','completed','not_run'):
                original=c.PREDECESSOR_ZERO_RECOGNITION['executions'][index][key]
                for value in (bool(original),float(original),str(original),None):
                    env,event,release=self.dispatch_fixture()
                    release['predecessor_zero_recognition']['executions'][index][key]=value
                    with self.subTest(predecessor=index,key=key,value=value),self.assertRaisesRegex(ValueError,'predecessor'):
                        c.fresh_run(env,event,release)
    def test_only_new_recovery3_source_identity_is_accepted(self):
        import branch_transport
        self.assertEqual(c.BRANCH,'research/qwen12-native-admission-recovery3')
        self.assertEqual(branch_transport.purpose_policy('native')['source_branch'],c.BRANCH)
        for branch in ('research/qwen12-native-admission-v1',
                       'research/qwen12-native-admission-recovery1',
                       'research/qwen12-native-admission-recovery2',
                       'research/qwen12-native-transport-preflight-v1',
                       'research/qwen12-native-transport-preflight-12m-v1',
                       'research/qwen12-native-admission-recovery30',
                       'research/qwen12-native-admission-recovery3-extra'):
            env,event,release=self.dispatch_fixture();env['GITHUB_REF']='refs/heads/'+branch
            with self.subTest(branch=branch),self.assertRaises(ValueError):c.fresh_run(env,event,release)
    def test_recovery3_still_rejects_rerun_and_existing_branch_push(self):
        for target,key,value in (('env','GITHUB_RUN_ATTEMPT','2'),('env','GITHUB_EVENT_NAME','workflow_dispatch'),
                                 ('event','created',False),('event','created',1)):
            env,event,release=self.dispatch_fixture();(env if target=='env' else event)[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):c.fresh_run(env,event,release)

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
        else:
            for key in c.DISPATCH_APPROVALS:
                if key!='new_transcripts_publication_approved':self.assertIs(release[key],False,key)
        push=text.split('  push:\n',1)[1].split('  pull_request:\n',1)[0]
        self.assertEqual(push,'    branches:\n      - '+c.BRANCH+'\n'+
                         '      - research/qwen12-native-transport-preflight-12m-v1\n')
        self.assertEqual(text.count('  qwen12:\n'),1)
        self.assertEqual(text.count('  contracts:\n'),1)
        native=text.split('  qwen12:\n',1)[1].split('  contracts:\n',1)[0]
        for field,expected in (('if',self.native_gate(release['enabled'])),
                ('needs','    needs: contracts'),('runs-on','    runs-on: ubuntu-24.04'),
                ('timeout-minutes','    timeout-minutes: 70')):
            self.assertEqual([line for line in native.splitlines() if line.startswith('    '+field+':')],
                             [expected])
        self.assertEqual(release['required_runtime_preconditions'],list(c.REQUIRED_RUNTIME_PRECONDITIONS))
        self.assertEqual(c.canonical(release['predecessor_zero_recognition']),
                         c.canonical(c.PREDECESSOR_ZERO_RECOGNITION))
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
    def test_disabled_recovery3_fixture_has_no_native_approvals(self):
        release,text=self.source_state_fixture(False)
        for key in c.DISPATCH_APPROVALS:
            with self.subTest(key=key):
                self.assertIs(release[key],key=='new_transcripts_publication_approved')
        self.assertIn(self.native_gate(False),text)
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
    def test_disabled_source_rejects_premature_native_approvals(self):
        for key in c.DISPATCH_APPROVALS:
            if key=='new_transcripts_publication_approved':continue
            release,text=self.source_state_fixture(False);release[key]=True
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.assert_source_activation_state(release,text)
    def test_source_rejects_mixed_activation_gates(self):
        for enabled in (False,True):
            release,_=self.source_state_fixture(enabled);_,text=self.source_state_fixture(not enabled)
            with self.subTest(enabled=enabled),self.assertRaises(AssertionError):
                self.assert_source_activation_state(release,text)
    def test_source_rejects_consumed_or_mixed_workflow_identity(self):
        for enabled in (False,True):
            release,text=self.source_state_fixture(enabled)
            for old in ('research/qwen12-native-admission-v1',
                        'research/qwen12-native-admission-recovery1',
                        'research/qwen12-native-admission-recovery2'):
                for part in ('trigger','guard','both'):
                    changed=text
                    if part in ('trigger','both'):
                        changed=changed.replace('      - '+c.BRANCH+'\n','      - '+old+'\n')
                    if part in ('guard','both'):
                        changed=changed.replace('refs/heads/'+c.BRANCH,'refs/heads/'+old)
                    with self.subTest(enabled=enabled,old=old,part=part),self.assertRaises(AssertionError):
                        self.assert_source_activation_state(release,changed)
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
                (fixture,'research/qwen12-native-transport-preflight-12m-v1','synthetic12m')):
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
    def test_recovery_identity_does_not_reset_consumed_local_gate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(host.MeteredDocker,'raw') as raw:
            docker=host.MeteredDocker(d,999999999999);gate=docker.gate('once')
            with self.assertRaises(ValueError):docker.gate('once')
            with self.assertRaises(FileExistsError):host.MeteredDocker(d,999999999999)
            self.assertTrue(gate.is_dir());raw.assert_not_called()
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
            row=docker.names['last-whisper'];row.update(create_attempted=True,created=True,container_id='a'*64,start_attempted=True,registered=True,reader=lambda:1800000001)
            docker.ledger.register('last-whisper','whisper',lambda:1799999999)
            calls=[]
            def raw(*args,**kwargs):
                calls.append(args)
                return '{"Paused":true}' if args[0]=='inspect' else ''
            docker.raw=raw
            with self.assertRaises(ValueError):docker.remove('last-whisper')
            self.assertTrue(row['removed']);self.assertIn(('rm','--force','a'*64),calls)
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


class CleanupStateTests(unittest.TestCase):
    """No Docker, network, model imports or credentials: fixed subprocess mocks."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.now=[100.0];clock=patch.object(host.time,'monotonic',side_effect=lambda:self.now[0]);clock.start();self.addCleanup(clock.stop)
        no_process=patch.object(host.subprocess,'run',side_effect=AssertionError('no real subprocess allowed'))
        no_process.start();self.addCleanup(no_process.stop)
        self.docker=host.MeteredDocker(self.temporary.name,200);self.docker.check=lambda:None
        self.cid='a'*64;self.calls=[];self.replies={};self.docker.raw=self.raw
    def raw(self,*args,timeout=5):
        self.calls.append((args,timeout));self.assertGreater(timeout,0);self.assertLessEqual(timeout,5)
        value=self.replies.get(args[0])
        if callable(value):return value(args,timeout)
        if isinstance(value,BaseException):raise value
        if value is not None:return value
        return {'create':self.cid+'\n','inspect':'{"Paused":true}','ps':''}.get(args[0],'')
    def owned(self,registered=True):
        self.docker.gate('owned');self.docker('create','--name','owned','image')
        row=self.docker.names['owned']
        if registered:
            row.update(start_attempted=True,registered=True,reader=lambda:12)
            self.docker.ledger.register('owned','setup',lambda:10)
        self.calls.clear();return row
    def uncertain_remove(self,output=''):
        self.replies['rm']=subprocess.TimeoutExpired(['PRIVATE_ARGV'],5,output=b'PRIVATE_STDOUT',stderr=b'PRIVATE_STDERR')
        self.replies['ps']=output
    def test_cleanup_success_closes_once_and_finish_is_read_only(self):
        row=self.owned();self.docker.remove('owned');first=list(self.calls)
        self.docker.remove('owned');self.docker.finish();self.docker.finish()
        self.assertEqual(self.calls,first);self.assertTrue(row['cpu_closed']);self.assertTrue(row['removed'])
        self.assertEqual([a[0] for a,t in self.calls],['pause','inspect','rm'])
        self.assertTrue(all(a[-1]==self.cid for a,t in self.calls))
    def test_rm_timeout_empty_daemon_query_proves_absence(self):
        row=self.owned();self.uncertain_remove();self.docker.remove('owned');self.docker.finish()
        self.assertTrue(row['removed']);self.assertTrue(row['cleanup_ok'])
        self.assertEqual(self.calls[-1][0],('ps','--all','--no-trunc','--filter','id='+self.cid,'--format','{{.ID}}'))
        self.assertEqual(self.docker.cleanup[0]['removal_proof'],'successful_empty_exact_id_query')
        encoded=c.canonical(self.docker.cleanup).decode()
        for private in ('PRIVATE_ARGV','PRIVATE_STDOUT','PRIVATE_STDERR'):self.assertNotIn(private,encoded)
        self.assertEqual(self.docker.cleanup[0]['operations'][-2]['error_type'],'TimeoutExpired')
    def test_nonempty_wrong_id_whitespace_and_failed_query_never_prove_absence(self):
        values=[self.cid+'\n','b'*64+'\n','\n',' ',self.cid[:12]+'\n',
            subprocess.CalledProcessError(1,['PRIVATE_ARGV'],stderr=b'PRIVATE_STDERR'),
            subprocess.TimeoutExpired(['PRIVATE_ARGV'],5)]
        for value in values:
            with self.subTest(value=type(value).__name__),tempfile.TemporaryDirectory() as d:
                self.docker=host.MeteredDocker(d,200);self.docker.check=lambda:None;self.docker.raw=self.raw
                self.replies={};row=self.owned();self.uncertain_remove(value)
                with self.assertRaises(ValueError):self.docker.remove('owned')
                first=list(self.calls)
                for _ in range(2):
                    with self.assertRaises(ValueError):self.docker.finish()
                    with self.assertRaises(ValueError):self.docker.remove('owned')
                self.assertEqual(first,self.calls);self.assertFalse(row['removed']);self.assertTrue(row['cpu_closed'])
                self.assertEqual(sum(a[0]=='rm' for a,t in self.calls),1)
                self.assertEqual(sum(a[0]=='ps' for a,t in self.calls),1)
    def test_failed_create_never_removes_name_or_unowned_id(self):
        for value in ('not-an-id','b'*12,'b'*64+'\nextra','b'*64+' ',
                subprocess.TimeoutExpired(['PRIVATE_ARGV'],5),subprocess.CalledProcessError(1,['PRIVATE_ARGV'])):
            with self.subTest(value=type(value).__name__),tempfile.TemporaryDirectory() as d:
                self.docker=host.MeteredDocker(d,200);self.docker.check=lambda:None;self.docker.raw=self.raw
                self.replies={'create':value};self.calls=[];self.docker.gate('owned')
                with self.assertRaises((ValueError,subprocess.SubprocessError)):self.docker('create','--name','owned','image')
                with self.assertRaises(ValueError):self.docker.finish()
                self.assertEqual([a[0] for a,t in self.calls],['create'])
                self.assertEqual(self.docker.cleanup[0]['ownership'],'unknown')
                self.assertIsNone(self.docker.names['owned']['container_id'])
    def test_duplicate_returned_id_cannot_grant_second_ownership(self):
        self.owned(False);self.docker.gate('other')
        with self.assertRaises(ValueError):self.docker('create','--name','other','image')
        with self.assertRaises(ValueError):self.docker.finish()
        self.assertFalse(self.docker.names['other']['created'])
        self.assertEqual(sum(a[0]=='rm' for a,t in self.calls),1)
    def test_reused_name_is_never_sent_after_create_and_wrong_inspect_id_stops(self):
        self.owned(False);self.replies['inspect']=json.dumps([{'Id':'b'*64}])
        with self.assertRaises(ValueError):self.docker('inspect','owned')
        self.docker('kill','--signal=USR1','owned');self.docker.remove('owned')
        self.assertEqual([a[-1] for a,t in self.calls],[self.cid]*3)
    def test_start_wrong_id_cannot_open_gate_or_claim_cpu_accounting(self):
        row=self.owned(False);self.replies['inspect']=json.dumps([{'Id':'b'*64}])
        with self.assertRaises(ValueError):self.docker('start','owned')
        with self.assertRaises(ValueError):self.docker.remove('owned')
        self.assertTrue(row['removed']);self.assertFalse(row['registered']);self.assertFalse((row['gate']/'go').exists())
        self.assertEqual(self.docker.cleanup[0]['accounting_status'],'final_usage_unverified')
        self.assertTrue(all(a[-1]==self.cid for a,t in self.calls))
    def test_start_and_whisper_call_shape_use_owned_id(self):
        row=self.owned(False)
        self.replies['inspect']=json.dumps([{'Id':self.cid,'State':{'Pid':123,'Running':True}}])
        def read(path,*a,**k):return '0::/mock-scope\n' if str(path).startswith('/proc/') else 'usage_usec 10\n'
        with patch.object(Path,'read_text',read),patch.object(Path,'stat',return_value=SimpleNamespace(st_ino=42)), \
                patch.object(Path,'is_symlink',return_value=False):
            self.docker('start','owned',timeout=30)
        self.assertTrue(row['registered']);self.assertTrue((row['gate']/'go').exists())
        self.replies['inspect']='{"Paused":true}';row['reader']=lambda:12
        self.docker('rm','--force','owned',timeout=.75)
        self.assertTrue(all(t<=.75 for a,t in self.calls if a[0] in ('pause','rm')))
        self.assertTrue(all(a[-1]==self.cid for a,t in self.calls))
    def test_pause_inspect_or_cpu_failure_still_removes_but_accounting_stays_invalid(self):
        for failing in ('pause','inspect','cpu'):
            with self.subTest(operation=failing),tempfile.TemporaryDirectory() as d:
                self.docker=host.MeteredDocker(d,200);self.docker.check=lambda:None;self.docker.raw=self.raw
                self.replies={};row=self.owned()
                if failing=='cpu':row['reader']=lambda:(_ for _ in ()).throw(ValueError('PRIVATE_CPU'))
                else:self.replies[failing]=subprocess.CalledProcessError(7,['PRIVATE_ARGV'],stderr=b'PRIVATE_STDERR')
                with self.assertRaises(ValueError):self.docker.remove('owned')
                self.assertTrue(row['removed']);self.assertFalse(row['cleanup_ok']);self.assertFalse(row['cpu_closed'])
                self.assertEqual(self.calls[-1][0],('rm','--force',self.cid))
                with self.assertRaises(ValueError):self.docker.finish()
    def test_final_cpu_invalid_is_not_repaired_by_proved_absence(self):
        row=self.owned();row['reader']=lambda:0;self.uncertain_remove()
        with self.assertRaises(ValueError):self.docker.remove('owned')
        self.assertTrue(row['removed']);self.assertFalse(row['cleanup_ok']);self.assertFalse(row['cpu_closed'])
    def test_cleanup_shared_fifteen_second_budget_has_no_retry(self):
        self.owned()
        def slow(args,timeout):
            self.now[0]+=timeout
            if args[0]=='inspect':return '{"Paused":true}'
            if args[0]=='rm':raise subprocess.TimeoutExpired(['PRIVATE_ARGV'],timeout)
            return ''
        self.replies={key:slow for key in ('pause','inspect','rm','ps')}
        with self.assertRaises(ValueError):self.docker.remove('owned')
        self.assertEqual(self.now[0],115);self.assertEqual([a[0] for a,t in self.calls],['pause','inspect','rm'])
        self.assertFalse(self.docker.names['owned']['removed'])
    def test_absence_query_uses_remaining_budget_not_new_five_seconds(self):
        self.owned()
        def slow(args,timeout):
            self.now[0]+=4 if args[0] in ('pause','inspect') else timeout
            if args[0]=='inspect':return '{"Paused":true}'
            if args[0]=='rm':raise subprocess.TimeoutExpired(['PRIVATE_ARGV'],timeout)
            return ''
        self.replies={key:slow for key in ('pause','inspect','rm','ps')};self.docker.remove('owned')
        self.assertEqual(self.calls[-1][1],2);self.assertEqual(self.now[0],115)
    def test_main_deadline_expiry_still_attempts_owned_cleanup_and_fails(self):
        for expire_at in ('before','pause','inspect'):
            with self.subTest(expire_at=expire_at),tempfile.TemporaryDirectory() as d:
                self.docker=host.MeteredDocker(d,self.now[0]+1);self.docker.check=lambda:None;self.docker.raw=self.raw
                self.replies={};self.owned()
                if expire_at=='before':self.now[0]+=2
                else:
                    def expire(args,timeout):
                        self.now[0]+=2;return '{"Paused":true}' if args[0]=='inspect' else ''
                    self.replies[expire_at]=expire
                with self.assertRaises(ValueError):self.docker.remove('owned')
                self.assertEqual(self.calls[-1][0],('rm','--force',self.cid))
                self.assertTrue(self.docker.cleanup[0]['host_deadline_overrun'])
    def test_cleanup_receipt_write_failure_stays_failed_without_repeating_removal(self):
        for filename in ('cpu-receipt.json','cleanup-receipt.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as d:
                self.docker=host.MeteredDocker(d,200);self.docker.check=lambda:None;self.docker.raw=self.raw
                self.replies={};row=self.owned();write=host.atomic
                def fail_receipt(path,value):
                    if path.name==filename:raise OSError('PRIVATE_RECEIPT_PATH')
                    return write(path,value)
                with patch.object(host,'atomic',side_effect=fail_receipt),self.assertRaises(ValueError):
                    self.docker.remove('owned')
                self.assertTrue(row['removed']);self.assertFalse(row['cleanup_ok']);first=list(self.calls)
                with self.assertRaises(ValueError):self.docker.finish()
                self.assertEqual(self.calls,first)
                self.assertEqual(self.docker.cleanup[0]['receipt_errors'][0]['error_type'],'OSError')
                self.assertNotIn('PRIVATE_',c.canonical(self.docker.cleanup).decode())
    def test_accounting_operations_reserve_time_for_one_stop(self):
        row=self.owned()
        def slow(args,timeout):
            self.now[0]+=timeout
            return '{"Paused":true}' if args[0]=='inspect' else ''
        self.replies={key:slow for key in ('pause','inspect','rm')}
        with self.assertRaises(ValueError):self.docker.remove('owned')
        self.assertEqual([a[0] for a,t in self.calls],['pause','inspect','rm'])
        self.assertTrue(row['removed']);self.assertFalse(row['cpu_closed']);self.assertEqual(self.now[0],115)

    def test_gate_result_is_strict_small_safe_projection(self):
        row=self.owned();path=row['gate']/'result.json'
        for value in ({'schema':'qwen12-gated-command-v1','exit_code':True},
                {'schema':'qwen12-gated-command-v1','exit_code':256},
                {'schema':'qwen12-gated-command-v1','exit_code':0,'private':'PRIVATE_GATE'}):
            c.atomic(path,value);self.assertEqual(self.docker.gate_result('owned')['status'],'invalid')
            with self.assertRaises(ValueError):self.docker.completed('owned')
        path.write_text('{"private":"PRIVATE_GATE"');self.assertEqual(self.docker.gate_result('owned')['status'],'invalid')
        c.atomic(path,{'schema':'qwen12-gated-command-v1','exit_code':7})
        with self.assertRaises(ValueError):self.docker.completed('owned')
        self.docker.remove('owned')
        self.assertEqual(self.docker.cleanup[0]['gate_result']['exit_code'],7)
        self.assertNotIn('PRIVATE_GATE',c.canonical(self.docker.cleanup).decode())
    def test_execute_failed_setup_charges_elapsed_for_asr_and_whisper(self):
        # Exercise the actual execute accumulator with both setup callbacks failing.
        for fail_stage in ('asr','whisper'):
            with self.subTest(stage=fail_stage),tempfile.TemporaryDirectory() as d:
                self.now[0]=100
                original=SimpleNamespace(host_capacity=lambda:{},pull_image=lambda *a:None)
                def setup(*args,**kwargs):
                    self.now[0]+=7
                    if fail_stage=='asr':raise RuntimeError('PRIVATE_SETUP')
                    c.atomic(args[3]/'setup-receipt.json',{'actual_download_bytes':0,'commands':[]})
                def whisper(*args,**kwargs):self.now[0]+=11;raise RuntimeError('PRIVATE_WHISPER')
                run={'run_id':'1','head_sha':'a'*40}
                with patch.object(sys,'path',list(sys.path)),patch.object(host,'fresh_run',return_value=run),patch.object(host,'load',return_value=original), \
                        patch.object(host.signal,'signal'),patch.object(host.signal,'setitimer'), \
                        patch.object(host.shutil,'disk_usage',return_value=SimpleNamespace(free=20*c.GIB)), \
                        patch.object(host,'fetch_blind',return_value={}),patch.object(host,'validate_blind',return_value={}), \
                        patch.object(host,'stage'),patch.object(host.shutil,'copyfile'), \
                        patch.object(host,'supervise',side_effect=setup),patch.object(host,'whisper_setup',side_effect=whisper), \
                        patch.object(host,'denominator',return_value=[]):
                    status=host.execute({'RUNNER_TEMP':d},{},{},{'inference_code':{},'setup_code':{}})
                receipt=c.read_json(Path(d)/(host.EXPERIMENT+'-1')/'host-receipt.json')
                self.assertEqual(status,1)
                self.assertEqual(receipt['combined_setup_seconds'],7 if fail_stage=='asr' else 18)
                self.assertEqual(receipt['stages']['asr_setup' if fail_stage=='asr' else 'whisper_setup'],'error')
                self.assertEqual(receipt['primary_error'],{'error_type':'RuntimeError'})
                self.assertNotIn('PRIVATE_',c.canonical(receipt).decode())

    def test_supervision_primary_cleanup_combinations_keep_original_error(self):
        image=c.read_json(host.ROOT/'research/source-screen32-v1/container-lock.json')['image']
        for primary_fails in (False,True):
            for cleanup_fails,retention_fails in ((False,False),(False,True),(True,False),(True,True)):
                with self.subTest(primary=primary_fails,cleanup=cleanup_fails,retention=retention_fails),tempfile.TemporaryDirectory() as d:
                    self.docker=host.MeteredDocker(d,200);self.docker.check=lambda:None;self.docker.raw=self.raw
                    self.replies={};self.calls=[];primary=RuntimeError('PRIVATE_PRIMARY')
                    original=SimpleNamespace(container_args=lambda name,image,*a:['create','--name',name,image,'old'],
                        validate_container_inspect=lambda *a:None)
                    captured={}
                    def create(args,timeout):captured['args']=args;return self.cid+'\n'
                    def inspect(args,timeout):
                        name=next(iter(self.docker.names));row=self.docker.names[name]
                        if '--format' in args:return '{"Paused":true}'
                        return json.dumps([{'Id':self.cid,'Mounts':[{'Destination':'/gate','Source':str(row['gate']),'RW':True}],
                            'Config':{'Cmd':list(captured['args'][captured['args'].index(image)+1:]),'Image':image,'Entrypoint':None}}])
                    self.replies.update(create=create,inspect=inspect)
                    real_call=host.MeteredDocker.__call__
                    def call(docker,*args,**kwargs):
                        if args[0]!='start':return real_call(docker,*args,**kwargs)
                        self.now[0]+=3
                        if primary_fails:raise primary
                        row=docker.names[args[-1]];row.update(start_attempted=True,registered=True,reader=lambda:12)
                        docker.ledger.register(args[-1],'setup',lambda:10)
                        c.atomic(row['gate']/'result.json',{'schema':'qwen12-gated-command-v1','exit_code':0})
                        return ''
                    if cleanup_fails:self.uncertain_remove(self.cid+'\n')
                    write=host.atomic
                    def retain(path,value):
                        if retention_fails and path.name=='supervision-receipt.json':raise OSError('PRIVATE_RETENTION')
                        return write(path,value)
                    with patch.object(host,'atomic',side_effect=retain),patch.object(host.MeteredDocker,'__call__',call),patch.dict(os.environ,{'GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'1'}):
                        if primary_fails or cleanup_fails or retention_fails:
                            expected=RuntimeError if primary_fails else ValueError if cleanup_fails else OSError
                            with self.assertRaises(expected) as caught:
                                host.supervise(original,self.docker,Path(d),Path(d),Path(d),Path(d),'setup',200)
                            if primary_fails:self.assertIs(caught.exception,primary)
                        else:host.supervise(original,self.docker,Path(d),Path(d),Path(d),Path(d),'setup',200)
                    record=(self.docker.supervisions[0] if retention_fails else
                        c.read_json(Path(d)/'supervision-receipt.json')['stages'][0])
                    self.assertEqual(record['primary_error'] is not None,primary_fails)
                    self.assertEqual(record['cleanup_error'] is not None,cleanup_fails)
                    self.assertGreaterEqual(record['wall_seconds'],3);self.assertEqual(record['phase_wall_seconds']['start'],3)
                    self.assertEqual('supervision-receipt.json' in host.collect(Path(d)),not retention_fails)
                    self.assertEqual('retention_error' in record,retention_fails)
                    self.assertNotIn('PRIVATE_',c.canonical(record).decode())


class PackageFailureDiagnosticTests(unittest.TestCase):
    identity={'repository':'jiying2007/kws-pipeline','run_id':1,'run_attempt':1,
        'job_key':'qwen12','executed_head_sha':'a'*40}
    secret='PRIVATE_TOKEN /home/private/audio.wav transcript env=secret argv=secret'

    def limit_error(self):
        from log_transport import PackageLimitError
        error=PackageLimitError('compressed_cap',compressed_bytes=12*c.MIB+1,
            expanded_bytes=24*c.MIB,member_count=3100)
        error.args=(self.secret,)
        return error

    def test_completed_zip_rejection_retains_actual_counts_and_original_error(self):
        import log_transport,zipfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c.write_once(root/'host-receipt.json',{'schema':'synthetic-only'})
            payload,_,_=log_transport.build_package(root,['host-receipt.json'],self.identity)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                expanded=sum(row.file_size for row in archive.infolist())
            observed=[]
            def refused(*args):
                try:return log_transport.build_package(root,['host-receipt.json'],self.identity)
                except log_transport.PackageLimitError as error:
                    observed.append(error);raise
            with patch.object(log_transport,'MAX_COMPRESSED',len(payload)-1), \
                 patch.object(host,'_package_budget_preflight',side_effect=refused), \
                 self.assertRaises(log_transport.PackageLimitError) as caught:
                host.package_budget_preflight(root,self.identity)
            self.assertIs(caught.exception,observed[0])
            result=c.read_json(root/'package-budget-preflight.json')
            self.assertEqual(result['status'],'rejected_before_any_audio_forward')
            self.assertEqual(result['new_native_calls'],0)
            diagnostic=result['diagnostic']
            self.assertEqual(diagnostic['reason_code'],'compressed_cap')
            self.assertEqual(diagnostic['stage'],'package_budget_preflight')
            self.assertEqual((diagnostic['compressed_bytes'],diagnostic['expanded_bytes'],diagnostic['member_count']),
                             (len(payload),expanded,1))
            self.assertIn('package-budget-preflight.json',host.collect(root))
            self.assertLess(len(c.canonical(result)),1024)

    def test_only_exact_known_exception_type_can_supply_sizing_fields(self):
        from log_transport import PackageLimitError
        class Hostile(PackageLimitError):
            def __str__(self):raise AssertionError('exception must not be stringified')
            def __repr__(self):raise AssertionError('exception must not be represented')
            def __getattribute__(self,name):
                if name in ('reason_code','compressed_bytes','expanded_bytes','member_count','args','__dict__','__traceback__'):
                    raise AssertionError('untrusted attributes must not be read')
                return super().__getattribute__(name)
        for error in (ValueError(self.secret),Hostile('compressed_cap',compressed_bytes=123)):
            error.reason_code='compressed_cap';error.compressed_bytes=123
            value=host.host_failure_diagnostic(error,'package_budget_preflight')
            self.assertEqual(value['reason_code'],'unknown_failure')
            self.assertTrue(all(value[key] is None for key in ('compressed_bytes','expanded_bytes','member_count')))
            self.assertNotIn('PRIVATE_',c.canonical(value).decode())

    def test_untrusted_reason_counts_and_stage_are_bounded_or_unknown(self):
        for number in (None,True,False,-1,2**63,2**1000,1.5,self.secret,[],{}):
            error=self.limit_error()
            for key in ('compressed_bytes','expanded_bytes','member_count'):setattr(error,key,number)
            value=host.host_failure_diagnostic(error,self.secret)
            self.assertEqual(value['stage'],'unknown')
            self.assertTrue(all(value[key] is None for key in ('compressed_bytes','expanded_bytes','member_count')))
            self.assertLess(len(c.canonical(value)),1024)
            self.assertNotIn('PRIVATE_',c.canonical(value).decode())
        error=self.limit_error();error.reason_code=self.secret
        value=host.host_failure_diagnostic(error,'package_budget_preflight')
        self.assertEqual(value['reason_code'],'unknown_failure')
        self.assertTrue(all(value[key] is None for key in ('compressed_bytes','expanded_bytes','member_count')))
        self.assertNotIn('PRIVATE_',c.canonical(value).decode())

    def test_early_limits_leave_unmeasured_aggregate_fields_unknown(self):
        from log_transport import PackageLimitError
        for reason in ('expanded_cap','member_cap','member_count_cap'):
            value=host.host_failure_diagnostic(PackageLimitError(reason,member_count=4097),'package_budget_preflight')
            self.assertEqual(value['reason_code'],reason);self.assertEqual(value['member_count'],4097)
            self.assertIsNone(value['compressed_bytes']);self.assertIsNone(value['expanded_bytes'])

    def test_unknown_preflight_failure_remains_failure_without_exception_payload(self):
        error=ValueError(self.secret)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(host,'_package_budget_preflight',side_effect=error),self.assertRaises(ValueError) as caught:
                host.package_budget_preflight(root,self.identity)
            self.assertIs(caught.exception,error)
            value=c.read_json(root/'package-budget-preflight.json')
            self.assertEqual(value['status'],'rejected_before_any_audio_forward')
            self.assertEqual(value['diagnostic']['reason_code'],'unknown_failure')
            self.assertNotIn('PRIVATE_',c.canonical(value).decode())

    def test_preflight_receipt_or_diagnostic_failure_does_not_replace_original(self):
        error=self.limit_error()
        for target in ('write_once','host_failure_diagnostic'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as d:
                with patch.object(host,'_package_budget_preflight',side_effect=error), \
                     patch.object(host,target,side_effect=OSError(self.secret)), \
                     self.assertRaises(type(error)) as caught:
                    host.package_budget_preflight(Path(d),self.identity)
                self.assertIs(caught.exception,error)

    def test_host_retains_safe_diagnostic_and_complete_failure_package(self):
        import branch_transport,worker_receipts,zipfile
        from contextlib import ExitStack,redirect_stdout,redirect_stderr
        # No subprocess/model/network: only the host orchestration and local codec run.
        for typed,receipt_fails,diagnostic_fails in ((True,False,False),(True,True,False),
                (False,False,False),(True,False,True)):
            with self.subTest(typed=typed,receipt_fails=receipt_fails,diagnostic_fails=diagnostic_fails),tempfile.TemporaryDirectory() as d:
                error=self.limit_error() if typed else ValueError(self.secret)
                original=SimpleNamespace(host_capacity=lambda:{},pull_image=lambda *a:None)
                calls=[]
                def setup(*args,**kwargs):
                    family=args[6];calls.append((family,kwargs.get('mode')))
                    if family=='setup':
                        c.atomic(args[3]/'setup-receipt.json',{'actual_download_bytes':0,'commands':[]})
                    else:
                        self.assertEqual(kwargs.get('mode'),'prepare')
                        c.write_once(args[5]/'native-config.json',{})
                        c.write_once(args[5]/'prepared.json',{'audio_forward_calls':0})
                        c.write_once(args[5]/'worker-complete.json',{'calls':0})
                    return 0
                write=host.write_once;diagnostic=host.host_failure_diagnostic
                def retain(path,value):
                    if receipt_fails and path.name=='package-budget-preflight.json':raise OSError(self.secret)
                    return write(path,value)
                def project(*args):
                    if diagnostic_fails:raise MemoryError(self.secret)
                    return diagnostic(*args)
                output=io.StringIO();run={'run_id':'1','head_sha':'a'*40}
                job=c.blind_job(c.read_json(HERE/'public-inputs.json')['records'])
                with ExitStack() as stack:
                    for context in (patch.object(sys,'path',list(sys.path)),patch.object(host,'fresh_run',return_value=run),
                        patch.object(host,'load',return_value=original),patch.object(host.signal,'signal'),
                        patch.object(host.signal,'setitimer'),patch.object(host.shutil,'disk_usage',return_value=SimpleNamespace(free=20*c.GIB)),
                        patch.object(host,'fetch_blind',return_value={}),patch.object(host,'validate_blind',return_value=job),
                        patch.object(host,'stage'),patch.object(host.shutil,'copyfile'),patch.object(host,'supervise',side_effect=setup),
                        patch.object(host,'whisper_setup',return_value={}),patch.object(worker_receipts,'build_policy',return_value={}),
                        patch.object(host,'_package_budget_preflight',side_effect=error),patch.object(host,'write_once',side_effect=retain),
                        patch.object(host,'host_failure_diagnostic',side_effect=project),redirect_stdout(output),redirect_stderr(output)):
                        stack.enter_context(context)
                    package=stack.enter_context(patch.object(branch_transport,'write_package'))
                    status=host.execute({'RUNNER_TEMP':d,'GITHUB_JOB':'qwen12'},{},
                        {'new_transcripts_publication_approved':True},{'inference_code':{},'setup_code':{}})
                self.assertEqual(status,1)
                self.assertEqual(calls,[('setup',None)]+[(family,'prepare') for family in c.FAMILIES])
                package.assert_called_once();payload=package.call_args.args[1]
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    receipt=json.loads(archive.read('host-receipt.json'))
                    outcomes=json.loads(archive.read('terminal-outcomes.json'))
                    self.assertEqual('package-budget-preflight.json' in archive.namelist(),not (receipt_fails or diagnostic_fails))
                    if not (receipt_fails or diagnostic_fails):
                        self.assertEqual(json.loads(archive.read('package-budget-preflight.json'))['diagnostic'],receipt['diagnostic'])
                    self.assertTrue(all(self.secret.encode() not in archive.read(name) for name in archive.namelist()))
                self.assertEqual(receipt['status'],'stopped_no_retry')
                self.assertEqual(receipt['primary_error'],{'error_type':'ValueError'})
                self.assertEqual(receipt['diagnostic']['stage'],'package_budget_preflight')
                self.assertEqual(receipt['diagnostic']['reason_code'],
                    'diagnostic_unavailable' if diagnostic_fails else 'compressed_cap' if typed else 'unknown_failure')
                self.assertEqual(len(outcomes),36);self.assertTrue(all(row['status']=='not_run' for row in outcomes))
                self.assertTrue(all(receipt['stages'][family]=='not_run' for family in c.FAMILIES))
                self.assertNotIn('PRIVATE_',output.getvalue())


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
