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
        release['remaining_slot_contract']=copy.deepcopy(c.remaining_slot_contract())
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
        release['remaining_slot_contract']=copy.deepcopy(c.remaining_slot_contract())
        release['required_runtime_preconditions']=list(c.REQUIRED_RUNTIME_PRECONDITIONS)
        release['evidence_transport']='github-repository-zip-chunks-v1'
        return env,event,release
    def test_all_five_consumed_predecessors_match_terminal_evidence(self):
        proof=c.remaining_slot_contract()
        self.assertEqual([row['run_id'] for row in proof['executions']],
                         [38154869005,38160295526,38167052146,38173984595,38180060677])
        for index,row in enumerate(proof['executions']):
            if index<3:
                terminal=c.read_json(HERE/'evidence'/('run-'+str(row['run_id']))/'terminal-status.json')
                self.assertEqual(row['executed_head_sha'],terminal['source']['commit'])
                self.assertEqual(row['archive_sha256'],terminal['evidence']['archive_sha256'])
            self.assertEqual((row['started'],row['completed'],row['error'],row['not_run']),
                             [(0,0,0,36),(0,0,0,36),(0,0,0,36),(1,0,1,35),(0,0,1,35)][index])
            self.assertEqual(len(row['started_members']),int(index==3))
        self.assertEqual(sum(row['started'] for row in proof['executions'])+c.LIMITS['new_starts'],36)
        self.assertEqual(proof['historical_consumed'][0],c.historical_slot())
        self.assertEqual(proof['executions'][3]['started_members'],[c.historical_slot()['started_member']])
    def test_original_four_and_historical_canonical_bytes_unchanged(self):
        proof=c.remaining_slot_contract()
        self.assertEqual(c.digest(c.canonical(proof['executions'][:4])),
                         '394f7d58b020500f14ba516fdfe39eda835b2393592441862430b995e89da86f')
        self.assertEqual(c.digest(c.canonical(c.historical_slot())),
                         'efd9d4d295de76ac3726f3598ef068ab8f3d7bb1635963ae51d4e2383af27cbb')
        self.assertEqual(c.canonical(c.historical_slot()),c._HISTORICAL_BYTES)

    def test_fifth_zero_actual_start_is_exact_inherited_error_reference(self):
        proof=c.remaining_slot_contract();fifth=proof['executions'][4]
        self.assertEqual(proof['schema'],'qwen12-fixed-remaining35-r1-v1')
        self.assertEqual(c.digest(c.canonical(fifth)),
                         '6d7d8203a86bfccc62d73ee05cca4072effbea8430ea8a8d93c4dcab5517810e')
        self.assertEqual([row['started'] for row in proof['executions']],[0,0,0,1,0])
        self.assertEqual([row['completed'] for row in proof['executions']],[0,0,0,0,0])
        self.assertEqual(fifth['started_members'],[])
        self.assertEqual((fifth['current_started'],fifth['current_completed'],fifth['cumulative_started']),
                         (0,0,1))
        self.assertEqual((fifth['terminal_reported_attempt_sum'],fifth['error'],fifth['not_run']),(1,1,35))
        self.assertEqual(fifth['terminal_outcomes'],{'path':'terminal-outcomes.json','bytes':21143,
                         'sha256':'588e0f4786d7d716770583d28edabca773031f953a65f394c224bfc028cadbfa'})
        self.assertEqual(fifth['terminal_historical_references'],[{
            'family':'qwen06','audio_id':'clip-000001','origin':'historical','status':'error','attempts':1,
            'historical_evidence':c.historical_slot()}])
        self.assertEqual(fifth['terminal_historical_references'][0]['historical_evidence']['started_member'],
                         proof['executions'][3]['started_members'][0])
        self.assertEqual(sum(row['started'] for row in proof['executions']),1)
        self.assertEqual(len(proof['historical_consumed']),1)
        self.assertEqual(proof['historical_starts'],1)
        self.assertEqual(proof['campaign_slots'],36)
        self.assertEqual(proof['maximum_new_starts'],35)
        self.assertEqual(proof['maximum_cumulative_starts'],36)

    def test_old_four_only_schema_and_release_are_rejected(self):
        for old_schema in ('qwen12-fixed-remaining35-v1','qwen12-fixed-remaining35-r1-v1'):
            for length in (4,5):
                if old_schema=='qwen12-fixed-remaining35-r1-v1' and length==5:continue
                env,event,release=self.dispatch_fixture()
                release['remaining_slot_contract']['schema']=old_schema
                release['remaining_slot_contract']['executions']=release['remaining_slot_contract']['executions'][:length]
                with self.subTest(schema=old_schema,length=length),self.assertRaisesRegex(ValueError,'predecessor'):
                    c.fresh_run(env,event,release)

    def test_fifth_historical_reference_never_becomes_a_second_start(self):
        changes=[{'started':1},{'completed':1},{'started_members':[c.historical_slot()['started_member']]},
                 {'current_started':1},{'current_completed':1},{'cumulative_started':2},
                 {'terminal_reported_attempt_sum':0},{'error':0},{'not_run':36},
                 {'terminal_historical_references':[]},
                 {'terminal_historical_references':[c.historical_slot()]}]
        for change in changes:
            env,event,release=self.dispatch_fixture()
            release['remaining_slot_contract']['executions'][4].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture()
        fifth=release['remaining_slot_contract']['executions'][4]
        fifth['started']=fifth['current_started']=1;fifth['cumulative_started']=2
        fifth['started_members']=[copy.deepcopy(c.historical_slot()['started_member'])]
        release['remaining_slot_contract']['historical_starts']=2
        release['remaining_slot_contract']['maximum_new_starts']=34
        release['remaining_slot_contract']['historical_consumed'].append(c.historical_slot())
        with self.assertRaisesRegex(ValueError,'predecessor'):c.fresh_run(env,event,release)

    def test_fifth_historical_role_original_member_and_terminal_census_are_fixed(self):
        for key,value in (('origin','current_run'),('status','not_run'),('attempts',0),
                          ('family','sensevoice'),('audio_id','clip-000002')):
            env,event,release=self.dispatch_fixture()
            release['remaining_slot_contract']['executions'][4]['terminal_historical_references'][0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
        for field in ('terminal_historical_references','started_members'):
            env,event,release=self.dispatch_fixture()
            fifth=release['remaining_slot_contract']['executions'][4]
            fifth[field].append(copy.deepcopy(fifth['terminal_historical_references'][0]))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)

    def test_every_consumed_run_id_is_rejected_with_new_ref(self):
        for row in c.remaining_slot_contract()['executions']:
            env,event,release=self.dispatch_fixture();env['GITHUB_RUN_ID']=str(row['run_id'])
            with self.subTest(run_id=row['run_id']),self.assertRaisesRegex(ValueError,'historical execution'):
                c.fresh_run(env,event,release)

    def test_recovery_predecessor_roster_is_exact(self):
        proof=c.remaining_slot_contract()
        for value in (None, {}, True, proof['executions'][0],
                      dict(proof,executions=proof['executions'][:1]),
                      dict(proof,executions=proof['executions'][1:]),
                      dict(proof,executions=proof['executions'][::-1]),
                      dict(proof,executions=proof['executions']+[proof['executions'][0]]),
                      dict(proof,maximum_cumulative_starts=72),
                      dict(proof,maximum_cumulative_starts=True),
                      dict(proof,executions=tuple(proof['executions'])),
                      dict(proof,planned_new_slots=tuple(proof['planned_new_slots']))):
            env,event,release=self.dispatch_fixture();release['remaining_slot_contract']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
        env,event,release=self.dispatch_fixture()
        del release['remaining_slot_contract']
        with self.assertRaisesRegex(ValueError,'predecessor'):c.fresh_run(env,event,release)
    def test_each_predecessor_changed_nonzero_or_omitted_is_rejected(self):
        changes={'run_id':38160295527,'run_attempt':2,'executed_head_sha':'a'*40,
                 'evidence_commit':'b'*40,'archive_sha256':'c'*64,
                 'started':1,'completed':1,'not_run':35}
        for index in range(5):
            env,event,release=self.dispatch_fixture()
            del release['remaining_slot_contract']['executions'][index]
            with self.subTest(predecessor=index,missing=True),self.assertRaisesRegex(ValueError,'predecessor'):
                c.fresh_run(env,event,release)
            for key,value in changes.items():
                for mode in ('changed','omitted'):
                    env,event,release=self.dispatch_fixture()
                    row=release['remaining_slot_contract']['executions'][index]
                    if mode=='omitted':del row[key]
                    else:row[key]=value if row[key]!=value else str(value)
                    with self.subTest(predecessor=index,key=key,mode=mode),self.assertRaisesRegex(ValueError,'predecessor'):
                        c.fresh_run(env,event,release)
            for key in ('run_id','run_attempt','started','completed','not_run'):
                original=c.remaining_slot_contract()['executions'][index][key]
                for value in (bool(original),float(original),str(original),None):
                    env,event,release=self.dispatch_fixture()
                    release['remaining_slot_contract']['executions'][index][key]=value
                    with self.subTest(predecessor=index,key=key,value=value),self.assertRaisesRegex(ValueError,'predecessor'):
                        c.fresh_run(env,event,release)
    def test_only_new_remaining35_r1_source_identity_is_accepted(self):
        import branch_transport
        self.assertEqual(c.BRANCH,'research/qwen12-native-admission-remaining35-r1-v1')
        self.assertEqual(branch_transport.purpose_policy('native')['source_branch'],c.BRANCH)
        for branch in ('research/qwen12-native-admission-v1',
                       'research/qwen12-native-admission-recovery1',
                       'research/qwen12-native-admission-recovery2',
                       'research/qwen12-native-transport-preflight-v1',
                       'research/qwen12-native-transport-preflight-12m-v1',
                       'research/qwen12-native-admission-recovery3',
                       'research/qwen12-native-admission-recovery30',
                       'research/qwen12-native-admission-recovery3-extra',
                       'research/qwen12-native-admission-remaining35-v1',
                       'research/qwen12-native-admission-remaining35-r1',
                       'research/qwen12-native-admission-remaining35-r1-v1-extra'):
            env,event,release=self.dispatch_fixture();env['GITHUB_REF']='refs/heads/'+branch
            with self.subTest(branch=branch),self.assertRaises(ValueError):c.fresh_run(env,event,release)
    def test_remaining35_r1_still_rejects_rerun_and_existing_branch_push(self):
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
        self.assertEqual(c.canonical(release['remaining_slot_contract']),
                         c.canonical(c.remaining_slot_contract()))
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
    def test_disabled_remaining35_r1_fixture_has_no_native_approvals(self):
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
                        'research/qwen12-native-admission-recovery2',
                        'research/qwen12-native-admission-recovery3',
                        'research/qwen12-native-admission-remaining35-v1'):
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
    def test_setup_failure_preserves_historical_error_and35_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            rows=c.denominator(c.blind_job(self.public['records']),d)
            self.assertEqual(len(rows),36);self.assertEqual(rows[0]['status'],'error')
            self.assertEqual(rows[0]['origin'],'historical');self.assertEqual(rows[0]['historical_evidence'],c.historical_slot())
            self.assertTrue(all(r['status']=='not_run' for r in rows[1:]))
    def test_consumed_attempt_stays_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'qwen06'/'clip-000002';p.mkdir(parents=True);c.write_once(p/'started.json',{})
            rows=c.denominator(c.blind_job(self.public['records']),d)
            self.assertEqual(rows[1]['attempts'],1);self.assertEqual(rows[1]['status'],'error')
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
            with self.assertRaisesRegex(ValueError,'stop threshold'):docker.check()
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
                self.assertEqual(receipt['primary_error']['error_type'],'RuntimeError')
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


class HostErrorDiagnosticTests(unittest.TestCase):
    secret='PRIVATE_TOKEN /home/private/audio.wav transcript argv=secret env=secret'
    redacted={'file':'<redacted>','function':'<redacted>','line':None}

    def assert_safe(self,value):
        raw=c.canonical(value).decode()
        self.assertLessEqual(len(raw.encode()),host.MAX_HOST_ERROR_BYTES)
        self.assertLessEqual(len(value.get('frames',[])),host.MAX_HOST_ERROR_FRAMES)
        self.assertLessEqual(set(value),{'error_type','returncode','errno','frames','frames_truncated'})
        for token in ('PRIVATE_','/home/private','transcript','argv=','env=',str(HERE)):
            self.assertNotIn(token,raw)
        return value

    def raised(self,error,function=None):
        try:
            if function is None:raise error
            function()
        except BaseException as caught:return caught

    def test_errno_is_fixed_integer_allowlist_without_filename_or_message(self):
        for number in (2,5,10,13,28,116):
            value=self.assert_safe(host.safe_error(OSError(number,self.secret,self.secret)))
            self.assertEqual(value['errno'],number);self.assertEqual(value['frames'],[])
        class PrivateInt(int):
            def __str__(self):raise AssertionError('do not convert')
        for number in (None,False,True,-1,0,4095,4096,2**1000,self.secret,PrivateInt(13)):
            error=OSError();error.errno=number
            self.assertNotIn('errno',self.assert_safe(host.safe_error(error)))

    def test_exception_subclass_hooks_and_private_class_names_are_ignored(self):
        class Hostile(OSError):
            def __str__(self):raise AssertionError('do not stringify')
            def __repr__(self):raise AssertionError('do not represent')
            def __getattribute__(self,name):raise AssertionError('do not read '+name)
            @property
            def errno(self):raise AssertionError('do not read errno property')
            @property
            def __traceback__(self):raise AssertionError('do not read traceback property')
        Hostile.__name__=self.secret
        error=Hostile(13,self.secret,self.secret)
        value=self.assert_safe(host.safe_error(self.raised(error)))
        self.assertEqual((value['error_type'],value['errno']),('OSError',13))
        self.assertEqual(value['frames'],[self.redacted])
        class Unknown(BaseException):
            def __str__(self):raise AssertionError('do not stringify')
            def __getattribute__(self,name):raise AssertionError('do not inspect')
        Unknown.__name__=self.secret
        self.assertEqual(host.safe_error(Unknown(self.secret)),{'error_type':'OtherError'})
        self.assertEqual(host.safe_error(object()),{'error_type':'OtherError'})

    def test_returncode_uses_bounded_stored_integer_without_subclass_properties(self):
        class Hostile(subprocess.CalledProcessError):
            def __getattribute__(self,name):raise AssertionError('do not inspect '+name)
            def __str__(self):raise AssertionError('do not stringify')
            @property
            def __dict__(self):raise AssertionError('do not read dict property')
        for number in (-255,-1,0,255,True,256,2**1000,self.secret):
            error=Hostile(number,[self.secret],output=self.secret,stderr=self.secret)
            value=self.assert_safe(host.safe_error(error))
            self.assertEqual(value['error_type'],'CalledProcessError')
            self.assertEqual('returncode' in value,type(number) is int and -255<=number<=255)
        class HostileFields(dict):
            def get(self,*args):raise AssertionError('do not call overridden dict method')
        error=subprocess.CalledProcessError(7,[self.secret])
        error.__dict__=HostileFields(returncode=7)
        self.assertEqual(self.assert_safe(host.safe_error(error))['returncode'],7)
        self.assertEqual(self.assert_safe(host.safe_error(subprocess.TimeoutExpired([self.secret],5,
            output=self.secret,stderr=self.secret)))['error_type'],'TimeoutExpired')

    def test_trusted_host_method_frame_uses_only_fixed_relative_source(self):
        import linecache,traceback
        error=OSError(5,self.secret,self.secret)
        with patch.object(host.subprocess,'run',side_effect=error), \
                patch.object(linecache,'getline',side_effect=AssertionError('source line read')), \
                patch.object(traceback,'format_exception',side_effect=AssertionError('traceback formatting')):
            observed=self.raised(error,lambda:host.MeteredDocker.raw(None,'inspect','owned'))
            value=self.assert_safe(host.safe_error(observed))
        trusted=[frame for frame in value['frames'] if frame['file']!='<redacted>']
        self.assertEqual(len(trusted),1)
        self.assertEqual((trusted[0]['file'],trusted[0]['function']),
            ('research/qwen12-native-admission-v1/hosted_run.py','raw'))
        self.assertGreater(trusted[0]['line'],0);self.assertEqual(value['errno'],5)

    def test_forged_source_names_and_equal_code_copies_are_not_trusted(self):
        import types
        for filename in (self.secret,str(HERE/'hosted_run.py')):
            scope={'error':OSError(13,self.secret)}
            exec(compile('def raw():\n    raise error\n',filename,'exec'),scope)
            value=self.assert_safe(host.safe_error(self.raised(scope['error'],scope['raw'])))
            self.assertTrue(all(frame==self.redacted for frame in value['frames']))
        original=host.MeteredDocker.raw.__code__;clone=original.replace()
        self.assertIsNot(clone,original);self.assertEqual(clone,original)
        replacement=types.FunctionType(clone,host.__dict__)
        error=OSError(13,self.secret)
        with patch.object(host.MeteredDocker,'raw',replacement),patch.object(host.subprocess,'run',side_effect=error):
            value=self.assert_safe(host.safe_error(self.raised(error,lambda:host.MeteredDocker.raw(None,'inspect','owned'))))
        self.assertTrue(all(frame==self.redacted for frame in value['frames']))
        self.assertNotIn(id(clone),host._HOST_DIAGNOSTIC_CODES)

    def test_long_traceback_and_cyclic_chains_are_bounded_and_not_followed(self):
        def recurse(count):
            if count:return recurse(count-1)
            raise OSError(5,self.secret)
        error=self.raised(None,lambda:recurse(200))
        error.__cause__=error;error.__context__=OSError(13,self.secret)
        value=self.assert_safe(host.safe_error(error))
        self.assertEqual(len(value['frames']),host.MAX_HOST_ERROR_FRAMES)
        self.assertIs(value['frames_truncated'],True);self.assertEqual(value['errno'],5)
        self.assertTrue(all(frame==self.redacted for frame in value['frames']))

    def test_diagnostic_failure_and_oversize_fall_back_to_primary_type(self):
        error=OSError(5,self.secret)
        with patch.object(host,'_host_error_frames',side_effect=MemoryError(self.secret)):
            self.assertEqual(host.safe_error(error),{'error_type':'OSError'})
        with patch.object(host,'MAX_HOST_ERROR_BYTES',1):
            self.assertEqual(host.safe_error(error),{'error_type':'OSError'})
        process=subprocess.CalledProcessError(7,[self.secret])
        with patch.object(host,'_host_error_frames',side_effect=MemoryError(self.secret)):
            self.assertEqual(host.safe_error(process),{'error_type':'CalledProcessError','returncode':7})

    def test_primary_and_cleanup_have_independent_errno_and_frames(self):
        primary=OSError(5,self.secret);cleanup=OSError(13,self.secret)
        with patch.object(host.subprocess,'run',side_effect=primary):
            first=self.assert_safe(host.safe_error(self.raised(primary,
                lambda:host.MeteredDocker.raw(None,'inspect','owned'))))
        with patch.object(host.subprocess,'run',side_effect=cleanup):
            second=self.assert_safe(host.safe_error(self.raised(cleanup,
                lambda:host.MeteredDocker.raw(None,'rm','--force','owned'))))
        self.assertEqual((first['errno'],second['errno']),(5,13))
        self.assertIsNot(first['frames'],second['frames'])
        self.assertEqual(host.safe_error(primary),first)


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
                self.assertEqual(receipt['primary_error']['error_type'],'ValueError')
                self.assertEqual(receipt['diagnostic']['stage'],'package_budget_preflight')
                self.assertEqual(receipt['diagnostic']['reason_code'],
                    'diagnostic_unavailable' if diagnostic_fails else 'compressed_cap' if typed else 'unknown_failure')
                self.assertEqual(len(outcomes),36);self.assertEqual(outcomes[0]['status'],'error')
                self.assertEqual(outcomes[0]['origin'],'historical')
                self.assertTrue(all(row['status']=='not_run' for row in outcomes[1:]))
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


class RemainingSlotTests(unittest.TestCase):
    """Pure metadata/control-flow mocks. No audio/model/network and no admission."""
    def setUp(self):
        self.job=c.original_job()

    def dispatch(self):
        return ContractTests().dispatch_fixture()

    def test_exact35_plan_partition_and_original_pins(self):
        all_slots=c.original_slots(self.job);plan=c.remaining_plan(self.job)
        self.assertEqual((len(all_slots),len(plan)),(36,35))
        old={c.slot_key(c.historical_slot())};new={c.slot_key(row) for row in plan}
        self.assertFalse(old & new);self.assertEqual(len(old|new),36)
        self.assertEqual([len(c.remaining_plan(self.job,f)) for f in c.FAMILIES],[11,12,12])
        self.assertEqual([r['audio_id'] for r in c.planned_clips(self.job,'qwen06')],
                         [f'clip-{i:06d}' for i in range(2,13)])
        self.assertEqual(len(c.canonical(self.job)),2415)
        self.assertEqual(c.digest(c.canonical(self.job)),c.ORIGINAL_JOB_SHA256)
        self.assertEqual(c.identity(HERE/'public-inputs.json')['sha256'],c.PUBLIC_INPUTS_SHA256)

    def test_original_job_all_alias_subset_order_and_extra_changes_reject(self):
        changes=[]
        for key,value in (('audio_id','clip-000001'),('wav_sha256','0'*64),
                          ('audio_path','audio/alias.wav'),('source_id','alias'),
                          ('pcm_sha256','0'*64),('audio_frames',1),('family','sensevoice')):
            job=copy.deepcopy(self.job);job['clips'][1][key]=value;changes.append(job)
        for rows in (self.job['clips'][::-1],self.job['clips'][1:],
                     self.job['clips']+[self.job['clips'][0]],self.job['clips'][1:]+self.job['clips'][:1],tuple(self.job['clips'])):
            job=copy.deepcopy(self.job);job['clips']=rows;changes.append(job)
        for job in changes:
            with self.subTest(job=job),self.assertRaises(ValueError):c.remaining_plan(job)
        for family in ('qwen','Qwen06',None,True):
            if family is None:continue
            with self.subTest(family=family),self.assertRaises(ValueError):c.planned_clips(self.job,family)

    def test_every_predecessor_and_plan_leaf_is_exact_and_typed(self):
        env,event,release=self.dispatch();contract=release['remaining_slot_contract']
        def leaves(value,path=()):
            if type(value) is dict:
                for key,child in value.items():yield from leaves(child,path+(key,))
            elif type(value) is list:
                for index,child in enumerate(value):yield from leaves(child,path+(index,))
            else:yield path,value
        for path,value in leaves(contract):
            mutations=(True,float(value),str(value)) if type(value) is int else ('unexpected',None)
            for bad_value in mutations:
                bad=copy.deepcopy(release);parent=bad['remaining_slot_contract']
                for key in path[:-1]:parent=parent[key]
                parent[path[-1]]=bad_value
                with self.subTest(path=path,value=bad_value),self.assertRaisesRegex(ValueError,'predecessor'):
                    c.fresh_run(env,event,bad)
        for key in contract:
            bad=copy.deepcopy(release);del bad['remaining_slot_contract'][key]
            with self.subTest(missing=key),self.assertRaises(ValueError):c.fresh_run(env,event,bad)

    def test_changed_or_unknown_started_rosters_and_historical_key_reject(self):
        env,event,release=self.dispatch()
        mutations=[]
        for index in range(5):
            bad=copy.deepcopy(release)
            bad['remaining_slot_contract']['executions'][index]['started_members'].append(
                {'path':'output/whisper/clip-000001/started.json','bytes':930,'sha256':'a'*64})
            mutations.append(bad)
        for field in ('executions','historical_consumed','planned_new_slots'):
            bad=copy.deepcopy(release);bad['remaining_slot_contract'][field].reverse();mutations.append(bad)
            bad=copy.deepcopy(release);bad['remaining_slot_contract'][field].append(copy.deepcopy(bad['remaining_slot_contract'][field][0]));mutations.append(bad)
        # A one-row list's reversal is not a mutation.
        for bad in mutations:
            if c.canonical(bad)==c.canonical(release):continue
            with self.assertRaises(ValueError):c.fresh_run(env,event,bad)
        release['predecessor_zero_recognition']={}
        with self.assertRaises(ValueError):c.fresh_run(env,event,release)

    def test_invalid_release_precedes_all_host_effects(self):
        env,event,release=self.dispatch();release['remaining_slot_contract']['maximum_new_starts']=36
        with patch.object(host,'load') as load,patch.object(host.time,'monotonic') as clock,             patch.object(host.urllib.request,'build_opener') as opener,patch.object(host,'MeteredDocker') as docker:
            with self.assertRaises(ValueError):host.execute(env,event,release,{})
            for mocked in (load,clock,opener,docker):mocked.assert_not_called()
        env,event,release=self.dispatch();env['GITHUB_RUN_ID']='38173984595'
        with self.assertRaisesRegex(ValueError,'historical execution'):c.fresh_run(env,event,release)

    def test_excluded_and_unknown_outputs_are_fatal(self):
        for name in ('qwen06/clip-000001/started.json','qwen06/clip-000001/completed.json',
                     'qwen06/clip-000001/raw-native-output.json','qwen06/clip-000001.native-observation.json',
                     'qwen06/clip-000013/started.json','fake/clip-000002/started.json',
                     'qwen06/renamed/started.json','qwen06/clip-000002/unknown.json','qwen06/unknown.json'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/name;p.parent.mkdir(parents=True);c.write_once(p,{})
                with self.subTest(name=name),self.assertRaises(ValueError):c.denominator(self.job,d)

    def mock_slot(self,root,slot,complete=True):
        directory=Path(root)/slot['family']/slot['audio_id'];directory.mkdir(parents=True)
        start={'schema':'retained-asr-started-v1','recognizer_id':slot['family'],
               'audio_id':slot['audio_id'],'file_sha256':slot['wav_sha256'],
               'pcm_sha256':slot['pcm_sha256'],'audio_frames':slot['audio_frames'],
               'source_id':slot['source_id'],'run_id':'999','executed_head_sha':'a'*40,
               'request_id':'qwen12-999-'+slot['family']+'-'+slot['audio_id'],
               'original_job_sha256':c.ORIGINAL_JOB_SHA256,
               'remaining_plan_sha256':c.digest(c.canonical(c.remaining_plan(self.job,slot['family'])))}
        c.write_once(directory/'started.json',start)
        if complete:
            c.write_once(directory/'raw-transcript.json',{'fixture':'invented-no-native-model-call'})
            c.write_once(directory/'completed.json',{**start,'schema':'retained-asr-completed-v1',
                'started_sha256':c.identity(directory/'started.json')['sha256'],
                'raw_output_sha256':c.identity(directory/'raw-transcript.json')['sha256'],
                'status':'success','decoder_completed':True})

    def test_mock35_complete_retains_old_error_and_only11_triplets(self):
        import native_worker
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for slot in c.remaining_plan(self.job):self.mock_slot(root,slot)
            for family in c.FAMILIES:
                completed=native_worker.worker_completion(self.job,root/family,family,'decode')
                c.write_once(root/family/'worker-complete.json',completed)
                actual=host.validate_worker_completion(self.job,root,family)
                self.assertEqual(actual['calls'],11 if family=='qwen06' else 12)
            rows=c.denominator(self.job,root);summary=c.terminal_summary(rows)
            self.assertEqual((summary['current_started'],summary['current_completed'],summary['cumulative_started']), (35,35,36))
            self.assertEqual(summary['complete_triplets'],11);self.assertTrue(summary['planned35_complete'])
            self.assertEqual(rows[0]['status'],'error');self.assertEqual(rows[0]['historical_evidence'],c.historical_slot())
            self.assertFalse((root/'qwen06/clip-000001').exists());self.assertEqual(len(list(root.glob('*/clip-*/started.json'))),35)
            self.assertEqual(len(rows),36);self.assertFalse(summary['training_admitted'])

    def test_mock_first_middle_final_failure_stops_no_later_calls(self):
        for failure_index in (0,4,17,23,34):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);calls=[]
                def decode(family,seconds):
                    for slot in c.remaining_plan(self.job,family):
                        index=len(calls);calls.append((family,slot['audio_id']))
                        self.mock_slot(root,slot,complete=index!=failure_index)
                        if index==failure_index:raise ValueError('synthetic stop first failure')
                with self.assertRaisesRegex(ValueError,'synthetic stop'):
                    host.recognition_stages(lambda *a:None,lambda:None,decode)
                summary=c.terminal_summary(c.denominator(self.job,root))
                self.assertEqual(len(calls),failure_index+1)
                self.assertEqual(summary['current_started'],failure_index+1)
                self.assertEqual(summary['current_completed'],failure_index)
                self.assertEqual(summary['current_errors'],1);self.assertEqual(summary['historical_errors'],1)
                self.assertEqual(summary['not_run'],34-failure_index)
                self.assertFalse(summary['planned35_complete'])
                self.assertFalse((root/'qwen06/clip-000001').exists())

    def test_invalid_accounting_preserves_primary_known_bytes_and_safe_package(self):
        import log_transport
        for illegal in ('qwen06/clip-000001/started.json','qwen06/clip-000002/unknown-private.json'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);output=root/'output';output.mkdir()
                self.mock_slot(output,c.remaining_plan(self.job)[0],complete=False)
                original=(output/'qwen06/clip-000002/started.json').read_bytes()
                path=output/illegal;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(c.canonical({'fixture':'unapproved-unknown-body' if 'unknown' in illegal else 'known-public-invalid-slot'}))
                primary={'error_type':'RuntimeError'};receipt={'primary_error':primary,'status':'stopped_no_retry'}
                outcomes,invalid=host.terminal_accounting(self.job,output,receipt)
                self.assertTrue(invalid);self.assertIs(receipt['primary_error'],primary)
                self.assertEqual(receipt['status'],'accounting_invalid');self.assertEqual(len(outcomes),36)
                self.assertEqual(outcomes[0]['status'],'error')
                self.assertTrue(all(row['status']=='accounting_invalid' and row['attempts'] is None for row in outcomes[1:]))
                self.assertIsNone(receipt['recognition_accounting']['current_started'])
                self.assertFalse(receipt['recognition_accounting']['planned35_complete'])
                c.write_once(root/'host-receipt.json',receipt);c.write_once(root/'terminal-outcomes.json',outcomes)
                allowed=host.collect(root)
                self.assertNotIn('output/qwen06/clip-000002/unknown-private.json',allowed)
                transport={'repository':'jiying2007/kws-pipeline','run_id':999,'run_attempt':1,
                           'job_key':'qwen12','executed_head_sha':'a'*40}
                payload,manifest,lines=log_transport.build_package(root,allowed,transport)
                recovered=log_transport.parse_job_log('\n'.join(lines),transport)
                self.assertEqual(recovered[0],payload)
                self.assertEqual(recovered[2]['archive_sha256'],c.digest(payload))
                import zipfile
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    self.assertEqual(archive.read('output/qwen06/clip-000002/started.json'),original)
                    self.assertIn('terminal-outcomes.json',archive.namelist())
                self.assertNotIn('unapproved-unknown-body',c.canonical(receipt).decode())

    def test_invalid_job_fallback_uses_only_fixed_trusted_original_roster(self):
        with tempfile.TemporaryDirectory() as d:
            receipt={'primary_error':{'error_type':'RuntimeError'}}
            bad=c.original_job();bad['clips'].reverse()
            outcomes,invalid=host.terminal_accounting(bad,Path(d),receipt)
            self.assertTrue(invalid);self.assertEqual(len(outcomes),36)
            self.assertEqual([row['audio_id'] for row in outcomes[:12]],
                             [f'clip-{i:06d}' for i in range(1,13)])
            self.assertEqual(receipt['primary_error'],{'error_type':'RuntimeError'})
            self.assertIsNone(receipt['recognition_accounting']['current_errors'])
            self.assertIsNone(receipt['recognition_accounting']['not_run'])

    def test_execute_invalid_accounting_stops_and_retains_complete_safe_package(self):
        import branch_transport,worker_receipts,zipfile,log_transport
        from contextlib import ExitStack,redirect_stdout
        for illegal in ('qwen06/clip-000001/started.json','qwen06/clip-000002/unknown-private.json'):
            with tempfile.TemporaryDirectory() as d:
                calls=[];original_bytes=[]
                original=SimpleNamespace(host_capacity=lambda:{},pull_image=lambda *a:None)
                def supervise(*args,**kwargs):
                    family=args[6];mode=kwargs.get('mode');calls.append((family,mode))
                    if family=='setup':c.atomic(args[3]/'setup-receipt.json',{'actual_download_bytes':0,'commands':[]})
                    elif mode=='prepare':
                        c.write_once(args[5]/'native-config.json',{})
                        c.write_once(args[5]/'prepared.json',{'audio_forward_calls':0})
                        c.write_once(args[5]/'worker-complete.json',{'calls':0})
                    else:
                        self.assertEqual((family,mode),('qwen06','decode'))
                        self.mock_slot(args[5].parent,c.remaining_plan(self.job)[0],complete=False)
                        original_bytes.append((args[5]/'clip-000002/started.json').read_bytes())
                        path=args[5].parent/illegal;path.parent.mkdir(parents=True,exist_ok=True)
                        c.write_once(path,{'fixture':'PRIVATE_UNKNOWN_BODY' if 'unknown' in illegal else 'public-invalid-slot'})
                        raise RuntimeError('PRIVATE_PRIMARY_ERROR')
                actual_copyfile=host.shutil.copyfile
                def copyfile(source,destination,*args,**kwargs):
                    if Path(source).name=='cpu_gate.py':return None
                    return actual_copyfile(source,destination,*args,**kwargs)
                with ExitStack() as stack:
                    for context in (patch.object(sys,'path',list(sys.path)),
                        patch.object(host,'fresh_run',return_value={'run_id':'999','head_sha':'a'*40}),
                        patch.object(host,'load',return_value=original),patch.object(host.signal,'signal'),
                        patch.object(host.signal,'setitimer'),patch.object(host.shutil,'disk_usage',return_value=SimpleNamespace(free=20*c.GIB)),
                        patch.object(host,'fetch_blind',return_value={}),patch.object(host,'validate_blind',return_value=self.job),
                        patch.object(host,'stage'),patch.object(host.shutil,'copyfile',side_effect=copyfile),patch.object(host,'supervise',side_effect=supervise),
                        patch.object(host,'whisper_setup',return_value={}),patch.object(worker_receipts,'build_policy',return_value={}),
                        patch.object(host,'package_budget_preflight',return_value={}),redirect_stdout(io.StringIO())):
                        stack.enter_context(context)
                    package=stack.enter_context(patch.object(branch_transport,'write_package'))
                    status=host.execute({'RUNNER_TEMP':d,'GITHUB_JOB':'qwen12'},{},
                        {'new_transcripts_publication_approved':True},{'inference_code':{},'setup_code':{}})
                self.assertEqual(status,1)
                self.assertEqual(calls,[('setup',None)]+[(f,'prepare') for f in c.FAMILIES]+[('qwen06','decode')])
                package.assert_called_once();payload=package.call_args.args[1];transport=package.call_args.args[2]
                log_transport.validate_archive(payload,transport)
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    self.assertEqual(archive.read('output/qwen06/clip-000002/started.json'),original_bytes[0])
                    self.assertNotIn('output/qwen06/clip-000002/unknown-private.json',archive.namelist())
                    self.assertTrue(all(b'PRIVATE_' not in archive.read(name) for name in archive.namelist()))
                    receipt=json.loads(archive.read('host-receipt.json'))
                    outcomes=json.loads(archive.read('terminal-outcomes.json'))
                self.assertEqual(receipt['status'],'accounting_invalid')
                self.assertEqual(receipt['primary_error']['error_type'],'RuntimeError')
                self.assertEqual(receipt['accounting_error']['error_type'],'ValueError')
                self.assertEqual(receipt['stages']['qwen06'],'error')
                self.assertEqual(receipt['stages']['sensevoice'],'not_run')
                self.assertEqual(receipt['stages']['whisper'],'not_run')
                self.assertIsNone(receipt['recognition_accounting']['cumulative_started'])
                self.assertEqual(outcomes[0]['status'],'error')
                self.assertTrue(all(r['status']=='accounting_invalid' and r['attempts'] is None for r in outcomes[1:]))

    def test_count_only_or_wrong_family_member_completion_is_rejected(self):
        import native_worker
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'qwen06').mkdir()
            c.write_once(root/'qwen06/worker-complete.json',{'calls':11,'completed':11,
                'audio_ids':[r['audio_id'] for r in c.remaining_plan(self.job,'qwen06')],
                'status':'all_planned_native_v2_complete'})
            with self.assertRaises(ValueError):host.validate_worker_completion(self.job,root,'qwen06')
            with self.assertRaises(ValueError):native_worker.worker_completion(self.job,root/'qwen06','qwen06','decode')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for row in c.remaining_plan(self.job,'qwen06'):self.mock_slot(root,row)
            receipt=native_worker.worker_completion(self.job,root/'qwen06','qwen06','decode')
            for field,value in (('calls',True),('calls',11.0),('completed',12),('audio_ids',receipt['audio_ids'][::-1])):
                bad=dict(receipt);bad[field]=value
                (root/'qwen06/worker-complete.json').write_bytes(c.canonical(bad))
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):host.validate_worker_completion(self.job,root,'qwen06')


class NativePlannedLoopTests(unittest.TestCase):
    """Exercise real loop bodies with only stdlib fakes, never model/audio I/O."""
    def run_loop(self,family='qwen06',fail=None,job=None):
        import contextlib
        import native_worker as n
        import native_locks
        import whisper_binding
        from unittest.mock import Mock
        job=c.original_job() if job is None else job
        events=[];sources={oid:(wav,pcm,frames) for oid,wav,pcm,frames in c.ORIGINAL_SOURCES}
        descriptors={oid:{'wav':{'bytes':44+2*frames},'raw_pcm16':{'frame_count':frames}}
                     for oid,(wav,pcm,frames) in sources.items()}
        def bind_wave(root,expect):
            events.append(('bind',expect.audio_id))
            return SimpleNamespace(descriptor=descriptors[expect.audio_id],audio_id=expect.audio_id)
        def infer(bound,*args):
            events.append(('infer',bound.audio_id))
            if bound.audio_id==fail:raise RuntimeError('invented middle failure')
            return {},{}
        def start(binding,directory,**kwargs):
            oid=kwargs['audio_id'];wav,pcm,frames=sources[oid]
            c.validate_start_slot(job,family,audio_id=oid,measured={'file_sha256':wav,'pcm_sha256':pcm,'audio_frames':frames},
                source_id='qwen12-'+wav,request_id=kwargs['request_id'],run_id='999')
            events.append(('start',oid));c.write_once(directory/'started.json',{'fixture':'only'})
            return SimpleNamespace(directory=directory,audio_id=oid)
        def complete(attempt,**kwargs):events.append(('complete',attempt.audio_id))
        locks={name:{'asset_lock':{},'asset_lock_sha256':'a'*64,'source_lock':{},'source_lock_sha256':'b'*64} for name in c.FAMILIES}
        worker=SimpleNamespace(THREAD_ENV={},OFFLINE_ENV={},RETAINED=HERE,
            verify_scope=lambda:None,prepare_caches=lambda:None,verify_sources=lambda:None,
            import_helpers=lambda:(None,None),bind_inputs=lambda p:(job,c.canonical(job),b'{}'),
            decode=lambda body,*a:json.loads(body),read_regular=lambda *a:c.canonical(locks),
            execution_contract=lambda *a:{},load_qwen_after_approval=lambda *a:SimpleNamespace(receipt={}),
            load_sense_after_approval=lambda *a:SimpleNamespace(receipt={}),
            validate_decoder=lambda *a:{oid:{'descriptor':descriptor} for oid,descriptor in descriptors.items()},
            WaveExpectation=lambda oid,*a:SimpleNamespace(audio_id=oid),bind_wave=bind_wave)
        adapter=SimpleNamespace(NativeCaptureAdapter=lambda *a,**k:SimpleNamespace(infer=infer))
        modules={'qwen12_retained_worker':worker,'qwen12_setup_binding':SimpleNamespace(verify_runtime=lambda *a:{}),
                 'qwen12_native_adapter':adapter,'qwen12_whisper_adapter':SimpleNamespace()}
        def pcm(path,wav):
            oid=next(oid for oid,row in sources.items() if row[0]==wav)
            events.append(('bind',oid));return oid,sources[oid][2]
        def whisper_decode(oid,frames,directory):
            events.append(('infer',oid))
            if oid==fail:raise RuntimeError('invented middle failure')
            return {}
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d);output=root/'output';output.mkdir()
            stack.enter_context(patch.dict(os.environ,{'KWS_RUN_ID':'999','KWS_EXECUTED_HEAD_SHA':'a'*40,'KWS_POLICY_SHA256':'b'*64}))
            stack.enter_context(patch.dict(sys.modules,{'common':SimpleNamespace(wav_bytes=pcm)}))
            module=stack.enter_context(patch.object(n,'module',side_effect=lambda name,path:modules[name]))
            for name,replacement in (('read_json',lambda *a:{}),('prepared_or_compare',lambda *a:False),
                    ('bind_predecode',lambda **kwargs:object()),('emit_start',start),('emit_completion',complete)):
                stack.enter_context(patch.object(n,name,side_effect=replacement))
            stack.enter_context(patch.object(n.signal,'setitimer'))
            stack.enter_context(patch.object(native_locks,'verify_installed_extension_sources',return_value={}))
            stack.enter_context(patch.object(native_locks,'build_native_locks',return_value=(b'{}',b'{}')))
            stack.enter_context(patch.object(native_locks,'build_whisper_locks',return_value=(b'{}',b'{}')))
            stack.enter_context(patch.object(whisper_binding,'construct_whisper',return_value=SimpleNamespace(decode=whisper_decode)))
            try:
                (n.whisper if family=='whisper' else lambda *a:n.qwen_sense(family,*a))(root/'runtime',root/'input',output,job,'decode')
            except BaseException as error:
                return events,error,sorted(p.name for p in output.iterdir()),module.call_count
            return events,None,sorted(p.name for p in output.iterdir()),module.call_count

    def test_real_qwen_loop_skips_old_before_binding_directory_and_call(self):
        events,error,names,_=self.run_loop()
        self.assertIsNone(error)
        expected=[f'clip-{i:06d}' for i in range(2,13)]
        for operation in ('bind','start','infer','complete'):
            self.assertEqual([oid for op,oid in events if op==operation],expected)
        self.assertNotIn('clip-000001',names)

    def test_real_qwen_loop_middle_failure_consumes_once_and_stops(self):
        events,error,names,_=self.run_loop(fail='clip-000005')
        self.assertIsInstance(error,RuntimeError)
        self.assertEqual([oid for op,oid in events if op=='infer'],[f'clip-{i:06d}' for i in range(2,6)])
        self.assertEqual([oid for op,oid in events if op=='complete'],[f'clip-{i:06d}' for i in range(2,5)])
        self.assertNotIn('clip-000001',names);self.assertNotIn('clip-000006',names)

    def test_real_whisper_loop_keeps_original12_and_stops_on_first_error(self):
        events,error,names,_=self.run_loop(family='whisper')
        self.assertIsNone(error)
        self.assertEqual([oid for op,oid in events if op=='infer'],[f'clip-{i:06d}' for i in range(1,13)])
        events,error,names,_=self.run_loop(family='whisper',fail='clip-000003')
        self.assertIsInstance(error,RuntimeError)
        self.assertEqual([oid for op,oid in events if op=='infer'],[f'clip-{i:06d}' for i in range(1,4)])
        self.assertNotIn('clip-000004',names)

    def test_fake_subset_or_reordered_loop_job_rejects_before_load_or_bind(self):
        for clips in (c.original_job()['clips'][:1],c.original_job()['clips'][1:],c.original_job()['clips'][::-1]):
            for family in ('qwen06','whisper'):
                job=c.original_job();job['clips']=clips
                events,error,names,loads=self.run_loop(family=family,job=job)
                self.assertIsInstance(error,ValueError);self.assertEqual(events,[])
                self.assertEqual(names,[]);self.assertEqual(loads,0)


class FilesystemSamplingTests(unittest.TestCase):
    def docker(self,root):return host.MeteredDocker(root,host.time.monotonic()+30)
    def partial(self,n=3):return dict(observed_bytes=n,vanished_entries=1,complete=False,missing_root=False,root_missing_at_start=False,coverage='incomplete_lower_bound')
    def test_static_bytes_and_root_json_keep_nofollow_rules(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();(r/'a.json').write_bytes(b'abc');(r/'b').write_bytes(b'12345');o=Path(td)/'outside';o.write_bytes(b'x'*4096);(r/'link.json').symlink_to(o);(r/'sub').mkdir();(r/'sub/c.json').write_bytes(b'zz')
            s={};self.assertEqual(host.tree_bytes(r,s),10+len(str(o)));self.assertTrue(s['complete']);self.assertEqual(s['symlinks'],1);self.assertEqual(host.tree_bytes(r,{},root_json_only=True),3)
    def test_one_nofollow_stat_per_entry(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat;calls=[]
            def observe(p,*a,**kw):
                if p=='item':calls.append(kw.copy())
                return real(p,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(host.tree_bytes(r),1)
            self.assertEqual(len(calls),1);self.assertIs(calls[0]['follow_symlinks'],False);self.assertIsInstance(calls[0]['dir_fd'],int)
    def test_real_unlink_then_stable_resample_returns_only_second(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);p=r/'vanish';p.write_bytes(b'1234');(r/'stable').write_bytes(b'abc');real=host.os.stat;done=False
            def observe(name,*a,**kw):
                nonlocal done
                if name=='vanish' and not done:done=True;p.unlink()
                return real(name,*a,**kw)
            with patch.object(host.os,'stat',observe):self.assertEqual(d.sample_bytes(r,'workspace',100),3)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual((row['scans'],row['selected_sample']),(2,2));self.assertEqual(row['last_samples'][0]['vanished_entries'],1);self.assertFalse(row['last_samples'][0]['complete']);self.assertEqual(row['last_samples'][1]['observed_bytes'],3)
    def test_two_partial_samples_fail_without_third(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial());return 3
            with patch.object(host,'tree_bytes',sample),self.assertRaisesRegex(ValueError,'one bounded resample'):d.sample_bytes(td,'workspace',10)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual(len(calls),2);self.assertIsNone(row['selected_sample']);self.assertEqual(row['vanished_entries'],2)
    def test_partial_over_cap_stops_before_resampling(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial(11));return 11
            with patch.object(host,'tree_bytes',sample),self.assertRaises(ValueError):d.sample_bytes(td,'workspace',10)
            self.assertEqual(len(calls),1);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['status'],'cap_exceeded')
    def test_complete_real_over_cap_stops(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);(Path(td)/'large').write_bytes(b'x'*11)
            with self.assertRaises(ValueError):d.sample_bytes(td,'workspace',10)
            self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1)
    def test_deadline_exhausted_does_not_start_resample(self):
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);calls=[]
            def sample(root,obs,**kw):calls.append(1);obs.update(self.partial());return 3
            with patch.object(host,'tree_bytes',sample),patch.object(host.time,'monotonic',side_effect=[d.deadline-1,d.deadline]),self.assertRaisesRegex(ValueError,'wall ceiling'):d.sample_bytes(td,'workspace',10)
            self.assertEqual(len(calls),1);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['status'],'deadline_exhausted')
    def test_non_enoent_fail_closed_and_keep_original_exception(self):
        import errno
        for number in (errno.EACCES,errno.EIO,errno.ELOOP,errno.ENOTDIR,None,9999):
            with self.subTest(errno=number),tempfile.TemporaryDirectory() as td:
                d=self.docker(td);(Path(td)/'item').write_bytes(b'x');real=host.os.stat;error=OSError(number,'PRIVATE','/PRIVATE')
                def stat(p,*a,**kw):
                    if p=='item':raise error
                    return real(p,*a,**kw)
                with patch.object(host.os,'stat',stat),self.assertRaises(OSError) as caught:d.sample_bytes(td,'workspace',100)
                self.assertIs(caught.exception,error);self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1);self.assertNotIn('PRIVATE',c.canonical(d.filesystem_sampling).decode())
    def test_partial_before_eio_preserves_counters_and_lower_bound(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td)
            def sample(root,obs,**kw):obs.update(self.partial(7));raise OSError(errno.EIO,'PRIVATE')
            with patch.object(host,'tree_bytes',sample),self.assertRaises(OSError):d.sample_bytes(td,'workspace',100)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual((row['vanished_entries'],row['incomplete_scans']),(1,1));self.assertEqual(row['last_samples'][0]['observed_bytes'],7)
    def test_optional_absence_and_required_workspace_missing_differ(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);self.assertEqual(d.sample_bytes(r/'output','public_output',100,optional_root=True),0)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertEqual(row['scans'],1);self.assertEqual(row['last_samples'][0]['coverage'],'absent_optional_root')
            r.rename(Path(td)/'moved')
            with self.assertRaises(OSError):d.sample_bytes(r,'workspace',100)
            self.assertEqual(d.filesystem_sampling['scopes']['workspace']['scans'],1)
    def test_owned_workspace_replaced_between_samples_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);r.rename(Path(td)/'old');r.mkdir()
            with self.assertRaisesRegex(ValueError,'workspace identity'):d.sample_bytes(r,'workspace',100)
    def test_root_lstat_open_replacement_rejects_and_closes(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();real=host.os.open;opened=[]
            def change(p,*a,**kw):
                if Path(p)==r:r.rename(Path(td)/'old');r.mkdir()
                fd=real(p,*a,**kw);opened.append(fd);return fd
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root directory identity'):host.tree_bytes(r)
            for fd in opened:
                with self.assertRaises(OSError):os.fstat(fd)
    def test_optional_root_seen_then_deleted_is_not_initial_absence(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();real=host.os.open
            def change(p,*a,**kw):
                if p=='output':out.rmdir()
                return real(p,*a,**kw)
            with patch.object(host.os,'open',change),self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(out,'public_output',100,optional_root=True)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertFalse(row['last_samples'][0]['root_missing_at_start']);self.assertEqual(row['absent_optional_roots'],0)
    def test_child_directory_replaced_by_symlink_or_other_directory_rejects(self):
        for symlink in (True,False):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as td:
                r=Path(td)/'task';r.mkdir();sub=r/'sub';sub.mkdir();out=Path(td)/'outside';out.mkdir();(out/'private').write_bytes(b'x'*10000);real=host.os.open
                def change(p,*a,**kw):
                    if p=='sub':
                        sub.rename(r/'old')
                        if symlink:sub.symlink_to(out,target_is_directory=True)
                        else:sub.mkdir()
                    return real(p,*a,**kw)
                with patch.object(host.os,'open',change),self.assertRaises((OSError,ValueError)):host.tree_bytes(r)
    def test_file_to_symlink_after_stat_counts_observed_file_only(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();p=r/'item';p.write_bytes(b'1234567');out=Path(td)/'outside';out.write_bytes(b'x'*10000);real=host.os.stat
            def change(name,*a,**kw):
                info=real(name,*a,**kw)
                if name=='item':p.unlink();p.symlink_to(out)
                return info
            with patch.object(host.os,'stat',change):self.assertEqual(host.tree_bytes(r),7)
            self.assertEqual(host.tree_bytes(r),len(str(out)))
    def test_all_descriptors_close_on_traversal_success_or_eio(self):
        import errno
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as td:
                r=Path(td);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x');opened=[];realopen=host.os.open;realstat=host.os.stat
                def observe(*a,**kw):fd=realopen(*a,**kw);opened.append(fd);return fd
                def stat(p,*a,**kw):
                    if fail and p=='item':raise OSError(errno.EIO,'PRIVATE')
                    return realstat(p,*a,**kw)
                with patch.object(host.os,'open',observe),patch.object(host.os,'stat',stat):
                    if fail:
                        with self.assertRaises(OSError):host.tree_bytes(r)
                    else:self.assertEqual(host.tree_bytes(r),1)
                self.assertGreaterEqual(len(opened),2)
                for fd in opened:
                    with self.assertRaises(OSError):os.fstat(fd)
    def test_real_concurrent_mutation_retains_non_atomic_sampling_semantics(self):
        import threading
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);paths=[r/str(i) for i in range(16)]
            for p in paths:p.write_bytes(b'x'*16)
            stop=threading.Event();errors=[]
            def churn():
                try:
                    while not stop.is_set():
                        for p in paths:
                            if stop.is_set():break
                            p.unlink();p.write_bytes(b'x'*16)
                except BaseException as error:errors.append(type(error).__name__)
            t=threading.Thread(target=churn);t.start()
            try:
                for _ in range(20):
                    obs={};amount=host.tree_bytes(r,obs);self.assertEqual(amount,obs['observed_bytes']);self.assertLessEqual(amount,256);self.assertEqual(obs['complete'],obs['vanished_entries']==0)
            finally:stop.set();t.join(2)
            self.assertFalse(t.is_alive());self.assertEqual(errors,[])
    def test_source_input_integrity_still_strict_for_missing_or_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'link').symlink_to('missing')
            for p in (r/'missing',r/'link'):
                with self.assertRaises((OSError,ValueError)):c.identity(p)
                with self.assertRaises((OSError,ValueError)):c.read_json(p)

    def test_early_cap_never_stats_the_next_entry(self):
        from contextlib import contextmanager
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'a_big').write_bytes(b'x'*11);(r/'z_after').write_bytes(b'x');scan=host.os.scandir;real=host.os.stat
            @contextmanager
            def ordered(fd):
                with scan(fd) as entries:yield iter(sorted(entries,key=lambda e:e.name))
            def stat(p,*a,**kw):
                if p=='z_after':raise AssertionError('read after known cap')
                return real(p,*a,**kw)
            obs={}
            with patch.object(host.os,'scandir',ordered),patch.object(host.os,'stat',stat),self.assertRaisesRegex(ValueError,'stop threshold'):host.tree_bytes(r,obs,limit=10)
            self.assertEqual(obs['observed_bytes'],11);self.assertEqual(obs['stop_reason'],'cap_exceeded')
    def test_entry_deadline_prevents_entry_metadata_access(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'item').write_bytes(b'x');real=host.os.stat
            def stat(p,*a,**kw):
                if p=='item':raise AssertionError('read after deadline')
                return real(p,*a,**kw)
            obs={}
            with patch.object(host.time,'monotonic',side_effect=[0,0,0,1]),patch.object(host.os,'stat',stat),self.assertRaisesRegex(ValueError,'wall ceiling'):host.tree_bytes(r,obs,deadline=1)
            self.assertEqual(obs['stop_reason'],'deadline_exhausted')
    def test_seen_optional_root_cannot_later_be_absent(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),0);out.rmdir()
            with self.assertRaisesRegex(ValueError,'root missing'):d.sample_bytes(out,'public_output',10,optional_root=True)
            row=d.filesystem_sampling['scopes']['public_output'];self.assertTrue(row['observed_root_once']);self.assertEqual(row['absent_optional_roots'],0)
    def test_legal_output_to_frozen_and_new_output_inode_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);out=r/'output';out.mkdir();(out/'old').write_bytes(b'123');self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),3)
            out.rename(r/'frozen');out.mkdir();(out/'new').write_bytes(b'45');self.assertEqual(d.sample_bytes(out,'public_output',10,optional_root=True),2)
            self.assertEqual(d.sample_bytes(r/'frozen','public_frozen',10,optional_root=True),3)
    def test_public_child_uses_open_workspace_fd_after_ancestor_rename(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'task';r.mkdir();d=self.docker(r);(r/'output').mkdir();(r/'output/old').write_bytes(b'123');outside=Path(td)/'outside';outside.mkdir();(outside/'output').mkdir();(outside/'output/private').write_bytes(b'x'*10000)
            real=host.os.open;changed=False
            def move(p,*a,**kw):
                nonlocal changed
                fd=real(p,*a,**kw)
                if Path(p)==r and not changed:changed=True;r.rename(Path(td)/'oldtask');r.symlink_to(outside,target_is_directory=True)
                return fd
            with patch.object(host.os,'open',move):self.assertEqual(d.sample_bytes(r/'output','public_output',100,optional_root=True),3)
            with self.assertRaises(OSError):d.sample_bytes(r/'output','public_output',100,optional_root=True)
    def test_public_aggregate_cap_does_not_reset_each_scope(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'output').mkdir();(r/'output/a').write_bytes(b'x'*6);(r/'whisper-setup').mkdir();(r/'whisper-setup/b').write_bytes(b'x'*5)
            with patch.dict(host.LIMITS,{'output_bytes':10}),self.assertRaisesRegex(ValueError,'stop threshold'):d.check()
            row=d.filesystem_sampling['scopes']['public_whisper_setup'];self.assertEqual(row['status'],'cap_exceeded');self.assertEqual(row['last_samples'][0]['observed_bytes'],5)

    def test_primary_scan_error_is_not_replaced_by_fd_cleanup_error(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'sub').mkdir();(r/'sub/item').write_bytes(b'x');realstat=host.os.stat;realclose=host.os.close;primary=OSError(errno.EACCES,'PRIVATE_PRIMARY');obs={}
            def stat(p,*a,**kw):
                if p=='item':raise primary
                return realstat(p,*a,**kw)
            def close(fd):realclose(fd);raise OSError(errno.EIO,'PRIVATE_CLOSE')
            with patch.object(host.os,'stat',stat),patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:host.tree_bytes(r,obs)
            self.assertIs(caught.exception,primary);self.assertEqual(obs['fd_close_error_count'],2);self.assertEqual(obs['fd_close_first_error']['errno'],errno.EIO);self.assertNotIn('PRIVATE',c.canonical(obs).decode())
    def test_fd_cleanup_error_without_primary_is_fatal(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            realclose=host.os.close;error=OSError(errno.EIO,'PRIVATE_CLOSE');obs={}
            def close(fd):realclose(fd);raise error
            with patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:host.tree_bytes(Path(td),obs)
            self.assertIs(caught.exception,error);self.assertEqual(obs['fd_close_error_count'],1)

    def test_close_enoent_is_never_treated_as_a_vanished_entry(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'sub').mkdir();realclose=host.os.close
            def close(fd):realclose(fd);raise OSError(errno.ENOENT,'PRIVATE_CLOSE')
            with patch.object(host.os,'close',close),self.assertRaises(OSError):d.sample_bytes(r,'workspace',100)
            row=d.filesystem_sampling['scopes']['workspace'];self.assertEqual(row['scans'],1);self.assertEqual(row['vanished_entries'],0);self.assertGreater(row['last_samples'][0]['fd_close_error_count'],0)
    def test_parent_close_failure_still_closes_workspace_anchor(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=self.docker(r);(r/'output/qwen06').mkdir(parents=True);realopen=host.os.open;realclose=host.os.close;opened=[];closed=[];error=OSError(errno.EIO,'PRIVATE_PARENT_CLOSE')
            def openfd(*a,**kw):fd=realopen(*a,**kw);opened.append(fd);return fd
            def close(fd):
                realclose(fd);closed.append(fd)
                if fd==opened[1]:raise error
            with patch.object(host.os,'open',openfd),patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:d.sample_bytes(r/'output/qwen06','phase_output',100)
            self.assertIs(caught.exception,error);self.assertEqual(set(opened),set(closed));self.assertEqual(len(opened),len(closed))
            for fd in opened:
                with self.assertRaises(OSError):os.fstat(fd)

    def test_handled_ambient_exception_does_not_hide_close_only_failure(self):
        import errno
        with tempfile.TemporaryDirectory() as td:
            d=self.docker(td);realclose=host.os.close;error=OSError(errno.EIO,'PRIVATE_CLOSE')
            def close(fd):realclose(fd);raise error
            try:raise ValueError('PRIVATE_HANDLED')
            except ValueError:
                with patch.object(host.os,'close',close),self.assertRaises(OSError) as caught:d.sample_bytes(Path(td),'workspace',100)
            self.assertIs(caught.exception,error);self.assertIsNone(d.filesystem_sampling['scopes']['workspace']['selected_sample'])

if __name__=='__main__':unittest.main()
