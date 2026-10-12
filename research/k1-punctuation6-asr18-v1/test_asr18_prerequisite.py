"""Actual local gate coverage with synthetic metadata; zero model/network/Docker work."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import campaign_contract as c
import prepare_freeze
import prepare_prerequisite as p
from test_control_fixture import synthetic_prerequisite, synthetic_enabled_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def fixture():
    release = c.disabled_release()
    for key in c.RELEASE_APPROVALS: release[key] = True
    for key in c.RELEASE_AUTHORITIES: release[key] = 'a'*64
    freeze = prepare_freeze.build(ROOT)
    raw = synthetic_prerequisite(release, root=ROOT, freeze=freeze)
    return release, freeze, raw


class PreparePrerequisite(unittest.TestCase):
    def test_checked_in_complete_prerequisite_is_exactly_pinned_and_asr_released(self):
        # Read the actual root-reviewed projection; never synthesize its proof.
        raw=c.read_bytes(HERE/'prepare-prerequisite.json',cap=65536)
        release=c.read_json(HERE/'execution-release.json')
        freeze=c.read_json(HERE/'execution-freeze.json')
        self.assertEqual(c.digest(raw),release['prepare_prerequisite_sha256'])
        value=p.validate_receipt(raw,release,source_freeze=freeze,source_root=ROOT)
        self.assertEqual(value['outcome'],'complete_prepare_verified')
        self.assertEqual(value['diagnostic_forward_calls'],0)
        for key in c.RELEASE_APPROVALS:self.assertIs(release[key],True)
        for key in ('enabled','root_publication_release','independent_source_review'):
            self.assertIs(c.read_json(HERE/'publication-release.json')[key],True)

    def test_explicit_unestablished_prerequisite_template_remains_closed(self):
        release=c.disabled_release()
        self.assertIsNone(release['prepare_prerequisite_sha256'])
        with self.assertRaisesRegex(ValueError,'unestablished'):
            p.validate_receipt(c.canonical(p.disabled_receipt()),release)

    def test_real_pure_gate_accepts_only_exact_root_pinned_complete_projection(self):
        release,freeze,raw=fixture()
        self.assertEqual(p.validate_receipt(raw,release,source_freeze=freeze)['diagnostic_forward_calls'],0)
        self.assertEqual(c.require_execution_release(release,freeze,
            expected_freeze_sha256=c.digest(c.canonical(freeze)),prepare_prerequisite_raw=raw),release)

    def test_all_actual_identity_fields_hash_bound_even_with_valid_syntax(self):
        release,freeze,raw=fixture();original=c.decode_json(raw)
        changes=[('numeric_job_id',777),('result_commit','9'*40),('archive_sha256','9'*64),
                 ('prepare_freeze_sha256','9'*64),('posterior_policy_sha256','9'*64)]
        for key,value in changes:
            bad=copy.deepcopy(original);bad[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'root-pinned'):
                p.validate_receipt(c.canonical(bad),release)
        for key,value in [('run_id',777),('executed_head_sha','9'*40)]:
            bad=copy.deepcopy(original);bad['identity'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'root-pinned'):
                p.validate_receipt(c.canonical(bad),release)

    def test_partial_fixture_publisher_only_null_old_and_wrong_scope_rejected_even_rehashed(self):
        release,_,raw=fixture();original=c.decode_json(raw)
        changes=[('evidence_kind','synthetic_zero_model_fixture'),('outcome','partial_prepare_archive_verified'),
            ('run_conclusion','failure'),('job_conclusion','failure'),('host_receipt_status','failed_no_retry'),
            ('step_conclusions',{'contracts':'success','host':'failure','publisher':'success'}),
            ('verified_prepared_families',['qwen06','sensevoice']),('verified_completed_model_constructions',2),
            ('diagnostic_forward_calls',None),('diagnostic_forward_calls',False),('diagnostic_forward_calls',1),
            ('source_ref','research/k1-punctuation6-prepare-diagnostic-v1'),
            ('result_ref','evidence/k1-punctuation6-prepare-diagnostic-v1'),
            ('source_ref','research/k1-punctuation6-prepare-diagnostic-r1'),
            ('result_ref','evidence/k1-punctuation6-prepare-diagnostic-r1'),
            ('workflow','.github/workflows/fixture.yml'),('acceptance',{}),('numeric_job_id',None),
            ('numeric_job_id',True),('result_commit',None),('archive_sha256',None),
            ('prepare_freeze_sha256',None),('posterior_policy_sha256',None)]
        for key,value in changes:
            bad=copy.deepcopy(original);bad[key]=value;body=c.canonical(bad)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                p.validate_receipt(body,{**release,'prepare_prerequisite_sha256':c.digest(body)})
        for key,values in [('run_id',p.SEALED_RUNS),('executed_head_sha',p.SEALED_SOURCES),
                           ('run_attempt',(2,True,None)),('job_key',('asr18',))]:
            for value in values:
                bad=copy.deepcopy(original);bad['identity'][key]=value;body=c.canonical(bad)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    p.validate_receipt(body,{**release,'prepare_prerequisite_sha256':c.digest(body)})

    def test_exact_no_private_allowlist_duplicate_noncanonical_rejected(self):
        release,_,raw=fixture();value=c.decode_json(raw)
        for key in ('custody_sha256','expected_sha256','inventory','observations'):
            bad={**value,key:'never_public'};body=c.canonical(bad)
            with self.assertRaises(ValueError):
                p.validate_receipt(body,{**release,'prepare_prerequisite_sha256':c.digest(body)})
        for body in (raw+b'\n',b'{"schema":null,'+raw[1:]):
            with self.assertRaises(ValueError):
                p.validate_receipt(body,{**release,'prepare_prerequisite_sha256':c.digest(body)})

    def test_same_six_input_binding_not_replaceable(self):
        release,_,raw=fixture()
        for key in p.INPUT_KEYS:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'same fixed six'):
                p.validate_receipt(raw,{**release,key:'e'*64})

    def test_scientific_identity_remeasures_bytes_and_ignores_only_four_controls(self):
        _,freeze,_=fixture()
        members={'source/'+name:c.read_bytes(ROOT/name) for name in p.SCIENTIFIC_MEMBERS}
        expected=p.scientific_identity(ROOT,freeze)
        self.assertEqual(p.scientific_identity_from_members(members,freeze),expected)
        changed=copy.deepcopy(freeze)
        for name in p.EXCLUDED:changed['inference_code'][name]={'bytes':1,'sha256':'0'*64}
        self.assertEqual(p.scientific_identity_from_members(members,changed),expected)
        name=p.SCIENTIFIC_MEMBERS[0]
        with self.assertRaisesRegex(ValueError,'byte identity'):
            p.scientific_identity_from_members({**members,'source/'+name:members['source/'+name]+b'\n'},freeze)
        changed=copy.deepcopy(freeze);changed['inference_code'].pop(name)
        with self.assertRaisesRegex(ValueError,'source roster'):p.scientific_identity(ROOT,changed)
        # A coherent caller-made freeze cannot replace the independently pinned prepare science.
        release,freeze,raw=fixture();changed=copy.deepcopy(freeze)
        body=members['source/'+name]+b'\n';changed_members={**members,'source/'+name:body}
        changed['inference_code'][name]={'bytes':len(body),'sha256':c.digest(body)}
        with self.assertRaisesRegex(ValueError,'unchanged scientific source'):
            p.validate_receipt(raw,release,source_freeze=changed,source_members=changed_members)

    def test_projector_only_copies_verified_allowlist_and_no_private_fingerprints(self):
        release,_,raw=fixture();value=c.decode_json(raw)
        posterior={'schema':'k1-prepare-posterior-result-v1',
            **{key:value[key] for key in p.POSTERIOR_KEYS},
            'public_prerequisite_binding':{key:value[key] for key in p.BINDING_KEYS},
            'publisher_step_success':True,'recognition_success_established':False,'retry_authorized':False,
            'consumed_asr_attempts':None,'sealed_original18_consumed_asr_attempts':None,
            'sealed_r1_18_consumed_asr_attempts':None,'custody_sha256':'PRIVATE','expected_sha256':'PRIVATE',
            'observations':{'arbitrary':'PRIVATE'}}
        self.assertEqual(p.project_posterior(posterior),value)
        self.assertNotIn(b'PRIVATE',c.canonical(p.project_posterior(posterior)))
        for key,bad in [('outcome','partial_prepare_archive_verified'),('consumed_asr_attempts',0),
                        ('sealed_r1_18_consumed_asr_attempts',0),('evidence_kind','synthetic_zero_model_fixture')]:
            with self.subTest(key=key),self.assertRaises(ValueError):p.project_posterior({**posterior,key:bad})

    def test_archive_validation_rechecks_prerequisite_bytes_before_any_outcome_path(self):
        import hosted_run as host
        from test_asr18_host import public_fixture
        files,identity=public_fixture()
        self.assertTrue(host.validate_public_evidence(files,identity))
        key='source/'+host.NEW+'prepare-prerequisite.json'
        original=c.decode_json(files[key])
        for field,bad in [('outcome','partial_prepare_archive_verified'),('archive_sha256','9'*64),
                          ('diagnostic_forward_calls',None),('scientific_source_sha256','9'*64)]:
            changed=dict(files);value={**original,field:bad};changed[key]=c.canonical(value)
            with self.subTest(field=field),self.assertRaises(ValueError):host.validate_public_evidence(changed,identity)
        changed=dict(files);changed.pop(key)
        with self.assertRaises((ValueError,KeyError)):host.validate_public_evidence(changed,identity)

    def test_transport_and_observer_never_relabel_the_two_sealed_asr_identities(self):
        import evidence_transport as transport
        import failure_observer as observer
        for run,head in zip(c.SEALED_ASR_RUNS,c.SEALED_ASR_SOURCES):
            for identity in ({'repository':p.REPOSITORY,'run_id':run,'run_attempt':1,'job_key':'asr18','executed_head_sha':'a'*40},
                {'repository':p.REPOSITORY,'run_id':123,'run_attempt':1,'job_key':'asr18','executed_head_sha':head}):
                with self.subTest(identity=identity),self.assertRaises(ValueError):transport.identity(identity)
                with self.subTest(identity=identity),self.assertRaises(ValueError):observer._identity(identity)

    def test_real_host_rejects_unestablished_prerequisite_before_event_api_claim_or_setup(self):
        # Real copied source and production gate, no patched gate/global shortcut.
        with synthetic_enabled_source(HERE) as source,tempfile.TemporaryDirectory() as name:
            release=c.read_json(source/'execution-release.json');release['prepare_prerequisite_sha256']=None
            (source/'execution-release.json').write_bytes(c.canonical(release))
            program="""import json,pathlib,runpy,sys
seen=[];effects=[]
def profile(frame,event,arg):
 if event=='call':seen.append(frame.f_code.co_name)
def audit(event,args):
 if event.startswith('socket.') or event in ('subprocess.Popen','os.mkdir','os.system'):effects.append(event)
sys.addaudithook(audit);sys.setprofile(profile);sys.argv=[sys.argv[1]]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as error:code=error.code
finally:sys.setprofile(None)
print(json.dumps({'code':code,'seen':seen,'effects':effects}))
"""
            flags=['-'+'O'*sys.flags.optimize] if sys.flags.optimize else []
            result=subprocess.run([sys.executable,'-I','-B',*flags,'-c',program,str(source/'hosted_run.py')],
                cwd=name,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','RUNNER_TEMP':name,
                    'GITHUB_EVENT_PATH':'/must-not-read'},capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            observed=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(observed['code'],2);self.assertEqual(observed['effects'],[])
            self.assertTrue('release_first' in observed['seen'])
            self.assertFalse(set(observed['seen'])&{'event_identity','source_claim','setup_runtimes','execute'})
            self.assertEqual(list(Path(name).iterdir()),[])


if __name__=='__main__':unittest.main()
