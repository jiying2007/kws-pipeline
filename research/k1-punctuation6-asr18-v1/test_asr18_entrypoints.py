"""Actual isolated command and shell integration; no model/Docker/network execution."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from test_control_fixture import APPROVALS, DISABLED_APPROVALS, synthetic_enabled_source
import campaign_contract as c
OPT=['-'+'O'*sys.flags.optimize] if sys.flags.optimize else []
SENTINELS={key:'SYNTHETIC_'+key+'_NOT_A_CREDENTIAL' for key in
    ('GITHUB_TOKEN','GH_TOKEN','KWS_EVIDENCE_TOKEN','ACTIONS_RUNTIME_TOKEN','ACTIONS_ID_TOKEN_REQUEST_TOKEN')}


@contextmanager
def disabled_source():
    """Explicit test-only control copy; never change the actual release bytes."""
    with tempfile.TemporaryDirectory(prefix='k1-explicit-disabled-') as name:
        target=Path(name)/'research/k1-punctuation6-asr18-v1'
        shutil.copytree(HERE,target)
        for filename,keys in DISABLED_APPROVALS.items():
            value=json.loads((target/filename).read_bytes())
            for key in keys:value[key]=False
            (target/filename).write_bytes(c.canonical(value))
        release=c.read_json(target/'execution-release.json')
        release['claim_admission_control_sha256']=c.digest((target/c.CLAIM_CONTROL_FILE).read_bytes())
        (target/'execution-release.json').write_bytes(c.canonical(release))
        yield target


class ActualEntrypoints(unittest.TestCase):
    def check_diagnostic(self,raw,*,role,stages,terminal=False,validation_code=None):
        def canonical(value):
            return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        lines=raw.splitlines()
        if terminal:
            self.assertEqual(lines.pop(),b'{"status":"blocked_no_retry"}')
        self.assertEqual(len(lines),len(stages)+1)
        for sequence,(line,stage) in enumerate(zip(lines[:-1],stages),1):
            row=json.loads(line)
            self.assertEqual(row,{'schema':'k1-asr18-safe-checkpoint-v1','role':role,
                'stage':stage,'sequence':sequence,'identity':None,'native_calls':'not_asserted'})
            self.assertEqual(line,canonical(row));self.assertLessEqual(len(line),1024)
        envelope=json.loads(lines[-1]);record=envelope['record']
        self.assertEqual(set(envelope),{'schema','record','record_bytes','record_sha256','persistence'})
        self.assertEqual(envelope['schema'],'k1-asr18-independent-process-diagnostic-envelope-v1')
        self.assertEqual(lines[-1],canonical(envelope))
        self.assertEqual(envelope['record_bytes'],len(canonical(record)))
        self.assertEqual(envelope['record_sha256'],hashlib.sha256(canonical(record)).hexdigest())
        self.assertLessEqual(envelope['record_bytes'],16384)
        self.assertLessEqual(len(lines[-1]),17408)
        self.assertEqual(envelope['persistence'],{'status':'unconfigured','error':None})
        self.assertEqual(set(record),{'schema','role','identity','stage','status','errors','attachment_error',
            'checkpoint_output_error','checkpoint_limit_reached','errors_truncated','claims','checkpoint_count',
            'original_writer_receipt_included','native_success_established','retry_authorized','limit'})
        self.assertEqual(record['limit'],'ordinary_exception_observation; abrupt_process_or_runner_loss_may_prevent_final_record')
        self.assertEqual(record['schema'],'k1-asr18-independent-process-diagnostic-v1')
        self.assertEqual(record['role'],role);self.assertEqual(record['stage'],stages[-1])
        self.assertEqual(record['status'],'failed_or_unconfirmed');self.assertIsNone(record['identity'])
        self.assertEqual(record['checkpoint_count'],len(stages))
        for key in ('native_success_established','retry_authorized','original_writer_receipt_included',
                    'checkpoint_limit_reached','errors_truncated'):self.assertIs(record[key],False)
        for key in ('attachment_error','checkpoint_output_error'):self.assertIsNone(record[key])
        self.assertEqual(record['claims'],{'scope':'fixed_new_campaign_local_observations_only',
            'snapshot_atomic':False,'consumed_attempts':None,'consumed_attempts_authority':'unknown',
            'source_claim':{'status':'unobserved','bytes':None,'sha256':None},
            'terminal_ledger':{'status':'unobserved','bytes':None,'sha256':None},
            'reported_terminal_attempts':None,'reported_terminal_accounting_valid':None,
            'observed_slot_directories':None,'validated_start_files_observed':None,'slots':[]})
        self.assertEqual(len(record['errors']),1)
        error=record['errors'][0]
        self.assertEqual(set(error),{'stage','detail','explanation'})
        self.assertEqual(error['stage'],stages[-1])
        self.assertEqual(error['explanation'],'operation_failed_at_fixed_stage; exception_content_omitted')
        expected_detail={'error_type':'UnlistedError' if stages==['publication_gate'] else 'ValueError',
            'error_text':'omitted_by_fixed_privacy_policy'}
        if validation_code is not None:expected_detail['validation_code']=validation_code
        self.assertEqual(error['detail'],expected_detail)

    def test_checked_in_asr_controls_are_explicitly_authorized_for_current_scope(self):
        for filename,keys in APPROVALS.items():
            value=json.loads((HERE/filename).read_bytes())
            self.assertTrue(all(value[key] is True for key in keys))
        release=json.loads((HERE/'execution-release.json').read_bytes())
        control_raw=(HERE/'claim-admission-control.json').read_bytes();control=json.loads(control_raw)
        self.assertEqual(release['schema'],'k1-asr18-execution-release-v2')
        self.assertEqual(release['claim_admission_mode'],'git-owner-v2')
        self.assertEqual(release['claim_admission_control_sha256'],hashlib.sha256(control_raw).hexdigest())
        self.assertEqual(control['schema'],'k1-asr18-claim-admission-control-v2')
        self.assertEqual(control['mode'],'git-owner-v2')
        for key in c.CLAIM_APPROVALS:self.assertIs(control[key],True)
        self.assertEqual(c.require_current_scope(control),c.current_scope())
        self.assertIsNone(control['workflow_id'])
        self.assertEqual(control['workflow_id_source'],'actual_run_api_plus_path')
        self.assertIs(release['historical_screening_reviewed'],True)

    def call(self,path,args=()):
        with tempfile.TemporaryDirectory(prefix='k1-entry-negative-') as name:
            temp=Path(name)
            env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','RUNNER_TEMP':name,
                 'GITHUB_EVENT_PATH':'/nonexistent/event',**SENTINELS}
            result=subprocess.run([sys.executable,'-I','-B',*OPT,str(path),*args],cwd=temp,env=env,
                capture_output=True,timeout=20)
            self.assertEqual(list(temp.iterdir()),[])
            for value in SENTINELS.values():self.assertNotIn(value.encode(),result.stdout+result.stderr)
            self.assertNotIn(b'ModuleNotFoundError',result.stderr)
            self.assertNotIn(b'ImportError',result.stderr)
            return result

    def test_real_workflow_host_command_hits_explicit_disabled_entry(self):
        with disabled_source() as source:
            value=self.call(source/'hosted_run.py')
        self.assertEqual((value.returncode,value.stderr),(2,b''))
        self.check_diagnostic(value.stdout,role='host',stages=['release'],terminal=True,validation_code='claim_admission_closed')

    def test_real_workflow_publisher_command_hits_explicit_disabled_gate(self):
        with disabled_source() as source:
            value=self.call(source/'evidence_publisher.py')
        self.assertEqual((value.returncode,value.stderr),(1,b'ASR18 publication disabled or review absent.\n'))
        self.check_diagnostic(value.stdout,role='publisher',stages=['publication_gate'])

    def test_actual_authorized_workflow_commands_reject_missing_context_without_effects(self):
        self.check_enabled_workflow_commands_reject_nonexecution_context(HERE)

    def test_synthetic_enabled_workflow_commands_reject_nonexecution_context_without_effects(self):
        with synthetic_enabled_source(HERE) as source:
            self.check_enabled_workflow_commands_reject_nonexecution_context(source)

    def check_enabled_workflow_commands_reject_nonexecution_context(self,source):
        # No gate patch, import-path injection or network blocker in the child.
        program='''import hashlib,json,os,pathlib,runpy,sys
path=pathlib.Path(sys.argv[1]);seen=[];effects=[]
def profile(frame,event,arg):
 if event in ('call','return') and frame.f_code.co_name in ('release_first','host_environment','early_environment','event_identity','publication_gate','bound_release','require_execution_release','_source_git','execute','run_once'):
  seen.append([event,frame.f_code.co_name,frame.f_code.co_filename])
def audit(event,args):
 if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.mkdir'):
  effects.append(event)
sys.addaudithook(audit);sys.setprofile(profile);sys.argv=[str(path)]
try:runpy.run_path(str(path),run_name='__main__')
except SystemExit as e:code=e.code
finally:sys.setprofile(None)
print(json.dumps({'code':code,'seen':seen,'effects':effects}))
'''
        cases=(('hosted_run.py',SENTINELS,'claim_host_credentials_rejected'),
               ('hosted_run.py',{},'claim_ordinal_attempt_rejected'),
               ('evidence_publisher.py',SENTINELS,None))
        for filename,credentials,host_code in cases:
            with self.subTest(filename=filename,credentials=bool(credentials)),tempfile.TemporaryDirectory(prefix='k1-enabled-cli-') as name:
                temp=Path(name);event=temp/'event.json';event.write_text('{}')
                expected=hashlib.sha256((source/'execution-freeze.json').read_bytes()).hexdigest()
                env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','RUNNER_TEMP':name,
                    'GITHUB_EVENT_PATH':str(event),'GITHUB_ACTIONS':'false','KWS_SOURCE_FREEZE_SHA256':expected,**credentials}
                value=subprocess.run([sys.executable,'-I','-B',*OPT,'-c',program,str(source/filename)],
                    cwd=temp,env=env,capture_output=True,timeout=20)
                self.assertEqual(value.returncode,0,value.stderr.decode())
                chunks=value.stdout.splitlines(keepends=True)
                observation=json.loads(chunks[-1])
                self.check_diagnostic(b''.join(chunks[:-1]),
                    role='host' if filename=='hosted_run.py' else 'publisher',
                    stages=['release','arguments'] if filename=='hosted_run.py' else
                           ['publication_gate','arguments','event_read','source_release'],
                    terminal=filename=='hosted_run.py',
                    validation_code=host_code)
                self.assertEqual(observation['effects'],[])
                self.assertEqual(sorted(x.name for x in temp.iterdir()),['event.json'])
                calls={row[1] for row in observation['seen'] if row[0]=='call'}
                self.assertFalse(calls & {'_source_git','execute','run_once'})
                if filename=='hosted_run.py':
                    self.assertEqual(observation['code'],2)
                    self.assertTrue({'release_first','host_environment'}<=calls)
                    self.assertEqual('early_environment' in calls,not bool(credentials))
                    self.assertNotIn('event_identity',calls)
                else:
                    self.assertEqual(observation['code'],'ASR18 publication blocked or unconfirmed; no retry.')
                    self.assertTrue({'publication_gate','bound_release','require_execution_release'}<=calls)
                    self.assertIn(['return','require_execution_release',str(source/'campaign_contract.py')],observation['seen'])
                for secret in SENTINELS.values():self.assertNotIn(secret.encode(),value.stdout+value.stderr)

    def test_real_claim_issuer_rejects_missing_context_before_client_or_private_directory(self):
        value=self.call(HERE/'claim_admission.py')
        self.assertEqual((value.returncode,value.stdout,value.stderr),(2,b'{"ready":false}\n',b''))

    def test_real_native_family_mode_commands_import_and_reject(self):
        for family in ('qwen06','sensevoice','whisper'):
            for mode in ('prepare','decode'):
                with self.subTest(family=family,mode=mode):
                    value=self.call(HERE/'native_worker.py',(family,mode))
                    self.assertEqual((value.returncode,value.stdout,value.stderr),(1,b'',b''))

    def test_passive_trace_proves_native_release_gate_not_caught_importerror(self):
        # Same -I import conditions: no source-directory injection in this wrapper.
        # The real worker's fixed _local bootstrap must reach the real contract gate.
        program='''import json,pathlib,runpy,sys
path=pathlib.Path(sys.argv[1]);seen=[]
def profile(frame,event,arg):
 if event=='call' and frame.f_code.co_name in ('_production_gate','require_execution_release'):
  seen.append([frame.f_code.co_name,frame.f_code.co_filename])
sys.setprofile(profile);sys.argv=[str(path),'qwen06','decode']
try:runpy.run_path(str(path),run_name='__main__')
except SystemExit as e:code=e.code
finally:sys.setprofile(None)
if code!=1:raise SystemExit(2)
print(json.dumps(seen))
'''
        with disabled_source() as source:
            value=subprocess.run([sys.executable,'-I','-B',*OPT,'-c',program,str(source/'native_worker.py')],
                capture_output=True,timeout=20,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'})
            self.assertEqual(value.returncode,0,value.stderr.decode())
            seen=json.loads(value.stdout)
            self.assertIn(['_production_gate',str(source/'native_worker.py')],seen)
            self.assertIn(['require_execution_release',str(source/'campaign_contract.py')],seen)

    def test_real_setup_entry_rejects_wrong_mount_before_scope_or_acquisition(self):
        for stage in ('setup','verify'):
            value=self.call(HERE/'setup_adapter.py',('--profile','asr','--stage',stage,'--runtime','/not-runtime'))
            self.assertNotEqual(value.returncode,0)
            self.assertIn(b'ValueError: Exact /runtime mount required',value.stderr)

    def test_retained_setup_child_bootstrap_reaches_same_mount_gate(self):
        # Extract the original locked literal without modifying its bytes.
        tree=ast.parse((HERE/'setup_adapter.py').read_bytes())
        expression=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
        bootstrap=ast.literal_eval(expression)
        value=subprocess.run([sys.executable,'-I','-B',*OPT,'-c',bootstrap,str(HERE/'setup_adapter.py'),
            'asr','/not-runtime','preflight'],capture_output=True,timeout=20,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'})
        self.assertNotEqual(value.returncode,0)
        self.assertIn(b'ValueError: Exact /runtime mount required',value.stderr)
        self.assertNotIn(b'ModuleNotFoundError',value.stderr)

    def test_cpu_handshake_without_command_never_starts_child(self):
        value=self.call(HERE/'cpu_gate.py')
        self.assertEqual((value.returncode,value.stdout,value.stderr),(2,b'',b''))

    def test_retained_whisper_isolated_cli_imports_without_runtime(self):
        base=ROOT/'research/source-screen32-whisper-v1/attempt-3'
        for name in ('bounded_connect.py','pip_acquire.py'):
            with self.subTest(name=name):
                value=self.call(base/name,('--help',))
                self.assertEqual(value.returncode,0,value.stderr.decode())
                self.assertIn(b'usage:',value.stdout)
        value=self.call(base/'phased_setup.py')
        self.assertEqual((value.returncode,value.stderr),(1,b''))
        self.assertEqual(ast.literal_eval(value.stdout.decode()),{'status':'failed_no_retry','reason_code':'entry_contract'})

    def test_exact_workflow_shell_gives_git_and_hash_tools_safe_env(self):
        text=(HERE/'workflow.yml').read_text()
        line=next(line.strip() for line in text.splitlines() if line.strip().startswith('export KWS_SOURCE_FREEZE_SHA256='))
        self.assertIn('env -i PATH=/usr/bin:/bin HOME=/nonexistent',line)
        self.assertIn('GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null',line)
        self.assertIn('core.fsmonitor=false',line)
        self.assertIn('core.hooksPath=/dev/null',line)
        with tempfile.TemporaryDirectory(prefix='k1-real-shell-') as name:
            root=Path(name);fake=root/'git';record=root/'git-observation.json'
            # Only Git itself is replaced; execute the exact reviewed real shell
            # env-cleaning, quoting, hash pipeline and export syntax unchanged.
            fake.write_text('#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\nPath('+repr(str(record))+').write_text(json.dumps({"env":dict(os.environ),"argv":sys.argv[1:]}))\nsys.stdout.buffer.write(b"synthetic committed freeze bytes")\n')
            fake.chmod(0o700)
            shell=line.replace('git --no-optional-locks',str(fake)+' --no-optional-locks')+'\nprintf "%s" "$KWS_SOURCE_FREEZE_SHA256"\n'
            value=subprocess.run(['/bin/bash','-e','-c',shell],capture_output=True,timeout=20,
                env={'PATH':'/usr/bin:/bin','HOME':'/sensitive/host-home','GITHUB_SHA':'a'*40,**SENTINELS})
            self.assertEqual(value.returncode,0,value.stderr.decode())
            self.assertEqual(value.stdout.decode(),hashlib.sha256(b'synthetic committed freeze bytes').hexdigest())
            observed=json.loads(record.read_bytes())
            for key in SENTINELS:self.assertNotIn(key,observed['env'])
            self.assertEqual(observed['env']['HOME'],'/nonexistent')
            self.assertEqual(observed['env']['GIT_CONFIG_GLOBAL'],'/dev/null')
            self.assertEqual(observed['env']['GIT_CONFIG_NOSYSTEM'],'1')
            self.assertEqual(observed['argv'],['--no-optional-locks','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null','show','a'*40+':research/k1-punctuation6-asr18-v1/execution-freeze.json'])
            for secret in SENTINELS.values():self.assertNotIn(secret,(value.stdout+value.stderr).decode())


if __name__=='__main__':unittest.main()
