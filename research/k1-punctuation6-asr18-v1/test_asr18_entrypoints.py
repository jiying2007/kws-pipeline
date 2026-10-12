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
OPT=['-'+'O'*sys.flags.optimize] if sys.flags.optimize else []
SENTINELS={key:'SYNTHETIC_'+key+'_NOT_A_CREDENTIAL' for key in
    ('GITHUB_TOKEN','GH_TOKEN','KWS_EVIDENCE_TOKEN','ACTIONS_RUNTIME_TOKEN','ACTIONS_ID_TOKEN_REQUEST_TOKEN')}


@contextmanager
def disabled_source():
    """Explicit test-only control copy; never change the actual release bytes."""
    with tempfile.TemporaryDirectory(prefix='k1-explicit-disabled-') as name:
        target=Path(name)/'research/k1-punctuation6-asr18-v1'
        shutil.copytree(HERE,target)
        for filename,keys in (
            ('execution-release.json',('enabled','historical_screening_reviewed','independent_source_review','root_execution_release')),
            ('publication-release.json',('enabled','independent_source_review','root_publication_release'))):
            value=json.loads((target/filename).read_bytes())
            for key in keys:value[key]=False
            (target/filename).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False))
        yield target


class ActualEntrypoints(unittest.TestCase):
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

    def test_real_workflow_host_command_hits_disabled_entry(self):
        with disabled_source() as source:value=self.call(source/'hosted_run.py')
        self.assertEqual((value.returncode,value.stdout,value.stderr),(2,b'{"status":"blocked_no_retry"}\n',b''))

    def test_real_workflow_publisher_command_hits_explicit_disabled_gate(self):
        with disabled_source() as source:value=self.call(source/'evidence_publisher.py')
        self.assertEqual((value.returncode,value.stdout,value.stderr),(1,b'',b'ASR18 publication disabled or review absent.\n'))


    def test_actual_enabled_workflow_commands_reject_nonexecution_context_without_effects(self):
        # Observe real production gates and Python audit events. No source-path
        # injection, gate patch, network blocker, or substituted release value.
        program='''import hashlib,json,os,pathlib,runpy,sys
path=pathlib.Path(sys.argv[1]);seen=[];effects=[]
def profile(frame,event,arg):
 if event in ('call','return') and frame.f_code.co_name in ('release_first','event_identity','publication_gate','bound_release','require_execution_release','_source_git','execute','run_once'):
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
        for filename in ('hosted_run.py','evidence_publisher.py'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory(prefix='k1-enabled-cli-') as name:
                temp=Path(name);event=temp/'event.json';event.write_text('{}')
                expected=hashlib.sha256((HERE/'execution-freeze.json').read_bytes()).hexdigest()
                env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','RUNNER_TEMP':name,
                    'GITHUB_EVENT_PATH':str(event),'GITHUB_ACTIONS':'false','KWS_SOURCE_FREEZE_SHA256':expected,**SENTINELS}
                value=subprocess.run([sys.executable,'-I','-B',*OPT,'-c',program,str(HERE/filename)],
                    cwd=temp,env=env,capture_output=True,timeout=20)
                self.assertEqual(value.returncode,0,value.stderr.decode())
                observation=json.loads(value.stdout.decode().splitlines()[-1])
                self.assertEqual(observation['effects'],[])
                self.assertEqual(sorted(x.name for x in temp.iterdir()),['event.json'])
                calls={row[1] for row in observation['seen'] if row[0]=='call'}
                self.assertFalse(calls & {'_source_git','execute','run_once'})
                if filename=='hosted_run.py':
                    self.assertEqual(observation['code'],2)
                    self.assertTrue({'release_first','event_identity'}<=calls)
                else:
                    self.assertEqual(observation['code'],'ASR18 publication blocked or unconfirmed; no retry.')
                    self.assertTrue({'publication_gate','bound_release','require_execution_release'}<=calls)
                    self.assertIn(['return','require_execution_release',str(HERE/'campaign_contract.py')],observation['seen'])
                for secret in SENTINELS.values():self.assertNotIn(secret.encode(),value.stdout+value.stderr)

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
