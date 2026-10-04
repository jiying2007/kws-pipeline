#!/usr/bin/env python3
"""Static/stdlib checks only; never install, download or construct a model."""
import ast,hashlib,json,os,pathlib,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import admit_run,pilot_common,runtime_lock

def main():
 for path in sorted(HERE.glob('*.py')):ast.parse(path.read_text(),filename=path.name)
 cfg=pilot_common.load_config();lock=runtime_lock.load()
 assert lock['status']=='input_lock_source_reviewed_pending_runtime_qualification' and lock['all_active_edges_satisfied']
 assert len(lock['packages'])==len({x['name'] for x in lock['packages']})
 assert lock['compressed_total_bytes']==sum(x['size'] for x in lock['packages'])==3007556262
 assert all(row['url'].startswith('https://files.pythonhosted.org/') and len(row['sha256'])==64 and row['size']>0 for row in lock['packages'])
 assert all(x['satisfied'] for x in lock['requirement_checks'])
 for path in sorted(HERE.glob('test_*.py')):
  subprocess.run([sys.executable,'-I','-S',str(path)],check=True,timeout=120)
 workflow=(HERE.parents[1]/admit_run.WORKFLOW).read_text()
 for value in ['group: research-cosy30-cross-voice-v1','cancel-in-progress: false','runs-on: ubuntu-24.04','retention-days: 1','github.run_attempt == 1','timeout-minutes: 60']:
  assert value in workflow,value
 for forbidden in ['workflow_dispatch:', 'pull_request_target:', 'schedule:', 'actions/cache','runs-on: self-hosted','ubuntu-latest','overwrite: true']:
  assert forbidden not in workflow,forbidden
 payload=admit_run.identity(HERE.parents[1]);arm=json.loads((HERE/'arm.json').read_text())
 armed=admit_run.check_activation(arm,payload)
 if os.environ.get('GITHUB_OUTPUT'):
  with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('armed='+str(armed).lower()+'\n')
 print(json.dumps({'status':'passed','package_count':len(lock['packages']),'source_payload_sha256':payload,'armed':armed,'generation_calls':0,'network':False,'model_load':False},sort_keys=True))
if __name__=='__main__':main()
