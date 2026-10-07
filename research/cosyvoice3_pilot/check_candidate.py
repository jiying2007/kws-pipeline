#!/usr/bin/env python3
"""No installs/imports/network: validate frozen public files and synthetic tests."""
import ast,hashlib,json,pathlib,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runtime_lock

def main():
 for path in sorted(HERE.glob('*.py')):ast.parse(path.read_text(),filename=path.name)
 lock=runtime_lock.load()
 assert lock['status']=='input_lock_source_reviewed_pending_runtime_qualification' and lock['all_active_edges_satisfied']
 assert len(lock['packages'])==len({x['name'] for x in lock['packages']})
 assert lock['compressed_total_bytes']==sum(x['size'] for x in lock['packages'])
 for row in lock['packages']:
  assert row['url'].startswith('https://files.pythonhosted.org/') and len(row['sha256'])==64 and row['size']>0
 assert all(x['satisfied'] for x in lock['requirement_checks'])
 for path in sorted(HERE.glob('test_*.py')):
  subprocess.run([sys.executable,'-I','-S',str(path)],check=True,timeout=120)
 workflow=(HERE.parents[1]/'.github/workflows/research-cosyvoice3-pilot.yml').read_text()
 assert 'group: research-cosyvoice3-pilot-v1' in workflow and 'cancel-in-progress: false' in workflow
 for forbidden in ['workflow_dispatch:', 'pull_request_target:', 'schedule:', 'actions/cache','runs-on: self-hosted','ubuntu-latest','overwrite: true']:
  assert forbidden not in workflow,forbidden
 assert 'runs-on: ubuntu-24.04' in workflow and 'retention-days: 1' in workflow and 'github.run_attempt == 1' in workflow
 print(json.dumps({'status':'passed','package_count':len(lock['packages']),'network':False,'model_load':False,'inference':False},sort_keys=True))
if __name__=='__main__':main()
