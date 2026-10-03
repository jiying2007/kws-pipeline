import contextlib,io,hashlib,importlib.util,json,os,pathlib,subprocess,sys,tempfile,types,unittest,zipfile
from unittest.mock import patch
D=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import runtime_lock
spec=importlib.util.spec_from_file_location('installer',D/'install_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class GateTests(unittest.TestCase):
 def wheel(self,d,extra=None,meta=None):
  p=pathlib.Path(d)/'demo-1.0-py3-none-any.whl';metadata=meta or b'Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n'
  with zipfile.ZipFile(p,'w') as z:
   z.writestr('demo-1.0.dist-info/METADATA',metadata)
   if extra:z.writestr(*extra)
  return p
 def test_wheel_manifest_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.wheel(d);x={'name':'demo','version':'1.0','metadata_sha256':hashlib.sha256(b'Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n').hexdigest()};r=m.inspect_wheel(p,x);self.assertEqual(r['bytes'],p.stat().st_size);self.assertGreater(r['expanded_allocation_bytes'],0)
 def test_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.wheel(d,('../evil','bad'))
   with self.assertRaises(RuntimeError):m.inspect_wheel(p,{'name':'demo','version':'1.0'})
 def test_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   z=zipfile.ZipInfo('link');z.external_attr=0o120777<<16;p=self.wheel(d,(z,'target'))
   with self.assertRaises(AssertionError):m.inspect_wheel(p,{'name':'demo','version':'1.0'})
 def test_metadata_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.wheel(d)
   with self.assertRaises(AssertionError):m.inspect_wheel(p,{'name':'demo','version':'1.0','metadata_sha256':'0'*64})
 def test_qualifier_nonrunner_guard(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);(p/'lock.json').write_text('{}');env={**os.environ};env.pop('GITHUB_ACTIONS',None)
   q=subprocess.run([sys.executable,D/'qualify_runtime.py','--lock',p/'lock.json','--output',p/'out.json'],env=env,capture_output=True,text=True)
   self.assertEqual(q.returncode,1);r=json.loads((p/'out.json').read_text());self.assertEqual(r['status'],'failed');self.assertIn('Runner-only',r['error']);self.assertEqual(r['versions'],{})
 def test_optimized_interpreter_rejected_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)
   for script in ['install_runtime.py','qualify_runtime.py']:
    q=subprocess.run([sys.executable,'-O',D/script,'--help'],capture_output=True,text=True)
    self.assertNotEqual(q.returncode,0);self.assertIn('Optimized Python',q.stderr)
 def runner(self,d):
  i=m.Installer.__new__(m.Installer);i.a=types.SimpleNamespace(receipts=pathlib.Path(d));i.budget=lambda additional=0:None;i.env=dict(os.environ);i.events=[];return i
 def test_subprocess_inherits_controller_group(self):
  with tempfile.TemporaryDirectory() as d:
   i=self.runner(d);i.run([sys.executable,'-c','import os; print(os.getpgrp())'],5,'group.log');self.assertEqual(int((pathlib.Path(d)/'group.log').read_text()),os.getpgrp())
 def test_log_hard_cap(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'MAX_LOG_BYTES',1024):
   i=self.runner(d)
   with self.assertRaises(RuntimeError):i.run([sys.executable,'-c',"print('x'*4096)"],5,'big.log')
   self.assertLessEqual((pathlib.Path(d)/'big.log').stat().st_size,1024)
 def test_aggregate_log_hard_cap(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'MAX_RECEIPT_BYTES',2048),patch.object(m,'RECEIPT_FINISH_RESERVE',512):
   (pathlib.Path(d)/'existing.log').write_bytes(b'a'*1024);i=self.runner(d)
   with self.assertRaises(RuntimeError):i.run([sys.executable,'-c',"print('x'*4096)"],5,'big.log')
   self.assertLessEqual(i.receipt_bytes(),1536)
 def test_source_receipt_and_member_manifest_consistency(self):
  receipts=json.loads((D/'sdist-review/sdist-receipts.json').read_text());members={x['name']:x for x in json.loads((D/'sdist-review/source-member-manifests.json').read_text())};lock={x['name']:x for x in runtime_lock.load()['packages']}
  self.assertEqual(set(members),{'pyworld','openai-whisper','wget','antlr4-python3-runtime'})
  for r in receipts:
   x=lock[r['name']];self.assertEqual((r['version'],r['url'],r['sha256'],r['bytes']),(x['version'],x['url'],x['sha256'],x['size']));row=members[r['name']];self.assertEqual(row['archive_sha256'],r['sha256']);self.assertEqual(row['total_regular_bytes'],sum(e['bytes'] for e in row['members']))
   for e in row['members']:m.safe_member(e['path']);self.assertRegex(e['sha256'],r'^[0-9a-f]{64}$')
 def test_failed_command_echoes_only_bounded_current_log_tail(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'MAX_FAILURE_TAIL_BYTES',128):
   (pathlib.Path(d)/'unrelated.log').write_text('never-publish-other-logs')
   i=self.runner(d);outer=io.StringIO()
   with contextlib.redirect_stdout(outer),self.assertRaisesRegex(RuntimeError,'failed with 7'):
    i.run([sys.executable,'-c',"import os; os.write(1,b'old-prefix:'+b'x'*4096+b'compiler diagnostic\\n'); raise SystemExit(7)"],5,'compile.log')
   output=outer.getvalue();lines=output.splitlines();meta=json.loads(lines[0])
   self.assertEqual(meta['event'],'failed_command_log_tail');self.assertEqual(meta['tail_bytes'],128)
   self.assertIn('compiler diagnostic',output);self.assertNotIn('old-prefix:',output);self.assertNotIn('never-publish-other-logs',output)
   self.assertTrue((pathlib.Path(d)/'compile.log').read_bytes().startswith(b'old-prefix:'))
 def test_successful_command_does_not_echo_log(self):
  with tempfile.TemporaryDirectory() as d:
   i=self.runner(d);outer=io.StringIO()
   with contextlib.redirect_stdout(outer):i.run([sys.executable,'-c',"print('normal install')"],5,'install.log')
   self.assertEqual(outer.getvalue(),'');self.assertEqual((pathlib.Path(d)/'install.log').read_text(),'normal install\n');self.assertEqual(i.events[-1]['returncode'],0)
if __name__=='__main__':unittest.main()
