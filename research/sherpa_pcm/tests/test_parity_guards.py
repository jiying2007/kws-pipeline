"""Model-free negative tests; never loads a shared library."""
import json,pathlib,tempfile,types,unittest
from unittest.mock import patch
import parity_inputs as g
class Guards(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.r=pathlib.Path(self.tmp.name);self.src=self.r/'src';self.src.mkdir();self.data=self.r/'data';self.data.mkdir();(self.data/'x.wav').write_bytes(b'fixture')
  runtime=self.r/'runtime';runtime.write_bytes(b'fake dependency, never loaded');lock={'host_runtime':{'runtime_libraries':[{'filename':'runtime','sha256':g.sha(runtime)}]}}
  for name in g.SOURCE_MEMBERS:
   p=self.src/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(lock) if name=='dependencies.lock.json' else name)
  self.ref=self.r/'ref.json';self.rows=[dict(recording=str(i),path='x.wav',file_sha256=g.sha(self.data/'x.wav')) for i in range(42)];self.ref.write_text(json.dumps({'recordings':self.rows}))
  lib=self.r/'fake.so';lib.write_bytes(b'fake library, never loaded');self.receipt={'stub_only':False,'files':{n:g.sha(self.src/n) for n in g.SOURCE_MEMBERS},'library_sha256':g.sha(lib),'runtime_libraries':[dict(filename='runtime',path=str(runtime),sha256=g.sha(runtime))]};self.receipt_path=self.r/'build.json';self.write_receipt()
  self.args=types.SimpleNamespace(output=self.r/'result.json',reference=self.ref,build_receipt=self.receipt_path,library=lib,data_root=self.data)
 def write_receipt(self):self.receipt_path.write_text(json.dumps(self.receipt))
 def check(self):
  with patch.object(g,'REFERENCE_SHA256',g.sha(self.ref)):return g.validate_inputs(self.args,self.src)
 def change_path(self,path):self.rows[0]['path']=path;self.ref.write_text(json.dumps({'recordings':self.rows}))
 def test_valid(self):self.assertEqual(len(self.check()['recordings']),42)
 def test_wrong_reference(self):
  with self.assertRaisesRegex(ValueError,'reference digest'):g.validate_inputs(self.args,self.src)
 def test_absolute(self):
  self.change_path(str(self.data/'x.wav'))
  with self.assertRaisesRegex(ValueError,'relative'):self.check()
 def test_traversal(self):
  self.change_path('../data/x.wav')
  with self.assertRaisesRegex(ValueError,'relative'):self.check()
 def test_symlink(self):
  (self.data/'link.wav').symlink_to(self.data/'x.wav');self.change_path('link.wav')
  with self.assertRaisesRegex(ValueError,'symlink'):self.check()
 def test_output_overwrite(self):
  self.args.output.write_text('preserve')
  with self.assertRaisesRegex(ValueError,'overwrite'):self.check()
  self.assertEqual(self.args.output.read_text(),'preserve')
 def test_stub(self):
  self.receipt['stub_only']=True;self.write_receipt()
  with self.assertRaisesRegex(ValueError,'non-stub'):self.check()
 def test_library_changed(self):
  self.args.library.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'library digest'):self.check()
 def test_source_changed(self):
  (self.src/'pcm_kws.cc').write_text('changed')
  with self.assertRaisesRegex(ValueError,'source digest'):self.check()
 def test_runtime_changed(self):
  (self.r/'runtime').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'dependency digest'):self.check()
if __name__=='__main__':unittest.main()
