"""Model-free lossless multipart transport and fail-closed assembly tests."""
import hashlib,json,pathlib,shutil,tempfile,unittest
from unpack_goldens import assemble_archive,unpack_archive,ARCHIVE_SHA256,ARCHIVE_BYTES
SOURCE=pathlib.Path(__file__).resolve().parents[1]/'fixtures'
class Parts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=pathlib.Path(self.temp.name);self.fixture=self.root/'fixture';shutil.copytree(SOURCE,self.fixture)
 def rebind_manifest(self,edit):
  p=self.fixture/'parts-manifest.json';m=json.loads(p.read_text());edit(m);p.write_text(json.dumps(m));i=self.fixture/'identity.json';identity=json.loads(i.read_text());identity['parts_manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();i.write_text(json.dumps(identity))
 def test_exact_original_archive(self):
  raw=assemble_archive(self.fixture);self.assertEqual(len(raw),ARCHIVE_BYTES);self.assertEqual(hashlib.sha256(raw).hexdigest(),ARCHIVE_SHA256);dest=unpack_archive(raw,self.root/'out');self.assertTrue((dest/'manifest.json').is_file())
 def test_wrong_part_bytes(self):
  p=self.fixture/'parts/000.bin';b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b)
  with self.assertRaisesRegex(ValueError,'digest'):assemble_archive(self.fixture)
 def test_missing(self):
  (self.fixture/'parts/001.bin').unlink()
  with self.assertRaisesRegex(ValueError,'coverage'):assemble_archive(self.fixture)
 def test_extra(self):
  (self.fixture/'parts/extra.bin').write_bytes(b'extra')
  with self.assertRaisesRegex(ValueError,'coverage'):assemble_archive(self.fixture)
 def test_reordered_rebound_manifest(self):
  self.rebind_manifest(lambda m:m['parts'].reverse())
  with self.assertRaisesRegex(ValueError,'order'):assemble_archive(self.fixture)
 def test_traversal_rebound_manifest(self):
  self.rebind_manifest(lambda m:m['parts'][0].update(path='../000.bin'))
  with self.assertRaisesRegex(ValueError,'path'):assemble_archive(self.fixture)
 def test_wrong_size_rebound_manifest(self):
  self.rebind_manifest(lambda m:m['parts'][0].update(bytes=49153))
  with self.assertRaisesRegex(ValueError,'size'):assemble_archive(self.fixture)
 def test_symlink_member(self):
  p=self.fixture/'parts/000.bin';original=self.root/'original.bin';p.rename(original);p.symlink_to(original)
  with self.assertRaisesRegex(ValueError,'file'):assemble_archive(self.fixture)
 def test_rebound_part_still_fails_whole_digest(self):
  p=self.fixture/'parts/000.bin';b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b);self.rebind_manifest(lambda m:m['parts'][0].update(sha256=hashlib.sha256(b).hexdigest()))
  with self.assertRaisesRegex(ValueError,'complete archive'):assemble_archive(self.fixture)
 def test_refuse_output_overwrite(self):
  out=self.root/'out';out.mkdir();(out/'keep').write_text('keep')
  with self.assertRaises(FileExistsError):unpack_archive(assemble_archive(self.fixture),out)
  self.assertEqual((out/'keep').read_text(),'keep')
if __name__=='__main__':unittest.main()
