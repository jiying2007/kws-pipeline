import importlib.util,json,pathlib,tempfile,unittest,sys,hashlib,struct
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manifest',ROOT/'manifest.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
E=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence');sys.argv=sys.argv[:1]
class ManifestTests(unittest.TestCase):
 def test_original(self):self.assertEqual(len(m.load(E/'weights-manifest.json',E/'donor-local.f32')),3027732)
 def test_rejections(self):
  original=json.loads((E/'weights-manifest.json').read_text())
  changes=[lambda x:x.update(version=True),lambda x:x.update(source_sha256='0'*64),lambda x:x['tensors'].pop(),lambda x:x['tensors'][1].update(offset=0),lambda x:x['tensors'][0].update(shape=[True]),lambda x:x['tensors'][0].update(sha256='0'*64),lambda x:x.update(payload_bytes=True)]
  for change in changes:
   with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
    x=json.loads(json.dumps(original));change(x);p=pathlib.Path(d)/'manifest.json';p.write_text(json.dumps(x))
    with self.assertRaises(ValueError):m.load(p,E/'donor-local.f32')
 def test_rebound_finite_weight(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);raw=bytearray((E/'donor-local.f32').read_bytes());raw[3200:3204]=struct.pack('<f',0.125)
   x=json.loads((E/'weights-manifest.json').read_text());x['payload_sha256']=hashlib.sha256(raw).hexdigest()
   t=x['tensors'][2];t['sha256']=hashlib.sha256(raw[t['offset']:t['offset']+t['bytes']]).hexdigest()
   (d/'data').write_bytes(raw);(d/'manifest').write_text(json.dumps(x))
   with self.assertRaisesRegex(ValueError,'canonical'):m.load(d/'manifest',d/'data')
if __name__=='__main__':unittest.main()
