"""No model imports; stdlib adversarial tests survive -O / -OO."""
import copy
import io
import os
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import zipfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_registry as b
REQUIRE_LIVE="--require-live" in sys.argv
if REQUIRE_LIVE:sys.argv.remove("--require-live")

HERE=Path(__file__).resolve().parent
class RegistryTests(unittest.TestCase):
 def test_saved_registry(self): b.validate(b.decode((HERE/'candidates.json').read_bytes()))
 def test_reject_escalation_and_misbindings(self):
  base=b.decode((HERE/'candidates.json').read_bytes())
  changes=[('training_admitted',True),('human_gold',True),('evaluation_gold',True),('source_trust_root_admitted',True),('speaker_id','Vivian'),('same_pcm_asr_binding_verified',True)]
  for key,value in changes:
   d=copy.deepcopy(base);d['records'][0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError): b.validate(d)
  for mutation in ('duplicate','role','wav_as_pcm','asr_pcm'):
   d=copy.deepcopy(base);r=d['records'][0]
   if mutation=='duplicate': d['records'][1]=copy.deepcopy(r)
   if mutation=='role': r['role_eligibility']['selection']=True
   if mutation=='wav_as_pcm':r['derived_pcm']['sha256']=r['derived_wav']['sha256']
   if mutation=='asr_pcm':r['asr'][0]['bound_pcm_sha256']=r['derived_pcm']['sha256']
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):b.validate(d)
 def test_json_fail_closed(self):
  for raw in (b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}'):
   with self.assertRaises(ValueError):b.decode(raw)
 def test_zip_boundaries(self):
  for names in (['../escape'],['/absolute'],['a\\b'],['same','same']):
   with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'x.zip'
    with zipfile.ZipFile(p,'w') as z:
     for name in names:z.writestr(name,b'x')
    with self.assertRaises(ValueError):b.archive(p,b.sha(p.read_bytes()))
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.zip'
   with zipfile.ZipFile(p,'w') as z:z.writestr('x',b'x')
   with self.assertRaises(ValueError):b.archive(p,'0'*64)
   link=Path(t)/'link.zip';link.symlink_to(p)
   with self.assertRaises(ValueError):b.archive(link,b.sha(p.read_bytes()))
 def test_real_pcm_not_container(self):
  stream=io.BytesIO();pcm=struct.pack('<hhh',-12,0,12)
  with wave.open(stream,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
  raw=stream.getvalue();d=b.pcm_identity(raw);self.assertEqual(d['sha256'],b.sha(pcm));self.assertNotEqual(d['sha256'],b.sha(raw))
  with self.assertRaises(ValueError):b.pcm_identity(raw[:-1])

 def test_boolean_types(self):
  base=b.decode((HERE/'candidates.json').read_bytes())
  for where,key in [('top','human_gold'),('row','training_admitted'),('row','same_pcm_asr_binding_verified'),('role','selection')]:
   d=copy.deepcopy(base);target=d if where=='top' else d['records'][0] if where=='row' else d['records'][0]['role_eligibility'];target[key]=0
   with self.subTest(where=where,key=key),self.assertRaises(ValueError):b.validate(d)
 def test_tampered_pins_and_source_never_execute(self):
  from unittest.mock import patch
  pins=b.decode((HERE/'input-pins.json').read_bytes())
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);marker=root/'executed';evil=("from pathlib import Path\nPath("+repr(str(marker))+").write_text('executed')\n").encode()
   hostile=dict(pins);hostile[b.INPUTS[4]]=b.sha(evil)
   pinfile=root/'pins.json';pinfile.write_text(b.canonical(hostile))
   with self.assertRaises(ValueError):b.verified_sources(root,pinfile)
   # Valid fixed pin file + a changed executable source must fail before module().
   for name in b.INPUTS:
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(evil)
   with patch.object(b,'module',side_effect=RuntimeError('module must not run')) as execute:
    with self.assertRaises(ValueError):b.build(root,root)
    execute.assert_not_called()
   self.assertFalse(marker.exists())

class LiveRegistryTests(unittest.TestCase):
 def test_live_regeneration_and_adapter(self):
  self.assertTrue(os.environ.get('KWS_REGISTRY_REPO') and os.environ.get('KWS_REGISTRY_ARTIFACTS'),'locally retained artifact ZIP unavailable; --require-live requires both source environment variables')
  sources={'repo':Path(os.environ['KWS_REGISTRY_REPO']),'artifacts':Path(os.environ['KWS_REGISTRY_ARTIFACTS'])}
  d=b.decode((HERE/'candidates.json').read_bytes());self.assertEqual(b.build(**sources),d)
  rows=b.verify_inputs(d,sources);self.assertEqual(len(rows),16);self.assertEqual(sum(r['target_class']=='KW1' for r in rows),5)
  d['records'][0]['asr'][0]['raw_text']='小吴'
  with self.assertRaises(ValueError):b.verify_inputs(d,sources)
def load_tests(loader, tests, pattern):
 suite=loader.loadTestsFromTestCase(RegistryTests)
 if REQUIRE_LIVE:suite.addTests(loader.loadTestsFromTestCase(LiveRegistryTests))
 return suite
if __name__=='__main__':unittest.main()
