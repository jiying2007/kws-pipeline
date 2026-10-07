"""Pure saved-data and tiny invented in-memory checks; no acoustic execution."""
import gzip,hashlib,importlib.util,json,os
from pathlib import Path
import struct,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('A20_N0_DATA_ROOT',ROOT.parents[2]/'kws-data/research/2026-10-04-n0-deterministic-controls'))
sys.path.insert(0,str(ROOT));import validate as V
sys.path.insert(0,str(ROOT/'src'));import generate_inputs as G
class PublicRecord(unittest.TestCase):
 def test_complete_saved_record(self):
  result=V.validate(ROOT,DATA);self.assertEqual(result['records'],6012);self.assertEqual(result['events'],0);self.assertFalse(result['projection_executed'])
 def test_staged_pin_cannot_publish(self):
  ref=V.parse((ROOT/'DATA-REFERENCE.json').read_bytes())
  if ref['publication_ready']:
   self.assertRegex(ref['commit'],r'^[0-9a-f]{40}$')
  else:
   with self.assertRaises(V.Invalid):V.validate(ROOT,DATA,require_published=True)
 def test_duplicate_json_key_rejected(self):
  with self.assertRaises(V.Invalid):V.parse(b'{"a":1,"a":2}')
 def test_nonfinite_json_rejected(self):
  for value in [b'NaN',b'Infinity',b'-Infinity']:
   with self.subTest(value=value),self.assertRaises(V.Invalid):V.parse(value)
 def test_unsafe_paths_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   for name in ['', '../x','a/../b','/absolute','a//b','./a','a\\b','a\0b']:
    with self.subTest(name=name),self.assertRaises(V.Invalid):V.safe_path(Path(td),name)
 def test_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'a').symlink_to(root/'b')
   with self.assertRaises(V.Invalid):V.safe_path(root,'a')
 def test_gzip_extra_member_trailing_truncation_rejected(self):
  b=b'pure invented JSON-like bytes\n';z=gzip.compress(b,mtime=0)
  self.assertEqual(V.unpack(z,len(b),V.sha(b)),b)
  for bad in [z+b'x',z+z,z[:-1],z[:10]]:
   with self.subTest(length=len(bad)),self.assertRaises(V.Invalid):V.unpack(bad,len(b),V.sha(b))
 def test_gzip_size_and_hash_rejected(self):
  b=b'invented'*100;z=gzip.compress(b,mtime=0)
  with self.assertRaises(V.Invalid):V.unpack(z,10,V.sha(b))
  with self.assertRaises(V.Invalid):V.unpack(z,len(b),'f'*64)
 def test_round_half_away_signed(self):
  for value,expected in [(-48,-2),(-32,-1),(-16,-1),(-15,0),(0,0),(15,0),(16,1),(32,1),(48,2)]:self.assertEqual(G.round32(value),expected)
 def test_tiny_generator_block_boundary_and_colored_startup(self):
  recipe=V.parse((DATA/'GENERATOR-RECIPE.json').read_bytes())
  for row in recipe['streams']:
   prefix=row['domain'].encode('ascii')+b'\0'+bytes.fromhex(row['seed_hex'])
   # Direct independent byte-index formulation, 33 samples total per case.
   stream=b''.join(hashlib.sha256(prefix+int(k).to_bytes(8,'little')).digest() for k in range(3))
   triangle=[stream[2*i]-stream[2*i+1] for i in range(33)]
   expected=triangle
   if row['recording']=='n0-colored-ma32':
    expected=[]
    for i in range(33):
     total=16*sum(triangle[max(0,i-31):i+1]);q=(abs(total)+16)//32;expected.append(-q if total<0 else q)
   # Transient stream prefix precedes the first pulse; full pulses not executed here.
   with self.subTest(recording=row['recording']):
    self.assertEqual(list(G.samples(row,33)),expected);self.assertEqual(list(G.samples(row,0)),[])
 def test_inputs_excluded_and_source_identity(self):
  self.assertFalse(any(p.suffix in ('.wav','.pcm','.so','.f32') for p in ROOT.rglob('*')))
  self.assertFalse(any(p.suffix in ('.wav','.pcm','.so','.f32') for p in DATA.rglob('*')))
  self.assertEqual(V.sha((ROOT/'src/generate_inputs.py').read_bytes()),'8dbb6c1a24f684bcd23df71f41950b16d4b2d31bd3dd83887d6982cab5fe99fb')
 def test_no_active_launcher(self):
  self.assertFalse((ROOT/'src/launch_once.py').exists())
  self.assertFalse((ROOT/'future').exists())
  archive=(ROOT/'archive/guard_probe_v2.py.txt').read_text();self.assertIn('DISABLED',archive.upper())
 def test_original_failed_and_revised_unknown_separate(self):
  old=V.parse((DATA/'evidence/strict-probe-failure.public.json').read_bytes());new=V.parse((DATA/'evidence/resource-record.public.json').read_bytes())
  self.assertEqual(old['status'],'FAILED_NO_RETRY');self.assertEqual(new['status'],'RAW_COMPLETE_PENDING_AUDIT')
  self.assertEqual(new['terminal_lifetime_io'],'UNKNOWN');self.assertEqual(new['lifetime_child_absence'],'NOT_PROVEN')
if __name__=='__main__':unittest.main()
