"""Invented parser/arithmetic tests, not additional core fixtures or native runs."""
import copy, hashlib, importlib.util, json, math, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'src/verify_saved.py'
spec=importlib.util.spec_from_file_location('validator',P);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def row(**kw):
 r=['0']*6
 for k,x in kw.items():r[int(k[1:])]=str(x)
 return r
class Tests(unittest.TestCase):
 def test_duplicate_keys(self):
  with self.assertRaises(ValueError):v.parse('{"x":1,"x":2}')
 def test_nonfinite_json(self):
  for x in ['NaN','Infinity','-Infinity']:
   with self.assertRaises(ValueError):v.parse('{"x":'+x+'}')
 def test_path_traversal(self):
  for p in ['../x','/absolute','x/../../y','x\\y','']:
   with self.assertRaises(ValueError):v.safe_relative(p)
 def test_relative_path(self):self.assertEqual(str(v.safe_relative('data/raw.jsonl')),'data/raw.jsonl')
 def test_ieee_nonfinite(self):
  for s in ['7f800000','7fc00001']:
   with self.assertRaises(ValueError):v.bitfloat(s,32)
 def test_ieee_width_and_uppercase(self):
  for s in ['3f8000','3F800000','garbage0']:
   with self.assertRaises(ValueError):v.bitfloat(s,32)
 def test_zero_bit_identity(self):self.assertNotEqual(v.f64bits(0.0),v.f64bits(-0.0))
 def test_raw_unknown_field(self):
  s={'records':1,'kinds':{'x':{'records':1,'fields':{'$':['dict'],'$.kind':['str']},'object_keys':{'$':[['kind']]}}},'string_classes':{'$.kind':{'values':['x']}}}
  v.validate_schema([{'kind':'x'}],s)
  with self.assertRaises(ValueError):v.validate_schema([{'kind':'x','audio':[0.1]}],s)
 def test_raw_unknown_string(self):
  s={'records':1,'kinds':{'x':{'records':1,'fields':{'$':['dict'],'$.kind':['str'],'$.id':['str']},'object_keys':{'$':[['id','kind']]}}},'string_classes':{'$.kind':{'values':['x']},'$.id':{'values':['fixture']}}}
  with self.assertRaises(ValueError):v.validate_schema([{'kind':'x','id':'unreviewed person'}],s)
 def test_raw_wrong_count(self):
  with self.assertRaises(ValueError):v.validate_schema([],{'records':1})
 def test_rne_cell(self):self.assertTrue(all(x>0 for x in v.strict_cell('1.0','3f800000')[:2]))
 def test_rne_wrong_bits(self):
  with self.assertRaises(ValueError):v.strict_cell('1.0','40000000')
 def test_rne_boundaries(self):
  with self.assertRaises(ValueError):v.strict_cell(str(v.Q(1)+v.Q(1,2**24)),'3f800000')
 def test_repeat_then_blank(self):
  self.assertEqual(v.collapse([1,1,0,1]),(1,1));self.assertEqual(v.collapse([1,1]),(1,))
 def test_rational_mass(self):
  rs=[row(t0='1/4',t1='3/4'),row(t0='1/2',t1='1/2')];g=v.groups(v.enumerate_paths(rs),1,0)
  self.assertEqual(g[(1,)],[v.Q(3,8),v.Q(1,2)])
 def test_bad_distribution(self):
  for rs in [[['1','1','0','0','0','0']],[['-1','2','0','0','0','0']],[['1']]]:
   with self.assertRaises(ValueError):v.enumerate_paths(rs)
 def test_bounded_oracle(self):
  with self.assertRaises(ValueError):v.enumerate_paths([['1/2','1/2','0','0','0','0']]*24)
 def test_joint_witness_distinct_tokens(self):
  rs=[row(t1=1),row(t2=1)];ps=v.enumerate_paths(rs)
  self.assertTrue(v.has_witness(ps,rs,1,0,[1,2],[0,3],[1,1]));self.assertFalse(v.has_witness(ps,rs,1,0,[1,2],[3,3],[1,1]))
 def test_blank_separation(self):
  rs=[row(t1=1),row(t1=1),row(t0=1),row(t1=1)];ps=v.enumerate_paths(rs)
  self.assertTrue(v.has_witness(ps,rs,3,0,[1,1],[3,9],[1,1]));self.assertFalse(v.has_witness(ps,rs,3,0,[1,1],[0,3],[1,1]))
 def test_late_peak_legitimate(self):
  rs=[row(t0='1/2',t1='1/2'),row(t0='1/4',t1='3/4')];ps=v.enumerate_paths(rs)
  self.assertTrue(v.has_witness(ps,rs,1,0,[1],[3],['3/4']));self.assertFalse(v.has_witness(ps,rs,1,0,[1],[3],['1/2']))
 def test_reset_absolute_coordinates(self):
  rs=[row(t1=1),row(t0=1),row(t1=1)];ps=v.enumerate_paths(rs)
  self.assertTrue(v.has_witness(ps,rs,2,1,[1],[6],[1]));self.assertFalse(v.has_witness(ps,rs,2,1,[1],[0],[1]))
 def test_source_text_rule(self):
  x=v.text_presence('你 好，小 窝。','')
  self.assertTrue(x['exact_keyword_text_hits']['你好小窝']);self.assertFalse(x['exact_keyword_text_hits']['小窝小窝'])
 def test_raw_text_not_substituted(self):
  self.assertFalse(any(v.text_presence('无关文本','你好小窝')['exact_keyword_text_hits'].values()))
 def test_optional_text_hash_mismatch(self):
  records=[{'source_row_index':i,'transcription_sha256':'0'*64,'raw_transcription_sha256':'0'*64} for i in range(20)]
  supplied=[{'source_row_index':i,'transcription':'invented','raw_transcription':'invented'} for i in range(20)]
  with self.assertRaises(ValueError):v.verify_optional_texts({'records':records},supplied)
 def test_publication_pin_closed_and_immutable(self):
  valid={'repository':'jiying2007/kws-data','candidate_path':'research/2026-10-04-domain-decoder-diagnostics','new_record_commit':'a'*40,'publication_ready':True,'manifest_sha256':'b'*64,'manifest_bytes':123}
  self.assertEqual(v.publication_pin(valid),'a'*40)
  mutations=[{'publication_ready':False},{'publication_ready':1},{'new_record_commit':'NOT_PUBLISHED'},{'new_record_commit':'main'},{'new_record_commit':'a'*39},{'repository':'someone/other-data'},{'candidate_path':'different-record'},{'manifest_sha256':'x'*64},{'manifest_bytes':True},{'manifest_bytes':0}]
  for change in mutations:
   with self.subTest(change=change),self.assertRaises(ValueError):v.publication_pin(dict(valid,**change))
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);body=b'{}';(root/'MANIFEST.json').write_bytes(body)
   bound=dict(valid,manifest_sha256=hashlib.sha256(body).hexdigest(),manifest_bytes=len(body))
   with patch.object(v,'read',return_value=bound):
    self.assertTrue(v.check_data_reference(root,True,'a'*40)['actual_checkout_commit_checked'])
    for wrong in [None,'b'*40,'main']:
     with self.subTest(actual_commit=wrong),self.assertRaises(ValueError):v.check_data_reference(root,True,wrong)
    (root/'MANIFEST.json').write_bytes(b'{ }')
    with self.assertRaisesRegex(ValueError,'manifest binding'):v.check_data_reference(root,True,'a'*40)
 def test_manifest_unlisted_file(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'MANIFEST.json').write_text('{"schema":"a20-domain-public-payload-manifest-v1","files":[]}');(root/'extra.txt').write_text('x')
   with self.assertRaisesRegex(ValueError,'unlisted'):v.check_manifest(root)
 def test_manifest_corrupt_hash(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'a.json').write_text('{}');(root/'MANIFEST.json').write_text(json.dumps({'schema':'a20-domain-public-payload-manifest-v1','files':[{'path':'a.json','bytes':2,'sha256':'0'*64}]}))
   with self.assertRaisesRegex(ValueError,'identity'):v.check_manifest(root)
 def test_manifest_symlink(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'real').write_text('x');(root/'link.json').symlink_to('real');(root/'MANIFEST.json').write_text(json.dumps({'schema':'a20-domain-public-payload-manifest-v1','files':[{'path':'link.json','bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}]}))
   with self.assertRaisesRegex(ValueError,'symlink'):v.check_manifest(root)
if __name__=='__main__':unittest.main(verbosity=2)
