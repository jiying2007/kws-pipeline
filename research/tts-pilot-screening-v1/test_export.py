import unittest,tempfile,json
from pathlib import Path
from screen import write,manifest
from test_screen import data
from export_public import export
class TestExport(unittest.TestCase):
 def test_allowlist_not_tree_archive(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);host=root/'host';derived=root/'derived';host.mkdir();derived.mkdir();d=data()
   for r in d['slots']:
    if r['engine']!='cosy3':r['status']='not_run';r['audio_quality_pass']=False
   write(derived/'manifest.json',d);write(host/'results.json',[])
   for name in ('runtime/weights.bin','code/source.py','setup-primary/log.json','report.json','raw-secret.json'):
    p=host/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('do not export')
   export(host,derived,root/'public');self.assertEqual({p.name for p in (root/'public').iterdir()},{'results.json','report.json','generation-receipts.json'});self.assertNotIn('do not export',''.join(p.read_text() for p in (root/'public').iterdir()))
 def test_manifest_deletion_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'manifest.json';d=data();d['slots'].pop();write(p,d)
   with self.assertRaises(ValueError):manifest(p)
 def test_manifest_target_swap_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'manifest.json';d=data();d['slots'][0]['target_text']='小窝小窝';write(p,d)
   with self.assertRaises(ValueError):manifest(p)
if __name__=='__main__':unittest.main()
