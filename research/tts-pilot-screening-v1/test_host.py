import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import host
class TestHost(unittest.TestCase):
 def test_setup_command(self):
  with tempfile.TemporaryDirectory() as t, patch.object(host.b,'host_capacity'),patch.object(host.b,'supervise_container',return_value={}) as run,patch.object(host.b.os,'getuid',return_value=1001):
   p=Path(t);host.launch(p,p,p,p/'out','image','qwen06',setup=True)
   args=run.call_args.args[0];self.assertIn('bridge',args);self.assertIn('--profile',args)
 def test_whisper_blind_command(self):
  with tempfile.TemporaryDirectory() as t, patch.object(host.b,'host_capacity'),patch.object(host.b,'supervise_container',return_value={}) as run,patch.object(host.b.os,'getuid',return_value=1001):
   p=Path(t);host.launch(p,p,p,p/'out','image','whisper',p/'input')
   args=run.call_args.args[0];self.assertIn('none',args);self.assertTrue(args[-1].endswith('whisper_worker.py'))
 def test_stage_excludes_targets_controls(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with self.assertRaisesRegex(ValueError,'adapter source drift'):
    host.stage(root/'live-rejected')
   fixture=root/'historical-source'
   historical={'research/qwen6_asr/'+name for name in
    ('core/asr_stage/adapters.py','core/asr_stage/execution.py','core/sense_binding.py')}
   lock=json.loads((host.HERE/'source-lock.json').read_bytes())
   for name,sha in lock['files'].items():
    source=host.ROOT/('research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/'+name if name in historical else name)
    self.assertFalse(source.is_symlink());raw=source.read_bytes()
    self.assertEqual(hashlib.sha256(raw).hexdigest(),sha)
    target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   with patch.object(host,'ROOT',fixture):host.stage(root/'staged')
   paths=[p for p in (root/'staged').rglob('*') if p.is_file()]
   self.assertFalse(any(p.name in ('execution-release.json','execution-freeze.json','hosted_run.py','plan.json','screen.py') for p in paths))
   self.assertFalse(any('你好小窝' in p.read_text(errors='replace') for p in paths))
if __name__=='__main__':unittest.main()
