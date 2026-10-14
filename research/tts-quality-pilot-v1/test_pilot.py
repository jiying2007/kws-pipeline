import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest
import pilot
class TestPilot(unittest.TestCase):
    def test_exact_matrix(self):
        for engine in pilot.MODELS:
            p = pilot.plan(engine)
            self.assertEqual(len(p),16)
            self.assertEqual(len({r['cell_id'] for r in p}),16)
            self.assertEqual(sum(r['resource_probe'] for r in p),1)
            self.assertEqual({r['text'] for r in p},set(pilot.TEXTS))
            self.assertEqual({r['speaker'] for r in p},set(pilot.VOICES))
    def test_cosy_not_fake_preset(self):
        with self.assertRaises(ValueError): pilot.plan('cosy3')
    def test_same_matched_conditions(self):
        for a,b in zip(pilot.plan('qwen06'),pilot.plan('qwen17')):
            self.assertEqual(pilot.kwargs(a),pilot.kwargs(b))
            self.assertEqual(a['seed'],b['seed'])
    def test_never_asr_gold(self):
        m=pilot.summarize([.1]*2400,24000,True)
        self.assertFalse(m['qualified'])
        self.assertFalse(pilot.qualified(m,{'three_asr_agree':True}))
        self.assertTrue(pilot.qualified(m,dict.fromkeys(('exact_target','no_missing_or_added_syllables','natural','no_tail_artifact'),True)))
    def test_bad_audio(self):
        for samples in [[],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError):pilot.summarize(samples,24000,True)
        self.assertFalse(pilot.summarize([0]*20,24000,True)['technically_valid'])
        self.assertFalse(pilot.summarize([1]*20,24000,True)['technically_valid'])
        self.assertFalse(pilot.summarize([.1]*20,24000,False)['technically_valid'])

class TestBindings(unittest.TestCase):
    def test_locks_and_retained_pins(self):
        import hashlib,json,importlib.util
        from pathlib import Path
        p=Path(pilot.__file__).parent
        spec=importlib.util.spec_from_file_location('original_setup',p.parent/'source-screen32-v1/setup_adapter.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for name,digest in m.RETAINED_PINS['tts'].items():
            self.assertEqual(hashlib.sha256((p.parent/'qwen6_tts'/name).read_bytes()).hexdigest(),digest)
        for engine in pilot.MODELS:
            lock=json.loads((p/(engine+'-model-lock.json')).read_text())
            self.assertEqual((lock['model_id'],lock['revision']),pilot.MODELS[engine])
            self.assertEqual(len(lock['files']),13)
            self.assertEqual(sum(x['size_bytes'] for x in lock['files']),lock['total_snapshot_bytes'])
            self.assertTrue(all(len(x['sha256'])==64 for x in lock['files']))
    def test_child_wrapper_entry(self):
        from pathlib import Path
        s=(Path(pilot.__file__).parent/'setup_adapter.py').read_text()
        self.assertIn('_run_child=base._run_child',s)
    def test_clean_environment_subprocess(self):
        import tempfile,shutil,subprocess,sys,os
        from pathlib import Path
        p=Path(pilot.__file__).parent
        with tempfile.TemporaryDirectory() as d:
            research=Path(d)/'research';research.mkdir()
            target=research/'tts-quality-pilot-v1'
            shutil.copytree(p,target,ignore=shutil.ignore_patterns('__pycache__'))
            old=research/'source-screen32-v1';old.mkdir()
            shutil.copyfile(p.parent/'source-screen32-v1/setup_adapter.py',old/'setup_adapter.py')
            for engine in pilot.MODELS:
                shutil.copyfile(target/(engine+'-model-lock.json'),target/'qwen-model-lock.json')
                code="import runpy,sys; m=runpy.run_path(sys.argv[1]); assert callable(m['_run_child']); print(m['engine'])"
                cmd=[sys.executable,'-I','-B','-c',code,str(target/'setup_adapter.py')]
                result=subprocess.run(cmd,env={'PATH':os.environ['PATH']},capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.strip(),engine)
                result=subprocess.run(cmd,env={'PATH':os.environ['PATH'],'TQP_ENGINE':'wrong'},capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('engine environment/lock mismatch',result.stderr)

if __name__ == '__main__': unittest.main()
