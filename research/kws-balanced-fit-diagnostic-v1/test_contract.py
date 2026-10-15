import tempfile
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import Budget, atomic_json, collapse, weights, select_pair

class ContractTests(unittest.TestCase):
    def test_archived_observation_metadata_and_statistics(self):
        from verify_archived_observation import verify, numeric_types
        self.assertEqual(verify()["result_files"], 608)
        for bad in ({"ctc_nll": True}, {"blank_mean_probability": False}, {"per_class_complete_rate": {"KW1": True, "KW2": 0.0}}, {"update": True}, {"target_tokens": [True]}, {"ctc_nll": float("nan")}, {"finite_grad_norm": float("inf")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                numeric_types(bad)

    def test_ctc_blanks_separate_repeats(self):
        self.assertEqual(collapse([3,3,0,3,4,4,0]),[3,3,4])
        self.assertEqual(collapse([0,0]),[])
    def test_weights(self):
        classes=['KW1']*5+['KW2']*11
        w=weights(classes,'B')
        self.assertAlmostEqual(sum(w[:5]),.5)
        self.assertAlmostEqual(sum(w[5:]),.5)
        self.assertEqual(weights(classes,'A'),[1/16]*16)
    def test_pair_stable_common_voice(self):
        rows=[dict(record_id='z',target_class='KW1',preset_voice_id='b'),dict(record_id='a',target_class='KW1',preset_voice_id='a'),dict(record_id='c',target_class='KW2',preset_voice_id='b')]
        self.assertEqual(select_pair(rows),['z','c'])
    def test_exclusive_results_and_claim(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'; claim=Path(d)/'campaign.json'; b=Budget(p,claim,'unit-test')
            with self.assertRaises(FileExistsError):Budget(p,claim,'unit-test')
            with self.assertRaises(FileExistsError):Budget(Path(d)/'different-output',claim,'unit-test')
            b.before_step('pair',1)
            with self.assertRaises(FileExistsError):atomic_json(p/'claim.json',{})
            b.count=1500
            with self.assertRaises(ValueError):b.before_step('B',601)
    def test_execution_disabled_before_import_or_input_read(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location("diagnostic_runner",Path(__file__).with_name("run_diagnostic.py"))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import json
        from unittest.mock import patch
        from types import SimpleNamespace
        plan=json.loads(Path(__file__).with_name('plan.json').read_text())
        self.assertIs(type(plan['execution_enabled']),bool)
        plan['execution_enabled']=False
        with patch.object(module,'read_checked',return_value=json.dumps(plan).encode()):
            with self.assertRaisesRegex(ValueError,"execution disabled"):
                module.run(SimpleNamespace(plan_sha256='isolated-disabled-test'))
        self.assertNotIn("torch",sys.modules)
    def test_read_checked_rejects_aliases(self):
        import run_diagnostic as runner
        import os
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'x')
            h=runner.sha(b'x')
            self.assertEqual(runner.read_checked(p,h),b'x')
            symlink=Path(d)/'s';symlink.symlink_to(p)
            with self.assertRaises(ValueError):runner.read_checked(symlink,h)
            hardlink=Path(d)/'h';os.link(p,hardlink)
            with self.assertRaises(ValueError):runner.read_checked(p,h)
    def test_invalid_token(self):
        for path in ([5],[True],[-1]):
            with self.assertRaises(ValueError):collapse(path)

if __name__=='__main__': unittest.main()
