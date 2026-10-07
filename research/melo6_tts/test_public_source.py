"""Standard-library source/binding checks; no install, model, or runtime."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent / 'melo6_asr'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


class PublicSourceProof(unittest.TestCase):
    def test_complete_asr_source_freeze_matches(self):
        freeze = json.loads((ASR / 'candidate-freeze.json').read_bytes())
        actual = {p.relative_to(ASR).as_posix() for p in ASR.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(freeze['files']) | {'candidate-freeze.json'})
        for name, digest in freeze['files'].items():
            self.assertEqual(sha(ASR / name), digest, name)

    def test_all_python_compiles_without_runtime_import(self):
        for root in (ASR, ROOT):
            for path in root.rglob('*.py'):
                compile(path.read_bytes(), path.name, 'exec')

    def test_public_plan_binds_projection_and_keeps_proposed_identity_explicit(self):
        import run_melo6
        projection = json.loads((ROOT / 'public-projection.json').read_bytes())
        original = projection['original_execution_plan_sha256']
        public = projection['public_execution_plan_sha256']
        self.assertNotEqual(original, public)
        run_melo6.validate_plan(ROOT / 'execution-plan.json', public)
        release = json.loads((ROOT.parent / 'melo6-source-screen-release.json').read_bytes())
        self.assertEqual(release['plan_sha256'], public)
        self.assertEqual(projection['proposed_execution_plan_sha256'], public)
        self.assertEqual(projection['execution_binding_status'], 'authorized_recovery_pending_independent_binding_review; no_inference_yet')
        self.assertEqual(release['tts_candidate_sha256'], sha(ROOT / 'adapter-freeze.json'))
        self.assertEqual(release['asr_candidate_sha256'], sha(ASR / 'candidate-freeze.json'))

    def test_fixed_asr_contract_bounds(self):
        constants = {}
        for node in ast.parse((ASR / 'core/asr6_contract.py').read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {'EXPERIMENT', 'MAX_FRAMES', 'MAX_ARCHIVE_BYTES'}:
                    constants[name] = eval(compile(ast.Expression(node.value), '<constant>', 'eval'), {'__builtins__': {}})
        self.assertEqual(constants, {'EXPERIMENT': 'melo-six-phrase-source-screen-asr6-v1',
                                    'MAX_FRAMES': 160000, 'MAX_ARCHIVE_BYTES': 2097152})

    def test_release_unapproved_no_generated_inputs_and_exact_workflow(self):
        namespace = {}
        exec(compile((ASR / 'release_gate.py').read_bytes(), '<release gate>', 'exec'), namespace)
        release_path = ROOT / 'asr-release-template.json'
        release = json.loads(release_path.read_bytes())
        self.assertIs(release['approved'], False)
        self.assertIsNone(release['blind_archive_sha256'])
        self.assertIsNone(release['blind_freeze_sha256'])
        with tempfile.TemporaryDirectory() as temporary:
            staged_asr = Path(temporary) / 'research/melo6_asr'
            staged_asr.mkdir(parents=True)
            staged_release = staged_asr.parent / 'melo6-source-screen-release.json'
            staged_release.write_bytes(release_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'approved'):
                namespace['verify_release'](staged_asr, 'asr', staged_release)
        workflow = ROOT.parents[1] / '.github/workflows/melo6-source-screen.yml'
        self.assertEqual(workflow.read_bytes(), (ASR / 'workflow.yml').read_bytes())


if __name__ == '__main__':
    unittest.main()
