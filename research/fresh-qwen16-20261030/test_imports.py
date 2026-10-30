"""Fresh subprocess verifies coexistence with retained stdlib launch modules."""
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class ImportIsolationTests(unittest.TestCase):
    def test_real_retained_host_import_without_models(self):
        code = '''import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('fresh_launcher', root / 'research/fresh-qwen16-20261030/run.py')
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)
if 'contract' in sys.modules:
    raise RuntimeError('fresh launcher contaminated retained contract namespace')
sys.path.insert(0, str(root / 'research/source-screen32-v1'))
import hosted_run
if not callable(hosted_run.host_capacity) or len(fresh.plan()) != 16:
    raise RuntimeError('launcher contract mismatch')
if any(name in sys.modules for name in ('torch', 'transformers', 'qwen_tts')):
    raise RuntimeError('model runtime imported by source-only test')
print('ISOLATED_IMPORT_PASS')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT)],
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ISOLATED_IMPORT_PASS')


if __name__ == '__main__':
    unittest.main()
