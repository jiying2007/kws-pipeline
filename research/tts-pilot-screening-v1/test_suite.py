"""Offline tests; works under isolated/no-site Python, no model imports/calls."""
from pathlib import Path
import argparse
import importlib.util
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-numeric',action='store_true');args=parser.parse_args()
    names=['test_screen','test_host','test_terminal','test_export']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    if importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('scipy') is not None:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_derive'))
    else:
        if args.require_numeric:raise RuntimeError('--require-numeric requires NumPy and SciPy; no skipped numerical coverage')
        def scipy_unavailable():raise unittest.SkipTest('NumPy/SciPy unavailable in isolated no-site runtime; resampling tests require separate dependency-enabled run')
        suite.addTest(unittest.FunctionTestCase(scipy_unavailable))
    count=suite.countTestCases()
    if count<22:raise RuntimeError('expected nonzero complete stdlib contract test inventory')
    print('PILOT_ASR_TEST_COUNT='+str(count),flush=True)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not (args.require_numeric and result.skipped) else 1
if __name__=='__main__':raise SystemExit(main())
