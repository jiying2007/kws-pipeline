"""Offline synthetic contracts plus retained numeric receipts; no corpus/models."""
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'overlap'));sys.path.insert(0,str(HERE/'machine-positive-control-v1'))

def main():
    loader=unittest.TestLoader()
    names=('test_alimeeting_eval','test_diagnostic_runner','test_compare_overlap','test_run_control','test_ab_data_admission','test_receipts')
    suite=unittest.TestSuite(loader.loadTestsFromName(name) for name in names)
    if loader.errors or suite.countTestCases()!=83:
        raise RuntimeError('Explicit offline suites failed discovery or returned unexpected test count')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped or result.testsRun!=suite.countTestCases():
        raise RuntimeError('All discovered tests must execute without skips')
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
