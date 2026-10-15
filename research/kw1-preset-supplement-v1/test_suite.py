#!/usr/bin/env python3
"""Explicit offline-only test inventory, fail closed on absent or skipped tests."""
import argparse
from pathlib import Path
import sys
import unittest
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
TESTS=('test_contract','test_worker','test_execution','test_blind_asr','test_docker_backend','test_archive_faults','test_atomic_export')

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--require-numeric',action='store_true');args=p.parse_args(argv)
    names=list(TESTS)
    if args.require_numeric:
        import numpy
        import scipy
        names.append('test_derive_numeric')
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    if suite.countTestCases()==0:raise SystemExit('zero tests forbidden')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 1
if __name__=='__main__':raise SystemExit(main())
