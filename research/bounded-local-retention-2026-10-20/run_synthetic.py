#!/usr/bin/env python3
"""Reconstruct an ephemeral research candidate and run invented-state tests only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HEADERS = ('kws3_diagnostic_ledger.h', 'kws3_diagnostic_ledger_impl.h', 'kws3_retention.h')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--cc', default='cc', choices=['cc', 'gcc', 'clang'])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    identity = json.loads((here / 'BUILD-IDENTITY.json').read_text())
    base = (args.repo_root or here.parents[1]) / identity['base_repo_directory']
    require(digest(base / 'kws3.c') == identity['base_source_sha256'], 'Public C base identity mismatch')
    require(digest(base / 'kws3.h') == identity['base_header_sha256'], 'Public header identity mismatch')
    support = ('candidate.patch', 'native_state_mock.c') + tuple('headers/' + h for h in HEADERS)
    require(set(identity['support_sha256']) == set(support), 'Support inventory mismatch')
    for name in support:
        require(digest(here / name) == identity['support_sha256'][name], 'Support identity mismatch: ' + name)
    with tempfile.TemporaryDirectory(prefix='bounded-local-retention-native-') as temp:
        root = Path(temp)
        candidate = root / 'source/candidate'
        candidate.mkdir(parents=True)
        mocks = root / 'native-mocks'
        mocks.mkdir()
        for name in ('kws3.c', 'kws3.h'):
            shutil.copyfile(base / name, candidate / name)
        for name in HEADERS:
            shutil.copyfile(here / 'headers' / name, candidate / name)
        subprocess.run(['patch', '--batch', '--forward', '-p1', '-d', str(candidate),
                        '-i', str(here / 'candidate.patch')], check=True, timeout=10,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(digest(candidate / 'kws3.c') == identity['reconstructed_candidate_sha256'],
                'Candidate reconstruction mismatch')
        shutil.copyfile(here / 'native_state_mock.c', mocks / 'retention_state_mock.c')
        exe = mocks / 'retention_state_mock'
        command = [args.cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-fno-fast-math',
                   '-ffp-contract=off', '-fexcess-precision=standard', '-O2',
                   str(mocks / 'retention_state_mock.c'), '-lm', '-o', str(exe)]
        subprocess.run(command, check=True, timeout=30)
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=10)
        result = json.loads(run.stdout)
        require(result['case_count'] == identity['native_state_cases'] and
                result['checks'] == identity['native_state_checks'], 'Native case/check count mismatch')
        require(all(result[k] == 0 for k in ('scientific_posterior_reads', 'model_forward_calls',
                                            'kws3_step_calls', 'kws3_process_calls')),
                'Synthetic-only scope violation')
        print(json.dumps({'status': 'PASS_RECONSTRUCTED_NATIVE_SYNTHETIC_ONLY',
                          'compiler': args.cc,
                          'candidate_sha256': identity['reconstructed_candidate_sha256'],
                          'case_count': result['case_count'], 'checks': result['checks'],
                          'production_files_modified': False, 'scientific_replays': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
