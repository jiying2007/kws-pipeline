#!/usr/bin/env python3
"""Read-only source-provenance clarification; never import or run the writer."""
import argparse
import hashlib
import json
from pathlib import Path

PLAN_SHA256 = 'cc947af40d436b1969fb78ae6c2c1c9f5953b4405f57b39949fb52a52ad273e8'
WRITER_SHA256 = '0a3a0e919f5707b30f13c5bc73f5babc3d0c66b8f1e77e430a4c49043b0ab835'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.output.resolve().parent != here:
        raise ValueError('output must be new file in independent audit directory')
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    raw = (here / 'analysis-plan.json').read_bytes()
    assert sha(raw) == PLAN_SHA256
    audit = json.loads(raw)
    pin = audit['inputs']['plan']
    raw = (args.base / pin['relative_path']).read_bytes()
    assert sha(raw) == pin['sha256']
    experiment = json.loads(raw)
    name = 'lr-stability-integrated/research/kws-lr-stability-v1/run_lr_stability.py'
    raw = (args.base / name).read_bytes()
    assert sha(raw) == WRITER_SHA256 == experiment['diagnostic_code_sha256']['run_lr_stability.py']
    report = {
        'schema': 'saved-prefix-source-provenance-addendum-v1',
        'status': 'PASS',
        'analysis_plan_sha256': PLAN_SHA256,
        'experiment_plan_sha256': pin['sha256'],
        'actual_observation_writer': {'relative_path': name, 'sha256': WRITER_SHA256},
        'correction': 'The analysis plan input named observation_writer is the grouped-balance helper, not the actual LR-stability observation writer. Its stored path and hash are accurate; this is a role-label correction. The actual writer is independently bound here through the original pinned experiment plan.',
        'original_observation_writer_input': audit['inputs']['observation_writer'],
        'actual_writer_reviewed_lines': {'frontend_call': [94, 95], 'LP_and_raw_path': [120, 129]},
        'analysis_definitions_changed': False,
        'numeric_report_changed': False,
        'model_calls': 0,
        'source_code_executed': False,
        'verifier_sha256': sha(Path(__file__).read_bytes()),
    }
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('PASS: actual observation writer matches its original pinned source hash')


if __name__ == '__main__':
    main()
