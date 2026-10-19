"""Run only artificial stdlib/oracle contracts; never a Torch numerical probe."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('research/independent-metadata-prep', 'test_metadata.py', 13),
    ('research/causal-kws-increment-2026-10-18/tests', 'test_core_contracts.py', 6),
    ('research/causal-kws-increment-2026-10-18/tests', 'test_policy_contracts.py', 4),
    ('research/fsmn2599-three-window-c/kws-unified-comparison-preparation-20261017', 'test_threshold_adapter.py', 6),
    ('research/causal-event-head', 'test_event_head_synthetic.py', 6),
    ('research/causal-mdtc', 'test_static.py', 8),
    ('research/evaluation_integrity', 'test_bound_metrics.py', 16),
    ('research/bounded-lifecycle-v2', 'test_controller.py', 17),
    ('research/fsmn2599-stream-c', 'test_arm_synthetic.py', 12),
    ('research/fsmn2599-stream-c', 'test_arm_dependencies.py', 19),
)

def verify_metadata_retention():
    # Reuse the repository's strict byte/mode/hash/symlink checker.
    sys.path.insert(0, str(ROOT / 'tools'))
    from verify_research_sources import check_file
    prefix = 'research/independent-metadata-prep/'
    manifest = json.loads((ROOT / prefix / 'RETENTION.json').read_text())
    expected = {prefix + name for name in
                ('metadata.py', 'test_metadata.py', 'README.md', 'ANONYMOUS-SUMMARY.json')}
    rows = manifest['files']
    if (manifest['schema'] != 'public-metadata-preparation-retention-v1'
            or len(rows) != len(expected) or {row['path'] for row in rows} != expected):
        raise ValueError('metadata preparation retention scope mismatch')
    for row in rows:
        check_file(ROOT, row)
    print('PASS: four public metadata preparation identities')


def main():
    verify_metadata_retention()
    mode = ['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else []
    for directory, script, expected in CASES:
        result = subprocess.run([sys.executable, '-S', '-B', *mode, script],
                                cwd=ROOT / directory, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=60, check=False)
        print(result.stdout, end='')
        if result.returncode != 0:
            raise RuntimeError(f'{script} failed: exit {result.returncode}')
        if not re.search(rf'Ran {expected} tests\b', result.stdout):
            raise RuntimeError(f'{script} did not execute the expected {expected} cases')
        if not re.search(r'^OK\s*$', result.stdout, re.MULTILINE):
            raise RuntimeError(f'{script} success completion missing or cases skipped')
    print('PASS: artificial stdlib/oracle contracts only; no numerical Torch revalidation')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
