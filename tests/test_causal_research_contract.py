"""Run only artificial stdlib/oracle contracts; never a Torch numerical probe."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('research/fsmn2599-three-window-c/kws-unified-comparison-preparation-20261017', 'test_threshold_adapter.py', 6),
    ('research/causal-event-head', 'test_event_head_synthetic.py', 6),
    ('research/causal-mdtc', 'test_static.py', 8),
)

def main():
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
