#!/usr/bin/env python3
"""Exercise the actual cleanup guard without GitHub/network/deletion calls."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = (ROOT / ".github/workflows/repository-cleanup.yml").read_text()
SNAPSHOT = json.loads((ROOT / "research/consolidation/branch-retention-2026-10-07.json").read_text())


def folded_env(name):
    return re.search(r"^  " + name + r": >-\n((?:    [^\n]+\n)+)", WORKFLOW, re.M).group(1).split()


class CleanupRetentionTests(unittest.TestCase):
    def test_exact_inventoried_names(self):
        expected = [row["name"] for row in SNAPSHOT["branches"]]
        self.assertEqual(len(expected), 61)
        self.assertEqual(len(set(expected)), len(expected))
        self.assertEqual(folded_env("RETAINED_CONSOLIDATION_BRANCHES"), expected)
        self.assertNotIn("main", expected)

    def check_guard_order(self, workflow):
        start = workflow.index('          for row in "${branch_rows[@]}"; do')
        loop = workflow[start:]
        guard = loop.index('if is_retained_consolidation "${branch}"; then')
        # Inspect the actual sole push, rather than binding guard order to the
        # obsolete unleased command. Reject additional/unguarded push owners.
        pushes = re.findall(r'^\s*(git [^\n]*\bpush\b[^\n]*)$', workflow, re.M)
        self.assertEqual(len(pushes), 1)
        deletion = pushes[0]
        self.assertEqual(shlex.split(deletion), [
            'git', '-C', '${graph}', 'push', '--quiet',
            '--force-with-lease=refs/heads/${branch}:${sha}',
            'origin', ':refs/heads/${branch}',
        ])
        for later in ('is_safe_prefix "${branch}"', 'is_explicitly_retired "${branch}"',
                      'merge-base --is-ancestor', 'closed_unmerged_at=', deletion):
            self.assertLess(guard, loop.index(later))
        self.assertIn('echo "skip consolidation retention: ${branch}"\n              continue', loop)
        for preflight in ('latest_sha=', 'latest_open_prs=', 'latest_active_runs=',
                          '"${latest_sha}" != "${sha}"',
                          '"${latest_open_prs}" != "0"',
                          '"${latest_active_runs}" != "0"'):
            self.assertLess(loop.index(preflight), loop.index(deletion))

    def test_guard_precedes_every_cleanup_eligibility_rule(self):
        self.check_guard_order(WORKFLOW)

    def test_rejects_unleased_or_early_deletion(self):
        with self.assertRaises(AssertionError):
            self.check_guard_order(WORKFLOW.replace(
                '--force-with-lease="refs/heads/${branch}:${sha}" ', ''))
        deletion = re.search(r'^            git [^\n]*\bpush\b[^\n]*$', WORKFLOW, re.M).group()
        early = WORKFLOW.replace(deletion + '\n', '')
        marker = '          for row in "${branch_rows[@]}"; do\n'
        early = early.replace(marker, marker + deletion + '\n')
        with self.assertRaises(AssertionError):
            self.check_guard_order(early)

    def test_actual_shell_guard_retains_old_closed_or_merged_heads(self):
        function = re.search(r"          is_retained_consolidation\(\) \{.*?\n          \}", WORKFLOW, re.S).group()
        guard = re.search(r'            if is_retained_consolidation "\$\{branch\}"; then.*?\n            fi', WORKFLOW, re.S).group()
        names = folded_env("RETAINED_CONSOLIDATION_BRANCHES")
        # Every retention name is tested with changed heads, old closed-unmerged
        # provenance and merged/retired eligibility. The real early continue must
        # stop the deletion-eligible suffix before any external command is needed.
        for reason in ("closed-unmerged-365-days", "merged-tip", "explicitly-retired"):
            script = 'set -eu\n' + function + '\nretained_consolidation=0\n'
            script += 'for branch in "$@"; do\n' + guard + '\nprintf "DELETE_ELIGIBLE:%s\\n" "$branch"\ndone\n'
            env = {"PATH": "/usr/bin:/bin", "RETAINED_CONSOLIDATION_BRANCHES": " ".join(names),
                   "sha": "b" * 40, "reason": reason}
            run = subprocess.run(["bash", "-c", script, "guard", *names, "research/not-in-snapshot"],
                                 env=env, text=True, capture_output=True, check=True)
            self.assertEqual(run.stdout.count("skip consolidation retention:"), 61)
            self.assertEqual([x for x in run.stdout.splitlines() if x.startswith("DELETE_ELIGIBLE:")],
                             ["DELETE_ELIGIBLE:research/not-in-snapshot"])


if __name__ == "__main__":
    unittest.main()
