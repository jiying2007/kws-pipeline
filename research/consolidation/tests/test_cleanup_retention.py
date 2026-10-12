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

    def test_exact_separate_execution_roster(self):
        expected = [
            "research/qwen12-native-admission-v1",
            "research/qwen12-native-admission-recovery1",
            "research/qwen12-native-transport-preflight-v1",
            "research/qwen12-native-admission-recovery2",
            "research/qwen12-native-transport-preflight-12m-v1",
            "research/qwen12-native-admission-recovery3",
            "research/qwen12-native-admission-remaining35-v1",
            "research/qwen12-native-admission-remaining35-r1-v1",
            "research/k1-punctuation6-asr18-v1",
        ]
        self.assertEqual(folded_env("RETAINED_EXECUTION_BRANCHES"), expected)
        self.assertTrue(set(expected).isdisjoint(folded_env("RETAINED_CONSOLIDATION_BRANCHES")))
        loop = WORKFLOW[WORKFLOW.index('          for row in "${branch_rows[@]}"; do'):]
        guard = loop.index('if is_retained_execution "${branch}"; then')
        for suffix in ('is_retained_consolidation "${branch}"', 'is_safe_prefix "${branch}"',
                       'is_explicitly_retired "${branch}"', 'merge-base --is-ancestor',
                       'closed_prs_json=', ' push '):
            self.assertLess(guard, loop.index(suffix))
        self.assertIn('echo "skip execution retention: ${branch}"\n              continue', loop)
        self.assertIn('retained_execution=${retained_execution}', loop)

    def cleanup_loop(self, branch, scenario, conclusion):
        # Execute the real workflow functions and complete branch loop. The only
        # replacements are external GitHub/Git commands; no network or real
        # repository deletion is reachable. Real jq parses the mocked API data.
        functions = WORKFLOW[WORKFLOW.index('          is_safe_prefix() {'):
                             WORKFLOW.index('          dry_run=false')]
        loop = WORKFLOW[WORKFLOW.index('          for row in "${branch_rows[@]}"; do'):]
        mocks = r'''
          gh() {
            echo "GH_QUERY:$*" >&2
            case "$*" in
              *actions/runs*)
                printf '[{"total_count":1,"workflow_runs":[{"id":1,"head_branch":"%s","status":"completed","conclusion":"%s"}]}]\n' "$TEST_BRANCH" "$CONCLUSION" ;;
              *state=open*) echo 0 ;;
              *state=closed*)
                case "$SCENARIO" in
                  merged-pr) printf '[{"merged_at":"2020-01-01T00:00:00Z","head":{"sha":"%s"}}]\n' "$SHA" ;;
                  closed-pr) printf '[{"merged_at":null,"closed_at":"2020-01-01T00:00:00Z","head":{"sha":"%s"}}]\n' "$SHA" ;;
                  *) echo '[]' ;;
                esac ;;
              *) return 90 ;;
            esac
          }
          git() {
            echo "GIT_QUERY:$*" >&2
            case "$*" in
              *show-ref*) return 0 ;;
              *merge-base*) [[ "$SCENARIO" == ancestor ]] ;;
              ls-remote*) printf '%s\trefs/heads/%s\n' "$SHA" "$TEST_BRANCH" ;;
              *" push "*) printf 'MOCK_DELETE:%s\n' "$TEST_BRANCH" ;;
              *) return 91 ;;
            esac
          }
          default_branch=main owner=owner graph=unused remote=unused
          default_ref=refs/remotes/origin/main dry_run=false
          deleted=0 candidates=0 manual_review=0 retained_design=0
          retained_consolidation=0 retained_execution=0
          branch_rows=("$(printf '%s\t%s\tfalse' "$TEST_BRANCH" "$SHA")")
          if [[ "$SCENARIO" == retired ]]; then RETIRED_BRANCHES="$TEST_BRANCH"; fi
        '''
        env = {
            "PATH": "/usr/bin:/bin", "GITHUB_REPOSITORY": "fixture/fixture",
            "SAFE_BRANCH_PREFIXES": re.search(r'^  SAFE_BRANCH_PREFIXES: "([^"\n]+)"', WORKFLOW, re.M).group(1),
            "RETIRED_BRANCHES": "",
            "RETAINED_CONSOLIDATION_BRANCHES": " ".join(folded_env("RETAINED_CONSOLIDATION_BRANCHES")),
            "RETAINED_EXECUTION_BRANCHES": " ".join(folded_env("RETAINED_EXECUTION_BRANCHES")),
            "RETAINED_DESIGN_BRANCHES": " ".join(folded_env("RETAINED_DESIGN_BRANCHES")),
            "CLOSED_PR_RETENTION_DAYS": "7", "SHA": "b" * 40,
            "TEST_BRANCH": branch, "SCENARIO": scenario, "CONCLUSION": conclusion,
        }
        return subprocess.run(["bash", "-euo", "pipefail", "-c", functions + mocks + loop],
                              env=env, text=True, capture_output=True, check=True)

    def test_execution_retention_precedes_all_actual_deletion_paths(self):
        for branch in folded_env("RETAINED_EXECUTION_BRANCHES"):
            for scenario in ("retired", "ancestor", "merged-pr", "closed-pr"):
                for conclusion in ("failure", "success"):
                    with self.subTest(branch=branch, scenario=scenario, conclusion=conclusion):
                        run = self.cleanup_loop(branch, scenario, conclusion)
                        self.assertIn("skip execution retention: " + branch, run.stdout)
                        self.assertIn("candidates=0 deleted=0", run.stdout)
                        self.assertIn("retained_execution=1", run.stdout)
                        self.assertNotIn("MOCK_DELETE:", run.stdout)
                        self.assertEqual(run.stderr, "")

    def test_unretained_branches_still_follow_actual_deletion_paths(self):
        # Exact matching must not silently retain future similarly named branches.
        controls = ["fix/ordinary-ephemeral", "research/qwen12-native-admission-v1-extra",
                    "research/qwen12-native-admission-recovery10",
                    "research/qwen12-native-admission-recovery20",
                    "research/qwen12-native-admission-recovery2-extra",
                    "research/qwen12-native-admission-recovery30",
                    "research/qwen12-native-admission-recovery3-extra",
                    "research/qwen12-native-admission-remaining35-v10",
                    "research/qwen12-native-admission-remaining35-v1-extra",
                    "research/qwen12-native-admission-remaining35-r1-v10",
                    "research/qwen12-native-admission-remaining35-r1-v1-extra",
                    "research/qwen12-native-transport-preflight-12m-v10",
                    "research/qwen12-native-transport-preflight-12m-v1-extra",
                    "research/qwen12-native-transport-preflight-v10",
                    "research/k1-punctuation6-asr18-v10",
                    "research/k1-punctuation6-asr18-v1-extra",
                    "research/k1-punctuation6-asr18-v2"]
        for branch in controls:
            for scenario in ("retired", "ancestor", "merged-pr", "closed-pr"):
                with self.subTest(branch=branch, scenario=scenario):
                    run = self.cleanup_loop(branch, scenario, "failure")
                    self.assertIn("MOCK_DELETE:" + branch, run.stdout)
                    self.assertIn("candidates=1 deleted=1", run.stdout)
                    self.assertIn("retained_execution=0", run.stdout)
                    self.assertIn("--force-with-lease=refs/heads/" + branch + ":" + "b" * 40, run.stderr)
        run = self.cleanup_loop("research/no-exact-closed-provenance", "diverged", "failure")
        self.assertIn("manual_review=1", run.stdout)
        self.assertNotIn("MOCK_DELETE:", run.stdout)

    def test_evidence_refs_remain_outside_cleanup_scope(self):
        for branch in ("evidence/qwen12-native-admission-v1",
                       "evidence/qwen12-native-admission-recovery1",
                       "evidence/qwen12-native-admission-recovery2",
                       "evidence/qwen12-native-admission-recovery3",
                       "evidence/qwen12-native-transport-preflight-v1",
                       "evidence/k1-punctuation6-asr18-v1"):
            for scenario in ("retired", "ancestor", "merged-pr", "closed-pr"):
                with self.subTest(branch=branch, scenario=scenario):
                    run = self.cleanup_loop(branch, scenario, "failure")
                    self.assertIn("candidates=0 deleted=0", run.stdout)
                    self.assertEqual(run.stderr, "")



if __name__ == "__main__":
    unittest.main()
