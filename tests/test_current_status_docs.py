#!/usr/bin/env python3
"""Small, offline navigation contract; no acoustic or live-CI validation."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
STATUS = 'docs/KWS_LANDING_STATUS.md'
ANCHOR = '#current-model-and-engineering-status'

class CurrentStatusDocs(unittest.TestCase):
    def text(self, path):
        return (ROOT / path).read_text(encoding='utf-8')

    def test_navigation_converges(self):
        for path, target in (
            ('README.md', STATUS + ANCHOR),
            ('README.zh-CN.md', STATUS + ANCHOR),
            ('docs/README.md', 'KWS_LANDING_STATUS.md' + ANCHOR),
            ('research/README.md', '../' + STATUS + ANCHOR),
        ):
            with self.subTest(path=path):
                self.assertIn('](' + target + ')', self.text(path))
        self.assertEqual(self.text(STATUS).count('## Current model and engineering status'), 1)

    def test_scope_and_failure_boundaries(self):
        current = self.text(STATUS).split('## Current model and engineering status\n', 1)[1].split('### Integration receipts', 1)[0]
        for term in ('Historical five-token SKU', 'Original FSMN B', 'Conservative OR',
                     'New eleven-record increment', 'Posterior-only C decoder',
                     'shipping_approved=false', 'recalibration_required=true',
                     'neither new head is promoted', 'not an independent validation win',
                     'atol=rtol=1e-5', '**FAIL**', 'chain is not integrated',
                     'SSC305/device qualification has not passed'):
            with self.subTest(term=term):
                self.assertIn(term, current)
        self.assertIn('历史五 token SKU', current)
        self.assertIn('两个新头均不晋升', current)
        self.assertIn('OR 链未集成', current)

    def test_receipts_are_merge_scoped_and_history_is_marked(self):
        text = self.text(STATUS)
        self.assertIn('push/main merge-commit runs', text)
        self.assertIn('not PR-head runs or new', text)
        self.assertIn('do not reclassify failed numerical evidence', text)
        self.assertIn('## 2026-10-16 data/model roadmap checkpoint', text)
        self.assertIn('## 2026-10-14 current source entry', text)  # Keep old links valid.
        research = self.text('research/causal-kws-increment-2026-10-18/README.md')
        self.assertIn('## Later integration status (2026-10-18)', research)
        self.assertIn('historical preparation', research)
        self.assertIn('Integration and remote CI are still pending.', research)  # Frozen preparation wording.
        self.assertIn('## Historical preparation snapshot', research)

if __name__ == '__main__':
    unittest.main()
