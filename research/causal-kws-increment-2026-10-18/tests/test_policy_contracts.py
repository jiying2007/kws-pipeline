"""Anonymous schema contracts; no source rows, recognizers or training data."""
import json
import unittest
from pathlib import Path

POLICY = json.loads((Path(__file__).resolve().parents[1] / 'policy' / 'research-tiers.json').read_text())

class PolicyContracts(unittest.TestCase):
    def test_evidence_tiers_remain_distinct(self):
        self.assertEqual(list(POLICY['aggregate_counts'].values()), [8, 3, 1])
        self.assertEqual(sum(POLICY['aggregate_counts'].values()), 12)
        self.assertEqual(len(POLICY['rules']), 3)
        self.assertEqual(POLICY['status'], 'POST_EVIDENCE_POLICY_NO_TRAINING')

    def test_majority_is_only_weak_candidate(self):
        r = POLICY['rules'][1]
        self.assertEqual(r['original_admission'], 'quarantine')
        self.assertEqual(r['preexisting_split'], 'train_voice_lineage_only')
        self.assertEqual(r['suggested_research_weight'], .1)
        self.assertFalse(r['weight_is_probability'])

    def test_abstention_does_not_create_negative_label(self):
        r = POLICY['rules'][2]
        self.assertEqual(r['suggested_research_weight'], 0.)
        self.assertFalse(r['negative_relabel_allowed'])
        for r in POLICY['rules']:
            self.assertFalse(r['human_gold'])
            self.assertFalse(r['final_acceptance_eligible'])
        self.assertTrue(all(POLICY['invariants'].values()))

    def test_abstract_vote_fixture_does_not_override_original_admission(self):
        # A vote tally is evidence only. No result authorizes a new label or split.
        fixtures = [('sealed-weak', ('T', 'T'), 2),
                    ('quarantine', ('T', 'T', 'D'), 2),
                    ('quarantine', ('D', 'D', 'D'), 0)]
        for admission, votes, matches in fixtures:
            self.assertEqual(sum(v == 'T' for v in votes), matches)
            self.assertIn(admission, ('sealed-weak', 'quarantine'))
        self.assertTrue(POLICY['invariants']['original_labels_unchanged'])
        self.assertTrue(POLICY['invariants']['train_dev_lineage_unchanged'])

if __name__ == '__main__':
    unittest.main()
