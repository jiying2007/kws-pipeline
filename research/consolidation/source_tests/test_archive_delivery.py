"""Offline consistency checks for archive delivery; never run restored content."""
import hashlib
import json
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / 'research/consolidation'


class ArchiveDeliveryTests(unittest.TestCase):
    def test_evidence_pins(self):
        decision = json.loads((C / 'archive-delivery-decision-2026-10-09.json').read_text())
        for row in decision['evidence']:
            raw = (ROOT / row['path']).read_bytes()
            self.assertEqual(len(raw), row['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])

    def test_history_and_cancellation_are_separate(self):
        historical = json.loads((C / 'source-retention-publication-2026-10-09.json').read_text())
        decision = json.loads((C / 'archive-delivery-decision-2026-10-09.json').read_text())
        self.assertEqual(historical['expanded_pipeline_source_copy'], 'NOT_PUSHED')
        self.assertEqual(decision['expanded_copy']['historical_publication_status'], 'NOT_PUSHED')
        self.assertEqual(decision['expanded_copy']['delivery_target'], 'CANCELLED')
        self.assertFalse(decision['expanded_copy']['all_500_bytes_archived_claim'])

    def test_complete_publication_restore_receipt(self):
        r = json.loads((C / 'archive-restoration-acceptance-2026-10-09.json').read_text())
        self.assertEqual(r['status'], 'PASS_FULL_PUBLICATION_RESTORE')
        self.assertEqual(r['commit'], 'd9a65cc616cbb0e55a99f4c0e77c16e29bc6622a')
        self.assertEqual(r['archive']['sha256'], '8719a8ed67f440afe4adc450d47418523a7595dd02fb674c77da686964c84368')
        self.assertEqual(r['restore']['members_verified'], 1886)
        self.assertEqual(r['restore']['unique_objects_verified'], 1403)
        self.assertEqual(r['restore']['published_bytes_verified'], 223112990)
        for key in ('missing_paths', 'extra_paths', 'hash_mismatches', 'size_mismatches'):
            self.assertEqual(r['restore'][key], 0)
        self.assertEqual(r['tests']['passed'], 29)
        self.assertEqual(r['tests']['failed'], 0)
        self.assertEqual(r['tests']['skipped'], 0)
        self.assertTrue(r['scope']['publication_bytes_only'])
        self.assertFalse(r['scope']['restored_content_executed'])

    def test_all_500_are_classified_without_overclaiming(self):
        r = json.loads((C / 'archive-expanded-copy-reconciliation-2026-10-09.json').read_text())
        rows = r['files']
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({x['path'] for x in rows}), 500)
        self.assertEqual(r['unclassified'], 0)
        self.assertEqual(r['counts'], {
            'archive_exact_bytes': 471, 'public_data_sidecar_exact_bytes': 18,
            'navigation_projection_full_original_in_archive': 4,
            'legacy_delivery_derived_or_packaging': 7})
        for kind, count in r['counts'].items():
            self.assertEqual(sum(x['classification'] == kind for x in rows), count)
        self.assertEqual(r['exact_byte_retention_count'], 489)
        self.assertFalse(r['all_500_original_bytes_in_archive'])
        decision = json.loads((C / 'archive-delivery-decision-2026-10-09.json').read_text())
        self.assertEqual(r['counts'], decision['expanded_copy']['counts'])

    def test_historical_ci_identity_is_immutable(self):
        # A local, non-active snapshot also works in shallow/source-only checkouts.
        # The decision records a past change, never the hash of every future CI.
        d = json.loads((C / 'archive-delivery-decision-2026-10-09.json').read_text())
        ci = d['ci_change']
        self.assertEqual(ci['after']['sha256'], 'fe89f15dbfbf3b063f44e4b0cec00d19a47049f174aca92d8eb3c9c7d975b288')
        self.assertEqual(ci['workflow'], '.github/workflows/research-source-consolidation.yml')
        snapshot = C / 'history/ci/research-source-consolidation-archive-delivery-2026-10-09.yml.txt'
        raw = snapshot.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ci['after']['sha256'])
        self.assertEqual(len(raw), ci['after']['bytes'])
        blob = b'blob ' + str(len(raw)).encode() + b'\0' + raw
        self.assertEqual(hashlib.sha1(blob).hexdigest(), ci['after']['git_blob_sha1'])
        self.assertEqual(ci['after']['path'], ci['workflow'])
        self.assertNotIn(ROOT / '.github/workflows', snapshot.parents)

    def test_current_ci_is_offline_and_integrated(self):
        # Check current structure separately from historical byte identity.
        text = (ROOT / '.github/workflows/ci.yml').read_text()
        jobs = re.split(r'(?m)^  (?=[a-z][a-z0-9-]*:\s*$)', text.split('\njobs:\n', 1)[1])
        local = next(part for part in jobs if part.startswith('python-contracts:'))
        self.assertNotRegex(local, r'(?m)^    (?:if|needs):')
        step = next(part for part in local.split('      - name: ')
                    if part.startswith('Verify local research source retention and delivery\n'))
        self.assertIn('python3 -B research/consolidation/source_tests/test_archive_delivery.py', step)
        self.assertNotRegex(step, r'(?m)^        (?:if|continue-on-error):')
        for forbidden in ('--fetch', 'pip install', 'verify_archive.py', 'curl ', 'wget '):
            self.assertNotIn(forbidden, step)

    def test_product_qualification_unchanged(self):
        decision = json.loads((C / 'archive-delivery-decision-2026-10-09.json').read_text())
        self.assertEqual(decision['qualification'], {
            'D20_raw_numerical_gate': 'FAIL', 'D90': 'NOT_RUN',
            'human_and_device_qualification': 'DEFERRED', 'shipping_approved': False,
            'historical_experiment_reproduction': 'NOT_CLAIMED'})
        self.assertFalse(json.loads((ROOT / 'configs/shipping.xiaowo.json').read_text())['shipping_approved'])

    def test_navigation_points_to_current_delivery(self):
        for name in ('README.md', 'README.zh-CN.md', 'research/README.md'):
            self.assertIn('ARCHIVE_DELIVERY_2026-10-09.md', (ROOT / name).read_text())
        research = (ROOT / 'research/README.md').read_text()
        current = 'diagnostic-readiness-2026-10-09/PUBLIC-PROTOCOL.zh-CN.md'
        historical = 'saved-diagnostics-2026-10-09/REPORT.zh-CN.md'
        self.assertLess(research.index(current), research.index(historical))
        for target in (current, historical, 'diagnostic-readiness-2026-10-09/PUBLIC-READINESS-SUMMARY.json'):
            self.assertTrue((ROOT / 'research' / target).is_file())


if __name__ == '__main__':
    unittest.main()
