"""Public synthetic fixtures only; no old data replay."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from metadata import exposure, mask, parse_textgrid, checked_mask, validate_partition


def grid(tiers, hi='2'):
    text = f'File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = {hi}\ntiers? <exists>\nsize = {len(tiers)}\nitem []:\n'
    for i, rows in enumerate(tiers, 1):
        text += f'item [{i}]:\nclass = "IntervalTier"\nname = "synthetic"\nxmin = 0\nxmax = {hi}\nintervals: size = {len(rows)}\n'
        for j, (start, end, label) in enumerate(rows, 1):
            label = label.replace('"', '""')
            text += f'intervals [{j}]:\nxmin = {start}\nxmax = {end}\ntext = "{label}"\n'
    return text.encode()


class MetadataTests(unittest.TestCase):
    def run_mask(self, tiers, **kw):
        return mask(grid(tiers), 10, 25, kw.get('terms', ['小窝', '小屋']), {'legacy': 'pending'})

    def test_rounding_priority_and_tail(self):
        result = self.run_mask([[('0.01', '1.99', '普通对话')], [('0.51', '0.91', '小屋')]])
        self.assertEqual([(x['start_sample'], x['end_sample'], x['category']) for x in result['intervals']], [
            (0, 1, 'unknown'), (1, 5, 'transcript_supported_candidate'),
            (5, 10, 'target_or_near_target_review'), (10, 19, 'transcript_supported_candidate'),
            (19, 25, 'unknown')])
        for key in ('human_gold', 'negative_admitted', 'audio_verified_this_run'):
            self.assertIs(result[key], False)
        self.assertIsNone(result['expected'])
        self.assertEqual(result['historical_pending'], {'legacy': 'pending'})
        self.assertEqual(result['new_independent_hours'], 0)
        self.assertEqual(result['accepted_negative_seconds'], 0)

    def test_no_cross_tier_or_segment_join_or_homophone_merge(self):
        result = self.run_mask([[('0', '1', '小'), ('1', '2', '屋')], [('0', '2', '窝屋')]], terms=['小屋', '屋屋'])
        self.assertEqual(result['intervals'][0]['category'], 'transcript_supported_candidate')
        self.assertEqual(result['cross_segment_search'], 'not_performed')
        self.assertFalse(result['screening_complete'])

    def test_blank_marker_and_gap_are_unknown(self):
        result = self.run_mask([[('0', '.2', ''), ('.4', '.6', '(SIL)'), ('.7', '.8', '...')]])
        self.assertEqual(result['intervals'], [dict(start_sample=0, end_sample=25, category='unknown')])

    def test_quoted_structural_text_does_not_split_parser(self):
        data = grid([[('0', '2', 'quoted "word"\nitem [7]:\nintervals [8]:')]])
        self.assertEqual(len(parse_textgrid(data)[1]), 1)
        self.assertEqual(len(parse_textgrid(b'\xff\xfe' + data.decode().encode('utf-16-le'))[1]), 1)

    def test_reject_invalid_timing_and_structure(self):
        for rows in ([('1', '.1', 'a')], [('0', '1', 'a'), ('.9', '2', 'b')], [('0', '3', 'a')], [('0', '0', 'a')]):
            with self.assertRaises(ValueError):
                parse_textgrid(grid([rows]))
        with self.assertRaises(ValueError):
            parse_textgrid(grid([[('0', '2', 'a')]]) + b'junk')
        with self.assertRaises(ValueError):
            mask(grid([[('0', '2', 'a')]]), 10, 19, ['x'], {})

    def test_review_term_inside_marker_is_still_review(self):
        result = self.run_mask([[('0', '2', '(小屋)')]])
        self.assertEqual(result['intervals'][0]['category'], 'target_or_near_target_review')

    def test_decimal_boundary_exact(self):
        result = self.run_mask([[('0.1', '0.3', 'abc')]])
        self.assertEqual(result['intervals'][1], dict(start_sample=1, end_sample=3, category='transcript_supported_candidate'))

    def test_partition_many_boundaries(self):
        for k in range(1, 99):
            result = self.run_mask([[('0.001', str(k / 100), 'abc')], [('0.99', '1.01', '小窝')]])
            rows = result['intervals']
            self.assertEqual(rows[0]['start_sample'], 0)
            self.assertEqual(rows[-1]['end_sample'], 25)
            self.assertTrue(all(a['end_sample'] == b['start_sample'] for a, b in zip(rows, rows[1:])))
            self.assertTrue(all(a['start_sample'] < a['end_sample'] for a in rows))

    def test_mixed_marker_blocks_overlapping_support(self):
        result = self.run_mask([[('0', '2', '普通(未知)文字')], [('0', '2', '另一说话人')]])
        self.assertEqual(result['intervals'], [dict(start_sample=0, end_sample=25, category='unknown')])

    def test_extended_unknown_markers(self):
        for text in ['{UNK}', '普通［未知］文字', 'UNK', 'unk', 'UNKNOWN', '未知', '〈未知〉']:
            result = self.run_mask([[('0', '2', text)], [('0', '2', '普通话')]])
            self.assertEqual(result['intervals'], [dict(start_sample=0, end_sample=25, category='unknown')])
        self.assertEqual(self.run_mask([[('0', '2', 'TRUNK')]])['intervals'][0]['category'], 'transcript_supported_candidate')

    def test_checked_identity_and_partition(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'S1.TextGrid'
            data = grid([[('0', '2', '普通话')]])
            path.write_bytes(data)
            contract = dict(session_id='S1', source_family='synthetic',
                textgrid_sha256=hashlib.sha256(data).hexdigest(), sample_rate=10,
                num_samples=25, review_terms=['小窝'], historical_pending={})
            self.assertEqual(checked_mask(path, contract)['session_id'], 'S1')
            for key, value in [('session_id', 'S2'), ('textgrid_sha256', '0' * 64), ('source_family', '')]:
                with self.assertRaises(ValueError):
                    checked_mask(path, dict(contract, **{key: value}))
        for rows in [[], [dict(start_sample=1, end_sample=25, category='unknown')],
                     [dict(start_sample=0, end_sample=24, category='unknown')]]:
            with self.assertRaises(ValueError):
                validate_partition(rows, 25)

    def test_session_channel_suffix_is_contact_clue(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source.json'
            path.write_text(json.dumps({'paths': ['S1_M2_CH1.wav', 'S1_M20_CH1.wav']}))
            result = exposure(['S1_M2', 'S1_M'], [path])
            self.assertEqual(len(result['candidates'][0]['scanned_ledger_matches']), 1)
            self.assertEqual(result['candidates'][1]['scanned_ledger_matches'], [])

    def test_absence_and_positive_clues_never_establish_independence(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'synthetic.json'
            path.write_text(json.dumps({'sources': ['/invented/R1_M2.wav', 'R1_M20']}))
            result = exposure(['R1_M2', 'R1_M3'], [path])
            self.assertEqual(len(result['candidates'][0]['scanned_ledger_matches']), 1)
            self.assertEqual(result['candidates'][1]['scanned_ledger_matches'], [])
            for row in result['candidates']:
                self.assertEqual(row['historical_exposure'], 'unknown')
                self.assertEqual(row['pretraining_exposure'], 'unknown')
                self.assertEqual(row['selection_independent'], 'unknown')
                self.assertFalse(row['negative_admitted'])
        self.assertEqual(exposure(['R1_M2'], [])['candidates'][0]['historical_exposure'], 'unknown')


if __name__ == '__main__':
    unittest.main()
