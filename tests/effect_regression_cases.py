"""No-model cases consumed by the official test_eval.py entry point."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'tools')]
import effect_evidence as evidence
import far_regression_receipt as gate


def result_fixture():
    return {'rows': [{'cell_id': cell, 'family': family, 'audio_id': 'audio-' + cell,
                      'wav_sha256': 'a' * 64, 'pcm_sha256': 'b' * 64, 'audio_frames': 16000,
                      'status': 'complete', 'raw_transcript': '你好，小屋。', 'normalized_transcript': '你好小屋'}
                     for cell in evidence.CELLS for family in evidence.FAMILIES]}


def saved_fixture(root):
    root = Path(root)
    folder = root / 'far-regression'; folder.mkdir()
    for name in gate.CONFIGS:
        shutil.copyfile(ROOT / 'configs' / name, folder / name)
    (folder / 'kws_raw_stream').write_bytes(b'fake runner: not executable')
    gate.write(folder / 'source-binding.json', {'schema': 'far-runtime-source-binding-v1', 'source_sha': 'a' * 40,
                'files': {'tools/kws_raw_stream.c': {'git_blob_sha1': 'b' * 40, 'sha256': 'c' * 64}}})
    (root / 'nightly-negative.tsv').write_text('fixture-only\n')
    model = gate.load(folder / 'shipping.xiaowo.json')['model']
    hashes = sorted(gate.KNOWN_FA_SOURCES)
    domain = [{'kind': 'confusable', 'wav_sha256': digest} for digest in hashes]
    domain_dir = root / 'nightly-domain'; domain_dir.mkdir()
    (domain_dir / 'domain-index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in domain))
    inputs = []
    for seed in gate.SEEDS:
        folder = root / f'far-run-{seed}'; folder.mkdir()
        (folder / 'detections.jsonl').write_text('')
        (folder / 'captures.jsonl').write_text('')
        schedule = [{'source_sha256': hashes[i % 2], 'coverage_required': True, 'gain': 0.9} for i in range(200)]
        (folder / 'hard-negative-injections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in schedule))
        row = dict(schema_version=2, seed=seed, seconds=7200, audio_hours=2.0, false_accepts=0,
                   full_negative_manifest_coverage=True, coverage_hard_negative_gain=0.9,
                   min_hard_negative_injections_per_clip=1, hard_negative_rate_per_minute=2.0,
                   hard_negative_injections=200, hard_negative_audio_seconds=200.0,
                   runner_sha256=gate.sha(root / 'far-regression/kws_raw_stream'),
                   model_sha256=model['model_sha256'], keyword_pack_sha256=model['keyword_pack_sha256'],
                   negative_manifest_sha256=gate.sha(root / 'nightly-negative.tsv'),
                   detections_sha256=gate.sha(folder / 'detections.jsonl'), captures_sha256=gate.sha(folder / 'captures.jsonl'),
                   hard_negative_injections_sha256=gate.sha(folder / 'hard-negative-injections.jsonl'))
        gate.write(folder / 'summary.json', row)
        inputs.append({'path': str(folder / 'summary.json'), 'sha256': gate.sha(folder / 'summary.json')})
    aggregate = dict(shards=4, seeds=list(gate.SEEDS), audio_hours=8.0, false_accepts=0,
                     far_per_hour=0.0, far_upper_bound_per_hour=gate.poisson_rate_upper(0, 8.0, 0.95),
                     confidence_level=0.95, max_far_per_hour=0.0, max_upper_bound_per_hour=0.4,
                     min_hard_negative_injections=800, min_hard_negative_audio_seconds=800.0,
                     hard_negative_injections=800, hard_negative_audio_seconds=800.0,
                     hard_negative_rate_per_minute=2.0, qualified=True, inputs=inputs)
    for key in ('runner_sha256', 'model_sha256', 'keyword_pack_sha256', 'negative_manifest_sha256'):
        aggregate[key] = row[key]
    gate.write(root / 'far-aggregate.json', aggregate)
    return root


class EffectEvidenceTests(unittest.TestCase):
    def test_retained_original_prefix_diagnostic_is_failed_evidence_not_qualification(self):
        root = ROOT / 'research/known-far-diagnostic-v1'
        result = json.loads((root / 'result.json').read_text())
        self.assertFalse(result['qualification_allowed'])
        self.assertFalse(result['shipping_approved'])
        self.assertFalse(result['human_gold'])
        self.assertEqual(result['original_run_id'], 38482297930)
        self.assertEqual(result['negative_wav_sha256_verified'], 224)
        self.assertEqual(result['domain_rows_identical_except_local_paths'], 240)
        self.assertEqual(hashlib.sha256((root / 'raw_stream_trace_diagnostic.c').read_bytes()).hexdigest(), result['trace_source_sha256'])
        total = 0
        for member in result['trace_members']:
            raw = (root / member['path']).read_bytes()
            self.assertEqual(len(raw), member['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), member['sha256'])
            decoded = gzip.decompress(raw)
            self.assertEqual(len(decoded), member['uncompressed_bytes'])
            self.assertEqual(hashlib.sha256(decoded).hexdigest(), member['uncompressed_sha256'])
            rows = [json.loads(line) for line in decoded.splitlines()]
            self.assertEqual(len(rows), member['frames'])
            self.assertEqual([r['end_sample'] for r in rows if r['detected']], member['detected_end_samples'])
            self.assertEqual(sum(r['detected'] for r in rows), 1)
            self.assertTrue(all(len(r['logits']) == 5 for r in rows))
            total += len(rows)
        self.assertEqual(total, 1051)
        self.assertEqual([r['detections'][0]['time_s'] for r in result['stream_reconstruction']], [7191.345, 2372.645])
        self.assertEqual(result['cold_start_rows'][0]['detections'], [])
        for member in result['original_metadata_members']:
            raw = (root / member['path']).read_bytes()
            self.assertEqual(len(raw), member['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), member['sha256'])
            decoded = gzip.decompress(raw)
            self.assertEqual(len(decoded), member['uncompressed_bytes'])
            self.assertEqual(hashlib.sha256(decoded).hexdigest(), member['uncompressed_sha256'])


    def test_fixed_denominator_and_no_roof_relaxation(self):
        summary = evidence.summarize(result_fixture())
        self.assertEqual((summary['denominator'], summary['complete_slots'], summary['target_consensus_count']), (6, 18, 0))
        self.assertEqual(summary['policy_accepted_consensus_count'], 0)
        self.assertFalse(summary['human_gold'])

    def test_missing_cell_stays_in_denominator(self):
        data = result_fixture(); data['rows'] = data['rows'][3:]
        summary = evidence.summarize(data)
        self.assertEqual(summary['denominator'], 6)
        self.assertEqual(summary['complete_slots'], 15)
        self.assertEqual(summary['disagreement_or_incomplete_count'], 1)

    def test_duplicate_and_foreign_slot_rejected(self):
        data = result_fixture(); data['rows'].append(data['rows'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'): evidence.summarize(data)
        data = result_fixture(); data['rows'][0]['cell_id'] = 'other'
        with self.assertRaises(ValueError): evidence.summarize(data)

    def test_normalization_cannot_replace_roof(self):
        data = result_fixture(); data['rows'][0]['normalized_transcript'] = '你好小窝'
        with self.assertRaisesRegex(ValueError, 'normalization'): evidence.summarize(data)

    def test_explicit_policy_does_not_grant_gold(self):
        policy = dict(evidence.DEFAULT_POLICY, policy_id='fixture-roof-diagnostic', accepted_alternatives=['你好小屋'])
        summary = evidence.summarize(result_fixture(), policy)
        self.assertEqual(summary['policy_accepted_consensus_count'], 6)
        self.assertEqual(summary['target_consensus_count'], 0)
        self.assertFalse(summary['training_admitted'])

    def test_listening_template_has_no_truth_or_asr_hints(self):
        template = evidence.listening_template(evidence.summarize(result_fixture()))
        self.assertEqual(len(template['records']), 6)
        for row in template['records']:
            self.assertIsNone(row['heard_text']); self.assertIsNone(row['acoustic_complete'])
            self.assertEqual(row['status'], 'pending'); self.assertFalse(row['human_gold'])
        self.assertNotIn('你好', json.dumps(template, ensure_ascii=False))

    def test_byte_disjoint_does_not_imply_family_disjoint(self):
        report = evidence.separation([dict(file_sha256='a', pcm_sha256='b')], [dict(file_sha256='c', pcm_sha256='d')])
        self.assertEqual(report['file_sha256']['status'], 'verified_disjoint')
        self.assertEqual(report['source_family_id']['status'], 'unknown')
        self.assertFalse(report['fully_disjoint'])

    def test_known_overlap_wins_even_with_missing_rows(self):
        report = evidence.separation([dict(speaker_id='same'), {}], [dict(speaker_id='same')])
        self.assertEqual(report['speaker_id']['status'], 'overlap')


class RegressionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = saved_fixture(self.temp.name)

    def test_recompute_saved_pass_without_model(self):
        result = gate.evaluate(self.root)
        self.assertTrue(result['qualified']); self.assertFalse(result['product_qualification_allowed'])

    def test_missing_raw_or_stale_pass_blocks(self):
        (self.root / 'far-run-3301/detections.jsonl').unlink()
        receipt = gate.produce(self.root, 'a' * 40, 1, 1)
        self.assertEqual(receipt['status'], 'unknown'); self.assertFalse(receipt['qualified'])

    def test_mutated_config_and_wrong_tuple_block(self):
        for path in ['far-regression/parameter-contract.json', 'far-regression/kws_raw_stream']:
            p = self.root / path; original = p.read_bytes(); p.write_bytes(original + b' ')
            with self.assertRaises(ValueError): gate.evaluate(self.root)
            p.write_bytes(original)

    def test_known_failure_source_missing_blocks(self):
        p = self.root / 'nightly-domain/domain-index.jsonl'; p.write_text('')
        with self.assertRaisesRegex(ValueError, 'applicability unknown'): gate.evaluate(self.root)

    def test_detection_tampering_blocks(self):
        (self.root / 'far-run-1103/detections.jsonl').write_text('{"keyword_id":2}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): gate.evaluate(self.root)

    def test_aggregate_verdict_not_trusted(self):
        p = self.root / 'far-aggregate.json'; data = gate.load(p); data['false_accepts'] = 2; gate.write(p, data)
        with self.assertRaisesRegex(ValueError, 'recomputed'): gate.evaluate(self.root)

    def test_receipt_attempt_and_source_are_bound(self):
        receipt = gate.produce(self.root, 'a' * 40, 1, 1); gate.write(self.root / 'far-regression/receipt.json', receipt)
        deployment = dict(source_sha='a' * 40, model_release={key: receipt['identity'][key] for key in ['model_sha256', 'keyword_pack_sha256']},
                          shipping_contract_sha256=receipt['identity']['shipping_contract_sha256'], nightly_contract_sha256=receipt['identity']['nightly_contract_sha256'])
        self.assertTrue(gate.verify_download(self.root, deployment, dict(id=1, run_attempt=1))['qualified'])
        with self.assertRaisesRegex(ValueError, 'attempt mismatch'): gate.verify_download(self.root, deployment, dict(id=1, run_attempt=2))

    def test_full_archive_digest_and_path_safety(self):
        p = self.root / 'artifact.zip'
        with zipfile.ZipFile(p, 'w') as out: out.writestr('../escape', 'bad')
        with self.assertRaisesRegex(ValueError, 'digest'): gate.extract_verified(p, 'sha256:' + '0' * 64, self.root / 'out')
        with self.assertRaisesRegex(ValueError, 'unsafe'): gate.extract_verified(p, 'sha256:' + gate.sha(p), self.root / 'out')

    def test_latest_failure_never_uses_old_success(self):
        old = dict(id=1, workflow_id=42, path=gate.WORKFLOW, repository={'full_name':'owner/repo'}, head_repository={'full_name':'owner/repo'},
                   head_sha='a' * 40, head_branch='main', run_number=1, run_attempt=1, event='schedule', status='completed', conclusion='success', updated_at='t1')
        new = dict(old, id=2, run_number=2, conclusion='failure', updated_at='t2')
        wf = dict(id=42, path=gate.WORKFLOW)
        pages = [{'total_count':2, 'workflow_runs':[old]}, {'total_count':2, 'workflow_runs':[new]}]
        with self.assertRaisesRegex(ValueError, 'older-green'): gate.select_run(pages, wf, 'owner/repo', 'a' * 40)
        new.update(conclusion='success')
        self.assertEqual(gate.select_run(pages, wf, 'owner/repo', 'a' * 40)['id'], 2)
        with self.assertRaisesRegex(ValueError, 'pagination'): gate.select_run(pages[:1], wf, 'owner/repo', 'a' * 40)
        new['head_repository'] = {'full_name':'attacker/repo'}
        with self.assertRaisesRegex(ValueError, 'scope'): gate.select_run(pages, wf, 'owner/repo', 'a' * 40)

    def test_shipping_and_producer_actually_consume_gate(self):
        nightly = (ROOT / gate.WORKFLOW).read_text()
        shipping = (ROOT / '.github/workflows/shipping-approval.yml').read_text()
        self.assertIn('      actions: read', shipping)
        self.assertIn('far_regression_receipt.py produce', nightly)
        self.assertIn('build/far-regression/', nightly)
        self.assertLess(shipping.index('far_regression_receipt.py fetch'), shipping.index('Assemble terminal shipping approval manifest'))
        self.assertLess(shipping.index('far_regression_receipt.py recheck'), shipping.index('gh release create "$TAG"'))
        self.assertIn("'current_tuple_regression':", shipping)


def run():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (EffectEvidenceTests, RegressionGateTests))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('effect/regression contract tests failed')
    return result.testsRun


if __name__ == '__main__':
    run()
