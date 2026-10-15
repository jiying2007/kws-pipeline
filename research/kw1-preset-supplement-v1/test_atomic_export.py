"""Transactional public export and independent verification, all synthetic."""
import copy
from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import export_public as exporter
import verify_public as verifier
import public_evidence as pe
from execution_common import canonical, sha
from test_archive_faults import make_work, put


def rehash(root):
    """Adversarial self-consistent manifest, not a trusted authorization anchor."""
    root = Path(root)
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(schema='kw1-public-package-v1', admission_sha256=sha(files['admission.json']),
                    plan_sha256=sha(files['plan.json']), payload_bytes=sum(map(len, files.values())),
                    files={name: dict(sha256=sha(raw), bytes=len(raw)) for name, raw in sorted(files.items())})
    put(root / 'manifest.json', manifest)


class AtomicExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.work = self.root / 'work'; self.output = self.root / 'public'

    def prepare(self, complete=True):
        self.plan, self.admission = make_work(self.work, complete)

    def publish(self):
        return exporter.export(self.plan, self.work, self.output, self.admission)

    def test_complete_16_plus_48_package_rebuild(self):
        self.prepare(); result = self.publish()
        self.assertTrue(result['verified']); self.assertTrue(result['identity_authenticated'])
        self.assertEqual(result['report_status'], 'completed'); self.assertEqual(result['ledger_slots'], 64)
        report = pe.json_bytes((self.output / 'report.json').read_bytes())
        self.assertEqual(report['consumed_calls'], {'tts': 16, 'asr': 48})
        self.assertEqual(report['target_qualified'], 16)
        self.assertIs(report['human_gold'], False); self.assertIs(report['training_admitted'], False)
        self.assertEqual(report['admission_status'], 'BLOCKED_NO_CALIBRATION_SELECTION_GOLD')
        self.assertFalse(verifier.verify(self.output)['identity_authenticated'])
        self.assertEqual(verifier.verify(self.output)['verification_scope'], 'internal_consistency_only')

    def test_finalization_ignores_damaged_report_and_private_bodies(self):
        self.prepare(); (self.work / 'report.json').write_bytes(b'CANARY_PRIVATE')
        (self.work / 'private.json').write_bytes(b'CANARY_PRIVATE')
        self.publish()
        self.assertNotIn(b'CANARY_PRIVATE', b''.join(p.read_bytes() for p in self.output.rglob('*') if p.is_file()))
        self.assertEqual((self.work / 'report.json').read_bytes(), b'CANARY_PRIVATE')

    def test_existing_complete_artifact_is_read_only(self):
        self.prepare(); self.publish()
        before = {p.relative_to(self.output).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
                  for p in self.output.rglob('*') if p.is_file()}
        (self.work / 'report.json').write_bytes(b'damaged derived report')
        result = self.publish(); self.assertTrue(result['verified'])
        after = {p.relative_to(self.output).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
                 for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_existing_foreign_admission_is_rejected_without_write(self):
        self.prepare(); self.publish(); original = (self.output / 'manifest.json').read_bytes()
        changed = copy.deepcopy(self.admission); changed['run_id'] += 1
        put(self.work / 'campaign.claim.json', changed)
        with self.assertRaises(ValueError): exporter.export(self.plan, self.work, self.output, changed)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), original)

    def test_reviewed_plan_source_hash_required(self):
        self.prepare(False); changed = copy.deepcopy(self.admission)
        changed['source_files']['research/kw1-preset-supplement-v1/plan.json'] = 'f' * 64
        put(self.work / 'campaign.claim.json', changed)
        with self.assertRaises(ValueError): exporter.export(self.plan, self.work, self.output, changed)
        self.assertFalse(self.output.exists())

    def test_existing_incomplete_output_never_replaced(self):
        self.prepare(False); self.output.mkdir(); (self.output / 'sentinel').write_bytes(b'keep')
        with self.assertRaises(ValueError): self.publish()
        self.assertEqual((self.output / 'sentinel').read_bytes(), b'keep')
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'sentinel'})

    def test_unknown_chains_export_only_metadata_reserved_64_slots(self):
        self.prepare(False)
        output = self.work / 'tts/qwen06/output'; output.mkdir(parents=True)
        (output / 'stage-claim.json').write_bytes(b'{"unknown":"CANARY_PRIVATE"}')
        result = self.publish(); self.assertEqual(result['report_status'], 'stopped_no_retry')
        report = pe.json_bytes((self.output / 'report.json').read_bytes())
        self.assertIsNone(report['consumed_calls']['tts']); self.assertIsNone(report['consumed_calls']['asr'])
        attempts = pe.json_bytes((self.output / 'ledger.json').read_bytes())
        self.assertEqual(len(attempts['slots']), 64)
        self.assertTrue(all(r['attempts'] is None and r['reserved_no_retry'] is True for r in attempts['slots']))
        self.assertNotIn(b'CANARY_PRIVATE', b''.join(p.read_bytes() for p in self.output.rglob('*') if p.is_file()))
        self.assertFalse((self.output / 'evidence/tts/qwen06/output/stage-claim.json').exists())

    def test_verify_failure_before_publish_cleans_temporary(self):
        self.prepare(False)
        with patch.object(verifier, 'verify', side_effect=ValueError('injected verifier failure')):
            with self.assertRaises(ValueError): self.publish()
        self.assertFalse(self.output.exists()); self.assertEqual(list(self.root.glob('.public.admission-*')), [])

    def test_publication_failure_cleans_temporary(self):
        self.prepare(False)
        with patch.object(exporter, 'atomic_publish', side_effect=OSError('injected publication failure')):
            with self.assertRaises(OSError): self.publish()
        self.assertFalse(self.output.exists()); self.assertEqual(list(self.root.glob('.public.admission-*')), [])

    def test_racing_incomplete_destination_is_never_replaced(self):
        self.prepare(False)
        def race(_source, dest):
            dest.mkdir(); (dest / 'sentinel').write_bytes(b'keep'); raise FileExistsError('race')
        with patch.object(exporter, 'atomic_publish', side_effect=race):
            with self.assertRaises(ValueError): self.publish()
        self.assertEqual((self.output / 'sentinel').read_bytes(), b'keep')
        self.assertEqual(list(self.root.glob('.public.admission-*')), [])

    def test_racing_complete_destination_is_verified(self):
        self.prepare(False)
        def race(source, dest):
            shutil.copytree(source, dest); raise FileExistsError('race')
        with patch.object(exporter, 'atomic_publish', side_effect=race):
            self.assertTrue(self.publish()['verified'])
        self.assertEqual(list(self.root.glob('.public.admission-*')), [])

    def test_kernel_no_replace_keeps_existing_directory(self):
        source, target = self.root / 'source', self.root / 'target'; source.mkdir(); target.mkdir()
        (source / 'new').write_bytes(b'new'); (target / 'old').write_bytes(b'old')
        with self.assertRaises(FileExistsError): exporter.atomic_publish(source, target)
        self.assertEqual((target / 'old').read_bytes(), b'old'); self.assertTrue((source / 'new').exists())

    def test_total_and_file_caps_fail_before_publish(self):
        self.prepare(False)
        for constant in ('MAX_TOTAL_BYTES', 'MAX_FILE_BYTES'):
            with self.subTest(cap=constant), patch.object(exporter, constant, 64):
                with self.assertRaises(ValueError): self.publish()
            self.assertFalse(self.output.exists())

    def test_strict_report_types_after_adversarial_rehash(self):
        self.prepare(); self.publish(); report_path = self.output / 'report.json'
        report = pe.json_bytes(report_path.read_bytes())
        for mutate in (lambda r: r['denominator'].__setitem__('design_slots', 16.0),
                       lambda r: r.__setitem__('human_gold', 0),
                       lambda r: r['slots'][0].__setitem__('native_attempts', True),
                       lambda r: r['slots'][0]['native_metrics'].__setitem__('clipped_fraction', 0)):
            altered = copy.deepcopy(report); mutate(altered); put(report_path, altered); rehash(self.output)
            with self.assertRaises(ValueError): verifier.verify(self.output, self.admission)
        put(report_path, report); rehash(self.output); self.assertTrue(verifier.verify(self.output)['verified'])

    def test_unknown_body_even_with_manifest_hash_is_forbidden(self):
        self.prepare(False); self.publish(); put(self.output / 'extra.json', {'secret': 'CANARY_PRIVATE'}); rehash(self.output)
        with self.assertRaises(ValueError): verifier.verify(self.output)

    def test_manifest_boolean_sizes_and_hardlinks_rejected(self):
        self.prepare(False); self.publish()
        manifest = pe.json_bytes((self.output / 'manifest.json').read_bytes())
        manifest['files']['report.json']['bytes'] = True; put(self.output / 'manifest.json', manifest)
        with self.assertRaises(ValueError): verifier.verify(self.output)
        rehash(self.output)
        os.link(self.output / 'report.json', self.root / 'linked-report')
        with self.assertRaises(ValueError): verifier.verify(self.output)

    def test_existing_symlink_and_private_admission_rejected(self):
        self.prepare(False); self.output.symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError): self.publish()
        self.output.unlink(); changed = dict(self.admission, secret='CANARY_PRIVATE'); put(self.work / 'campaign.claim.json', changed)
        with self.assertRaises(ValueError): self.publish()

    def test_known_resource_limits_preserved_unknown_memory_not_measured(self):
        self.prepare(False)
        from docker_backend import DockerBackend
        backend = DockerBackend.__new__(DockerBackend)
        backend._failed = False; backend._next = 3
        backend._summaries = [dict(stage=name, status='complete', attempt=1, wall_seconds=1.5)
                              for name in ('image', 'setup_qwen06', 'tts_qwen06')]
        backend._runtime_cleanups = [dict(runtime='runtime-qwen06', status='removed_after_container_cleanup',
            logical_regular_bytes=123, scope='only this campaign owned runtime; claims/results/source code retained')]
        put(self.work / 'backend/summary.json', backend.public_summary())
        self.publish(); report = pe.json_bytes((self.output / 'report.json').read_bytes())
        resources = report['resources']['container_evidence']
        self.assertTrue(resources['verified']); self.assertEqual(resources['body']['runtime_cleanups'][0]['logical_regular_bytes'], 123)
        stage = resources['body']['stages'][-1]
        self.assertEqual(stage['limits']['memory_bytes'], 12 * 1024**3)
        self.assertIsNone(stage['memory_observation']['process_peak_rss_kib'])
        self.assertIsNone(stage['memory_observation']['cgroup_memory_peak_bytes'])

    def test_complete_calls_without_final_cleanup_never_complete(self):
        self.prepare(); path = self.work / 'backend/summary.json'; original = path.read_bytes()
        cases = ('missing', 'malformed', 'failed_cleanup', 'incomplete_cleanups', 'incomplete_summary',
                 'unknown_exit', 'unknown_oom', 'unknown_host', 'unknown_configuration', 'wall_limit')
        for kind in cases:
            with self.subTest(kind=kind):
                if path.exists(): path.unlink()
                if kind == 'malformed': path.write_bytes(b'CANARY_PRIVATE')
                elif kind != 'missing':
                    value = pe.json_bytes(original)
                    if kind == 'failed_cleanup':
                        value['status'] = 'stopped_no_retry'; value['runtime_cleanups'][-1]['status'] = 'failed_no_retry'
                    elif kind == 'incomplete_cleanups': value['runtime_cleanups'].pop()
                    elif kind == 'incomplete_summary': value['status'] = 'started'
                    elif kind == 'unknown_exit': value['stages'][-1].pop('exit_code')
                    elif kind == 'unknown_oom': value['stages'][-1].pop('oom_killed')
                    elif kind == 'unknown_host': value['stages'][-1].pop('host')
                    elif kind == 'unknown_configuration': value['stages'][-1]['container_configuration_verified'] = False
                    elif kind == 'wall_limit': value['stages'][-1]['wall_seconds'] = 4800.1
                    else: raise AssertionError('unknown test case')
                    put(path, value)
                output = self.root / ('public-' + kind)
                result = exporter.export(self.plan, self.work, output, self.admission)
                self.assertEqual(result['report_status'], 'stopped_no_retry')
                report = pe.json_bytes((output / 'report.json').read_bytes())
                self.assertEqual(report['consumed_calls'], {'tts': 16, 'asr': 48})
                self.assertIn(report['stopped_reason'], ('container_resource_unverified', 'container_stage_or_cleanup_incomplete'))


if __name__ == '__main__':
    unittest.main()
