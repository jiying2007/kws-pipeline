"""Synthetic contract and filesystem fault checks only; never a release."""
import copy
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import asr18_metadata as metadata
import campaign_contract as c


def invented_projection():
    rows = []
    for index, cell in enumerate(metadata.CELLS, 1):
        rows.append({'cell_id': cell, 'status': 'complete', 'attempts': 1,
            'wav_sha256': c.digest(('invented-wav-' + str(index)).encode()),
            'pcm_sha256': c.digest(('invented-pcm-' + str(index)).encode()),
            'raw_wav_sha256': c.digest(('invented-raw-wav-' + str(index)).encode()),
            'raw_float_sha256': c.digest(('invented-raw-float-' + str(index)).encode()),
            'source_frames': 1920, 'frames': 1280, 'sample_rate_hz': 16000,
            'channels': 1, 'sample_width_bytes': 2, 'eos_verified': True,
            'generation_iterations': 2, 'quality_flags': ['silent'] if index == 2 else []})
    return metadata.screen_metadata(rows, [{'wav_sha256': rows[2]['wav_sha256'], 'pcm_sha256': 'e' * 64}])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.projection = invented_projection()
        self.blind = metadata.blind_metadata(self.projection)

    def test_exact_eighteen_slots_and_exclusions_retained(self):
        slots = c.fixed_slots(self.projection, self.blind['job'], self.blind['execution_plan'])
        self.assertEqual(len(slots), 18)
        self.assertEqual([(x['family'], x['cell_id']) for x in slots],
            [(family, cell) for family in c.FAMILIES for cell in c.CELLS])
        ledger = c.initial_ledger(self.projection)
        self.assertEqual(sum(x['status'] == 'not_run_duplicate' for x in ledger), 3)
        self.assertEqual(sum(x['status'] == 'not_run_quality' for x in ledger), 3)
        self.assertEqual(c.validate_ledger(self.projection, ledger)['denominator'], 18)

    def test_starts_after_first_failure_or_notrun_rejected(self):
        for status in ('started', 'failed_no_retry', 'not_run'):
            ledger = c.initial_ledger(self.projection)
            eligible = [x for x in ledger if x['status'] == 'not_run']
            eligible[0].update(status=status, attempts=0 if status == 'not_run' else 1)
            eligible[1].update(status='complete', attempts=1)
            with self.assertRaises(ValueError): c.validate_ledger(self.projection, ledger)

    def test_no_identity_replacement_or_hiding_quality_rows(self):
        ledger = c.initial_ledger(self.projection)
        ledger[1].update(status='complete', attempts=1)
        with self.assertRaises(ValueError): c.validate_ledger(self.projection, ledger)
        with self.assertRaises(ValueError): c.validate_ledger(self.projection, ledger[:-1])
        job = copy.deepcopy(self.blind['job']); job['clips'][0]['audio_id'] = 'clip-new-retry'
        with self.assertRaises(ValueError): c.validate_blind_job(job, self.blind['execution_plan'])

    def test_exact_fixed_order_differs_from_sha_order(self):
        job, plan = self.blind['job'], self.blind['execution_plan']
        c.validate_blind_job(job, plan)
        for family in c.FAMILIES:
            self.assertEqual([x['audio_id'] for x in c.planned_clips(job, plan, family)], plan['audio_order'])
        mutated = copy.deepcopy(plan); mutated['audio_order'].reverse()
        with self.assertRaises(ValueError): c.fixed_slots(self.projection, job, mutated)

    def test_zero_eligible_blocks_setup(self):
        projection = copy.deepcopy(self.projection)
        for row in projection['cells']: row['disposition'] = 'not_run_duplicate'
        blind = metadata.blind_metadata(projection)
        with self.assertRaises(ValueError): c.validate_blind_job(blind['job'], blind['execution_plan'])
        self.assertEqual(len(c.initial_ledger(projection)), 18)

    def test_limits_are_fresh_bounded_and_old_exhausted(self):
        self.assertEqual(c.LIMITS['old_calls_allowed'], 0)
        self.assertEqual(c.LIMITS['campaign_slots'], 18)
        self.assertEqual(c.LIMITS['cpu_threshold_seconds'], {'setup': 3600, 'qwen06': 600, 'sensevoice': 600, 'whisper': 1800})
        self.assertEqual(c.LIMITS['cpu_total_threshold_seconds'], 6600)
        self.assertEqual(c.LIMITS['acquisition_reserved_bytes'], 6056574976)
        self.assertFalse(hasattr(c, 'remaining_plan'))
        self.assertFalse(hasattr(c, 'ORIGINAL_SOURCES'))

    def test_default_disabled_and_all_actual_authorities_null(self):
        release = c.disabled_release(); c.validate_release_shape(release)
        self.assertTrue(all(release[k] is None for k in c.RELEASE_AUTHORITIES))
        with self.assertRaises(ValueError): c.require_execution_release(release, {}, expected_freeze_sha256='a' * 64)
        release['enabled'] = True
        with self.assertRaises(ValueError): c.require_execution_release(release, {}, expected_freeze_sha256='a' * 64)
        release = c.disabled_release(); release['max_new_asr_calls'] = True
        with self.assertRaises(ValueError): c.validate_release_shape(release)

    def test_external_source_hash_cannot_default_to_self_measurement(self):
        release = c.disabled_release()
        for key in c.RELEASE_APPROVALS: release[key] = True
        for key in c.RELEASE_AUTHORITIES: release[key] = 'a' * 64
        freeze = {'synthetic': 'not actual execution authority'}
        with self.assertRaises(ValueError): c.require_execution_release(release, freeze)
        with self.assertRaises(ValueError): c.require_execution_release(release, freeze, expected_freeze_sha256='b' * 64)
        # A self-consistent source digest is insufficient without actual prepare proof.
        with self.assertRaises(ValueError):
            c.require_execution_release(release, freeze, expected_freeze_sha256=c.digest(c.canonical(freeze)))

    def test_actual_input_module_anonymous_binding_integrates(self):
        from test_asr18_binding import setup_binding
        from input_binding import verify_screening_receipt
        raw, kwargs = setup_binding()
        result = verify_screening_receipt(raw, **kwargs)
        value, blind = result['worker_input_binding'], result['blind']
        self.assertEqual(c.validate_input_binding(value, blind['job'], blind['execution_plan']), value)
        release = c.disabled_release()
        for key in c.RELEASE_APPROVALS: release[key] = True
        for key in c.RELEASE_AUTHORITIES: release[key] = 'a' * 64
        release.update(actual_worker_binding_sha256=c.digest(c.canonical(value)),
            actual_screening_receipt_sha256=value['screening_receipt_sha256'],
            actual_tts_authority_sha256=value['tts_authority_sha256'],
            actual_opaque_order_sha256=c.digest(c.canonical(blind['execution_plan'])))
        self.assertEqual(c.validate_worker_admission(release, value, blind['job'], blind['execution_plan']), value)
        for key in ('actual_worker_binding_sha256', 'actual_screening_receipt_sha256',
                    'actual_tts_authority_sha256', 'actual_opaque_order_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_worker_admission({**release, key: 'e' * 64}, value, blind['job'], blind['execution_plan'])

    def test_archived_v1_source_claim_binds_context_and_retains_history1(self):
        release = c.disabled_release()
        for key in c.RELEASE_APPROVALS: release[key] = True
        for key in c.RELEASE_AUTHORITIES: release[key] = 'a' * 64
        claim = {'schema': 'k1-asr18-once-source-claim-v1',
            'identity': {'repository': 'jiying2007/kws-pipeline', 'executed_head_sha': 'c' * 40,
                'run_id': 123, 'run_attempt': 1, 'job_key': 'asr18'},
            'source_freeze_sha256': 'f' * 64, 'release_sha256': c.digest(c.canonical(release)),
            'source_branch': release['source_ref'], 'result_branch': release['evidence_ref'],
            'workflow': release['workflow'], 'numeric_job_id': 456,
            'prior_public_claim_history_sha256': release['prior_public_claim_history_sha256'],
            'api_preflight_verified': True, 'source_branch_workflow_runs': 1,
            'maximum_new_asr_calls': 18, 'old_calls_allowed': 0, 'new_audio_forwards_at_claim': 0}
        def validate(value, **changes):
            body = c.canonical(value)
            args = {'expected_sha256': c.digest(body), 'release': release,
                'source_freeze_sha256': 'f' * 64, 'run_id': '123', 'executed_head_sha': 'c' * 40}
            args.update(changes)
            return c.validate_legacy_source_claim(body, **args)
        self.assertEqual(validate(claim), claim)
        for key, value in (('source_branch_workflow_runs', 2), ('old_calls_allowed', 1),
            ('maximum_new_asr_calls', 19), ('new_audio_forwards_at_claim', 1),
            ('numeric_job_id', True), ('api_preflight_verified', 1)):
            with self.subTest(key=key), self.assertRaises(ValueError): validate({**claim, key: value})
        for changed in ({'run_id': '124'}, {'executed_head_sha': 'd' * 40},
                        {'source_freeze_sha256': 'e' * 64}, {'expected_sha256': '1' * 64}):
            with self.subTest(changed=changed), self.assertRaises(ValueError): validate(claim, **changed)
        bad = copy.deepcopy(claim); bad['identity']['run_attempt'] = 2
        with self.assertRaises(ValueError): validate(bad)

    def test_retained_pretty_locks_are_read_without_rewriting(self):
        root = HERE.parents[1]
        path = root / 'research/source-screen32-whisper-v1/source-lock.json'
        before = path.read_bytes()
        self.assertEqual(c.read_json(path)['schema'], 'whisper-screen-official-source-lock-v1')
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_noncanonical_and_nonfinite_json_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x": 1}', b'{"x":NaN}'):
            with self.assertRaises(ValueError): c.decode_json(raw)


class DurableIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='k1-invented-io-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_open_flags_and_regular_readonly_write(self):
        calls, real_open = [], os.open
        def observe(path, flags, *args, **kwargs):
            calls.append((path, flags)); return real_open(path, flags, *args, **kwargs)
        with patch.object(os, 'open', observe): c.write_once(self.root / 'claim.json', {'invented': True})
        flags = next(flags for path, flags in calls if path == 'claim.json' and flags & os.O_WRONLY)
        self.assertTrue(flags & os.O_EXCL and flags & os.O_NOFOLLOW and flags & os.O_CREAT)
        self.assertEqual(c.read_json(self.root / 'claim.json'), {'invented': True})
        with self.assertRaises(FileExistsError): c.write_once(self.root / 'claim.json', {'invented': False})
        self.assertFalse((self.root / 'claim.json').stat().st_mode & 0o222)

    def test_symlink_and_hardlink_refused_without_side_effects(self):
        target = self.root / 'target'; target.mkdir()
        (self.root / 'alias').symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError): c.write_once(self.root / 'alias' / 'claim.json', {})
        self.assertFalse((target / 'claim.json').exists())
        c.write_once(target / 'good.json', {'invented': True})
        (target / 'hard.json').hardlink_to(target / 'good.json')
        with self.assertRaises(ValueError): c.read_bytes(target / 'hard.json')
        (target / 'link.json').symlink_to(target / 'good.json')
        with self.assertRaises(OSError): c.read_bytes(target / 'link.json')
        with self.assertRaises(FileExistsError): c.write_once(target / 'link.json', {})

    def test_read_fault_closes_all_fds(self):
        c.write_once(self.root / 'data.json', {'invented': True})
        before = len(list(Path('/proc/self/fd').iterdir()))
        with patch.object(os, 'read', side_effect=OSError('invented read fault')):
            with self.assertRaises(OSError): c.read_bytes(self.root / 'data.json')
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_parent_component_open_fault_closes_all_fds(self):
        before = len(list(Path('/proc/self/fd').iterdir()))
        original = os.open
        def failure(path, *args, **kwargs):
            if path == self.root.name: raise OSError('invented parent open fault')
            return original(path, *args, **kwargs)
        with patch.object(os, 'open', failure):
            with self.assertRaises(OSError): c.write_once(self.root / 'data.json', {})
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)


if __name__ == '__main__':
    unittest.main()
