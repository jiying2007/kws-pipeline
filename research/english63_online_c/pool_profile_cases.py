#!/usr/bin/env python3
"""Invented saved-record checks; never execute a model, C binary or real PCM."""
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('en63_pool_saved_cases', HERE / 'pool_profile.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def invented_trace():
    # Binary fields are fabricated without feeding audio or running a network.
    values = [0.0] * 35 + [1.0, 0.0, 0.0]
    return P.TRACE_HEADER + b''.join(P.TRACE_RECORD.pack(0, 400 + index * 160, *values, 0)
                                    for index in range(13134))


def invented_pass(plan, trace, output=Path('/invented/result/pool-profile')):
    timed = plan['phase'] == 'timed'
    candidate = plan['variant'] == 1
    body = {'schema': 'english63-pool-native-pass-v1', 'kind': plan['kind'], 'variant': plan['variant'],
            'phase': plan['phase'], 'affinity_cpu': 2, 'affinity_single_cpu': True,
            'pcm_samples': 2101760, 'frames': 13134, 'feed_calls': 13136,
            'proposal_count': trace['proposal_count'], 'signature64': trace['signature64'],
            'state_bytes': 38192, 'weight_parameter_bytes': (6531 if plan['kind'] == 'gru' else 4803) * 4,
            'frontend_table_bytes': 3894, 'input_pcm_bytes': 4203520,
            'cpu_ns': (18000000 if candidate else 20000000) if timed else 0,
            'wall_ns': 30000000 if timed else 0,
            'feed_ns': [900 if candidate else 1000] * 13136 if timed else [],
            'trace_io_enabled': plan['phase'] == 'trace', 'eof_padding_samples': 0, 'eof_events': 0,
            'inference_end_high_water_rss_kib': 1000,
            'timed_region': 'preloaded-pcm-en63_feed-and-constant-signature-callback'}
    lifetime = {'schema': 'english63-pool-child-lifetime-v1', 'child_exit_code': 0, 'child_signaled': False,
                'process_lifetime_peak_rss_kib': 1200, 'child_user_us': 30000, 'child_system_us': 1000}
    return {**plan, 'status': 'COMPLETED', 'exit_code': 0, 'timed_out': False,
            'binary_sha256': 'b' * 64 if candidate else 'a' * 64,
            'command': [str(output / ('en63_pool_' + plan['variant_name'])), '--supervise', plan['kind'],
                        plan['phase'], '/invented/temporary/full-original.pcm',
                        str(output / (plan['kind'] + '-' + plan['variant_name'] + '.trace'))
                        if plan['phase'] == 'trace' else '-', '2'],
            'worker': body, 'lifetime': lifetime,
            'raw_stdout': json.dumps(body) + '\n' + json.dumps(lifetime) + '\n', 'raw_stderr': ''}


def rebind_stdout(record):
    record['raw_stdout'] = json.dumps(record['worker']) + '\n' + json.dumps(record['lifetime']) + '\n'


def invented_measurements(trace, output=Path('/invented/result/pool-profile')):
    return {'schema': P.MEASUREMENTS_SCHEMA, 'initial_allowed_cpus': [2, 5], 'selected_cpu': 2,
            'final_allowed_cpus': [2, 5], 'native_group_elapsed_seconds': 30.0,
            'passes': [invented_pass(plan, trace, output) for plan in P.schedule()]}


def request_bytes(head):
    return encoded({'schema': P.REQUEST_SCHEMA, 'experiment': P.EXPERIMENT, 'request_id': P.REQUEST_ID,
                    'pipeline_repository': 'jiying2007/kws-pipeline',
                    'entrypoint': 'research/english63_online_c/pool_profile.py',
                    'source_artifacts': P.SOURCE_ARTIFACTS, 'native_state_ancestor': P.ANCESTOR,
                    'profile': P.PROFILE, 'optimizer_updates': 0, 'cpu_threads': 1,
                    'max_execution_seconds': 600, 'max_verification_seconds': 180,
                    'new_audio_generated': False, 'new_tts_asr_ipa_calls': 0,
                    'new_independent_audio_seconds': 0, 'formal_training_admitted': False,
                    'human_gold': False, 'model_promotion_allowed': False, 'pipeline_commit': head,
                    'reason': 'Fabricated source-only receipt, not an executed experiment.'})


@contextmanager
def invented_saved_bundle(raw, trace):
    """A deliberately fake parent and ELF envelope; no assets/binaries are loaded."""
    header = b'invented fixture header without weights\n'
    header_sha = P.c.digest(header)
    ancestor = {**P.ANCESTOR, 'header_sha256': header_sha}
    old_rows, total = [], 0
    for index in range(63):
        count = 33360 if index < 62 else 33440
        old_rows.append({'cell_id': 'invented-' + str(index), 'member': 'invented/' + str(index),
                         'wav_sha256': '1' * 64, 'pcm_sha256': '2' * 64,
                         'action': 'INVENTED_NEVER_RUN', 'source_family_id': 'invented',
                         'source_samples': count, 'concatenation_start_sample': total,
                         'test_start_within_source': 0, 'test_samples_used': min(count, max(0, 128246 - total)),
                         'human_gold': False, 'training_admitted': False})
        total += count
    old_input_raw = encoded({'rows': old_rows})
    old_request_raw = encoded({'pipeline_commit': ancestor['pipeline_commit']})
    ancestor['request_sha256'] = P.c.digest(old_request_raw)
    old_files = dict.fromkeys(P.STATE_FILES - {'REPLAY.json'}, 'c' * 64)
    old_files.update({'en63_generated.h': header_sha, 'INPUTS.json': P.c.digest(old_input_raw)})
    old_replay_raw = encoded({'files_sha256': old_files})
    parent_files = {'state-contracts/' + key: value for key, value in old_files.items()}
    parent_files.update({'state-contracts/REPLAY.json': P.c.digest(old_replay_raw),
                         'REQUEST.json': ancestor['request_sha256']})
    old_terminal_raw = encoded({'status': 'SUCCEEDED', 'run_id': str(ancestor['run_id']), 'run_attempt': '1',
                                'pipeline_commit': ancestor['pipeline_commit'], 'data_commit': ancestor['data_commit'],
                                'request_sha256': ancestor['request_sha256'], 'files_sha256': parent_files})
    ancestor['terminal_sha256'] = P.c.digest(old_terminal_raw)
    parent_raws = {'TERMINAL.json': old_terminal_raw, 'REQUEST.json': old_request_raw,
                   'state-contracts/INPUTS.json': old_input_raw, 'state-contracts/REPLAY.json': old_replay_raw}
    evidence = {name: {'raw_utf8': body.decode(), **P.artifact_identity(body)} for name, body in parent_raws.items()}
    with tempfile.TemporaryDirectory() as temporary, patch.multiple(
            P, HEADER_SHA=header_sha, ANCESTOR=ancestor,
            PARENT_EVIDENCE={name: P.c.digest(body) for name, body in parent_raws.items()},
            BASELINE_TRACE_SHA=dict.fromkeys(P.c.KINDS, trace['sha256'])):
        root = Path(temporary)
        head = 'b' * 40
        run = {'repository': 'jiying2007/kws-data', 'head': 'a' * 40, 'run_id': '123456', 'run_attempt': '1'}
        request = request_bytes(head)
        claim = encoded({'schema': 'kws-research-v4-native-c-pool-start-claim-v1',
                         'request_sha256': P.c.digest(request), 'pipeline_commit': head,
                         'source_state_artifact_id': ancestor['artifact_id'], 'run_id': run['run_id'],
                         'run_attempt': '1', 'optimizer_updates': 0,
                         'status': 'CONSUMED_BEFORE_PAIRED_NATIVE_C_POOL_EXPERIMENT'})
        inputs = {'schema': 'english63-pool-profile-inputs-v1', 'native_state_ancestor': ancestor,
                  'source_header_sha256': header_sha, 'source_tts_zip_sha256': P.c.TTS_SHA,
                  'source_review_json_sha256': P.c.REVIEW_SHA, 'source_weights_sha256': P.c.SOURCE_WEIGHTS,
                  'source_audio_clips': 63, 'source_pcm_samples': total, 'unique_input_samples': total,
                  'test_pcm_bytes': total * 2, 'test_pcm_sha256': 'd' * 64,
                  'rows': [{**row, 'test_samples_used': row['source_samples']} for row in old_rows],
                  'pcm_operation': 'byte-preserving-full63-concatenation',
                  'new_audio_generation_or_resampling': False, 'parent_evidence': evidence,
                  'request_raw_utf8': request.decode(), 'request_sha256': P.c.digest(request),
                  'claim_raw_utf8': claim.decode(), 'claim_sha256': P.c.digest(claim)}
        binaries = {}
        for variant in P.PROFILE['variants']:
            body = b'\x7fELF' + ('invented-not-executable-' + variant).encode()
            (root / ('en63_pool_' + variant)).write_bytes(body)
            binaries[variant] = P.c.digest(body)
        source = P.source_hashes()
        build = {'schema': 'english63-pool-native-build-v1', 'compiler': '/invented/gcc',
                 'compiler_realpath': '/invented/gcc', 'compiler_sha256': 'e' * 64,
                 'compiler_version': 'INVENTED-NOT-EXECUTED', 'source_root': str(P.ROOT), 'output_root': str(root),
                 'source_sha256': source, 'source_header_sha256': header_sha, 'public_run': run,
                 'pipeline_head': head, 'compile_flags': list(P.FLAGS), 'machine': 'invented',
                 'libc': ['invented', '0'], 'physical_ssc305': False,
                 'compiles': [{'variant_name': variant, 'variant': value,
                               'command': P.command_for('/invented/gcc', P.ROOT, root, variant),
                               'exit_code': 0, 'stdout': '', 'stderr': '', 'binary_sha256': binaries[variant]}
                              for variant, value in P.PROFILE['variants'].items()]}
        measurements = invented_measurements(trace, root)
        for row in measurements['passes']:
            row['binary_sha256'] = binaries[row['variant_name']]
        measurements.update({'public_run': run, 'pipeline_head': head, 'request_sha256': P.c.digest(request),
                             'source_integrity': {
                                 'source_before': source, 'source_after': source,
                                 'header_before': header_sha, 'header_after': header_sha,
                                 'tts_before': P.c.TTS_SHA, 'tts_after': P.c.TTS_SHA,
                                 'review_before': P.c.REVIEW_SHA, 'review_after': P.c.REVIEW_SHA,
                                 'pcm_before': 'd' * 64, 'pcm_after': 'd' * 64,
                                 'parent_terminal_before': ancestor['terminal_sha256'],
                                 'parent_terminal_after': ancestor['terminal_sha256'],
                                 'parent_files_before': parent_files, 'parent_files_after': parent_files,
                                 'binaries_before': binaries, 'binaries_after': binaries,
                                 'request_before': P.c.digest(request), 'request_after': P.c.digest(request),
                                 'claim_before': P.c.digest(claim), 'claim_after': P.c.digest(claim)}})
        (root / 'en63_generated.h').write_bytes(header)
        (root / 'native.stderr').write_bytes(b'')
        for kind in P.c.KINDS:
            for variant in P.PROFILE['variants']:
                (root / (kind + '-' + variant + '.trace')).write_bytes(raw)
        for name, body in [('PROTOCOL.json', P.protocol()), ('INPUTS.json', inputs),
                           ('BUILD.json', build), ('MEASUREMENTS.json', measurements)]:
            (root / name).write_bytes(encoded(body))
        with zipfile.ZipFile(root / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in P.SOURCES:
                archive.writestr(name, (P.ROOT / name).read_bytes())
        models = P.measurements_result(measurements, dict.fromkeys(P.c.KINDS, trace), binaries)
        summary = P.summary_from({**inputs, 'file_sha256': P.c.digest((root / 'INPUTS.json').read_bytes())}, build,
                                 {**measurements, 'file_sha256': P.c.digest((root / 'MEASUREMENTS.json').read_bytes())},
                                 models, source)
        (root / 'SUMMARY.json').write_bytes(encoded(summary))
        receipt = P.saved_receipt(root)
        (root / 'REPLAY.json').write_bytes(encoded(receipt))
        yield root, receipt


class SavedRecordContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = invented_trace()
        cls.trace = P.trace_observation(cls.raw)
        cls.binaries = {'baseline': 'a' * 64, 'candidate': 'b' * 64}

    def test_exact_pass_budget_and_nonindependent_exposure(self):
        order = P.schedule()
        self.assertEqual(len(order), 32)
        self.assertEqual([row['run_index'] for row in order], list(range(32)))
        self.assertEqual([row['kind'] for row in order], ['gru'] * 16 + ['dsc_tcn'] * 16)
        self.assertEqual([row['variant'] for row in order[:4]], [0, 1, 0, 1])
        self.assertEqual([row['variant'] for row in order[4:16]], [0, 1, 1, 0] * 3)
        self.assertEqual(P.PROFILE['full_passes'] * P.PROFILE['frames_per_pass'], 420288)
        self.assertEqual(P.PROFILE['full_passes'] * P.PROFILE['source_pcm_samples'], 67256320)
        self.assertEqual(P.protocol()['new_independent_audio_samples'], 0)
        self.assertFalse(P.protocol()['runtime_default_changed'])

    def test_real_entry_refuses_local_before_any_child_or_input(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(P.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(ValueError, 'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY'):
                P.run(None)
            child.assert_not_called()

    def test_public_identity_rejects_second_attempt_other_repo_and_bad_head(self):
        base = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-data',
                'GITHUB_RUN_ID': '123456', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40}
        for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REPOSITORY', 'jiying2007/kws-pipeline'),
                           ('GITHUB_RUN_ID', '0'), ('GITHUB_SHA', 'g' * 40)]:
            with self.subTest(key=key), patch.dict(os.environ, {**base, key: value}, clear=True):
                with self.assertRaises(ValueError):
                    P.require_public_actions()

    def test_child_environment_contains_no_credentials_or_parent_run_secrets(self):
        with patch.dict(os.environ, {'PATH': '/usr/bin', 'GITHUB_TOKEN': 'fixture-secret',
                                    'AWS_ACCESS_KEY_ID': 'fixture-secret', 'SECRET': 'fixture-secret'}, clear=True):
            env = P.native_environment()
        self.assertNotIn('fixture-secret', str(env))
        self.assertEqual(env['OMP_NUM_THREADS'], '1')

    def test_trace_complete_clock_and_signature(self):
        self.assertEqual(self.trace['frames'], 13134)
        self.assertEqual(self.trace['last_sample'], 2101680)
        self.assertEqual(self.trace['proposal_count'], 0)
        self.assertEqual(len(self.trace['signature64']), 16)

    def test_fabricated_trace_clock_nan_truncation_and_spurious_event_rejected(self):
        bad_clock, bad_nan, bad_event = bytearray(self.raw), bytearray(self.raw), bytearray(self.raw)
        struct.pack_into('<Q', bad_clock, 16 + 4, 401)
        struct.pack_into('<f', bad_nan, 16 + 12, float('nan'))
        struct.pack_into('<I', bad_event, 16 + 164, 1)
        for raw in (bytes(bad_clock), bytes(bad_nan), bytes(bad_event), self.raw[:-168], self.raw + self.raw[-168:]):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                P.trace_observation(raw)

    def test_decoder_uses_float32_threshold_three_confirmations_and_cooldown(self):
        chunks = [P.TRACE_HEADER]
        for frame in range(1, 13135):
            probabilities = [0.1, 0.8, 0.1]
            event = 1 if frame >= 52 and (frame - 52) % 102 == 0 else 0
            chunks.append(P.TRACE_RECORD.pack(0, 400 + (frame - 1) * 160, *([0.0] * 35 + probabilities), event))
        result = P.trace_observation(b''.join(chunks))
        self.assertEqual(result['proposals'][0]['frame'], 52)
        self.assertEqual(result['proposals'][1]['frame'], 154)

    def test_nearest_rank_quantiles_are_not_interpolation(self):
        self.assertEqual(P.percentile(list(range(1, 13137)), 99), 13005)
        self.assertEqual(P.percentile([4, 1, 3, 2], 50), 2)
        with self.assertRaises(ValueError):
            P.percentile([True, 2], 99)

    def test_benefit_and_no_benefit_are_both_valid_results(self):
        measurements = invented_measurements(self.trace)
        rows = P.measurements_result(measurements, dict.fromkeys(P.c.KINDS, self.trace), self.binaries)
        self.assertEqual([row['decision'] for row in rows], ['BENEFIT', 'BENEFIT'])
        for record in measurements['passes']:
            if record['phase'] == 'timed' and record['variant'] == 1:
                record['worker']['cpu_ns'] = 21000000
                rebind_stdout(record)
        rows = P.measurements_result(measurements, dict.fromkeys(P.c.KINDS, self.trace), self.binaries)
        self.assertEqual([row['decision'] for row in rows], ['NO_BENEFIT', 'NO_BENEFIT'])

    def test_missing_duplicate_or_reordered_pass_is_rejected(self):
        for action in ('missing', 'duplicate', 'order'):
            measurements = invented_measurements(self.trace)
            if action == 'missing':
                measurements['passes'].pop()
            elif action == 'duplicate':
                measurements['passes'][6] = measurements['passes'][4]
            else:
                measurements['passes'][4:6] = measurements['passes'][4:6][::-1]
            with self.subTest(action=action), self.assertRaises(ValueError):
                P.measurements_result(measurements, dict.fromkeys(P.c.KINDS, self.trace), self.binaries)

    def test_native_raw_stdout_and_parsed_body_cannot_disagree(self):
        plan = P.schedule()[4]
        record = invented_pass(plan, self.trace)
        record['worker']['cpu_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'RAW_NATIVE_STDOUT_BINDING'):
            P.pass_observation(record, plan, 2, self.trace, self.binaries)

    def test_bool_float_exit_and_native_counter_are_rejected_even_rebound(self):
        plan = P.schedule()[4]
        for target, key, value in [('record', 'exit_code', False), ('record', 'exit_code', 0.0),
                                   ('worker', 'variant', False), ('worker', 'cpu_ns', True),
                                   ('lifetime', 'child_exit_code', False), ('worker', 'frames', 13134.0)]:
            record = invented_pass(plan, self.trace)
            (record if target == 'record' else record[target])[key] = value
            rebind_stdout(record)
            with self.subTest(target=target, key=key, value=value), self.assertRaises(ValueError):
                P.pass_observation(record, plan, 2, self.trace, self.binaries)

    def test_timed_phase_requires_all_original_feed_timings(self):
        plan = P.schedule()[4]
        for timings in ([1000] * 13135, [1000] * 13135 + [0], [1000] * 13135 + [True]):
            record = invented_pass(plan, self.trace)
            record['worker']['feed_ns'] = timings
            rebind_stdout(record)
            with self.assertRaisesRegex(ValueError, 'TIMED_FEED_OBSERVATIONS'):
                P.pass_observation(record, plan, 2, self.trace, self.binaries)

    def test_warmup_cannot_be_reported_as_timed_or_emit_a_trace(self):
        plan = P.schedule()[2]
        for key, value in [('cpu_ns', 1), ('feed_ns', [1]), ('trace_io_enabled', True)]:
            record = invented_pass(plan, self.trace)
            record['worker'][key] = value
            rebind_stdout(record)
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.pass_observation(record, plan, 2, self.trace, self.binaries)

    def test_fixed_affinity_command_and_signature_bind_all_passes(self):
        plan = P.schedule()[4]
        for field, value in [('affinity_cpu', 5), ('signature64', 'f' * 16), ('state_bytes', 38193),
                              ('proposal_count', 1)]:
            record = invented_pass(plan, self.trace)
            record['worker'][field] = value
            rebind_stdout(record)
            with self.subTest(field=field), self.assertRaises(ValueError):
                P.pass_observation(record, plan, 2, self.trace, self.binaries)
        record = invented_pass(plan, self.trace)
        record['command'][3] = 'trace'
        with self.assertRaisesRegex(ValueError, 'NATIVE_FIXED_COMMAND'):
            P.pass_observation(record, plan, 2, self.trace, self.binaries)

    def test_cpu_set_minimum_is_checked_not_only_reported(self):
        measurements = invented_measurements(self.trace)
        measurements['initial_allowed_cpus'] = [1, 2, 5]
        with self.assertRaisesRegex(ValueError, 'INITIAL_SINGLE_CPU'):
            P.measurements_result(measurements, dict.fromkeys(P.c.KINDS, self.trace), self.binaries)

    def test_canonical_type_checks_and_json_duplicate_rejection(self):
        self.assertFalse(P.exact({'exit': False}, {'exit': 0}))
        self.assertFalse(P.exact({'exit': 0.0}, {'exit': 0}))
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                P.parse_json(raw)

    def test_fixed_request_rejects_budget_change_and_wrong_checkout(self):
        head = 'b' * 40
        raw = request_bytes(head)
        P.validate_request(raw, head)
        changed = json.loads(raw)
        changed['profile']['full_passes'] = 34
        with self.assertRaises(ValueError):
            P.validate_request(encoded(changed), head)
        with self.assertRaises(ValueError):
            P.validate_request(raw, 'c' * 40)

    def test_saved_verification_refuses_input_arguments_without_model_activity(self):
        with patch.object(P, 'run') as execute, patch.object(P, 'verify_saved') as verify:
            with self.assertRaisesRegex(ValueError, 'SAVED_VERIFY_ACCEPTS_NO_RUNTIME_INPUTS'):
                P.main(['--verify-result-dir', '/invented', '--tts-zip', '/invented.zip'])
            execute.assert_not_called()
            verify.assert_not_called()

    def test_flat_closed_set_rejects_missing_extra_empty_and_nested_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in P.RESULT_FILES - {'REPLAY.json'}:
                (root / name).write_bytes(b'invented')
            self.assertEqual(len(P.artifact_manifest(root)), 14)
            for name in ('extra', 'TERMINAL.json', 'FAILED.json'):
                (root / name).write_bytes(b'invented')
                with self.assertRaises(ValueError):
                    P.artifact_manifest(root)
                (root / name).unlink()
            (root / 'nested').mkdir()
            with self.assertRaises(ValueError):
                P.artifact_manifest(root)
            (root / 'nested').rmdir()
            (root / 'BUILD.json').write_bytes(b'')
            with self.assertRaises(ValueError):
                P.artifact_manifest(root)
            (root / 'BUILD.json').unlink()
            with self.assertRaises(ValueError):
                P.artifact_manifest(root)

    def test_compile_commands_differ_only_in_macro_and_output(self):
        baseline = P.command_for('/usr/bin/gcc', '/invented/source', '/invented/output', 'baseline')
        candidate = P.command_for('/usr/bin/gcc', '/invented/source', '/invented/output', 'candidate')
        differences = [(a, b) for a, b in zip(baseline, candidate) if a != b]
        self.assertEqual(differences, [('-DEN63_POOL_TIME_MAJOR=0', '-DEN63_POOL_TIME_MAJOR=1'),
                                      ('/invented/output/en63_pool_baseline', '/invented/output/en63_pool_candidate')])

    def test_parent_original_raw_bytes_cannot_be_replaced_with_rehashed_json(self):
        with self.assertRaisesRegex(ValueError, 'PARENT_ORIGINAL_BYTES'):
            P.parent_evidence_check({name: {'raw_utf8': '{}', 'bytes': 2, 'sha256': P.c.digest(b'{}')}
                                     for name in P.PARENT_EVIDENCE})

    def test_complete_fifteen_file_saved_roundtrip_executes_nothing(self):
        with patch.object(P.subprocess, 'Popen') as child, invented_saved_bundle(self.raw, self.trace) as (root, receipt):
            self.assertEqual(P.verify_saved(root), receipt)
            self.assertEqual(len(list(root.iterdir())), 15)
            self.assertEqual(receipt['model_forward_calls'], 0)
            child.assert_not_called()

    def test_rehashed_forged_benefit_is_rejected_by_saved_measurement_recomputation(self):
        with invented_saved_bundle(self.raw, self.trace) as (root, receipt):
            summary = json.loads((root / 'SUMMARY.json').read_bytes())
            summary['per_kind'][0]['paired_cpu_reduction_median'] = 0.9
            (root / 'SUMMARY.json').write_bytes(encoded(summary))
            receipt['summary_sha256'] = P.c.digest((root / 'SUMMARY.json').read_bytes())
            receipt['files_sha256']['SUMMARY.json'] = receipt['summary_sha256']
            receipt['files']['SUMMARY.json'] = P.artifact_identity((root / 'SUMMARY.json').read_bytes())
            (root / 'REPLAY.json').write_bytes(encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'SUMMARY_RECOMPUTED'):
                P.verify_saved(root)


if __name__ == '__main__':
    unittest.main()
