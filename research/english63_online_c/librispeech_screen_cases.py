#!/usr/bin/env python3
"""Invented source boundaries, native tensors and saved records; no real input."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_librispeech_screen_fixture', HERE / 'librispeech_screen.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def encoded(value):
    return (S.p.canonical(value) + '\n').encode('utf-8')


def invented_header():
    def array(kind, name, values):
        return 'static const ' + kind + ' ' + name + '[]={' + ','.join(map(str, values)) + '};\n'
    header = '#include "en63_stream.h"\n'
    for name, values in (('w', [1] * 400), ('mean', [0] * 32), ('std', [1] * 32), ('mel', [1] * 32)):
        header += array('float', name, values)
    for name, values in (('start', [0] * 32), ('count', [1] * 32), ('offset', list(range(33)))):
        header += array('uint16_t', name, values)
    for kind, size, tag in (('gru', 6531, 1), ('dsc_tcn', 4803, 2)):
        header += 'static const float ' + kind + '_p[' + str(size) + ']={[' + str(size - 2) + ']=4};\n'
        header += ('static const en63_model en63_' + kind + '_model={' + str(tag) + ',' + str(size) + ',' +
                   kind + '_p,w,mean,std,start,count,offset,mel,32,"invented-not-a-checkpoint"};\n')
    return (header + '#define EN63_FRONTEND_TABLE_BYTES 2178u\n').encode('utf-8')


def native_fixture(compiler, flags, root):
    if (HERE / 'en63_generated.h').exists() or (HERE / 'en63_generated.h').is_symlink():
        raise ValueError('Invented header shadowed by source-directory original header')
    (root / 'en63_generated.h').write_bytes(invented_header())
    binary = root / 'invented-librispeech'
    subprocess.run([compiler, *flags, '-DEN63_POOL_TIME_MAJOR=0', '-I', str(HERE), '-I', str(root),
                    str(HERE / 'en63_stream.c'), str(HERE / 'librispeech_screen.c'), '-lm', '-o', str(binary)],
                   check=True, timeout=90)
    # Seventy files exceed the old replay CLI's 64-file bound. The first exact
    # splice occurs at the first real event, and the final feed is 84 samples.
    counts = [8560] + [401] * 68 + [576]
    paths, rows, offset, reference = [], [], 0, bytearray()
    for index, count in enumerate(counts):
        raw = struct.pack('<h', index * 53 - 1500) * count
        path = root / ('part-' + str(index) + '.pcm')
        path.write_bytes(raw)
        paths.append(path)
        rows.append({'member': 'invented-' + str(index), 'sample_count': count,
                     'concatenation_start_sample': offset})
        reference.extend(raw)
        offset += count
    pcm = root / 'full-original.pcm'
    if S.join_pcm_files(paths, pcm) != offset or pcm.read_bytes() != reference:
        raise ValueError('Invented file joins inserted/removed/reordered bytes')
    cpu = min(os.sched_getaffinity(0))
    traces = []
    for kind in S.p.c.KINDS:
        trace = root / (kind + '.trace')
        raw = subprocess.check_output([str(binary), '--supervise', kind, str(pcm), str(trace), str(cpu), str(offset)],
                                      text=True, timeout=60, env=S.p.native_environment())
        body, lifetime = [json.loads(line) for line in raw.splitlines()]
        observed = S.trace_observation(trace.read_bytes(), rows, offset)
        if not (body['frames'] == 226 and body['feed_calls'] == 228 and body['final_feed_samples'] == 84 and
                body['pcm_samples'] == offset == 36404 and body['eof_events'] == 0 and
                body['eof_padding_samples'] == 0 and body['reset_calls'] == 0 and
                body['discontinuity_calls'] == 0 and body['initializations'] == 1 and
                observed['proposal_count'] == 2 and observed['events'][0]['frame'] == 52 and
                observed['events'][0]['available_sample'] == 8560 and
                observed['events'][0]['keyword'] == 'Hey Vosen' and
                observed['events'][0]['source_interval_member'] == 'invented-0' and
                observed['events'][0]['exact_splice_boundary'] is True and
                observed['events'][1]['frame'] == 154 and lifetime['child_exit_code'] == 0 and
                lifetime['child_signaled'] is False):
            raise ValueError('Invented stream/decoder/clock/EOF/file-boundary contract failed')
        traces.append(trace.read_bytes())
    if traces[0] != traces[1]:
        raise ValueError('Equivalent invented classifiers produced different complete traces')
    # A length mismatch must fail before a trace/model replay is started.
    for delta in (-1, 1):
        rejected = root / ('rejected-' + str(delta) + '.trace')
        result = subprocess.run([str(binary), 'gru', str(pcm), str(rejected), str(cpu), str(offset + delta)],
                                capture_output=True, text=True, timeout=10, env=S.p.native_environment())
        if result.returncode == 0 or rejected.exists():
            raise ValueError('Incorrect PCM length was accepted')
    print('PASS invented LibriSpeech C: 70 source files, 226 frames/model, exact splice event, partial84 EOF; real model/audio calls=0')


def invented_reference():
    result, rows, offset = {}, [], 0
    for index in range(517):
        count = 113435 if index < 516 else 58646164 - 113435 * 516
        name = 'invented-' + format(index, '04d')
        row = {'member': name, 'flac_sha256': 'a' * 64, 'transcript_sha256': 'b' * 64,
               'speaker_source_id': str(index % 40), 'sample_count': count,
               'human_gold': False, 'trusted_negative': False, 'training_admitted': False}
        result[name] = row
        rows.append({**row, 'pcm_sha256': 'c' * 64, 'concatenation_start_sample': offset})
        offset += count
    return result, rows


def invented_pass(kind, root, cpu=2):
    body = {'schema': 'english63-librispeech-native-pass-v1', 'kind': kind,
            'pcm_samples': 58646164, 'frames': 366537, 'feed_calls': 366539, 'final_feed_samples': 84,
            'proposal_count': 0, 'state_bytes': 38192,
            'weight_parameter_bytes': (6531 if kind == 'gru' else 4803) * 4,
            'frontend_table_bytes': 3894, 'pool_time_major': 0, 'affinity_cpu': cpu, 'affinity_single_cpu': True,
            'initializations': 1, 'reset_calls': 0, 'discontinuity_calls': 0, 'finish_calls': 1,
            'eof_padding_samples': 0, 'eof_events': 0, 'cpu_ns': 1000000, 'wall_ns': 2000000}
    lifetime = {'schema': 'english63-librispeech-child-lifetime-v1', 'child_exit_code': 0,
                'child_signaled': False, 'process_lifetime_peak_rss_kib': 4000}
    return {'kind': kind, 'status': 'COMPLETED', 'exit_code': 0, 'timed_out': False,
            'command': [str(root / 'en63_librispeech'), '--supervise', kind,
                        '/invented/full-original.pcm', str(root / (kind + '.trace')), str(cpu), '58646164'],
            'raw_stdout': encoded(body).decode() + encoded(lifetime).decode(), 'raw_stderr': ''}


@contextmanager
def invented_bundle(raw):
    reference, rows = invented_reference()
    header = b'INVENTED-HEADER-NOT-EXECUTABLE\n'
    with tempfile.TemporaryDirectory() as temporary, patch.object(S.p, 'HEADER_SHA', S.digest(header)), \
            patch.object(S, 'reference_rows', return_value=reference), \
            patch.object(S.p, 'parent_evidence_check', return_value={'TERMINAL.json': {'files_sha256': {}}}):
        root = Path(temporary)
        run = {'repository': 'jiying2007/kws-data', 'head': 'a' * 40, 'run_id': '123456', 'run_attempt': '1'}
        head = 'b' * 40
        request = encoded({**S.request_fixed(), 'pipeline_commit': head, 'reason': 'invented fixture'})
        claim = encoded({'schema': 'kws-research-v4-native-c-librispeech-start-claim-v1',
                         'request_sha256': S.digest(request), 'pipeline_commit': head,
                         'source_state_artifact_id': 14141171320, 'source_negative_artifact_id': 14090383839,
                         'run_id': run['run_id'], 'run_attempt': '1', 'optimizer_updates': 0,
                         'status': 'CONSUMED_BEFORE_NATIVE_C_LIBRISPEECH_SCREEN'})
        inputs = {'schema': 'english63-librispeech-inputs-v1', 'native_state_ancestor': S.p.ANCESTOR,
                  'negative_reference': S.REFERENCE, 'corpus': S.CORPUS, 'header_sha256': S.p.HEADER_SHA,
                  'rows': rows, 'concatenated_pcm_sha256': 'd' * 64,
                  'pcm_decoder': {'numpy': '2.3.5', 'soundfile': '0.13.1', 'libsndfile': 'invented',
                                  'sample_type': 'little-endian-PCM16', 'resampled': False, 'normalized': False},
                  'parent_evidence': {}, 'reference_raw_utf8': 'INVENTED-REFERENCE',
                  'request_raw_utf8': request.decode(), 'request_sha256': S.digest(request),
                  'claim_raw_utf8': claim.decode(), 'claim_sha256': S.digest(claim)}
        binary = b'\x7fELF-INVENTED-NOT-EXECUTABLE'
        (root / 'en63_generated.h').write_bytes(header)
        (root / 'en63_librispeech').write_bytes(binary)
        source = S.source_hashes()
        build = {'schema': 'english63-librispeech-build-v1', 'compiler': '/invented/gcc',
                 'compiler_realpath': '/invented/gcc', 'compiler_sha256': 'e' * 64,
                 'compiler_version': 'INVENTED-NOT-EXECUTED',
                 'command': S.command_for('/invented/gcc', S.ROOT, root), 'exit_code': 0, 'stdout': '', 'stderr': '',
                 'source_root': str(S.ROOT), 'output_root': str(root), 'source_sha256': source,
                 'binary_sha256': S.digest(binary), 'header_sha256': S.p.HEADER_SHA, 'public_run': run,
                 'pipeline_head': head, 'physical_ssc305': False, 'machine': 'invented', 'libc': ['invented', '0']}
        integrity = {'source_before': source, 'source_after': source,
                     'header_before': S.p.HEADER_SHA, 'header_after': S.p.HEADER_SHA,
                     'corpus_before': S.CORPUS['sha256'], 'corpus_after': S.CORPUS['sha256'],
                     'reference_before': S.REFERENCE['member_sha256'], 'reference_after': S.REFERENCE['member_sha256'],
                     'pcm_before': 'd' * 64, 'pcm_after': 'd' * 64, 'parent_files_before': {}, 'parent_files_after': {},
                     'binary_before': S.digest(binary), 'binary_after': S.digest(binary),
                     'request_before': S.digest(request), 'request_after': S.digest(request),
                     'claim_before': S.digest(claim), 'claim_after': S.digest(claim)}
        measurements = {'schema': 'english63-librispeech-measurements-v1', 'public_run': run, 'pipeline_head': head,
                        'request_sha256': S.digest(request), 'initial_allowed_cpus': [2, 3], 'selected_cpu': 2,
                        'final_allowed_cpus': [2, 3], 'native_group_elapsed_seconds': 1,
                        'passes': [invented_pass(kind, root) for kind in S.p.c.KINDS], 'source_integrity': integrity}
        for kind in S.p.c.KINDS:
            (root / (kind + '.trace')).write_bytes(raw)
        (root / 'native.stderr').write_bytes(b'')
        for name, body in (('PROTOCOL.json', S.protocol()), ('INPUTS.json', inputs),
                           ('BUILD.json', build), ('MEASUREMENTS.json', measurements)):
            (root / name).write_bytes(encoded(body))
        with zipfile.ZipFile(root / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in S.SOURCES:
                archive.writestr(name, (S.ROOT / name).read_bytes())
        trace = S.trace_observation(raw, rows)
        models = [S.pass_observation(record, kind, build, 2, trace)
                  for kind, record in zip(S.p.c.KINDS, measurements['passes'])]
        summary = S.summary_from({**inputs, 'file_sha256': S.sha_file(root / 'INPUTS.json')}, build,
                                 {**measurements, 'file_sha256': S.sha_file(root / 'MEASUREMENTS.json')}, models)
        (root / 'SUMMARY.json').write_bytes(encoded(summary))
        receipt = S.saved_receipt(root)
        (root / 'REPLAY.json').write_bytes(encoded(receipt))
        yield root, receipt


class SavedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = S.TRACE_HEADER + b''.join(S.TRACE_RECORD.pack(i, 400 + (i - 1) * 160, 1, 0, 0, 0)
                                          for i in range(1, 366538))

    def test_local_entry_rejected_before_inputs_or_child(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(S.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY'):
                S.run(None)
            process.assert_not_called()

    def test_saved_only_rejects_original_input_arguments(self):
        with patch.object(S, 'run') as execute, patch.object(S, 'verify_saved') as verify:
            with self.assertRaisesRegex(ValueError, 'SAVED_ONLY_MODE'):
                S.main(['--verify-result-dir', '/invented', '--source-tar', '/invented.tar.gz'])
            execute.assert_not_called()
            verify.assert_not_called()

    def test_fixed_request_rejects_budget_and_model_order_changes(self):
        self.assertEqual(S.protocol()['schema'], 'english63-librispeech-screen-protocol-v1')
        self.assertEqual(S.protocol()['classes'], {'background': 0, 'Hey Vosen': 1, 'Hey Mero': 2})
        request = {**S.request_fixed(), 'pipeline_commit': 'a' * 40, 'reason': 'invented'}
        S.validate_request(encoded(request), 'a' * 40)
        for key, value in (('full_passes', 3), ('native_child_seconds', 181), ('models', ['dsc_tcn', 'gru'])):
            changed = json.loads(encoded(request))
            changed['profile'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                S.validate_request(encoded(changed), 'a' * 40)

    def test_source_reorder_splice_and_role_changes_rejected(self):
        reference, rows = invented_reference()
        S.verify_roster(rows, reference)
        for key, value in (('sample_count', 1), ('concatenation_start_sample', 1), ('human_gold', True),
                           ('flac_sha256', 'e' * 64), ('transcript_sha256', 'f' * 64), ('speaker_source_id', '999')):
            changed = json.loads(encoded(rows))
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                S.verify_roster(changed, reference)
        with self.assertRaises(ValueError):
            S.verify_roster(rows[::-1], reference)

    def test_trace_clock_probability_event_and_tail_forgery_rejected(self):
        _, rows = invented_reference()
        for offset, format_, value in ((16, '<Q', 2), (24, '<Q', 401), (32, '<f', float('nan')), (44, '<I', 1)):
            bad = bytearray(self.raw)
            struct.pack_into(format_, bad, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                S.trace_observation(bytes(bad), rows)
        for raw in (self.raw[:-1], self.raw + self.raw[-32:]):
            with self.assertRaises(ValueError):
                S.trace_observation(raw, rows)

    def test_class_names_follow_original_training_bridge(self):
        rows = [{'member': 'invented-one', 'sample_count': 8560, 'concatenation_start_sample': 0}]
        for chosen, name in ((1, 'Hey Vosen'), (2, 'Hey Mero')):
            probabilities = [0.1, 0.1, 0.1]
            probabilities[chosen] = 0.8
            raw = S.TRACE_HEADER + b''.join(S.TRACE_RECORD.pack(
                frame, 400 + (frame - 1) * 160, *probabilities, chosen if frame == 52 else 0)
                for frame in range(1, 53))
            observed = S.trace_observation(raw, rows, 8560)
            self.assertEqual(observed['proposal_count'], 1)
            self.assertEqual(observed['events'][0]['keyword'], name)

    def test_complete_saved_roundtrip_has_no_executable_activity(self):
        with invented_bundle(self.raw) as (root, receipt), patch.object(S.subprocess, 'Popen') as process:
            self.assertEqual(S.verify_saved(root), receipt)
            self.assertEqual({p.name for p in root.iterdir()}, S.RESULT_FILES)
            self.assertEqual(receipt['model_forward_calls'], 0)
            process.assert_not_called()

    def test_rehashed_summary_cannot_forge_event_count_or_gold_far(self):
        with invented_bundle(self.raw) as (root, receipt):
            summary = json.loads((root / 'SUMMARY.json').read_bytes())
            summary['models'][0]['proposal_count'] = 1
            summary['continuous_negative_far_per_hour'] = 0
            (root / 'SUMMARY.json').write_bytes(encoded(summary))
            receipt['files']['SUMMARY.json'] = S.p.artifact_identity((root / 'SUMMARY.json').read_bytes())
            receipt['files_sha256']['SUMMARY.json'] = S.sha_file(root / 'SUMMARY.json')
            receipt['summary_sha256'] = receipt['files_sha256']['SUMMARY.json']
            (root / 'REPLAY.json').write_bytes(encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'SUMMARY_RECOMPUTED'):
                S.verify_saved(root)

    def test_rehashed_trace_event_forgery_rejected_by_saved_decoder(self):
        with invented_bundle(self.raw) as (root, receipt):
            path = root / 'gru.trace'
            raw = bytearray(path.read_bytes())
            struct.pack_into('<I', raw, 16 + 51 * 32 + 28, 1)
            path.write_bytes(raw)
            receipt['files']['gru.trace'] = S.p.artifact_identity(bytes(raw))
            receipt['files_sha256']['gru.trace'] = S.sha_file(path)
            (root / 'REPLAY.json').write_bytes(encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'RAW_PROBABILITY_DECODER_EVENT'):
                S.verify_saved(root)

    def test_trace_pair_pass_order_and_affinity_are_fixed(self):
        with invented_bundle(self.raw) as (root, _):
            measurements = json.loads((root / 'MEASUREMENTS.json').read_bytes())
            measurements['passes'].reverse()
            (root / 'MEASUREMENTS.json').write_bytes(encoded(measurements))
            with self.assertRaisesRegex(ValueError, 'NATIVE_TERMINAL'):
                S.verify_saved(root)


def run(compiler, flags, directory):
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(SavedContracts))
    if not result.wasSuccessful():
        raise ValueError('Invented LibriSpeech saved-record tests failed')
    native_fixture(compiler, flags, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise ValueError('LibriSpeech source fixture compiler unavailable')
    flags = list(S.p.FLAGS)
    if args.sanitize:
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    with tempfile.TemporaryDirectory(prefix='en63-librispeech-invented-') as directory:
        run(compiler, flags, Path(directory))


if __name__ == '__main__':
    main()
