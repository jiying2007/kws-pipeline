"""Offline positive/tamper regressions for the public PR450 evidence pointer."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('durable', ROOT / 'tools/verify_durable_trace.py')
DURABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DURABLE)
DIRECTORY = ROOT / DURABLE.DIRECTORY


class DurableTraceTests(unittest.TestCase):
    def setUp(self):
        self.pointer = DURABLE.read_json(DIRECTORY / 'DURABLE-TRACE.json')
        self.readiness = DURABLE.read_json(DIRECTORY / 'PUBLIC-READINESS-SUMMARY.json')

    def test_current_pinned_metadata_passes_without_archive_access(self):
        with mock.patch('builtins.open', side_effect=AssertionError('no additional I/O allowed')), \
             mock.patch.object(Path, 'open', side_effect=AssertionError('no additional I/O allowed')):
            result = DURABLE.verify(self.pointer, self.readiness)
        self.assertEqual(result['original_members'], 44)
        self.assertEqual(result['wrapper_parts'], 3)
        self.assertEqual(result['archive_bytes_read'], 0)
        self.assertIs(result['restored_bytes_verified'], False)
        self.assertIs(result['execution_ready'], False)

    def test_original_three_silent_corruptions_now_fail(self):
        mutations = [lambda p: p.update(source_commit='0'*40),
                     lambda p: p['original_artifact'].update(sha256='0'*64),
                     lambda p: p['wrapper'].update(parts=[])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                pointer = copy.deepcopy(self.pointer)
                mutation(pointer)
                with self.assertRaises(ValueError):
                    DURABLE.verify(pointer, self.readiness)

    def test_every_frozen_pointer_scalar_and_container_is_enforced(self):
        # Every identity leaf plus missing/empty list/dict/extra nested key.
        def variants(value):
            if isinstance(value, dict):
                yield {}
                extra = copy.deepcopy(value); extra['unexpected'] = 0; yield extra
                for key, child in value.items():
                    missing = copy.deepcopy(value); del missing[key]; yield missing
                    for replacement in variants(child):
                        changed = copy.deepcopy(value); changed[key] = replacement; yield changed
            elif isinstance(value, list):
                yield []
                yield value + value[:1]
                for index, child in enumerate(value):
                    for replacement in variants(child):
                        changed = copy.deepcopy(value); changed[index] = replacement; yield changed
            elif isinstance(value, bool):
                yield not value
                yield int(value)
            elif isinstance(value, int):
                yield value + 1
                yield str(value)
            else:
                yield '0' * len(value)
        fields = ('source_repository', 'source_commit', 'source_path', 'source_url',
                  'original_artifact', 'wrapper', 'verification_inputs', 'scope')
        count = 0
        for field in fields:
            for replacement in variants(self.pointer[field]):
                changed = copy.deepcopy(self.pointer); changed[field] = replacement
                with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                    DURABLE.verify(changed, self.readiness)
                count += 1
        self.assertEqual(count, 127)

    def test_part_order_and_verification_input_order_are_explicit(self):
        for field, nested in (('wrapper', 'parts'), ('verification_inputs', None)):
            pointer = copy.deepcopy(self.pointer)
            (pointer[field][nested] if nested else pointer[field]).reverse()
            with self.assertRaises(ValueError):
                DURABLE.verify(pointer, self.readiness)

    def test_consistent_but_wrong_wrapper_or_original_identity_still_fails(self):
        pointer = copy.deepcopy(self.pointer)
        pointer['wrapper']['parts'][0]['bytes'] += 1
        pointer['wrapper']['archive_bytes'] += 1
        with self.assertRaises(ValueError):
            DURABLE.verify(pointer, self.readiness)
        pointer = copy.deepcopy(self.pointer)
        readiness = copy.deepcopy(self.readiness)
        pointer['original_artifact']['sha256'] = readiness['shipping_saved_evidence']['artifact_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            DURABLE.verify(pointer, readiness)

    def test_readiness_artifact_member_and_status_tampering_fail(self):
        first = next(iter(self.readiness['shipping_saved_evidence']['member_identities']))
        mutations = [lambda r: r.update(execution_ready=True),
                     lambda r: r.update(D20_original_status='PASS'),
                     lambda r: r.update(D90_status='PASS'),
                     lambda r: r.update(new_model_calls=1),
                     lambda r: r.update(new_operator_calls=False),
                     lambda r: r['shipping_saved_evidence'].update(artifact_id=0),
                     lambda r: r['shipping_saved_evidence'].update(run_id=0),
                     lambda r: r['shipping_saved_evidence'].update(artifact_sha256='0'*64),
                     lambda r: r['shipping_saved_evidence']['member_identities'].pop(first),
                     lambda r: r['shipping_saved_evidence']['member_identities'][first].update(sha256='0'*64),
                     lambda r: r['shipping_saved_evidence']['member_identities'][first].update(bytes=0)]
        for mutation in mutations:
            readiness = copy.deepcopy(self.readiness); mutation(readiness)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                DURABLE.verify(self.pointer, readiness)

    def test_json_reader_rejects_duplicate_keys_bad_encoding_and_oversize(self):
        for raw in (b'{"key":1,"key":2}', b'{"value":NaN}', b'\xff', b' '*65537):
            with mock.patch.object(Path, 'open', return_value=io.BytesIO(raw)):
                with self.assertRaises((ValueError, UnicodeError)):
                    DURABLE.read_json(Path('unused'))

    def test_navigation_uses_one_current_checklist_and_marks_old_protocol_historical(self):
        research = (ROOT / 'research/README.md').read_text()
        checklist = (ROOT / 'research/d20-diagnostic-admission-v1/PLAN-SCHEMA.md').read_text()
        historical = (DIRECTORY / 'PUBLIC-PROTOCOL.zh-CN.md').read_text()
        self.assertEqual(research.count('d20-diagnostic-admission-v1/PLAN-SCHEMA.md#current-admission-checklist'), 1)
        self.assertIn('Historical PR #502 saved-evidence supplement', research)
        self.assertEqual(checklist.count('## Current admission checklist'), 1)
        self.assertIn('Under the scope, verify exact dependencies and import-only peak', checklist)
        self.assertIn('不能作为当前执行清单', historical)
        self.assertNotIn('先安装/核对确切依赖', historical)

    def test_real_cli_pass_and_each_original_single_mutation_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / DURABLE.DIRECTORY
            directory.mkdir(parents=True)
            (directory / 'PUBLIC-READINESS-SUMMARY.json').write_text(json.dumps(self.readiness))
            for field in ('valid', 'commit', 'original_sha', 'parts'):
                pointer = copy.deepcopy(self.pointer)
                if field == 'commit': pointer['source_commit'] = '0'*40
                if field == 'original_sha': pointer['original_artifact']['sha256'] = '0'*64
                if field == 'parts': pointer['wrapper']['parts'] = []
                (directory / 'DURABLE-TRACE.json').write_text(json.dumps(pointer))
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_durable_trace.py'),
                                         '--root', temp], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, field == 'valid', result.stderr)


if __name__ == '__main__':
    unittest.main()
