"""Only fake external exceptions; no runtime imports, sessions or model calls."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import run_single_melo as runner


class FakeExternalOrtError(Exception):
    pass


def fail_with_external_chain(*args):
    local_only = 'UNRELATED_LOCAL_VALUE_MUST_NOT_BE_CAPTURED'
    message = 'EXTERNAL_GRAPH_MESSAGE /private/model.onnx'
    cause = 'EXTERNAL_CHAIN_CAUSE /private/cache'
    try:
        raise RuntimeError(cause)
    except RuntimeError as error:
        raise FakeExternalOrtError(message) from error


class PrivateCauseProof(unittest.TestCase):
    def test_child_retains_chain_before_sanitized_record_without_locals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original_write = runner.write_json
            failed_stage_seen = []
            def ordered_write(path, value, **kwargs):
                if path.name == 'stage.json' and value.get('status') == 'failed_no_retry':
                    self.assertTrue((root / runner.PRIVATE_CAUSE_FILENAME).is_file())
                    failed_stage_seen.append(True)
                return original_write(path, value, **kwargs)
            with patch.dict(os.environ, {'MELO_SINGLE_SUPERVISED_PLAN_SHA256':'a'*64}), \
                 patch.object(runner.os,'getpgrp',return_value=os.getpid()), \
                 patch.object(runner.resource,'setrlimit'), \
                 patch.object(runner,'validate_plan',side_effect=fail_with_external_chain), \
                 patch.object(runner,'write_json',side_effect=ordered_write):
                self.assertEqual(runner.armed_child(Path('not-read'),root,Path('not-read'),'a'*64),1)
            raw_path = root / runner.PRIVATE_CAUSE_FILENAME
            private = json.loads(raw_path.read_bytes())
            sanitized = (root / 'stage.json').read_text()
            self.assertEqual(stat.S_IMODE(raw_path.stat().st_mode),0o600)
            self.assertIn('EXTERNAL_GRAPH_MESSAGE',private['traceback_chain'])
            self.assertIn('EXTERNAL_CHAIN_CAUSE',private['traceback_chain'])
            self.assertIn('direct cause',private['traceback_chain'])
            self.assertNotIn('UNRELATED_LOCAL_VALUE_MUST_NOT_BE_CAPTURED',private['traceback_chain'])
            self.assertNotIn('EXTERNAL_GRAPH_MESSAGE',sanitized)
            self.assertNotIn('EXTERNAL_CHAIN_CAUSE',sanitized)
            self.assertNotIn('/private/',sanitized)
            self.assertFalse(private['capture_locals'])
            self.assertFalse(private['public_artifact_allowed'])
            self.assertFalse(private['truncated'])
            self.assertEqual(private['traceback_total_bytes'],private['traceback_retained_bytes'])
            self.assertEqual(failed_stage_seen,[True])

    def test_json_escaping_and_unicode_cannot_exceed_private_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            error = FakeExternalOrtError(('中\n\t"' * 40000) + 'TAIL_SENTINEL')
            receipt = runner.retain_private_cause(root,error)
            raw = (root / runner.PRIVATE_CAUSE_FILENAME).read_bytes()
            value = json.loads(raw)
            self.assertLessEqual(len(raw),64*1024)
            self.assertTrue(value['truncated'])
            self.assertTrue(receipt['truncated'])
            self.assertGreater(value['traceback_total_bytes'],value['traceback_retained_bytes'])
            self.assertEqual(value['traceback_retained_bytes'],len(value['traceback_chain'].encode('utf-8')))
            self.assertNotIn('TAIL_SENTINEL',value['traceback_chain'])

    def test_private_receipt_uses_existing_receipt_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'occupied.json').write_bytes(b' ' * runner.RECEIPT_CAP)
            with self.assertRaisesRegex(ValueError,'PRIVATE_CAUSE_RECEIPT_BUDGET'):
                runner.retain_private_cause(root,FakeExternalOrtError('small cause'))
            self.assertFalse((root / runner.PRIVATE_CAUSE_FILENAME).exists())


if __name__ == '__main__':
    unittest.main()
