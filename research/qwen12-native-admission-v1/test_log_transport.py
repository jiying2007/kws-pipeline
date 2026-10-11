"""Synthetic-only log transport tests. No recognizer, network, or real audio."""
import base64
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_transport as t

IDENTITY = {"repository": "jiying2007/kws-pipeline", "run_id": 123,
    "run_attempt": 1, "job_key": "qwen12-native", "executed_head_sha": "a" * 40}

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "receipts").mkdir()
        (self.root / "receipts/fixture.json").write_text(json.dumps({"synthetic_fixture": True,
            "native_completion": "unknown", "random": random.Random(7).randbytes(16000).hex()}))
        self.allowed = ["receipts/fixture.json"]
        self.payload, self.manifest, self.lines = t.build_package(self.root, self.allowed, IDENTITY)

    def tearDown(self):
        self.temp.cleanup()

    def log(self, lines=None):
        return "\ufeff2026-10-11T10:20:30.1234567Z Runner fixture\n" + "\n".join(
            "2026-10-11T10:20:30.1234567Z " + line for line in (self.lines if lines is None else lines)) + "\n"

    def mutate(self, index, fn):
        lines = list(self.lines)
        record = json.loads(lines[index][len(t.PREFIX):])
        fn(record)
        lines[index] = t.PREFIX + t.canonical(record).decode()
        return lines

    def test_round_trip_timestamps_bom(self):
        payload, manifest, receipt = t.parse_job_log(self.log(), IDENTITY)
        self.assertEqual(payload, self.payload)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(receipt["archive_sha256"], t.sha(self.payload))
        self.assertIsNone(receipt["github_actions_artifact_id"])

    def test_deterministic(self):
        self.assertEqual(t.build_package(self.root, self.allowed, IDENTITY)[0], self.payload)

    def test_plain_lines(self):
        self.assertEqual(t.parse_job_log("\n".join(self.lines), IDENTITY)[0], self.payload)

    def test_missing_chunk(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.lines[:2] + self.lines[3:]), IDENTITY)

    def test_duplicate_chunk(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.lines[:2] + [self.lines[1]] + self.lines[2:]), IDENTITY)

    def test_reordered_chunks(self):
        lines = list(self.lines); lines[1], lines[2] = lines[2], lines[1]
        with self.assertRaises(ValueError): t.parse_job_log(self.log(lines), IDENTITY)

    def test_truncated_no_end(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.lines[:-1]), IDENTITY)

    def test_truncated_chunk(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.mutate(1, lambda r: r.update(data=r['data'][:-4]))), IDENTITY)

    def test_mixed_run(self):
        other = dict(IDENTITY, run_id=124)
        with self.assertRaises(ValueError): t.parse_job_log(self.log(), other)

    def test_mixed_head_job_attempt(self):
        for key, value in [("executed_head_sha", "b" * 40), ("job_key", "other"), ("run_attempt", 2)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                t.parse_job_log(self.log(), dict(IDENTITY, **{key: value}))

    def test_mixed_chunk(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.mutate(1, lambda r: r.update(header_sha256='b'*64))), IDENTITY)

    def test_duplicate_package(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.lines + self.lines), IDENTITY)

    def test_wrong_payload_digest(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.mutate(0, lambda r: r.update(archive_sha256='b'*64))), IDENTITY)

    def test_oversized_header(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.mutate(0, lambda r: r.update(uncompressed_bytes=t.MAX_UNCOMPRESSED+1))), IDENTITY)

    def test_bool_integer(self):
        with self.assertRaises(ValueError): t.parse_job_log(self.log(self.mutate(1, lambda r: r.update(index=False))), IDENTITY)

    def test_bool_footer(self):
        (self.root / 'small.json').write_text('{}')
        _, _, lines = t.build_package(self.root, ['small.json'], IDENTITY)
        footer = json.loads(lines[-1][len(t.PREFIX):]); footer['chunk_count'] = True
        lines[-1] = t.PREFIX + t.canonical(footer).decode()
        with self.assertRaises(ValueError): t.parse_job_log(self.log(lines), IDENTITY)

    def test_unsafe_names(self):
        for name in ['../outside.json', '/absolute.json', 'a/../b.json', 'a//b.json', '.env', 'audio.wav', 'model.pt', 'weights.safetensors', 'secrets.key', 'a\0.json', 'a\n.json', 'a\x7f.json']:
            with self.subTest(name=name), self.assertRaises(ValueError): t.build_package(self.root, [name], IDENTITY)

    def test_member_symlink(self):
        (self.root / 'alias.json').symlink_to(self.root / self.allowed[0])
        with self.assertRaises((ValueError, OSError)): t.build_package(self.root, ['alias.json'], IDENTITY)

    def test_directory_symlink(self):
        (self.root / 'alias').symlink_to(self.root / 'receipts', target_is_directory=True)
        with self.assertRaises((ValueError, OSError)): t.build_package(self.root, ['alias/fixture.json'], IDENTITY)

    def test_disguised_wav(self):
        (self.root / 'audio.json').write_bytes(b'RIFF\0\0\0\0WAVE')
        with self.assertRaises(ValueError): t.build_package(self.root, ['audio.json'], IDENTITY)

    def test_oversized_member(self):
        (self.root / 'large.json').write_bytes(b'x' * (t.MAX_MEMBER+1))
        with self.assertRaises(ValueError): t.build_package(self.root, ['large.json'], IDENTITY)

    def test_compressed_cap(self):
        for seed, name in [(9, 'random-a.json'), (10, 'random-b.json')]:
            (self.root / name).write_bytes(random.Random(seed).randbytes(t.MAX_COMPRESSED//2+5000))
        with self.assertRaisesRegex(ValueError, 'compressed evidence cap'):
            t.build_package(self.root, ['random-a.json', 'random-b.json'], IDENTITY)

    def test_large_timestamped_log_roundtrip(self):
        (self.root / 'large-fixture.json').write_text(json.dumps({'synthetic_fixture': True,
            'hex': random.Random(42).randbytes(820000).hex()}))
        payload, manifest, lines = t.build_package(self.root, ['large-fixture.json'], IDENTITY)
        log = self.log(lines)
        self.assertGreater(len(log.encode()), 1024 * 1024)
        self.assertLessEqual(len(payload), t.MAX_COMPRESSED)
        observed, observed_manifest, _ = t.parse_job_log(log, IDENTITY)
        self.assertEqual(observed, payload)
        self.assertEqual(observed_manifest, manifest)

    def test_no_execution(self):
        marker = self.root / 'should-not-exist'
        (self.root / 'request.json').write_text(json.dumps({'instruction': f'write {marker}'}))
        _, _, lines = t.build_package(self.root, ['request.json'], IDENTITY)
        t.parse_job_log(self.log(lines), IDENTITY)
        self.assertFalse(marker.exists())

    def test_duplicate_json_key(self):
        lines = list(self.lines)
        lines[0] = lines[0].replace('{', '{"kind":"header",', 1)
        with self.assertRaises(ValueError): t.parse_job_log(self.log(lines), IDENTITY)

    def test_unrelated_log_cannot_substitute(self):
        with self.assertRaises(ValueError): t.parse_job_log('ordinary log says successful', IDENTITY)

if __name__ == '__main__': unittest.main(verbosity=2)
