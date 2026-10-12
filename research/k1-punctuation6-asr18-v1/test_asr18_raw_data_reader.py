"""Synthetic bytes and stdlib fake HTTP only; never actual data or execution."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import zipfile
import unittest
from unittest import mock
import urllib.error

import raw_data_reader as r


def synthetic_authority(payload=b"synthetic-not-a-real-archive"):
    freeze_raw = (r.HOST_TTS_VERIFIER_ROOT / r.FREEZE_PATH).read_bytes()
    files = json.loads(freeze_raw)["files"]
    inp = {"repository": "jiying2007/kws-data", "data_commit": "d" * 40, "source_commit": "a" * 40,
        "run_id": 701, "run_attempt": 1, "job_id": 702, "job_key": "k1-tts",
        "source_freeze_sha256": r.sha(freeze_raw), "archive_sha256": r.sha(payload), "archive_bytes": len(payload),
        "outer_manifest_sha256": "e" * 64, "member_manifest_sha256": "f" * 64}
    authority = {"schema": "k1-asr18-reviewed-tts-authority-v1", "input": inp,
                 "source_tree": "c" * 40, "data_parent": "a" * 40, "source_files": files}
    chunks = [payload[start:start + r.MAX_CHUNK] for start in range(0, len(payload), r.MAX_CHUNK)]
    manifest = {"schema": "k1-data-branch-manifest-v1", "identity": r.run_identity(authority),
        "purpose": "tts", "fixture_only": False, "source_closure_sha256": inp["source_freeze_sha256"],
        "numeric_job_id": 702, "archive_bytes": len(payload), "archive_sha256": r.sha(payload),
        "chunk_bytes": r.MAX_CHUNK, "chunk_count": len(chunks), "chunks": [
            {"index": index, "path": r.DATA_DIRECTORY + f"/chunks/{index:04d}.bin", "bytes": len(raw), "sha256": r.sha(raw)}
            for index, raw in enumerate(chunks)]}
    raw = r.canonical(manifest)
    inp["outer_manifest_sha256"] = r.sha(raw)
    return authority, raw, chunks


class FakeResponse(io.BytesIO):
    def __init__(self, raw, url, status=200, encoding="identity"):
        super().__init__(raw)
        self.url, self.status, self.headers = url, status, {"Content-Encoding": encoding}
        self.read_sizes = []
    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)
    def getcode(self):
        return self.status
    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, bodies, *, status=200, encoding="identity", url_override=None):
        self.bodies, self.calls, self.responses = list(bodies), [], []
        self.status, self.encoding, self.url_override = status, encoding, url_override
    def open(self, request, timeout):
        self.calls.append((request.full_url, request.method, dict(request.header_items()), timeout))
        raw = self.bodies.pop(0)
        if isinstance(raw, BaseException):
            raise raw
        response = FakeResponse(raw, self.url_override or request.full_url, self.status, self.encoding)
        self.responses.append(response)
        return response


_SYNTHETIC_ARCHIVE_SCRIPT = r'''import sys, pathlib, json, hashlib
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "research/k1-punctuation6-v1"))
from test_tts_semantics import archive, IDENTITY
from test_tts_public_schema import private_setup, model_load
import tts_public_schema as schema
from tts_semantics import validate_tts_members
from data_transport import build_archive, canonical
members = archive()
source = json.loads(members["source.json"])
freeze_raw = (root / "research/k1-punctuation6-v1/tts-source-freeze.json").read_bytes()
freeze = json.loads(freeze_raw)
source["source_files"] = freeze["files"]
source["frozen_files"] = len(freeze["files"])
source["source_freeze_sha256"] = hashlib.sha256(freeze_raw).hexdigest()
claim = json.loads(members["claim.json"])
claim["source_freeze_sha256"] = source["source_freeze_sha256"]
members["claim.json"] = canonical(claim)
private = private_setup()
packages, sources, assets, _ = schema.inventory()
for name, pin in packages.items():
    private['wheels'][name] = {'path': 'downloads/' + pin['filename'],
        'bytes': pin['size_bytes'] if pin['kind'] == 'wheel' else 50000,
        'sha256': pin['sha256'] if pin['kind'] == 'wheel' else 'a' * 64,
        'name': pin['name'], 'version': pin['version'], 'metadata_sha256': pin['metadata_sha256'] or 'b' * 64,
        'metadata_bytes': pin.get('metadata_bytes') or 1024, 'requires_dist': list(pin['requires_dist'])}
private['bootstrap_pip'] = {'source': 'exact_python_bundled_ensurepip_not_independent_lock',
    'filename': 'pip-25.0.1-py3-none-any.whl', 'bytes': 123, 'sha256': 'a' * 64, 'version': '25.0.1', 'metadata_sha256': 'b' * 64}
private['commands'] = [{'argv': ['/synthetic/interpreter'], 'log': 'logs/synthetic', 'stage': 'verify_setup',
    'package': None, 'command_role': 'verification', 'status': 'success', 'returncode': 0}]
private['verification'] = {'installed': [], 'source_members': [dict(row, bytes=123) for row in sources],
    'model_assets': {'qwen_tts': assets}, 'disk': {key: 0 for key in ('installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes')},
    'python': dict(schema.PYTHON)}
private['import_preflight'] = {'schema': 'native-import-preflight-v1', 'profile': 'tts', 'stage': 'complete',
    'status': 'success', 'weight_loads': 0, 'model_constructions': 0, 'forward_calls': 0, 'raw_text_disclosed': False,
    'versions': {'numpy': packages['numpy']['version']},
    'native_io': {'frames': 12, 'float32_values_sha256': 'c' * 64, 'float_wav_sha256': 'd' * 64,
        'pcm_wav_sha256': 'e' * 64, 'pcm16_values': [-32768, -32768, -16384, -2, -1, 0, 0, 1, 16384, 32767, 32767, 32767],
        'bundled_native_module': '_soundfile_data/libsndfile_x86_64.so', 'bundled_native_sha256': 'f' * 64,
        'libsndfile_version': '1.2.2', 'empty_nonfinite_rejection_checked': True}}
source["runtime_setup"] = schema.project_setup(private)
source["runtime_model_load"] = schema.project_load(model_load('/synthetic/runtime'), '/synthetic/runtime')
if len(sys.argv) > 2:
    source[sys.argv[2]] = {"status": "not_available"}
members["source.json"] = canonical(source)
sys.stdout.buffer.write(build_archive(members, IDENTITY, "tts", validate_tts_members))
'''


def synthetic_complete_archive(unavailable=None):
    with tempfile.TemporaryFile() as output:
        args = [sys.executable, "-I", "-S", "-B", "-c", _SYNTHETIC_ARCHIVE_SCRIPT, str(r.HOST_TTS_VERIFIER_ROOT)]
        if unavailable:
            args.append(unavailable)
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.PIPE,
                                timeout=20, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        if result.returncode:
            raise AssertionError(result.stderr.decode())
        output.seek(0)
        payload = output.read(r.MAX_ARCHIVE + 1)
    authority, raw, chunks = synthetic_authority(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        authority["input"]["member_manifest_sha256"] = r.sha(archive.read(r.MEMBER_MANIFEST))
    return payload, authority


class RawReaderTests(unittest.TestCase):
    def test_maximum_archive_exact_49_gets_and_manifest_counted(self):
        payload = b"x" * r.MAX_ARCHIVE
        authority, manifest, chunks = synthetic_authority(payload)
        opener = FakeOpener([manifest, *chunks])
        reader = r.RawDataReader(_opener=opener)
        result = reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(result["archive"], payload)
        self.assertEqual(reader.gets, 49)
        self.assertEqual(reader.response_body_bytes, len(manifest) + len(payload))
        self.assertLess(reader.response_body_bytes, r.MAX_BODY)
        self.assertTrue(all(call[1] == "GET" and "Range" not in call[2] and "?" not in call[0] for call in opener.calls))
        self.assertTrue(all("/" + authority["input"]["data_commit"] + "/" in call[0] for call in opener.calls))
        self.assertFalse(result["accounting"]["execution_enabled"])

    def test_one_reader_cannot_retry_after_success(self):
        authority, manifest, chunks = synthetic_authority()
        reader = r.RawDataReader(_opener=FakeOpener([manifest, *chunks]))
        kwargs = {"expected_authority_sha256": r.metadata.digest(authority)}
        reader.read_complete(authority, **kwargs)
        with self.assertRaises(ValueError): reader.read_complete(authority, **kwargs)
        self.assertEqual(reader.gets, 2)

    def test_unknown_authority_blocks_before_first_get(self):
        authority, manifest, chunks = synthetic_authority()
        for value in (None, "a" * 64):
            reader = r.RawDataReader(_opener=FakeOpener([manifest, *chunks]))
            with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=value)
            self.assertEqual(reader.gets, 0)

    def test_unknown_actual_fields_and_wrong_job_or_parent_rejected(self):
        authority = synthetic_authority()[0]
        for key, value in (("source_commit", None), ("data_commit", None), ("run_id", True), ("job_key", "writer-fixture"), ("run_attempt", 2)):
            changed = deepcopy(authority); changed["input"][key] = value
            with self.assertRaises(ValueError): r.validate_authority(changed, expected_authority_sha256=r.metadata.digest(changed))
        changed = deepcopy(authority); changed["data_parent"] = "e" * 40
        with self.assertRaises(ValueError): r.validate_authority(changed, expected_authority_sha256=r.metadata.digest(changed))

    def test_fixed_complete_source_roster_and_pinned_verifier(self):
        authority = synthetic_authority()[0]
        for change in (lambda a: a["source_files"].pop(next(iter(a["source_files"]))),
                       lambda a: a["source_files"].update({"private/inventory.json": {"bytes": 1, "sha256": "e" * 64}}),
                       lambda a: a["source_files"][r.SOURCE_DIRECTORY + "/tts_semantics.py"].update(sha256="e" * 64)):
            modified = deepcopy(authority); change(modified)
            with self.assertRaises(ValueError): r.validate_authority(modified, expected_authority_sha256=r.metadata.digest(modified))

    def test_error_response_body_counted_and_terminal(self):
        authority = synthetic_authority()[0]
        reader = r.RawDataReader(_opener=FakeOpener([b"error page"], status=500))
        with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(reader.response_body_bytes, len(b"error page"))
        self.assertEqual(reader.gets, 1)
        self.assertTrue(reader.failed)
        with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))

    def test_http_error_exception_body_counted(self):
        authority = synthetic_authority()[0]
        url = "https://raw.githubusercontent.com/jiying2007/kws-data/" + authority["input"]["data_commit"] + "/" + r.DATA_DIRECTORY + "/manifest.json"
        error = urllib.error.HTTPError(url, 403, "blocked", {}, io.BytesIO(b"denied"))
        reader = r.RawDataReader(_opener=FakeOpener([error]))
        with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(reader.response_body_bytes, 6)
        self.assertEqual(reader.gets, 1)

    def test_overlimit_manifest_probe_byte_counted(self):
        authority = synthetic_authority()[0]
        reader = r.RawDataReader(_opener=FakeOpener([b"x" * (r.MAX_MANIFEST + 20)]))
        with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(reader.response_body_bytes, r.MAX_MANIFEST + 1)

    def test_overlimit_chunk_probe_byte_counted(self):
        authority, manifest, chunks = synthetic_authority(b"x" * r.MAX_CHUNK)
        reader = r.RawDataReader(_opener=FakeOpener([manifest, chunks[0] + b"x"]))
        with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(reader.response_body_bytes, len(manifest) + r.MAX_CHUNK + 1)

    def test_total_body_cap_never_reads_extra_detection_byte(self):
        reader = r.RawDataReader(_opener=FakeOpener([b"x" * 100]))
        reader.response_body_bytes = r.MAX_BODY - 3
        with self.assertRaises(ValueError): reader._get("a" * 40, r.DATA_DIRECTORY + "/manifest.json", r.MAX_MANIFEST)
        self.assertEqual(reader.response_body_bytes, r.MAX_BODY)

    def test_redirect_or_compressed_body_counted_then_rejected(self):
        authority = synthetic_authority()[0]
        for kwargs in ({"status": 302}, {"url_override": "https://other.test/data"}, {"encoding": "gzip"}):
            reader = r.RawDataReader(_opener=FakeOpener([b"bad"], **kwargs))
            with self.assertRaises(ValueError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
            self.assertEqual(reader.response_body_bytes, 3)
            self.assertEqual(reader.gets, 1)
        self.assertIsNone(r.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test"))

    def test_failed_open_consumes_get_without_retry(self):
        authority = synthetic_authority()[0]
        opener = FakeOpener([OSError("synthetic")])
        reader = r.RawDataReader(_opener=opener)
        with self.assertRaises(OSError): reader.read_complete(authority, expected_authority_sha256=r.metadata.digest(authority))
        self.assertEqual(reader.gets, 1)
        self.assertEqual(reader.response_body_bytes, 0)

    def test_manifest_omission_duplicate_path_or_self_asserted_source_rejected(self):
        for mutate in (lambda m: m["chunks"].pop(), lambda m: m["chunks"][0].update(path="arbitrary"),
                       lambda m: m["identity"].update(executed_head_sha="b" * 40),
                       lambda m: m.update(fixture_only=True), lambda m: m.update(numeric_job_id=True)):
            authority, raw, _ = synthetic_authority()
            m = json.loads(raw); mutate(m); raw = r.canonical(m)
            authority["input"]["outer_manifest_sha256"] = r.sha(raw)
            with self.assertRaises(ValueError): r.parse_manifest(raw, authority)

    def test_duplicate_noncanonical_json_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{ "x":1}', b'{"x":NaN}', b'[]'):
            with self.assertRaises(ValueError): r.decode(raw)

    def test_real_reference_closure_only_synthetic_authority(self):
        authority = synthetic_authority()[0]
        snapshot = r.verified_source_snapshot(authority, expected_authority_sha256=r.metadata.digest(authority),
            expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])
        self.assertEqual(len(snapshot), 43)
        self.assertEqual(len(authority["source_files"]), 42)
        self.assertFalse(r.EXECUTION_ENABLED)
        self.assertFalse(any("__pycache__" in name for name in snapshot))

    def test_missing_unreviewed_or_wrong_freeze_blocks(self):
        authority = synthetic_authority()[0]
        for expected in (None, "a" * 64):
            with self.assertRaises(ValueError): r.verified_source_snapshot(authority,
                expected_authority_sha256=r.metadata.digest(authority), expected_source_freeze_sha256=expected)

    def test_extra_file_and_symlink_source_rejected(self):
        authority = synthetic_authority()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.py").write_bytes(b"pass")
            with mock.patch.object(r, "HOST_TTS_VERIFIER_ROOT", root), self.assertRaises(ValueError):
                r.verified_source_snapshot(authority, expected_authority_sha256=r.metadata.digest(authority),
                    expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])
            (root / "unexpected.py").unlink()
            (root / "alias").symlink_to(r.HOST_TTS_VERIFIER_ROOT)
            with mock.patch.object(r, "HOST_TTS_VERIFIER_ROOT", root), self.assertRaises(ValueError):
                r.verified_source_snapshot(authority, expected_authority_sha256=r.metadata.digest(authority),
                    expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])

    def test_isolated_verifier_rejects_nonzip_without_model_or_network(self):
        authority = synthetic_authority()[0]
        with self.assertRaisesRegex(ValueError, "semantic/source verification failed"):
            r.verify_complete_package(b"synthetic-not-a-real-archive", authority,
                expected_authority_sha256=r.metadata.digest(authority),
                expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])


    def test_complete_synthetic_zip_passes_pinned_semantics_without_runtime(self):
        payload, authority = synthetic_complete_archive()
        members = r.verify_complete_package(payload, authority,
            expected_authority_sha256=r.metadata.digest(authority),
            expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])
        self.assertEqual(len(members), 23)
        self.assertEqual(json.loads(members["outcomes.json"])["status"], "completed_six_once")
        self.assertFalse(r.EXECUTION_ENABLED)

    def test_complete_claim_cannot_hide_missing_setup_or_load_evidence(self):
        for unavailable in ("runtime_setup", "runtime_model_load"):
            payload, authority = synthetic_complete_archive(unavailable)
            with self.assertRaisesRegex(ValueError, "semantic/source verification failed"):
                r.verify_complete_package(payload, authority,
                    expected_authority_sha256=r.metadata.digest(authority),
                    expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])

    def test_complete_zip_wrong_tree_job_member_manifest_or_trailer_fails(self):
        payload, authority = synthetic_complete_archive()
        for key, value in (("source_tree", "e" * 40), ("job_id", 703), ("member_manifest_sha256", "e" * 64)):
            changed = deepcopy(authority)
            (changed if key == "source_tree" else changed["input"])[key] = value
            with self.assertRaises(ValueError):
                r.verify_complete_package(payload, changed, expected_authority_sha256=r.metadata.digest(changed),
                    expected_source_freeze_sha256=changed["input"]["source_freeze_sha256"])
        changed = deepcopy(authority)
        changed["input"].update(archive_bytes=len(payload) + 1, archive_sha256=r.sha(payload + b"x"))
        with self.assertRaises(ValueError):
            r.verify_complete_package(payload + b"x", changed, expected_authority_sha256=r.metadata.digest(changed),
                expected_source_freeze_sha256=changed["input"]["source_freeze_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
