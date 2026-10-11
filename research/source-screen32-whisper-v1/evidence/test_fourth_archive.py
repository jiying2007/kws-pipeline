#!/usr/bin/env python3
"""Offline retained-byte and invented adversarial archive tests; no acquisition."""
import copy
import binascii
import io
import json
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_fourth_archive as archive


class FourthArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carrier, cls.manifest, cls.files = archive.read_archive()

    def temp(self):
        directory = tempfile.TemporaryDirectory(prefix='.fourth-archive-test-', dir=HERE)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def copied_run(self):
        root = self.temp() / 'run'
        shutil.copytree(archive.RUN, root)
        return root

    def fixture(self, entries=None):
        entries = entries or [('ok.json', b'{"safe":true}')]
        buffer = io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            warnings.simplefilter('ignore', UserWarning)
            for name, content in entries:
                z.writestr(name, content)
        data = buffer.getvalue()
        rows = [{'path': n.filename if isinstance(n, zipfile.ZipInfo) else n, **archive.identity(b)} for n, b in entries]
        manifest = {'schema': 'whisper-screen-compact-archive-v1', 'archive_bytes': len(data),
                    'archive_sha256': archive.sha(data), 'logical_bytes': sum(len(b) for _, b in entries),
                    'chunks': [], 'members': rows}
        return data, manifest

    def changed_record(self, name, change):
        files = self.files.copy()
        record = archive.read_json(files[name])
        change(record)
        files[name] = json.dumps(record, ensure_ascii=False).encode()
        freeze = archive.read_json(files['artifact-freeze.json'])
        freeze['files'][name] = archive.identity(files[name])
        files['artifact-freeze.json'] = json.dumps(freeze).encode()
        return files

    def test_real_retained_archive_and_full_native_chains(self):
        result = archive.verify()
        self.assertEqual(result['completed_native_eot'], 16)
        self.assertEqual(result['archive_members'], 431)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('whisper', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_exact_zip_restoration_is_exclusive(self):
        path = self.temp() / 'terminal.zip'
        archive.verify(output=path)
        self.assertEqual(path.read_bytes(), self.carrier)
        with self.assertRaises(FileExistsError):
            archive.verify(output=path)
        self.assertEqual(path.read_bytes(), self.carrier)

    def test_changed_chunk(self):
        root = self.copied_run()
        path = root / 'chunks/terminal.zip.part-000'
        data = bytearray(path.read_bytes()); data[20] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'chunk identity'):
            archive.read_archive(root)

    def test_missing_or_extra_chunk(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                root = self.copied_run()
                if extra:
                    (root / 'chunks/extra').write_bytes(b'extra')
                else:
                    (root / 'chunks/terminal.zip.part-001').unlink()
                with self.assertRaises(ValueError):
                    archive.read_archive(root)

    def test_reordered_chunks(self):
        root = self.copied_run()
        path = root / 'ARCHIVE.json'; value = json.loads(path.read_text())
        value['chunks'].reverse(); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'chunk order'):
            archive.read_archive(root)

    def test_symlink_chunk_is_rejected(self):
        root = self.copied_run()
        path = root / 'chunks/terminal.zip.part-000'
        path.unlink(); path.symlink_to(archive.RUN / 'chunks/terminal.zip.part-000')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            archive.read_archive(root)

    def test_manifest_schema_and_caps(self):
        data, good = self.fixture()
        for key, value in [('schema', 'unknown'), ('archive_bytes', archive.MAX_ARCHIVE + 1),
                           ('logical_bytes', archive.MAX_LOGICAL + 1), ('members', [])]:
            bad = copy.deepcopy(good); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                archive.decode_archive(data, bad)
        bad = copy.deepcopy(good); bad['members'][0]['bytes'] = archive.MAX_MEMBER + 1
        with self.assertRaisesRegex(ValueError, 'size cap'):
            archive.decode_archive(data, bad)

    def test_changed_member_manifest(self):
        data, manifest = self.fixture()
        manifest['members'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'member hash'):
            archive.decode_archive(data, manifest)

    def test_duplicate_members(self):
        data, manifest = self.fixture([('same.json', b'{}'), ('same.json', b'{}')])
        with self.assertRaisesRegex(ValueError, 'duplicate member'):
            archive.decode_archive(data, manifest)
        manifest['members'][1]['path'] = 'other.json'
        with self.assertRaisesRegex(ValueError, 'closed member set'):
            archive.decode_archive(data, manifest)

    def test_unlisted_member(self):
        data, manifest = self.fixture([('a.json', b'{}'), ('b.json', b'{}')])
        manifest['members'].pop(); manifest['logical_bytes'] = 2
        with self.assertRaisesRegex(ValueError, 'member count'):
            archive.decode_archive(data, manifest)

    def test_unsafe_names(self):
        for name in ('../escape', '/absolute', 'a/../escape', 'a//b', 'a\\b', 'C:drive', 'a/./b', 'a\x00b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.safe_name(name)

    def test_archived_nul_name_cannot_alias_manifest_name(self):
        raw, manifest = self.fixture([('ok.jsonZX', b'{}')])
        forged = bytearray(raw)
        central = forged.index(b'PK\x01\x02')
        forged[30 + 7] = 0
        forged[central + 46 + 7] = 0
        forged = bytes(forged)
        manifest['archive_sha256'] = archive.sha(forged)
        manifest['members'][0]['path'] = 'ok.json'
        with zipfile.ZipFile(io.BytesIO(forged)) as z:
            self.assertEqual(z.infolist()[0].filename, 'ok.json')
            self.assertEqual(z.read('ok.json'), b'{}')
        with self.assertRaisesRegex(ValueError, 'invalid path'):
            archive.decode_archive(forged, manifest)

    def test_nonregular_archive_member(self):
        info = zipfile.ZipInfo('link'); info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        data, manifest = self.fixture([(info, b'target')])
        with self.assertRaisesRegex(ValueError, 'non-regular'):
            archive.decode_archive(data, manifest)

    def test_unsupported_compression(self):
        info = zipfile.ZipInfo('a.json'); info.compress_type = zipfile.ZIP_BZIP2
        data, manifest = self.fixture([(info, b'{}')])
        with self.assertRaisesRegex(ValueError, 'compression method'):
            archive.decode_archive(data, manifest)

    def test_forged_short_size_cannot_hide_deflate_expansion(self):
        raw, manifest = self.fixture([('a.json', b'{}' + b'X' * 5000)])
        forged = bytearray(raw)
        central = forged.index(b'PK\x01\x02')
        for offset in (14, central + 16):
            struct.pack_into('<I', forged, offset, binascii.crc32(b'{}') & 0xffffffff)
        for offset in (22, central + 24):
            struct.pack_into('<I', forged, offset, 2)
        forged = bytes(forged)
        manifest['archive_sha256'] = archive.sha(forged)
        manifest['logical_bytes'] = 2
        manifest['members'][0].update(archive.identity(b'{}'))
        # Demonstrates why ZipFile.read alone is insufficient for a generic loader.
        with zipfile.ZipFile(io.BytesIO(forged)) as z:
            self.assertEqual(z.read('a.json'), b'{}')
        with self.assertRaisesRegex(ValueError, 'deflate EOF/tails|raw expansion size'):
            archive.decode_archive(forged, manifest)

    def test_duplicate_or_nonfinite_json(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
            data, manifest = self.fixture([('a.json', raw)])
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                archive.decode_archive(data, manifest)

    def test_exact_carrier_anchor_is_independent_of_manifest(self):
        root = self.copied_run()
        path = root / 'chunks/terminal.zip.part-000'
        raw = bytearray(path.read_bytes()); raw[20] ^= 1; raw = bytes(raw); path.write_bytes(raw)
        manifest = json.loads((root / 'ARCHIVE.json').read_text())
        manifest['chunks'][0].update(archive.identity(raw))
        (root / 'ARCHIVE.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'fourth-run ZIP identity'):
            archive.read_archive(root)

    def test_receipt_and_provenance_mutations(self):
        for name in ('verification-receipt.json', 'scope-verification.json', 'provenance.json'):
            root = self.copied_run(); path = root / name
            if name == 'provenance.json':
                value = json.loads(path.read_text()); value['limitations'] = []; path.write_text(json.dumps(value))
            else:
                path.write_bytes(path.read_bytes() + b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.verify(root)

    def test_forged_shipping_and_terminal_claims_fail(self):
        for name, change in [('host-receipt.json', lambda r: r.update(shipping_approved=True)),
                             ('terminal-outcomes.json', lambda r: r[0].update(status='not_run'))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.verify_records(self.changed_record(name, change))

    def test_broken_native_trace_fails_even_with_refrozen_member(self):
        files = self.changed_record('whisper/clip-000001/update-001.json', lambda r: r.update(completed=True))
        with self.assertRaises(ValueError):
            archive.verify_records(files)

    def test_forged_pip_and_resource_scope_fails(self):
        changes = [('setup/bounded-acquisition/wheel-01-worker-result.json', lambda r: r.update(reviewed_source_matches=False)),
                   ('setup/bounded-acquisition/wheel-01-worker-scope.json', lambda r: r['limits'].update(network='bridge'))]
        for name, change in changes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.verify_records(self.changed_record(name, change))


if __name__ == '__main__':
    unittest.main()
