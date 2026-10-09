import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_fetcher', ROOT / 'fetch_public_helper.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class PublicHelperFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.helper = b'# synthetic helper bytes; never executed\n'
        self.make_archive()

    def make_archive(self, duplicate=False, missing=False):
        blob = io.BytesIO()
        with zipfile.ZipFile(blob, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('objects/unrelated', b'unrelated object must not be extracted')
            if not missing:
                archive.writestr('objects/helper', self.helper)
                if duplicate:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        archive.writestr('objects/helper', self.helper)
        raw = blob.getvalue()
        self.parts = [raw[:len(raw)//2], raw[len(raw)//2:]]
        self.lock = dict(base_url='https://invalid.example/pinned',
                         archive_bytes=len(raw), archive_sha256=hashlib.sha256(raw).hexdigest(),
                         helper_object='objects/helper', helper_bytes=len(self.helper),
                         helper_sha256=hashlib.sha256(self.helper).hexdigest(),
                         parts=[dict(path=str(i), bytes=len(part), sha256=hashlib.sha256(part).hexdigest())
                                for i, part in enumerate(self.parts)])

    def opener(self, url, timeout):
        self.assertEqual(timeout, 30)
        return io.BytesIO(self.parts[int(url.rsplit('/', 1)[-1])])

    def fetch(self):
        return fetcher.fetch_helper(self.lock, self.tmp.name, opener=self.opener)

    def test_extracts_exactly_one_verified_helper(self):
        self.assertEqual(self.fetch(), self.helper)
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ['public-objects.zip'])

    def test_oversized_part_rejected(self):
        self.parts[0] += b'x'
        with self.assertRaisesRegex(ValueError, 'exceeds pinned size'): self.fetch()

    def test_truncated_part_rejected(self):
        self.parts[0] = self.parts[0][:-1]
        with self.assertRaisesRegex(ValueError, 'part size/hash mismatch'): self.fetch()

    def test_wrong_part_hash_rejected(self):
        self.lock['parts'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'part size/hash mismatch'): self.fetch()

    def test_wrong_archive_hash_rejected(self):
        self.lock['archive_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'archive size/hash mismatch'): self.fetch()

    def test_duplicate_helper_rejected(self):
        self.make_archive(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'missing, duplicated'): self.fetch()

    def test_missing_helper_rejected(self):
        self.make_archive(missing=True)
        with self.assertRaisesRegex(ValueError, 'missing, duplicated'): self.fetch()

    def test_wrong_helper_hash_rejected(self):
        self.lock['helper_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'helper size/hash mismatch'): self.fetch()

    def test_wrong_helper_zip_size_rejected(self):
        self.lock['helper_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'wrong-sized'): self.fetch()


if __name__ == '__main__':
    unittest.main()
