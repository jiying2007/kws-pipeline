import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_archive as v

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'chunks').mkdir()
        self.make()
    def tearDown(self):
        self.tmp.cleanup()
    def make(self, zip_name='safe/item.txt'):
        b = b'invented evidence\n'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(zip_name, b)
        raw = stream.getvalue()
        (self.root / 'chunks/part').write_bytes(raw)
        self.manifest = {'schema':'cosy49-public-archive-v1','archive_bytes':len(raw),
                        'archive_sha256':v.digest(raw),'logical_bytes':len(b),
                        'chunks':[{'path':'chunks/part','bytes':len(raw),'sha256':v.digest(raw)}],
                        'members':[{'path':'safe/item.txt','bytes':len(b),'sha256':v.digest(b)}]}
        self.save()
    def save(self):
        (self.root/'ARCHIVE.json').write_text(json.dumps(self.manifest))
    def test_restore(self):
        out=self.root/'out'
        self.assertEqual(v.verify(self.root,out)['files'],1)
        self.assertEqual((out/'safe/item.txt').read_bytes(),b'invented evidence\n')
    def test_corrupt_chunk(self):
        p=self.root/'chunks/part';b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_member_hash(self):
        self.manifest['members'][0]['sha256']='0'*64;self.save()
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_missing_chunk(self):
        (self.root/'chunks/part').unlink()
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_extra_chunk(self):
        (self.root/'chunks/extra').write_bytes(b'x')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_unsafe_member(self):
        self.make('../escape')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_chunk_total_before_reads(self):
        self.manifest['archive_bytes']+=1;self.save()
        with self.assertRaisesRegex(ValueError,'chunk total'):v.verify(self.root)
    def test_windows_drive_path(self):
        self.make('C:/escape')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_duplicate_manifest(self):
        p=self.root/'ARCHIVE.json';p.write_text(p.read_text()[:-1]+',"schema":"cosy49-public-archive-v1"}')
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_symlink_chunk(self):
        p=self.root/'chunks/part';b=p.read_bytes();p.unlink();q=self.root/'real';q.write_bytes(b);p.symlink_to(q)
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_extra_dangling_symlink(self):
        (self.root/'chunks/extra').symlink_to(self.root/'absent')
        with self.assertRaisesRegex(ValueError,'symlink in chunk tree'):v.verify(self.root)
    def test_existing_output(self):
        out=self.root/'out';out.mkdir()
        with self.assertRaises(ValueError):v.verify(self.root,out)
    def test_duplicate_member(self):
        self.manifest['members'].append(self.manifest['members'][0]);self.save()
        with self.assertRaises(ValueError):v.verify(self.root)
    def test_size_cap(self):
        self.manifest['logical_bytes']=v.MAX_ARCHIVE+1;self.save()
        with self.assertRaises(ValueError):v.verify(self.root)

if __name__=='__main__':unittest.main()
