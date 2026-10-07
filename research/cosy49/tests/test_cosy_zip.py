"""Invented in-memory ZIP safety fixtures only; no bundle code execution."""
import copy,hashlib,io,stat,sys,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fetch_inputs import cosy_zip_members

class ZipTests(unittest.TestCase):
    def fixture(self,alter=None):
        entries=[]
        for n in range(19):entries.append((f'fixture/features/{n:02}.f32le',f'native/{n:02}.f32le',b'x'+bytes([n]),None))
        if alter:entries=alter(entries)
        out=io.BytesIO();members=[]
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED)as z:
            for path,dest,raw,mode in entries:
                i=zipfile.ZipInfo(path,(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=(0o600 if mode is None else mode)<<16;z.writestr(i,raw)
                members.append(dict(path=path,destination=dest,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        raw=out.getvalue();return raw,dict(archive=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),unpacked_bytes=sum(x['bytes']for x in members),members=members))
    def test_exact_inert_members(self):
        raw,m=self.fixture();result=cosy_zip_members(raw,m);self.assertEqual(len(result),19);self.assertEqual(result['native/00.f32le'],b'x\0')
    def test_whole_archive_hash_rejected(self):
        raw,m=self.fixture();m['archive']['sha256']='f'*64
        with self.assertRaises(ValueError):cosy_zip_members(raw,m)
    def test_member_hash_size_and_count_rejected(self):
        for change in ('hash','size','count'):
            raw,m=self.fixture()
            if change=='hash':m['archive']['members'][0]['sha256']='f'*64
            elif change=='size':m['archive']['members'][0]['bytes']+=1;m['archive']['unpacked_bytes']+=1
            else:m['archive']['members'].pop()
            with self.assertRaises(ValueError):cosy_zip_members(raw,m)
    def test_traversal_rejected_even_with_matching_hashes(self):
        def alter(entries):
            path,dest,raw,mode=entries[0];entries[0]=('../outside.f32le',dest,raw,mode);return entries
        raw,m=self.fixture(alter)
        with self.assertRaisesRegex(ValueError,'safe closed paths'):cosy_zip_members(raw,m)
    def test_escaping_destination_rejected(self):
        raw,m=self.fixture();m['archive']['members'][0]['destination']='../outside.f32le'
        with self.assertRaisesRegex(ValueError,'safe closed paths'):cosy_zip_members(raw,m)
    def test_symlink_and_executable_modes_rejected(self):
        for mode in (stat.S_IFLNK|0o777,stat.S_IFREG|0o755,stat.S_IFCHR|0o600):
            def alter(entries):
                p,d,b,_=entries[0];entries[0]=(p,d,b,mode);return entries
            raw,m=self.fixture(alter)
            with self.assertRaisesRegex(ValueError,'symlink device or executable'):cosy_zip_members(raw,m)
    def test_duplicate_destination_rejected(self):
        raw,m=self.fixture();m['archive']['members'][1]['destination']=m['archive']['members'][0]['destination']
        with self.assertRaisesRegex(ValueError,'duplicate'):cosy_zip_members(raw,m)
    def test_unpacked_bomb_cap_rejected(self):
        raw,m=self.fixture();m['archive']['members'][0]['bytes']=2*1024**2;m['archive']['unpacked_bytes']=sum(x['bytes']for x in m['archive']['members'])
        with self.assertRaisesRegex(ValueError,'unpacked cap'):cosy_zip_members(raw,m)
if __name__=='__main__':unittest.main()
