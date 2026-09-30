"""Losslessly assemble small pinned fixture parts in memory, then safely unpack."""
import argparse,hashlib,io,json,pathlib,tarfile
ARCHIVE_BYTES=555581
ARCHIVE_SHA256='ef729d8a4e0170bddeeec67d2095f38f53e67b363fc9d5a56ce8af2fea8e6f54'
PART_BYTES=49152

def require(condition,message):
 if not condition:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def assemble_archive(fixture):
 identity=json.loads((fixture/'identity.json').read_text())
 require(identity['bytes']==ARCHIVE_BYTES and identity['sha256']==ARCHIVE_SHA256,'original archive identity changed')
 require(identity['storage']=='lossless-parts-v1' and identity['parts_manifest']=='parts-manifest.json','unsupported part storage')
 m=fixture/'parts-manifest.json';raw=m.read_bytes();require(len(raw)<16384 and digest(raw)==identity['parts_manifest_sha256'],'part manifest digest mismatch');manifest=json.loads(raw)
 require(set(manifest)=={'schema_version','chunk_limit_bytes','archive_bytes','archive_sha256','parts'},'part manifest keys')
 require(manifest['schema_version']==1 and manifest['chunk_limit_bytes']==PART_BYTES and manifest['archive_bytes']==ARCHIVE_BYTES and manifest['archive_sha256']==ARCHIVE_SHA256,'part manifest identity mismatch')
 parts=manifest['parts'];count=(ARCHIVE_BYTES+PART_BYTES-1)//PART_BYTES;require(len(parts)==count,'part count mismatch')
 directory=fixture/'parts';require(directory.is_dir() and not directory.is_symlink(),'invalid part directory');require({x.name for x in directory.iterdir()}=={f'{i:03d}.bin' for i in range(count)},'part member coverage mismatch')
 assembled=bytearray()
 for i,part in enumerate(parts):
  expected=PART_BYTES if i<count-1 else ARCHIVE_BYTES-PART_BYTES*(count-1)
  require(set(part)=={'index','path','bytes','sha256'} and type(part['index']) is int and part['index']==i and part['path']==f'parts/{i:03d}.bin' and type(part['bytes']) is int and part['bytes']==expected,'part order/path/size mismatch')
  p=fixture/part['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==expected,'part file/size mismatch');b=p.read_bytes();require(len(b)==expected and digest(b)==part['sha256'],'part digest mismatch');assembled.extend(b)
 result=bytes(assembled);require(len(result)==ARCHIVE_BYTES and digest(result)==ARCHIVE_SHA256,'complete archive digest mismatch');return result

def unpack_archive(raw,output):
 require(len(raw)==ARCHIVE_BYTES and digest(raw)==ARCHIVE_SHA256,'unpack requires original pinned bytes')
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
  members=t.getmembers();require(len(members)<2000 and sum(m.size for m in members)<32*1024*1024,'archive member/size bound');seen=set()
  for m in members:
   p=pathlib.PurePosixPath(m.name);require(m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]=='donor-frontend-goldens' and m.name not in seen and 0<=m.size<1024*1024,'unsafe archive member');seen.add(m.name)
  output.mkdir(parents=True,exist_ok=False)
  for m in members:
   dest=output/pathlib.PurePosixPath(m.name);dest.parent.mkdir(parents=True,exist_ok=True);f=t.extractfile(m);require(f is not None,'missing archive stream');b=f.read();require(len(b)==m.size,'truncated archive member')
   with dest.open('xb') as out:out.write(b)
 return output/'donor-frontend-goldens'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();fixture=pathlib.Path(__file__).resolve().parents[1]/'fixtures';print(unpack_archive(assemble_archive(fixture),a.output))
if __name__=='__main__':main()
