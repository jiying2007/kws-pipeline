"""Read-only last gate immediately before the bounded public upload."""
from pathlib import Path
from common import read_json,require,digest
ROOT=Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    ready=read_json(root/'work/PUBLICATION-READY.json')
    out=root/'publication'
    require({p.name for p in out.iterdir()}=={'token-preparation-artifact.tar.gz','token-preparation-artifact.tar.gz.sha256'},'closed publication pair')
    archive=out/ready['archive'];receipt=out/ready['sha256_receipt']
    require(not archive.is_symlink()and not receipt.is_symlink(),'regular publication files')
    require(archive.stat().st_size==ready['gzip_bytes']and digest(archive)==ready['sha256'],'publication readback hash')
    require(receipt.read_text()==ready['sha256']+'  token-preparation-artifact.tar.gz\n','publication hash receipt')
    require(archive.stat().st_size+receipt.stat().st_size==ready['publication_total_bytes']<=8*1024**2,'publication byte cap')
    require(ready['retention_days']==1 and ready['upload_performed']is False,'bounded upload contract')
    return True

if __name__=='__main__':
    verify();print('Public artifact pair verified; maximum8MiB and retention1day')
