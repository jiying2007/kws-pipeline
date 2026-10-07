from pathlib import Path
from common import require,read_json,digest
ROOT=Path(__file__).resolve().parents[1]
r=read_json(ROOT/'work/PUBLICATION-READY.json');name='a20-cosy49-artifact.tar.gz';p=ROOT/'publication'/name
require(r['archive']==name and p.stat().st_size==r['bytes']and digest(p)==r['sha256']and r['bytes']<=20*1024**2,'exact bounded publication')
require({x.name for x in p.parent.iterdir()}=={name,name+'.sha256'},'exact publication pair')
require((p.parent/(name+'.sha256')).read_text()==r['sha256']+'  '+name+'\n','archive checksum sidecar')
print('Exact public artifact pair verified')
