"""Closed read-only public transports: prepared bytes and the original A20 only."""
import argparse,hashlib,io,tarfile,time,urllib.request,zlib
from pathlib import Path
from urllib.parse import urlparse
from common import check_context,read_json,require,safe,write_bytes,write_json,digest
ROOT=Path(__file__).resolve().parents[1]
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('input redirect not authorized')
def validate_request(e,commits):
    u=urlparse(e['raw_url'])
    require(u.scheme=='https'and u.netloc=='raw.githubusercontent.com'and not u.query and not u.fragment,'public input URL')
    require(e['commit']in commits.get(e['repository'],set()),'immutable input commit')
    require(e['raw_url']=='https://raw.githubusercontent.com/'+e['repository']+'/'+e['commit']+'/'+e['path'],'URL mapping')
    require(type(e['bytes'])is int and 0<e['bytes']<=1024**2,'input request size')
def receive(response,e):
    require(response.status==200 and response.geturl()==e['raw_url'],'exact input response')
    n=response.headers.get('Content-Length');require(n is None or (n.isdigit()and int(n)==e['bytes']),'input declared bytes')
    raw=bytearray()
    while True:
        b=response.read(min(65536,e['bytes']+1-len(raw)))
        if not b:break
        raw.extend(b);require(len(raw)<=e['bytes'],'input actual byte cap')
    require(len(raw)==e['bytes']and hashlib.sha256(raw).hexdigest()==e['sha256'],'input hash and length')
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['git_blob_sha1'],'Git blob identity')
    return bytes(raw)
def gunzip_exact(compressed,e):
    require(len(compressed)==e['compressed_bytes']and hashlib.sha256(compressed).hexdigest()==e['compressed_sha256'],'compressed identity')
    d=zlib.decompressobj(16+zlib.MAX_WBITS);raw=d.decompress(compressed,e['bytes']+1)
    require(len(raw)==e['bytes']and d.eof and not d.unused_data and not d.unconsumed_tail,'single bounded gzip stream')
    require(hashlib.sha256(raw).hexdigest()==e['sha256'],'uncompressed identity');return raw
def assemble(e,data):
    if 'file'in e:return data[e['file']['raw_url']]
    require(e['transport']=='ordered-gzip-shards','transport schema')
    descriptor=__import__('json').loads(data[e['object_descriptor']['raw_url']])
    require(descriptor['sha256']==e['sha256']and descriptor['size_bytes']==e['bytes']and descriptor['compression']=='gzip','object descriptor')
    require(descriptor['compressed_sha256']==e['compressed_sha256']and descriptor['compressed_size_bytes']==e['compressed_bytes'],'compressed descriptor')
    expected=[dict(path='archive/shards/'+s['path'].rsplit('/',1)[-1],sha256=s['sha256'],size_bytes=s['bytes'])for s in e['shards']]
    require(descriptor['shards']==expected,'ordered shard descriptor')
    return gunzip_exact(b''.join(data[x['raw_url']]for x in e['shards']),e)
def prepared_members(raw,manifest):
    a=manifest['archive'];require(len(raw)==a['bytes']and hashlib.sha256(raw).hexdigest()==a['sha256'],'prepared archive identity')
    d=zlib.decompressobj(16+zlib.MAX_WBITS);expanded=d.decompress(raw,8*1024**2+1)
    require(len(expanded)<=8*1024**2 and d.eof and not d.unconsumed_tail and not d.unused_data,'bounded prepared gzip')
    results={}
    with tarfile.open(fileobj=io.BytesIO(expanded),mode='r:')as t:
        members=t.getmembers();require(len(members)==len(a['members']),'prepared member count')
        for m,e in zip(members,a['members']):
            require(m.isfile()and not m.issym()and not m.islnk()and m.name==e['path']and m.size==e['bytes']and m.name not in results,'prepared member schema')
            b=t.extractfile(m).read(e['bytes']+1)
            require(len(b)==e['bytes']and hashlib.sha256(b).hexdigest()==e['sha256'],'prepared member identity');results[m.name]=b
    return results
def run(context):
    check_context(ROOT,context,'inputs')
    model=read_json(ROOT/'metadata/MODEL-INPUTS.json');prep=read_json(ROOT/'metadata/PREPARED-INPUTS.json')
    require(prep['status']=='VERIFIED_PUBLIC_PIN'and type(prep['commit'])is str and len(prep['commit'])==40,'prepared public pin required')
    require(len(prep['requests'])==3 and sum(e['bytes']for e in prep['requests'])==prep['archive']['bytes'],'exact three prepared shards')
    commits={'jiying2007/kws-data':{'7af8f8597b0b7fbe8761c9a2400f2e4028395551',prep['commit']},'jiying2007/kws-pipeline':{'804553286fd9caa294769cc4d44cc0c43dc66e88'}}
    requests=model['requests']+prep['requests'];require(len({e['raw_url']for e in requests})==len(requests),'unique requests')
    require(sum(e['bytes']for e in requests)<=8*1024**2,'closed input download budget')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect());data={};receipts=[];deadline=time.monotonic()+300
    for e in requests:
        validate_request(e,commits);require(time.monotonic()<deadline,'input deadline')
        with opener.open(urllib.request.Request(e['raw_url'],headers={'User-Agent':'a20-fixed300/1','Accept-Encoding':'identity'}),timeout=30)as response:b=receive(response,e)
        data[e['raw_url']]=b;receipts.append(dict(path=e['path'],commit=e['commit'],bytes=len(b),sha256=e['sha256'],git_blob_sha1=e['git_blob_sha1']))
    for e in model['objects']:write_bytes(safe(ROOT,e['destination']),assemble(e,data),5*1024**2)
    for i,e in enumerate(model['notices']):write_bytes(ROOT/f'work/inputs/notices/{i:02d}.txt',data[e['raw_url']],128*1024)
    raw=b''.join(data[e['raw_url']]for e in prep['requests'])
    for name,b in prepared_members(raw,prep).items():write_bytes(safe(ROOT/'work/prepared',name),b,1024**2)
    write_json(ROOT/'work/artifact/input-reconstitution.json',dict(schema='a20-fixed300-inputs-v1',requests=receipts,request_count=len(requests),download_bytes=sum(e['bytes']for e in requests),model_payload_count=1,audio_downloads=0,feature_extractions=0,control_forwards=0,prepared_archive_sha256=prep['archive']['sha256']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-context',type=Path);a=p.parse_args()
    if a.execute_context is None:raise SystemExit('PREPARATION_ONLY: no released input context')
    from safe_failure import save
    try:run(read_json(a.execute_context))
    except BaseException as e:save(ROOT,e,'inputs','INPUT_RECONSTITUTION');raise SystemExit(1)
