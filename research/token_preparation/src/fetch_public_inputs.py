"""Exact published-input reconstitution only; no full archive/model discovery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from urllib.parse import urlparse
import zlib
from common import check_context,read_json,require,safe,write_bytes,write_json,digest
ROOT=Path(__file__).resolve().parents[1]
MAP_SHA='d2eb3c8127373939b89e29e8e866309ac3239bc20e03c79be7b69e9dc1f02105'
COMMITS={'jiying2007/kws-data':'7af8f8597b0b7fbe8761c9a2400f2e4028395551',
         'jiying2007/kws-pipeline':'804553286fd9caa294769cc4d44cc0c43dc66e88'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('input redirect not authorized')


def validate_request(e):
    url=e['raw_url'];u=urlparse(url)
    require(u.scheme=='https'and u.netloc=='raw.githubusercontent.com'and not u.query and not u.fragment,'public input URL')
    require(e['repository']in COMMITS and e['commit']==COMMITS[e['repository']],'immutable input commit')
    require(url=='https://raw.githubusercontent.com/'+e['repository']+'/'+e['commit']+'/'+e['path'],'URL mapping')
    require(type(e['bytes'])is int and 0<e['bytes']<=5*1024*1024,'input request size')


def receive(response,e):
    require(response.status==200 and response.geturl()==e['raw_url'],'exact input response')
    length=response.headers.get('Content-Length')
    require(length is None or (length.isdigit()and int(length)==e['bytes']),'input declared bytes')
    raw=bytearray()
    while True:
        block=response.read(min(65536,e['bytes']+1-len(raw)))
        if not block:break
        raw.extend(block);require(len(raw)<=e['bytes'],'input actual byte cap')
    require(len(raw)==e['bytes']and hashlib.sha256(raw).hexdigest()==e['sha256'],'input response hash/length')
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['git_blob_sha1'],'published Git blob identity')
    return bytes(raw)


def gunzip_exact(compressed,entry):
    require(len(compressed)==entry['compressed_bytes']and hashlib.sha256(compressed).hexdigest()==entry['compressed_sha256'],'gzip transport identity')
    d=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=d.decompress(compressed,entry['bytes']+1)
    require(len(raw)==entry['bytes']and d.eof and not d.unused_data and not d.unconsumed_tail,'bounded single gzip stream')
    require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'reconstituted identity')
    return raw


def assemble(entry,requests):
    if 'file'in entry:return requests[entry['file']['raw_url']]
    require(entry['transport']=='ordered-gzip-shards','exact transport')
    descriptor=json.loads(requests[entry['object_descriptor']['raw_url']])
    require(descriptor['sha256']==entry['sha256']and descriptor['size_bytes']==entry['bytes']and descriptor['compression']=='gzip','object descriptor')
    require(descriptor['compressed_sha256']==entry['compressed_sha256']and descriptor['compressed_size_bytes']==entry['compressed_bytes'],'compressed descriptor')
    expected=[dict(path='archive/shards/'+s['path'].rsplit('/',1)[-1],sha256=s['sha256'],size_bytes=s['bytes'])for s in entry['shards']]
    require(descriptor['shards']==expected,'exact ordered shards')
    return gunzip_exact(b''.join(requests[s['raw_url']]for s in entry['shards']),entry)


def run(context):
    check_context(ROOT,context,'inputs')
    path=ROOT/'metadata/PUBLIC-INPUT-MAP.json';require(digest(path)==MAP_SHA,'public map')
    m=read_json(path);requests=m['request_allowlist']
    require(len(requests)==155 and len({e['raw_url']for e in requests})==155 and sum(e['bytes']for e in requests)==6097691,'closed request budget')
    require(shutil.disk_usage(ROOT).free>=4*1024**3,'input free disk floor')
    cache=ROOT/'work/input-transports';cache.mkdir(mode=0o700,exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect());data={};receipts=[];deadline=time.monotonic()+300
    for i,e in enumerate(requests):
        validate_request(e);require(time.monotonic()<deadline,'input wall deadline')
        with opener.open(urllib.request.Request(e['raw_url'],headers={'User-Agent':'a20-public-preparation/1','Accept-Encoding':'identity'}),timeout=30)as response:
            raw=receive(response,e)
        write_bytes(cache/f'{i:03d}.bin',raw,5*1024*1024);data[e['raw_url']]=raw
        receipts.append(dict(url=e['raw_url'],bytes=len(raw),sha256=e['sha256'],git_blob_sha1=e['git_blob_sha1']))
    entries=m['audio']+[m['checkpoint']]+m['python_sources']+m['native_frontend']+m['reference_metadata']+m['rights_assets']
    for e in entries:
        raw=assemble(e,data);p=safe(ROOT,e['destination']);write_bytes(p,raw,5*1024*1024)
    for i,e in enumerate(m['notices']):write_bytes(ROOT/f'work/inputs/metadata/notices/{i:02d}.txt',data[e['raw_url']],128*1024)
    outputs=read_json(ROOT/'metadata/PUBLIC-INPUTS.json')['outputs']
    for e in outputs:
        p=safe(ROOT/'work/inputs',e['path']);require(p.stat().st_size==e['bytes']and digest(p)==e['sha256'],'final input identities')
    write_json(ROOT/'work/artifact/input-reconstitution.json',dict(schema='a20-public-input-reconstitution-v1',requests=receipts,outputs=outputs,
        request_count=155,download_bytes=6097691,full_archive_downloaded=False,model_payload_count=1,waveform_count=32,model_calls=0))
    print('Public inputs reconstituted and verified;32 WAVs and exact original A20 only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-context',type=Path);a=p.parse_args()
    if a.execute_context is None:raise SystemExit('PREPARATION_ONLY: no execution context')
    from safe_failure import save
    try:run(read_json(a.execute_context))
    except BaseException as error:
        save(ROOT,error,'inputs','PUBLIC_INPUT_RECONSTITUTION')
        raise SystemExit(1)
