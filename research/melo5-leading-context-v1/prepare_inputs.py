"""Explicitly fetch hash-pinned public saved evidence; tests never fetch inputs."""
import argparse, hashlib, http.client, json, re, ssl, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify(raw,row):
    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('size/hash mismatch: '+row['name'])
    return raw
def fetch(row):
    connection=http.client.HTTPSConnection('raw.githubusercontent.com',timeout=30,context=ssl.create_default_context())
    try:
        connection.request('GET','/'+row['repository']+'/'+row['commit']+'/'+row['path'],headers={'Accept-Encoding':'identity'})
        response=connection.getresponse()
        if response.status!=200 or response.getheader('Content-Encoding') not in (None,'identity'):raise ValueError('public GET failed: '+row['name'])
        return verify(response.read(row['bytes']+1),row)
    finally:connection.close()
def prepare(output,source_dir=None):
    manifest=json.loads((ROOT/'metadata/inputs.json').read_text());rows=manifest['files'];bodies=[]
    for row in rows:
        if (row['repository'] not in ('jiying2007/kws-pipeline','jiying2007/kws-data') or re.fullmatch('[0-9a-f]{40}',row['commit']) is None or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None or type(row['bytes']) is not int or not 0<row['bytes']<=1048576):raise ValueError('invalid immutable pin')
        for key in ('name','path'):
            if Path(row[key]).is_absolute() or '..' in Path(row[key]).parts:raise ValueError('nonlocal input path')
        destination=output/row['name']
        if destination.exists():raw=verify(destination.read_bytes(),row)
        elif source_dir is not None:raw=verify((source_dir/row['name']).read_bytes(),row)
        else:raw=fetch(row)
        bodies.append((destination,raw))
    for destination,raw in bodies:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():destination.write_bytes(raw)
    return dict(files=len(rows),bytes=sum(len(raw) for _,raw in bodies))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-dir',type=Path);a=p.parse_args()
    try:print(json.dumps(prepare(a.output_dir,a.source_dir),sort_keys=True))
    except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as e:raise SystemExit('Preparation failed: '+str(e))
