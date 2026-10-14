"""Read-only public small-package verification. Never builds or calls a model.
This checks archived receipts, not missing raw trace bytes. The independent
full-local audit supplies raw-trace verification; missing bodies stay explicit.
"""
import hashlib,json,pathlib,sys
D=pathlib.Path(sys.argv[1]) if len(sys.argv)==2 else pathlib.Path(__file__).resolve().parent

def require(value,message):
    if not value: raise ValueError(message)
def read(name):return json.loads((D/name).read_text())
m=read('publication-manifest.json')
for row in m['files']:
    p=pathlib.PurePosixPath(row['path'])
    require(not p.is_absolute() and '..' not in p.parts,'unsafe path')
    b=(D/p).read_bytes()
    require(len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],'member hash')
r=read('run/receipt.json');a=read('run/analysis.json');s=read('diagnosis-summary.json')
require(r['status']=='pass' and r['completed_samples']==13,'completion')
require(len(r['calls'])==39 and len({(x['sample_id'],x['kind']) for x in r['calls']})==39,'calls')
require(all(x['attempt']==1 and x['exit_code']==0 for x in r['calls']),'attempts')
require({k:sum(x['kind']==k for x in r['calls']) for k in ('dump','replay','path')}=={'dump':13,'replay':13,'path':13},'budget')
require(len(a)==13 and len({x['sample_id'] for x in a})==13,'sample count')
require(sum(x['frames'] for x in a)==1375 and sum(x['speech_frames'] for x in a)==879,'frame counts')
require(sum(x['top1_histogram'].get('0',0) for x in a)==1317 and sum(x['top1_histogram'].get('3',0) for x in a)==0,'tokens')
require(all(x['events_byte_identical'] and all(k['detections']==0 for k in x['paths']['keywords']) for x in a),'event agreement')
require(s['all_blank_clips']==6 and s['frames']==1375,'summary')
print('PASS: small-package hashes/receipts; no model execution; raw trace bodies excluded')
