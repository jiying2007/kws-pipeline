"""Read-only receipt checker. Never imports torch or invokes inference."""
import hashlib,json,pathlib,sys,math
D=pathlib.Path(sys.argv[1]) if len(sys.argv)==2 else pathlib.Path(__file__).resolve().parent
def check(v,m):
 if not v:raise ValueError(m)
def load(n):return json.loads((D/n).read_text())
for row in load('publication-manifest.json')['files']:
 p=pathlib.PurePosixPath(row['path']);check(not p.is_absolute() and '..' not in p.parts,'path');b=(D/p).read_bytes();check(len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],'hash')
r=load('run/receipt.json');a=load('run/analysis.json');s=load('summary.json')
check(r['status']=='pass' and r['float_forward_calls']==13 and r['no_new_c_dump'],'scope')
check(len(r['calls'])==13 and len({x['sample_id'] for x in r['calls']})==13 and all(x['attempt']==1 and x['status']=='success' for x in r['calls']),'calls')
check(len(a)==13 and sum(x['frames'] for x in a)==1375,'frames')
check(sum(x['top1_agreement_frames'] for x in a)==1372,'top1')
check(all(x['top1_counts']['float'][3]==0 and x['top1_counts']['c'][3]==0 for x in a),'xiao')
check(abs(s['global_logit_rmse']-math.sqrt(sum(x['frames']*x['logit_rmse']**2 for x in a)/1375))<1e-12,'rmse')
print('PASS: archived float/C receipts; raw checkpoint/audio/logit bodies excluded; no model execution')
