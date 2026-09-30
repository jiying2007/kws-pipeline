"""Explicit v2 final-feature acceptance; never relabel v1 intermediate failure."""
import argparse,hashlib,json,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=pathlib.Path);p.add_argument('--goldens',required=True,type=pathlib.Path);p.add_argument('--contract',required=True,type=pathlib.Path);p.add_argument('--output-dir',required=True,type=pathlib.Path);a=p.parse_args();root=pathlib.Path(__file__).resolve().parent;c=json.loads(a.contract.read_text());assert c['status']=='approved-for-v2-acceptance';a.output_dir.mkdir(parents=True,exist_ok=False)
for script in ['test_stream_contract.py','test_cmvn_contract.py']:subprocess.run([sys.executable,str(root/script),str(a.library)],check=True)
layerfile=a.output_dir/'v1-layers.json';run=subprocess.run([sys.executable,str(root/'compare_goldens.py'),'--library',str(a.library),'--goldens',str(a.goldens),'--output',str(layerfile)],capture_output=True,text=True);(a.output_dir/'v1-layer-log.txt').write_text(run.stdout+run.stderr)
if not layerfile.exists():raise RuntimeError('layer comparison did not produce evidence')
l=json.loads(layerfile.read_text());assert run.returncode in [0,1];assert len(l['layers'])==384 and l['cases']==39 and l['frames']==159
allowed={'power','mel_power'};hard_pass=all(x['layer'] in allowed for x in l['failures'])
assert len({(x['case'],x['layer']) for x in l['layers']})==len(l['layers'])
fftfile=a.output_dir/'fft-invariants.json';subprocess.run([sys.executable,str(root/'test_fft_invariants.py'),'--library',str(a.library),'--contract',str(a.contract),'--goldens',str(a.goldens),'--output',str(fftfile)],check=True)
f=json.loads(fftfile.read_text());assert f['library_sha256']==l['library_sha256'];assert f['passed']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'contract_name':c['name'],'contract_sha256':sha(a.contract),'passed':hard_pass and f['passed'],'implemented_scope':['C11 fbank80','independent C400affine'],'not_implemented':['C context/splice','C skip3','FSMN','decoder','target qualification'],'v1_overall_passed':l['passed'],'v1_intermediate_failures_preserved':[x for x in l['failures'] if x['layer'] in allowed],'unexpected_hard_failures':[x for x in l['failures'] if x['layer'] not in allowed],'library_sha256':l['library_sha256'],'golden_manifest_sha256':l['golden_manifest_sha256'],'v1_report_sha256':sha(layerfile),'fft_report_sha256':sha(fftfile),'driver_sha256':sha(pathlib.Path(__file__))}
(a.output_dir/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
