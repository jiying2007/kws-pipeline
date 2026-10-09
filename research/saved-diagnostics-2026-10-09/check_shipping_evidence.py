#!/usr/bin/env python3
"""Read-only shipping source attribution; no audio/model forward."""
import argparse, pathlib, json, hashlib
EXPECTED_SOURCES={'far-aggregate.json': 'a9514e28f24904e6256fe994b806b0f09e00c0ef889605232680a3d08c230d53', 'far-run-1103/detection-source-map.jsonl': 'e25d1c2b87737ab9cc4c028351e28ba5637616d31d50cfed054b5e2425df6f61', 'far-run-3301/detection-source-map.jsonl': '72c5ad42d75431b5f64729471adebf2bf9878790fcc4587413ff74770b7b055c'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import sys
 if sys.flags.optimize:raise RuntimeError("optimized Python disables evidence assertions; refused")
 p=argparse.ArgumentParser();p.add_argument('artifact',type=pathlib.Path);p.add_argument('--zip',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();root=a.artifact
 if a.output.exists():raise FileExistsError('output already exists; evidence will not be overwritten')
 for name,expected in EXPECTED_SOURCES.items():
  if sha(root/name)!=expected:raise ValueError('source identity mismatch: '+name)
 aggregate=json.loads((root/'far-aggregate.json').read_text());report={'artifact_id':11585689303,'run_id':37860130019,'artifact_zip_sha256':'77e3d3fda81bef000c72927bf196dbed1b680e431c4455ee122ecdbdef1a38d0','source_members_verified':EXPECTED_SOURCES,'zip_hash_verified':False,'model_sha256':aggregate['model_sha256'],'events':[],'model_scope':'shipping model, not D20/D90','root_cause':'UNDETERMINED: source attribution is not a saved decoder/logit/VAD state trace'}
 if a.zip:
  if sha(a.zip)!=report['artifact_zip_sha256']:raise ValueError('ZIP identity mismatch')
  report['zip_hash_verified']=True
 else:report['artifact_zip_sha256_is_declared_source_reference_only']=True
 for seed,t,conf,tokens in [(1103,7191.345,.603046,['xiao3','xiao3','xiao3','wo1']),(3301,2372.645,.962665,['xiao3','wo1','xiao3','ni3'])]:
  f=root/f'far-run-{seed}/detection-source-map.jsonl';events=[json.loads(line) for line in f.read_text().splitlines()];assert len(events)==1;e=events[0];d=e['detection'];active=e['active_injection'];prior=e['previous_injection'];assert d['time_s']==t and d['confidence']==conf and d['keyword_id']==2 and active['domain']['tokens']==tokens
  report['events'].append({'seed':seed,'time_s':t,'confidence':conf,'keyword':2,'source_map_sha256':sha(f),'active_tokens':tokens,'active_offset_s':t-active['start_second'],'prior_tokens':prior['domain']['tokens'],'prior_end_to_detection_s':t-prior['end_second'],'gap_between_prior_end_and_active_start_s':active['start_second']-prior['end_second'],'source_sha256':active['source_sha256'],'scene':active['domain']['scene']})
 a.output.open('x').write(json.dumps(report,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
