"""Mark a verified failed upstream probe without inventing a model receipt."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from screen import validate_manifest,require,canonical,third_policy,THIRD_POLICIES

def finalize(admission,path):
    admitted=json.loads(Path(admission).read_bytes());path=Path(path);data=json.loads(path.read_bytes())
    require(admitted.get('third_policy') in THIRD_POLICIES,'explicit admitted third policy')
    require(data.get('third_policy','disagreement_only')=='disagreement_only','expanded policy may only be selected once from admission')
    data['third_policy']=admitted['third_policy']
    failed=admitted['failed_probe_lanes'];require(set(failed)<= {'qwen06','qwen17'} and len(failed)==len(set(failed)),'failed probe lane identities')
    for row in data['slots']:
        if row['engine'] in failed:
            require(row['status']=='not_run','failed lane must not restore or schedule any model audio')
            if row['sample_id'].endswith('-00'):row.update(status='failed',failure_type='UpstreamProbeFailed',reason='Verified upstream probe pipeline failed; synthesis-attempt count is not inferred.')
    validate_manifest(data);path.write_bytes(canonical(data));return data
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();finalize(a.admission,a.manifest)
