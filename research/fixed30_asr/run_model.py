"""Fresh supervised model process; no downloader and no repeated clip attempts."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from fixed30_contract import canonical,digest,decode,validate_decoder
from asr_stage.execution import execute_primary

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=('sensevoice','qwen06'),required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--supervisor-pid',type=int,required=True)
    a=p.parse_args();runtime=ROOT/'runtime'
    if os.environ.get('FIXED30_SUPERVISED')!='1' or os.getpgrp()!=os.getpid() or os.getppid()!=a.supervisor_pid: raise ValueError('Fresh supervised model process required')
    plan=decode((runtime/'prepared'/(a.name+'.plan.json')).read_bytes(),a.plan_sha256)
    # Verify installed bodies/metadata and source pins again before ASR imports.
    from setup_locked import verify_setup
    verify_setup(runtime)
    declaration=plan['contract']
    if declaration['candidate_count']!=30: raise ValueError('Wrong call budget')
    with (runtime/(a.name+'.claim')).open('x') as f: f.write(a.plan_sha256+'\n')
    result=execute_primary(plan,runtime/'raw'/a.name,(runtime/'prepared/decoder-inputs.json').read_bytes())
    return 0 if result['status']=='complete' else 1

if __name__=='__main__': raise SystemExit(main())
