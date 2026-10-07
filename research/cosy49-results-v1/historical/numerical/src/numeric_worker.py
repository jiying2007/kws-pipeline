"""One gated candidate reference OR native phase; never starts the next phase."""
import argparse,os,sys,resource,time
from pathlib import Path
import common

def run(release_path,phase):
 r=common.configure(release_path,phase)
 ledger=common.read(common.ROOT/phase/'work/EXECUTION-LEDGER.json')
 common.require(ledger['status']=='START_RESERVED_NO_RETRY'and ledger['release_sha256']==common.sha(release_path),'supervised exclusive phase reservation')
 common.write(common.OUT/'worker-claim.json',dict(status='CLAIMED_NO_RETRY',phase=phase,release_sha256=common.sha(release_path)))
 import numpy as np
 common.require(np.__version__=='2.3.5'and 'torch'not in sys.modules,'exact numpy no Torch')
 cases,inputs=common.load_inputs(np);sys.path.insert(0,str(common.CANONICAL))
 from algorithms import reference_run,native_run
 begin=time.monotonic()
 result=reference_run(np,r,cases,inputs,common.OUT)if phase=='reference'else native_run(np,r,cases,inputs,common.OUT)
 common.identities(r);common.allocation_check()
 result.update(schema='a20-candidate-numerical-report-v1',phase=phase,terminal_state_sha256=r['terminal_state_sha256'],native_bundle_sha256=r['native_bundle']['sha256'],protocol_sha256=common.CONTRACT_SHA,source_freeze_sha256=r['source_freeze_sha256'],release_sha256=common.sha(release_path),elapsed_seconds=time.monotonic()-begin,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,old_A20_verdict_inherited=False,actual_audio_calls=0,automatic_next_stage=False)
 common.write_compact(common.OUT/'report.json',result);common.allocation_check()
 return 0 if result['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path);p.add_argument('--phase',choices=['reference','native']);a=p.parse_args()
 if not a.release or not a.phase:raise SystemExit('PREPARATION_ONLY: explicit supervised numerical phase required')
 try:raise SystemExit(run(a.release,a.phase))
 except Exception as e:
  if common.OUT and common.OUT.exists():common.write(common.OUT/'safe-failure.json',dict(status='FAILED_NO_RETRY',exception_type=type(e).__name__,raw_message_captured=False,counts='REFER_TO_COMPLETE_SAVED_FILES_OR_UNKNOWN'))
  raise SystemExit(1)
