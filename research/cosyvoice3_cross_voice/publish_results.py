#!/usr/bin/env python3
"""Runner-only two-artifact publication with no model or semantic inspection."""
import json,os,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from pilot_common import load_config,require_runner,job_gate,RESERVE,OUTPUT_LIMIT
import packager

def main():
 require_runner()
 job=pathlib.Path(os.environ['RUNNER_TEMP'])/'cosy30-job'
 if not job.exists():
  print(json.dumps({'publication':'not_prepared','reason':'no_admitted_work_tree'}));return 0
 destination=ROOT/'cosy30-artifacts'
 try:
  # Reserve space for staging and uncompressed upload framing before copying.
  job_gate(job,RESERVE+2*OUTPUT_LIMIT)
  summary=packager.pack(job,load_config(),destination)
  job_gate(job,RESERVE+summary['staged_bytes']+packager.FRAMING_ALLOWANCE)
 except Exception as error:
  for name in packager.ARTIFACTS:
   marker=destination/name/'artifact-manifest.json'
   if marker.is_file():marker.unlink()
  reason=str(error) if isinstance(error,packager.GateError) else type(error).__name__
  print(json.dumps({'publication':'failed','reason':reason}));return 1
 print(json.dumps({'publication':'verified','status':summary['status'],'counts':summary['counts'],
   'staged_bytes':summary['staged_bytes'],'publication_envelope_bytes':summary['publication_envelope_bytes'],
   'initial_listening_enabled':summary['initial_listening_enabled'],'heldout_semantic_inspection':False,
   'artifact_names':list(packager.ARTIFACTS),'retention_days':1},sort_keys=True))
 return 0
if __name__=='__main__':raise SystemExit(main())
