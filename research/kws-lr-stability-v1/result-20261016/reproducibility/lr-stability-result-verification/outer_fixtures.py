"""Synthetic outer process evidence, never invokes subprocess or runner."""
import json
from pathlib import Path
import verify_result as a
v=a.v

def outer(base,repo,freeze):
    evidence=base/'lr-stability-execution-evidence';evidence.mkdir()
    command=[str(base/'fit-venv/bin/python'),'-I','-B',str(base/'lr-stability-integrated/research/kws-lr-stability-v1/run_lr_stability.py'),'--repo',str(base/'lr-stability-integrated'),'--artifacts',str(base.parent/'recovery-kws-20261015/artifacts'),'--wheelhouse',str(base/'runtime-work/wheels'),'--output',str(base/'lr-stability-result'),'--claim-root',str(base/'lr-stability-campaign-ledger'),'--plan-sha256',freeze['plan_sha256'],'--historical-result',str(base/'grouped-balance-result'),'--historical-manifest',str(base/'grouped-balance-result-acceptance/hash-manifest.json')]
    cmdsha=v.sha(json.dumps(command,separators=(',',':')).encode())
    intent=dict(schema='research-lr-stability-local-launch-intent-v1',campaign_id=a.CAMPAIGN,parent_verified_source_commit=freeze['reviewed_source_commit'],plan_sha256=freeze['plan_sha256'],runner_sha256=freeze['runner_sha256'],launcher_sha256=freeze['launcher_sha256'],command_sha256=cmdsha,max_optimizer_updates=600,max_model_forward_calls=608,new_tts_calls=0,new_asr_calls=0,max_probe_forwards=0,retry=False,actual_optimizer_updates='UNKNOWN_UNTIL_RESULT_VERIFICATION',wall_timeout_seconds=3600,cpu_limit_seconds=3600,virtual_address_space_limit_bytes=8*1024**3,per_file_limit_bytes=16*1024**2,runner_evidence_directory_limit_bytes=512*1024**2,production_qualified=False,reusable_weights_exported=False,reviewed_source_pins=freeze['reviewed_source_pins'],independent_new_budget=True,previous_campaign_budget_reused=False,claim_relative='research-budget/kws-lr-stability-20261016-v1.claim.json',core_dump_limit_bytes=0,cpu_affinity=[0],started_at_utc='2026-10-15T00:00:00+00:00')
    process=dict(schema='research-lr-stability-local-process-result-v1',exit_code=0,wall_timeout=False,wall_seconds=1.0,user_cpu_seconds=1.0,system_cpu_seconds=0.0,child_max_rss_kib=1000,max_rss_scope='Linux RUSAGE_CHILDREN, not cgroup or SSC305',stdout_sha256=v.sha(b''),stderr_sha256=v.sha(b''),launcher_sha256=freeze['launcher_sha256'],command_sha256=cmdsha,actual_optimizer_updates='VERIFY_CALL_AND_STEP_LEDGER',production_qualified=False)
    for name,obj in [('launch-intent.json',intent),('process-start.json',dict(pid=1,command_sha256=cmdsha)),('process-result.json',process)]: (evidence/name).write_bytes(v.encoded(obj))
    for name in ('stdout.log','stderr.log'):(evidence/name).write_bytes(b'')
    return evidence/'process-result.json'
