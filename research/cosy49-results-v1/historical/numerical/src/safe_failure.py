"""Bounded diagnostic identities only; no raw exception text or paths."""
SAFE_CODES=frozenset(['MONITOR_EVIDENCE_LIMIT','TRACKED_PROCESS_VISIBILITY_REQUIRED','CHILD_VISIBILITY_REQUIRED','MANDATORY_PROC_FIELDS_UNAVAILABLE','OWNED_ROOT_UNAVAILABLE','OWNED_ROOT_GROUP_ESCAPE','OWNED_CHILD_GROUP_ESCAPE','OWNED_IDENTITY_CHANGED_DURING_SAMPLE','OWNED_PID_REUSE','PROC_INVENTORY_LIMIT','PROC_INVENTORY_SCHEMA','PROC_INVENTORY_LIST_UNAVAILABLE','PROC_INVENTORY_STAT_UNAVAILABLE','PROC_DISAPPEARANCE_AMBIGUOUS','PROC_STAT_SCHEMA','PROC_STAT_PID','PROC_STAT_FIELDS','PROC_STAT_NEGATIVE','PROC_STAT_GROUP','CLOCK_TICK_RATE','NO_OWNED_LIVE_SAMPLE','PROCESS_LIMIT','RSS_LIMIT','THREAD_LIMIT','CPU_LIMIT','WALL_LIMIT','LOG_OUTPUT_LIMIT','ARTIFACT_WORK_OUTPUT_LIMIT','AGGREGATE_CPU_LIMIT','PROCESS_REAP','CHILD_FAILURE_OR_NO_LIVE_GUARD_SAMPLE'])
def describe(root,error,phase,stage):
 message=str(error)
 return dict(schema='a20-candidate-safe-failure-v1',phase=phase,stage=stage,exception_type=type(error).__name__,error_code=message if message in SAFE_CODES else 'UNCLASSIFIED_GUARD_ERROR',raw_message_captured=False)
def cleanup_status(value):
 out={k:value.get(k)for k in ['returncode','kill_requested','kill_exit_race','reaped']}
 out['errors']=[dict(exception_type=e.get('type','UNKNOWN'),errno=e.get('errno'),raw_message_captured=False)for e in value.get('errors',[])]
 if 'kill_exit_race_evidence'in value:
  e=value['kill_exit_race_evidence'];out['kill_exit_race_evidence']=dict(exception_type=e.get('type','UNKNOWN'),errno=e.get('errno'),raw_message_captured=False)
 out.update(group_absence_proven=False,oom_inference=False)
 return out
