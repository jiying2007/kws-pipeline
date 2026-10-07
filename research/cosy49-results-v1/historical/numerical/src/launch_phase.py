"""One separately authorized phase. No chained reference/native/acoustic runs."""
import argparse,math,os,resource,signal,sys,time
from pathlib import Path
import common,guard_reuse as guard

def launch(path,phase):
 r=common.configure(path,phase);phase_root=common.ROOT/phase;phase_root.mkdir(exist_ok=True);work=phase_root/'work';work.mkdir(exist_ok=False);out=work/'artifact';out.mkdir()
 common.write(work/'EXECUTION-LEDGER.json',dict(status='START_RESERVED_NO_RETRY',phase=phase,release_sha256=common.sha(path),source_freeze_sha256=r['source_freeze_sha256'],retry_allowed=False))
 guard.ROOT=phase_root;guard.OUTPUT_CAP=common.read(common.ROOT/'PROTOCOL.json')['phase_budgets'][phase]['output_bytes']
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 hard=math.ceil(time.process_time()+600)+1;resource.setrlimit(resource.RLIMIT_CPU,(hard,hard))
 def deadline(*args):raise RuntimeError('NUMERICAL_GLOBAL_WALL_LIMIT')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(900);start=time.monotonic();cpu=guard.accounted_cpu()
 record=guard.phase(phase,[sys.executable,'-B',str(common.ROOT/'src/numeric_worker.py'),'--release',str(Path(path).resolve()),'--phase',phase],start=start,cpu_start=cpu,phase_wall=900,file_cap=20*1024**2)
 common.identities(r);metrics=dict(wall_seconds=time.monotonic()-start,cpu_seconds=guard.accounted_cpu()-cpu);signal.alarm(0)
 passed=record['status']=='PASS'and metrics['wall_seconds']<=900 and metrics['cpu_seconds']<=600
 common.write(work/'supervision.json',dict(schema='a20-candidate-numerical-supervision-v1',passed=passed,phase=phase,record=record,metrics=metrics,output_cap_bytes=guard.OUTPUT_CAP,raw_logs='PRIVATE_RUN_LOCAL_ONLY',automatic_next_stage=False,independent_saved_output_audit_required=True))
 common.write(work/'EXECUTION-TERMINAL.json',dict(status='SUCCESS_PENDING_SAVED_AUDIT'if passed else'FAILED_NO_RETRY',automatic_next_stage=False))
 return 0 if passed else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path);p.add_argument('--phase',choices=['reference','native']);a=p.parse_args()
 if not a.release or not a.phase:raise SystemExit('PREPARATION_ONLY: exact separately reviewed numerical phase release required')
 raise SystemExit(launch(a.release,a.phase))
