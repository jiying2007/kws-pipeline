"""Always-run failure evidence finalizer: reads existing bytes only, never executes a phase."""
import resource,signal,os
from pathlib import Path
from common import read_json,require,write_json,digest
from artifacts import collect_phase_logs,package,journal_counts,publication_complete
ROOT=Path(__file__).resolve().parents[1]
def main():
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(20*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));signal.alarm(60)
    ledger=ROOT/'work/EXECUTION-LEDGER.json'
    if not ledger.exists():print('No admitted attempt; no finalization');return
    if publication_complete(ROOT):print('Verified existing publication retained');return
    damaged_ready=None;ready=ROOT/'work/PUBLICATION-READY.json'
    if ready.exists():
        require(ready.is_file()and not ready.is_symlink(),'safe incomplete publication receipt')
        damaged_ready=dict(bytes=ready.stat().st_size,sha256=digest(ready),status='INVALID_OR_INCOMPLETE_CONTENT_UNKNOWN')
        hold=ROOT/'work/PUBLICATION-READY.invalid';require(not hold.exists(),'one publication receipt recovery only');os.replace(ready,hold)
    l=read_json(ledger);require(l['status']=='START_RESERVED_NO_RETRY','reserved attempt required')
    out=ROOT/'work/artifact';out.mkdir(exist_ok=True);collect_phase_logs(ROOT)
    try:counts=journal_counts(out/'call-journal.jsonl')
    except (ValueError,KeyError,TypeError,OSError):counts=dict(status='UNKNOWN_DAMAGED_JOURNAL',exact_completed_count=False)
    report=dict(schema='a20-fixed300-saved-only-recovery-v1',status='FAILED_NO_RETRY',counts=counts,source_freeze_sha256=l['source_freeze_sha256'],release_sha256=l['release_sha256'],no_new_calculation=True,no_resume=True,missing_resources='UNKNOWN_UNLESS_EXISTING_RESOURCES_FILE_IS_VALID',qualification=False,damaged_publication_receipt=damaged_ready)
    if not (out/'saved-only-recovery.json').exists():write_json(out/'saved-only-recovery.json',report)
    package(ROOT,'FAILED_NO_RETRY')
if __name__=='__main__':main()
