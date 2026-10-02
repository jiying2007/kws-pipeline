"""Bounded Linux process-group supervision; sampled RSS is not a cgroup limit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

GIB = 1024**3


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable not found')


def group_rss(pgid):
    total, members = 0, []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal(): continue
        try:
            fields = (entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
                total += int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        except (OSError,ValueError,IndexError): pass
    return total,members


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--max-seconds',type=int,default=2400)
    ap.add_argument('--max-rss-gib',type=int,default=12)
    ap.add_argument('--minimum-initial-available-gib',type=int,default=12)
    ap.add_argument('--progress-dir',type=Path,help='ASR output directory for hard per-clip timeout')
    ap.add_argument('--per-clip-seconds',type=int,default=120)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if args.command[:1]==['--']: args.command=args.command[1:]
    if not args.command or args.receipt.exists(): raise RuntimeError('new receipt and explicit command required')
    if available()<args.minimum_initial_available_gib*GIB: raise RuntimeError('insufficient host memory before start')
    start=time.monotonic(); peak=0; minimum=available(); reason=None
    def signal_stop(signum,frame):
        nonlocal reason
        reason='supervisor_signal_'+str(signum)
    previous={s:signal.signal(s,signal_stop) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    process=subprocess.Popen(args.command,start_new_session=True,env=dict(os.environ,FIRERED_SUPERVISOR_PID=str(os.getpid())))
    observed_starts={}
    def kill(sig):
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
    try:
        while process.poll() is None:
            rss,_=group_rss(process.pid);peak=max(peak,rss);minimum=min(minimum,available())
            if rss>args.max_rss_gib*GIB:reason='sampled_process_group_rss_limit'
            if available()<1*GIB:reason='host_memory_reserve'
            if time.monotonic()-start>args.max_seconds:reason='wall_time_limit'
            if args.progress_dir is not None and args.progress_dir.is_dir():
                for path in args.progress_dir.glob('clip-*.started.json'):
                    clip=path.name.removesuffix('.started.json')
                    observed_starts.setdefault(clip,time.monotonic())
                    if not any((args.progress_dir/(clip+suffix)).exists() for suffix in ('.raw.json','.error.json')):
                        if time.monotonic()-observed_starts[clip]>args.per_clip_seconds:
                            reason='per_clip_hard_wall_limit_'+clip
            if reason:
                kill(signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:kill(signal.SIGKILL);process.wait(timeout=10)
                break
            time.sleep(.1)
    finally:
        kill(signal.SIGTERM)
        time.sleep(.1)
        _,members=group_rss(process.pid)
        if members:kill(signal.SIGKILL);time.sleep(.1)
        _,members=group_rss(process.pid)
        data={'returncode':process.poll(),'stop_reason':reason,'wall_seconds':time.monotonic()-start,'sampled_group_peak_rss_bytes':peak,'host_minimum_available_bytes':minimum,'remaining_group_members':members,'limit_kind':'100ms sampled /proc RSS, not kernel-enforced memory isolation'}
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x') as f:json.dump(data,f,indent=2)
        for s,handler in previous.items():signal.signal(s,handler)
    if reason or process.returncode!=0 or members:raise SystemExit(1)


if __name__=='__main__':main()
