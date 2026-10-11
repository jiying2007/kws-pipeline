"""Fixed host handshake keeps a cgroup alive for exact cumulative CPU metering."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time


def main():
    gate=Path('/gate'); started=time.monotonic()
    if len(sys.argv)<2 or not gate.is_dir():return 2
    # The host registers this still-live container before any third-party child.
    while not (gate/'go').exists():
        if time.monotonic()-started>30:return 2
        time.sleep(.02)
    try:
        child=subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        signal.signal(signal.SIGUSR1,lambda signum,frame:child.send_signal(signal.SIGUSR1))
        code=child.wait()
    except BaseException:
        code=255
    with (gate/'result.json').open('x') as f:
        json.dump({'schema':'qwen12-gated-command-v1','exit_code':code},f);f.flush();os.fsync(f.fileno())
    # Host pauses the scope, reads final usage and removes it. Never restart.
    while True:time.sleep(.1)


if __name__=='__main__':raise SystemExit(main())
