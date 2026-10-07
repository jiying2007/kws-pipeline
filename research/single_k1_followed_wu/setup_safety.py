"""Reusable setup guards. Importing this module installs/downloads nothing."""
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import time

MIB=1024**2

def require(ok,code):
    if not ok:raise ValueError(code)

def run_bounded(command,log_directory,name,deadline_monotonic,*,env=None,
                installed_file_cap=256*MIB,combined_log_cap=4*MIB):
    """Separate installer-file and shared-log ceilings; retain stdout/stderr.

    The caller owns the aggregate expanded-wheel/workspace/download budgets.
    This helper does not authorize an install and does not retry failures.
    """
    root=Path(log_directory)
    require(root.is_dir() and not root.is_symlink(),'LOG_DIRECTORY')
    require(name and all(c.isalnum() or c in '-_' for c in name),'LOG_NAME')
    stdout=root/(name+'.stdout.log');stderr=root/(name+'.stderr.log')
    require(not stdout.exists() and not stderr.exists(),'NO_LOG_RESUME')
    require(time.monotonic()<deadline_monotonic,'WALL_DEADLINE')
    prior=sum(p.stat().st_size for p in root.glob('*.log') if p.is_file())
    require(prior<=combined_log_cap,'COMBINED_LOG_CAP')
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           start_new_session=True,env=env,
                           preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,
                                                               (installed_file_cap,installed_file_cap)))
    selector=selectors.DefaultSelector();count=0
    try:
        with stdout.open('xb') as out,stderr.open('xb') as err:
            selector.register(child.stdout,selectors.EVENT_READ,out)
            selector.register(child.stderr,selectors.EVENT_READ,err)
            while selector.get_map():
                require(time.monotonic()<deadline_monotonic,'WALL_DEADLINE')
                for key,_ in selector.select(.025):
                    block=os.read(key.fileobj.fileno(),65536)
                    if not block:
                        selector.unregister(key.fileobj);continue
                    count+=len(block)
                    require(prior+count<=combined_log_cap,'COMBINED_LOG_CAP')
                    key.data.write(block)
            child.wait(timeout=max(.001,deadline_monotonic-time.monotonic()))
        require(child.returncode==0,'COMMAND_FAILED')
    finally:
        selector.close()
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()
        child.stdout.close();child.stderr.close()
    return {'returncode':child.returncode,'new_log_bytes':count,
            'combined_log_bytes':prior+count,'stdout':stdout.name,'stderr':stderr.name}

def parse_smoke_stdout(path,expected):
    """Exactly one complete JSON value with exact keys, values and scalar types.

    Native stderr is retained separately. Never discard arbitrary stdout lines
    to recover a structured record; malformed or multiple values fail closed.
    """
    raw=Path(path).read_bytes()
    require(len(raw)<=64*1024,'SMOKE_RECORD_CAP')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'DUPLICATE_JSON_KEY');result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=pairs)
    def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
    require(canonical(value)==canonical(expected),'SMOKE_IDENTITY')
    return value
