"""Fixed numeric same-job storage observations, never cleanup or subprocesses."""
from pathlib import Path
import json
import os
import re
import shutil
import time
from contextlib import contextmanager

STAGES=('initial','before_image','after_image','before_setup_primary','after_setup_primary',
        'before_setup_whisper','after_setup_whisper','before_prepare_qwen06','after_prepare_qwen06',
        'before_prepare_sensevoice','after_prepare_sensevoice','before_prepare_whisper','after_prepare_whisper')

def require(ok,why):
    if not ok:raise ValueError(why)

def disk(path):
    usage=shutil.disk_usage(path);dev=Path(path).stat().st_dev
    values=dict(total_bytes=usage.total,used_bytes=usage.used,free_bytes=usage.free,
                device_major=os.major(dev),device_minor=os.minor(dev))
    require(all(type(v) is int and v>=0 for v in values.values()) and usage.free<=usage.total,
            'bounded numeric disk snapshot')
    values['status']='available'
    return values

def observe_disk(path):
    try:return disk(path)
    except OSError as error:
        return dict(status='unavailable',errno=error.errno if type(error.errno) is int else 0)

@contextmanager
def observed(capacity,before,after):
    capacity.record(before)
    try:yield
    except BaseException:
        try:capacity.record(after)
        except Exception:pass  # preserve the primary stage failure, never continue
        raise
    else:capacity.record(after)

class Capacity:
    def __init__(self,work,workspace,env,started):
        self.work=Path(work);self.workspace=Path(workspace);self.started=started;self.rows=[]
        require(re.fullmatch('[0-9a-f]{40}',env['GITHUB_SHA']) is not None and
                re.fullmatch('[1-9][0-9]{0,19}',env['GITHUB_RUN_ID']) is not None,'same job identity')
        self.identity=dict(run_id=env['GITHUB_RUN_ID'],source_head=env['GITHUB_SHA'])
    def record(self,stage):
        require(stage in STAGES and stage not in [r['stage'] for r in self.rows] and
                len(self.rows)<len(STAGES),'fixed once-only storage checkpoints')
        if self.rows:require(STAGES.index(stage)>STAGES.index(self.rows[-1]['stage']),'storage checkpoint order')
        workspace=observe_disk(self.work)
        try: same_device=self.work.stat().st_dev==self.workspace.stat().st_dev
        except OSError:same_device=None
        row=dict(stage=stage,elapsed_milliseconds=max(0,int((time.monotonic()-self.started)*1000)),
                 root=observe_disk(Path('/')),workspace=workspace,work_workspace_same_device=same_device)
        docker=Path('/var/lib/docker')
        row['docker']=observe_disk(docker)
        self.rows.append(row)
        public=self.work/'aggregate-only';public.mkdir(exist_ok=True)
        target=public/'capacity.json';temporary=public/'capacity.next.json'
        with temporary.open('x') as stream:
            json.dump(dict(schema='native-prepare-same-job-capacity-v1',**self.identity,
                required_initial_free_bytes=25*1024**3,primary_initial_free_bytes=16*1024**3,
                whisper_initial_free_bytes=8*1024**3,observations=self.rows,
                runner_capacity_cleanup_performed=False,retained_setup_temp_and_container_cleanup_unchanged=True,
                snapshots_are_sequential_not_atomic=True,previous_runner_capacity_is_not_evidence=True),stream,
                sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(target)
        return row
