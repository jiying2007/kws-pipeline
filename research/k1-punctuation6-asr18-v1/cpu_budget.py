"""Cumulative cgroup CPU stop thresholds plus an inference process CPU guard.

The host poll/kill target is not a kernel lifetime quota and does not promise a
finite overshoot if the host is suspended or removal fails. Retain final usage
before removing every cgroup; missing final usage invalidates accounting.
"""
from pathlib import Path
import ctypes
import errno
import os
import platform
import resource
import time
from campaign_contract import LIMITS, require


def parse_cpu_stat(text):
    rows={}
    for line in text.splitlines():
        pair=line.split(); require(len(pair)==2 and pair[0] not in rows,'cpu.stat shape')
        require(pair[1].isdigit(),'cpu.stat nonnegative integer'); rows[pair[0]]=int(pair[1])
    require('usage_usec' in rows,'cgroup aggregate usage required')
    return rows['usage_usec']


class CpuLedger:
    def __init__(self, clock=time.monotonic):
        self.clock=clock; self.entries={}; self.failed=False; self.last_poll=None; self.max_poll_gap=0

    def register(self,name,stage,reader):
        require(not self.failed and name not in self.entries and stage in LIMITS['cpu_threshold_seconds'],'fresh CPU scope')
        value=reader(); require(type(value) is int and value>=0,'initial CPU usage')
        # Charge from cgroup creation, including startup before registration.
        self.entries[name]={'stage':stage,'reader':reader,'usage_usec':value,'closed':False,'final_verified':False}
        return self.check()

    def sample(self):
        now=self.clock()
        if self.last_poll is not None:self.max_poll_gap=max(self.max_poll_gap,now-self.last_poll)
        self.last_poll=now
        for row in self.entries.values():
            if row['closed']:continue
            value=row['reader']()
            require(type(value) is int and value>=row['usage_usec'],'CPU accounting must not reset')
            row['usage_usec']=value
        return self.check()

    def check(self):
        totals={stage:sum(r['usage_usec'] for r in self.entries.values() if r['stage']==stage)
                for stage in LIMITS['cpu_threshold_seconds']}
        exceeded=[s for s,v in totals.items() if v>=LIMITS['cpu_threshold_seconds'][s]*1000000]
        if sum(totals.values())>=LIMITS['cpu_total_threshold_seconds']*1000000:exceeded.append('total')
        if exceeded:self.failed=True
        return exceeded

    def close(self,name,final_usage):
        row=self.entries[name]; require(not row['closed'],'single CPU finalization')
        require(type(final_usage) is int and final_usage>=row['usage_usec'],'verified final cgroup usage')
        row.update(usage_usec=final_usage,closed=True,final_verified=True); self.check()

    def receipt(self):
        values={n:{k:v for k,v in r.items() if k!='reader'} for n,r in self.entries.items()}
        return {'schema':'qwen12-cumulative-cpu-threshold-v1','containers':values,
            'total_usage_usec':sum(r['usage_usec'] for r in self.entries.values()),
            'thresholds_seconds':LIMITS['cpu_threshold_seconds'],'total_threshold_seconds':6600,
            'threshold_exceeded':self.failed,'all_final_usage_verified':all(r['final_verified'] for r in self.entries.values()),
            'usage_scope':'controlled container cgroups through acknowledged pause before removal; excludes host/image/log/cleanup after pause',
            'configured_poll_seconds':0.2,'maximum_observed_poll_gap_seconds':self.max_poll_gap,
            'kill_command_timeout_seconds':5,'hard_cumulative_kernel_quota':False,
            'overshoot_bound':'No unconditional finite bound: host suspension or unconfirmed kill invalidates claim.'}


def inference_filter():
    """x86_64: disallow new processes; allow only CLONE_THREAD clones.

    clone3 returns ENOSYS so glibc may use its ordinary thread clone fallback.
    Fork/vfork/non-thread clone fail EPERM. Existing Docker seccomp still applies.
    """
    LD=0x20; JEQ=0x15; JSET=0x45; RET=0x06
    kill=0x80000000; allow=0x7fff0000; deny=0x00050000|errno.EPERM
    return [(LD,0,0,4),(JEQ,1,0,0xc000003e),(RET,0,0,kill),
        (LD,0,0,0),(JSET,0,1,0x40000000),(RET,0,0,kill),(JEQ,0,1,57),(RET,0,0,deny),
        (JEQ,0,1,58),(RET,0,0,deny),(JEQ,0,1,435),(RET,0,0,0x00050000|errno.ENOSYS),
        (JEQ,0,3,56),(LD,0,0,16),(JSET,1,0,0x10000),(RET,0,0,deny),(RET,0,0,allow)]


def install_inference_guard(family):
    require(family in ('qwen06','sensevoice','whisper'),'inference family')
    require(platform.machine()=='x86_64' and len(list(Path('/proc/self/task').iterdir()))==1,'single initial x86 process')
    class Filter(ctypes.Structure):
        _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):
        _fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
    entries=inference_filter(); array=(Filter*len(entries))(*(Filter(*x) for x in entries)); program=Program(len(entries),array)
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(38,1,0,0,0)==0,'no-new-privileges before seccomp')
    require(libc.prctl(22,2,ctypes.byref(program))==0,'inference non-thread process guard')
    seconds=LIMITS['cpu_threshold_seconds'][family]
    resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
    require(resource.getrlimit(resource.RLIMIT_CPU)==(seconds,seconds),'inference CPU process limit')
    return {'schema':'qwen12-inference-process-guard-v1','family':family,
        'rlimit_cpu_seconds':seconds,'scope':'all main-process threads; non-thread process creation blocked',
        'scheduler_granularity_not_zero_overshoot':True,'fork_vfork_nonthread_clone':'EPERM','clone3':'ENOSYS'}
