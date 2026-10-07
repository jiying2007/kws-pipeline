"""Owned-process observation fallback for exact unavailable proc children endpoints.

No process is spawned here. Discovery and limits are sampled, never a claim of
continuous lifetime absence. Ambiguous or non-ENOENT reads fail closed.
"""
import errno,os
from pathlib import Path
from common import require
INVENTORY_CAP=4096

def parse_stat(pid,text):
 require(type(pid)is int and pid>0 and type(text)is str and len(text)<=8192,'PROC_STAT_SCHEMA')
 head,sep,tail=text.rpartition(')');require(bool(sep)and head.split('(',1)[0].strip()==str(pid),'PROC_STAT_PID')
 words=tail.split();require(len(words)>=20,'PROC_STAT_FIELDS')
 try:values={k:int(words[i])for k,i in [('ppid',1),('pgrp',2),('session',3),('utime',11),('stime',12),('cutime',13),('cstime',14),('starttime',19)]}
 except (ValueError,IndexError):raise ValueError('PROC_STAT_FIELDS')from None
 require(all(v>=0 for v in values.values()),'PROC_STAT_NEGATIVE')
 require(values['pgrp']>0 and values['session']>0,'PROC_STAT_GROUP')
 return dict(pid=pid,state=words[0],**values)

def read_stat(pid):return parse_stat(pid,(Path('/proc')/str(pid)/'stat').read_text())
def list_pids():
 values=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   values.append(int(p.name));require(len(values)<=INVENTORY_CAP,'PROC_INVENTORY_LIMIT')
 require(all(p>0 for p in values)and len(values)==len(set(values)),'PROC_INVENTORY_SCHEMA')
 return sorted(values)
def pid_absent(pid):
 path=Path('/proc')/str(pid)
 try:path.stat();return False
 except FileNotFoundError as e:
  require(e.errno==errno.ENOENT and e.filename==str(path),'PROC_DISAPPEARANCE_AMBIGUOUS');return True

def stat_inventory():
 rows={};disappeared=[]
 try:pids=list_pids()
 except OSError:raise ValueError('PROC_INVENTORY_LIST_UNAVAILABLE')from None
 for pid in pids:
  try:rows[pid]=read_stat(pid)
  except FileNotFoundError as e:
   exact=e.errno==errno.ENOENT and e.filename==str(Path('/proc')/str(pid)/'stat')
   try:gone=exact and pid_absent(pid)
   except OSError:gone=False
   require(gone,'PROC_INVENTORY_STAT_UNAVAILABLE');disappeared.append(pid)
  except (OSError,UnicodeError):raise ValueError('PROC_INVENTORY_STAT_UNAVAILABLE')from None
 return rows,disappeared

def select_owned(rows,root_pid,root_birth,previous=None,inventory_cap=INVENTORY_CAP,process_cap=4):
 previous=previous or {};require(type(rows)is dict and 0<len(rows)<=inventory_cap,'PROC_INVENTORY_LIMIT')
 require(root_pid in rows,'OWNED_ROOT_UNAVAILABLE');root=rows[root_pid]
 require(root['starttime']==root_birth,'OWNED_PID_REUSE')
 require(root['pgrp']==root_pid and root['session']==root_pid,'OWNED_ROOT_GROUP_ESCAPE')
 selected={p for p,r in rows.items()if r['pgrp']==root_pid or r['session']==root_pid};selected.add(root_pid)
 for pid,birth in previous.items():
  if pid in rows:
   require(rows[pid]['starttime']==birth,'OWNED_PID_REUSE');selected.add(pid)
 while True:
  more=selected|{p for p,r in rows.items()if r['ppid']in selected}
  if more==selected:break
  selected=more
 require(len(selected)<=process_cap,'PROCESS_LIMIT')
 for pid in selected:
  r=rows[pid];require(r['pgrp']==root_pid and r['session']==root_pid,'OWNED_CHILD_GROUP_ESCAPE')
 return sorted(selected)

def same_identity(a,b):return all(a[k]==b[k]for k in ('pid','ppid','pgrp','session','starttime'))

def observe_inventory(proc,max_processes,reader):
 if proc.poll()is not None:return None
 rows,disappeared=stat_inventory();root=proc.pid
 if root not in rows and proc.poll()is not None:return None
 require(root in rows,'OWNED_ROOT_UNAVAILABLE')
 birth=getattr(proc,'_a20_root_birth',rows[root]['starttime']);previous=getattr(proc,'_a20_observed_owned_births',{})
 for pid in previous:
  if pid not in rows and pid not in disappeared:
   try:gone=pid_absent(pid)
   except OSError:gone=False
   require(gone,'TRACKED_PROCESS_VISIBILITY_REQUIRED');disappeared.append(pid)
 selected=select_owned(rows,root,birth,previous,process_cap=max_processes)
 states=[];used=[];cpu=0.;availability={};ticks=os.sysconf('SC_CLK_TCK');require(type(ticks)is int and ticks>0,'CLOCK_TICK_RATE')
 for pid in selected:
  try:
   require(os.getpgid(pid)==root and os.getsid(pid)==root,'OWNED_CHILD_GROUP_ESCAPE')
   before=read_stat(pid);require(same_identity(rows[pid],before),'OWNED_IDENTITY_CHANGED_DURING_SAMPLE')
   s=reader(pid)
   require(not s['critical_errors'],'MANDATORY_PROC_FIELDS_UNAVAILABLE')
   mode=s.get('children_observation');require(mode in ('AVAILABLE_AT_SAMPLE','NOT_AVAILABLE'),'CHILD_VISIBILITY_REQUIRED')
   after=read_stat(pid);require(same_identity(before,after),'OWNED_IDENTITY_CHANGED_DURING_SAMPLE')
   require(os.getpgid(pid)==root and os.getsid(pid)==root,'OWNED_CHILD_GROUP_ESCAPE')
  except (FileNotFoundError,ProcessLookupError):
   if pid==root and proc.poll()is not None:return None
   try:gone=pid_absent(pid)
   except OSError:gone=False
   require(gone,'PROC_DISAPPEARANCE_AMBIGUOUS');disappeared.append(pid);continue
  require(type(s.get('VmRSS'))is int and s['VmRSS']>=0 and type(s.get('Threads'))is int and s['Threads']>=1,'MANDATORY_PROC_FIELDS_UNAVAILABLE')
  cpu+=sum(after[k]for k in ('utime','stime','cutime','cstime'))/ticks;states.append(s);used.append(pid);availability[str(pid)]=mode
 require(len(states)>0,'NO_OWNED_LIVE_SAMPLE')
 proc._a20_root_birth=birth;proc._a20_observed_owned_births={p:rows[p]['starttime']for p in used}
 return dict(rss=sum(s['VmRSS']*1024 for s in states),vmsize=sum(s.get('VmSize',0)*1024 for s in states),threads=sum(s['Threads']for s in states),processes=len(states),cpu=cpu,
  io={k:sum(s['io'][k]['value']for s in states)if all(s['io'][k]['status']=='AVAILABLE'for s in states)else None for k in ('rchar','wchar','read_bytes','write_bytes')},
  observation=dict(method='BOUNDED_PROC_STAT_OWNED_PGID_SID_PPID_SNAPSHOT',owned_root=root,root_starttime=birth,owned_pids=used,children_observation=availability,disappeared_during_snapshot=sorted(set(disappeared)&(set(previous)|set(selected)|{root})),unrelated_disappeared_count=len(set(disappeared)-(set(previous)|set(selected)|{root})),snapshot_atomic=False,continuous_lifetime_absence_or_peak_proven=False))


def compact_observation(observed,elapsed_ms):
 require(len(observed['owned_pids'])<=4 and len(observed['disappeared_during_snapshot'])<=8,'MONITOR_EVIDENCE_LIMIT')
 require(0<=observed['unrelated_disappeared_count']<=INVENTORY_CAP,'MONITOR_EVIDENCE_LIMIT')
 return [elapsed_ms,observed['root_starttime'],observed['owned_pids'],[1 if observed['children_observation'][str(p)]=='AVAILABLE_AT_SAMPLE' else 0 for p in observed['owned_pids']],observed['disappeared_during_snapshot'],observed['unrelated_disappeared_count']]
