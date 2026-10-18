"""Narrow online merger, integer acquisition cursor. No model/branch state."""
class OnlineMerge:
 def __init__(self):
  self.last_cursor=-1;self.last_kept={};self.rows=0;self.suppressed=0
 def step(self,actual_supplied_samples,b_events,a_kw2_events):
  if type(actual_supplied_samples) is not int or actual_supplied_samples<=self.last_cursor:raise ValueError('strict real acquisition cursor')
  self.last_cursor=actual_supplied_samples;self.rows+=1;out=[];suppressed=[]
  # Caller gathers BOTH branches for this one context before invoking merger.
  for source,events in (('B',b_events),('A_KW2',a_kw2_events)):
   for ev in events:
    if ev['keyword'] not in (0,1) or (source=='A_KW2' and ev['keyword']!=1):raise ValueError('keyword branch')
    if ev['actual_supplied_samples']!=actual_supplied_samples:raise ValueError('not the callback cursor')
    k=ev['keyword'];last=self.last_kept.get(k)
    record=dict(ev,source=source)
    if last is not None and actual_supplied_samples-last['actual_supplied_samples']<=8000:
     self.suppressed+=1;suppressed.append(dict(event=record,kept=last));continue
    self.last_kept[k]=record;out.append(record)
  return out,suppressed
