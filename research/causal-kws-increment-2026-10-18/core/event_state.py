class EventReplay:
    """Two independent trigger states. Actual timestamps only; no score filtering.
    Replay each threshold afresh, re-arm after two subthreshold frames plus cooldown.
    Hysteresis uses 0.6*threshold. Per-word cooldown fixed 1000ms.
    """
    def __init__(self,thresholds,cooldown_ms=1000):
        if len(thresholds)!=2 or any(not 0<t<1 for t in thresholds):raise ValueError('thresholds')
        self.thresholds=thresholds;self.cooldown_ms=cooldown_ms;self.reset()
    def reset(self):self.armed=[True,True];self.low=[0,0];self.last=[-float('inf')]*2;self.time=-float('inf')
    def step(self,available_ms,probabilities):
        if available_ms<=self.time or len(probabilities)!=2 or any(not 0<=p<=1 for p in probabilities):raise ValueError('stream ordering/probabilities')
        self.time=available_ms;events=[]
        for k,(p,th) in enumerate(zip(probabilities,self.thresholds)):
            if p<.6*th:self.low[k]+=1
            else:self.low[k]=0
            if not self.armed[k] and self.low[k]>=2 and available_ms-self.last[k]>=self.cooldown_ms:self.armed[k]=True
            if self.armed[k] and p>=th:
                events.append(dict(keyword=k,available_ms=available_ms,score=p));self.last[k]=available_ms;self.armed[k]=False;self.low[k]=0
        return events
