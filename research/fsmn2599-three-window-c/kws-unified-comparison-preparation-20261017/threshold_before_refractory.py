"""Separate synthetic-only candidate: score gate after exact dedup, before refractory.
The source-pinned three-window beam, local reset, and original theta0 file stay unchanged.
"""
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
SOURCE=P.parent/'kws-fsmn-three-window-adapter-20261017'
for name,expected in json.loads((P/'adapter-source-pins.json').read_text()).items():
    if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('adapter source identity mismatch: '+name)
sys.path.insert(0,str(SOURCE))
from three_window_events import ThreeWindowEvents, DecoderError


class ThresholdBeforeRefractory(ThreeWindowEvents):
    def __init__(self,recording_id,threshold):
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('fixed finite threshold in [0,1] required')
        self.threshold=float(threshold)
        super().__init__(recording_id)

    def reset_all(self,recording_id):
        super().reset_all(recording_id)
        self.threshold_rejections=0

    def _merge(self,candidates,row_index):
        # Same merge order, duplicate retention and shared refractory as pinned v2.
        # The only semantic addition is the explicit score gate marked below.
        current_frame=3*row_index
        while self._seen_heap and self._seen_heap[0][0]<=current_frame-378:
            _,key=heapq.heappop(self._seen_heap)
            self._seen.remove(key)
        unique={}
        for c in sorted(candidates,key=self._order):
            key=(c['keyword_id'],c['node_frames'])
            if key in unique or key in self._seen:
                self.duplicate_candidates+=1
                continue
            unique[key]=c
        events=[]
        for key,c in sorted(unique.items(),key=lambda item:self._order(item[1])):
            self._seen.add(key)
            heapq.heappush(self._seen_heap,(c['end_frame'],key))
            # A rejected candidate does not start/extend the global refractory.
            # It DOES retain its exact dedup key; local beam was already reset.
            if c['score']<self.threshold:
                self.threshold_rejections+=1
                continue
            if self.last_accepted_end!=-1 and c['end_frame']-self.last_accepted_end<50:
                self.refractory_rejections+=1
                continue
            self.last_accepted_end=c['end_frame']
            event=dict(c,recording_id=self.recording_id,event_index=self.events_emitted,
                       diagnostic_candidate=True,grid_seconds=.01)
            self.events_emitted+=1
            events.append(event)
        if len(self._seen)>378:
            raise DecoderError('deduplication bound exceeded')
        self.max_dedup_keys=max(self.max_dedup_keys,len(self._seen))
        return events
