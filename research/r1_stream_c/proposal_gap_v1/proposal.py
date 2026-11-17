#!/usr/bin/env python3
"""Pure token postprocessing; no audio, tensor framework, or model execution."""
from __future__ import annotations

WORDS = ((1027, 1401, 1462, 2093), (1462, 2093, 1462, 2093))
UINT64_MAX = (1 << 64) - 1


class Proposal:
    """The original CTC suffix policy with a separately enabled blank expiry."""

    def __init__(self, max_blank_gap=0):
        if type(max_blank_gap) is not int or not 0 <= max_blank_gap <= 150:
            raise ValueError("max_blank_gap must be an integer in 0..150")
        self.limit = max_blank_gap
        self.rows = self.events = self.last = self.blanks = self.expirations = 0
        self.ids, self.frames, self.cooldown = [], [], [0, 0]

    def step(self, token, frame=None):
        frame = self.rows if frame is None else frame
        if (type(token) is not int or not 0 <= token < 2601 or
                type(frame) is not int or frame != self.rows or
                frame > UINT64_MAX - 51):
            raise ValueError("invalid token or nonsequential/overflowing frame")
        self.rows += 1
        if token == 0:
            self.last = 0
            if self.limit:
                self.blanks = min(self.limit, self.blanks + 1)
                if self.blanks == self.limit and self.ids:
                    self.ids.clear()
                    self.frames.clear()
                    self.expirations += 1
            return None
        self.blanks = 0
        if token == self.last:
            return None
        self.last = token
        self.ids = (self.ids + [token])[-4:]
        self.frames = (self.frames + [frame])[-4:]
        if len(self.ids) != 4 or frame - self.frames[0] > 150:
            return None
        for index, word in enumerate(WORDS):
            if tuple(self.ids) == word and frame >= self.cooldown[index]:
                self.cooldown[index] = frame + 50
                self.events += 1
                return {"keyword_id": index + 1, "model_frame": frame,
                        "token_frames": list(self.frames)}
        return None


def replay(tokens, max_blank_gap=0):
    decoder = Proposal(max_blank_gap)
    events = []
    for token in tokens:
        event = decoder.step(token)
        if event is not None:
            events.append(event)
    return {"events": events, "frames": decoder.rows,
            "prefix_expirations": decoder.expirations,
            "max_blank_gap": max_blank_gap}


def event_blank_runs(tokens, event):
    """Observed blank runs inside the matched token span, not silence/word times."""
    begin, end = event["token_frames"][0], event["token_frames"][-1]
    maximum = run = 0
    for token in tokens[begin:end + 1]:
        run = run + 1 if token == 0 else 0
        maximum = max(maximum, run)
    return maximum
