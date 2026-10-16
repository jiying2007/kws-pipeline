"""Explicit direct-class event policy and the existing scorer's event schema.

Probabilities are three mutually exclusive classes, not pinyin tokens. This
research policy does not claim equivalence to the shipping C/CTC decoder.
"""
from __future__ import annotations

import math


def class_target(expected_keywords, *, annotation_status):
    """Map an explicitly complete single-window label to a direct class.

    Uses the existing clip-presence expected_keywords convention. The caller
    must additionally establish that the labeled content fits at the chosen
    causal endpoint. Missing/partial labels never become background; multiple or
    repeated keywords need separate labeled windows, not an arbitrary choice.
    """
    if annotation_status != "complete" or type(expected_keywords) is not list:
        raise ValueError("an explicit complete expected_keywords list is required")
    if not expected_keywords:
        return 0
    if (len(expected_keywords) != 1 or type(expected_keywords[0]) is not int
            or expected_keywords[0] not in (1, 2)):
        raise ValueError("each labeled window must contain exactly one target or background")
    return expected_keywords[0]


class DirectEventHead:
    """Argmax + threshold + consecutive frames + per-word cooldown + rearming.

    Keep the same instance across all chunks of a recording. An emitted class
    stays latched while it remains above threshold and the unique argmax, even
    after cooldown. Background, a tie, a low score or another class rearms it.
    No EOF flush/padding is performed. Create/reset only at a recording boundary
    or declared discontinuity, together with frontend/model and sample clocks.
    """

    def __init__(self, thresholds, *, min_consecutive_frames, cooldown_frames):
        if type(thresholds) not in (list, tuple) or len(thresholds) != 2 or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 1
            for x in thresholds
        ):
            raise ValueError("supply two finite per-keyword thresholds in (0,1]")
        if type(min_consecutive_frames) is not int or not 1 <= min_consecutive_frames <= 1000:
            raise ValueError("min_consecutive_frames must be 1..1000")
        if type(cooldown_frames) is not int or not 0 <= cooldown_frames <= 10000:
            raise ValueError("cooldown_frames must be 0..10000")
        self.thresholds = tuple(float(x) for x in thresholds)
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_frames = cooldown_frames
        self.reset()

    def reset(self):
        self.frame = -1
        self.candidate = 0
        self.streak = 0
        self.latched = 0
        self.last_hit = {}

    def accept(self, probabilities):
        if type(probabilities) not in (list, tuple) or len(probabilities) != 3 or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("expected finite normalized [background,keyword1,keyword2] probabilities")
        # Validate before advancing any timeline/state.
        maximum = max(probabilities)
        winners = [i for i, p in enumerate(probabilities) if p == maximum]
        current = winners[0] if len(winners) == 1 else 0
        if current and probabilities[current] < self.thresholds[current - 1]:
            current = 0
        self.frame += 1
        if current != self.candidate:
            self.candidate, self.streak = current, 0
        self.streak = self.streak + 1 if current else 0
        if current != self.latched:
            self.latched = 0
        if (not current or self.streak < self.min_consecutive_frames
                or self.latched == current
                or self.frame - self.last_hit.get(current, -10**12) < self.cooldown_frames):
            return []
        self.latched = current
        self.last_hit[current] = self.frame
        return [{"keyword_id": current, "confidence": float(probabilities[current]),
                 "time_s": (400 + self.frame * 320) / 16000}]


def stream_events(model, feature_chunks, event_head, *, recording):
    """Evaluate one unpadded recording's [time,32] feature chunks continuously.

    Reuse training/frontend.py's complete-window features or the existing
    research/model_comparison/runner.py iter_wav_features bridge. Model state,
    event streak, latch, cooldown and frame clock persist across empty/uneven
    chunks. The feature producer must preserve its own overlap/PCEN state.
    Only logmel is currently used by that existing WAV bridge. Frames must be
    consecutive from recording sample 0, at 16 kHz / 400 samples / hop 320;
    cropped, skipped or other-geometry frames have incompatible event times.
    """
    import torch

    if not isinstance(recording, str) or not recording.strip():
        raise ValueError("recording must be a nonempty string")
    if type(event_head) is not DirectEventHead or event_head.frame != -1:
        raise ValueError("start each recording with a fresh event head")
    if model.training:
        raise ValueError("set model.eval() before continuous inference")
    parameter = next(model.parameters())
    state = model.init_state(1, device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        for chunk in feature_chunks:
            if not isinstance(chunk, torch.Tensor) or chunk.ndim != 2:
                raise ValueError("feature chunks must be [time,32]")
            logits, state = model.forward_chunk(chunk.unsqueeze(0), state)
            for probabilities in logits[:, 0].softmax(-1).tolist():
                for event in event_head.accept(probabilities):
                    yield {"recording": recording, **event}
