#!/usr/bin/env python3
"""Development-only boundary-continuity regression with non-contradictory labels.

Positive stress keeps both sides from ONE original full-keyword utterance and
inserts silence at a CTC-aligned internal token boundary. Boundary negatives
concatenate TWO independently recorded half utterances separated by silence.
The two constructions intentionally have different source semantics; this tool
must never relabel the stitched negative as a legal within-word pause.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'eval'), str(ROOT / 'tools')]

from diagnose_sequence_margin_runtime_gap import log_softmax, read_trace_logits
from frozen_speech_ablation import resolve, sha, verify_pool, write
from startup_context_evaluation import collect, measure, pcm, wav
from wake_pressure_balance import keyword_target_sequences

POLICY = 'boundary-continuity-corrected-labels-v1'
SPLITS = ('train', 'calibration', 'test')
SPLIT_ROLES = {
    'train': 'development-fit',
    'calibration': 'development-calibration',
    'test': 'development-feedback',
}
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
EVALUATION_LEAD_SAMPLES = 16000
DEFAULT_GAP_MS = 400


def _finite_rows(logits: list[list[float]]) -> None:
    if not logits or not all(isinstance(row, list) and len(row) >= 2 for row in logits):
        raise ValueError('forced alignment requires nonempty logits with vocab >=2')
    width = len(logits[0])
    if any(len(row) != width or any(not math.isfinite(v) for v in row) for row in logits):
        raise ValueError('forced alignment logits must be rectangular and finite')


def ctc_viterbi_alignment(logits: list[list[float]], target: tuple[int, ...]) -> list[int]:
    """Return one deterministic best CTC state index per frame.

    State sequence is blank,t0,blank,t1,... . Transition rules are standard CTC;
    ties prefer the smaller predecessor state so evidence is reproducible.
    """
    _finite_rows(logits)
    if not target or any(type(t) is not int or t <= 0 or t >= len(logits[0]) for t in target):
        raise ValueError('forced alignment target contains invalid token id')
    labels: list[int] = [0]
    for token in target:
        labels.extend((token, 0))
    states = len(labels)
    neg = float('-inf')
    probs = log_softmax(logits)
    previous = [neg] * states
    previous[0] = probs[0][0]
    if states > 1:
        previous[1] = probs[0][labels[1]]
    parents: list[list[int]] = [[-1] * states]

    for frame in range(1, len(probs)):
        current = [neg] * states
        row_parent = [-1] * states
        for state, label in enumerate(labels):
            choices = [(previous[state], state)]
            if state > 0:
                choices.append((previous[state - 1], state - 1))
            if state > 1 and label != 0 and label != labels[state - 2]:
                choices.append((previous[state - 2], state - 2))
            best_score, best_parent = max(choices, key=lambda item: (item[0], -item[1]))
            if math.isfinite(best_score):
                current[state] = best_score + probs[frame][label]
                row_parent[state] = best_parent
        previous = current
        parents.append(row_parent)

    finals = [states - 1]
    if states > 1:
        finals.append(states - 2)
    final = max(finals, key=lambda state: (previous[state], -state))
    if not math.isfinite(previous[final]):
        raise ValueError('no finite CTC alignment for target')
    path = [final]
    for frame in range(len(probs) - 1, 0, -1):
        parent = parents[frame][path[-1]]
        if parent < 0:
            raise ValueError('incomplete CTC alignment backtrace')
        path.append(parent)
    path.reverse()
    for index in range(len(target)):
        if (2 * index + 1) not in path:
            raise ValueError('CTC alignment did not consume every target token')
    return path


def internal_split_from_alignment(logits: list[list[float]], target: tuple[int, ...],
                                  event: dict, lead_samples: int = 0) -> dict:
    if len(target) < 2 or len(target) % 2:
        raise ValueError('boundary continuity requires an even-length wake target')
    if type(lead_samples) is not int or lead_samples < 0 or lead_samples % FRAME_HOP_SAMPLES:
        raise ValueError('alignment lead must be nonnegative and hop aligned')
    start = _event_sample(event, 'start_s') + lead_samples
    end = _event_sample(event, 'end_s') + lead_samples
    eligible = [i for i in range(len(logits))
                if start <= i * FRAME_HOP_SAMPLES + FRAME_LENGTH_SAMPLES <= end]
    if not eligible:
        raise ValueError('wake event contains no posterior frames')
    local = [logits[i] for i in eligible]
    path = ctc_viterbi_alignment(local, target)
    middle = len(target) // 2
    left_state, right_state = 2 * (middle - 1) + 1, 2 * middle + 1
    left = [eligible[i] for i, state in enumerate(path) if state == left_state]
    right = [eligible[i] for i, state in enumerate(path) if state == right_state]
    if not left or not right or max(left) >= min(right):
        raise ValueError('CTC alignment has no ordered internal token boundary')
    left_end = max(left) * FRAME_HOP_SAMPLES + FRAME_LENGTH_SAMPLES
    right_start = min(right) * FRAME_HOP_SAMPLES
    absolute_split = ((left_end + right_start) // 2 // FRAME_HOP_SAMPLES) * FRAME_HOP_SAMPLES
    split = absolute_split - lead_samples
    raw_start = _event_sample(event, 'start_s')
    raw_end = _event_sample(event, 'end_s')
    if split % FRAME_HOP_SAMPLES or not raw_start < split < raw_end:
        raise ValueError('aligned internal split lies outside the annotated wake event')
    return {
        'split_samples': split,
        'left_token_last_frame': max(left),
        'right_token_first_frame': min(right),
        'eligible_first_frame': eligible[0],
        'eligible_last_frame': eligible[-1],
        'alignment_policy': 'ctc-viterbi-event-window-token-midpoint-v1',
    }


def _event_sample(event: dict, key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'invalid event {key}')
    sample = round(float(value) * 16000.0)
    if sample < 0 or abs(sample / 16000.0 - float(value)) > 1.0e-8:
        raise ValueError(f'event {key} is not sample aligned')
    return sample


def paused_expected(event: dict, split_samples: int, gap_samples: int,
                    lead_samples: int = EVALUATION_LEAD_SAMPLES) -> dict:
    if type(split_samples) is not int or type(gap_samples) is not int or gap_samples <= 0:
        raise ValueError('invalid pause geometry')
    start, end = _event_sample(event, 'start_s'), _event_sample(event, 'end_s')
    if not start < split_samples < end:
        raise ValueError('pause split must be strictly inside wake event')
    return {**event, 'start_s': (start + lead_samples) / 16000.0,
            'end_s': (end + lead_samples + gap_samples) / 16000.0}


def fullword_pause(raw: bytes, split_samples: int, gap_samples: int) -> bytes:
    samples = len(raw) // 2
    if (not isinstance(raw, bytes) or not raw or len(raw) % 2 or type(split_samples) is not int
            or not 0 < split_samples < samples or split_samples % FRAME_HOP_SAMPLES
            or type(gap_samples) is not int or gap_samples <= 0 or gap_samples % FRAME_HOP_SAMPLES):
        raise ValueError('invalid fullword pause input')
    byte_split = split_samples * 2
    return raw[:byte_split] + b'\0\0' * gap_samples + raw[byte_split:]


def stitched_boundary(left: bytes, right: bytes, gap_samples: int) -> tuple[bytes, int]:
    if (not isinstance(left, bytes) or not isinstance(right, bytes) or not left or not right
            or len(left) % 2 or len(right) % 2 or type(gap_samples) is not int
            or gap_samples <= 0 or gap_samples % FRAME_HOP_SAMPLES):
        raise ValueError('invalid stitched boundary input')
    alignment_pad = (-(len(left) // 2)) % FRAME_HOP_SAMPLES
    silence = alignment_pad + gap_samples
    return left + b'\0\0' * silence + right, silence


def same_voice_half_pairs(rows: list[dict]) -> dict[int, tuple[int, int]]:
    lookup: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for index, row in enumerate(rows):
        voice = row.get('source_provenance', {}).get('voice_id')
        ids = tuple(row.get('target_ids', ()))
        if voice and len(ids) > 0 and not row.get('expected'):
            lookup.setdefault((voice, ids), []).append(index)
    result: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        if len(row.get('expected', ())) != 1:
            continue
        voice = row.get('source_provenance', {}).get('voice_id')
        target = tuple(row.get('target_ids', ()))
        if not voice or len(target) < 2 or len(target) % 2:
            raise ValueError('positive wake rows require voice id and even target length')
        middle = len(target) // 2
        halves = []
        for half in (target[:middle], target[middle:]):
