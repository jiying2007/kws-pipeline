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
