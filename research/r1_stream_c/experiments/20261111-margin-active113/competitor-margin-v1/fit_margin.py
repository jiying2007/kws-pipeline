#!/usr/bin/env python3
"""Isolated NumPy diagnostic: one fixed balanced/margin max-OTHER candidate.

No teacher/backbone execution, PCM, ASR or TTS. This is not the canonical
Torch training runtime and cannot promote/replace R1. `score` consumes saved
predictions only, allowing reporting repair without training or head inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

ROOT = Path('/workspace/scratch/fec7ccd5bb4b')
OLD = Path('/workspace/scratch/1e1fe64498b2')
INPUTS = OLD/'r1-local/evidence-original/KWS_FSMN_R1_EVIDENCE_20261110/inputs'
SAVED = OLD/'r1-local/evidence-qemu'
HEAD = '13661a38064247d38bca88287e48db85bf2ccd79'
INPUT_SHA = '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab'
LABEL_SHA = '8d0b622ebd50570239e45f3092e97e8ea4fa646f051dedddec1291a8225ac7b2'
SAVED_SHA = '8a3d0fbd2822c383b40db2159785419a3b85ab926f06a14c3b3da2a056dd1e44'
MODEL_SHA = '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
FRONTEND_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
DETECTOR_SHA = '35c2b2a36e0f29ee572ee25e368e10a0fde97e36dd8d5e0ea95f71788f6e5594'
DETECTOR = ROOT/'competitor-screen/execution/replay_detector'
PRECISE_IDS = (0, 1027, 1401, 1462, 2093)
HERE = Path(__file__).resolve().parent
CATEGORY_NAMES = ('blank', 'keyword', 'other')
CONFIG = {'seed': 20261111, 'input_dim': 140, 'hidden_dim': 64, 'output_dim': 1,
    'activation': 'relu', 'dtype': 'float64', 'updates': 2000, 'batch_frames': 256,
    'loss': 'mean(SmoothL1(z-y; beta=1) + 0.5*max(0,delta-s*(p-z))**2)', 'lr': 0.001,
    'weight_decay': 0.0001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
    'gradient_clipping': False, 'normalization_std_floor': 1e-6,
    'sampling': 'iid uniform original-winner category; uniform eligible train clip within category; uniform frame within clip/category; with replacement',
    'category_names': list(CATEGORY_NAMES), 'category_probability': [1/3, 1/3, 1/3],
    'category_balance': 'equal probabilities, not exact equal counts within each batch',
    'margin_weight': 1.0, 'margin_cap_standardized': 1.0,
    'margin_definition': 'p=(max_exact_raw-train_y_mean)/train_y_std; s=+1 for original precise winner else -1; delta=min(abs(max_exact_raw-max_other_raw)/train_y_std,1), computed before centering',
    'margin_constants': 'fixed teacher values from known train only; never candidate confidence or development data',
    'train_category_frame_counts': [7904, 616, 559], 'train_precise_other_ties_allowed': 0,
    'joint_change': 'category sampling and margin loss changed together; not an isolated ablation',
    'normalization': 'pooled 9079 known-train frames only; population std ddof=0',
    'fit_absolute_seconds': 600, 'early_stop': False, 'sweep': False, 'retry': False,
    'candidate_tie_rule': 'precise wins when max_precise >= predicted_other; first original ID among precise ties',
    'development_gate': 'all 34 complete E-event byte streams including clocks exactly equal saved R1',
    'train_gate': 'all 96 candidate E-event multisets subset of saved R1; no added wrong-keyword weak-label event',
    'unknown_indices_excluded_from_fit_and_scoring': [96, 97, 98, 99],
    'historical_compact_affine_sum_other': '2026-09-30 FAIL_FIDELITY_UNCHANGED',
    'prior_uniform_clip_mlp64': 'REJECTED_FIXED_RESEARCH_SCREEN_UNCHANGED',
    'runtime_classification': 'isolated NumPy float64 head diagnostic; not canonical Torch retraining',
    'model_promotion_authorized': False}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), 'nonregular or linked file: '+path.name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'duplicate JSON key'); value[key] = item
        return value
    def bad(value): raise ValueError('nonfinite JSON')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def configure_runtime():
    global np
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    need(sys.version_info[:3] == (3, 12, 14) and np.__version__ == '2.3.5', 'isolated diagnostic runtime changed')
    need(len(os.listdir('/proc/self/task')) == 1, 'more than one runtime thread')
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'threads_observed': 1,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'classification': CONFIG['runtime_classification'], 'canonical_torch_retraining': False}


def preregistration(expected_sha, extract=None):
    path = HERE/'PLAN.json'
    need(sha(path) == expected_sha, 'preregistered plan changed')
    plan = load_json(path)
    need(plan['schema'] == 'r1-max-other-margin-v1-preregistered-plan-v1' and
         plan['config'] == CONFIG and plan['candidate_source_sha256'] == sha(Path(__file__)),
         'preregistered policy/source changed')
    if extract is not None:
        need(str(extract) == plan['extraction_path'] and
             sha(extract/'RESULT.json') == plan['extraction_result_sha256'] and
             sha(extract/'PLAN.json') == plan['extraction_plan_sha256'], 'preregistered extraction changed')
    for relative, digest in plan['prior_rejected_run_sha256'].items():
        need(sha(ROOT/'competitor-model'/relative) == digest, 'prior rejected run changed')
    return plan


def read_data(extract):
    need(sha(INPUTS/'INPUT_MANIFEST.json') == INPUT_SHA and
         sha(INPUTS/'original-observations.jsonl') == LABEL_SHA and
         sha(SAVED/'FILE_MANIFEST.json') == SAVED_SHA, 'original input/label/trace binding changed')
    original = load_json(INPUTS/'INPUT_MANIFEST.json')['records']
    observations = [json.loads(x) for x in (INPUTS/'original-observations.jsonl').read_text().splitlines()]
    obs = {r['source_id']: r for r in observations}
    need(len(original) == len(observations) == len(obs) == 134, 'input/label denominator')
    result = load_json(extract/'RESULT.json')
    for key, expected in {'status': 'PASS_FIXED_TEACHER_EXTRACTION', 'source_head': HEAD,
        'model_sha256': MODEL_SHA, 'frontend_sha256': FRONTEND_SHA,
        'input_manifest_sha256': INPUT_SHA, 'saved_trace_manifest_sha256': SAVED_SHA,
        'extraction_count': 134, 'actual_frames': 12534}.items():
        need(result.get(key) == expected, 'teacher extraction binding: '+key)
    rows = result['records']; need([r['index'] for r in rows] == list(range(134)), 'extraction index order')
    dtype = np.dtype([('frame', '<u8'), ('source', '<u8'), ('delivery', '<u8'),
        ('original_best_token', '<u4'), ('other_best_token', '<u4'), ('hidden', '<f8', (140,)),
        ('exact_logits', '<f8', (5,)), ('max_other', '<f8')], align=False)
    need(dtype.itemsize == 1200, 'independent binary record layout')
    traces = load_json(SAVED/'FILE_MANIFEST.json'); records = []
    ids = np.array(PRECISE_IDS, dtype=np.uint32)
    for index, (row, source) in enumerate(zip(rows, original)):
        need(source['index'] == index and row['path'] == f'hidden/{index:03d}.r1h', 'hidden record path')
        path = extract/row['path']; payload = path.read_bytes()
        need(len(payload) == row['bytes'] and sha(path) == row['sha256'], 'hidden record byte identity')
        need(len(payload) >= 64 and (len(payload)-64) % 1200 == 0, 'hidden file length')
        header = struct.unpack('<8s6I5IIQ', payload[:64])
        need(header == (b'R1HD140\0', 1, 64, 1200, 140, 5, 2601, *PRECISE_IDS, 1, 0), 'hidden header')
        array = np.frombuffer(payload, dtype=dtype, offset=64)
        need(len(array) == row['frames'] == source['feature_shape'][0], 'hidden frame denominator')
        need(np.isfinite(array['hidden']).all() and np.isfinite(array['exact_logits']).all() and
             np.isfinite(array['max_other']).all(), 'nonfinite teacher data')
        need(np.array_equal(array['frame'], np.arange(len(array), dtype=np.uint64)) and
             np.array_equal(array['source'], 720+480*array['frame']) and
             np.array_equal(array['delivery'], np.minimum(((array['source']+319)//320)*320, source['samples'])),
             'hidden frame clocks')
        need(np.all(array['other_best_token'] < 2601) and
             not np.isin(array['other_best_token'], ids).any(), 'other teacher class')
        precise = ids[np.argmax(array['exact_logits'], axis=1)]; maximum = np.max(array['exact_logits'], axis=1)
        other_wins = (array['max_other'] > maximum) | ((array['max_other'] == maximum) & (array['other_best_token'] < precise))
        need(np.array_equal(np.where(other_wins, array['other_best_token'], precise), array['original_best_token']),
             'teacher full-argmax tie reconstruction')
        observation = obs[source['source_id']]
        need(observation['split'] == source['split'], 'original split mismatch')
        split = 'unknown' if observation['ctc_weight'] == 0 else observation['split']
        need(split in ('train', 'development', 'unknown') and observation['ctc_weight'] in (0, 0.5, 1.0), 'eligibility')
        trace_name = f'private-traces/{index:03d}-arm_par4.txt'; trace = SAVED/trace_name
        need(sha(trace) == traces[trace_name]['sha256'] and trace.stat().st_size == traces[trace_name]['bytes'], 'baseline trace')
        need(row['trace_sha256'] == traces[trace_name]['sha256'] and row['pcm_sha256'] == source['pcm_sha256'] and
             row['fe_trace_equal'] is True and row['returncode'] == 0 and row['original_split'] == source['split'] and
             row['fit_excluded_unknown'] == (split == 'unknown'), 'teacher per-record identity/parity')
        trace_lines = trace.read_bytes().splitlines(keepends=True)
        expected = b''.join(line for line in trace_lines if line.startswith(b'E,'))
        baseline_frames = [tuple(map(int, line.strip().split(b',')[1:])) for line in trace_lines if line.startswith(b'F,')]
        hidden_frames = [(int(a['frame']), int(a['source']), int(a['delivery']), int(a['original_best_token'])) for a in array]
        need(baseline_frames == hidden_frames, 'hidden teacher frame/argmax parity against saved original trace')
        references = {k['keyword_id']: k['reference'] for k in observation['keywords'] if k['known']}
        if split != 'unknown': need(set(references) == {1, 2}, 'known weak keyword reference')
        records.append({'index': index, 'split': split, 'array': array, 'path': path,
            'sha256': row['sha256'], 'expected': expected, 'references': references})
    need([r['index'] for r in records if r['split'] == 'unknown'] == [96, 97, 98, 99], 'unknown membership')
    for split, clips, frames in (('train', 96, 9079), ('development', 34, 3176), ('unknown', 4, 279)):
        selected = [r for r in records if r['split'] == split]
        need(len(selected) == clips and sum(len(r['array']) for r in selected) == frames, 'split denominator')
    return records


def forward(parameters, x):
    z = x @ parameters['w1'] + parameters['b1']
    activated = np.maximum(z, 0.0)
    return activated @ parameters['w2'] + parameters['b2'][0], z, activated


def loss_gradient(parameters, x, target, precise, sign, required_margin):
    prediction, pre, activated = forward(parameters, x)
    error = prediction-target; absolute = np.abs(error)
    regression = np.where(absolute <= 1.0, 0.5*error*error, absolute-0.5)
    deficit = np.maximum(0.0, required_margin-sign*(precise-prediction))
    penalty = 0.5*deficit*deficit
    loss = np.mean(regression+penalty)
    derivative = (np.clip(error, -1.0, 1.0)+sign*deficit)/len(x)
    hidden_gradient = derivative[:, None]*parameters['w2'][None, :]*(pre > 0.0)
    gradients = {'w1': x.T @ hidden_gradient, 'b1': hidden_gradient.sum(axis=0),
                 'w2': activated.T @ derivative, 'b2': np.array([derivative.sum()], dtype=np.float64)}
    parts = {'smooth_l1_standardized': float(np.mean(regression)),
             'margin_penalty_standardized': float(np.mean(penalty)),
             'margin_violating_frames': int(np.count_nonzero(deficit > 0.0))}
    return float(loss), gradients, parts


def gradient_check():
    x = np.array([[-.3,.2,.7],[.8,-.4,.1],[.2,.5,-.6],[.1,-.9,.3],
                  [-.7,.3,.2],[.4,.6,-.2],[-.5,-.1,.8],[.6,-.8,-.4]], dtype=np.float64)
    parameters = {'w1': np.array([[.3,-.2],[.1,.4],[-.5,.2]]), 'b1': np.array([.07,-.03]),
                  'w2': np.array([.6,-.4]), 'b2': np.array([.11])}
    prediction, pre, _ = forward(parameters, x)
    need(np.min(np.abs(pre)) > 1e-4, 'invented fixture touches a ReLU kink')
    target = prediction-np.array([.4,-.8,1.3,-1.7,.2,-.3,1.2,-1.4])
    sign = np.array([1.,1.,1.,1.,-1.,-1.,-1.,-1.])
    teacher_gap = np.array([.25,.6,1.5,2.,.4,.7,1.8,1.6])
    precise = target+sign*teacher_gap
    required_margin = np.minimum(np.abs(precise-target), 1.0)
    # All teacher constants above are held fixed during finite differences.
    hinge_argument = required_margin-sign*(precise-prediction)
    need(np.min(np.abs(hinge_argument)) > 1e-4, 'invented fixture touches a hinge boundary')
    need(np.array_equal(hinge_argument > 0, np.array([True,False,True,False,False,True,False,True])),
         'invented fixture does not cover both hinge signs/branches')
    _, gradients, _ = loss_gradient(parameters, x, target, precise, sign, required_margin)
    errors = []; checked = 0
    for name, values in parameters.items():
        for index in np.ndindex(values.shape):
            old = float(values[index]); values[index] = old+1e-6
            plus, _, _ = loss_gradient(parameters, x, target, precise, sign, required_margin)
            values[index] = old-1e-6; minus, _, _ = loss_gradient(parameters, x, target, precise, sign, required_margin)
            values[index] = old; numerical = (plus-minus)/(2e-6); analytic = float(gradients[name][index])
            error = abs(numerical-analytic); errors.append(error); checked += 1
            need(error <= 2e-7+1e-5*abs(numerical), 'finite-difference gradient sanity failure')
    return {'status': 'PASS_INVENTED_ARRAY_GRADIENT_CHECK', 'parameters_checked': checked,
            'maximum_absolute_error': max(errors), 'finite_difference_step': 1e-6,
            'branches': ['active/inactive ReLU', 'positive/negative quadratic and linear SmoothL1',
                         'active/inactive squared hinge on precise and OTHER signs',
                         'teacher-capped and uncapped margins'],
            'teacher_constants_held_fixed': True,
            'real_model_or_audio_used': False}


def training_pools(train, starts):
    """Fixed categories from original teacher winners, never candidate predictions."""
    categories = []
    pools = {name: [] for name in CATEGORY_NAMES}
    for clip, row in enumerate(train):
        best = row['array']['original_best_token']
        category = np.where(best == 0, 0, np.where(np.isin(best, PRECISE_IDS[1:]), 1, 2)).astype(np.int8)
        categories.append(category)
        for kind, name in enumerate(CATEGORY_NAMES):
            frames = np.flatnonzero(category == kind)+starts[clip]
            if len(frames): pools[name].append((clip, frames))
    categories = np.concatenate(categories)
    need(np.bincount(categories, minlength=3).tolist() == CONFIG['train_category_frame_counts'],
         'train teacher category counts changed')
    prepared = []; descriptions = {}
    for name in CATEGORY_NAMES:
        need(len(pools[name]) > 0, 'empty training category: '+name)
        clips = np.array([clip for clip, frames in pools[name]], dtype=np.int64)
        lengths = np.array([len(frames) for clip, frames in pools[name]], dtype=np.int64)
        offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)[:-1]))
        frames = np.concatenate([frames for clip, frames in pools[name]])
        prepared.append({'clips': clips, 'lengths': lengths, 'starts': offsets, 'frames': frames})
        descriptions[name] = {'eligible_train_clips': len(clips), 'eligible_train_frames': len(frames)}
    return categories, prepared, descriptions


def sample_batch(rng, pools):
    category = rng.integers(0, 3, size=256)
    selected = np.empty(256, dtype=np.int64); clips = np.empty(256, dtype=np.int64)
    for kind, pool in enumerate(pools):
        positions = np.flatnonzero(category == kind)
        if not len(positions): continue
        chosen = rng.integers(0, len(pool['clips']), size=len(positions))
        offsets = rng.integers(0, pool['lengths'][chosen])
        selected[positions] = pool['frames'][pool['starts'][chosen]+offsets]
        clips[positions] = pool['clips'][chosen]
    return selected, clips, category


def save_npz(path, arrays):
    with path.open('xb') as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())


def fixed_precise_rows():
    model = OLD/'r1-local/integration-original/model/weights/model-int8.r1hq8'
    need(sha(model) == MODEL_SHA, 'fixed teacher model changed')
    data = model.read_bytes(); quant = 88+390474*4; scales = quant+2601*140+4
    biases = scales+2601*8; precise = biases+2601*4
    need(len(data) == precise+5*140*4 == 1960140, 'fixed model layout')
    w = np.frombuffer(data, dtype='<f4', count=700, offset=precise).reshape(5, 140).copy()
    b = np.frombuffer(data, dtype='<f4', count=2601, offset=biases)[list(PRECISE_IDS)].copy()
    return w, b


def fit(extract, out, preregistration_sha):
    need(not out.exists(), 'fresh fit directory required; no retry or overwrite')
    predeclared = preregistration(preregistration_sha, extract)
    need(str(out) == predeclared['output_path'], 'fixed single-candidate output changed')
    runtime = configure_runtime()
    need(load_json(ROOT/'competitor-screen/execution/AGGREGATE.json')['status'] ==
         'PASS_SAVED_EVENT_EQUIVALENCE_ONLY', 'saved OTHER-collapse precondition absent')
    records = read_data(extract); train = [r for r in records if r['split'] == 'train']
    eligible = [r for r in records if r['split'] != 'unknown']
    os.umask(0o077); out.mkdir(parents=True)
    (out/'predictions').mkdir(); (out/'source').mkdir()
    source_path = out/'source/fit_margin.py'
    with source_path.open('xb') as stream:
        stream.write(Path(__file__).read_bytes()); stream.flush(); os.fsync(stream.fileno())
    with (out/'source/PREREGISTRATION.json').open('xb') as stream:
        stream.write((HERE/'PLAN.json').read_bytes()); stream.flush(); os.fsync(stream.fileno())
    plan = {'schema': 'r1-max-other-margin-v1-fixed-diagnostic-plan-v1', 'config': CONFIG, 'runtime': runtime,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'source_head': HEAD,
        'script_sha256': sha(Path(__file__)), 'retained_fit_source': 'source/fit_margin.py', 'extraction_path': str(extract),
        'preregistered_plan_sha256': preregistration_sha, 'retained_preregistration': 'source/PREREGISTRATION.json',
        'extraction_result_sha256': sha(extract/'RESULT.json'), 'extraction_plan_sha256': sha(extract/'PLAN.json'),
        'input_manifest_sha256': INPUT_SHA, 'original_observations_sha256': LABEL_SHA,
        'saved_trace_manifest_sha256': SAVED_SHA, 'teacher_model_sha256': MODEL_SHA,
        'train_records': 96, 'train_frames': 9079, 'exposed_development_records': 34,
        'exposed_development_frames': 3176, 'excluded_unknown_records': 4,
        'final_head_recordings': 130, 'training_head_batches': 2000, 'sampled_train_frames': 512000,
        'backbone_or_teacher_forwards': 0, 'pcm_reads': 0, 'tts': 0, 'asr': 0,
        'learned_parameters': 9089, 'fixed_precise_parameters': 705,
        'checkpoint_and_all_final_other_logits_saved_before_scoring': True}
    need(sha(source_path) == plan['script_sha256'] and sha(out/'source/PREREGISTRATION.json') == preregistration_sha,
         'retained fit source/preregistration drift')
    save_json(out/'PLAN.json', plan)
    state = {'schema': 'r1-max-other-margin-v1-fit-terminal-v1', 'status': 'RUNNING',
        'plan_sha256': sha(out/'PLAN.json'), 'optimizer_updates_completed': 0,
        'final_predictions_completed': 0, 'backbone_or_teacher_forwards': 0,
        'model_promotion_authorized': False, 'checkpoint': None}
    start = time.monotonic()
    try:
        save_json(out/'GRADIENT_CHECK.json', gradient_check())
        x = np.concatenate([r['array']['hidden'] for r in train], axis=0)
        raw_target = np.concatenate([r['array']['max_other'] for r in train])
        raw_precise = np.concatenate([np.max(r['array']['exact_logits'], axis=1) for r in train])
        original_best = np.concatenate([r['array']['original_best_token'] for r in train])
        sign = np.where(np.isin(original_best, PRECISE_IDS), 1.0, -1.0)
        need(np.count_nonzero(raw_precise == raw_target) == 0 and np.all(sign*(raw_precise-raw_target) > 0.0),
             'unexpected training teacher precise/OTHER tie or sign')
        x_mean = x.mean(axis=0); x_scale = np.maximum(x.std(axis=0), 1e-6)
        y_mean = float(raw_target.mean()); y_scale = max(float(raw_target.std()), 1e-6)
        x = (x-x_mean)/x_scale; target = (raw_target-y_mean)/y_scale
        precise = (raw_precise-y_mean)/y_scale
        required_margin = np.minimum(np.abs(raw_precise-raw_target)/y_scale, 1.0)
        lengths = np.array([len(r['array']) for r in train], dtype=np.int64)
        starts = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)[:-1]))
        categories, pools, pool_descriptions = training_pools(train, starts)
        constants = {'x_mean': x_mean, 'x_scale': x_scale, 'y_mean': np.array([y_mean]), 'y_scale': np.array([y_scale]),
            'teacher_other_standardized': target, 'teacher_max_precise_standardized': precise,
            'winner_sign': sign, 'required_margin_standardized': required_margin, 'frame_category': categories,
            'frame_clip_index': np.repeat(np.arange(96, dtype=np.int64), lengths)}
        save_npz(out/'TRAIN_CONSTANTS.npz', constants)
        state['train_constants'] = {'path': 'TRAIN_CONSTANTS.npz', 'bytes': (out/'TRAIN_CONSTANTS.npz').stat().st_size,
                                    'sha256': sha(out/'TRAIN_CONSTANTS.npz')}
        save_json(out/'TRAIN_SETUP.json', {'train_only': True, 'training_teacher_precise_other_ties': 0,
            'category_pools': pool_descriptions, 'frame_category_counts': np.bincount(categories, minlength=3).tolist(),
            'teacher_margin_capped_frames': int(np.count_nonzero(np.abs(raw_precise-raw_target)/y_scale > 1.0)),
            'train_constants': state['train_constants'], 'normalization_frames': 9079, 'normalization_clips': 96,
            'preregistered_plan_sha256': preregistration_sha, 'config': CONFIG})
        rng = np.random.default_rng(CONFIG['seed'])
        p = {'w1': rng.normal(0.0, math.sqrt(2/140), (140,64)), 'b1': np.zeros(64),
             'w2': rng.normal(0.0, math.sqrt(1/64), 64), 'b2': np.zeros(1)}
        moment = {k: np.zeros_like(v) for k, v in p.items()}
        variance = {k: np.zeros_like(v) for k, v in p.items()}
        clip_exposures = np.zeros(96, dtype=np.int64)
        category_exposures = np.zeros(3, dtype=np.int64)
        category_clip_exposures = np.zeros((3, 96), dtype=np.int64)
        def expired(signum, frame): raise TimeoutError('fixed absolute fit budget 600 seconds exhausted')
        previous_handler = signal.signal(signal.SIGALRM, expired)
        fit_start = time.monotonic(); signal.setitimer(signal.ITIMER_REAL, 600)
        try:
            with (out/'loss.jsonl').open('x', encoding='utf-8') as log:
                for update in range(1, 2001):
                    selected, chosen, chosen_categories = sample_batch(rng, pools)
                    need(np.array_equal(categories[selected], chosen_categories), 'sampling category mismatch')
                    loss, gradients, parts = loss_gradient(p, x[selected], target[selected], precise[selected],
                                                          sign[selected], required_margin[selected])
                    need(math.isfinite(loss) and all(np.isfinite(g).all() for g in gradients.values()), 'nonfinite fit')
                    for key in p:
                        moment[key] *= .9; moment[key] += .1*gradients[key]
                        variance[key] *= .999; variance[key] += .001*gradients[key]*gradients[key]
                        p[key] *= 1-.001*.0001
                        p[key] -= .001*(moment[key]/(1-.9**update))/(np.sqrt(variance[key]/(1-.999**update))+1e-8)
                    need(all(np.isfinite(v).all() for v in p.values()), 'nonfinite fitted parameters')
                    clip_exposures += np.bincount(chosen, minlength=96)
                    category_counts = np.bincount(chosen_categories, minlength=3)
                    category_exposures += category_counts
                    for kind in range(3):
                        category_clip_exposures[kind] += np.bincount(chosen[chosen_categories == kind], minlength=96)
                    state['optimizer_updates_completed'] = update
                    log.write(json.dumps({'update': update, 'total_loss_standardized': loss, **parts,
                        'batch_frames': 256, 'selected_train_frame_indices_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                        'selected_categories_sha256': hashlib.sha256(chosen_categories.astype('<i8').tobytes()).hexdigest(),
                        'batch_category_counts': category_counts.tolist(),
                        'elapsed_fit_seconds': time.monotonic()-fit_start}, allow_nan=False)+'\n')
                    log.flush()
                    if update % 100 == 0: os.fsync(log.fileno())
                os.fsync(log.fileno())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)
        state['fit_seconds'] = time.monotonic()-fit_start
        need(state['fit_seconds'] <= 600 and len(os.listdir('/proc/self/task')) == 1, 'fit resource boundary')
        need(int(clip_exposures.sum()) == int(category_exposures.sum()) == int(category_clip_exposures.sum()) == 512000,
             'fixed sampled-frame denominator')
        precise_w, precise_b = fixed_precise_rows()
        metadata = {'schema': 'r1-max-other-margin-v1-numpy-checkpoint-v1', 'config': CONFIG,
            'runtime': runtime, 'plan_sha256': sha(out/'PLAN.json'), 'script_sha256': plan['script_sha256'],
            'preregistered_plan_sha256': preregistration_sha, 'train_constants_sha256': state['train_constants']['sha256'],
            'train_setup_sha256': sha(out/'TRAIN_SETUP.json'),
            'extraction_result_sha256': plan['extraction_result_sha256'], 'teacher_model_sha256': MODEL_SHA,
            'updates_completed': 2000, 'rng_state': rng.bit_generator.state,
            'normalization_derived_only_from_known_train': True,
            'fixed_precise_evaluation': 'reused exact saved C logits; no reassociated NumPy recomputation',
            'standalone_full_model': False, 'model_promotion_authorized': False}
        arrays = {**p, **{'adam_m_'+k: v for k, v in moment.items()},
            **{'adam_v_'+k: v for k, v in variance.items()}, 'x_mean': x_mean, 'x_scale': x_scale,
            'y_mean': np.array([y_mean]), 'y_scale': np.array([y_scale]),
            'precise_ids': np.array(PRECISE_IDS, dtype=np.uint32), 'precise_weights_f32': precise_w,
            'precise_biases_f32': precise_b, 'train_clip_sample_counts': clip_exposures,
            'train_category_sample_counts': category_exposures, 'train_category_clip_sample_counts': category_clip_exposures,
            'metadata_json': np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))}
        checkpoint = out/'checkpoint.npz'; save_npz(checkpoint, arrays)
        with np.load(checkpoint, allow_pickle=False) as saved:
            need(set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items()), 'checkpoint readback')
            final = {k: saved[k].copy() for k in p}
        state['checkpoint'] = {'path': 'checkpoint.npz', 'bytes': checkpoint.stat().st_size, 'sha256': sha(checkpoint)}
        predictions = []
        for row in eligible:
            index = row['index']; other, _, _ = forward(final, (row['array']['hidden']-x_mean)/x_scale)
            other = other*y_scale+y_mean; need(np.isfinite(other).all(), 'nonfinite final predicted logits')
            path = out/'predictions'/f'{index:03d}.npy'
            with path.open('xb') as stream:
                np.save(stream, other.astype('<f8'), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            predictions.append({'index': index, 'split': row['split'], 'path': f'predictions/{index:03d}.npy',
                'bytes': path.stat().st_size, 'sha256': sha(path), 'frames': len(other), 'hidden_sha256': row['sha256']})
            state['final_predictions_completed'] += 1
        save_json(out/'PREDICTIONS.json', {'schema': 'r1-max-other-margin-v1-final-predictions-v1', 'checkpoint': state['checkpoint'],
            'extraction_result_sha256': plan['extraction_result_sha256'], 'records': predictions,
            'all_final_predictions_persisted_before_scoring': True, 'final_head_recordings': 130,
            'excluded_unknown_indices': [96,97,98,99]})
        need(all(sha(r['path']) == r['sha256'] for r in records) and sha(extract/'RESULT.json') == plan['extraction_result_sha256'] and
             sha(Path(__file__)) == plan['script_sha256'] and sha(HERE/'PLAN.json') == preregistration_sha and
             sha(out/'TRAIN_CONSTANTS.npz') == state['train_constants']['sha256'], 'post-fit source/extraction/constant drift')
        preregistration(preregistration_sha, extract)
        state['status'] = 'PASS_FIXED_FIT_AND_PREDICTIONS_NOT_PROMOTED'
    except BaseException as exc:
        state['status'] = 'STOPPED_NO_RETRY'; state['failure'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        state['wall_seconds'] = time.monotonic()-start
        save_json(out/'FIT_TERMINAL.json', state)
    score(out, out/'score')


def metric(values):
    return {'mae': float(np.mean(values)), 'p95_absolute_error': float(np.quantile(values, .95)),
            'max_absolute_error': float(np.max(values)), 'frames': len(values)}


def score(run, out):
    """Saved-prediction scoring only; deliberately never calls forward()."""
    configure_runtime(); need(not out.exists(), 'fresh score directory required; no overwrite')
    plan = load_json(run/'PLAN.json'); predictions = load_json(run/'PREDICTIONS.json')
    need(plan['config'] == CONFIG and plan['retained_fit_source'] == 'source/fit_margin.py' and
         sha(run/'source/fit_margin.py') == plan['script_sha256'], 'fixed scoring policy/retained fit source changed')
    need(plan['retained_preregistration'] == 'source/PREREGISTRATION.json' and
         sha(run/'source/PREREGISTRATION.json') == plan['preregistered_plan_sha256'], 'retained preregistration changed')
    scoring_source_sha = sha(Path(__file__))
    need(predictions['all_final_predictions_persisted_before_scoring'] is True and predictions['final_head_recordings'] == 130,
         'all final logits must exist before scoring')
    need(sha(run/'checkpoint.npz') == predictions['checkpoint']['sha256'], 'retained checkpoint changed')
    extract = Path(plan['extraction_path'])
    need(sha(extract/'RESULT.json') == plan['extraction_result_sha256'] == predictions['extraction_result_sha256'], 'extraction changed')
    records = read_data(extract); selected = [r for r in records if r['split'] != 'unknown']
    need([r['index'] for r in predictions['records']] == [r['index'] for r in selected], '130 prediction membership')
    need(sha(DETECTOR) == DETECTOR_SHA and sha(ROOT/'matrix-build/sdk/par4/src/fsmn_detector.c') ==
         '5ef3fc5410d30588cb516253b3f40e9299316178d5e54fe46ad22fd720314dcb', 'unchanged C detector binding')
    cached = {}
    for descriptor, row in zip(predictions['records'], selected):
        path = run/descriptor['path']
        need(descriptor['path'] == f"predictions/{row['index']:03d}.npy" and descriptor['split'] == row['split'] and
             path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'] and
             descriptor['hidden_sha256'] == row['sha256'], 'saved prediction binding')
        other = np.load(path, allow_pickle=False)
        need(other.dtype == np.dtype('<f8') and other.shape == (len(row['array']),) and np.isfinite(other).all(), 'saved prediction shape/value')
        cached[row['index']] = other
    os.umask(0o077); out.mkdir(parents=True)
    for name in ('claims', 'tokens', 'events', 'logs'): (out/name).mkdir()
    save_json(out/'PLAN.json', {'schema': 'r1-max-other-margin-v1-saved-score-plan-v1',
        'training_plan_sha256': sha(run/'PLAN.json'), 'predictions_sha256': sha(run/'PREDICTIONS.json'),
        'preregistered_plan_sha256': plan['preregistered_plan_sha256'],
        'script_sha256': scoring_source_sha, 'retained_fit_script_sha256': plan['script_sha256'],
        'scoring_source_differs_from_fit': scoring_source_sha != plan['script_sha256'], 'detector_binary_sha256': DETECTOR_SHA,
        'maximum_detector_replays': 130, 'new_head_or_backbone_forwards': 0, 'retries': 0,
        'development_gate': CONFIG['development_gate'], 'train_gate': CONFIG['train_gate'],
        'tie_rule': CONFIG['candidate_tie_rule'], 'calibration_or_threshold_changes': False})
    terminal = {'schema': 'r1-max-other-margin-v1-saved-score-terminal-v1', 'status': 'RUNNING', 'completed': []}
    ids = np.array(PRECISE_IDS, dtype=np.uint32); errors = {'train': [], 'development': []}
    try:
        with (out/'scores.jsonl').open('x', encoding='utf-8') as log:
            for row in selected:
                index = row['index']; array = row['array']; other = cached[index]
                max_precise = np.max(array['exact_logits'], axis=1); precise = ids[np.argmax(array['exact_logits'], axis=1)]
                candidate = np.where(max_precise >= other, precise, 1).astype(np.uint32)
                reference = np.where(np.isin(array['original_best_token'], ids), array['original_best_token'], 1)
                token_path = out/'tokens'/f'{index:03d}.txt'
                with token_path.open('x', encoding='ascii') as stream:
                    for frame in range(len(array)):
                        stream.write(f"F,{frame},{int(array['source'][frame])},{int(array['delivery'][frame])},{int(candidate[frame])}\n")
                    stream.flush(); os.fsync(stream.fileno())
                argv = [str(DETECTOR), 'original', str(token_path)]
                save_json(out/'claims'/f'{index:03d}.json', {'index': index, 'argv': argv, 'attempt': 1,
                    'saved_prediction_sha256': sha(run/f'predictions/{index:03d}.npy'), 'new_neural_forwards': 0})
                events = out/'events'/f'{index:03d}.txt'; error_log = out/'logs'/f'{index:03d}.stderr'
                with events.open('xb') as so, error_log.open('xb') as se:
                    process = subprocess.run(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, timeout=2, check=False)
                need(process.returncode == 0 and error_log.stat().st_size == 0 and events.stat().st_size <= 65536, 'C detector scoring failure')
                actual = events.read_bytes(); actual_lines = actual.splitlines(); expected_lines = row['expected'].splitlines()
                new_events = list((Counter(actual_lines)-Counter(expected_lines)).elements())
                missing_events = list((Counter(expected_lines)-Counter(actual_lines)).elements())
                words = [int(line.split(b',')[2]) for line in actual_lines]
                wrong = sum(row['references'][word] == 0 for word in words)
                baseline_wrong = sum(row['references'][int(line.split(b',')[2])] == 0 for line in expected_lines)
                added_wrong = sum(row['references'][int(line.split(b',')[2])] == 0 for line in new_events)
                err = np.abs(other-array['max_other']); errors[row['split']].append(err)
                transitions = Counter(f'{int(before)}->{int(after)}' for before, after in zip(reference, candidate) if before != after)
                scored = {'index': index, 'split': row['split'], 'frames': len(array),
                    'events_exact': actual == row['expected'], 'new_events': len(new_events), 'missing_events': len(missing_events),
                    'wrong_keyword_events_machine_weak': wrong, 'baseline_wrong_keyword_events_machine_weak': baseline_wrong,
                    'added_wrong_keyword_events_machine_weak': added_wrong, 'actual_events': len(words), 'reference_events': len(expected_lines),
                    'frame_category_changes': int(np.count_nonzero(candidate != reference)), 'changed_categories': dict(transitions),
                    'candidate_precise_other_ties': int(np.count_nonzero(max_precise == other)),
                    'teacher_precise_other_ties': int(np.count_nonzero(max_precise == array['max_other'])),
                    'max_other_error': metric(err), 'events_sha256': sha(events),
                    'weak_keywords': {str(k): {'reference': row['references'][k], 'proposal_present': k in words} for k in (1,2)}}
                log.write(json.dumps(scored, sort_keys=True, allow_nan=False)+'\n'); log.flush(); os.fsync(log.fileno())
                terminal['completed'].append(scored)
        need(len(terminal['completed']) == 130, 'score denominator')
        need(sha(run/'PREDICTIONS.json') == load_json(out/'PLAN.json')['predictions_sha256'] and
             all(sha(run/r['path']) == r['sha256'] for r in predictions['records']) and
             sha(Path(__file__)) == scoring_source_sha, 'post-score predictions/source changed')
        train_rows = [r for r in terminal['completed'] if r['split'] == 'train']
        dev_rows = [r for r in terminal['completed'] if r['split'] == 'development']
        dev_pass = len(dev_rows) == 34 and all(r['events_exact'] for r in dev_rows)
        train_pass = len(train_rows) == 96 and all(r['new_events'] == 0 and r['added_wrong_keyword_events_machine_weak'] == 0 for r in train_rows)
        terminal['development_exact_retention'] = dev_pass; terminal['train_no_new_wrong_events'] = train_pass
        terminal['status'] = 'PASS_EXPOSED_DEVELOPMENT_SCREEN_NOT_PROMOTED' if dev_pass and train_pass else 'REJECTED_FIXED_RESEARCH_SCREEN'
    except BaseException as exc:
        terminal['status'] = 'STOPPED_SCORING_RETAINED_PREDICTIONS'; terminal['failure'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        save_json(out/'TERMINAL.json', terminal)
    groups = {}
    for split in ('train', 'development'):
        subset = [r for r in terminal['completed'] if r['split'] == split]
        groups[split] = {'records': len(subset), 'frames': sum(r['frames'] for r in subset),
            'event_exact_records': sum(r['events_exact'] for r in subset),
            'new_events': sum(r['new_events'] for r in subset), 'missing_events': sum(r['missing_events'] for r in subset),
            'wrong_keyword_events_machine_weak': sum(r['wrong_keyword_events_machine_weak'] for r in subset),
            'baseline_wrong_keyword_events_machine_weak': sum(r['baseline_wrong_keyword_events_machine_weak'] for r in subset),
            'added_wrong_keyword_events_machine_weak': sum(r['added_wrong_keyword_events_machine_weak'] for r in subset),
            'frame_category_changes': sum(r['frame_category_changes'] for r in subset),
            'candidate_precise_other_ties': sum(r['candidate_precise_other_ties'] for r in subset),
            'teacher_precise_other_ties': sum(r['teacher_precise_other_ties'] for r in subset),
            'max_other_error': metric(np.concatenate(errors[split])),
            'weak_keywords': {str(k): {'positive_records': sum(r['weak_keywords'][str(k)]['reference'] == 1 for r in subset),
                'hit_positive_records': sum(r['weak_keywords'][str(k)]['reference'] == 1 and r['weak_keywords'][str(k)]['proposal_present'] for r in subset),
                'negative_records': sum(r['weak_keywords'][str(k)]['reference'] == 0 for r in subset),
                'false_positive_records': sum(r['weak_keywords'][str(k)]['reference'] == 0 and r['weak_keywords'][str(k)]['proposal_present'] for r in subset)} for k in (1,2)}}
    aggregate = {'schema': 'r1-max-other-margin-v1-diagnostic-aggregate-v1', 'status': terminal['status'],
        'checkpoint_sha256': predictions['checkpoint']['sha256'], 'fit_plan_sha256': sha(run/'PLAN.json'),
        'preregistered_plan_sha256': plan['preregistered_plan_sha256'], 'joint_change': CONFIG['joint_change'],
        'predictions_sha256': sha(run/'PREDICTIONS.json'), 'score_terminal_sha256': sha(out/'TERMINAL.json'),
        'runtime': plan['runtime'], 'optimizer_updates': 2000, 'sampled_train_frames': 512000,
        'final_head_recordings': 130, 'scoring_detector_only_replays': 130,
        'teacher_or_backbone_forwards_in_fit_and_score': 0, 'unknown_scored_or_fitted': 0,
        'hidden140': 'unchanged cached teacher embeddings; no learned encoder or new hidden computation',
        'groups': groups, 'development_exact_retention': terminal['development_exact_retention'],
        'train_no_new_wrong_events': terminal['train_no_new_wrong_events'],
        'model_promotion_authorized': False, 'default_r1_changed': False,
        'historical_compact_affine_sum_other': 'FAIL_FIDELITY_UNCHANGED',
        'prior_uniform_clip_mlp64': 'REJECTED_FIXED_RESEARCH_SCREEN_UNCHANGED',
        'limitations': ['Exposed synthetic machine-weak development; no independent FAR/FRR.',
            'Max-other scalar is not full 2601-class probability or CTC equivalence.',
            'No standalone C export, target-board speed, calibration or promotion established.',
            'Isolated NumPy diagnostic is not canonical Torch retraining.']}
    save_json(out/'AGGREGATE.json', aggregate); print(json.dumps(aggregate, sort_keys=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('fit')
    training.add_argument('--extract', type=Path, required=True); training.add_argument('--output', type=Path, required=True)
    training.add_argument('--plan-sha256', required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--run', type=Path, required=True); scoring.add_argument('--output', type=Path, required=True)
    sanity = commands.add_parser('gradient-check')
    sanity.add_argument('--output', type=Path, required=True)
    sanity.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'fit': fit(args.extract.resolve(), args.output.resolve(), args.plan_sha256)
    elif args.command == 'score': score(args.run.resolve(), args.output.resolve())
    else:
        need(not args.output.exists(), 'fresh gradient-check output required')
        preregistration(args.plan_sha256)
        runtime = configure_runtime(); result = gradient_check()
        result.update({'runtime': runtime, 'script_sha256': sha(Path(__file__)), 'preregistered_plan_sha256': args.plan_sha256})
        os.umask(0o077); save_json(args.output, result); print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
