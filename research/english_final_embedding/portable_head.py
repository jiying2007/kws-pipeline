"""Export and replay one fixed research head; never supplies labels or admission.

This module performs no model download or automatic execution. Its future caller
must authenticate the saved NPZ and embedding provenance before invoking it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

SHAPES = {'mean96': (96,), 'scale96': (96,), 'w1': (32, 1536),
          'b1': (32,), 'w2': (3, 32), 'b2': (3,)}
FLOATS = 49475
MAGIC = b'EHKWS01\0'
STREAM_MAGIC = b'EHSTR01\0'
DIRECTORY = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate(weights):
    import numpy as np
    need(set(weights) == set(SHAPES), 'EXACT_HEAD_WEIGHT_KEYS')
    for key, shape in SHAPES.items():
        value = weights[key]
        need(isinstance(value, np.ndarray) and value.dtype == np.float32 and
             value.shape == shape and bool(np.isfinite(value).all()), 'HEAD_WEIGHT_ARRAY:' + key)
    need(bool((weights['scale96'] > 0).all()), 'POSITIVE_NORMALIZATION_SCALE')
    return weights


def load_weights(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        need(len(archive.files) == len(set(archive.files)) == len(SHAPES), 'HEAD_ARCHIVE_KEYS')
        weights = {key: archive[key].copy() for key in archive.files}
    return validate(weights)


def export_blob(weights):
    validate(weights)
    body = b''.join(weights[key].astype('<f4', copy=False).tobytes(order='C') for key in SHAPES)
    need(len(body) == 4 * FLOATS, 'HEAD_BINARY_GEOMETRY')
    return MAGIC + body


def stream_blob(embeddings, samples, discontinuities=None):
    import numpy as np
    need(isinstance(embeddings, np.ndarray) and embeddings.dtype == np.float32 and
         embeddings.ndim == 2 and embeddings.shape[1] == 96 and
         1 <= len(embeddings) <= 1000000 and bool(np.isfinite(embeddings).all()), 'EMBEDDING_STREAM')
    need(len(samples) == len(embeddings) and all(isinstance(int(x), int) and
         int(x) == x and 0 <= int(x) <= 2 ** 64 - 1 for x in samples), 'ABSOLUTE_SAMPLE_CLOCK')
    if discontinuities is None:
        discontinuities = [False] * len(samples)
    need(len(discontinuities) == len(samples) and
         all(type(x) is bool for x in discontinuities), 'DISCONTINUITY_MARKERS')
    return STREAM_MAGIC + struct.pack('<I', len(samples)) + b''.join(
        struct.pack('<QI', int(sample), int(gap)) + row.astype('<f4', copy=False).tobytes()
        for row, sample, gap in zip(embeddings, samples, discontinuities))


def reference(weights, embeddings, samples, discontinuities=None):
    """NumPy saved-weight inference and fixed uncalibrated proposal policy."""
    import numpy as np
    validate(weights)
    stream_blob(embeddings, samples, discontinuities)
    gaps = discontinuities or [False] * len(samples)
    history, outputs = [], []
    candidate = confirmations = blocked_until = last_sample = 0
    for row, sample, gap in zip(embeddings, samples, gaps):
        sample = int(sample)
        if gap:
            history = []
            candidate = confirmations = blocked_until = 0
        need(sample >= 16000 and sample > last_sample and sample <= 2 ** 64 - 1 - 13 * 1280 and
             (not history or sample - last_sample == 1280), 'STREAM_CLOCK_CONTINUITY')
        normalized = ((row - weights['mean96']) / weights['scale96']).astype(np.float32)
        need(bool(np.isfinite(normalized).all()), 'FINITE_NORMALIZED_EMBEDDING')
        history.append(normalized)
        history = history[-16:]
        last_sample = sample
        result = {'sample': sample, 'ready': len(history) == 16, 'keyword': 0,
                  'logits': [0.0] * 3, 'probabilities': [0.0] * 3}
        if len(history) == 16:
            flattened = np.stack(history).reshape(1536)
            hidden = np.maximum(weights['w1'] @ flattened + weights['b1'], 0)
            logits = weights['w2'] @ hidden + weights['b2']
            need(bool(np.isfinite(logits).all()), 'FINITE_HEAD_LOGITS')
            exp = np.exp((logits - logits.max()).astype(np.float64))
            probabilities = (exp / exp.sum()).astype(np.float32)
            selected = int(probabilities.argmax())
            if selected == 0 or probabilities[selected] < np.float32(0.80) or sample < blocked_until:
                candidate = confirmations = 0
            else:
                confirmations = confirmations + 1 if selected == candidate else 1
                candidate = selected
                if confirmations >= 3:
                    result['keyword'] = selected
                    blocked_until = sample + 13 * 1280
                    candidate = confirmations = 0
            result.update(logits=logits.tolist(), probabilities=probabilities.tolist())
        outputs.append(result)
    return outputs


def compile_replay(output, compiler='cc', extra_flags=()):
    """Build only; callers decide whether host or ARM execution is authorized."""
    subprocess.run([compiler, '-std=c99', '-O2', '-ffp-contract=off', '-fno-fast-math',
                    '-Wall', '-Wextra', '-Werror', *extra_flags,
                    str(DIRECTORY / 'english_causal_head.c'), str(DIRECTORY / 'head_replay.c'),
                    '-lm', '-o', str(output)], check=True)


def check_parity(weights, embeddings, samples, work, runner, discontinuities=None):
    """runner is an explicit [host-executable] or [qemu-arm, ..., executable]."""
    import numpy as np
    work = Path(work)
    work.mkdir(exist_ok=False)
    weight_bytes = export_blob(weights)
    stream_bytes = stream_blob(embeddings, samples, discontinuities)
    (work / 'head.bin').write_bytes(weight_bytes)
    (work / 'stream.bin').write_bytes(stream_bytes)
    expected = reference(weights, embeddings, samples, discontinuities)
    result = subprocess.run([*runner, str(work / 'head.bin'), str(work / 'stream.bin')],
                            check=True, text=True, capture_output=True)
    lines = result.stdout.splitlines()
    need(len(lines) == len(expected), 'C_REPLAY_ROW_COUNT')
    maximum_logit_error = maximum_probability_error = 0.0
    events = []
    for line, row in zip(lines, expected):
        parts = line.split()
        need(len(parts) == 9, 'C_REPLAY_ROW_SHAPE')
        need((int(parts[0]), bool(int(parts[1])), int(parts[2])) ==
             (row['sample'], row['ready'], row['keyword']), 'C_CLOCK_READY_EVENT_PARITY')
        values = np.asarray([float(x) for x in parts[3:]], dtype=np.float64)
        need(bool(np.isfinite(values).all()), 'C_REPLAY_FINITE')
        maximum_logit_error = max(maximum_logit_error, float(np.abs(values[:3] - row['logits']).max()))
        maximum_probability_error = max(maximum_probability_error,
                                        float(np.abs(values[3:] - row['probabilities']).max()))
        if row['keyword']:
            events.append({'sample': row['sample'], 'keyword': row['keyword']})
    need(maximum_logit_error <= 1e-4 and maximum_probability_error <= 2e-6, 'C_NUMERICAL_PARITY')
    return {'schema': 'english-causal-head-c-parity-v1',
            'head_binary_sha256': hashlib.sha256(weight_bytes).hexdigest(),
            'stream_binary_sha256': hashlib.sha256(stream_bytes).hexdigest(),
            'input_embeddings': len(expected), 'head_outputs': sum(r['ready'] for r in expected),
            'max_abs_logit_error': maximum_logit_error,
            'max_abs_probability_error': maximum_probability_error, 'events': events,
            'threshold': 0.80, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
            'formal_qualification': False, 'ssc305_qualification': False}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--weights', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args(argv)
    raw = export_blob(load_weights(args.weights))
    with args.output.open('xb') as dst:
        dst.write(raw)
    print(json.dumps({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'model_promotion_allowed': False}))


if __name__ == '__main__':
    main()
