#!/usr/bin/env python3
"""Portable M2 saved-logit analysis; no model, frontend, or stateful decoder.

Call analyze(original_raw, positive300_raw, leading_raw) with three JSONL paths.
The returned mapping contains no input paths or private acquisition metadata.
The numeric helper functions below preserve the previously reviewed arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path

"""Saved-log numeric helpers only; no runtime or audio access."""
from decimal import Decimal, localcontext
import math
import struct

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def top3_stateless(p):
    # Value-only six-element nth_element + first-two sort, source lines 47-74.
    q = list(range(6))
    lo, hi = 0, 6
    gt = lambda a, b: p[a] > p[b]
    while hi - lo > 3:
        a, b, c = lo + 1, lo + (hi - lo) // 2, hi - 1
        if gt(q[a], q[b]):
            median = b if gt(q[b], q[c]) else (c if gt(q[a], q[c]) else a)
        else:
            median = a if gt(q[a], q[c]) else (c if gt(q[b], q[c]) else b)
        q[lo], q[median] = q[median], q[lo]
        first, last, pivot = lo + 1, hi, q[lo]
        while True:
            while gt(q[first], pivot):
                first += 1
            last -= 1
            while gt(pivot, q[last]):
                last -= 1
            if first >= last:
                break
            q[first], q[last] = q[last], q[first]
            first += 1
        if first <= 2:
            lo = first
        else:
            hi = first
    for i in range(lo + 1, hi):
        v, j = q[i], i
        while j > lo and gt(v, q[j - 1]):
            q[j] = q[j - 1]
            j -= 1
        q[j] = v
    if gt(q[1], q[0]):
        q[0], q[1] = q[1], q[0]
    return q[:3]

def probabilities(logits):
    x = [f32(v) for v in logits]
    maximum = max(x)
    ex = [math.exp(v - maximum) for v in x]
    denominator = 0.0
    for v in ex:  # sequential, matching C token order
        denominator += v
    p64 = [v / denominator for v in ex]
    p32 = [f32(v) for v in p64]
    # Independent high precision arithmetic check, not another runtime replay.
    with localcontext() as ctx:
        ctx.prec = 80
        xd = [Decimal.from_float(v) for v in x]
        ed = [(v - max(xd)).exp() for v in xd]
        denominator_d = sum(ed)
        reference = [v / denominator_d for v in ed]
        reference_f32 = [f32(float(v)) for v in reference]
        error = max(abs(Decimal.from_float(a) - b) for a, b in zip(p64, reference))
    return x, p32, reference_f32, float(error)

SYMBOLS = ['<blank>', '你', '好', '小', '窝', '屋']


def analyze(original_raw, positive300_raw, leading_raw):
    """Return compact token support and numeric checks from three saved traces."""
    inputs = [('original_eof', original_raw, 0),
              ('tail300', positive300_raw, 0),
              ('leading1500_tail300', leading_raw, 150)]
    summaries, scalar_comparison, errors = [], [], []
    disagreements = row_count = 0
    for name, path, offset in inputs:
        data = Path(path).read_bytes()
        rows = []
        for callback in (json.loads(line) for line in data.splitlines()):
            if callback.get('kind') != 'callback' or callback.get('recording') != 'M2':
                continue
            centers, logits_rows = callback['centers'], callback['logits']
            if len(centers) != len(logits_rows):
                raise ValueError('Saved center/logit row count mismatch')
            for index, (center, logits) in enumerate(zip(centers, logits_rows)):
                if len(logits) != 6 or not all(math.isfinite(v) for v in logits):
                    raise ValueError('Six finite saved logits required')
                _, posterior, reference, error = probabilities(logits)
                disagreements += sum(a != b for a, b in zip(posterior, reference))
                errors.append(error)
                top = top3_stateless(posterior)
                eligible = [token for token in top if posterior[token] > 0.05]
                rows.append(dict(absolute_center=center,
                                 source_relative_center=center-offset,
                                 searched=index < callback['decoder_rows_decoded'],
                                 probabilities_f32=posterior,
                                 top3_ids=top, eligible_ids=eligible))
        if not rows:
            raise ValueError('Saved trace has no M2 callback rows')
        row_count += len(rows)
        selected = [dict(absolute_center=row['absolute_center'],
                         source_relative_center=row['source_relative_center'],
                         searched=row['searched'],
                         tokens=[dict(id=token, symbol=SYMBOLS[token],
                                      posterior=row['probabilities_f32'][token])
                                 for token in row['eligible_ids'] if token in (3, 4)])
                    for row in rows if any(token in row['eligible_ids'] for token in (3, 4))]
        support = {SYMBOLS[token]: [row['source_relative_center'] for row in rows
                                   if token in row['eligible_ids']]
                   for token in (3, 4)}
        # Ordered support is a row-local property, not beam/CTC simulation.
        last, witness = None, []
        for token in (3, 4, 3, 4):
            options = [row for row in rows if token in row['eligible_ids'] and
                       (last is None or row['source_relative_center'] > last)]
            if not options:
                witness = []
                break
            chosen = options[0]
            last = chosen['source_relative_center']
            witness.append(dict(token=SYMBOLS[token],
                                absolute_center=chosen['absolute_center'],
                                source_relative_center=last,
                                posterior=chosen['probabilities_f32'][token]))
        summaries.append(dict(condition=name,
                              raw_sha256=hashlib.sha256(data).hexdigest(),
                              M2_model_rows=len(rows),
                              M2_decoder_rows_searched=sum(row['searched'] for row in rows),
                              subtract_prefix_fbank_frames=offset,
                              eligible_source_relative_centers=support,
                              eligible_rows=selected,
                              strictly_ordered_row_support_witness=witness))
        center72 = [row for row in rows if row['source_relative_center'] == 72]
        if len(center72) != 1:
            raise ValueError('Exactly one M2 source-relative center72 required')
        row = center72[0]
        scalar_comparison.append(dict(condition=name, symbol='小', token_id=3,
                                      absolute_center=row['absolute_center'],
                                      source_relative_center=72,
                                      posterior=row['probabilities_f32'][3],
                                      top3_ids=row['top3_ids'],
                                      eligible=3 in row['eligible_ids']))
    if disagreements:
        raise ValueError('Binary32 probabilities disagree with 80-digit reference')
    return dict(
        schema='melo5-leading-M2-portable-saved-logit-token-support-v1',
        status='PASS_STATELESS_SAVED_LOGIT_COMPARISON',
        frontend_calls=0, model_calls=0, decoder_calls=0,
        method='Recast saved %.9g logits to binary32; source-order max-shift double exp and sequential sum; round probabilities to binary32; source-equivalent strict-comparator top3; strict >0.05 cutoff unchanged.',
        numeric_check=dict(decimal_reference_digits=80,
                           binary32_reference_disagreements=disagreements,
                           max_binary64_absolute_difference=max(errors),
                           posterior_values_checked=row_count*6),
        conditions=summaries,
        center72_small_token_comparison=scalar_comparison,
        interpretation=[
            'Eligible row-local token support is a descriptive property of saved logits, not a new stateful decoder run or an explanation of a beam path.',
            'The existing saved event is the evidence for M2 K2 activation. Ordered support only characterizes the difference accompanying it.',
            'Source-relative centers subtract 150 fbank frames from the new stream; they are model/frontend coordinates, not acoustic word boundaries.',
            'Do not assign the sole earlier eligible 小 to the first or second spoken occurrence. Leading context changes frontend, model and decoder history together.',
            '80-digit agreement checks this finite sample after binary32 rounding; it does not prove universal libm bit identity.',
            'Bulk posterior rows are omitted. The three saved input traces and this module reproduce all numeric checks and compact results.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_raw', type=Path)
    parser.add_argument('positive300_raw', type=Path)
    parser.add_argument('leading_raw', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.original_raw, args.positive300_raw, args.leading_raw)
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
