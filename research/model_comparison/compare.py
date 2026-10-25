"""Read-only, fail-closed development comparison; no model/audio execution.

Receipts must describe an independently frozen selection set. This validates
consistency of supplied evidence, not truth of measurements or qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

WORDS = ('你好小窝', '小窝小窝')
SPLITS = ('train', 'calibration', 'selection')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            name + ' must be finite and positive')
    return value

def count(value, name):
    require(type(value) is int and value >= 0, name + ' must be a nonnegative integer')
    return value

UNKNOWN_IDENTITIES = frozenset(('', 'unknown', 'none', 'null', 'pending', 'n/a', 'na',
                                'unset', 'unspecified', 'missing', 'tbd', '-', '?'))
SOURCE_ID_FIELDS = ('source_id', 'source_family', 'source_family_id', 'family_id',
                    'derivation_family', 'derivation_family_id')
PCM_ID_FIELDS = ('pcm_sha256', 'source_pcm', 'source_pcm_sha256', 'original_pcm_sha256', 'reference_pcm_sha256')
AUDIO_ID_FIELDS = ('file_sha256', 'source_wav_sha256', 'source_audio_sha256',
                   'original_audio_sha256', 'reference_audio_sha256')
LINEAGE_FIELDS = SOURCE_ID_FIELDS + PCM_ID_FIELDS + AUDIO_ID_FIELDS + ('preset_voice_id',)

def identity(value, key):
    require(isinstance(value, str) and value.strip().casefold() not in UNKNOWN_IDENTITIES,
            'unknown or missing identity: ' + key)
    value = value.strip()
    normalized = value.casefold()
    require(not any(normalized.startswith(word + separator) for word in UNKNOWN_IDENTITIES if len(word) > 1
                    for separator in ('_', '-', ':', ' ')), 'placeholder identity: ' + key)
    if key in PCM_ID_FIELDS or key in AUDIO_ID_FIELDS:
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid PCM digest')
    return value

def validate_split(rows):
    require(isinstance(rows, list) and 0 < len(rows) <= 30000, 'empty or oversized split inventory')
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen, present = {}, set()
    for i, row in enumerate(rows):
        split = row['split']
        require(split in SPLITS, 'protected/final evidence cannot be comparison input')
        present.add(split)
        links = row.get('lineage_links', [])
        require(isinstance(links, list) and all(isinstance(x, dict) for x in links), 'invalid lineage links')
        for key in ('speaker_id', 'session_id', 'source_id', 'pcm_sha256'):
            require(key in row, 'missing ' + key)
        for record in [row, *links]:
            for key in ('speaker_id', 'session_id', 'pcm_sha256', *LINEAGE_FIELDS):
                if key not in record:
                    continue
                value = identity(record[key], key)
                # Original PCM and rendered PCM share a namespace. Family aliases
                # also share a namespace, connecting multi-step augmentations.
                # Match effect_dataset_split: preset voices remain a shared
                # leakage link even when engines assign different speaker IDs.
                namespace = ('preset-voice' if key == 'preset_voice_id' else
                             'pcm' if key in PCM_ID_FIELDS else
                             'audio-file' if key in AUDIO_ID_FIELDS else
                             'source' if key in SOURCE_ID_FIELDS else key)
                node = (namespace, value)
                if node in seen:
                    parent[find(i)] = find(seen[node])
                else:
                    seen[node] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), set()).add(row['split'])
    require(all(len(splits) == 1 for splits in components.values()), 'cross-split leakage: connected source component')
    require(present == set(SPLITS), 'all three splits required')
    return digest(sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)))

def compare(plan, receipts):
    require(plan['development_only'] is True and plan['protected_evidence_used'] is False,
            'development-only independent evidence required')
    split_hash = validate_split(plan['split_inventory'])
    gate = positive(plan['max_far_per_hour'], 'FAR gate')
    budget = plan['budget']
    for key in ('max_training_seconds', 'max_peak_rss_bytes', 'max_model_bytes', 'max_macs_per_frame'):
        positive(budget[key], key)
    require(type(budget['sample_exposures']) is int and budget['sample_exposures'] > 0, 'invalid exposure budget')
    require(type(plan['seed']) is int, 'invalid seed')
    require(len(receipts) == 2 and {r['arm'] for r in receipts} == {'control', 'treatment'}, 'two unique paired arms required')
    common = ('source_sha', 'runner_id', 'frontend_sha256', 'tokens_sha256',
              'optimizer_sha256', 'decoder_sha256', 'selection_stream_sha256',
              'calibration_stream_sha256')
    for key in common:
        require(isinstance(receipts[0][key], str) and bool(receipts[0][key]), 'missing ' + key)
        require(receipts[0][key] == receipts[1][key], 'unmatched ' + key)
    # Matching arms are not enough: calibration must not be reused as selection.
    # Canonical digests prevent placeholders or alternate spellings from hiding
    # an explicit same-stream contradiction. Different hashes alone do not prove
    # independent sources; validate_split still checks the supplied lineage.
    for key in ('calibration_stream_sha256', 'selection_stream_sha256'):
        value = receipts[0][key]
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'invalid stream SHA-256: ' + key)
    require(receipts[0]['calibration_stream_sha256'] != receipts[0]['selection_stream_sha256'],
            'calibration and selection streams must differ')
    results = {}
    for r in receipts:
        require(r['status'] == 'MEASURED', 'NOT_RUN or synthetic fixture is not measurement')
        require(r['split_sha256'] == split_hash, 'split identity mismatch')
        require(r['seed'] == plan['seed'], 'seed mismatch')
        require(r['sample_exposures'] == budget['sample_exposures'], 'unequal sample exposure')
        require(r['threshold_source'] == 'calibration-only' and r['selection_used_for_tuning'] is False,
                'thresholds must freeze before selection')
        require(isinstance(r['thresholds_sha256'], str) and len(r['thresholds_sha256']) == 64,
                'missing frozen threshold digest')
        hours = positive(r['negative_hours'], 'continuous negative hours')
        fa = count(r['false_accept_events'], 'FA events')
        require(hours == receipts[0]['negative_hours'], 'unequal negative duration')
        far = fa / hours
        resource_ok = True
        for field, limit in (('training_seconds', 'max_training_seconds'), ('peak_rss_bytes', 'max_peak_rss_bytes'),
                             ('model_bytes', 'max_model_bytes'), ('macs_per_frame', 'max_macs_per_frame')):
            resource_ok &= positive(r[field], field) <= budget[limit]
        require(set(r['per_keyword']) == set(WORDS), 'both wake words required')
        frr = {}
        for word in WORDS:
            n = count(r['per_keyword'][word]['positives'], 'positive events')
            miss = count(r['per_keyword'][word]['false_rejects'], 'FR events')
            require(n > 0 and miss <= n, 'invalid per-word denominator')
            require(n == receipts[0]['per_keyword'][word]['positives'], 'unmatched positive denominator')
            frr[word] = miss / n
        results[r['arm']] = {'far_per_hour': far, 'frr': frr,
                             'far_gate_pass': far <= gate, 'resource_gate_pass': bool(resource_ok)}
    c, t = results['control'], results['treatment']
    eligible = all(r['far_gate_pass'] and r['resource_gate_pass'] for r in (c, t))
    dominates = eligible and all(t['frr'][w] <= c['frr'][w] for w in WORDS) and any(t['frr'][w] < c['frr'][w] for w in WORDS)
    return {'schema_version': 1, 'scope': 'paired-development-only', 'shipping_approved': False,
            'split_sha256': split_hash, 'results': results,
            'treatment_per_word_dominates_at_shared_gate': dominates,
            'limitations': ['Observed FAR gate only; no population FAR confidence claim.',
                            'Receipt consistency is not independent evidence authentication.',
                            'Hosted resource measurements do not qualify SSC305.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.plan.read_text()),
                            [json.loads(p.read_text()) for p in (args.control, args.treatment)]),
                     ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
