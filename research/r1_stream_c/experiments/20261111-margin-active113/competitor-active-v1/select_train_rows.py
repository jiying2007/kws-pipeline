#!/usr/bin/env python3
"""Freeze the uncapped union of cached training max-OTHER winner IDs.

This reads only eligible training .r1h caches. It does not evaluate a head,
read PCM, train, or inspect development/unknown hidden-cache contents.
"""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

ROOT = Path('/workspace/scratch/fec7ccd5bb4b')
HERE = ROOT / 'competitor-active-v1'
INPUTS = Path('/workspace/scratch/1e1fe64498b2/r1-local/evidence-original/KWS_FSMN_R1_EVIDENCE_20261110/inputs')
EXTRACTION = ROOT / 'competitor-model/extraction'
MODEL = Path('/workspace/scratch/1e1fe64498b2/r1-local/integration-original/model/weights/model-int8.r1hq8')
PRECISE = (0, 1027, 1401, 1462, 2093)
PINS = {
    'input_manifest': '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab',
    'labels': '8d0b622ebd50570239e45f3092e97e8ea4fa646f051dedddec1291a8225ac7b2',
    'extraction_result': '316befa7a528315edebe107dfe1371aacc144d997271cf7b1b64c7986baef448',
    'teacher_model': '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4',
    'par4_core': 'f400d8691d4aefdb72e73f020e2a16bf9a729e109f4efdfbe2a444506a61ef91',
}

def need(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'regular non-symlink file required')
    return path.read_bytes()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def pinned(path, expected):
    data = read(path)
    need(sha(data) == expected, 'pinned input changed: ' + str(path))
    return data

def main():
    os.umask(0o077)
    output = HERE / 'SELECTION.json'
    need(not output.exists() and not output.is_symlink(), 'fresh selection required; no retry/overwrite')
    manifest = json.loads(pinned(INPUTS / 'INPUT_MANIFEST.json', PINS['input_manifest']))
    label_bytes = pinned(INPUTS / 'original-observations.jsonl', PINS['labels'])
    labels = {}
    for line in label_bytes.splitlines():
        row = json.loads(line)
        need(row['source_id'] not in labels, 'duplicate source label')
        labels[row['source_id']] = row
    result = json.loads(pinned(EXTRACTION / 'RESULT.json', PINS['extraction_result']))
    audit_path = ROOT / 'competitor-model/EXTRACTION_AUDIT.json'
    audit_bytes = read(audit_path)
    audit = json.loads(audit_bytes)
    need(result['status'] == 'PASS_FIXED_TEACHER_EXTRACTION' and
         audit['status'] == 'PASS_RETAINED_EXTRACTION_AUDIT' and
         audit['result_sha256'] == PINS['extraction_result'], 'extraction not audited')
    records = {row['index']: row for row in result['records']}
    need(len(records) == len(result['records']) == 134, 'extraction inventory changed')
    need(len(manifest['records']) == len(labels) == 134, 'input/label inventory changed')
    eligible = []
    for row in manifest['records']:
        label = labels[row['source_id']]
        need(row['split'] == label['split'] and row['pcm_sha256'] == label['pcm_sha256'],
             'source label join changed')
        if row['split'] == 'train' and label['ctc_weight'] in (0.5, 1.0):
            need(label['evaluation_status'] == 'evaluated' and
                 all(k['known'] is True for k in label['keywords']), 'unknown training labels')
            eligible.append((row, label))
    eligible.sort(key=lambda pair: pair[0]['index'])
    need([row['index'] for row, _ in eligible] == list(range(96)), 'eligible training set changed')

    expected_header = struct.pack('<8s6I5IIQ', b'R1HD140\0', 1, 64, 1200,
                                  140, 5, 2601, *PRECISE, 1, 0)
    counts = collections.Counter()
    train_sources = []
    for row, label in eligible:
        index = row['index']
        retained = records[index]
        need(retained['path'] == f'hidden/{index:03d}.r1h' and
             retained['original_split'] == 'train' and
             retained['fit_excluded_unknown'] is False and retained['fe_trace_equal'] is True and
             retained['pcm_sha256'] == row['pcm_sha256'], 'training cache metadata mismatch')
        path = EXTRACTION / retained['path']
        data = pinned(path, retained['sha256'])
        need(len(data) == retained['bytes'] == 64 + retained['frames'] * 1200 and
             data[:64] == expected_header, 'training cache shape/header mismatch')
        local = collections.Counter()
        for frame in range(retained['frames']):
            offset = 64 + frame * 1200
            need(struct.unpack_from('<Q', data, offset)[0] == frame, 'cached frame index changed')
            other = struct.unpack_from('<I', data, offset + 28)[0]
            need(other < 2601 and other not in PRECISE, 'invalid cached OTHER winner')
            local[other] += 1
        counts.update(local)
        train_sources.append({
            'index': index, 'source_id': row['source_id'], 'split': 'train',
            'ctc_weight': label['ctc_weight'], 'cache_path': str(path),
            'bytes': len(data), 'sha256': retained['sha256'], 'frames': retained['frames'],
            'pcm_sha256': row['pcm_sha256'], 'unique_other_winners': len(local),
        })
    need(len(train_sources) == 96 and sum(counts.values()) == 9079, 'training denominator changed')
    ids = sorted(counts)
    need(ids and len(ids) <= 2596, 'invalid uncapped union')
    ids_le = b''.join(struct.pack('<I', value) for value in ids)

    # Byte identities only: no matrix/vector arithmetic or model evaluation.
    model = pinned(MODEL, PINS['teacher_model'])
    need(len(model) == 1960140 and model[:8] == b'R1HQ8V1\0', 'model layout changed')
    quant = 88 + 390474 * 4
    scale = quant + 2601 * 140 + 4
    bias = scale + 2601 * 8
    precise = bias + 2601 * 4
    need(precise + 5 * 140 * 4 == len(model), 'model offsets changed')
    selected_rows = []
    for token in ids:
        q = model[quant + token*140:quant + (token+1)*140]
        s = model[scale + token*8:scale + (token+1)*8]
        b = model[bias + token*4:bias + (token+1)*4]
        selected_rows.append({'id': token, 'quantized_row_bytes': 140,
            'quantized_row_sha256': sha(q), 'scale_f64le_sha256': sha(s),
            'bias_f32le_sha256': sha(b), 'row_scale_bias_sha256': sha(q+s+b)})
    precise_rows = []
    for slot, token in enumerate(PRECISE):
        w = model[precise + slot*560:precise + (slot+1)*560]
        b = model[bias + token*4:bias + (token+1)*4]
        precise_rows.append({'id': token, 'weights_f32le_sha256': sha(w),
                             'bias_f32le_sha256': sha(b)})
    core_path = ROOT / 'matrix-build/sdk/par4/core/r1dfsmn.c'
    pinned(core_path, PINS['par4_core'])
    n = len(ids)
    receipt = {
        'schema': 'r1-train-active-other-selection-v1',
        'candidate': 'R1-TRAIN-ACTIVE-OTHER-V1',
        'status': 'FROZEN_TRAIN_ONLY_UNION_NOT_SCORED',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_rule': 'Sorted union of other_best_token over every frame of the 96 known eligible training clips; no top-k, cap, tuning, or development selection.',
        'eligibility': 'INPUT_MANIFEST source_id join original observations: split=train, ctc_weight in {0.5,1.0}, evaluated and known labels.',
        'selection_source_sha256': sha(read(Path(__file__))),
        'input_manifest_sha256': PINS['input_manifest'],
        'observation_labels_sha256': PINS['labels'],
        'extraction_result_sha256': PINS['extraction_result'],
        'extraction_audit_sha256': sha(audit_bytes),
        'teacher_model_sha256': PINS['teacher_model'],
        'frontend_sha256': result['frontend_sha256'],
        'saved_trace_manifest_sha256': result['saved_trace_manifest_sha256'],
        'tested_source_head': result['source_head'],
        'canonical_source_commit': '8d72fe408ba6801c2e98cbcbbbdf6ee55a26ddda',
        'par4_core_sha256': PINS['par4_core'],
        'training_clips': 96, 'training_frames': 9079,
        'training_sources': train_sources,
        'precise_ids': list(PRECISE), 'selected_other_ids': ids,
        'selected_other_count': n, 'selected_output_count': n + 5,
        'selected_other_ids_u32le_sha256': sha(ids_le),
        'training_other_winner_frame_counts': {str(token): counts[token] for token in ids},
        'selected_original_quantized_rows': selected_rows,
        'preserved_precise_rows': precise_rows,
        'original_row_values_exported': False,
        'row_arithmetic_contract': 'For each selected original OTHER row: four double sums over i=0,4,...,136, then (s0+s1)+(s2+s3), multiply original float64 scale, add original float32 bias cast to double. Precise rows use original scalar i=0..139 double accumulation and original float32 bias.',
        'argmax_contract': 'Compare all selected OTHER and five precise scores using their original IDs; maximum score wins and the lowest original ID wins ties. Preserve the original winning ID for the unchanged detector.',
        'training_coverage': 'Every cached full-OTHER winning ID is included by construction. Equality of max-OTHER and full selected argmax follows only when original row arithmetic and tie order are preserved; C implementation verification is still required.',
        'development_coverage': 'NOT_INSPECTED_OR_SCORED',
        'unknown_coverage': 'NOT_INSPECTED_OR_SCORED',
        'cost_estimate': {
            'baseline_head_dot_product_multiplications_per_frame': 2601*140,
            'candidate_head_dot_product_multiplications_per_frame': (n+5)*140,
            'unchanged_backbone_dense_multiplications_per_frame': 382000,
            'baseline_head_payload_bytes_excluding_original_header_and_backbone': 398156,
            'candidate_original_row_parameter_bytes': n*152 + 5*564,
            'candidate_id_map_u32_bytes': (n+5)*4,
            'candidate_head_payload_with_id_map_before_format_header_padding': n*152+5*564+(n+5)*4,
            'frozen_backbone_prefix_bytes': 390474*4,
            'compact_model_payload_with_ids_before_new_header_padding': 390474*4+n*152+5*564+(n+5)*4,
            'baseline_model_file_bytes': 1960140,
            'compact_logit_buffer_bytes': (n+5)*8,
            'baseline_logit_buffer_bytes': 2601*8,
            'not_a_measured_speed_or_memory_result': True,
        },
        'development_hidden_files_read': 0, 'unknown_hidden_files_read': 0,
        'hidden_values_or_logits_decoded_for_selection': 0,
        'pcm_reads': 0, 'head_forward_calls': 0, 'backbone_forward_calls': 0,
        'detector_replays': 0, 'training_updates': 0,
        'default_r1_changed': False, 'model_promotion_authorized': False,
        'prior_mlp_candidate_status': 'REJECTED_FIXED_RESEARCH_SCREEN_UNCHANGED',
        'original_fp32_numeric_gate': 'FAIL_UNCHANGED',
        'next_execution': 'Requires a separately locked, bounded cached-head scoring plan. No candidate scoring is authorized by this selection receipt.',
    }
    # Identity recheck before the single exclusive write; no development caches opened.
    for row in train_sources:
        need(sha(read(Path(row['cache_path']))) == row['sha256'], 'training cache drift during selection')
    pinned(INPUTS / 'INPUT_MANIFEST.json', PINS['input_manifest'])
    pinned(INPUTS / 'original-observations.jsonl', PINS['labels'])
    pinned(EXTRACTION / 'RESULT.json', PINS['extraction_result'])
    pinned(MODEL, PINS['teacher_model'])
    need(sha(read(audit_path)) == sha(audit_bytes), 'audit changed during selection')
    data = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    with output.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': receipt['status'], 'selected_other_count': n,
        'selected_output_count': n+5, 'training_frames': 9079,
        'selection_receipt_bytes': len(data), 'selection_receipt_sha256': sha(data),
        'cost_estimate': receipt['cost_estimate'], 'head_or_backbone_calls': 0,
        'development_hidden_files_read': 0}, sort_keys=True))

if __name__ == '__main__':
    main()
