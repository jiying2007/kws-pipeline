#!/usr/bin/env python3
"""Fixed ACTIVE113 byte format. Standard-library only; no model execution."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path('/workspace/scratch/fec7ccd5bb4b')
BASE = ROOT/'matrix-build/sdk/par4'
SELECTION = ROOT/'competitor-active-v1/SELECTION.json'
SELECTION_SHA = '8289f2ca46e40717747c1ed8d08152d8db2fbadcac5f4a84fdbb20d06375dc42'
MODEL = Path('/workspace/scratch/1e1fe64498b2/r1-local/integration-original/model/weights/model-int8.r1hq8')
MODEL_SHA = '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
FRONTEND = MODEL.with_name('frontend.f32le')
FRONTEND_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
BASE_FLOAT_SHA = 'e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb'
PRECISE = (0, 1027, 1401, 1462, 2093)
OTHER_COUNT = 113
HEADER_BYTES = 256
PREFIX_FLOATS = 390474
HIDDEN = 140
OFFSETS = {'ids': 256, 'trunk': 728, 'quant': 1562624, 'scales': 1578448,
           'other_bias': 1579352, 'precise_weights': 1579804, 'precise_bias': 1582604}
SIZES = {'ids': 472, 'trunk': 1561896, 'quant': 15820, 'scales': 904,
         'other_bias': 452, 'precise_weights': 2800, 'precise_bias': 20}
PACKAGE_BYTES = 1582624
OLD_QUANT = 88+PREFIX_FLOATS*4
OLD_SCALE = OLD_QUANT+2601*140+4
OLD_BIAS = OLD_SCALE+2601*8
OLD_PRECISE = OLD_BIAS+2601*4
POLICY = 'original-id-ties-actual-other-id'
HERE = Path(__file__).resolve().parent
BASE_NAMES = ('core/r1dfsmn.c','core/r1dfsmn.h','core/r1d_stream.c','core/r1d_stream.h',
    'core/r1d_pcm.c','core/r1d_pcm.h','core/reference64_fbank.c','core/reference64_fbank.h',
    'core/sha256.c','core/sha256.h','core/model_identity.h','core/vendor/fft_twiddles_f64.h',
    'include/fsmn_sdk.h','include/fsmn_detector.h','src/fsmn_sdk.c','src/fsmn_detector.c')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), 'regular file required: '+path.name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for k,v in items:
            need(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    def bad(value): raise ValueError('nonfinite JSON')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)


def save(path, value):
    import os
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def selection():
    need(sha(SELECTION) == SELECTION_SHA, 'fixed selection changed')
    result = load(SELECTION)
    ids = tuple(result['selected_other_ids'])
    need(len(ids) == OTHER_COUNT and tuple(sorted(set(ids))) == ids and
         not set(ids).intersection(PRECISE) and all(0 <= i < 2601 for i in ids), 'selected IDs')
    need(tuple(result['precise_ids']) == PRECISE and result['teacher_model_sha256'] == MODEL_SHA and
         result['selected_other_count'] == OTHER_COUNT and result['selected_output_count'] == 118,
         'selection model/shape binding')
    need([row['id'] for row in result['selected_original_quantized_rows']] == list(ids), 'selected row order')
    return result, ids


def header():
    data = bytearray(HEADER_BYTES)
    data[:8] = b'R1A113V1'
    fields = (1,HEADER_BYTES,PACKAGE_BYTES,PREFIX_FLOATS,HIDDEN,OTHER_COUNT,5,6,2601,
              OFFSETS['ids'],OFFSETS['trunk'],OFFSETS['quant'],OFFSETS['scales'],
              OFFSETS['other_bias'],OFFSETS['precise_weights'],OFFSETS['precise_bias'])
    struct.pack_into('<16I', data, 8, *fields)
    for offset, value in ((72,MODEL_SHA),(104,SELECTION_SHA),(136,FRONTEND_SHA),(168,BASE_FLOAT_SHA)):
        data[offset:offset+32] = bytes.fromhex(value)
    return bytes(data)


def base_binding():
    receipt = ROOT/'matrix-build/BUILD_RECEIPT.json'
    need(sha(receipt) == 'e3b5335d67d9db5c43878ad2c520c3493ddb52c183e6662c2a3efd689f88b477',
         'admitted base SDK build receipt changed')
    original = load(receipt)['generated_sources']['par4']; result = {}
    for name in BASE_NAMES:
        need(sha(BASE/name) == original[name]['sha256'], 'base SDK source drift: '+name)
        result[name] = original[name]['sha256']
    return result


def source_gate(expected_sha):
    need(sha(HERE/'PLAN.json') == expected_sha, 'SDK preparation plan changed')
    plan = load(HERE/'PLAN.json')
    need(plan['policy'] == POLICY and plan['selection_sha256'] == SELECTION_SHA, 'SDK plan policy')
    need(sha(ROOT/'competitor-active-v1/execution/AGGREGATE.json') == plan['cached_screen_aggregate_sha256'],
         'admitting cached screen changed')
    for name, value in plan['source_sha256'].items():
        need(sha(HERE/name) == value, 'SDK preparation source changed: '+name)
    return plan


def screen_gate(path, expected_sha):
    """Only the frozen cached-head screen can admit this separate SDK stage."""
    need(path.resolve() == (ROOT/'competitor-active-v1/execution/AGGREGATE.json').resolve(), 'screen path')
    need(sha(path) == expected_sha, 'cached screen receipt changed')
    result = load(path)
    need(result.get('schema') == 'active113-cached-head-aggregate-v1' and
         result.get('status') == 'PASS_RETAINED_EVENT_SCREEN_NOT_PROMOTED', 'passing active screen required')
    need(result.get('selection_sha256') == SELECTION_SHA and result.get('declared_plan_sha256') ==
         '86cd4dccfb35bce57739a68a8b6170b7ff52e732cf7ef36d9715eba36e5753bf', 'cached screen selection/plan binding')
    policy=result['policy']; source=result['source_identity']
    need(policy['between_precise_and_other_tie'] == policy['within_other_tie'] == 'lowest_original_id' and
         policy['detector_other_token'] == 'actual_original_winner_id' and policy['sentinel_mapping'] is False and
         policy['threshold_changes'] is False and policy['top_k_or_cap'] is False, 'cached screen policy binding')
    need(source['original_model_sha256'] == MODEL_SHA and source['frontend_sha256'] == FRONTEND_SHA and
         source['tested_source_head'] == '13661a38064247d38bca88287e48db85bf2ccd79' and
         source['par4_core_sha256'] == sha(BASE/'core/r1dfsmn.c'), 'cached screen source binding')
    for key in ('training_max_other_bit_equal','train_no_new_event','development_complete_event_retention',
                'all_final_predictions_saved_before_scoring','all_claims_completed'):
        need(result.get(key) is True, 'cached screen gate: '+key)
    need(result['selected_other_count'] == 113 and result['selected_output_count'] == 118 and
         result['groups']['train']['records'] == 96 and result['groups']['train']['frames'] == 9079 and
         result['groups']['train']['max_other_bit_equal_frames'] == 9079 and
         result['groups']['train']['original_token_id_changes'] == 0 and result['groups']['train']['new_events'] == 0 and
         result['groups']['development']['records'] == 34 and result['groups']['development']['frames'] == 3176 and
         result['groups']['development']['event_exact_records'] == 34, 'cached screen denominator/parity')
    return result


def finite_f32(data, positive=False):
    values = [x[0] for x in struct.iter_unpack('<f', data)]
    need(all(math.isfinite(x) and (not positive or x > 0) for x in values), 'invalid f32 section')


def validate_package(data, ids):
    need(len(data) == PACKAGE_BYTES and data[:HEADER_BYTES] == header(), 'compact header/size')
    need(struct.unpack_from('<118I', data, OFFSETS['ids']) == PRECISE+ids, 'compact ID mapping')
    need(data[OFFSETS['quant']+SIZES['quant']:OFFSETS['scales']] == b'\0'*4, 'compact alignment padding')
    for name in ('trunk','other_bias','precise_weights','precise_bias'):
        finite_f32(data[OFFSETS[name]:OFFSETS[name]+SIZES[name]])
    finite_f32(data[OFFSETS['trunk']+400*4:OFFSETS['trunk']+800*4], positive=True)
    need(all(x != 128 for x in data[OFFSETS['quant']:OFFSETS['quant']+SIZES['quant']]), 'forbidden -128 quant value')
    scales = struct.unpack_from('<113d', data, OFFSETS['scales'])
    need(all(math.isfinite(x) and x > 0 for x in scales), 'invalid active scale')


def pack_bytes():
    selected, ids = selection(); need(sha(MODEL) == MODEL_SHA, 'original model changed')
    old = MODEL.read_bytes()
    need(len(old) == 1960140 and old[:8] == b'R1HQ8V1\0' and
         struct.unpack_from('<4I',old,8) == (1,2601,140,5) and
         old[24:88] == BASE_FLOAT_SHA.encode() and old[OLD_SCALE-4:OLD_SCALE] == b'\0'*4,
         'original model format')
    blocks = {'ids':struct.pack('<118I',*(PRECISE+ids)),
              'trunk':old[88:OLD_QUANT], 'quant':b'', 'scales':b'', 'other_bias':b'',
              'precise_weights':old[OLD_PRECISE:OLD_PRECISE+2800], 'precise_bias':b''}
    for descriptor in selected['selected_original_quantized_rows']:
        i = descriptor['id']; q = old[OLD_QUANT+i*140:OLD_QUANT+(i+1)*140]
        scale = old[OLD_SCALE+i*8:OLD_SCALE+(i+1)*8]; bias = old[OLD_BIAS+i*4:OLD_BIAS+(i+1)*4]
        need(digest(q) == descriptor['quantized_row_sha256'] and
             digest(scale) == descriptor['scale_f64le_sha256'] and digest(bias) == descriptor['bias_f32le_sha256'] and
             digest(q+scale+bias) == descriptor['row_scale_bias_sha256'], 'selected row bytes')
        blocks['quant'] += q; blocks['scales'] += scale; blocks['other_bias'] += bias
    for k, descriptor in enumerate(selected['preserved_precise_rows']):
        i = PRECISE[k]; bias = old[OLD_BIAS+i*4:OLD_BIAS+(i+1)*4]
        row = blocks['precise_weights'][k*560:(k+1)*560]
        need(descriptor['id'] == i and digest(row) == descriptor['weights_f32le_sha256'] and
             digest(bias) == descriptor['bias_f32le_sha256'], 'precise row bytes')
        blocks['precise_bias'] += bias
    data = bytearray(PACKAGE_BYTES); data[:HEADER_BYTES] = header()
    for name, block in blocks.items():
        need(len(block) == SIZES[name], 'compact section size'); data[OFFSETS[name]:OFFSETS[name]+len(block)] = block
    data = bytes(data); validate_package(data,ids)
    return data, {name:{'offset':OFFSETS[name],'bytes':len(block),'sha256':digest(block)} for name,block in blocks.items()}
