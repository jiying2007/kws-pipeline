#!/usr/bin/env python3
"""Opt-in R1 strong-FSMN weight-only head INT8 packer; no model inference.

The source float32 R1 model remains immutable. Output is an owner-held research
asset, never a redistributable model license. Rowwise symmetric [-127,127].
Five blank/keyword competition rows retain exactly the original FP32 weights.
"""
import argparse
import hashlib
import math
from pathlib import Path
import struct

BASE_SHA = 'e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb'
BASE_FLOATS = 757215
PACK_SHA = '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
HEAD_START = 390474
HEAD_BIAS = 754614
HEAD_ROWS = 2601
HEAD_COLS = 140
PRECISE_IDS = (0, 1027, 1401, 1462, 2093)
HEADER = struct.Struct('<8s4I64s')
MAGIC = b'R1HQ8V1\0'
# header + prefix f32 + q8 + four-byte alignment + scales f64 + bias f32 + selected exact rows
assert len(MAGIC) == 8
assert HEAD_START + HEAD_ROWS * HEAD_COLS == HEAD_BIAS
assert HEAD_BIAS + HEAD_ROWS == BASE_FLOATS
assert HEADER.size == 88
Q_END = HEADER.size + HEAD_START * 4 + HEAD_ROWS * HEAD_COLS
PAD = (-Q_END) % 8
PACKAGE_BYTES = Q_END + PAD + HEAD_ROWS * 8 + HEAD_ROWS * 4 + len(PRECISE_IDS) * HEAD_COLS * 4
assert PACKAGE_BYTES == 1960140

def digest(data):
    return hashlib.sha256(data).hexdigest()

def pack(data: bytes) -> bytes:
    if len(data) != BASE_FLOATS * 4 or digest(data) != BASE_SHA:
        raise ValueError('base R1 payload identity mismatch')
    # Floats copied as little-endian original bytes; no numerical-source rewrites.
    values = struct.unpack('<' + str(BASE_FLOATS) + 'f', data)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('non-finite source weights')
    if not all(x > 0 for x in values[400:800]):
        raise ValueError('invalid CMVN divisor')
    q = bytearray()
    scales = []
    for o in range(HEAD_ROWS):
        w = values[HEAD_START + o * HEAD_COLS:HEAD_START + (o + 1) * HEAD_COLS]
        maximum = max(abs(float(x)) for x in w)
        scale = maximum / 127.0 if maximum else 1.0
        row = [max(-127, min(127, round(float(x) / scale))) for x in w]
        if any(abs(v) > 127 for v in row):
            raise ValueError('out-of-range quantized head')
        q.extend((v & 0xff) for v in row)
        scales.append(scale)
    output = bytearray(HEADER.pack(MAGIC, 1, HEAD_ROWS, HEAD_COLS, len(PRECISE_IDS), BASE_SHA.encode('ascii')))
    output.extend(data[:HEAD_START * 4])
    output.extend(q)
    output.extend(b'\0' * PAD)
    output.extend(struct.pack('<' + str(HEAD_ROWS) + 'd', *scales))
    output.extend(data[HEAD_BIAS * 4:])
    for o in PRECISE_IDS:
        output.extend(data[(HEAD_START + o * HEAD_COLS)*4:(HEAD_START+(o+1)*HEAD_COLS)*4])
    if len(output) != PACKAGE_BYTES:
        raise ValueError('package length mismatch')
    result=bytes(output)
    if digest(result)!=PACK_SHA:
        raise ValueError('quantized checkpoint identity mismatch')
    return result

def validate(data: bytes):
    if len(data) != PACKAGE_BYTES:
        raise ValueError('q8 package length mismatch')
    magic,version,rows,cols,critical,source = HEADER.unpack_from(data)
    if (magic,version,rows,cols,critical,source) != (MAGIC,1,HEAD_ROWS,HEAD_COLS,len(PRECISE_IDS),BASE_SHA.encode('ascii')):
        raise ValueError('q8 package header mismatch')
    if any(data[Q_END:Q_END+PAD]):
        raise ValueError('nonzero alignment pad')
    if digest(data)!=PACK_SHA:
        raise ValueError('quantized checkpoint identity mismatch')
    return {'schema':'r1-reference-qhead8-v1','model_id':'R1-DEV-FSMN-20261107',
            'original_model_sha256':BASE_SHA,'quantized_bytes':len(data),
            'quantized_sha256':digest(data),'weight_storage_saved_bytes':BASE_FLOATS*4-len(data),
            'critical_ids':PRECISE_IDS,'quantization':'per-row symmetric int8, double scale; 5 exact fp32 rows',
            'reference_fp32_compat_gate':'FAIL_UNCHANGED','calibrated_far_frr':False,
            'physical_ssc305':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('destination exists; refusing overwrite')
    data=pack(a.source.read_bytes());status=validate(data)
    # No attempt/retry semantics are implied by static, owner-local generation.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(data)
    import json
    print(json.dumps(status,sort_keys=True))
if __name__=='__main__':main()
