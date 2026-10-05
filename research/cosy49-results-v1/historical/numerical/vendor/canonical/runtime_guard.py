"""Small scalar environment probes only; no PCM/FFT/model/native candidate."""
import struct
import sys
import numpy as np


def bits64(value):
    return struct.unpack('<Q', struct.pack('<d', value))[0]


def validate_runtime():
    assert sys.flags.optimize == 0 and sys.byteorder == 'little'
    assert sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
    # Runtime operands prevent constant folding. Together the midpoint-even and
    # three-quarter-ulp cases reject upward/downward/toward-zero and tie-away.
    one = float.fromhex('0x1p0')
    half_ulp = float.fromhex('0x1p-53')
    three_quarters = float.fromhex('0x1.8p-53')
    assert bits64(one + half_ulp) == 0x3ff0000000000000, 'binary64 RNE required'
    assert bits64(one + three_quarters) == 0x3ff0000000000001, 'binary64 RNE required'
    assert bits64(-one - half_ulp) == 0xbff0000000000000, 'binary64 RNE required'
    assert bits64(-one - three_quarters) == 0xbff0000000000001, 'binary64 RNE required'
    tiny64 = struct.unpack('<d', struct.pack('<Q', 1))[0]
    normal64 = float.fromhex('0x1p-1022')
    half = float.fromhex('0x1p-1')
    assert bits64(tiny64 * one) == 1, 'binary64 DAZ/FTZ rejected'
    assert bits64(normal64 * half) == 0x0008000000000000, 'binary64 FTZ rejected'
    tiny32 = np.array([1, 0x00400000, 0x00800000], dtype='<u4').view('<f4')
    assert [float(v) for v in tiny32] == [2.**-149, 2.**-127, 2.**-126]
    assert np.array([2.**-149, 2.**-127], dtype='<f4').view('<u4').tolist() == [1, 0x00400000]
    assert (tiny32 * np.float32(1)).view('<u4').tolist() == [1, 0x00400000, 0x00800000], 'binary32 DAZ/FTZ rejected'
    assert (tiny32[2:] * np.float32(.5)).view('<u4').tolist() == [0x00400000], 'binary32 FTZ rejected'
    return dict(binary64_rne_scalar_probes=True, binary32_gradual_underflow=True,
                binary64_gradual_underflow=True, environment_modified=False,
                candidate_calls=0, model_calls=0, pcm_frames=0)
