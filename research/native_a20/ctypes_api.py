"""Separate FFT64 opaque-state ABI. Import/load never initializes or steps A20."""
import ctypes as C
import json
from pathlib import Path

F32P = C.POINTER(C.c_float)
I16P = C.POINTER(C.c_int16)
U64P = C.POINTER(C.c_uint64)


class PCM_Batch(C.Structure):
    _fields_ = [('call_index', C.c_uint64), ('available_samples', C.c_uint64),
                ('call_samples', C.c_size_t), ('waveform_samples', C.c_size_t),
                ('fbank_rows', C.c_size_t), ('splice_rows', C.c_size_t),
                ('selected_rows', C.c_size_t), ('is_final_short', C.c_uint32),
                ('fbank', F32P), ('rows', F32P), ('centers', U64P),
                ('wave_samples', C.c_uint32), ('feature_count', C.c_uint32),
                ('offset', C.c_uint32)]


class Decoder_Result(C.Structure):
    _fields_ = [('valid', C.c_int32), ('state', C.c_int32), ('keyword', C.c_int32),
                ('start_frame', C.c_int64), ('end_frame', C.c_int64),
                ('score', C.c_double), ('rows_decoded', C.c_size_t)]


class Stream_Batch(C.Structure):
    _fields_ = [('frontend', C.POINTER(PCM_Batch)), ('logits', F32P),
                ('cache', F32P), ('result', C.POINTER(Decoder_Result))]


Stream_Callback = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(Stream_Batch))
PCM_Callback = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PCM_Batch))
Trace_Callback = C.CFUNCTYPE(None, C.c_void_p, C.c_uint64, C.c_size_t, F32P, F32P, F32P)


class Aligned_Arena:
    def __init__(self, size, alignment):
        assert size > 0 and alignment > 0 and alignment & (alignment-1) == 0
        self.backing = C.create_string_buffer(size+alignment-1)
        address = (C.addressof(self.backing)+alignment-1) & -alignment
        self.pointer = C.c_void_p(address)
        self.size = size
        self.alignment = alignment


def load(path, resource_path=None):
    lib = C.CDLL(str(path))
    for name in ['a20fft64s_state_bytes', 'a20fft64s_state_alignment',
                 'donor_fft64_pcm_state_bytes', 'donor_fft64_pcm_state_alignment',
                 'donor_fft64_state_bytes', 'donor_fft64_state_alignment']:
        fn = getattr(lib, name); fn.argtypes = []; fn.restype = C.c_size_t
    for name, args, result in [
        ('a20fft64s_init', [C.c_void_p, F32P, C.c_size_t], C.c_int),
        ('a20fft64s_reset', [C.c_void_p], C.c_int),
        ('a20fft64s_set_trace', [C.c_void_p, F32P, Trace_Callback, C.c_void_p], C.c_int),
        ('a20fft64s_feed', [C.c_void_p, I16P, C.c_size_t, Stream_Callback, C.c_void_p], C.c_int),
        ('a20fft64s_finish', [C.c_void_p, Stream_Callback, C.c_void_p], C.c_int),
        ('a20fft64s_cache', [C.c_void_p], F32P),
        ('a20fft64s_pcm', [C.c_void_p], C.c_void_p),
        ('donor_fft64_pcm_init', [C.c_void_p], C.c_int),
        ('donor_fft64_pcm_reset', [C.c_void_p], C.c_int),
        ('donor_fft64_pcm_feed', [C.c_void_p, I16P, C.c_size_t, PCM_Callback, C.c_void_p], C.c_int),
        ('donor_fft64_pcm_finish', [C.c_void_p, PCM_Callback, C.c_void_p], C.c_int),
        ('donor_fft64_environment_ok', [], C.c_int)]:
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = result
    if resource_path is None:
        resource_path = Path(path).with_name('resource-sizes.json')
    sizes = json.loads(Path(resource_path).read_text())
    assert C.sizeof(PCM_Batch) == sizes['pcm_batch_bytes']
    assert C.sizeof(Stream_Batch) == sizes['stream_batch_bytes']
    assert C.sizeof(Decoder_Result) == sizes['decoder_result_bytes']
    assert lib.a20fft64s_state_bytes() == sizes['stream_bytes']
    assert lib.a20fft64s_state_alignment() == sizes['stream_alignment']
    assert lib.donor_fft64_pcm_state_bytes() == sizes['pcm_bytes']
    assert lib.donor_fft64_pcm_state_alignment() == sizes['pcm_alignment']
    assert lib.donor_fft64_state_bytes() == sizes['frontend_bytes']
    assert lib.donor_fft64_state_alignment() == sizes['frontend_alignment']
    return lib, sizes
