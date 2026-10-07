"""Fixed future Melo output conversion; no TTS/ASR imports or model calls.

Uses installed NumPy/SciPy and standard-library WAV I/O. Nothing runs on import.
PCM16 floor/saturation is checked against the prior 12-sample SoundFile fixture.
"""
import io
import math
import wave

NATIVE_RATE = 44100
PCM_RATE = 16000
MAX_SECONDS = 10


def check_native(samples):
    import numpy as np
    x = np.asarray(samples)
    if x.dtype != np.dtype('float32') or x.ndim != 1 or not 0 < len(x) <= NATIVE_RATE * MAX_SECONDS:
        raise ValueError('expected finite mono float32 native audio, 0 < duration <= 10s')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite native audio')
    return x


def native_wav(samples):
    from scipy.io.wavfile import write
    x = check_native(samples)
    out = io.BytesIO()
    write(out, NATIVE_RATE, x)
    data = out.getvalue()
    if len(data) > x.size * 4 + 4096:
        raise ValueError('native WAV header cap')
    return data


def quantize_pcm16(samples):
    import numpy as np
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('nonfinite/non-mono derivative')
    # Match the already calibrated libsndfile PCM16 floor and saturation rule.
    return np.clip(np.floor(y * 32768.0), -32768, 32767).astype('<i2')


def derivative_wav(samples):
    import numpy as np
    from scipy.signal import resample_poly
    x = check_native(samples)
    y = resample_poly(x.astype(np.float64), up=160, down=441,
                      window=('kaiser', 5.0), padtype='constant', cval=0.0)
    if len(y) != math.ceil(len(x) * 160 / 441) or not 0 < len(y) <= PCM_RATE * MAX_SECONDS:
        raise ValueError('unexpected derivative frame count')
    pcm = quantize_pcm16(y)
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(PCM_RATE)
        f.writeframes(pcm.tobytes())
    data = out.getvalue()
    if len(data) != 44 + 2 * len(pcm):
        raise ValueError('noncanonical PCM WAV')
    return data, {'source_frames': len(x), 'derivative_frames': len(y),
                  'resampled_peak': float(np.max(np.abs(y))),
                  'prequantization_saturation_count': int(np.count_nonzero((y >= 1) | (y < -1))),
                  'gain_normalization': False, 'trimmed': False}
