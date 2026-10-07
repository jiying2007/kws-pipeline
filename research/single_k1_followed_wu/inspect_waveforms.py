"""Pure stdlib measurement of the frozen heuristic; no inference or audio mutation."""
from pathlib import Path
import sys
import hashlib
import json
import math
import struct
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1])
SPEC = SOURCE / 'heuristic-preregistered.json'
spec_bytes = SPEC.read_bytes()
spec = json.loads(spec_bytes)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def runs(mask):
    start = None
    out = []
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def interval(start, end, sample_rate, count):
    return {'start_sample': start, 'end_sample_exclusive': end,
            'start_s': start / sample_rate, 'end_s_exclusive': end / sample_rate,
            'duration_ms': 1000 * (end - start) / sample_rate,
            'position': 'whole_file' if start == 0 and end == count else
                        'leading_edge' if start == 0 else
                        'trailing_edge' if end == count else 'interior',
            'speech_identity': 'unknown'}


def read_wav(path, expected_hash):
    raw = path.read_bytes()
    observed_hash = sha256(raw)
    if observed_hash != expected_hash:
        raise ValueError(f'{path.name}: input identity mismatch')
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('Expected RIFF WAVE')
    chunks = {}
    pos = 12
    while pos + 8 <= len(raw):
        kind = raw[pos:pos + 4]
        size = struct.unpack_from('<I', raw, pos + 4)[0]
        value = raw[pos + 8:pos + 8 + size]
        if len(value) != size:
            raise ValueError('Truncated chunk')
        if kind in chunks:
            raise ValueError('Repeated chunk not supported')
        chunks[kind] = value
        pos += 8 + size + size % 2
    encoding, channels, sr, byte_rate, block_align, bits = struct.unpack_from('<HHIIHH', chunks[b'fmt '])
    if channels != 1:
        raise ValueError('Expected mono')
    data = chunks[b'data']
    if len(data) % block_align:
        raise ValueError('Nonintegral PCM frames')
    if encoding == 3 and bits == 32:
        samples = [x[0] for x in struct.iter_unpack('<f', data)]
        subtype = 'IEEE_FLOAT_32'
    elif encoding == 1 and bits == 16:
        samples = [x[0] / 32768 for x in struct.iter_unpack('<h', data)]
        subtype = 'PCM_SIGNED_16'
    else:
        raise ValueError(f'Unexpected encoding={encoding}, bits={bits}')
    if not all(math.isfinite(x) for x in samples):
        raise ValueError('Nonfinite PCM')
    if sr % 100:
        raise ValueError('10 ms not an integer number of samples')
    return samples, sr, subtype, observed_hash, sha256(data), len(raw)


results = []
for item in spec['inputs']:
    path = SOURCE / item['file']
    x, sr, subtype, wav_hash, data_hash, byte_count = read_wav(path, item['sha256'])
    n = len(x)
    zero_runs = runs([v == 0.0 for v in x])
    window = sr // 100
    n_windows = n // window
    rms = [math.sqrt(math.fsum(v * v for v in x[k * window:(k + 1) * window]) / window)
           for k in range(n_windows)]
    low_runs = runs([v < 0.001 for v in rms])
    low_candidates = [(start * window, end * window) for start, end in low_runs if end - start >= 2]
    result = {
        'file': path.name, 'sha256': wav_hash, 'pcm_data_sha256': data_hash,
        'byte_count': byte_count, 'encoding': subtype, 'channels': 1,
        'sample_rate_hz': sr, 'samples': n, 'duration_s': n / sr,
        'exact_zero_runs': [interval(a, b, sr, n) for a, b in zero_runs],
        'exact_zero_candidate_gaps_ge_20ms': [interval(a, b, sr, n) for a, b in zero_runs if (b - a) * 1000 >= 20 * sr],
        'low_rms_candidate_gaps': [interval(a, b, sr, n) for a, b in low_candidates],
        'rms_window_samples': window, 'rms_full_windows': n_windows,
        'excluded_final_partial_window': interval(n_windows * window, n, sr, n) if n_windows * window < n else None,
        'valid_external_terminal_wo_interval': None,
        'clause_gap_identity': 'unknown',
        'word_or_phoneme_boundaries': 'unknown',
        'input_unchanged': sha256(path.read_bytes()) == wav_hash
    }
    results.append(result)

report = {
    'measured_at_utc': datetime.now(timezone.utc).isoformat(),
    'heuristic_file_sha256': sha256(spec_bytes),
    'script_sha256': sha256(Path(__file__).read_bytes()),
    'waveforms': results,
    'audio_perception_limit': 'This numerical analysis exposes PCM and acoustic gaps, not recognized words or independent word/phone timestamps. No ASR, forced alignment, KWS, TTS or model was invoked.',
    'conclusion': 'The fixed waveform measurements do not identify which, if any, acoustic gap is the comma/clause gap and cannot establish a valid external terminal-窝 interval. Confirmed full words do not supply frame boundaries.',
    'smallest_annotation_needed': 'One human pass on this exact original N1: mark the audible onset and offset of the final 窝 in 你好小窝, explicitly distinguishing it from the later 屋, and report start/end sample indices or times with uncertainty. If either boundary is not audibly localizable, record uncertain/unusable rather than inventing it. Record the WAV hash and freeze this interval before viewing KWS outputs.'
}
Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'waveforms': [{key: value for key, value in r.items() if key not in ('exact_zero_runs', 'file', 'pcm_data_sha256')} for r in results], 'exact_zero_run_counts': [len(r['exact_zero_runs']) for r in results], 'conclusion': report['conclusion']}, ensure_ascii=False, indent=2))
