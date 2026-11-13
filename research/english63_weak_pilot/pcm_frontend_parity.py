#!/usr/bin/env python3
"""Source-bound English63 streaming PCM-to-log-Mel frontend parity.

Research ONLY. This does not implement a wake-event detector, certify labels,
train a model, select a threshold, measure FAR/FRR, or qualify SSC305.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import wave
import zipfile

POLICY = 'english63-original-pcm-streaming-frontend-parity-v1'
TTS_SHA256 = 'f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc'
REVIEW_SHA256 = '89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12'
CHUNKS = (159, 160, 161, 1600, 3199)
TOL = 1e-5
FRAME = 400
HOP = 160
MAX_SAMPLES = 16000 * 12
CELL = re.compile(r'en-[0-9a-f]{24}\Z')
SPEAKERS = {'dylan', 'eric', 'ono_anna', 'sohee', 'vivian', 'serena', 'uncle_fu'}


def ensure(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FrameBuffer:
    """Carry only the 25 ms unfinished window between arbitrary PCM chunks."""
    def __init__(self):
        self.tail = []
        self.samples = 0
        self.frames = 0
        self.maximum_buffer_samples = 0

    def push(self, values):
        # Accept float32 PCM sample iterables; never pad partial tail at EOF.
        ensure(0 < len(values) <= max(CHUNKS), 'EN_PCM_FEED_BUDGET')
        self.samples += len(values)
        ensure(self.samples <= MAX_SAMPLES, 'EN_PCM_TOTAL_BUDGET')
        self.tail.extend(values)
        self.maximum_buffer_samples = max(self.maximum_buffer_samples,
                                          len(self.tail))
        out = []
        while len(self.tail) >= FRAME:
            out.append(self.tail[:FRAME])
            del self.tail[:HOP]
            self.frames += 1
        ensure(len(self.tail) < FRAME and self.frames <= 1200,
               'EN_PCM_FRAME_BUDGET')
        return out

    def finish(self):
        ensure(FRAME <= self.samples <= MAX_SAMPLES and
               self.frames == 1 + (self.samples - FRAME) // HOP and
               50 <= self.frames <= 1200,
               'EN_PCM_END_GEOMETRY')
        return self.frames


def verify_receipt(report):
    """Audio/Torch/numpy-free integrity check, suitable for isolated source CI."""
    ensure(type(report) is dict and report.get('schema') == POLICY and
           report.get('status') == 'MEASURED_ORIGINAL_ENGLISH63_PCM_FRONTEND_PARITY_NOT_KWS' and
           report.get('source_tts_zip_sha256') == TTS_SHA256 and
           report.get('source_review_json_sha256') == REVIEW_SHA256 and
           report.get('runner_sha256') == digest(Path(__file__).read_bytes()) and
           report.get('audio_clips_verified') == 63 and
           report.get('model_inference_executed') is False and
           report.get('train_or_threshold_tuned') is False and
           report.get('new_tts_asr_ipa_calls') == 0 and
           report.get('human_gold') is False and
           report.get('independent_frr') == 'NOT_RUN' and
           report.get('continuous_far_per_hour') == 'NOT_RUN' and
           report.get('ssc305_qualified') is False and
           report.get('continuous_wake_detector_tested') is False and
           report.get('formal_training_admitted') is False and
           report.get('model_promotion_allowed') is False,
           'EN_PCM_RECEIPT_UNSAFE')
    plans = report.get('plans')
    ensure(type(plans) is list and [x.get('chunk_samples') for x in plans] == list(CHUNKS),
           'EN_PCM_RECEIPT_PLAN_DRIFT')
    for v in plans:
        err = v.get('max_abs_logmel_error')
        ensure(type(v.get('clips')) is int and v['clips'] == 63 and
               type(v.get('frames')) is int and v['frames'] == 13010 and
               type(v.get('max_buffer_samples')) is int and
               FRAME <= v['max_buffer_samples'] <= FRAME + v['chunk_samples'] - 1 and
               type(err) in (int, float) and math.isfinite(err) and 0 <= err <= TOL,
               'EN_PCM_RECEIPT_NUMERIC_INVALID')
    return True


def mel_bank():
    """Frozen 32-bin frontend from the saved English63 weak research recipe."""
    import numpy as np
    freq = np.linspace(0.0, 8000.0, 257, dtype=np.float64)
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    knots = mel_to_hz(np.linspace(hz_to_mel(70), hz_to_mel(7600), 34))
    bank = np.zeros((32, 257), np.float64)
    for i in range(32):
        left, center, right = knots[i:i+3]
        bank[i, :] = np.maximum(0, np.minimum((freq-left)/(center-left),
                                              (right-freq)/(right-center)))
        bank[i] /= max(1.e-12, bank[i].sum())
    return bank.astype(np.float32)


def logmel(frames, bank):
    import numpy as np
    window = np.hanning(FRAME).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, n=512, axis=1)) ** 2
    return np.log(np.maximum(power @ bank.T, 1e-8)).astype(np.float32)


def pcm_from_wav(raw, row):
    import numpy as np
    ensure(digest(raw) == row.get('wav_sha256'), 'EN_PCM_WAV_SHA_CHANGED')
    with wave.open(io.BytesIO(raw), 'rb') as w:
        ensure(w.getnchannels() == 1 and w.getsampwidth() == 2 and
               w.getframerate() == 16000 and w.getcomptype() == 'NONE',
               'EN_PCM_WAV_GEOMETRY')
        count = w.getnframes()
        ensure(8000 <= count <= MAX_SAMPLES, 'EN_PCM_WAV_DURATION')
        data = w.readframes(count)
        ensure(len(data) == count*2 and not w.readframes(1),
               'EN_PCM_WAV_TRUNCATION')
    ensure(digest(data) == row.get('pcm_sha256'), 'EN_PCM_RAW_SHA_CHANGED')
    samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
    ensure(bool(np.isfinite(samples).all()), 'EN_PCM_NONFINITE')
    return samples


def run(source_zip: Path, review_json: Path, output: Path):
    import numpy as np
    ensure(source_zip.is_absolute() and review_json.is_absolute() and
           source_zip.is_file() and review_json.is_file() and
           not source_zip.is_symlink() and not review_json.is_symlink() and
           source_zip.stat().st_size < 16*1024*1024 and
           review_json.stat().st_size < 1024*1024 and
           output.is_absolute() and output.parent.is_dir() and
           not output.exists() and not output.parent.is_symlink(),
           'EN_PCM_CREATE_ONLY_INPUTS')
    ensure(digest(source_zip.read_bytes()) == TTS_SHA256 and
           digest(review_json.read_bytes()) == REVIEW_SHA256,
           'EN_PCM_SOURCE_IDENTITY')
    review = json.loads(review_json.read_bytes())
    rows = review.get('rows')
    ensure(review.get('source_tts_run_id') == 43242278962 and
           review.get('schema') == 'english63-independent-evidence-joined-review-priority-v1' and
           review.get('status') == 'MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD' and
           review.get('human_gold') is False and
           review.get('training_admitted') is False and
           review.get('audio_rows_reused') == 63 and
           type(rows) is list and len(rows) == 63,
           'EN_PCM_SOURCE_POLICY')
    bank = mel_bank()
    results = {n: {'chunk_samples': n, 'clips': 0, 'frames': 0,
                   'max_abs_logmel_error': 0.0, 'max_buffer_samples': 0}
               for n in CHUNKS}
    seen, role_by_speaker = set(), {}
    with zipfile.ZipFile(source_zip) as z:
        wav_paths = {p for p in z.namelist() if p.endswith('/audio-16k.wav')}
        ensure(len(wav_paths) == 63, 'EN_PCM_WAV_MEMBER_COUNT')
        for row in rows:
            cell, role, speaker = (row.get('cell_id'), row.get('role'),
                                   row.get('source_family_id'))
            ensure(type(cell) is str and CELL.fullmatch(cell) and
                   cell not in seen and role in ('train-candidate', 'development-only') and
                   speaker in {'qwen3-preset-' + s for s in SPEAKERS} and
                   row.get('training_admitted') is False and
                   row.get('human_gold') is False and
                   row.get('trusted_negative') is False and
                   row.get('phonetic_label_authenticated') is False,
                   'EN_PCM_ROW_POLICY')
            seen.add(cell)
            role_by_speaker.setdefault(speaker, set()).add(role)
            name = ('english-seven-voice-actual/capture/cells/' +
                    cell + '/audio-16k.wav')
            ensure(name in wav_paths, 'EN_PCM_WAV_MEMBER_MISSING')
            wav_paths.remove(name)
            samples = pcm_from_wav(z.read(name), row)
            reference = logmel(np.lib.stride_tricks.sliding_window_view(
                samples, FRAME)[::HOP], bank)
            ensure(50 <= len(reference) <= 1200 and np.isfinite(reference).all(),
                   'EN_PCM_REFERENCE_INVALID')
            for n, stats in results.items():
                framer = FrameBuffer()
                offset = 0
                max_err = 0.0
                for p in range(0, len(samples), n):
                    frames = framer.push(samples[p:p+n])
                    if frames:
                        x = logmel(np.asarray(frames, dtype=np.float32), bank)
                        expected = reference[offset:offset+len(frames)]
                        ensure(x.shape == expected.shape, 'EN_PCM_FRAME_COUNT_DRIFT')
                        max_err = max(max_err, float(np.max(np.abs(x - expected))))
                        offset += len(frames)
                ensure(framer.finish() == len(reference) == offset and
                       np.isfinite(max_err) and max_err <= TOL,
                       'EN_PCM_NUMERIC_OR_GEOMETRY_DRIFT')
                stats['clips'] += 1
                stats['frames'] += offset
                stats['max_abs_logmel_error'] = max(stats['max_abs_logmel_error'], max_err)
                stats['max_buffer_samples'] = max(stats['max_buffer_samples'],
                                                  framer.maximum_buffer_samples)
        ensure(not wav_paths and len(seen) == 63 and
               len(role_by_speaker) == 7 and
               all(len(v) == 1 for v in role_by_speaker.values()) and
               sum('development-only' in v for v in role_by_speaker.values()) == 2,
               'EN_PCM_ROLE_OR_MEMBER_DRIFT')
    result = {
        'schema': POLICY,
        'status': 'MEASURED_ORIGINAL_ENGLISH63_PCM_FRONTEND_PARITY_NOT_KWS',
        'source_tts_zip_sha256': TTS_SHA256,
        'source_review_json_sha256': REVIEW_SHA256,
        'runner_sha256': digest(Path(__file__).read_bytes()),
        'audio_clips_verified': 63, 'plans': list(results.values()),
        'runtime': {'numpy': np.__version__, 'python': sys.version.split()[0]},
        'model_inference_executed': False, 'train_or_threshold_tuned': False,
        'new_tts_asr_ipa_calls': 0, 'human_gold': False,
        'independent_frr': 'NOT_RUN', 'continuous_far_per_hour': 'NOT_RUN',
        'ssc305_qualified': False, 'continuous_wake_detector_tested': False,
        'formal_training_admitted': False, 'model_promotion_allowed': False,
    }
    verify_receipt(result)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    output.chmod(0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tts-zip', required=True, type=Path)
    parser.add_argument('--review-json', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.tts_zip, args.review_json, args.output)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        print('EN_PCM_FRONTEND_STOP: ' + str(e), file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'plans': result['plans'],
                      'model_promotion_allowed': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
