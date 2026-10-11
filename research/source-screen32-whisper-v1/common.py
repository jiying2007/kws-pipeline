"""Fixed-run metadata and byte checks. Standard library only; never loads models."""
from pathlib import Path
import hashlib
import json
import os
import re
import stat
import wave

EXPERIMENT = 'qwen16-whisper-small-third-once-v1'
BRANCH = 'research/' + EXPERIMENT
JOB_SHA = '425a705515339ff7054e4f1148971a342e5f882a1e9b84ac2be144a3d7aeecc5'
MODEL_SHA = '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'
SOURCE_REVISION = '86098128c0b4f24f0e2aa2994de830614b474227'
MODEL_URL = 'https://openaipublic.azureedge.net/main/whisper/models/' + MODEL_SHA + '/small.pt'
LIMITS = {'clips': 16, 'new_whisper_calls': 16, 'new_tts_calls': 0,
          'new_qwen_calls': 0, 'new_sensevoice_calls': 0, 'job_seconds': 2400,
          'cell_seconds': 300, 'load_seconds': 1200, 'setup_seconds': 1200,
          'download_bytes': 1536 * 1024**2, 'output_bytes': 64 * 1024**2}
BOUNDARY = {'source_role': 'source_screen_only', 'training_admitted': False,
            'shipping_approved': False, 'human_gold': False,
            'acoustic_verbatim_completeness': 'unknown'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def regular(path, maximum=None):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'aliased file path')
    s = path.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'unaliased regular file required')
    require(maximum is None or s.st_size <= maximum, 'file size bound')
    return s


def read_json(path, maximum=8 * 1024**2):
    regular(path, maximum)
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def write_once(path, value):
    with Path(path).open('xb') as target:
        target.write(canonical(value)); target.flush(); os.fsync(target.fileno())


def atomic(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as target:
        target.write(canonical(value)); target.flush(); os.fsync(target.fileno())
    os.replace(temp, path)


def identity(path):
    size = regular(path).st_size
    return {'bytes': size, 'sha256': file_hash(path)}


def verify_inventory(root, expected):
    root = Path(root)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in root.rglob('*')), 'inventory aliases')
    require(actual == set(expected), 'inventory exact membership')
    for name, row in expected.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'inventory path')
        require(identity(root / name) == row, 'inventory byte identity')


def bind_input(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'blind input root')
    require({p.name for p in root.iterdir()} == {'job.json', 'audio'}, 'audio-only input membership')
    require(file_hash(root / 'job.json') == JOB_SHA, 'exact retained sixteen job')
    job = read_json(root / 'job.json', 65536)
    require(set(job) == {'schema', 'clips'} and job['schema'] == 'blind-source-screen32-job-v1', 'job schema')
    require(len(job['clips']) == 16, 'fixed sixteen denominator')
    hashes = []
    for index, row in enumerate(job['clips'], 1):
        require(set(row) == {'audio_id', 'audio_path', 'wav_sha256'}, 'blind input fields')
        digest = row['wav_sha256']
        require(re.fullmatch('[0-9a-f]{64}', digest) and row['audio_id'] == f'clip-{index:06d}' and
                row['audio_path'] == 'audio/' + digest + '.wav', 'blind identity')
        hashes.append(digest)
        wav_bytes(root / row['audio_path'], digest)
    require(hashes == sorted(set(hashes)), 'fixed content order')
    require({p.name for p in (root / 'audio').iterdir()} == {s + '.wav' for s in hashes}, 'audio set')
    return job


def wav_bytes(path, expected_sha):
    regular(path, 960_044)
    require(file_hash(path) == expected_sha, 'WAV byte identity')
    with wave.open(str(path), 'rb') as source:
        frames = source.getnframes()
        require((source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) ==
                (16000, 1, 2, 'NONE'), '16k mono PCM16 required')
        require(0 < frames <= 480000, 'Whisper short input bound; never trim')
        pcm = source.readframes(frames + 1)
    require(len(pcm) == frames * 2, 'complete PCM frame bytes')
    return pcm, frames
