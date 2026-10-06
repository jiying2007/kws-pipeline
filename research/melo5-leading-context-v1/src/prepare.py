"""Pure retained WAV header/byte checks; no inference or audio playback."""
from pathlib import Path
import hashlib,json,struct
ROOT=Path(__file__).resolve().parents[1]
def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')

def inspect_wav(data):
    if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('RIFF/WAVE required')
    if struct.unpack_from('<I', data, 4)[0] + 8 != len(data):
        raise ValueError('RIFF size mismatch')
    if data[12:16] != b'fmt ' or struct.unpack_from('<I', data, 16)[0] != 16:
        raise ValueError('exact retained PCM header required')
    if struct.unpack_from('<HHIIHH', data, 20) != (1, 1, 16000, 32000, 2, 16):
        raise ValueError('16k mono little-endian PCM16 required')
    if data[36:40] != b'data':
        raise ValueError('exact retained data region required')
    n = struct.unpack_from('<I', data, 40)[0]
    if n % 2 or n + 44 != len(data):
        raise ValueError('PCM region length mismatch')
    return dict(frames=n//2, data_offset=44, pcm_bytes=n, pcm_sha256=sha(data[44:]), wav_sha256=sha(data), wav_bytes=len(data))

LEADING_SAMPLES = 24000
APPENDED_SAMPLES = 4800
EXPECTED_IDS = ['M1', 'M2', 'M3', 'M4', 'M5']
EXPECTED_TEXTS = [r['declared_text'] for r in json.loads((ROOT/'metadata/manifest.json').read_text())['rows']]

def verify_context(original, modified):
    before = inspect_wav(original); after = inspect_wav(modified)
    if after['frames'] != before['frames'] + LEADING_SAMPLES + APPENDED_SAMPLES:
        raise ValueError('exact 24000 leading and 4800 appended samples required')
    left = 44 + LEADING_SAMPLES * 2
    right = left + before['pcm_bytes']
    if modified[44:left] != b'\0' * (LEADING_SAMPLES * 2):
        raise ValueError('digital-zero leading context required')
    if modified[left:right] != original[44:]:
        raise ValueError('unaltered entire original PCM required')
    if modified[right:] != b'\0' * (APPENDED_SAMPLES * 2):
        raise ValueError('digital-zero appended context required')
    return True

