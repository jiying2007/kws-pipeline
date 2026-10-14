import os
import json,hashlib,wave
from pathlib import Path
def require(ok, why):
    if not ok: raise ValueError(why)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def write(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())

def validate_wav(path, expected):
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'unaliased WAV')
    require(path.stat().st_size <= 384044, '12 second maximum')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'audio digest')
    with wave.open(str(path)) as w:
        require((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype()) == (16000,1,2,'NONE'), 'derived PCM16 mono 16k only')
        require(0 < w.getnframes() <= 192000, 'full input maximum')
