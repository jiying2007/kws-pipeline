#!/usr/bin/env python3
"""Offline, standard-library saved-byte verification/restoration. No scientific execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_ARCHIVE = 128 * 1024 * 1024
MAX_FILES = 1000
MAX_FILE = 64 * 1024 * 1024

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_name(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value and '\x00' not in value,
            'invalid path')
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ('', '.', '..') for x in value.split('/')),
            'unsafe path')
    return value

def read_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON')
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)

def verify(data_dir, output=None):
    data_dir = Path(data_dir)
    manifest_path = data_dir / 'ARCHIVE.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'manifest type')
    require(manifest_path.stat().st_size <= 1024 * 1024, 'manifest size')
    m = read_json(manifest_path.read_bytes())
    require(m['schema'] == 'cosy49-public-archive-v1', 'manifest schema')
    require(type(m['archive_bytes']) is int and 0 < m['archive_bytes'] <= MAX_ARCHIVE, 'archive cap')
    require(type(m['logical_bytes']) is int and 0 < m['logical_bytes'] <= MAX_ARCHIVE, 'logical cap')
    require(0 < len(m['members']) <= MAX_FILES and 0 < len(m['chunks']) <= MAX_FILES, 'entry cap')
    expected = {}
    for row in m['members']:
        name = safe_name(row['path'])
        require(name not in expected, 'duplicate member')
        require(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_FILE, 'member cap')
        expected[name] = row
    require(sum(e['bytes'] for e in expected.values()) == m['logical_bytes'], 'logical total')
    for row in m['chunks']:
        require(type(row['bytes']) is int and 0 < row['bytes'] <= 1565280, 'chunk cap')
    require(sum(row['bytes'] for row in m['chunks']) == m['archive_bytes'], 'chunk total')
    chunks = []
    seen = set()
    for row in m['chunks']:
        name = safe_name(row['path'])
        require(name not in seen and name.startswith('chunks/'), 'duplicate or misplaced chunk')
        seen.add(name)
        path = data_dir / name
        require(not any(p.is_symlink() for p in [path, *path.parents] if p != data_dir.parent), 'symlink chunk')
        require(path.is_file() and path.stat().st_size == row['bytes'], 'chunk size')
        require(type(row['bytes']) is int and 0 < row['bytes'] <= 1565280, 'chunk cap')
        b = path.read_bytes()
        require(digest(b) == row['sha256'], 'chunk hash')
        chunks.append(b)
    chunk_paths = list((data_dir / 'chunks').rglob('*'))
    require(not any(p.is_symlink() for p in chunk_paths), 'symlink in chunk tree')
    actual_chunks = {str(p.relative_to(data_dir)) for p in chunk_paths if p.is_file()}
    require(actual_chunks == seen, 'extra or missing chunk')
    archive = b''.join(chunks)
    require(len(archive) == m['archive_bytes'] and digest(archive) == m['archive_sha256'], 'archive identity')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        infos = z.infolist()
        require(len(infos) == len(expected), 'ZIP member count')
        names = [safe_name(i.filename) for i in infos]
        require(len(set(names)) == len(names) and set(names) == set(expected), 'ZIP closed allowlist')
        for info in infos:
            e = expected[info.filename]
            require(not info.is_dir() and not (info.flag_bits & 1), 'ZIP kind/encryption')
            mode = (info.external_attr >> 16) & 0xFFFF
            require(stat.S_IFMT(mode) in (0, stat.S_IFREG), 'ZIP non-regular member')
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'ZIP method')
            require(info.file_size == e['bytes'], 'ZIP member size')
            b = z.read(info)
            require(len(b) == e['bytes'] and digest(b) == e['sha256'], 'member hash')
        # All input bytes pass before creating any output. Never overwrite.
        if output is not None:
            output = Path(output)
            require(not output.exists() and not output.is_symlink(), 'output must be new')
            output.mkdir(parents=False)
            for name in names:
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    f.write(z.read(name))
    return {'status': 'PASS_SAVED_BYTES_ONLY', 'files': len(expected),
            'bytes': m['logical_bytes'], 'archive_bytes': m['archive_bytes'],
            'new_model_frontend_decoder_training_calls': 0}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir, args.output), sort_keys=True))
