"""Fetch a bounded pinned public archive; materialize exactly one helper object."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile


def verify_helper(raw, lock):
    if len(raw) != lock['helper_bytes'] or hashlib.sha256(raw).hexdigest() != lock['helper_sha256']:
        raise ValueError('public helper size/hash mismatch; no code executed')
    return raw


def fetch_helper(lock, destination, *, opener=urllib.request.urlopen):
    """Caller explicitly opts into the public download; never extract other objects."""
    archive_path = Path(destination) / 'public-objects.zip'
    archive_hash = hashlib.sha256()
    archive_bytes = 0
    with archive_path.open('xb') as archive:
        for part in lock['parts']:
            part_hash = hashlib.sha256()
            retained = 0
            with opener(lock['base_url'] + '/' + part['path'], timeout=30) as response:
                while True:
                    chunk = response.read(min(1024 * 1024, part['bytes'] - retained + 1))
                    if not chunk:
                        break
                    retained += len(chunk)
                    if retained > part['bytes']:
                        raise ValueError('public archive part exceeds pinned size')
                    archive.write(chunk)
                    part_hash.update(chunk)
                    archive_hash.update(chunk)
                    archive_bytes += len(chunk)
            if retained != part['bytes'] or part_hash.hexdigest() != part['sha256']:
                raise ValueError('public archive part size/hash mismatch')
    if archive_bytes != lock['archive_bytes'] or archive_hash.hexdigest() != lock['archive_sha256']:
        raise ValueError('public archive size/hash mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if info.filename == lock['helper_object']]
        if len(members) != 1 or members[0].file_size != lock['helper_bytes']:
            raise ValueError('public helper ZIP member missing, duplicated or wrong-sized')
        with archive.open(members[0]) as stream:
            raw = stream.read(lock['helper_bytes'] + 1)
    return verify_helper(raw, lock)


def cached_helper(lock, cache_directory, destination, *, opener=urllib.request.urlopen):
    """Use verified public source bytes only; a cache hit never establishes trust.

    A corrupt hit fails closed instead of silently replacing evidence. A missing
    entry follows the original full pinned archive/part verification path.
    """
    import os
    import re
    import tempfile

    digest = lock['helper_sha256']
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('invalid public helper cache identity')
    if type(lock['helper_bytes']) is not int or not 0 < lock['helper_bytes'] <= 1024 * 1024:
        raise ValueError('invalid public helper cache size')
    cache = Path(cache_directory)
    entry = cache / (digest + '.py')
    try:
        with entry.open('rb') as stream:
            return verify_helper(stream.read(lock['helper_bytes'] + 1), lock)
    except FileNotFoundError:
        pass
    raw = fetch_helper(lock, destination, opener=opener)
    # Never cache the archive, other members, weights, logs, or private inputs.
    cache.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + digest + '-', dir=cache)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
        os.replace(temporary, entry)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return verify_helper(raw, lock)
