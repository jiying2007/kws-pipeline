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
