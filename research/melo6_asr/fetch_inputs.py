"""Fetch exactly the two approved blind files from one immutable public data commit.

No model imports, credentials, redirects, retries, package setup, or decoding.
The source freeze binds this helper and its transport manifest before any GET.
"""
import hashlib
import http.client
import json
from pathlib import Path
import re
import ssl

REPOSITORY = 'jiying2007/kws-data'
PREFIX = 'research/2026-10-05-melo6-source-screen/recovery/blind/'
NAMES = ('blind-inputs.zip', 'blind-input-freeze.json')
CAPS = {'blind-inputs.zip': 2 * 1024**2, 'blind-input-freeze.json': 16384}
RELEASE_KEYS = {'blind-inputs.zip': 'blind_archive_sha256',
                'blind-input-freeze.json': 'blind_freeze_sha256'}


def require(value, code):
    if not value:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_source(source):
    require(type(source) is dict and set(source) == {'schema', 'repository', 'commit', 'files'}, 'INPUT_SOURCE_FIELDS')
    require(source['schema'] == 'melo6-public-data-input-source-v1' and source['repository'] == REPOSITORY, 'INPUT_SOURCE_TARGET')
    require(type(source['commit']) is str and re.fullmatch(r'[0-9a-f]{40}', source['commit'])
            and source['commit'] != '0' * 40, 'IMMUTABLE_DATA_COMMIT')
    require(type(source['files']) is list and len(source['files']) == 2, 'EXACT_TWO_INPUTS')
    for name, row in zip(NAMES, source['files']):
        require(type(row) is dict and set(row) == {'name', 'path', 'bytes', 'sha256'}, 'INPUT_FILE_FIELDS')
        require(row['name'] == name and row['path'] == PREFIX + name, 'FIXED_INPUT_PATH')
        require(type(row['bytes']) is int and 0 < row['bytes'] <= CAPS[name], 'INPUT_SIZE_CAP')
        require(type(row['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'INPUT_SHA256')
    return source


def get_once(path, expected, connection_factory=http.client.HTTPSConnection):
    # HTTPSConnection does not follow Location headers or use ambient proxy credentials.
    connection = connection_factory('raw.githubusercontent.com', timeout=30,
                                    context=ssl.create_default_context())
    try:
        connection.request('GET', path, headers={'Accept': 'application/octet-stream',
                                               'Accept-Encoding': 'identity'})
        response = connection.getresponse()
        require(response.status == 200, 'INPUT_HTTP_STATUS')
        length = response.getheader('Content-Length')
        require(length is None or length == str(expected['bytes']), 'INPUT_CONTENT_LENGTH')
        require(response.getheader('Content-Encoding') in (None, 'identity'), 'INPUT_CONTENT_ENCODING')
        raw = response.read(expected['bytes'] + 1)
        require(len(raw) == expected['bytes'], 'INPUT_RESPONSE_BYTES')
        require(sha(raw) == expected['sha256'], 'INPUT_RESPONSE_SHA256')
        return raw
    finally:
        connection.close()


def fetch(source, incoming, downloader=get_once):
    source = validate_source(source)
    incoming = Path(incoming)
    require(incoming.name == 'incoming' and incoming.parent.name == 'runtime', 'INPUT_DESTINATION')
    require(not incoming.exists() and not incoming.is_symlink()
            and not incoming.parent.exists() and not incoming.parent.is_symlink(), 'FRESH_INPUT_RUNTIME')
    # Keep both verified bodies in memory; a failed GET never leaves a partial handoff.
    bodies = []
    for row in source['files']:
        path = '/' + REPOSITORY + '/' + source['commit'] + '/' + row['path']
        raw = downloader(path, row)
        require(type(raw) is bytes and len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'INPUT_TRANSPORT_IDENTITY')
        bodies.append(raw)
    incoming.parent.mkdir()
    incoming.mkdir()
    for row, raw in zip(source['files'], bodies):
        with (incoming / row['name']).open('xb') as output:
            output.write(raw)
    return {'schema': 'melo6-input-transport-receipt-v1', 'repository': REPOSITORY,
            'commit': source['commit'], 'files': source['files'], 'get_requests': 2,
            'response_body_bytes': sum(len(raw) for raw in bodies),
            'tls_certificate_verification': True, 'redirects': 0, 'retries': 0}


def main():
    root = Path(__file__).resolve().parent
    require(root.name == 'melo6_asr' and root.parent.name == 'research', 'INPUT_SOURCE_SCOPE')
    release = json.loads((root.parent / 'melo6-source-screen-release.json').read_bytes())
    require(release.get('approved') is True, 'APPROVED_RELEASE_REQUIRED')
    freeze_path = root / 'candidate-freeze.json'
    require(not freeze_path.is_symlink() and sha(freeze_path.read_bytes()) == release['asr_candidate_sha256'], 'SOURCE_FREEZE_HASH')
    freeze = json.loads(freeze_path.read_bytes())
    require(freeze.get('schema') == 'melo6-component-code-freeze-v1' and freeze.get('component') == 'asr', 'SOURCE_FREEZE_SCOPE')
    files = [p for p in root.rglob('*') if p.is_file() and p != freeze_path]
    require(not any(p.is_symlink() for p in root.rglob('*')) and
            {p.relative_to(root).as_posix() for p in files} == set(freeze['files']), 'SOURCE_FREEZE_MEMBERSHIP')
    for name, digest in freeze['files'].items():
        require(sha((root / name).read_bytes()) == digest, 'SOURCE_FREEZE_BYTES')
    source = validate_source(json.loads((root / 'input-source.json').read_bytes()))
    for row in source['files']:
        require(row['sha256'] == release[RELEASE_KEYS[row['name']]], 'INPUT_RELEASE_BINDING')
    receipt = fetch(source, root / 'runtime/incoming')
    from release_gate import verify_release
    verify_release(root, 'asr', root.parent / 'melo6-source-screen-release.json')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('{"status":"blocked","stage":"immutable_public_data_fetch","raw_exception_disclosed":false}')
        raise SystemExit(1)
