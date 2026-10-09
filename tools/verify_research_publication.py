#!/usr/bin/env python3
"""Validate publication metadata; never execute archived research or fetch objects.

CI explicitly fetches only the two pinned JSON metadata files (2 MiB combined).
Offline validation requires their exact bytes via --metadata-dir. This checks
internal consistency and pinned identities, not remote PR freshness or acoustics.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'research/consolidation/source-retention-publication-2026-10-09.json'
PREFIX = 'research/2026-10-08-d20-d90-host-research/'
REPO = 'jiying2007/kws-data'
CATALOG_STATUS = ('PLANNED_NOT_PUBLISHED; planned_pipeline_path is not a live repository path. '
                  'Source/report bytes are published only in this recoverable archive.')
MAX_METADATA_BYTES = 2 * 1024 * 1024
WORKFLOWS = {
    '.github/workflows/host-research-publication.yml': 379143432,
    '.github/workflows/research-retention.yml': 377351813,
    '.github/workflows/assets.yml': 370944134,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def safe_path(value):
    return (isinstance(value, str) and bool(value) and not value.startswith('/')
            and '\\' not in value and '\x00' not in value
            and all(x not in ('', '.', '..') for x in value.split('/')))


def unique_json(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_json)


def validate_index(index):
    require(set(index) == {'schema', 'repository', 'base_commit', 'date', 'publication_scope', 'expanded_pipeline_source_copy', 'source_repository', 'source_pull_request', 'archive', 'files', 'source_ci_runs', 'observed_at', 'snapshot_semantics'}, 'index fields')
    require(index['schema'] == 'research-source-retention-publication-v1', 'schema')
    require(index['repository'] == 'jiying2007/kws-pipeline' and index['source_repository'] == REPO, 'repository')
    require(index['publication_scope'] == 'PUBLIC_METADATA_ONLY', 'scope')
    require(index['expanded_pipeline_source_copy'] == 'NOT_PUSHED', 'expanded source status')
    require(digest(index['base_commit'], 40), 'base commit')
    observed = dt.datetime.fromisoformat(index['observed_at'].replace('Z', '+00:00'))
    require(observed.tzinfo is not None and observed.date().isoformat() == index['date'], 'observation date')
    require(index['snapshot_semantics'] == 'Point-in-time public metadata observation; not a live PR status or acoustic qualification.', 'snapshot semantics')
    pr = index['source_pull_request']
    require(pr['number'] == 27 and pr['url'] == f'https://github.com/{REPO}/pull/27', 'PR identity')
    require(digest(pr['head_commit'], 40), 'PR head')
    require(pr['state'] == 'open' and pr['draft'] is True and pr['merged'] is False, 'PR snapshot status')
    require(set(pr) == {'number', 'url', 'head_commit', 'state', 'draft', 'merged'}, 'PR snapshot fields')
    archive = index['archive']
    require(set(archive) == {'commit', 'bytes', 'sha256', 'logical_member_count', 'unique_object_count', 'part_count'}, 'archive fields')
    require(digest(archive['commit'], 40) and digest(archive['sha256'], 64), 'archive hashes')
    for key in ('bytes', 'logical_member_count', 'unique_object_count', 'part_count'):
        require(integer(archive[key], 1), 'archive count: ' + key)
    require(archive['part_count'] <= 128, 'bounded part count')
    require(archive['logical_member_count'] <= 10000, 'bounded member count')
    require(archive['unique_object_count'] <= archive['logical_member_count'], 'object/member counts')
    require(isinstance(index['files'], list), 'files type')
    rows = {}
    for row in index['files']:
        require(set(row) == {'path', 'commit', 'git_blob_sha1', 'sha256', 'bytes', 'url'}, 'file fields')
        path = row['path']
        require(safe_path(path) and path.startswith(PREFIX) and path not in rows, 'unsafe/duplicate file path')
        require(digest(row['commit'], 40) and digest(row['git_blob_sha1'], 40) and digest(row['sha256'], 64), 'file hashes')
        require(integer(row['bytes'], 1), 'file bytes')
        require(row['url'] == f'https://github.com/{REPO}/blob/{row["commit"]}/{path}', 'file URL binding')
        rows[path] = row
    expected = {PREFIX + name for name in ('ARCHIVE.json', 'CATALOG.json')}
    expected |= {PREFIX + f'objects.zip.part-{i:03d}' for i in range(archive['part_count'])}
    require(set(rows) == expected, 'exact metadata/contiguous part set')
    require(rows[PREFIX + 'ARCHIVE.json']['commit'] == archive['commit'], 'archive commit')
    parts = [rows[PREFIX + f'objects.zip.part-{i:03d}'] for i in range(archive['part_count'])]
    require(all(p['commit'] == archive['commit'] for p in parts), 'part commits')
    require(sum(p['bytes'] for p in parts) == archive['bytes'], 'part byte sum')
    runs = index['source_ci_runs']
    require(isinstance(runs, list) and bool(runs), 'CI runs')
    ids, identities = set(), set()
    for run in runs:
        require(set(run) == {'run_id', 'repository', 'workflow_id', 'path', 'event', 'run_attempt', 'head_sha', 'status', 'conclusion', 'url'}, 'CI fields')
        require(integer(run['run_id'], 1) and run['run_id'] not in ids, 'CI run ID')
        ids.add(run['run_id'])
        require(run['repository'] == REPO and run['head_sha'] == pr['head_commit'], 'CI exact head/repository')
        require(run['path'] in WORKFLOWS and run['workflow_id'] == WORKFLOWS[run['path']], 'CI workflow identity')
        require(run['event'] in ('push', 'pull_request') and integer(run['run_attempt'], 1), 'CI event/attempt')
        require(run['status'] == 'completed' and run['conclusion'] == 'success', 'CI success snapshot')
        require(run['url'] == f'https://github.com/{REPO}/actions/runs/{run["run_id"]}', 'CI URL')
        identity = (run['path'], run['event'])
        require(identity not in identities, 'duplicate CI workflow/event')
        identities.add(identity)
    require({(p, 'pull_request') for p in WORKFLOWS} <= identities, 'required PR workflows')
    return rows


def validate_metadata(index, metadata):
    rows = validate_index(index)
    require(set(metadata) == {'ARCHIVE.json', 'CATALOG.json'}, 'two metadata inputs required')
    require(sum(len(raw) for raw in metadata.values()) <= MAX_METADATA_BYTES, 'metadata budget')
    decoded = {}
    for name, raw in metadata.items():
        row = rows[PREFIX + name]
        require(len(raw) == row['bytes'], name + ' bytes')
        require(hashlib.sha256(raw).hexdigest() == row['sha256'], name + ' SHA256')
        blob = b'blob ' + str(len(raw)).encode() + b'\0' + raw
        require(hashlib.sha1(blob).hexdigest() == row['git_blob_sha1'], name + ' Git blob SHA1')
        decoded[name] = parse(raw)
    archive, catalog = decoded['ARCHIVE.json'], decoded['CATALOG.json']
    require(archive['schema'] == 'kws-public-research-archive-v1', 'archive schema')
    require(catalog['schema'] == 'kws-public-research-catalog-v1', 'catalog schema')
    require(catalog['pipeline_source_copy_status'] == CATALOG_STATUS, 'catalog publication status')
    for left, right in [('archive_bytes', 'bytes'), ('archive_sha256', 'sha256'), ('members', 'logical_member_count'), ('unique_objects', 'unique_object_count')]:
        require(archive[left] == index['archive'][right], 'archive identity/count: ' + left)
    require(len(archive['parts']) == index['archive']['part_count'], 'archive part count')
    for n, part in enumerate(archive['parts']):
        require(part['path'] == f'objects.zip.part-{n:03d}', 'archive part order')
        row = rows[PREFIX + part['path']]
        require(part['bytes'] == row['bytes'] and part['sha256'] == row['sha256'], 'archive part identity')
    members = catalog['members']
    require(len(members) == archive['members'], 'catalog member count')
    paths, objects, pipeline_paths = set(), {}, set()
    require(len(set(catalog['packages'])) == len(catalog['packages']), 'duplicate packages')
    for member in members:
        require(safe_path(member['path']) and member['path'] not in paths, 'catalog member path')
        paths.add(member['path'])
        require(member['source_package'] in catalog['packages'] and member['path'].startswith(member['source_package'] + '/'), 'catalog package')
        for field in ('original_sha256', 'published_sha256'):
            require(digest(member[field], 64), 'catalog hash')
        for field in ('original_bytes', 'published_bytes'):
            require(integer(member[field]), 'catalog bytes')
        obj = member['object']
        require(obj == 'objects/' + member['published_sha256'], 'catalog CAS binding')
        require(obj not in objects or objects[obj] == member['published_bytes'], 'CAS size conflict')
        objects[obj] = member['published_bytes']
        require('pipeline_path' not in member, 'NOT_PUSHED permits planned_pipeline_path only')
        # A planned mapping does not claim a file exists in pipeline.
        for field in ('pipeline_path', 'planned_pipeline_path'):
            if field not in member:
                continue
            path = member[field]
            require(safe_path(path) and path == 'research/host-research-2026-10-08/sources/' + member['path'] and path not in pipeline_paths, 'pipeline mapping')
            pipeline_paths.add(path)
    require(len(objects) == archive['unique_objects'], 'catalog unique objects')
    require(sum(m['original_bytes'] for m in members) == archive['original_bytes_total'], 'original byte total')
    require(sum(m['published_bytes'] for m in members) == archive['published_bytes_total'], 'published byte total')


def load_metadata(index, directory=None):
    rows = validate_index(index)
    result = {}
    require(sum(rows[PREFIX + n]['bytes'] for n in ('ARCHIVE.json', 'CATALOG.json')) <= MAX_METADATA_BYTES, 'combined metadata budget')
    for name in ('ARCHIVE.json', 'CATALOG.json'):
        row = rows[PREFIX + name]
        require(row['bytes'] <= MAX_METADATA_BYTES, 'metadata budget')
        if directory is not None:
            with (directory / name).open('rb') as stream:
                result[name] = stream.read(row['bytes'] + 1)
        else:
            # Construct from validated identity, never follow an index-supplied URL.
            url = f'https://raw.githubusercontent.com/{REPO}/{row["commit"]}/{row["path"]}'
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args, **kwargs):
                    raise ValueError('metadata redirects prohibited')
            with urllib.request.build_opener(NoRedirect).open(url, timeout=30) as stream:
                result[name] = stream.read(row['bytes'] + 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / INDEX)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--metadata-dir', type=Path)
    group.add_argument('--fetch-metadata', action='store_true', help='Explicit network access for two pinned public JSON files only')
    args = parser.parse_args()
    try:
        index = parse(args.index.read_bytes())
        validate_metadata(index, load_metadata(index, args.metadata_dir))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('FAIL publication metadata: ' + str(exc), file=sys.stderr)
        return 1
    print('PASS publication metadata identities and offline semantics; no live-status or acoustic claim')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
