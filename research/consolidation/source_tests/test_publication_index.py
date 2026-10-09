"""Offline adversarial tests: invented catalog bytes, never remote access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('publication', ROOT / 'tools/verify_research_publication.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def fixture():
    index = json.loads((ROOT / v.INDEX).read_text())
    sha = hashlib.sha256(b'invented').hexdigest()
    member = dict(path='invented/file.txt', source_package='invented', original_sha256=sha,
                  original_bytes=8, published_sha256=sha, published_bytes=8,
                  transformations=[], object='objects/' + sha)
    cat = dict(schema='kws-public-research-catalog-v1', packages=['invented'], members=[member],
               pipeline_source_copy_status=v.CATALOG_STATUS)
    archive = dict(schema='kws-public-research-archive-v1', archive_bytes=12,
                   archive_sha256='a' * 64, parts=[dict(path='objects.zip.part-000', bytes=12, sha256='b' * 64)],
                   members=1, unique_objects=1, original_bytes_total=8, published_bytes_total=8)
    index['archive'].update(bytes=12, sha256='a' * 64, logical_member_count=1, unique_object_count=1, part_count=1)
    index['files'] = index['files'][:3]
    index['files'][2].update(bytes=12, sha256='b' * 64)
    metadata = {}
    for name, obj in [('ARCHIVE.json', archive), ('CATALOG.json', cat)]:
        metadata[name] = json.dumps(obj).encode()
    rebind(index, metadata)
    return index, metadata


def rebind(index, metadata):
    for name, raw in metadata.items():
        row = next(r for r in index['files'] if r['path'] == v.PREFIX + name)
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   git_blob_sha1=hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest())


class PublicationTests(unittest.TestCase):
    def test_valid_offline_fixture(self):
        v.validate_metadata(*fixture())

    def test_catalog_can_use_new_commit(self):
        index, metadata = fixture()
        row = index['files'][1]
        row['commit'] = 'd' * 40
        row['url'] = f'https://github.com/{v.REPO}/blob/{row["commit"]}/{row["path"]}'
        v.validate_metadata(index, metadata)

    def test_index_mutations(self):
        cases = [
            (('schema',), 'parse-only'), (('repository',), 'other/repo'),
            (('base_commit',), 'main'), (('expanded_pipeline_source_copy',), 'PUSHED'),
            (('publication_scope',), 'PRIVATE'), (('observed_at',), '2026-10-09'),
            (('date',), '2026-10-08'), (('snapshot_semantics',), 'live'),
            (('source_pull_request', 'head_commit'), 'main'),
            (('source_pull_request', 'merged'), True), (('source_pull_request', 'draft'), False),
            (('source_pull_request', 'url'), 'https://example.com'),
            (('archive', 'logical_member_count'), 2), (('archive', 'logical_member_count'), 10001), (('archive', 'unique_object_count'), True),
            (('archive', 'part_count'), 2), (('archive', 'part_count'), 10**12), (('archive', 'bytes'), 13),
            (('archive', 'sha256'), 'c' * 64),
            (('files', 0, 'path'), v.PREFIX + '../ARCHIVE.json'),
            (('files', 0, 'url'), 'https://example.com/a'),
            (('files', 0, 'bytes'), True), (('files', 0, 'sha256'), '0' * 64),
            (('files', 0, 'git_blob_sha1'), '0' * 40),
            (('files', 2, 'commit'), '0' * 40), (('files', 2, 'sha256'), '0' * 64),
            (('source_ci_runs', 0, 'head_sha'), '0' * 40),
            (('source_ci_runs', 0, 'repository'), 'other/repo'),
            (('source_ci_runs', 0, 'run_id'), True),
            (('source_ci_runs', 0, 'workflow_id'), 1),
            (('source_ci_runs', 0, 'path'), '.github/workflows/evil.yml'),
            (('source_ci_runs', 0, 'event'), 'workflow_dispatch'),
            (('source_ci_runs', 0, 'run_attempt'), 0),
            (('source_ci_runs', 0, 'conclusion'), 'failure'),
            (('source_ci_runs', 0, 'status'), 'queued'),
            (('source_ci_runs', 0, 'url'), 'https://example.com'),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                index, metadata = fixture()
                target = index
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    v.validate_metadata(index, metadata)

    def test_duplicate_missing_parts_and_ci(self):
        for field in ('files', 'source_ci_runs'):
            for operation in ('duplicate', 'remove'):
                with self.subTest(field=field, operation=operation):
                    index, metadata = fixture()
                    if operation == 'duplicate':
                        index[field].append(copy.deepcopy(index[field][0]))
                    else:
                        index[field].pop(0)
                    with self.assertRaises(ValueError):
                        v.validate_metadata(index, metadata)

    def test_metadata_semantics_even_with_recomputed_hashes(self):
        cases = [('ARCHIVE.json', 'members', 2), ('ARCHIVE.json', 'unique_objects', 2),
                 ('ARCHIVE.json', 'published_bytes_total', 7),
                 ('CATALOG.json', 'pipeline_source_copy_status', 'PUBLISHED'),
                 ('CATALOG.json', 'schema', 'wrong'), ('CATALOG.json', 'packages', ['wrong'])]
        for name, key, value in cases:
            with self.subTest(name=name, key=key):
                index, metadata = fixture()
                obj = json.loads(metadata[name])
                obj[key] = value
                metadata[name] = json.dumps(obj).encode()
                rebind(index, metadata)
                with self.assertRaises(ValueError):
                    v.validate_metadata(index, metadata)

    def test_catalog_member_tampering(self):
        cases = [('path', '../escape'), ('object', 'objects/' + '0' * 64),
                 ('published_bytes', -1), ('original_sha256', 'bad'),
                 ('source_package', 'unknown'), ('pipeline_path', '/tmp/escape')]
        for key, value in cases:
            with self.subTest(key=key):
                index, metadata = fixture()
                cat = json.loads(metadata['CATALOG.json'])
                cat['members'][0][key] = value
                metadata['CATALOG.json'] = json.dumps(cat).encode()
                rebind(index, metadata)
                with self.assertRaises(ValueError):
                    v.validate_metadata(index, metadata)

    def test_planned_mapping_is_not_a_published_path_claim(self):
        index, metadata = fixture()
        cat = json.loads(metadata['CATALOG.json'])
        member = cat['members'][0]
        member['planned_pipeline_path'] = 'research/host-research-2026-10-08/sources/' + member['path']
        metadata['CATALOG.json'] = json.dumps(cat).encode()
        rebind(index, metadata)
        v.validate_metadata(index, metadata)
        member['pipeline_path'] = member['planned_pipeline_path']
        metadata['CATALOG.json'] = json.dumps(cat).encode()
        rebind(index, metadata)
        with self.assertRaises(ValueError):
            v.validate_metadata(index, metadata)

    def test_raw_bytes_and_json_duplicates(self):
        index, metadata = fixture()
        metadata['CATALOG.json'] += b' '
        with self.assertRaises(ValueError):
            v.validate_metadata(index, metadata)
        with self.assertRaises(ValueError):
            v.parse('{"a": 1, "a": 2}')

    def test_metadata_budget(self):
        index, metadata = fixture()
        metadata['CATALOG.json'] = b'x' * (v.MAX_METADATA_BYTES + 1)
        with self.assertRaises(ValueError):
            v.validate_metadata(index, metadata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
