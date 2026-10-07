"""Pure transport/target failure tests; never contacts a network or model."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from fetch_inputs import CAPS, NAMES, PREFIX, REPOSITORY, fetch, get_once, validate_source


def fixture():
    bodies = [b'opaque zip fixture', b'opaque freeze fixture']
    source = {'schema': 'melo6-public-data-input-source-v1', 'repository': REPOSITORY,
              'commit': 'a' * 40, 'files': [
                  {'name': name, 'path': PREFIX + name, 'bytes': len(body),
                   'sha256': hashlib.sha256(body).hexdigest()}
                  for name, body in zip(NAMES, bodies)]}
    return source, bodies


class InputFetchTests(unittest.TestCase):
    def test_wrong_targets_paths_shapes_and_hashes_fail(self):
        source, _ = fixture()
        mutations = []
        for key, values in {'repository': ['other/kws-data', 'jiying2007/kws-pipeline'],
                            'commit': [None, 'main', '0' * 40, 'A' * 40, 'a' * 39, '../main'],
                            'schema': ['other']}.items():
            for value in values:
                item = copy.deepcopy(source); item[key] = value; mutations.append(item)
        for key, values in {'name': ['other', '../blind-inputs.zip'], 'path': ['/absolute', PREFIX + '../blind-inputs.zip'],
                            'bytes': [0, -1, True, 3.0, CAPS[NAMES[0]] + 1],
                            'sha256': [None, 'z' * 64, 'a' * 63]}.items():
            for value in values:
                item = copy.deepcopy(source); item['files'][0][key] = value; mutations.append(item)
        for files in [[], source['files'] * 2, list(reversed(source['files']))]:
            item = copy.deepcopy(source); item['files'] = files; mutations.append(item)
        item = copy.deepcopy(source); item['extra'] = True; mutations.append(item)
        item = copy.deepcopy(source); item['files'][0]['extra'] = True; mutations.append(item)
        for item in mutations:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_source(item)

    def test_exact_two_gets_and_verified_bytes(self):
        source, bodies = fixture(); calls = []
        def download(path, row):
            calls.append(path); return bodies[len(calls) - 1]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'runtime/incoming'
            receipt = fetch(source, target, download)
            self.assertEqual(calls, ['/' + REPOSITORY + '/' + source['commit'] + '/' + PREFIX + name for name in NAMES])
            self.assertEqual({p.name for p in target.iterdir()}, set(NAMES))
            self.assertEqual([(target / name).read_bytes() for name in NAMES], bodies)
            self.assertEqual(receipt['get_requests'], 2)
            self.assertEqual(receipt['response_body_bytes'], sum(map(len, bodies)))
            with self.assertRaises(ValueError): fetch(source, target, download)
            self.assertEqual(len(calls), 2)

    def test_failed_or_tampered_get_leaves_no_handoff_and_has_no_retry(self):
        source, bodies = fixture()
        for bad in [b'', bodies[1][:-1], bodies[1] + b'x', b'x' * len(bodies[1]), None]:
            calls = []
            def download(path, row):
                calls.append(path)
                if len(calls) == 1: return bodies[0]
                if bad is None: raise OSError('fake network failure')
                return bad
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / 'runtime/incoming'
                with self.assertRaises((ValueError, OSError)): fetch(source, target, download)
                self.assertEqual(len(calls), 2); self.assertFalse(target.parent.exists())

    def test_fresh_destination_required_before_any_get(self):
        source, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'runtime').mkdir()
            for target in [root / 'runtime/incoming', root / 'wrong/incoming', root / 'runtime/other']:
                with self.assertRaises(ValueError):
                    fetch(source, target, lambda *_: self.fail('Unexpected GET'))

    def test_http_status_size_encoding_hash_tls_and_single_request(self):
        source, bodies = fixture(); row = source['files'][0]
        for status, length, encoding, body, valid in [
            (200, str(row['bytes']), None, bodies[0], True),
            (200, None, 'identity', bodies[0], True),
            (302, str(row['bytes']), None, bodies[0], False),
            (403, str(row['bytes']), None, bodies[0], False),
            (200, '999', None, bodies[0], False),
            (200, str(row['bytes']), 'gzip', bodies[0], False),
            (200, str(row['bytes']), None, bodies[0][:-1], False),
            (200, None, None, bodies[0] + b'x', False),
            (200, None, None, b'x' * len(bodies[0]), False)]:
            calls = []; closed = []
            class Response:
                def getheader(self, key):
                    return {'Content-Length': length, 'Content-Encoding': encoding}.get(key)
                def read(self, size):
                    self_outer.assertEqual(size, row['bytes'] + 1); return body[:size]
            response = Response(); response.status = status; self_outer = self
            class Connection:
                def __init__(self, host, timeout, context):
                    self_outer.assertEqual(host, 'raw.githubusercontent.com')
                    self_outer.assertEqual(timeout, 30)
                    self_outer.assertTrue(context.check_hostname)
                    self_outer.assertEqual(context.verify_mode, 2)
                def request(self, method, path, headers):
                    calls.append((method, path)); self_outer.assertNotIn('Authorization', headers)
                def getresponse(self): return response
                def close(self): closed.append(True)
            with self.subTest(status=status, length=length, encoding=encoding, valid=valid):
                if valid: self.assertEqual(get_once('/fixed', row, Connection), bodies[0])
                else:
                    with self.assertRaises(ValueError): get_once('/fixed', row, Connection)
                self.assertEqual(calls, [('GET', '/fixed')]); self.assertEqual(closed, [True])


if __name__ == '__main__': unittest.main()
