import copy
import unittest
from unittest import mock
import io
import json
import pathlib
import tempfile
import types
import subprocess
import admission
import contract

# Synthetic metadata only: these hashes do not identify real numerical inputs.
def fixture():
    p={'schema':'d20-saved-single-op-plan-v1','limits':dict(contract.LIMITS),
       'anchor':'C_SAVED_OBSERVED','input_chaining':False,'historical_chunk_equivalence':False,
       'original_status':{'D20':'FAIL','D90':'NOT_RUN'},'jobs':[],
       'frozen_sources':dict(contract.FROZEN_SOURCES), 'gates':copy.deepcopy(contract.GATES),
       'required_backend':{'torch':'2.11.0+cpu','numpy':'1.26.4'}}
    for r in range(19):
        for s in range(21):
            for b in ('C','Torch'):
                n=400 if s==0 else contract.DIMS[s-1]
                j=dict(row=r,stage=s,backend=b,input_origin='frozen_saved_pre_input',dtype='<f4',order='C',
                       input_shape=[n],output_shape=[contract.DIMS[s]],input_sha256='1'*64,input_bytes=n*4)
                if s in contract.MEMORY_STAGES:
                    j.update(cache_origin='C_SAVED_OBSERVED',cache_shape=[128,11,4],cache_bytes=22528,cache_sha256='2'*64)
                p['jobs'].append(j)
    return p

class ContractTests(unittest.TestCase):
    def test_valid_synthetic_metadata(self):
        result=contract.validate(json.loads(json.dumps(fixture())))
        self.assertEqual(result['planned_rows'],798)
        self.assertIs(result['execution_ready'],False)
        self.assertIs(result['input_authenticity_verified'],False)
    def test_json_integer_metadata_rejects_equal_floats_and_booleans(self):
        # JSON round trips distinguish integers from numerically equal floats.
        # No saved input, archive, backend or operator is read or executed.
        for key in ('input_shape', 'output_shape', 'cache_shape'):
            job = 8 if key == 'cache_shape' else 0
            for dimension in range(len(fixture()['jobs'][job][key])):
                for convert in (float, bool):
                    plan = fixture()
                    shape = plan['jobs'][job][key]
                    shape[dimension] = convert(shape[dimension])
                    with self.subTest(key=key, dimension=dimension, type=convert.__name__):
                        with self.assertRaises(ValueError):
                            contract.validate(json.loads(json.dumps(plan)))
        for key, job in (('input_bytes', 0), ('cache_bytes', 8), ('row', 0), ('stage', 0)):
            for convert in (float, bool):
                plan = fixture()
                plan['jobs'][job][key] = convert(plan['jobs'][job][key])
                with self.subTest(key=key, type=convert.__name__):
                    with self.assertRaises(ValueError):
                        contract.validate(json.loads(json.dumps(plan)))
        for key in contract.LIMITS:
            for convert in (float, bool):
                plan = fixture()
                plan['limits'][key] = convert(plan['limits'][key])
                with self.subTest(limit=key, type=convert.__name__):
                    with self.assertRaises(ValueError):
                        contract.validate(json.loads(json.dumps(plan)))

    def test_json_gates_require_finite_numbers_without_changing_tolerances(self):
        for name, pair in contract.GATES.items():
            for index in range(len(pair)):
                for value in (False, True, '0', '0.0', None, float('nan'), float('inf'), -float('inf')):
                    plan = fixture()
                    plan['gates'][name][index] = value
                    with self.subTest(gate=name, index=index, value=value):
                        with self.assertRaises(ValueError):
                            contract.validate(json.loads(json.dumps(plan)))
        # Integer and float zero remain equivalent numeric thresholds.
        for zero in (0, 0.0):
            plan = fixture()
            for pair in plan['gates'].values():
                if pair[1] == 0:
                    pair[1] = zero
            result = contract.validate(json.loads(json.dumps(plan)))
            self.assertIs(result['execution_ready'], False)
            self.assertIs(result['numerical_admission'], False)

    def test_reject_invalid(self):
        cases=[lambda p:p['frozen_sources'].update(weights='4'*64),
               lambda p:p['gates'].update(raw=[1e-3,1e-5]),
               lambda p:p['required_backend'].update(torch='2.11.0'),
               lambda p:p['limits'].update(operator_rows=1596),
               lambda p:p.update(input_chaining=True),lambda p:p.update(anchor='Torch_RECONSTRUCTED'),
               lambda p:p.update(historical_chunk_equivalence=True),
               lambda p:p['jobs'].pop(),lambda p:p['jobs'].__setitem__(1,copy.deepcopy(p['jobs'][0])),
               lambda p:p['jobs'][1].update(input_sha256='3'*64),
               lambda p:p['jobs'][8].update(cache_origin='Torch_RECONSTRUCTED'),
               lambda p:p['jobs'][8].update(cache_shape=[11,128,4]),
               lambda p:p['jobs'][0].update(input_shape=[140]),
               lambda p:p['jobs'][0].update(dtype='float64'),lambda p:p['jobs'][0].update(input_bytes=4),
               lambda p:p['jobs'][0].update(input_sha256='x'*64),
               lambda p:p['original_status'].update(D20='PASS')]
        for mutate in cases:
            p=fixture(); mutate(p)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):contract.validate(p)

class InventorySafetyTests(unittest.TestCase):
    def test_bounded_reads(self):
        with mock.patch('builtins.open',return_value=io.BytesIO(b'x'*(admission.MAX_READ_BYTES+1))):
            self.assertIsNone(admission.read('/proc/self/cgroup'))
        self.assertIsNone(admission.read('/private/not-allowlisted'))
    def test_sanitized_versions_and_limits(self):
        for value in ('/private/file', 'bad\nversion', 'x'*65, None):
            self.assertIsNone(admission.version(value))
        self.assertEqual(admission.version('2.11.0+cpu'),'2.11.0+cpu')
        self.assertEqual(admission.memory_limit('536870912\n'),536870912)
        self.assertIsNone(admission.memory_limit('/private/file'))
        self.assertIsNone(admission.memory_limit('9'*20))
    def test_child_exception_redacted(self):
        with mock.patch('admission.subprocess.run',side_effect=OSError('/private/secret')):
            result=admission.cpu_probe()
        self.assertEqual(result['status'],'probe_unavailable')
        self.assertNotIn('private',json.dumps(result))
    def test_structured_cgroups_only(self):
        def fake_read(path):
            return {'/proc/self/mountinfo':'private/secret - cgroup2 private/host',
                    '/proc/self/cgroup':'0::/private/group',
                    '/sys/fs/cgroup/memory.max':'536870912'}.get(path)
        with mock.patch('admission.read',side_effect=fake_read), \
             mock.patch('admission.cpu_probe',return_value={'status':'probe_failed'}), \
             mock.patch('admission.importlib.metadata.Distribution.discover',return_value=iter(())):
            result=admission.inspect()
        self.assertNotIn('private',json.dumps(result))
        self.assertEqual(result['cgroup_capabilities']['mount_count'],1)
        self.assertIs(result['execution_ready'],False)

class DistributionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.directory = self.root / 'torch-2.11.0.dist-info'
        self.directory.mkdir()
        self.file = self.directory / 'METADATA'
        self.good = b'Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0+cpu\n'
        self.file.write_bytes(self.good)
        patch = mock.patch.object(admission.site, 'getsitepackages', return_value=[str(self.root)])
        patch.start()
        self.addCleanup(patch.stop)

    def test_real_metadata_without_package_pth_or_customization_import(self):
        for name in ('torch.py', 'sitecustomize.py'):
            (self.root / name).write_text("raise AssertionError('must not execute')\n")
        (self.root / 'bad.pth').write_text('import must_not_execute_pth\n')
        before = set(admission.sys.modules)
        self.assertEqual(admission.metadata_version('torch'), '2.11.0+cpu')
        self.assertEqual(set(admission.sys.modules) - before, set())
        with self.assertRaises(admission.importlib.metadata.PackageNotFoundError):
            admission.metadata_version('numpy')

    def test_exact_byte_boundary_and_actual_read_size(self):
        self.file.write_bytes(self.good + b'\n' + b'x' * (admission.MAX_READ_BYTES - len(self.good) - 1))
        original = admission.os.fdopen
        reads = []
        def wrapped(*args, **kwargs):
            source = original(*args, **kwargs)
            proxy = mock.MagicMock(wraps=source)
            proxy.__enter__.return_value = proxy
            proxy.__exit__.side_effect = source.__exit__
            proxy.read.side_effect = lambda size: (reads.append(size), source.read(size))[1]
            return proxy
        with mock.patch.object(admission.os, 'fdopen', side_effect=wrapped):
            self.assertEqual(admission.metadata_version('torch'), '2.11.0+cpu')
        self.assertEqual(reads, [admission.MAX_READ_BYTES + 1])
        self.file.write_bytes(self.file.read_bytes() + b'x')
        with self.assertRaises(ValueError):
            admission.metadata_version('torch')

    def test_byte_cap_does_not_trust_stat_size_hint(self):
        self.file.write_bytes(self.good + b'\n' + b'x'*admission.MAX_READ_BYTES)
        understated = types.SimpleNamespace(st_mode=admission.stat.S_IFREG, st_size=1)
        with mock.patch.object(admission.os, 'fstat', return_value=understated):
            with self.assertRaisesRegex(ValueError, 'exceeds byte cap'):
                admission.metadata_version('torch')

    def test_nonregular_metadata_and_unsupported_finder_fail_closed(self):
        self.file.unlink()
        self.file.mkdir()
        with self.assertRaises(ValueError):
            admission.metadata_version('torch')
        for candidate in (object(), admission.importlib.metadata.PathDistribution(self.root / 'torch.egg-info'),
                          admission.importlib.metadata.PathDistribution(str(self.directory))):
            with mock.patch.object(admission.importlib.metadata.Distribution, 'discover',
                                   return_value=iter([candidate])):
                with self.assertRaises(ValueError):
                    admission.metadata_version('torch')

    def test_duplicate_installs_fail_without_reading_version(self):
        extra = self.root / 'torch-9.0.dist-info'
        extra.mkdir()
        (extra / 'METADATA').write_bytes(self.good)
        with mock.patch.object(admission.importlib.metadata.PathDistribution, 'read_text',
                               side_effect=AssertionError('unbounded API forbidden')):
            with self.assertRaises(ValueError):
                admission.metadata_version('torch')

    def test_invalid_metadata_fail_closed(self):
        cases = [self.good + b'Version: 2.11.0+cpu\n',
                 self.good + b'Name: torch\n',
                 self.good + b'Metadata-Version: 2.1\n',
                 self.good.replace(b'Name: torch', b'Name: numpy'),
                 self.good.replace(b'Version: 2.11.0+cpu\n', b''),
                 self.good.replace(b'2.11.0+cpu', b'2.11.0\n +cpu'),
                 self.good.replace(b'2.11.0+cpu', b'/private/secret'),
                 self.good.replace(b'2.11.0+cpu', b'x'*65),
                 self.good + b'\n\xff',
                 b'\xef\xbb\xbf' + self.good,
                 b'malformed header\n' + self.good]
        for raw in cases:
            with self.subTest(raw=raw):
                self.file.write_bytes(raw)
                with self.assertRaises((ValueError, UnicodeError, admission.email.errors.MessageError)):
                    admission.metadata_version('torch')
                with mock.patch.object(admission, 'read', return_value=None), \
                     mock.patch.object(admission, 'cpu_probe', return_value={'status':'probe_failed'}):
                    result = admission.inspect()
                self.assertIsNone(result['installed_distribution_metadata']['torch'])
                self.assertIs(result['execution_ready'], False)
                self.assertIn('torch_metadata_missing_mismatched_or_unreadable', result['blockers'])
                self.assertNotIn('private', json.dumps(result))

    def test_symlink_file_and_directory_rejected(self):
        saved = self.root / 'unrelated-file'
        self.file.rename(saved)
        self.file.symlink_to(saved)
        with self.assertRaises(OSError):
            admission.metadata_version('torch')
        self.file.unlink()
        self.file.write_bytes(self.good)
        outside = self.root / 'elsewhere'
        self.directory.rename(outside)
        self.directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            admission.metadata_version('torch')

    def test_unknown_name_never_discovers_or_reads(self):
        with mock.patch.object(admission.importlib.metadata.Distribution, 'discover') as discover:
            with self.assertRaises(ValueError):
                admission.metadata_version('/private/secret')
        discover.assert_not_called()

    def test_matching_metadata_never_grants_execution(self):
        numpy = self.root / 'numpy-1.26.4.dist-info'
        numpy.mkdir()
        (numpy / 'METADATA').write_bytes(b'Metadata-Version: 2.1\nName: numpy\nVersion: 1.26.4\n')
        with mock.patch.object(admission, 'read', return_value=None), \
             mock.patch.object(admission, 'cpu_probe', return_value={'status':'probe_failed'}):
            result = admission.inspect()
        self.assertEqual(result['installed_distribution_metadata'], admission.REQUIRED)
        self.assertIs(result['execution_ready'], False)
        self.assertEqual(result['numerical_imports'], 0)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['operator_calls'], 0)
        self.assertIn('numerical_authorization_required', result['blockers'])

    def test_public_unbounded_metadata_apis_are_unused(self):
        with mock.patch.object(admission.importlib.metadata.PathDistribution, 'read_text',
                               side_effect=AssertionError('unbounded API forbidden')):
            self.assertEqual(admission.metadata_version('torch'), '2.11.0+cpu')


if __name__=='__main__':unittest.main()
