import copy
import unittest
from unittest import mock
import io
import json
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
        result=contract.validate(fixture())
        self.assertEqual(result['planned_rows'],798)
        self.assertIs(result['execution_ready'],False)
        self.assertIs(result['input_authenticity_verified'],False)
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

if __name__=='__main__':unittest.main()
