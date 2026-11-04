"""Adversarial invented bytes only; no model, filesystem dependency execution."""
from pathlib import Path
import ast
import copy
import json
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import frontend_verifier as v

def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class Tests(unittest.TestCase):
    def setUp(self):
        attrs=dict(fs=16000,window='hamming',n_mels=3,frame_length=25,frame_shift=10,filter_length_min=-1,
            filter_length_max=-1,lfr_m=1,lfr_n=1,dither=1.0,snip_edges=True,upsacle_samples=True)
        vals=[list(struct.unpack('<3f',struct.pack('<3f',.1,.2,.3))),[1.,1.,1.]]
        self.bodies={name:('invented '+name).encode() for name in v.DEPS}
        self.bodies['cmvn_file']=b'<AddShift> 3 3\n<LearnRateCoef> 1 [ 0.1 0.2 0.3 ]\n<Rescale> 3 3\n<LearnRateCoef> 1 [ 1 1 1 ]\n'
        self.snap=dict(schema='observed-sense-frontend-v1',frontend_class='funasr.frontends.wav_frontend.WavFrontend',
            attributes=attrs,attribute_types={k:type(x).__name__ for k,x in attrs.items()},
            attribute_source='actual_loaded_WavFrontend_attributes_no_defaults',dependencies={},
            native_execution_mappings=dict(backend_binary=True,backend_core_library=True),
            cmvn=dict(device='cpu',dtype='float32',shape=[2,3],encoding='ieee754-binary32-le',values=vals,
                sha256=v.sha(struct.pack('<6f',.1,.2,.3,1.,1.,1.))),observation_stage='loaded_object_observation',
            dither_rng_reproducibility='not established',training_admitted=False)
        self.bind=dict(schema='reviewed-sense-frontend-dependencies-v1',snapshot_sha256='',dependencies={})
        self.inventory=[]
        for name,b in self.bodies.items():
            identity=dict(bytes=len(b),sha256=v.sha(b));self.snap['dependencies'][name]=identity
            self.bind['dependencies'][name]=dict(identity,reference=dict(archive='reviewed',member='deps/'+name))
            self.inventory.append(dict(identity,path='deps/'+name))
    def verify(self):
        self.bind['snapshot_sha256']=v.sha(raw(self.snap))
        return v.verify_frontend_dependency(raw(self.snap),raw(self.bind),lambda ref:self.bodies[ref['member'].split('/')[-1]],self.inventory)
    def test_success_no_admission(self):self.assertFalse(self.verify()['training_admitted'])
    def test_extra_field(self):
        self.snap['unknown']='x'
        with self.assertRaises(ValueError):self.verify()
    def test_missing_field(self):
        del self.snap['cmvn']
        with self.assertRaises(ValueError):self.verify()
    def test_bool_is_not_integer(self):
        self.snap['attributes']['n_mels']=True;self.snap['attribute_types']['n_mels']='bool'
        with self.assertRaisesRegex(ValueError,'attribute type'):self.verify()
    def test_type_label_mismatch(self):
        self.snap['attribute_types']['dither']='int'
        with self.assertRaises(ValueError):self.verify()
    def test_cmvn_hash_wrong(self):
        self.snap['cmvn']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'tensor hash'):self.verify()
    def test_cmvn_shape_bool(self):
        self.snap['cmvn']['shape']=[True,3]
        with self.assertRaises(ValueError):self.verify()
    def test_cmvn_not_exact_float32(self):
        self.snap['cmvn']['values'][0][0]=.1
        with self.assertRaisesRegex(ValueError,'exact observed float32'):self.verify()
    def test_cmvn_values_cannot_be_integers(self):
        self.snap['cmvn']['values'][1][0]=1
        with self.assertRaises(ValueError):self.verify()
    def test_original_dependency_bytes_drift(self):
        self.bodies['tokenizer_file']=b'changed'
        with self.assertRaisesRegex(ValueError,'raw bytes changed'):self.verify()
    def test_reviewed_identity_mismatch(self):
        self.bind['dependencies']['parser_source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'reviewed dependency identity'):self.verify()
    def test_inventory_membership_missing(self):
        self.inventory=[]
        with self.assertRaises(ValueError):self.verify()
    def test_inventory_membership_ambiguous(self):
        self.inventory.append(copy.deepcopy(self.inventory[0]))
        with self.assertRaisesRegex(ValueError,'uniquely match'):self.verify()
    def test_no_snapshot_inventory_needed(self):
        self.assertFalse(any(r['sha256']==v.sha(raw(self.snap)) for r in self.inventory));self.verify()
    def test_nested_dependency_membership_required(self):
        self.inventory=self.inventory[:-1]
        with self.assertRaises(ValueError):self.verify()
    def test_path_traversal(self):
        self.bind['dependencies']['frontend_source']['reference']['member']='../evil'
        with self.assertRaisesRegex(ValueError,'safe exact'):self.verify()
    def test_self_reported_training_not_allowed(self):
        self.snap['training_admitted']=True
        with self.assertRaises(ValueError):self.verify()
    def test_unknown_or_missing_dependency(self):
        del self.bind['dependencies']['backend_binary']
        with self.assertRaises(ValueError):self.verify()
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):v.parse(b'{"a":1,"a":2}')
    def test_nonfinite_json(self):
        for b in (b'{"a":NaN}',b'{"a":Infinity}'):
            with self.assertRaises(ValueError):v.parse(b)
    def test_cmvn_file_tensor_disagreement(self):
        name='cmvn_file';self.bodies[name]=self.bodies[name].replace(b'0.1',b'0.9')
        identity=dict(bytes=len(self.bodies[name]),sha256=v.sha(self.bodies[name]))
        self.snap['dependencies'][name]=identity;self.bind['dependencies'][name].update(identity)
        for item in self.inventory:
            if item['path']=='deps/'+name:item.update(identity)
        with self.assertRaisesRegex(ValueError,'differs from original CMVN'):self.verify()
    def test_duplicate_cmvn_blocks(self):
        b=self.bodies['cmvn_file']*2
        with self.assertRaisesRegex(ValueError,'single CMVN block'):v.cmvn_from_file(b)
    def test_nan_cmvn_file(self):
        with self.assertRaises(ValueError):v.cmvn_from_file(self.bodies['cmvn_file'].replace(b'0.1',b'nan'))
    def test_bad_snapshot_binding_hash(self):
        self.bind['snapshot_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'reviewed snapshot hash'):v.verify_frontend_dependency(raw(self.snap),raw(self.bind),lambda r:b'',self.inventory)
    def test_python38_syntax(self):
        for name in ('frontend_verifier.py','test_frontend_verifier.py'):
            ast.parse((Path(__file__).parent/name).read_text(),feature_version=(3,8))

if __name__=='__main__':unittest.main()
