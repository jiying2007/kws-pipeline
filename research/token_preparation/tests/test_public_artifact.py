"""Pure invented fixtures only: no real feature/model/audio reads or execution."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import sys
_SOURCE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_SOURCE_ROOT/'src'))
sys.path.insert(0,str(_SOURCE_ROOT/'src/deps'))

import public_artifact as gate

PUBLIC_URL = 'https://github.com/example/public-research/blob/' + 'a'*40 + '/NOTICE'
WHEEL_URL = 'https://download-r2.pytorch.org/whl/cpu/torch-2.12.1%2Bcpu-test.whl'


def source_fixture(root):
    """Invented source-tree metadata; copied names/geometry are public identities."""
    package = root / gate.PACKAGE_RELATIVE
    metadata = package / 'metadata'; metadata.mkdir(parents=True)
    train = {'development_audio_included':False, 'rows':[
        {'recording':name, 'model_rows':rows, 'role':'train'} for name,rows in gate.GEOMETRY]}
    rights_raws = {name: gate.canonical({'fixture':'Invented public notice.'}) if name.endswith('.json')
                    else b'Invented public notice.\n' for name in gate.RIGHTS_BASENAMES}
    rights = {'source_metadata':[{'file':'rights-metadata/'+name, 'sha256':gate.digest(raw),
                                  'bytes':len(raw), 'source_url':PUBLIC_URL}
                                 for name,raw in sorted(rights_raws.items())],
              'required_output_notices':['Research only; commercial_output_license=not-established.',
                                         'Preserve original public source notices; no new grant.']}
    files = {'metadata/TRAIN32.json': gate.canonical(train),
             'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json':gate.canonical(rights),
             'src/deps/exact-wheels.lock.json':gate.canonical({'url':WHEEL_URL}),
             'src/fixture.py':b'# Invented non-executable fixture source.\n'}
    files.update({'metadata/rights/'+name:raw for name,raw in rights_raws.items()})
    entries = []
    for relative, raw in files.items():
        p = package / relative; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
        entries.append({'path':gate.PACKAGE_RELATIVE + '/' + relative,
                        'bytes':len(raw), 'sha256':gate.digest(raw)})
    workflow = root/'.github/workflows/preparation.yml'; workflow.parent.mkdir(parents=True)
    workflow.write_bytes(b'# Invented workflow; no executable steps.\n')
    entries.append({'path':'.github/workflows/preparation.yml', 'bytes':workflow.stat().st_size,
                    'sha256':gate.digest(workflow.read_bytes())})
    raw = gate.canonical({'schema':'a20-execution-source-freeze-v1','files':entries})
    (package/'SOURCE-FREEZE.json').write_bytes(raw)
    context = {'source_root':root, 'source_freeze_sha256':gate.digest(raw),
               'release_sha256':'c'*64,'rights_privacy_reviewed':True}
    return context


def payload_fixture(source, success=True):
    """All-zero FP32 is invented, never calculated from a waveform or model."""
    status = 'SUCCESS' if success else 'FAILED'
    docs = {name:{'schema':'invented-pure-test-fixture'} for name in gate.JSON_NAMES} if success else {}
    docs['source-identities.json'] = dict(source['identity'])
    docs['rights-and-provenance.json'] = {
        'commercial_output_license':'not-established',
        'distribution_scope':'bounded-public-research-preparation-only', 'new_license_grant':False,
        'source_provenance_sha256':source['identity']['rights_provenance_sha256'],
        'source_metadata':source['rights']['source_metadata'],
        'required_output_notices':source['rights']['required_output_notices']}
    docs['final-status.json'] = {'status':status, 'training_authorized':False, 'training_update_count':0,
        'native_feature_count':32 if success else 0, 'official_feature_count':20 if success else 0,
        'control_logits_count':1 if success else 0,'model_forward_count':1 if success else None,
        'phases':{p:'SUCCEEDED' if success else 'UNKNOWN' for p in gate.PHASES}}
    if not success:
        docs['final-status.json']['failure_code'] = 'INVENTED_TEST_FAILURE'
        return {name:gate.canonical(doc) for name,doc in docs.items()}
    raws = {name:b'\x00'*(4*gate.math.prod(shape)) for name,shape in gate.RAW_SHAPES.items()}
    for group in ('native','official'):
        docs[group + '-features.json'].update(frontend_passes=32 if group=='native' else 20, model_calls=0)
        docs[group + '-features.json']['rows'] = [
            {'recording':Path(name).stem,'file':name,'shape':gate.RAW_SHAPES[name],
             'bytes':len(raw),'sha256':gate.digest(raw)}
            for name,raw in raws.items() if name.startswith(group+'/')]
    docs['initial-control.json'] = {'raw_file':'a20-initial-logits.f32le','raw_shape':[32,95,6],
        'raw_sha256':gate.digest(raws['a20-initial-logits.f32le']), 'model_forwards':1,'cmvn_calls':1,
        'optimizer_constructions':0,'backwards':0,'optimizer_steps':0,'dropout_calls':0}
    docs['preparation-result.json'] = {'native_waveform_passes':32,'official_waveform_passes':20,
        'model_forwards':1,'optimizer_constructions':0,'backwards':0,'updates':0,'decoder_calls':0,
        'development_calls':0,'F_arm_calls':0,'training_authorized':False,
        'source_freeze_sha256':source['identity']['source_freeze_sha256']}
    docs['synthetic-ctc.json'] = {'status':'PASS','model_calls':0}
    docs['dependencies.json'] = {'url':WHEEL_URL}
    return dict(raws, **{name:gate.canonical(doc) for name,doc in docs.items()})


def change_json(blobs, name, action):
    result = dict(blobs); obj = gate.parse_json(result[name]); action(obj)
    result[name] = gate.canonical(obj)
    return result


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.context = source_fixture(self.base/'source')
        self.source = gate.validate_sourcecontext(self.context)
        self.failed = payload_fixture(self.source, False)

    def tearDown(self):
        self.temp.cleanup()

    def assertBlocked(self, blobs, status='FAILED', code=None):
        with self.assertRaises(gate.GateError) as cm:
            gate.build_archive(blobs,status,self.source)
        if code:
            self.assertEqual(str(cm.exception),code)

    def write_input(self, blobs):
        root = self.base/'input'; root.mkdir()
        for name,raw in blobs.items():
            path = root/name; path.parent.mkdir(exist_ok=True); path.write_bytes(raw)
        return root

    def test_geometry_exact(self):
        self.assertEqual(sum(gate.math.prod(s)*4 for n,s in gate.RAW_SHAPES.items() if n.startswith('native/')),2899200)
        self.assertEqual(sum(gate.math.prod(s)*4 for n,s in gate.RAW_SHAPES.items() if n.startswith('official/')),1627200)
        self.assertEqual(gate.math.prod(gate.RAW_SHAPES['a20-initial-logits.f32le'])*4,72960)
        self.assertEqual(len(gate.RAW_SHAPES),53)

    def test_deterministic_success_and_readback(self):
        blobs = payload_fixture(self.source)
        first = gate.build_archive(blobs,'SUCCESS',self.source)
        second = gate.build_archive(dict(reversed(list(blobs.items()))),'SUCCESS',self.source)
        self.assertEqual(first[:2],second[:2])
        packed,receipt,manifest,tar_size = first
        self.assertIn(gate.digest(packed).encode(),receipt)
        self.assertLessEqual(tar_size,gate.TAR_CAP)
        self.assertFalse(manifest['missing_raw_members'])
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed)),mode='r:') as archive:
            self.assertEqual(len(archive.getmembers()),68)
            for info in archive:
                self.assertTrue(info.isfile()); self.assertEqual(info.mtime,0)
                self.assertEqual(info.uname,''); self.assertEqual(info.gname,'')
            archived_manifest=json.load(archive.extractfile(gate.MANIFEST_NAME))
            for member in archived_manifest['members']:
                raw=archive.extractfile(member['path']).read()
                self.assertEqual(len(raw),member['bytes'])
                self.assertEqual(gate.digest(raw),member['sha256'])

    def test_minimal_failure_explicit_unknown(self):
        _,_,manifest,_ = gate.build_archive(self.failed,'FAILED',self.source)
        self.assertEqual(len(manifest['missing_raw_members']),53)
        self.assertEqual(manifest['artifact_counts']['native_feature_count'],0)
        self.assertIn('never proof',manifest['missing_means'])

    def test_partial_failure_has_exact_declared_member(self):
        name = 'native/'+gate.GEOMETRY[0][0]+'.f32le'; raw = b'\x00'*(gate.GEOMETRY[0][1]*1600)
        blobs=dict(self.failed);blobs[name]=raw
        blobs['native-features.json']=gate.canonical({'rows':[{'file':name,'recording':gate.GEOMETRY[0][0],
            'shape':gate.RAW_SHAPES[name],'bytes':len(raw),'sha256':gate.digest(raw)}]})
        blobs=change_json(blobs,'final-status.json',lambda obj:obj.update(native_feature_count=1))
        gate.build_archive(blobs,'FAILED',self.source)

    def test_success_missing_feature_rejected(self):
        blobs=payload_fixture(self.source);del blobs[next(n for n in blobs if n.startswith('native/'))]
        self.assertBlocked(blobs,'SUCCESS','required_raw_missing')

    def test_unknown_member_and_manifest_injection(self):
        for name in ('raw.wav','checkpoint.pt','wheel.whl','stdout.txt',gate.MANIFEST_NAME,
                     'native/unexpected.f32le','official/qwen3-kw1-dylan.f32le'):
            with self.subTest(name=name):
                self.assertBlocked(dict(self.failed,**{name:b'bad'}),code='unknown_member')

    def test_symlink_and_unknown_directory(self):
        root=self.write_input(self.failed);(root/'secret').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(gate.GateError,'symlink_member'):
            gate.package_artifact(root,self.base/'output','FAILED',self.context)
        (root/'secret').unlink();(root/'wheel-cache').mkdir()
        with self.assertRaisesRegex(gate.GateError,'unknown_directory'):
            gate.package_artifact(root,self.base/'output','FAILED',self.context)

    def test_hardlink_rejected(self):
        root=self.write_input(self.failed)
        os.link(root/'final-status.json',self.base/'duplicate')
        with self.assertRaisesRegex(gate.GateError,'nonregular_or_hardlinked_member'):
            gate.package_artifact(root,self.base/'output','FAILED',self.context)

    def test_wrong_byte_length(self):
        blobs=payload_fixture(self.source);name=next(iter(gate.RAW_SHAPES));blobs[name]=blobs[name][:-4]
        self.assertBlocked(blobs,'SUCCESS','raw_shape_or_length_mismatch')

    def test_nonfinite_fp32(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(value=value):
                blobs=payload_fixture(self.source);name=next(iter(gate.RAW_SHAPES))
                blobs[name]=struct.pack('<f',value)+blobs[name][4:]
                self.assertBlocked(blobs,'SUCCESS','nonfinite_fp32')

    def test_raw_evidence_hash_and_shape_rejected(self):
        for patch in ({'sha256':'0'*64},{'shape':[1,400]},{'bytes':0}):
            blobs=change_json(payload_fixture(self.source),'native-features.json',lambda obj:obj['rows'][0].update(patch))
            self.assertBlocked(blobs,'SUCCESS','feature_evidence_identity')

    def test_raw_without_evidence(self):
        name=next(iter(gate.RAW_SHAPES));blobs=dict(self.failed);blobs[name]=b'\x00'*(4*gate.math.prod(gate.RAW_SHAPES[name]))
        self.assertBlocked(blobs,code='raw_without_feature_evidence')

    def test_control_wrong_shape_and_hash(self):
        blobs=change_json(payload_fixture(self.source),'initial-control.json',lambda obj:obj.update(raw_shape=[95,32,6]))
        self.assertBlocked(blobs,'SUCCESS','control_evidence_identity')

    def test_false_success_and_counts_rejected(self):
        for patch in ({'status':'SUCCESS'},{'native_feature_count':1},{'model_forward_count':False},
                      {'training_update_count':1},{'phases':{}},{'failure_code':'Exception /home/runner'}):
            with self.subTest(patch=patch):
                blobs=change_json(self.failed,'final-status.json',lambda obj:obj.update(patch))
                self.assertBlocked(blobs)

    def test_failed_omitted_model_count_not_invented(self):
        self.assertBlocked(change_json(self.failed,'final-status.json',lambda obj:obj.pop('model_forward_count')),
                           code='model_forward_count_required')

    def test_source_pin_or_file_drift(self):
        bad=dict(self.context,source_freeze_sha256='0'*64)
        with self.assertRaisesRegex(gate.GateError,'source_freeze_pin_mismatch'):
            gate.validate_sourcecontext(bad)
        path=self.context['source_root']/gate.PACKAGE_RELATIVE/'src/fixture.py'
        path.write_bytes(path.read_bytes().replace(b'Invented',b'Altered!'))
        with self.assertRaisesRegex(gate.GateError,'source_entry_mismatch'):
            gate.validate_sourcecontext(self.context)

    def test_review_required(self):
        for value in (False,None,'true'):
            with self.assertRaisesRegex(gate.GateError,'rights_privacy_review_required'):
                gate.validate_sourcecontext(dict(self.context,rights_privacy_reviewed=value))

    def test_source_identity_wrong(self):
        self.assertBlocked(change_json(self.failed,'source-identities.json',lambda obj:obj.update(train32_sha256='0'*64)),
                           code='source_identity_mismatch')

    def test_rights_cannot_expand(self):
        for patch in ({'commercial_output_license':'Apache-2.0'},{'new_license_grant':True},
                      {'required_output_notices':[]},{'distribution_scope':'commercial'}):
            blobs=change_json(self.failed,'rights-and-provenance.json',lambda obj:obj.update(patch))
            self.assertBlocked(blobs,code='rights_scope_or_provenance_mismatch')

    def test_privacy_recursive_denials(self):
        bad_values=['/home/runner/.secret','/workspace/private/path','C:\\Users\\Alice\\file',
            'file:///tmp/private','https://chatgpt.com/private','https://github.com/example/private?token=bad',
            'http://github.com/example/file','https://github.com.evil.test/file',
            'ghp_'+'a'*40,'sk-'+'a'*30,'Bearer ' + 'z'*30,'secret=abcde',
            'alice@example.com','127.0.0.1','runner.internal',
            '12345678-1234-1234-1234-123456789abc','a'*120,'%2Fhome%2Frunner%2Fhidden']
        for value in bad_values:
            with self.subTest(value=value[:30]):
                self.assertBlocked(change_json(self.failed,'final-status.json',lambda obj:obj.update(details=[{'value':value}])))
        for key in ('hostname','boot_id','environment_variables','api_key','access-token','stdout','stderr'):
            self.assertBlocked(change_json(self.failed,'final-status.json',lambda obj:obj.update({key:'hidden'})))

    def test_public_urls_only_frozen(self):
        gate.privacy_check({'source':PUBLIC_URL,'wheel':WHEEL_URL,'state_sha256':'a'*64},self.source['allowed_urls'])
        with self.assertRaisesRegex(gate.GateError,'unapproved_url'):
            gate.privacy_check({'source':'https://github.com/example/new'},self.source['allowed_urls'])

    def test_json_duplicate_nan_huge_and_deep(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e10000}'):
            with self.subTest(raw=raw):
                blobs=dict(self.failed);blobs['final-status.json']=raw;self.assertBlocked(blobs)
        blobs=dict(self.failed);blobs['resources.json']=b' '*(gate.MIB+1)
        self.assertBlocked(blobs,code='json_size_cap')
        blobs=dict(self.failed);blobs['dependencies.json']=b' '*(gate.MIB//2+1)
        self.assertBlocked(blobs,code='json_size_cap')
        deep={}; obj=deep
        for _ in range(26): obj['child']={};obj=obj['child']
        with self.assertRaisesRegex(gate.GateError,'json_complexity_cap'):
            gate.privacy_check(deep)

    def test_actual_payload_tar_and_total_caps(self):
        # Small invented fixtures, lowered internal constants exercise final actual-size caps.
        for name,cap,error in (('PAYLOAD_CAP',1,'payload_cap'),('TAR_CAP',1,'tar_cap'),
                                ('GZIP_CAP',1,'publication_total_cap')):
            with self.subTest(name=name), mock.patch.object(gate,name,cap):
                self.assertBlocked(self.failed,code=error)

    def test_package_writes_only_pair_and_readback(self):
        root=self.write_input(self.failed); output=self.base/'output'
        result=gate.package_artifact(root,output,'FAILED',self.context)
        self.assertEqual(sorted(p.name for p in output.iterdir()),sorted([gate.ARCHIVE_NAME,gate.RECEIPT_NAME]))
        self.assertEqual(result['sha256'],gate.digest((output/gate.ARCHIVE_NAME).read_bytes()))
        self.assertFalse(result['upload_performed']);self.assertEqual(result['retention_days'],1)
        with self.assertRaisesRegex(gate.GateError,'output_directory_not_empty'):
            gate.package_artifact(root,output,'FAILED',self.context)

    def test_package_failure_writes_nothing(self):
        root=self.write_input(dict(self.failed,**{'password.txt':b'no'}));output=self.base/'output'
        with self.assertRaises(gate.GateError):gate.package_artifact(root,output,'FAILED',self.context)
        self.assertFalse(output.exists())

    def test_output_nested_in_input_rejected(self):
        root=self.write_input(self.failed)
        with self.assertRaisesRegex(gate.GateError,'output_directory_overlap'):
            gate.package_artifact(root,root/'publicationout','FAILED',self.context)

    def test_source_member_scope_is_closed(self):
        for name in ('research/token_preparation/work/raw.json', 'research/token_preparation/src/model.pt',
                     'research/token_preparation/metadata/audio.wav', '../private.json',
                     'other-repository/private.json', '.github/workflows/nested/file.yml'):
            with self.subTest(name=name):
                try: accepted=gate._source_member(name)
                except gate.GateError: accepted=False
                self.assertFalse(accepted)
        for name in ('research/token_preparation/tests/test_gate.py',
                     'research/token_preparation/src/deps/requirements.proposed.lock',
                     '.github/workflows/prepare.yml'):
            self.assertTrue(gate._source_member(name))

    def test_missing_success_json_and_bad_status(self):
        blobs=payload_fixture(self.source);del blobs['resources.json']
        self.assertBlocked(blobs,'SUCCESS','required_evidence_missing')
        self.assertBlocked(self.failed,'COMPLETE','invalid_status')

    def test_integer_shapes_and_execution_counts(self):
        blobs=change_json(payload_fixture(self.source),'initial-control.json',lambda obj:obj.update(raw_shape=[32.0,95,6]))
        self.assertBlocked(blobs,'SUCCESS','control_evidence_identity')
        blobs=change_json(payload_fixture(self.source),'native-features.json',lambda obj:obj.update(frontend_passes=31))
        self.assertBlocked(blobs,'SUCCESS','feature_execution_counts_mismatch')
        blobs=change_json(payload_fixture(self.source),'preparation-result.json',lambda obj:obj.update(updates=1))
        self.assertBlocked(blobs,'SUCCESS','preparation_scope_mismatch')

    def test_cli_default_noexec(self):
        result=subprocess.run([sys.executable,str(Path(gate.__file__))],check=True,capture_output=True,text=True)
        doc=json.loads(result.stdout)
        self.assertEqual(doc['status'],'PREPARATION_ONLY_NO_EXECUTION')
        self.assertFalse(doc['package_performed']);self.assertFalse(doc['upload_performed'])


if __name__=='__main__':unittest.main()
