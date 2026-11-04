"""Invented bytes, real retained dataclass structure, no framework imports."""
from pathlib import Path
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import sense_capture as c
spec=importlib.util.spec_from_file_location('old_capture_fixtures',REPO/'research/sense-native-capture-20261104/test_sense_capture.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
ADAPTER=REPO/'research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/research/qwen6_asr/core/asr_stage/adapters.py'
node=next(n for n in ast.parse(ADAPTER.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PreparedRuntime')
ns={'dataclass':dataclass};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ADAPTER),'exec'),ns)
PreparedRuntime=ns['PreparedRuntime']

class AdaptedTests(fixtures.Tests):
    def setUp(self):
        super().setUp()
        old=self.paths['tokenizer_file'];target=self.root/c.TOKENIZER_NAME;old.rename(target);self.paths['tokenizer_file']=target
        self.tokenizer.bpemodel=str(target)
        identity=dict(self.reviewed['tokenizer_file'])
        receipt={'assets':{'model_id':c.SENSE_MODEL_ID,'revision':c.SENSE_REVISION,
                          'verified_files':[dict(filename=c.TOKENIZER_NAME,**identity)]},
                 'tokenizer_identity':{'sha256':identity['sha256']}}
        self.runtime=PreparedRuntime(c.SENSE_MODEL_ID,self.runtime.wrapper,receipt,{},set(),{})
        for field,value in [('TOKENIZER_ROOT',self.root),('TOKENIZER_IDENTITY',identity)]:
            p=patch.object(c,field,value);p.start();self.addCleanup(p.stop)
    def test_changed_tokenizer_fails(self):
        self.tokenizer.sp.serialized_model_proto=lambda:b'wrong'
        with self.assertRaises(ValueError):self.capture()
        self.assertEqual(self.original_calls,0)
    def bind(self):return c.loaded_tokenizer_path(self.runtime,self.reviewed)
    def test_exact_dataclass_has_no_fabricated_field(self):
        self.assertFalse(hasattr(self.runtime,'tokenizer_file'));self.assertEqual(self.bind(),self.paths['tokenizer_file'])
    def test_missing_bpemodel_rejected(self):
        del self.tokenizer.bpemodel
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_types_relative_roots_names_rejected(self):
        for value in (None,self.paths['tokenizer_file'],'',c.TOKENIZER_NAME,'/other/'+c.TOKENIZER_NAME,str(self.root/'wrong')):
            self.tokenizer.bpemodel=value
            with self.assertRaises(ValueError):self.bind()
    def test_symlink_rejected(self):
        target=self.paths['tokenizer_file'];real=self.root/'other';target.rename(real);target.symlink_to(real)
        with self.assertRaises(ValueError):self.bind()
    def test_hardlink_rejected(self):
        import os
        os.link(self.paths['tokenizer_file'],self.root/'linked')
        with self.assertRaises(ValueError):self.bind()
    def test_missing_and_duplicate_receipt_rows(self):
        rows=self.runtime.receipt['assets']['verified_files']
        self.runtime.receipt['assets']['verified_files']=[]
        with self.assertRaises(ValueError):self.bind()
        self.runtime.receipt['assets']['verified_files']=rows*2
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_receipt_hash_rejected(self):
        self.runtime.receipt['assets']['verified_files'][0]['sha256']='f'*64
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_model_identity_rejected(self):
        self.runtime.model_id='wrong'
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_revision_rejected(self):
        self.runtime.receipt['assets']['revision']='wrong'
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_loaded_tokenizer_receipt_rejected(self):
        self.runtime.receipt['tokenizer_identity']['sha256']='f'*64
        with self.assertRaises(ValueError):self.bind()
    def test_changed_file_rejected(self):
        self.paths['tokenizer_file'].write_bytes(b'mutated')
        with self.assertRaises(ValueError):self.bind()
    def test_changed_serialized_processor_rejected(self):
        self.tokenizer.sp.serialized_model_proto=lambda:b'other'
        with self.assertRaises(ValueError):self.bind()
    def test_nonbytes_processor_rejected(self):
        self.tokenizer.sp.serialized_model_proto=lambda:bytearray(self.paths['tokenizer_file'].read_bytes())
        with self.assertRaises(ValueError):self.bind()
    def test_prospective_inference_rechecks_path_before_forward(self):
        cap=self.capture();self.tokenizer.bpemodel='/other/'+c.TOKENIZER_NAME
        with self.assertRaises(ValueError):self.invoke(cap)
        self.assertEqual(self.original_calls,0);self.assertTrue(cap.dead)
    def test_outer_capture_attributes_are_real_dataclass_fields(self):
        allowed=set(PreparedRuntime.__dataclass_fields__)
        tree=ast.parse((ROOT/'sense_capture.py').read_text())
        direct={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='loaded'}
        indirect={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and
                  isinstance(n.value,ast.Attribute) and n.value.attr=='loaded' and
                  isinstance(n.value.value,ast.Name) and n.value.value.id=='self'}
        self.assertTrue((direct|indirect)<=allowed,(direct,indirect,allowed))

class StaticTests(unittest.TestCase):
    def test_production_constants_match_original_asset_lock(self):
        locks=json.loads((REPO/'research/qwen6_asr/model-locks.json').read_text())
        row=next(r for r in locks['sensevoice']['asset_lock']['files'] if r['filename']==c.TOKENIZER_NAME)
        self.assertEqual(c.TOKENIZER_IDENTITY,{'bytes':row['bytes'],'sha256':row['sha256']})
        self.assertEqual(str(c.TOKENIZER_ROOT),'/runtime/models/sensevoice')
        self.assertEqual(c.SENSE_MODEL_ID,locks['sensevoice']['asset_lock']['model_id'])
        self.assertEqual(c.SENSE_REVISION,locks['sensevoice']['asset_lock']['revision'])
    def test_no_framework_import(self):
        for name in ('torch','numpy','funasr','kaldi_native_fbank','sentencepiece'):self.assertNotIn(name,sys.modules)
    def test_python38_syntax(self):ast.parse((ROOT/'sense_capture.py').read_text(),feature_version=(3,8))

if __name__=='__main__':unittest.main()
