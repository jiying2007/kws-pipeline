"""Independent-review regressions: source patch type guards and 32MiB ceilings."""
from pathlib import Path
import json
import sys
import textwrap
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sense_capture as c
import frontend_verifier as v
import test_sense_capture as fixtures
import test_frontend_verifier as verifier_fixtures

class Tests(unittest.TestCase):
    setUp=fixtures.Tests.setUp
    tearDown=fixtures.Tests.tearDown
    def added_guard(self,row):
        lines=(Path(__file__).parent/'sense-frontend-resolver-review.patch').read_text().splitlines(True)
        additions=[];in_hunk=False
        for line in lines:
            if line.startswith('@@'):
                if in_hunk:break
                in_hunk=True;continue
            if in_hunk and line.startswith('+'):additions.append(line[1:])
        namespace=dict(refs_row=row,identities={'sensevoice':{'model_family':'sensevoice'},'qwen06':{'model_family':'qwen3-asr'},'whisper':{'model_family':'whisper'}},expected_refs=set())
        exec(compile(textwrap.dedent(''.join(additions)),'<actual-proposed-resolver-guard>','exec'),namespace)
        return namespace['expected_refs']
    def test_patch_unhashable_ids_raise_valueerror(self):
        for value in ([],{},['sensevoice'],{'id':'sensevoice'}):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'ID must be a string'):
                self.added_guard({'recognizer_id':value})
    def test_patch_other_invalid_ids_raise_valueerror(self):
        for value in (None,True,1,1.0):
            with self.subTest(value=value),self.assertRaises(ValueError):self.added_guard({'recognizer_id':value})
    def test_only_sense_gets_extra_reference(self):
        self.assertEqual(self.added_guard({'recognizer_id':'sensevoice'}),{'frontend_runtime_binding'})
        for ident in ('qwen06','whisper','unknown'):
            self.assertEqual(self.added_guard({'recognizer_id':ident}),set())
    def test_regular_exact_limit_and_one_byte_over(self):
        path=self.root/'bounded-binary'
        with path.open('wb') as stream:stream.truncate(32*1024**2)
        self.assertEqual(len(c.regular(path,32*1024**2)),32*1024**2)
        with path.open('ab') as stream:stream.write(b'x')
        with self.assertRaisesRegex(ValueError,'bounded regular'):c.regular(path,32*1024**2)
    def test_producer_and_prepare_use_32mib(self):
        original=c.regular;calls=[]
        def observed(path,maximum):
            if Path(path)==self.paths['backend_binary']:calls.append(maximum)
            return original(path,maximum)
        with patch.object(c,'regular',side_effect=observed):
            c.snapshot(self.frontend,self.tokenizer,self.paths['tokenizer_file'],self.paths['parser_source'],self.reviewed)
            fixtures.Tests.prepared(self)
        self.assertEqual(calls,[32*1024**2]*3)
    def test_snapshot_verifier_boundary(self):
        fixture=verifier_fixtures.Tests();fixture.setUp()
        fixture.snap['dependencies']['backend_binary']['bytes']=32*1024**2
        v.validate_snapshot(verifier_fixtures.raw(fixture.snap))
        fixture.snap['dependencies']['backend_binary']['bytes']+=1
        with self.assertRaisesRegex(ValueError,'byte bounds'):v.validate_snapshot(verifier_fixtures.raw(fixture.snap))
    def test_schema_matches_existing_resolver_bound(self):
        schema=json.loads((Path(__file__).parent/'frontend-observation.schema.json').read_text())
        self.assertEqual(schema['properties']['dependencies']['properties']['backend_binary']['properties']['bytes']['maximum'],32*1024**2)

    def mapping(self,path,permissions='r-xp',inode=None):
        import os
        st=path.stat()
        return ('00001000-00002000 %s 00000000 %x:%x %d %s\n' %
            (permissions,os.major(st.st_dev),os.minor(st.st_dev),st.st_ino if inode is None else inode,path)).encode()
    def test_actual_mapping_identity_without_loading(self):
        path=self.paths['backend_core_library']
        self.assertTrue(c.verify_executable_mapping(path,self.mapping(path)))
        for raw in (self.mapping(path,'r--p'),self.mapping(path,inode=path.stat().st_ino+1),
                    self.mapping(path).rstrip()+b' (deleted)\n'):
            with self.assertRaises(ValueError):c.verify_executable_mapping(path,raw)
    def test_actual_mapping_symlink_rejected(self):
        path=self.root/'alias';path.symlink_to(self.paths['backend_core_library'])
        with self.assertRaises(ValueError):c.verify_executable_mapping(path,self.mapping(path))
    def test_core_library_required_by_verifier(self):
        fixture=verifier_fixtures.Tests();fixture.setUp()
        del fixture.snap['dependencies']['backend_core_library']
        with self.assertRaises(ValueError):v.validate_snapshot(verifier_fixtures.raw(fixture.snap))
        fixture.setUp();fixture.snap['native_execution_mappings']['backend_core_library']=False
        with self.assertRaisesRegex(ValueError,'observed executable'):v.validate_snapshot(verifier_fixtures.raw(fixture.snap))
    def test_both_native_members_same_32mib_schema_bound(self):
        schema=json.loads((Path(__file__).parent/'frontend-observation.schema.json').read_text())
        for name in ('backend_binary','backend_core_library'):
            self.assertEqual(schema['properties']['dependencies']['properties'][name]['properties']['bytes']['maximum'],32*1024**2)

if __name__=='__main__':unittest.main()
