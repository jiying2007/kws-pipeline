"""Metadata/HTTP fixtures only. No install, wheel download or model/DSP calls."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest
import test_exact_wheels as fixtures
from test_exact_wheels import Response,fixture
import exact_wheels as x
import public_artifact as gate


class MetadataRecoveryTests(unittest.TestCase):
    setUp=fixtures.PureTests.setUp
    tearDown=fixtures.PureTests.tearDown
    inspect=fixtures.PureTests.inspect
    def make(self,requirements):
        raw=b'Metadata-Version: 2.4\nName: sample\nVersion: 1.0\nRequires-Python: >=3.9\n'
        raw+=b''.join(('Requires-Dist: '+r+'\n').encode()for r in requirements)+b'\n'
        wheel,p=fixture(metadata=raw);p['requires_dist']=requirements
        return raw,wheel,p

    def test_all_eleven_have_metadata_sha_and_same_body_budget(self):
        lock=x.load_lock()
        self.assertEqual(len(lock['packages']),11)
        self.assertEqual(lock['total_download_bytes'],220935082)
        self.assertTrue(all(len(p['wheel_metadata_sha256'])==64 for p in lock['packages']))

    def test_jinja_semicolon_spacing_exact_pin_reproduces_old_failure(self):
        actual='Babel>=2.7 ; extra == "i18n"'
        md,raw,p=self.make([actual]);old=copy.deepcopy(p)
        old['requires_dist']=['Babel>=2.7; extra == "i18n"']
        with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):self.inspect(raw,old)
        self.assertEqual(self.inspect(raw,p)['metadata']['requires_dist'],[actual])

    def test_fsspec_quote_parenthesis_exact_pin(self):
        actual="backports-zstd; (python_version < '3.14') and extra == 'test-full'"
        md,raw,p=self.make([actual]);old=copy.deepcopy(p)
        old['requires_dist']=['backports-zstd; python_version < "3.14" and extra == "test-full"']
        with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):self.inspect(raw,old)
        self.assertEqual(self.inspect(raw,p)['metadata']['requires_dist'],[actual])

    def test_counter_order_still_ignored_but_multiplicity_enforced(self):
        md,raw,p=self.make(['x>=1','y<3']);p['requires_dist'].reverse()
        self.inspect(raw,p)
        p['requires_dist'].append('x>=1')
        with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):self.inspect(raw,p)

    def test_unknown_semantically_equal_text_still_rejected(self):
        md,raw,p=self.make(['x>=1; extra == "doc"'])
        p['requires_dist']=["x>=1; extra == 'doc'"]
        with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):self.inspect(raw,p)

    def test_optional_missing_and_real_version_change_rejected(self):
        for altered in ([],['x>=2; extra == "doc"']):
            md,raw,p=self.make(['x>=1; extra == "doc"']);p['requires_dist']=altered
            with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):self.inspect(raw,p)

    def test_metadata_hash_even_with_equal_fields(self):
        md,raw,p=self.make(['x>=1']);p['wheel_metadata_sha256']='0'*64
        with self.assertRaisesRegex(x.Rejected,'WHEEL_METADATA_HASH_MISMATCH'):self.inspect(raw,p)

    def test_unexpected_metadata_text_is_hashed_not_disclosed(self):
        md,raw,p=self.make(['x>=1'])
        hostile=b'Name: sample\nRequires-Dist: hidden @ https://private.invalid/token-secret\n\n'
        report=x.metadata_evidence(hostile,x.parse_headers(hostile),p)
        self.assertFalse(report['metadata_hash_matches'])
        self.assertIsNone(report['observed_requires_dist']['values'])
        self.assertNotIn('private.invalid',json.dumps(report))
        self.assertNotIn('token-secret',json.dumps(report))
        gate.privacy_check(report)

    def test_failure_keeps_exact_package_hash_fields_and_completed_download(self):
        md,raw,p=self.make(['x>=1 ; extra == "doc"']);p['requires_dist']=['x>=1; extra == "doc"']
        states=[];lock={'packages':[p],'total_download_bytes':len(raw)}
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'),patch.object(x,'load_lock',return_value=lock),patch.object(x,'exact_https_open',return_value=Response(raw,p['url']))as opened:
            with self.assertRaisesRegex(x.Rejected,'METADATA_REQUIRES_DIST_MISMATCH'):
                x.download_all(self.root,x.release_template()|{'approved':True},lambda o:states.append(copy.deepcopy(o)))
        last=states[-1]
        self.assertEqual(opened.call_count,1)
        self.assertEqual(last['status'],'FAILED_NO_RETRY')
        self.assertEqual(last['request_attempts_reserved'],1)
        self.assertEqual(last['successful_download_bytes'],len(raw))
        self.assertEqual(len(last['successful_downloads']),1)
        self.assertEqual(last['inspections_completed'],[])
        self.assertEqual(last['current']['package']['project'],'sample')
        obs=last['current']['metadata']
        self.assertEqual(obs['observed_metadata_sha256'],p['wheel_metadata_sha256'])
        self.assertNotEqual(obs['expected_requires_dist']['values'],obs['observed_requires_dist']['values'])
        gate.privacy_check(last)

    def test_http_failure_has_reservation_not_false_completed_request(self):
        md,raw,p=self.make([]);states=[]
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'),patch.object(x,'load_lock',return_value={'packages':[p],'total_download_bytes':len(raw)}),patch.object(x,'exact_https_open',return_value=Response(b'private raw failure',p['url'],status=403))as opened:
            with self.assertRaisesRegex(x.Rejected,'HTTP_STATUS_REJECTED'):
                x.download_all(self.root,x.release_template()|{'approved':True},lambda o:states.append(copy.deepcopy(o)))
        last=states[-1]
        self.assertEqual(opened.call_count,1)
        self.assertEqual(last['request_attempts_reserved'],1)
        self.assertEqual(last['successful_downloads'],[])
        self.assertEqual(last['successful_download_bytes'],0)
        self.assertNotIn('private raw failure',json.dumps(last))

    def test_existing_dependencies_json_keeps_progress_after_failure(self):
        import launch_once as launch
        from common import write_json,read_json
        original=Path(__file__).resolve().parents[1]
        temp=self.root/'package';(temp/'work').mkdir(parents=True);(temp/'metadata').mkdir()
        for name in ('SOURCE-FREEZE.json','PROTOCOL.json'):(temp/name).write_text('{}')
        for name in ('PUBLIC-INPUT-MAP.json','TRAIN32.json'):(temp/'metadata'/name).write_text('{}')
        (temp/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json').write_bytes((original/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json').read_bytes())
        progress={'status':'FAILED_NO_RETRY','current':{'stage':'METADATA_OBSERVED','project':'jinja2'},'successful_download_bytes':123}
        write_json(temp/'work/wheel-progress.json',progress)
        out=self.root/'out';out.mkdir()
        with patch.object(launch,'ROOT',temp):
            launch.finish_evidence(out,'FAILED',[{'name':'wheels','status':'FAILED_NO_RETRY'}],{},dict(wall_ms=1,accounted_cpu_ms=1),'0'*64)
        saved=read_json(out/'dependencies.json')
        self.assertEqual(saved['wheel_progress'],progress)
        self.assertEqual(saved['installation']['status'],'NOT_COMPLETED')
        gate.privacy_check(saved)

