"""Pure tests only: invented bytes; socket/process entrypoints fail if touched."""
import contextlib
import copy
from email.message import Message
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import sys
_SOURCE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_SOURCE_ROOT/'src'))
sys.path.insert(0,str(_SOURCE_ROOT/'src/deps'))

import exact_wheels as x


class Response:
    def __init__(self, body, url, length=None, status=200):
        self.data=io.BytesIO(body)
        self.status=status
        self.url=url
        self.headers=Message()
        if length is not False:
            self.headers['Content-Length']=str(len(body) if length is None else length)

    def __enter__(self):return self

    def __exit__(self,*args):return False

    def read(self,size):
        return self.data.read(size)

    def geturl(self):
        return self.url


def fixture(extra=None, *, metadata=None, wheel=None, compression=zipfile.ZIP_DEFLATED):
    data=io.BytesIO()
    meta=metadata or b'Metadata-Version: 2.4\nName: sample\nVersion: 1.0\nRequires-Python: >=3.9\nLicense: MIT\n\nexample\n'
    desc=wheel or b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n'
    entries=[('sample/__init__.py',b'pass\n'),('sample-1.0.dist-info/METADATA',meta),
             ('sample-1.0.dist-info/WHEEL',desc),('sample-1.0.dist-info/RECORD',b''),
             ('sample-1.0.dist-info/licenses/LICENSE',b'invented MIT fixture\n')]+(extra or [])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with zipfile.ZipFile(data,'w',compression=compression) as z:
            for name, body in entries:
                z.writestr(name,body)
    raw=data.getvalue()
    p={'project':'sample','version':'1.0','requires_python':'>=3.9','requires_dist':[],
       'filename':'sample-1.0-py3-none-any.whl','bytes':len(raw),'sha256':x.sha256(raw),
       'url':'https://files.pythonhosted.org/packages/sample-1.0-py3-none-any.whl',
       'matched_tag':'py3-none-any','license_metadata_sha256':x.sha256(b'MIT'),
       'metadata_url':'https://pypi.org/pypi/sample/1.0/json','metadata_sha256':'0'*64,
       'wheel_metadata_sha256':x.sha256(meta)}
    return raw,p


class PureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='.test-',dir=Path(__file__).parent)
        self.root=Path(self.temp.name)
        self.patches=[patch('socket.create_connection',side_effect=AssertionError('NETWORK_NOT_ALLOWED')),
                      patch('socket.socket',side_effect=AssertionError('NETWORK_NOT_ALLOWED')),
                      patch('subprocess.Popen',side_effect=AssertionError('CHILD_NOT_ALLOWED')),
                      patch.object(x,'exact_https_open',side_effect=AssertionError('NETWORK_NOT_ALLOWED'))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def inspect(self, raw, package, cap=x.UNCOMPRESSED_CAP):
        target=self.root/package['filename']
        target.write_bytes(raw)
        return x.inspect_wheel(target,package,x.Budget(uncompressed_cap=cap))

    def reject_zip(self, extra=None, code=None, **kwargs):
        raw,p=fixture(extra,**kwargs)
        with self.assertRaisesRegex(x.Rejected,code or '.'):
            self.inspect(raw,p)

    def test_nested_vendor_metadata_is_data_not_top_level_distribution(self):
        raw,p=fixture(extra=[('sample/_vendor/vendor-1.0.dist-info/METADATA',b'Name: vendor\nVersion: 1.0\n')])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/p['filename'];path.write_bytes(raw)
            result=x.inspect_wheel(path,p,x.Budget())
            self.assertEqual(result['project'],'sample')


    def test_actual_lock_exact_hashes_and_eleven(self):
        lock=x.load_lock()
        self.assertEqual(len(lock['packages']),11)
        self.assertEqual(sum(p['bytes'] for p in lock['packages']),220935082)
        self.assertEqual(lock['redirects'],[])

    def test_default_cli_is_read_only(self):
        before=set(self.root.iterdir())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(x.main([]),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'PREPARATION_ONLY_NO_NETWORK_NO_INSTALL')
        self.assertEqual(before,set(self.root.iterdir()))

    def test_download_without_release_rejected_without_network(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(x.main(['download','--workspace',str(self.root)]),2)
        self.assertEqual(json.loads(out.getvalue())['code'],'EXECUTION_NOT_RELEASED')
        self.assertFalse((self.root/'wheels').exists())

    def test_release_exact_closed_schema(self):
        release=x.release_template()
        with self.assertRaisesRegex(x.Rejected,'EXECUTION_NOT_RELEASED'):x.validate_release(release)
        release['approved']=True
        x.validate_release(release)
        release['extra']='ignored?'
        with self.assertRaises(x.Rejected):x.validate_release(release)

    def test_locked_urls_and_tags(self):
        for p in x.load_lock()['packages']:
            x.safe_url(p['url'])
            self.assertIn(p['matched_tag'],x.wheel_tags(p['filename']))

    def test_bad_urls(self):
        for url in ['http://files.pythonhosted.org/a.whl','https://evil.test/a.whl',
                    'https://files.pythonhosted.org:443/a.whl','https://u:p@files.pythonhosted.org/a.whl',
                    'https://files.pythonhosted.org/a.whl?q=secret','https://files.pythonhosted.org/a.whl#x',
                    'https://files.pythonhosted.org/a.tar.gz']:
            with self.subTest(url=url),self.assertRaises(x.Rejected):x.safe_url(url)

    def test_redirect_handler_never_follows(self):
        for u in ['https://files.pythonhosted.org/other.whl','https://download-r2.pytorch.org/x.whl','http://evil.test']:
            with self.assertRaisesRegex(x.Rejected,'REDIRECT_REJECTED'):
                x.RejectRedirects().redirect_request(None,None,302,'Found',{},u)

    def test_receive_exact_bytes_and_hash(self):
        body=b'invented wheel fixture'
        _,p=fixture();p.update(bytes=len(body),sha256=x.sha256(body))
        budget=x.Budget()
        r=x.receive_wheel(Response(body,p['url']),p,self.root/'result.whl',budget)
        self.assertEqual(r['bytes'],len(body));self.assertEqual(budget.download,len(body))
        self.assertEqual(stat.S_IMODE((self.root/'result.whl').stat().st_mode),0o600)

    def test_receive_rejects_http_and_header_variants(self):
        body=b'abc';_,p=fixture();p.update(bytes=3,sha256=x.sha256(body))
        cases=[]
        cases.append(Response(body,p['url'],status=206))
        cases.append(Response(body,'https://files.pythonhosted.org/alternate.whl'))
        cases.append(Response(body,p['url'],length=False))
        cases.append(Response(body,p['url'],length=4))
        cases.append(Response(body,p['url'],length='3, 3'))
        r=Response(body,p['url']);r.headers['Content-Length']='3';cases.append(r)
        r=Response(body,p['url']);r.headers['Transfer-Encoding']='chunked';cases.append(r)
        r=Response(body,p['url']);r.headers['Content-Encoding']='gzip';cases.append(r)
        for i,r in enumerate(cases):
            target=self.root/f'{i}.whl'
            with self.subTest(i=i),self.assertRaises(x.Rejected):
                x.receive_wheel(r,p,target,x.Budget())
            self.assertFalse(target.exists())

    def test_receive_rejects_truncation_oversize_hash_and_cleans(self):
        _,p=fixture();p.update(bytes=3,sha256=x.sha256(b'abc'))
        for i,body in enumerate([b'ab',b'abcd',b'xyz']):
            target=self.root/f'{i}.whl'
            with self.subTest(i=i),self.assertRaises(x.Rejected):
                x.receive_wheel(Response(body,p['url'],length=3),p,target,x.Budget())
            self.assertFalse(target.exists())

    def test_receive_budget_and_deadline(self):
        _,p=fixture();p.update(bytes=3,sha256=x.sha256(b'abc'))
        for opts in [{'budget':x.Budget(download_cap=2)},{'budget':x.Budget(),'deadline':0}]:
            with self.assertRaises(x.Rejected):
                x.receive_wheel(Response(b'abc',p['url']),p,self.root/'x.whl',**opts)
            self.assertFalse((self.root/'x.whl').exists())

    def test_receive_does_not_overwrite(self):
        _,p=fixture();p.update(bytes=3,sha256=x.sha256(b'abc'))
        target=self.root/'keep';target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            x.receive_wheel(Response(b'abc',p['url']),p,target,x.Budget())
        self.assertEqual(target.read_bytes(),b'keep')

    def test_receive_disk_failure_cleans(self):
        _,p=fixture();p.update(bytes=3,sha256=x.sha256(b'abc'))
        def disk_failure():raise x.Rejected('FREE_DISK_FLOOR')
        with self.assertRaisesRegex(x.Rejected,'FREE_DISK_FLOOR'):
            x.receive_wheel(Response(b'abc',p['url']),p,self.root/'x.whl',x.Budget(),free_check=disk_failure)
        self.assertFalse((self.root/'x.whl').exists())

    def test_good_zip_and_public_report(self):
        raw,p=fixture()
        r=self.inspect(raw,p)
        self.assertEqual(r['metadata']['name'],'sample')
        self.assertEqual(len(r['license_members']),1)
        self.assertNotIn(str(self.root),json.dumps(r))
        self.assertEqual(r['sha256'],p['sha256'])

    def test_zip_traversal(self):
        for name in ['../escape','/absolute','a/../escape','a//b','./a','a\\b','C:evil','a\x00evil']:
            with self.subTest(name=name):
                # zipfile sanitizes NUL at creation, so test ZipInfo directly.
                if '\x00' in name:
                    info=zipfile.ZipInfo(name)
                    with self.assertRaises(x.Rejected):x.safe_member(info)
                else:self.reject_zip([(name,b'x')])

    def test_zip_duplicate(self):
        self.reject_zip([('sample/__init__.py',b'changed')],'ZIP_DUPLICATE_MEMBER')

    def test_zip_file_directory_collision(self):
        self.reject_zip([('clash',b'x'),('clash/inside',b'x')],'ZIP_FILE_DIRECTORY_COLLISION')

    def test_zip_symlink_and_special_modes(self):
        for mode in [stat.S_IFLNK|0o777,stat.S_IFCHR|0o600,stat.S_IFIFO|0o600,stat.S_IFREG|0o4755]:
            info=zipfile.ZipInfo('special');info.create_system=3;info.external_attr=mode<<16
            with self.subTest(mode=mode):self.reject_zip([(info,b'target')],'ZIP_SPECIAL_FILE_REJECTED')

    def test_zip_encryption_flag(self):
        info=zipfile.ZipInfo('secret');info.flag_bits|=1
        with self.assertRaisesRegex(x.Rejected,'ZIP_ENCRYPTED_REJECTED'):x.safe_member(info)

    def test_zip_unknown_compression(self):
        info=zipfile.ZipInfo('z');info.compress_type=zipfile.ZIP_BZIP2
        with self.assertRaisesRegex(x.Rejected,'ZIP_COMPRESSION_REJECTED'):x.safe_member(info)

    def test_zip_declared_member_size(self):
        info=zipfile.ZipInfo('huge');info.file_size=x.MEMBER_CAP+1
        with self.assertRaisesRegex(x.Rejected,'ZIP_MEMBER_SIZE_REJECTED'):x.safe_member(info)

    def test_zip_bomb_ratio(self):
        info=zipfile.ZipInfo('bomb');info.file_size=1001;info.compress_size=1
        with self.assertRaisesRegex(x.Rejected,'ZIP_RATIO_REJECTED'):x.safe_member(info)

    def test_zip_aggregate_cap(self):
        raw,p=fixture()
        with self.assertRaisesRegex(x.Rejected,'UNCOMPRESSED_CAP_EXCEEDED'):self.inspect(raw,p,cap=1)

    def test_zip_foreign_dist_info(self):
        self.reject_zip([('alien-1.0.dist-info/METADATA',b'x')],'ZIP_FOREIGN_DIST_INFO')

    def test_zip_prefix_suffix_comment(self):
        raw,p=fixture()
        for changed in [b'executable'+raw,raw+b'trailer']:
            q=p|{'bytes':len(changed),'sha256':x.sha256(changed)}
            with self.assertRaises(x.Rejected):self.inspect(changed,q)

    def test_zip_hash_precedes_parsing(self):
        raw,p=fixture();p['sha256']='f'*64
        with self.assertRaisesRegex(x.Rejected,'WHEEL_SHA256_MISMATCH'):self.inspect(raw,p)

    def test_metadata_critical_fields(self):
        base=b'Metadata-Version: 2.4\nName: sample\nVersion: 1.0\nRequires-Python: >=3.9\n\n'
        for changed in [base.replace(b'Name: sample',b'Name: foreign'),
                        base.replace(b'Version: 1.0',b'Version: 2.0'),
                        base.replace(b'>=3.9',b'>=3.13'),
                        base.replace(b'\n\n',b'\nRequires-Dist: evil\n\n'),
                        base.replace(b'\n\n',b'\nName: sample\n\n')]:
            self.reject_zip(metadata=changed)

    def test_wheel_tags_and_purelib(self):
        for wheel in [b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: cp313-cp313-any\n\n',
                      b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n\n',
                      b'Wheel-Version: 2.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n']:
            self.reject_zip(wheel=wheel)

    def test_python_spec_rejects_unknown_or_wrong(self):
        for spec in ['>=3.13','!=3.12.3','~=3.12','https://evil']:
            with self.assertRaises(x.Rejected):x.python_spec_matches(spec)
        self.assertTrue(x.python_spec_matches('!=3.14.1,>=3.12'))

    def test_aggregate_budget_across_wheels(self):
        b=x.Budget(uncompressed_cap=10)
        b.add_uncompressed(6)
        with self.assertRaises(x.Rejected):b.add_uncompressed(5)

    def test_workspace_mode_and_symlink(self):
        self.assertEqual(x.workspace_root(self.root),self.root)
        child=self.root/'child';child.mkdir(mode=0o755)
        with self.assertRaises(x.Rejected):x.workspace_root(child)
        link=self.root/'link';link.symlink_to(self.root)
        with self.assertRaises(x.Rejected):x.workspace_root(link)

    def test_wheelhouse_closed_set(self):
        wheels=self.root/'wheels';wheels.mkdir()
        (wheels/'extra.whl').write_bytes(b'')
        with self.assertRaisesRegex(x.Rejected,'WHEELHOUSE_NOT_CLOSED'):x.inspect_wheelhouse(self.root)

    def test_venv_argv_only(self):
        release=x.release_template()|{'approved':True}
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'):
            argv=x.build_venv_argv(self.root,release)
        self.assertEqual(argv[1:5],['-I','-m','venv','--copies'])
        self.assertFalse((self.root/'venv').exists())

    def test_install_argv_only_closed_flags(self):
        release=x.release_template()|{'approved':True}
        (self.root/'requirements.offline.lock').write_bytes((x.HERE/'requirements.proposed.lock').read_bytes())
        (self.root/'venv'/'bin').mkdir(parents=True)
        (self.root/'venv'/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        (self.root/'venv'/'bin'/'python').write_bytes(b'fixture only')
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'),patch.object(x,'inspect_wheelhouse'):
            argv=x.build_install_argv(self.root,release)
        for flag in ['--no-index','--no-deps','--require-hashes','--only-binary=:all:','--isolated','--require-virtualenv']:
            self.assertIn(flag,argv)
        self.assertEqual(argv[1:4],['-I','-m','pip'])
        self.assertFalse(any('https:' in arg for arg in argv))

    def test_pipeline_fixture_download_inspect_no_install(self):
        raw,p=fixture()
        lock={'packages':[p],'total_download_bytes':len(raw)}
        release=x.release_template()|{'approved':True}
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'), \
             patch.object(x,'load_lock',return_value=lock), \
             patch.object(x,'exact_https_open',return_value=Response(raw,p['url'])) as opened:
            result=x.download_all(self.root,release)
            self.assertEqual(result['status'],'PASS_DOWNLOADED_AND_INSPECTED_NO_INSTALL')
            self.assertEqual(result['actual_download_bytes'],len(raw))
            self.assertEqual(opened.call_count,1)
            with self.assertRaisesRegex(x.Rejected,'DOWNLOAD_OUTPUT_ALREADY_EXISTS'):
                x.download_all(self.root,release)
            self.assertEqual(opened.call_count,1)
        self.assertFalse((self.root/'venv').exists())

    def test_pipeline_failure_terminal_no_retry_or_fallback(self):
        raw,p=fixture()
        lock={'packages':[p],'total_download_bytes':len(raw)}
        release=x.release_template()|{'approved':True}
        with patch.object(x,'validate_runtime'),patch.object(x,'check_free'), \
             patch.object(x,'load_lock',return_value=lock), \
             patch.object(x,'exact_https_open',return_value=Response(raw,p['url'],status=403)) as opened:
            with self.assertRaisesRegex(x.Rejected,'HTTP_STATUS_REJECTED'):
                x.download_all(self.root,release)
            self.assertEqual(opened.call_count,1)
            self.assertEqual(list((self.root/'wheels').iterdir()),[])
            self.assertFalse((self.root/'requirements.offline.lock').exists())
            with self.assertRaisesRegex(x.Rejected,'DOWNLOAD_OUTPUT_ALREADY_EXISTS'):
                x.download_all(self.root,release)
            self.assertEqual(opened.call_count,1)

    def test_metadata_declared_license_must_exist(self):
        meta=b'Metadata-Version: 2.4\nName: sample\nVersion: 1.0\nRequires-Python: >=3.9\nLicense-File: MISSING\n\n'
        self.reject_zip(metadata=meta,code='DECLARED_LICENSE_FILE_MISSING')

    def test_torch_style_metadata_hash_checked(self):
        raw,p=fixture();p['wheel_metadata_sha256']='1'*64
        with self.assertRaisesRegex(x.Rejected,'WHEEL_METADATA_HASH_MISMATCH'):
            self.inspect(raw,p)

    def test_metadata_caps_before_allocating(self):
        with patch.object(x,'METADATA_CAP',16):
            self.reject_zip(code='METADATA_CAP_EXCEEDED')

    def test_corrupt_crc_or_payload_blocks(self):
        raw,p=fixture(compression=zipfile.ZIP_STORED)
        changed=raw.replace(b'pass\n',b'fail\n',1)
        p.update(bytes=len(changed),sha256=x.sha256(changed))
        with self.assertRaises(zipfile.BadZipFile):self.inspect(changed,p)

    def test_cli_error_redacts_private_paths(self):
        with patch.object(x,'load_lock',side_effect=OSError('/private/user/secret')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(x.main([]),2)
        self.assertNotIn('/private',out.getvalue())


if __name__=='__main__':unittest.main(verbosity=2)
