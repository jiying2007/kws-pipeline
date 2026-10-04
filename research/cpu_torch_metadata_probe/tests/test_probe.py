import hashlib,importlib.util,io,json,os,sys,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'probe.py';s=importlib.util.spec_from_file_location('probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Response:
 def __init__(self,url,body=b'',headers=None,status=200):self.url=url;self.body=body;self.headers=headers or {};self.status=status;self.read_calls=0
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read(self,n):self.read_calls+=1;return self.body[:n]
class Opener:
 def __init__(self,r):self.r=r;self.calls=[]
 def open(self,q,timeout):self.calls.append((q.full_url,q.method,timeout));return self.r
class Tests(unittest.TestCase):
 def link(self):return ('<a href="'+m.WHEEL+'#sha256='+m.WHEEL_SHA+'" data-core-metadata="sha256='+m.META_SHA+'" data-dist-info-metadata="sha256='+m.META_SHA+'">wheel</a>').encode()
 def env(self):return dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY='jiying2007/kws-pipeline',GITHUB_EVENT_NAME='push',GITHUB_REF=m.BRANCH,GITHUB_RUN_ATTEMPT='1',RUNNER_OS='Linux',RUNNER_ARCH='X64',ImageOS='ubuntu24',GITHUB_SHA='a'*40)
 def test_exact_index(self):self.assertEqual(m.parse_index(self.link())['metadata_sha256'],m.META_SHA)
 def test_duplicate_missing_changed_index_rejected(self):
  for b in [b'',self.link()*2,self.link().replace(m.META_SHA.encode(),b'a'*64)]:
   with self.assertRaises(m.Invalid):m.parse_index(b)
 def test_head_never_reads_body(self):
  r=Response(m.WHEEL,b'do-not-read',{'Content-Length':'123'});o=Opener(r);a=[];self.assertEqual(m.read_request(*m.REQUESTS[1],o,a),123);self.assertEqual(r.read_calls,0);self.assertEqual(a[0]['body_bytes_read'],0)
 def test_head_missing_size_rejected(self):
  for x in [None,'0','-2','1.5','nan']:
   r=Response(m.WHEEL,headers={}if x is None else{'Content-Length':x})
   with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[1],Opener(r),[])
 def test_cap_rejected(self):
  with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[2],Opener(Response(m.REQUESTS[2][0],b'a'*200001)),[])
 def test_declared_lengths_strict_cap_and_short_body(self):
  for body,size in [(b'x','200001'),(b'x','0'),(b'x','-1'),(b'x','nan'),(b'x','2')]:
   r=Response(m.REQUESTS[2][0],body,{'Content-Length':size})
   with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[2],Opener(r),[])
   if size!='2':self.assertEqual(r.read_calls,0)
  r=Response(m.REQUESTS[2][0],b'x'*200000,{'Content-Length':'200000'});a=[]
  self.assertEqual(len(m.read_request(*m.REQUESTS[2],Opener(r),a)),200000);self.assertEqual(a[0]['body_bytes_read'],200000)
 def test_unknown_length_never_reads_probe_byte(self):
  class Sized(Response):
   def read(self,n):self.last_n=n;return super().read(n)
  r=Sized(m.REQUESTS[2][0],b'x'*200001);a=[]
  with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[2],Opener(r),a)
  self.assertEqual(r.last_n,200000);self.assertEqual(a[0]['body_bytes_read'],200000)
  r=Response(m.REQUESTS[2][0],b'abc');self.assertEqual(m.read_request(*m.REQUESTS[2],Opener(r),[]),b'abc')
 def test_unlisted_url_and_method_rejected(self):
  for args in [(m.WHEEL,'GET',200000),(m.INDEX,'HEAD',0),('https://example.com','GET',1)]:
   with self.assertRaises(m.Invalid):m.read_request(*args,Opener(None),[])
 def test_redirect_rejected(self):
  with self.assertRaises(m.Invalid):m.NoRedirect().redirect_request(None,None,302,'',{},m.WHEEL)
  with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[1],Opener(Response(m.WHEEL+'?other',headers={'Content-Length':'1'})),[])
 def test_environment_and_attempt_gate(self):
  self.assertEqual(m.validate_environment(self.env())['GITHUB_REF'],m.BRANCH)
  for k,v in [('GITHUB_ACTIONS','false'),('GITHUB_REPOSITORY','other/repo'),('GITHUB_EVENT_NAME','pull_request'),('GITHUB_REF','refs/heads/main'),('GITHUB_RUN_ATTEMPT','2'),('RUNNER_OS','Windows'),('RUNNER_ARCH','ARM64'),('ImageOS','ubuntu22'),('GITHUB_SHA','bad')]:
   e=self.env();e[k]=v
   with self.assertRaises(m.Invalid):m.validate_environment(e)
 def test_metadata_hash_and_identity(self):
  b=b'Name: torch\nVersion: 2.12.1+cpu\nRequires-Python: >=3.10\nRequires-Dist: filelock\nLicense: BSD-3-Clause\n\n'
  with patch.object(m,'META_SHA',hashlib.sha256(b).hexdigest()):self.assertEqual(m.parse_metadata(b)['requires_dist'],['filelock'])
  with self.assertRaises(m.Invalid):m.parse_metadata(b)
 def test_http_error_record_no_body_no_retry(self):
  class ErrorOpener:
   def __init__(self):self.calls=0
   def open(self,*args,**kw):self.calls+=1;raise urllib.error.HTTPError(m.WHEEL,403,'Forbidden',{},io.BytesIO(b'not-read'))
  o=ErrorOpener();r=[]
  with self.assertRaises(m.Invalid):m.read_request(*m.REQUESTS[1],o,r)
  self.assertEqual(o.calls,1);self.assertEqual(r[0]['status'],403);self.assertEqual(r[0]['body_bytes_read'],0)
 def test_main_no_live_network_outside_github(self):
  with patch.dict(os.environ,{},clear=True),patch('sys.stdout',new_callable=io.StringIO)as st,patch.object(m.urllib.request,'build_opener')as op:self.assertEqual(m.main(),2);op.assert_not_called();self.assertIn('FAILED_NO_RETRY',st.getvalue())
if __name__=='__main__':unittest.main()
