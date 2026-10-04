#!/usr/bin/env python3
"""One standard-runner metadata-only check; no install, wheel body, or model."""
import hashlib,html.parser,json,os,platform,sys,time,urllib.error,urllib.request
INDEX='https://download.pytorch.org/whl/cpu/torch/'
WHEEL='https://download-r2.pytorch.org/whl/cpu/torch-2.12.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
WHEEL_SHA='ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df'
META_SHA='2dd7308d91027a8d24fe97f0982dca9a304814248869fb289c1d54e8e6c388ff'
BRANCH='refs/heads/research/a20-cpu-metadata-probe-20261004'
REQUESTS=((INDEX,'GET',1048576),(WHEEL,'HEAD',0),(WHEEL+'.metadata','GET',200000))
class Invalid(ValueError):pass
def require(ok,msg):
 if not ok:raise Invalid(msg)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):raise Invalid('redirect prohibited')
class Index(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   a=dict(attrs)
   if a.get('href')==WHEEL+'#sha256='+WHEEL_SHA:self.links.append(a)
def parse_index(raw):
 p=Index();p.feed(raw.decode('utf-8'));require(len(p.links)==1,'exact official link absent/duplicated')
 a=p.links[0]
 require(a.get('data-core-metadata')=='sha256='+META_SHA and a.get('data-dist-info-metadata')=='sha256='+META_SHA,'metadata identity drift')
 return {'index_bytes':len(raw),'index_sha256':hashlib.sha256(raw).hexdigest(),'exact_link_sha256':WHEEL_SHA,'metadata_sha256':META_SHA}
def parse_metadata(raw):
 import email.parser
 require(hashlib.sha256(raw).hexdigest()==META_SHA,'metadata hash mismatch')
 p=email.parser.BytesParser().parsebytes(raw)
 require(p['Name']=='torch' and p['Version']=='2.12.1+cpu','unexpected package identity')
 return {'name':p['Name'],'version':p['Version'],'requires_python':p['Requires-Python'],'requires_dist':p.get_all('Requires-Dist',[]),'license':p['License'],'license_expression':p['License-Expression'],'license_files':p.get_all('License-File',[]),'metadata_bytes':len(raw),'metadata_sha256':hashlib.sha256(raw).hexdigest()}
def validate_environment(env):
 require(env.get('GITHUB_ACTIONS')=='true','GitHub Actions only; local live requests prohibited')
 require(env.get('GITHUB_REPOSITORY')=='jiying2007/kws-pipeline','wrong public repository')
 require(env.get('GITHUB_EVENT_NAME')=='push' and env.get('GITHUB_REF')==BRANCH,'wrong event/ref')
 require(env.get('GITHUB_RUN_ATTEMPT')=='1','retry prohibited')
 require(env.get('RUNNER_OS')=='Linux' and env.get('RUNNER_ARCH')=='X64' and env.get('ImageOS')=='ubuntu24','wrong standard runner image')
 require(sys.version_info[:2]==(3,12),'CPython3.12 required')
 sha=env.get('GITHUB_SHA','');require(len(sha)==40 and all(x in '0123456789abcdef'for x in sha),'invalid commit identity')
 return {k:env.get(k)for k in ['GITHUB_REPOSITORY','GITHUB_SHA','GITHUB_REF','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','RUNNER_OS','RUNNER_ARCH','ImageOS','ImageVersion']}
def read_request(url,method,cap,opener,records):
 require((url,method,cap)in REQUESTS,'request outside fixed allowlist')
 record={'url':url,'method':method,'body_bytes_read':0};records.append(record)
 request=urllib.request.Request(url,method=method,headers={'User-Agent':'kws-pipeline-metadata-probe/1','Accept-Encoding':'identity'})
 try:
  with opener.open(request,timeout=20)as r:
   record['status']=r.status;require(r.status==200,'non200 HTTP response');require(r.url==url,'unexpected resolved URL')
   record['headers']={k:r.headers.get(k)for k in ['Content-Length','Content-Type','Date','Server']}
   if method=='HEAD':
    size=r.headers.get('Content-Length');require(size is not None and size.isdecimal() and int(size)>0,'missing/invalid wheel size')
    record['wheel_body_bytes_read']=0;return int(size)
   size=r.headers.get('Content-Length')
   if size is not None:
    require(size.isascii() and size.isdecimal() and 0<int(size)<=cap,'invalid/oversized metadata Content-Length')
    expected=int(size);b=r.read(expected);record['body_bytes_read']=len(b);require(len(b)==expected,'short metadata body')
   else:
    b=r.read(cap);record['body_bytes_read']=len(b);require(len(b)<cap,'metadata cap reached without EOF evidence')
   return b
 except urllib.error.HTTPError as e:
  record.update(status=e.code,error_type=type(e).__name__)
  record['headers']={k:e.headers.get(k)for k in ['Content-Length','Content-Type','Date','Server']}
  raise Invalid('HTTP '+str(e.code)+' on fixed '+method+' metadata request') from None
 except urllib.error.URLError as e:
  record['error_type']=type(e).__name__;raise Invalid('network request failed')from None

def main():
 begin=time.monotonic();records=[];report={'schema':'official-cpu-torch-actions-metadata-probe-v1','status':'FAILED_NO_RETRY','requests':records,'wheel_body_downloads':0,'install_attempts':0,'model_audio_training_calls':0,'cache_and_artifact_uploads':0,'environment_is_separate_from_prior_local_403':True}
 try:
  report['runner']=validate_environment(os.environ);report['python']=platform.python_version();report['libc']=platform.libc_ver()
  opener=urllib.request.build_opener(NoRedirect())
  raw=read_request(*REQUESTS[0],opener,records);report['index']=parse_index(raw)
  report['wheel_content_length']=read_request(*REQUESTS[1],opener,records)
  report['metadata']=parse_metadata(read_request(*REQUESTS[2],opener,records))
  report['status']='PASS_METADATA_ONLY_NO_INSTALL';rc=0
 except (Invalid,ValueError,OSError,UnicodeError)as e:report['error_type']=type(e).__name__;report['error']=str(e)[:180];rc=2
 report['elapsed_wall_s']=time.monotonic()-begin
 print('CPU_METADATA_RESULT='+json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False))
 return rc
if __name__=='__main__':sys.exit(main())
