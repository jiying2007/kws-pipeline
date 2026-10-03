#!/usr/bin/env python3
"""Install the reviewed exact input lock on an admitted public runner only.
All candidate artifacts are streamed one at a time, hashed, inspected, installed
without dependency resolution, then removed from this script's own staging.
"""
import argparse, hashlib, json, os, pathlib, platform, selectors, shutil, stat, subprocess, sys, tarfile, time, traceback, urllib.parse, urllib.request, venv, zipfile
if sys.flags.optimize: raise RuntimeError('Optimized Python bypasses qualification assertions and is forbidden')
RESERVE=1_073_741_824
TOTAL_CEILING=14_000_000_000
BUILD_SCRATCH_ALLOWANCE=536_870_912
MAX_LOG_BYTES=1_048_576
MAX_FAILURE_TAIL_BYTES=16_384
MAX_RECEIPT_BYTES=16_777_216
RECEIPT_FINISH_RESERVE=1_048_576
BUILD_NAMES=['pip','setuptools','wheel','packaging','numpy','cython']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe_member(name):
    p=pathlib.PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise RuntimeError('Unsafe archive path: '+name)

def inspect_wheel(path,expected):
    block=os.statvfs(path.parent).f_frsize or os.statvfs(path.parent).f_bsize
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();assert len(entries)<100000
        for e in entries:
            safe_member(e.filename);assert not stat.S_ISLNK(e.external_attr>>16),'Archive symlink'
        metas=[e for e in entries if e.filename.endswith('.dist-info/METADATA')];assert len(metas)==1
        b=z.read(metas[0]);mh=hashlib.sha256(b).hexdigest()
        if expected.get('metadata_sha256'):assert mh==expected['metadata_sha256'],f"{expected['name']} METADATA hash"
        from email import message_from_bytes
        m=message_from_bytes(b);canonical=lambda n:n.lower().replace('_','-').replace('.','-')
        assert canonical(m['Name'])==expected['name'] and m['Version']==expected['version']
        return {'expanded_bytes':sum(e.file_size for e in entries),'expanded_allocation_bytes':sum(((e.file_size+block-1)//block)*block for e in entries),'filesystem_allocation_block_bytes':block,'metadata_sha256':mh,'requires_dist':m.get_all('Requires-Dist',[]),'requires_python':m.get('Requires-Python'),'filename':path.name,'sha256':digest(path),'bytes':path.stat().st_size}

def echo_failure_tail(log):
    """Echo only this failed command's bounded log tail into the outer phase log."""
    with log.open('rb') as source:
        size=source.seek(0,os.SEEK_END);source.seek(max(0,size-MAX_FAILURE_TAIL_BYTES))
        tail=source.read(MAX_FAILURE_TAIL_BYTES)
    print(json.dumps({'event':'failed_command_log_tail','log':log.name,'stored_log_bytes':size,'tail_bytes':len(tail),'tail_start_byte':max(0,size-len(tail))}),flush=True)
    print(tail.decode('utf-8','replace'),flush=True)
    print(json.dumps({'event':'end_failed_command_log_tail','log':log.name}),flush=True)

class Installer:
    def __init__(self,a):
        self.a=a;self.raw=a.lock.read_bytes();self.lock=json.loads(self.raw);self.pkgs={x['name']:x for x in self.lock['packages']}
        self.initial_free=shutil.disk_usage(a.work.parent).free;self.peak_consumed=0;self.min_free=self.initial_free;self.events=[];self.realized=[]
        self.env={**os.environ,'PIP_CONFIG_FILE':'/dev/null','PIP_NO_INDEX':'1','PIP_NO_CACHE_DIR':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONNOUSERSITE':'1','SETUPTOOLS_USE_DISTUTILS':'local','CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MAX_JOBS':'2','CMAKE_BUILD_PARALLEL_LEVEL':'2','SOURCE_DATE_EPOCH':'1750809600'}
        for key in ['PYTHONOPTIMIZE','PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_TARGET','PIP_PREFIX','PIP_ROOT','PIP_USER']:
            self.env.pop(key,None)
        self.report={'schema':'cosyvoice3.runtime-install.v1','status':'failed','lock_sha256':hashlib.sha256(self.raw).hexdigest(),'initial_free_bytes':self.initial_free,'total_ceiling_bytes':TOTAL_CEILING,'reserve_bytes':RESERVE,'events':self.events,'python_version':platform.python_version(),'package_input_count':len(self.pkgs)}
    def budget(self,additional=0):
        free=shutil.disk_usage(self.a.work.parent).free;consumed=max(0,self.initial_free-free);self.peak_consumed=max(self.peak_consumed,consumed);self.min_free=min(self.min_free,free)
        if free<RESERVE+additional or consumed+additional+RESERVE>TOTAL_CEILING:raise RuntimeError(f'Disk gate failed: free={free}, consumed={consumed}, additional={additional}')
    def receipt_bytes(self):
        return sum(p.stat().st_size for p in self.a.receipts.rglob('*') if p.is_file())
    def run(self,cmd,timeout,log_name,env=None):
        self.budget();t=time.monotonic();log=self.a.receipts/log_name;written=0
        # Inherit the controller-owned process group; never let pip/compilers escape it.
        with log.open('xb') as f:
            p=subprocess.Popen(list(map(str,cmd)),env=env or self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ);eof=False
            try:
                while not (p.poll() is not None and eof):
                    self.budget()
                    if self.receipt_bytes()>MAX_RECEIPT_BYTES-RECEIPT_FINISH_RESERVE:raise RuntimeError('Aggregate receipt/log quota exceeded')
                    if time.monotonic()-t>timeout:raise TimeoutError('Subprocess wall limit: '+log_name)
                    for key,_ in selector.select(.2):
                        b=os.read(key.fileobj.fileno(),65536)
                        if not b:selector.unregister(key.fileobj);eof=True;continue
                        remaining=min(MAX_LOG_BYTES-written,MAX_RECEIPT_BYTES-RECEIPT_FINISH_RESERVE-self.receipt_bytes())
                        if len(b)>remaining:
                            if remaining>0:f.write(b[:remaining]);f.flush();written+=remaining
                            raise RuntimeError('Per-command or aggregate log quota exceeded: '+log_name)
                        f.write(b);f.flush();written+=len(b)
                if p.returncode:raise RuntimeError(f'{log_name} failed with {p.returncode}; inspect log')
            except BaseException:
                if p.poll() is None:p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                # Read only this known command log, never recursively publish receipts.
                f.flush();echo_failure_tail(log)
                # The outer controller terminates this entire owned group on phase failure.
                raise
            finally:
                selector.close();p.stdout.close()
        self.budget();self.events.append({'event':'command','log':log.name,'log_bytes':written,'elapsed_seconds':time.monotonic()-t,'returncode':p.returncode})
    def fetch(self,x):
        u=urllib.parse.urlsplit(x['url']);assert u.scheme=='https' and u.hostname=='files.pythonhosted.org' and not u.query
        assert pathlib.PurePosixPath(x['filename']).name==x['filename'];assert isinstance(x['size'],int) and x['size']>0
        assert len(x['sha256'])==64;self.budget(x['size']+16*1024*1024)
        p=self.a.work/'staging'/x['filename'];part=p.with_name(p.name+'.part');h=hashlib.sha256();count=0;t=time.monotonic()
        with urllib.request.urlopen(x['url'],timeout=60) as r,part.open('xb') as f:
            final=urllib.parse.urlsplit(r.url);assert final.scheme=='https' and final.hostname=='files.pythonhosted.org'
            while True:
                b=r.read(1024*1024)
                if not b:break
                count+=len(b);assert count<=x['size'],'Oversize download';f.write(b);h.update(b);self.budget()
                if time.monotonic()-t>900:raise TimeoutError('Artifact acquisition exceeded 900 seconds')
        assert count==x['size'] and h.hexdigest()==x['sha256'],f"{x['name']} artifact hash or byte-size mismatch"
        part.replace(p);self.events.append({'event':'verified_download','name':x['name'],'version':x['version'],'bytes':count,'sha256':h.hexdigest()});return p
    def install_wheel(self,python,p,x,label):
        info=inspect_wheel(p,x);self.budget(info['expanded_allocation_bytes']+64*1024*1024)
        self.run([python,'-m','pip','install','--no-index','--no-cache-dir','--no-deps','--disable-pip-version-check',p],600,f'install-{label}-{x["name"]}.log')
        self.realized.append({**info,'name':x['name'],'version':x['version'],'input_sha256':x['sha256'],'environment':label,'kind':'built-wheel' if x['kind']=='sdist' else 'official-wheel'})
    def build(self,builder,x):
        p=self.fetch(x);build=self.a.work/'build'/x['name'];build.mkdir();out=build/'out';out.mkdir()
        # Archive listing only; pip safely unpacks under our bounded TMPDIR.
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                entries=z.infolist();expanded=sum(e.file_size for e in entries)
                for e in entries:safe_member(e.filename);assert not stat.S_ISLNK(e.external_attr>>16)
        else:
            with tarfile.open(p) as z:
                entries=z.getmembers();expanded=sum(e.size for e in entries)
                for e in entries:safe_member(e.name);assert e.isfile() or e.isdir()
        assert expanded<32*1024*1024 and len(entries)<5000;self.budget(expanded+BUILD_SCRATCH_ALLOWANCE)
        env={**self.env,'TMPDIR':str(build),'PYTHONPATH':str(self.a.work/'offline-guard'),'COSYVOICE_NETWORK_LOG':str(self.a.receipts/'build-network-attempts.jsonl')}
        if (self.a.receipts/'build-network-attempts.jsonl').exists():assert not (self.a.receipts/'build-network-attempts.jsonl').read_text().strip()
        self.run([builder,'-m','pip','wheel','--no-index','--no-cache-dir','--no-deps','--no-build-isolation','--wheel-dir',out,p],900,f'build-{x["name"]}.log',env)
        assert not (self.a.receipts/'build-network-attempts.jsonl').exists() or not (self.a.receipts/'build-network-attempts.jsonl').read_text().strip(),'Blocked network attempt during source build'
        wheels=list(out.glob('*.whl'));assert len(wheels)==1;self.install_wheel(self.runtime,wheels[0],x,'runtime')
        p.unlink();shutil.rmtree(build)
    def execute(self):
        assert os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_REPOSITORY')=='jiying2007/kws-pipeline','Admitted public runner only'
        assert platform.python_version()=='3.12.14' and platform.system()=='Linux' and platform.machine()=='x86_64'
        runner_temp=pathlib.Path(os.environ['RUNNER_TEMP']).resolve();assert self.a.work.resolve().is_relative_to(runner_temp) and self.a.work.resolve()!=runner_temp
        assert self.lock['all_active_edges_satisfied'] and self.lock['status']=='input_lock_source_reviewed_pending_runtime_qualification'
        assert len(self.pkgs)==len(self.lock['packages']);assert set(self.lock['sdists'])=={'pyworld','openai-whisper','wget','antlr4-python3-runtime'}
        compiler=shutil.which('g++');assert compiler,'Inherited C++ compiler required; no system package install allowed'
        c=subprocess.run([compiler,'--version'],text=True,capture_output=True,timeout=15);assert c.returncode==0;self.report['compiler']={'path':compiler,'version_output':c.stdout}
        self.a.work.mkdir(exist_ok=False);(self.a.work/'staging').mkdir();(self.a.work/'build').mkdir();self.budget(512*1024*1024)
        guard=self.a.work/'offline-guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text("import sys,os,json\ndef audit(event,args):\n if event in {'socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.sendto'}:\n  path=os.environ['COSYVOICE_NETWORK_LOG'];size=os.path.getsize(path) if os.path.exists(path) else 0\n  if size<64000:\n   with open(path,'a') as f:f.write(json.dumps({'event':event,'args':repr(args)[:1000]})+'\\n')\n  raise RuntimeError('Offline source build forbids networking')\nsys.addaudithook(audit)\n")
        venv.EnvBuilder(with_pip=True,system_site_packages=False).create(self.a.work/'venv');self.runtime=self.a.work/'venv/bin/python'
        venv.EnvBuilder(with_pip=True,system_site_packages=False).create(self.a.work/'builder');builder=self.a.work/'builder/bin/python'
        # Build dependency order is explicit; no resolver or implicit source fallback runs.
        for n in BUILD_NAMES:
            x=self.pkgs[n];assert x['kind']=='wheel';p=self.fetch(x)
            self.install_wheel(builder,p,x,'builder');self.install_wheel(self.runtime,p,x,'runtime');p.unlink()
        self.run([builder,'-m','pip','check'],120,'builder-pip-check.log')
        for x in self.lock['packages']:
            if x['kind']=='wheel' and x['name'] not in BUILD_NAMES:
                p=self.fetch(x);self.install_wheel(self.runtime,p,x,'runtime');p.unlink()
        for x in self.lock['packages']:
            if x['kind']=='sdist':self.build(builder,x)
        self.run([self.runtime,'-m','pip','check'],120,'runtime-pip-check.log')
        # Only ephemeral paths created by this process are removed.
        shutil.rmtree(self.a.work/'builder');shutil.rmtree(self.a.work/'build');shutil.rmtree(self.a.work/'staging');shutil.rmtree(self.a.work/'offline-guard');self.budget()
        self.report.update({'status':'installed_pending_offline_qualification','python_executable':str(self.runtime),'final_free_bytes':shutil.disk_usage(self.a.work).free,'model_acquisition_admitted':False})
    def finish(self):
        self.report.update({'peak_observed_consumed_bytes':self.peak_consumed,'minimum_observed_free_bytes':self.min_free,'sampling_interval_seconds':.2,'per_command_log_limit_bytes':MAX_LOG_BYTES,'aggregate_receipt_limit_bytes':MAX_RECEIPT_BYTES,'budget_note':'Observed whole-filesystem delta; measured at admission, each download chunk, and every 0.2 seconds during subprocesses. Transient between-sample peaks are not proven. Final source/reference+model admission is separate.'})
        outputs={'install-receipt.json':json.dumps(self.report,indent=2),'realized-wheel-lock.json':json.dumps({'schema':'cosyvoice3.realized-wheel-lock.v1','input_lock_sha256':self.report['lock_sha256'],'wheels':self.realized},indent=2)}
        new_bytes=sum(len(v.encode()) for v in outputs.values());assert new_bytes<=RECEIPT_FINISH_RESERVE
        assert self.receipt_bytes()+new_bytes<=MAX_RECEIPT_BYTES,'Final receipt quota exceeded'
        for name,value in outputs.items():(self.a.receipts/name).write_text(value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=pathlib.Path,required=True);ap.add_argument('--work',type=pathlib.Path,required=True);ap.add_argument('--receipts',type=pathlib.Path,required=True);a=ap.parse_args();a.lock=a.lock.resolve();a.work=a.work.resolve();a.receipts=a.receipts.resolve();a.receipts.mkdir(parents=True,exist_ok=True)
    if any((a.receipts/n).exists() for n in ['install-receipt.json','realized-wheel-lock.json']):raise RuntimeError('Refusing to overwrite existing install receipts')
    i=Installer(a)
    try:i.execute()
    except BaseException as exc:i.report['error']=f'{type(exc).__name__}: {exc}';i.report['traceback']=traceback.format_exc()
    finally:i.finish()
    print(json.dumps({'status':i.report['status'],'error':i.report.get('error'),'python_executable':i.report.get('python_executable')}));return 0 if i.report['status']=='installed_pending_offline_qualification' else 1
if __name__=='__main__':sys.exit(main())
