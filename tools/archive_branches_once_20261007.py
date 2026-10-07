#!/usr/bin/env python3
"""One-purpose archival coordinator. Templates are locked until reviewed publication.
Only tag create/no-op and frozen branch deletions can reach the mutation transport.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ZERO = '0' * 40
TAG = 'refs/tags/archive/branches-2026-10-07'
REPOS = {'kws-pipeline': ('1349639240',60,57), 'kws-data': ('1397319075',22,22)}
HOLDS = {'refs/heads/feat/restricted-development-dataset-iteration-v1',
         'refs/heads/training/keyword-set-contract-v1', 'refs/heads/training/keyword-set-identity-v1'}
FROZEN_FIELDS = ('full_name', 'repository_id', 'snapshot_baseline_main', 'archive_tree_oid', 'archive_docs', 'ordered_archive_parents', 'original_count', 'delete_count', 'delete_refs', 'keep_refs')
FROZEN_BINDINGS = {'kws-pipeline': '93849ef82299bddd2ed7c6c357629a2e86ecbc972e77b71921a64e04690b75ed', 'kws-data': 'da3562c7887b435478a71657189fe463f6955d08bab8e33ea121a40608d0f5b1'}
SHA = re.compile(r'[0-9a-f]{40}\Z')

class Stop(RuntimeError): pass
class ApiError(Stop):
    def __init__(self, status, errors):
        self.status, self.errors = status, errors
        super().__init__('API rejected request; status=' + str(status))


def require(ok, message):
    if not ok: raise Stop(message)

def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out

def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(Stop('nonfinite JSON')))

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha256(raw): return hashlib.sha256(raw).hexdigest()
def valid_oid(value): return isinstance(value,str) and SHA.fullmatch(value) and value != ZERO

def frozen_rows(repo): return repo['delete_refs'] + repo['keep_refs']

def cas_signature(errors, wrong_before, archive):
    return [{'type':x.get('type'),'message':str(x.get('message','')).replace(wrong_before,'<WRONG_BEFORE>').replace(archive,'<ARCHIVE>')} for x in errors]

def explicit_cas_signature(signature):
    return bool(signature) and all(x.get('type')=='UNPROCESSABLE' and
           '<WRONG_BEFORE>' in x.get('message','') and '<ARCHIVE>' in x.get('message','') and
           ('expected' in x['message'].lower() or 'before' in x['message'].lower()) and
           not any(term in x['message'].lower() for term in ('fast-forward','fast forward','ancestor','force','permission','forbidden'))
           for x in signature)

def validate_plan(p):
    require(p['schema']=='kws-one-shot-archive-execution-v1' and p['publication_ready'] is True,
            'unpublished or unapproved template')
    require(p['owner']=='jiying2007' and p['owner_id']=='33591504' and
            p['expected_dispatch_actor_id']=='33591504' and p['tag']==TAG, 'identity constants')
    require(set(p['repositories'])==set(REPOS), 'repository allowlist')
    require(p['resource_limits']=={'timeout_seconds':1200,'max_object_bytes':5368709120,
            'max_objects':500000,'max_disk_bytes':6442450944}, 'resource budget changed')
    for name,(identity,count,delete_count) in REPOS.items():
        r=p['repositories'][name];rows=frozen_rows(r)
        require(sha256(canonical({key:r[key] for key in FROZEN_FIELDS}))==FROZEN_BINDINGS[name],
                'original approved inventory changed')
        require(r['full_name']=='jiying2007/'+name and r['repository_id']==identity,
                'repository binding')
        require(r['default_branch']=='main' and r['original_count']==count and
                r['delete_count']==delete_count and len(rows)==count and
                len(r['delete_refs'])==delete_count, 'frozen counts')
        names=[x['name'] for x in rows]
        require(len(set(names))==count and all(x.startswith('refs/heads/') for x in names)
                and 'refs/heads/main' not in names, 'invalid original ref inventory')
        require(set(x['name'] for x in r['keep_refs'])==(HOLDS if name=='kws-pipeline' else set()),
                'held ref inventory')
        for row in rows:
            require(set(row)=={'name','before_oid','tree_oid'} and valid_oid(row['before_oid'])
                    and valid_oid(row['tree_oid']), 'invalid frozen row')
            require(not any(s in row['name'] for s in ('..','@{','\\','\0','\n','\r',' ')), 'unsafe ref')
        require(valid_oid(r['archive_commit_oid']) and valid_oid(r['archive_tree_oid']),
                'archive object pin is unresolved')
        parents=r['ordered_archive_parents']
        require(len(parents)==count+1 and all(valid_oid(x) for x in parents) and
                parents[0]==r['snapshot_baseline_main'] and
                set(parents[1:])==set(x['before_oid'] for x in rows), 'archive parent inventory')
        require([x['path'] for x in r['archive_docs']]==['BRANCHES.json','README.md'], 'archive document inventory')
        require(isinstance(r['protection_snapshot'],dict) and r['protection_snapshot'] and
                isinstance(r['required_success_workflows'],list) and r['required_success_workflows'],
                'rules/checks pins unresolved')
        require(r['main_protected_expected'] is (name=='kws-pipeline'), 'unexpected protection policy')
    pointer=p['pointer_zip']
    require(pointer=={'repository':'jiying2007/kws-pipeline','blob_oid':'f72723835df35121419a3c5698cebf5368f21a60',
            'bytes':827058,'sha256':'c5b466d8091f06fcdf6579770d9962b8fc44449287b9608ceb59de8f7c688e02',
            'must_be_reachable_from_archive':True}, 'pointer identity or reachability requirement changed')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise Stop('HTTP redirect refused')

class GitHub:
    """No retry. Own-repository bearer token only, never Git auth or peer auth."""
    def __init__(self, repository, token):
        require(repository in ['jiying2007/'+x for x in REPOS] and bool(token), 'transport identity')
        self.repository, self.__token = repository, token
        self.__opener=urllib.request.build_opener(NoRedirect())
    def request(self, method, path, payload=None, authenticated=True):
        require(path.startswith('/') and not path.startswith('//') and '\\' not in path,
                'API path rejected')
        if authenticated and path!='/graphql':
            require(path.startswith('/repos/'+self.repository+'/') or path=='/repos/'+self.repository,
                    'cross-repository token use rejected')
        headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10',
                 'User-Agent':'kws-one-shot-archive'}
        if authenticated: headers['Authorization']='Bearer '+self.__token
        if payload is not None:headers['Content-Type']='application/json'
        raw=None if payload is None else canonical(payload)
        request=urllib.request.Request('https://api.github.com'+path,data=raw,headers=headers,method=method)
        try:
            with self.__opener.open(request,timeout=45) as response:
                body=response.read(16*1024*1024+1)
                require(len(body)<=16*1024*1024,'API response too large')
                data=decode(body)
        except urllib.error.HTTPError as error:
            raise ApiError(error.code,[]) from None
        except (urllib.error.URLError,TimeoutError,OSError):
            raise Stop('Transport failure; write outcome may be unknown. Do not retry.') from None
        if isinstance(data,dict) and data.get('errors'): raise ApiError(200,data['errors'])
        return data
    def read(self, repository, suffix=''):
        require(repository in ['jiying2007/'+x for x in REPOS], 'read repository allowlist')
        return self.request('GET','/repos/'+repository+suffix,authenticated=(repository==self.repository))
    def pages(self, repository, path):
        rows=[]
        for page in range(1,101):
            sep='&' if '?' in path else '?'
            part=self.read(repository,path+sep+'per_page=100&page='+str(page))
            require(isinstance(part,list),'pagination response shape')
            rows.extend(part)
            if len(part)<100:return rows
        raise Stop('pagination exceeded closed limit')
    def graphql(self, query, variables):
        result=self.request('POST','/graphql',{'query':query,'variables':variables})
        require(isinstance(result.get('data'),dict),'malformed GraphQL response; outcome uncertain')
        return result['data']
    def public(self, repository, suffix):
        require(repository in ['jiying2007/'+x for x in REPOS], 'public repository allowlist')
        return self.request('GET','/repos/'+repository+suffix,authenticated=False)
    def protection_snapshot(self, repository):
        rows=self.pages(repository,'/rulesets?includes_parents=true')
        details=[]
        for row in sorted(rows,key=lambda x:x['id']):
            value=self.read(repository,'/rulesets/'+str(row['id']))
            details.append({k:value[k] for k in ('id','name','target','source_type','source','enforcement',
                                               'conditions','rules','bypass_actors')})
        branch=self.read(repository,'/branches/main')
        return {'rulesets':details,'branch_protected':branch['protected'],
                'legacy_protection_summary':branch['protection']}
    def assert_successful_workflows(self, repository, sha, required_paths, current_run_id):
        results=[]
        for page in range(1,101):
            data=self.public(repository,'/actions/runs?head_sha='+sha+'&event=push&per_page=100&page='+str(page))
            rows=data['workflow_runs'];results.extend(rows)
            if len(rows)<100:break
        else:raise Stop('workflow pagination exceeded limit')
        for path in required_paths:
            runs=[x for x in results if x['path']==path and x['head_sha']==sha
                  and x['head_branch']=='main' and x['event']=='push'
                  and str(x['id'])!=str(current_run_id)]
            require(bool(runs),'applicable exact-source workflow missing: '+path)
            latest=max(runs,key=lambda x:(x['run_number'],x.get('run_attempt',1),x['id']))
            require(latest['status']=='completed' and latest['conclusion']=='success',
                    'applicable exact-source workflow not successful: '+path)
    def assert_no_active_work(self, repository, names):
        short={x.removeprefix('refs/heads/') for x in names}
        # Public reads avoid requesting pull-requests/actions token scopes.
        for page in range(1,101):
            rows=self.public(repository,'/pulls?state=open&per_page=100&page='+str(page))
            for pr in rows:
                require(not (pr['head']['ref'] in short and pr['head']['repo'] is not None
                            and pr['head']['repo']['full_name']==repository), 'deletion candidate has open PR')
            if len(rows)<100:break
        else:raise Stop('PR pagination exceeded limit')
        for status in ('queued','in_progress','waiting','pending','requested'):
            for page in range(1,101):
                data=self.public(repository,'/actions/runs?status='+status+'&per_page=100&page='+str(page))
                rows=data['workflow_runs']
                require(not any(x['head_branch'] in short for x in rows),'deletion candidate has active workflow')
                if len(rows)<100:break
            else:raise Stop('active-run pagination exceeded limit')
    def update_refs(self, repository_node_id, updates):
        result=self.graphql('mutation($i:UpdateRefsInput!){updateRefs(input:$i){clientMutationId}}',
            {'i':{'repositoryId':repository_node_id,'refUpdates':updates}})
        require(set(result)=={'updateRefs'} and isinstance(result['updateRefs'],dict) and
                set(result['updateRefs'])=={'clientMutationId'},'malformed mutation acknowledgement; do not retry')
        return result


def guard_context(plan, ctx, own):
    r=plan['repositories'][own]
    require(ctx['repository']==r['full_name'] and ctx['repository_id']==r['repository_id'] and
            ctx['repository_owner_id']=='33591504' and ctx['actor_id']=='33591504' and
            ctx['actor']=='jiying2007' and ctx['triggering_actor']=='jiying2007', 'run identity')
    require(ctx['event_name']=='workflow_dispatch' and ctx['ref']=='refs/heads/main' and
            ctx['run_attempt']=='1' and ctx['operation'] in ('archive','prune'), 'event/phase/rerun')
    require(valid_oid(ctx['reviewed_source_sha']) and ctx['sha']==ctx['workflow_sha']==ctx['reviewed_source_sha'],
            'reviewed execution source mismatch')
    digest=ctx['reviewed_cas_signature_sha256']
    require((ctx['operation']=='archive' and digest=='') or
            (ctx['operation']=='prune' and re.fullmatch(r'[0-9a-f]{64}',digest) and digest!='0'*64),
            'archive requires empty CAS pin; prune requires reviewed signature SHA-256')
    expected=r['full_name']+'/'+plan['source_identity']['workflow_path']+'@refs/heads/main'
    require(ctx['workflow_ref']==expected,'workflow source path')


def ref_entry(name,before,after):
    return {'name':name,'beforeOid':before,'afterOid':after,'force':False}

def assert_write_scope(repo, updates, mode, wrong_before=None):
    require(updates and len({x['name'] for x in updates})==len(updates),'duplicate/empty mutation')
    A=repo['archive_commit_oid']
    if mode=='archive': expected=[ref_entry(TAG,ZERO,A)]
    elif mode=='tag-guard': expected=[ref_entry(TAG,A,A)]
    elif mode=='negative-tag-guard':
        require(valid_oid(wrong_before) and wrong_before!=A,'negative guard needs distinct nonzero source')
        expected=[ref_entry(TAG,wrong_before,A)]
    elif mode=='prune':
        expected=[ref_entry(TAG,A,A)]+[ref_entry(r['name'],r['before_oid'],ZERO) for r in repo['delete_refs']]
    else: raise Stop('unknown mutation mode')
    require(updates==expected,'mutation exceeded exact allowlist')
    require(all(x['name']!= 'refs/heads/main' and x['name'] not in HOLDS for x in updates),
            'main/held refs must never be mutation targets')

class Coordinator:
    def __init__(self, plan, context, api, restore):
        validate_plan(plan)
        self.plan,self.ctx,self.api,self.restore=plan,context,api,restore
        self.own=context['repository'].split('/')[-1]
        require(self.own in REPOS,'unknown own repo')
        guard_context(plan,context,self.own)
        self.repo=plan['repositories'][self.own]
        self.mutations=[];self.archive_source_not_ancestor=False
    def tag(self, repo):
        try: obj=self.api.read(repo['full_name'],'/git/ref/tags/archive/branches-2026-10-07')
        except ApiError as e:
            if e.status==404:return None
            raise
        require(obj['ref']==TAG and obj['object']['type']=='commit','tag must be exact lightweight commit ref')
        return obj['object']['sha']
    def branches(self, repo):
        rows=self.api.pages(repo['full_name'],'/branches')
        require(len({x['name'] for x in rows})==len(rows),'duplicate branch response')
        return {'refs/heads/'+x['name']: x['commit']['sha'] for x in rows}
    def check_source(self):
        r=self.repo;meta=self.api.read(r['full_name'])
        require(str(meta['id'])==r['repository_id'] and str(meta['owner']['id'])=='33591504'
                and meta['full_name']==r['full_name'] and meta['default_branch']=='main'
                and meta['visibility']=='public' and not meta['archived'], 'live repository identity')
        branch=self.api.read(r['full_name'],'/branches/main')
        require(branch['commit']['sha']==self.ctx['reviewed_source_sha'] and
                branch['protected'] is r['main_protected_expected'],'live main source/protection changed')
        # Explicit injected read uses GraphQL/public rules, not a permission escalation.
        require(self.api.protection_snapshot(r['full_name'])==r['protection_snapshot'],
                'protection/rules changed')
        self.api.assert_successful_workflows(r['full_name'],self.ctx['reviewed_source_sha'],
                                            r['required_success_workflows'],self.ctx['run_id'])
        self.node_id=meta['node_id']
    def check_original_heads(self, branches):
        for row in frozen_rows(self.repo):
            require(branches.get(row['name'])==row['before_oid'],'original head changed or missing: '+row['name'])
        self.api.assert_no_active_work(self.repo['full_name'],[r['name'] for r in self.repo['delete_refs']])
    def verify_server_archive(self, repo):
        commit=self.api.read(repo['full_name'],'/git/commits/'+repo['archive_commit_oid'])
        require(commit['sha']==repo['archive_commit_oid'] and commit['tree']['sha']==repo['archive_tree_oid']
                and [p['sha'] for p in commit['parents']]==repo['ordered_archive_parents'], 'server archive topology')
        tree=self.api.read(repo['full_name'],'/git/trees/'+repo['archive_tree_oid'])
        require(not tree.get('truncated',False),'truncated tree')
        expected=[{'path':x['path'],'mode':x['mode'],'type':x['type'],'sha':x['git_blob_oid']} for x in repo['archive_docs']]
        actual=[{k:x[k] for k in ('path','mode','type','sha')} for x in tree['tree']]
        require(actual==expected,'archive tree is not the two exact documents')
        for row in repo['archive_docs']:
            blob=self.api.read(repo['full_name'],'/git/blobs/'+row['git_blob_oid'])
            require(blob['encoding']=='base64','blob encoding')
            raw=base64.b64decode(blob['content'].replace('\n',''),validate=True)
            require(len(raw)==row['bytes'] and sha256(raw)==row['sha256'],'archive document bytes')
    def write(self, mode, updates):
        assert_write_scope(self.repo,updates,mode,self.ctx['reviewed_source_sha'])
        self.mutations.append({'mode':mode,'updates':updates})
        return self.api.update_refs(self.node_id,updates)
    def prove_both(self):
        proofs={}
        for name,repo in self.plan['repositories'].items():
            require(self.tag(repo)==repo['archive_commit_oid'],'both exact tags required')
            self.verify_server_archive(repo)
            proof=self.restore(repo,self.plan)
            require(proof['status']=='PASS' and proof['archive_oid']==repo['archive_commit_oid']
                    and proof['restored_heads']==repo['original_count']
                    and proof['full_object_identity'] is True and proof['restored_trees'] is True
                    and proof['tag_only_fetch'] is True and proof['external_payload_gap'] is False,
                    'incomplete real restoration proof')
            if repo['full_name']==self.plan['pointer_zip']['repository']:
                require(proof['pointer_zip'] is True,'pointer ZIP proof missing')
            if name==self.own:
                require(proof['wrong_before_not_ancestor'] is True,'workflow source is an archive ancestor')
                self.archive_source_not_ancestor=True
            proofs[name]=proof
        return proofs
    def verify_noop_guard(self, diagnostic_only=False):
        A=self.repo['archive_commit_oid']
        require(self.archive_source_not_ancestor,'safe nonzero wrong-before proof missing')
        require(self.tag(self.repo)==A,'tag changed before guard test')
        self.write('tag-guard',[ref_entry(TAG,A,A)])
        require(self.tag(self.repo)==A,'tag changed after same-value guard')
        expected=self.ctx['reviewed_cas_signature_sha256']
        if not diagnostic_only:
            require(bool(re.fullmatch(r'[0-9a-f]{64}',expected)) and expected!='0'*64,
                    'reviewed CAS signature SHA-256 missing; prune blocked')
        try:self.write('negative-tag-guard',[ref_entry(TAG,self.ctx['reviewed_source_sha'],A)])
        except ApiError as error:
            actual=cas_signature(error.errors,self.ctx['reviewed_source_sha'],A)
            if diagnostic_only:
                require(error.status==200 and actual,'negative guard did not return a GraphQL rejection')
                require(self.tag(self.repo)==A,'tag changed during guard diagnostic')
                return {'status':'REQUIRES_REVIEW','observed_error_signature':actual,
                        'signature_sha256':sha256(canonical(actual)),
                        'explicit_expected_oid_evidence':explicit_cas_signature(actual)}
            require(error.status==200 and explicit_cas_signature(actual) and sha256(canonical(actual))==expected,
                    'wrong-before test did not return reviewed explicit CAS mismatch')
        else:raise Stop('server ignored mismatched beforeOid; deletion blocked')
        require(self.tag(self.repo)==A,'tag changed during guard test')
    def run(self):
        self.check_source();before=self.branches(self.repo);self.check_original_heads(before)
        if self.ctx['operation']=='archive':
            require(self.tag(self.repo) is None,'archive phase already consumed; existing tag stops')
            self.verify_server_archive(self.repo)
            self.check_source();self.check_original_heads(self.branches(self.repo))
            self.write('archive',[ref_entry(TAG,ZERO,self.repo['archive_commit_oid'])])
            require(self.tag(self.repo)==self.repo['archive_commit_oid'],'archive tag readback failed')
            proof=self.restore(self.repo,self.plan)
            require(proof['status']=='PASS','archive restoration failed')
            require(self.branches(self.repo)==before,'branch changed during archive; reconcile read-only')
            require(proof['wrong_before_not_ancestor'] is True,'workflow source is an archive ancestor')
            self.archive_source_not_ancestor=True
            diagnostic=self.verify_noop_guard(diagnostic_only=True)
            return {'phase':'archive','proof':proof,'deletions':0,'tag_guard_diagnostic':diagnostic}
        proofs=self.prove_both()
        self.check_source();self.check_original_heads(self.branches(self.repo))
        require(self.branches(self.repo)==before,'branch inventory changed before prune')
        self.verify_noop_guard()
        self.check_source();final_branches=self.branches(self.repo)
        require(final_branches==before,'branch inventory changed after guard tests')
        self.check_original_heads(final_branches)
        updates=[ref_entry(TAG,self.repo['archive_commit_oid'],self.repo['archive_commit_oid'])]
        updates += [ref_entry(r['name'],r['before_oid'],ZERO) for r in self.repo['delete_refs']]
        for repo in self.plan['repositories'].values():
            require(self.tag(repo)==repo['archive_commit_oid'],'archive tag changed after cross-repository proof')
        self.write('prune',updates)
        after=self.branches(self.repo)
        expected={k:v for k,v in before.items() if k not in {x['name'] for x in self.repo['delete_refs']}}
        require(after==expected,'post-prune branch inventory differs; reconcile read-only')
        require(self.tag(self.repo)==self.repo['archive_commit_oid'],'post-prune tag mismatch')
        self.check_source()
        post=self.prove_both()
        return {'phase':'prune','deleted_refs':self.repo['delete_count'],'before_proofs':proofs,'after_proofs':post}

class GitRestorer:
    """Fetches only a public archive tag; never checks out or runs historical files."""
    def __init__(self, source_sha=None):
        self.deadline=time.monotonic()+1200;self.source_sha=source_sha
    def fetch_tag(self, command, repo):
        command('fetch','--quiet','--no-tags','--no-recurse-submodules',
                'https://github.com/'+repo['full_name']+'.git',TAG+':'+TAG)
    def __call__(self, repo, plan):
        with tempfile.TemporaryDirectory(prefix='kws-tag-proof-') as directory:
            return self._restore(Path(directory),repo,plan)
    def _restore(self, root, repo, plan):
        limits=plan['resource_limits'];bare=root/'archive.git';home=root/'home';home.mkdir()
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(home),
             'LANG':'C','LC_ALL':'C','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null',
             'GIT_TERMINAL_PROMPT':'0','GIT_NO_REPLACE_OBJECTS':'1','GIT_LFS_SKIP_SMUDGE':'1'}
        prefix=['git','-c','core.hooksPath=/dev/null','-c','credential.helper=',
                '-c','protocol.file.allow=never','-c','protocol.ext.allow=never',
                '-c','protocol.https.allow=always','-c','http.followRedirects=false',
                '-c','fetch.fsckObjects=true','-c','transfer.fsckObjects=true','--git-dir='+str(bare)]
        def budget():
            require(time.monotonic()<self.deadline,'restoration deadline exceeded')
            size=0
            for path in root.rglob('*'):
                try:
                    if path.is_file():size+=path.stat().st_size
                except FileNotFoundError:pass  # Git may atomically rename a just-counted pack
            require(size<=limits['max_disk_bytes'],'restoration disk budget exceeded')
        def child_limits():
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE,(limits['max_disk_bytes'],limits['max_disk_bytes']))
            resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        def command(*args, extra_env=None):
            budget()
            merged={**env,**(extra_env or {})}
            out=root/'command.stdout';err=root/'command.stderr'
            with out.open('wb') as stdout,err.open('wb') as stderr:
                process=subprocess.Popen(prefix+list(args),env=merged,stdout=stdout,stderr=stderr,
                                         start_new_session=True,preexec_fn=child_limits)
                try:
                    while process.poll() is None:
                        budget();time.sleep(.05)
                    budget();require(process.returncode==0,'Git verification command failed: '+args[0])
                except BaseException:
                    if process.poll() is None:
                        import signal
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            require(out.stat().st_size<=32*1024*1024,'Git output budget exceeded')
            result=out.read_bytes();out.unlink();err.unlink();return result
        command('init','--bare','--quiet',str(bare))
        self.fetch_tag(command,repo)
        require(command('for-each-ref','--format=%(refname)').decode().splitlines()==[TAG],
                'fetch contained other refs')
        require(not (bare/'shallow').exists() and not (bare/'objects/info/alternates').exists(),
                'shallow or alternate recovery not accepted')
        require(command('rev-parse',TAG).decode().strip()==repo['archive_commit_oid'],'fetched tag OID')
        require(command('show','-s','--format=%P',TAG).decode().strip().split()==repo['ordered_archive_parents'],
                'fetched archive parents')
        require(command('rev-parse',TAG+'^{tree}').decode().strip()==repo['archive_tree_oid'],'fetched archive tree')
        require(command('ls-tree','--name-only',TAG).decode().splitlines()==['BRANCHES.json','README.md'],
                'fetched archive contains extra files')
        for doc in repo['archive_docs']:
            raw=command('show',TAG+':'+doc['path'])
            require(len(raw)==doc['bytes'] and sha256(raw)==doc['sha256'],'fetched document bytes')
        command('fsck','--full','--strict','--no-reflogs')
        objects=command('rev-list','--objects','--no-object-names',TAG).decode().splitlines()
        require(objects and len(objects)==len(set(objects)) and len(objects)<=limits['max_objects']
                and all(valid_oid(x) for x in objects),'object inventory limit/identity')
        pointer=plan['pointer_zip'];pointer_ok=False;byte_count=0;blob_count=0;digest=hashlib.sha256()
        stderr=(root/'batch.stderr').open('wb')
        process=subprocess.Popen(prefix+['cat-file','--batch'],env=env,stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,stderr=stderr,start_new_session=True,preexec_fn=child_limits)
        try:
            for number,oid in enumerate(objects):
                if number%128==0:budget()
                process.stdin.write((oid+'\n').encode());process.stdin.flush()
                header=process.stdout.readline(256).decode().strip().split()
                require(len(header)==3 and header[0]==oid and header[1] in ('commit','tree','blob','tag'),
                        'object response identity')
                kind=header[1];size=int(header[2]);byte_count+=size
                require(size>=0 and byte_count<=limits['max_object_bytes'],'object-byte budget exceeded')
                git_hash=hashlib.sha1((kind+' '+str(size)).encode()+b'\0');content_hash=hashlib.sha256()
                left=size;start=b'';tree=bytearray() if kind=='tree' else None
                while left:
                    chunk=process.stdout.read(min(left,1024*1024));require(chunk,'truncated object stream')
                    if len(start)<4096:start=(start+chunk)[:4096]
                    git_hash.update(chunk);content_hash.update(chunk);left-=len(chunk)
                    if tree is not None:
                        require(len(tree)+len(chunk)<=32*1024*1024,'tree object budget exceeded');tree.extend(chunk)
                require(process.stdout.read(1)==b'\n' and git_hash.hexdigest()==oid,'actual Git object bytes mismatch')
                if kind=='blob':
                    blob_count+=1
                    require(not (start.startswith((b'version https://git-lfs.github.com/spec/v1',
                               b'version https://hawser.github.com/spec/v1',b'version http://git-media.io/v/2'))
                               or (start.startswith(b'version ') and b'oid sha256:' in start and b'\nsize ' in start)),
                            'LFS pointer found; external bytes not archived')
                if tree is not None:
                    cursor=0
                    while cursor<len(tree):
                        end=tree.find(0,cursor);require(end>=0 and end+21<=len(tree),'invalid tree framing')
                        entry=tree[cursor:end];mode,sep,name=entry.partition(b' ')
                        require(sep and name and mode in (b'100644',b'100755',b'40000',b'120000'),
                                'gitlink or unsupported tree mode; external payload gap')
                        cursor=end+21
                    require(cursor==len(tree),'tree object trailing bytes')
                digest.update(canonical({'oid':oid,'type':kind,'bytes':size,'sha256':content_hash.hexdigest()})+b'\n')
                if oid==pointer['blob_oid']:
                    require(kind=='blob' and size==pointer['bytes'] and content_hash.hexdigest()==pointer['sha256'],
                            'pointer ZIP bytes mismatch');pointer_ok=True
            process.stdin.close();process.wait(timeout=max(1,self.deadline-time.monotonic()))
            require(process.returncode==0,'object stream command failed')
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            stderr.close()
        if repo['full_name']==pointer['repository']:
            require(pointer_ok,'pointer ZIP absent from archive reachability')
            for source in ('3385bf2a7f6838b743546f2ee9f481e2fa7ee975','94b9fbf562aca9487beb8096bc3f50924369a3ce'):
                entry=command('ls-tree',source,'--','research/fixed30_asr/fixed30-blind-inputs.zip').decode().strip()
                require(entry=='100644 blob '+pointer['blob_oid']+'\tresearch/fixed30_asr/fixed30-blind-inputs.zip',
                        'pointer ZIP original source/path binding')
        for index,row in enumerate(frozen_rows(repo)):
            restored='refs/heads/restore/'+str(index)
            command('update-ref',restored,row['before_oid'])  # local bare repository only
            require(command('rev-parse',restored+'^{tree}').decode().strip()==row['tree_oid'],'restored tree identity')
            index_file=root/('index-'+str(index))
            extra={'GIT_INDEX_FILE':str(index_file)}
            command('read-tree',restored,extra_env=extra)
            require(command('write-tree',extra_env=extra).decode().strip()==row['tree_oid'],'reconstructed tree identity')
            index_file.unlink()
        budget()
        return {'status':'PASS','archive_oid':repo['archive_commit_oid'],'restored_heads':len(frozen_rows(repo)),
                'full_object_identity':True,'restored_trees':True,'tag_only_fetch':True,'external_payload_gap':False,
                'pointer_zip':pointer_ok,'wrong_before_not_ancestor':bool(valid_oid(self.source_sha)) and self.source_sha not in objects,'objects':len(objects),'blobs':blob_count,'object_bytes':byte_count,
                'object_receipt_sha256':digest.hexdigest()}


def main():
    import signal
    def timeout(*_):raise Stop('overall operation deadline exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    plan_path=Path(os.environ['ARCHIVE_PLAN_FILE']);raw=plan_path.read_bytes()
    require(sha256(raw)==os.environ['ARCHIVE_PLAN_SHA256'],'plan source hash')
    require(sha256(Path(__file__).read_bytes())==os.environ['ARCHIVE_SCRIPT_SHA256'],'script source hash')
    keys=('repository','repository_id','repository_owner_id','actor_id','actor','triggering_actor',
          'event_name','ref','run_attempt','operation','reviewed_source_sha','reviewed_cas_signature_sha256','sha','workflow_sha','workflow_ref','run_id')
    context={key:os.environ['ARCHIVE_CTX_'+key.upper()] for key in keys}
    token=os.environ.pop('GH_TOKEN','')
    # Never inherit the job token into Git subprocesses or save it to any file.
    api=GitHub(context['repository'],token)
    coordinator=Coordinator(decode(raw),context,api,GitRestorer(context['reviewed_source_sha']))
    result=coordinator.run()
    print(json.dumps(result,sort_keys=True))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
            f.write('Verified archival operation\n\n'+json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except (Stop,KeyError,ValueError,TypeError,subprocess.SubprocessError) as error:
        print('STOP: '+str(error)+'; no automatic retry. Reconcile server state read-only.')
        raise SystemExit(1)
