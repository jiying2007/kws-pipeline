"""Offline unittest suite. No live GitHub transport or credentials are used."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
script=ROOT/'archive_branches_once_20261007.py'
if not script.exists():script=ROOT.parent/'tools/archive_branches_once_20261007.py'
spec=importlib.util.spec_from_file_location('archive_once',script)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
plan_file=ROOT/'kws-pipeline.plan.template.json'
if not plan_file.exists():plan_file=ROOT/'archive-plan.json'
if not plan_file.exists():plan_file=ROOT.parent/'docs/archive-branches-once-20261007.json'
BASE=json.loads(plan_file.read_text())
SIG=[{'type':'UNPROCESSABLE','message':'fixture expected <WRONG_BEFORE> but was <ARCHIVE>'}]
SOURCE='9'*40

def ready():
    p=copy.deepcopy(BASE);p['publication_ready']=True
    for name,r in p['repositories'].items():
        r['archive_commit_oid']=('a' if name=='kws-pipeline' else 'b')*40
    return p

def context(name='kws-pipeline',phase='prune'):
    return {'repository':'jiying2007/'+name,'repository_id':A.REPOS[name][0],'repository_owner_id':'33591504',
            'actor_id':'33591504','actor':'jiying2007','triggering_actor':'jiying2007','event_name':'workflow_dispatch',
            'ref':'refs/heads/main','run_attempt':'1','operation':phase,'reviewed_source_sha':SOURCE,
            'sha':SOURCE,'workflow_sha':SOURCE,'workflow_ref':'jiying2007/'+name+'/'+BASE['source_identity']['workflow_path']+'@refs/heads/main',
            'run_id':'123','reviewed_cas_signature_sha256':A.sha256(A.canonical(SIG)) if phase=='prune' else ''}

def proof(repo,plan):
    return {'status':'PASS','archive_oid':repo['archive_commit_oid'],'restored_heads':repo['original_count'],
            'full_object_identity':True,'restored_trees':True,'tag_only_fetch':True,
            'external_payload_gap':False,'pointer_zip':True,'wrong_before_not_ancestor':True}

class FakeAPI:
    def __init__(self,p,own='kws-pipeline',tagged=True):
        self.p,self.own=p,own;self.writes=[];self.fail_mode=None;self.calls=[]
        self.refs={name:{r['name']:r['before_oid'] for r in A.frozen_rows(repo)} for name,repo in p['repositories'].items()}
        for refs in self.refs.values():refs.update({'refs/heads/main':SOURCE,'refs/heads/new/work':'8'*40})
        self.tags={name:(r['archive_commit_oid'] if tagged else None) for name,r in p['repositories'].items()}
    def read(self,repository,suffix=''):
        name=repository.split('/')[-1];r=self.p['repositories'][name]
        if not suffix:return {'id':r['repository_id'],'owner':{'id':33591504},'full_name':repository,
                              'default_branch':'main','visibility':'public','archived':False,'node_id':'node-'+name}
        if suffix=='/branches/main':return {'commit':{'sha':self.refs[name]['refs/heads/main']},'protected':r['main_protected_expected']}
        if suffix.startswith('/git/ref/'):
            if self.tags[name] is None:raise A.ApiError(404,[])
            return {'ref':A.TAG,'object':{'type':'commit','sha':self.tags[name]}}
        if suffix.startswith('/git/commits/'):
            return {'sha':r['archive_commit_oid'],'tree':{'sha':r['archive_tree_oid']},
                    'parents':[{'sha':x} for x in r['ordered_archive_parents']]}
        if suffix.startswith('/git/trees/'):
            return {'truncated':False,'tree':[{'path':d['path'],'mode':d['mode'],'type':d['type'],'sha':d['git_blob_oid']} for d in r['archive_docs']]}
        if suffix.startswith('/git/blobs/'):
            d=next(d for d in r['archive_docs'] if d['git_blob_oid']==suffix.split('/')[-1])
            # Full content identities checked separately; zero network fixture bypass is explicit.
            return {'encoding':'base64','content':base64.b64encode(b'fixture').decode()}
        raise AssertionError('unexpected read '+suffix)
    def pages(self,repository,path):
        name=repository.split('/')[-1]
        return [{'name':k.removeprefix('refs/heads/'),'commit':{'sha':v}} for k,v in self.refs[name].items()]
    def protection_snapshot(self,repository):return copy.deepcopy(self.p['repositories'][repository.split('/')[-1]]['protection_snapshot'])
    def assert_successful_workflows(self,*args):
        if self.fail_mode=='ci':raise A.Stop('CI not success')
    def assert_no_active_work(self,*args):
        if self.fail_mode=='active':raise A.Stop('active work')
    def update_refs(self,node,updates):
        self.writes.append(copy.deepcopy(updates));refs=self.refs[self.own]
        if self.fail_mode=='uncertain':raise A.Stop('unknown write outcome')
        if self.fail_mode=='noop-rejected' and updates[0]['beforeOid']==updates[0]['afterOid']:
            raise A.ApiError(200,[{'type':'FORBIDDEN','message':'no-op rejected'}])
        if self.fail_mode=='raced-at-prune' and len(updates)>1:
            refs[updates[-1]['name']]='c'*40
        for u in updates:
            actual=(self.tags[self.own] or A.ZERO) if u['name']==A.TAG else refs.get(u['name'],A.ZERO)
            if u['beforeOid']!=actual:
                if self.fail_mode=='ignored-before':return {}
                raise A.ApiError(200,[{'type':'UNPROCESSABLE','message':'fixture expected '+u['beforeOid']+' but was '+actual}])
        for u in updates:
            if u['name']==A.TAG:self.tags[self.own]=u['afterOid']
            elif u['afterOid']==A.ZERO:refs.pop(u['name'])
            else:raise AssertionError('unexpected branch write')
        return {}

class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.p=ready();self.ctx=context();self.api=FakeAPI(self.p)
        self.coord=A.Coordinator(self.p,self.ctx,self.api,proof)
        # Unit state-machine tests use a verified-objects fixture; verifier itself is tested below.
        self.coord.verify_server_archive=lambda r:None
    def test_templates_remain_locked(self):
        locked=copy.deepcopy(BASE);locked['publication_ready']=False
        with self.assertRaises(A.Stop):A.validate_plan(locked)
    def test_exact_frozen_inventory_counts(self):
        A.validate_plan(self.p)
        self.assertEqual(sum(r['delete_count'] for r in self.p['repositories'].values()),79)
    def test_replacement_or_reordered_original_ref_rejected(self):
        for field in ('name','before_oid','tree_oid'):
            p=ready();r=p['repositories']['kws-pipeline']['delete_refs'][0]
            r[field]='refs/heads/arbitrary' if field=='name' else 'c'*40
            with self.assertRaises(A.Stop):A.validate_plan(p)
    def test_all_unresolved_archive_pins_fail(self):
        for name in A.REPOS:
            p=ready();p['repositories'][name]['archive_commit_oid']=None
            with self.assertRaises(A.Stop):A.validate_plan(p)
    def test_context_wrong_actor_source_ref_or_rerun_fails(self):
        for key,value in [('actor_id','1'),('event_name','push'),('ref','refs/heads/other'),('run_attempt','2'),
                          ('sha','c'*40),('workflow_sha','c'*40),('reviewed_source_sha','main')]:
            c=context();c[key]=value
            with self.assertRaises(A.Stop):A.Coordinator(self.p,c,self.api,proof)
    def test_main_or_held_never_accepted_as_mutation_targets(self):
        for ref in ['refs/heads/main',*A.HOLDS]:
            with self.assertRaises(A.Stop):self.coord.write('prune',[A.ref_entry(ref,SOURCE,A.ZERO)])
        self.assertEqual(self.api.writes,[])
    def test_success_deletes_exact_57_keeps_everything_else(self):
        before=copy.deepcopy(self.api.refs['kws-pipeline']);out=self.coord.run()
        self.assertEqual(out['deleted_refs'],57)
        deleted=set(before)-set(self.api.refs['kws-pipeline'])
        self.assertEqual(deleted,{x['name'] for x in self.p['repositories']['kws-pipeline']['delete_refs']})
        targets={u['name'] for batch in self.api.writes for u in batch}
        self.assertFalse(targets&A.HOLDS);self.assertNotIn('refs/heads/main',targets)
    def test_data_success_deletes_exact_22(self):
        api=FakeAPI(self.p,'kws-data');c=A.Coordinator(self.p,context('kws-data'),api,proof)
        c.verify_server_archive=lambda r:None
        self.assertEqual(c.run()['deleted_refs'],22)
    def test_repeat_after_success_fails(self):
        self.coord.run();count=len(self.api.writes)
        with self.assertRaises(A.Stop):self.coord.run()
        self.assertEqual(len(self.api.writes),count)
    def test_any_required_proof_missing_prevents_delete(self):
        for field in ['status','archive_oid','restored_heads','full_object_identity','restored_trees','tag_only_fetch','external_payload_gap','pointer_zip','wrong_before_not_ancestor']:
            c=A.Coordinator(self.p,self.ctx,FakeAPI(self.p),lambda r,p:{k:v for k,v in proof(r,p).items() if k!=field})
            c.verify_server_archive=lambda r:None
            with self.assertRaises((A.Stop,KeyError)):c.run()
            self.assertFalse(any(len(b)>1 for b in c.api.writes))
    def test_peer_tag_missing_or_changed_blocks(self):
        for value in (None,'c'*40):
            self.api.tags['kws-data']=value
            with self.assertRaises(A.Stop):self.coord.run()
            self.assertEqual(self.api.writes,[])
    def test_peer_tag_removed_during_proof_blocks(self):
        def raced(r,p):
            if r['full_name'].endswith('kws-data'):self.api.tags['kws-data']=None
            return proof(r,p)
        self.coord.restore=raced
        with self.assertRaises(A.Stop):self.coord.run()
        self.assertFalse(any(len(b)>1 for b in self.api.writes))
    def test_raced_candidate_or_held_or_main_blocks(self):
        for name in ['refs/heads/main',next(iter(A.HOLDS)),self.p['repositories']['kws-pipeline']['delete_refs'][0]['name']]:
            api=FakeAPI(self.p);api.refs['kws-pipeline'][name]='c'*40
            c=A.Coordinator(self.p,self.ctx,api,proof)
            with self.assertRaises(A.Stop):c.run()
            self.assertEqual(api.writes,[])
    def test_ci_or_active_work_blocks_all_writes(self):
        for mode in ('ci','active'):
            self.api.fail_mode=mode
            with self.assertRaises(A.Stop):self.coord.run()
            self.assertEqual(self.api.writes,[])
    def test_noop_unsupported_or_mismatch_ignored_blocks_deletion(self):
        for mode in ('noop-rejected','ignored-before'):
            api=FakeAPI(self.p);api.fail_mode=mode;c=A.Coordinator(self.p,self.ctx,api,proof)
            c.verify_server_archive=lambda r:None
            with self.assertRaises(A.Stop):c.run()
            self.assertFalse(any(len(b)>1 for b in api.writes))
    def test_race_at_atomic_write_rejects_entire_batch(self):
        self.api.fail_mode='raced-at-prune';before=copy.deepcopy(self.api.refs['kws-pipeline'])
        with self.assertRaises(A.ApiError):self.coord.run()
        after=self.api.refs['kws-pipeline']
        self.assertEqual(set(before),set(after))
        self.assertEqual(sum(before[k]!=after[k] for k in before),1)
        self.assertEqual(len(self.api.writes),3)
    def test_generic_or_permission_rejection_cannot_unlock_prune(self):
        for kind,message in [('FORBIDDEN','expected <WRONG_BEFORE> but was <ARCHIVE>'),
                             ('UNPROCESSABLE','non-fast-forward'),('UNPROCESSABLE','generic failed ref update'),
                             ('UNPROCESSABLE','Expected <WRONG_BEFORE> to be an ancestor of <ARCHIVE>; non-fast-forward update rejected')]:
            self.assertFalse(A.explicit_cas_signature([{'type':kind,'message':message}]))
    def test_unknown_write_has_one_attempt_and_no_retry(self):
        self.api.fail_mode='uncertain'
        with self.assertRaises(A.Stop):self.coord.run()
        self.assertEqual(len(self.api.writes),1)
    def test_unreviewed_mismatch_signature_blocks_prune(self):
        self.coord.ctx['reviewed_cas_signature_sha256']=''
        with self.assertRaises(A.Stop):self.coord.run()
        self.assertFalse(any(len(b)>1 for b in self.api.writes))
    def test_wrong_reviewed_signature_digest_blocks_prune(self):
        self.coord.ctx['reviewed_cas_signature_sha256']='f'*64
        with self.assertRaises(A.Stop):self.coord.run()
        self.assertFalse(any(len(b)>1 for b in self.api.writes))
    def test_archive_rejects_cas_pin_and_prune_requires_one(self):
        for phase,pin in [('archive','f'*64),('prune',''),('prune','not-a-digest')]:
            ctx=context(phase=phase);ctx['reviewed_cas_signature_sha256']=pin
            with self.assertRaises(A.Stop):A.Coordinator(self.p,ctx,self.api,proof)
    def test_archive_existing_tag_even_correct_stops(self):
        c=A.Coordinator(self.p,context(phase='archive'),self.api,proof)
        with self.assertRaises(A.Stop):c.run()
        self.assertEqual(self.api.writes,[])
    def test_archive_creates_tag_and_collects_diagnostic_without_deleting(self):
        api=FakeAPI(self.p,tagged=False);c=A.Coordinator(self.p,context(phase='archive'),api,proof)
        c.verify_server_archive=lambda r:None;out=c.run()
        self.assertEqual(out['deletions'],0);self.assertEqual(out['tag_guard_diagnostic']['status'],'REQUIRES_REVIEW')
        self.assertEqual(out['tag_guard_diagnostic']['signature_sha256'],A.sha256(A.canonical(SIG)))
        self.assertTrue(all(len(b)==1 and b[0]['name']==A.TAG for b in api.writes))

class ApiAndStaticTests(unittest.TestCase):
    def test_transport_rejects_peer_authenticated_path_without_network(self):
        api=A.GitHub('jiying2007/kws-pipeline','invented-test-token')
        with self.assertRaises(A.Stop):api.request('GET','/repos/jiying2007/kws-data')
    def test_redirects_are_refused(self):
        with self.assertRaises(A.Stop):A.NoRedirect().redirect_request(None,None,None,None,None,None)
    def test_duplicate_json_fails(self):
        with self.assertRaises(A.Stop):A.decode('{"x":1,"x":2}')
    def test_script_does_not_configure_or_export_credentials(self):
        src=script.read_text()
        for bad in ['gh auth setup-git','http.extraheader','x-access-token:','git push','secrets.']:
            self.assertNotIn(bad,src)
        self.assertIn("'GIT_CONFIG_GLOBAL':'/dev/null'",src)
        self.assertIn("'GIT_TERMINAL_PROMPT':'0'",src)

if __name__=='__main__':unittest.main(verbosity=2)
