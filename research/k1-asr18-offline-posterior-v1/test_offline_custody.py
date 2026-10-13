"""Local files and isolated CLI tests. All records are explicitly synthetic."""
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import asr_posterior as a
import fixture_cases as f
CLI_FLAGS=['-I','-S','-B']+(['-'+'O'*sys.flags.optimize] if sys.flags.optimize else [])


def write_bundle(root, case):
    root=Path(root);root.mkdir(exist_ok=True);known=a.locators(case.expected);rows=[]
    bodies={k:(v if type(v) is bytes else a.canonical(v)) for k,v in case.records.items() if k in known}
    bodies.update(outer_manifest=case.outer_raw,archive=case.payload,producer_log=case.logs.get('producer',b''))
    if case.expected['recovery'] is not None:bodies['recovery_log']=case.logs.get('recovery',b'')
    bodies.update({'chunk_%04d'%i:raw for i,(_,raw) in enumerate(case.chunks)})
    for role,body in sorted(bodies.items()):
        path=role+'.bin';(root/path).write_bytes(body)
        rows.append({'role':role,'path':path,'bytes':len(body),'sha256':a.sha(body),'locator':sorted(known[role])[0]})
    (root/'expected.json').write_bytes(a.canonical(case.expected))
    (root/'custody.json').write_bytes(a.canonical({'schema':'k1-asr18-offline-custody-v1','entries':rows}))
    return dict(root=root,expected_path=root/'expected.json',expected_sha256=a.sha((root/'expected.json').read_bytes()),custody_path=root/'custody.json',custody_sha256=a.sha((root/'custody.json').read_bytes()))

class CustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.normal=f.normal_case();cls.recovered=f.recovery_case()
    def bundle(self,case=None):
        directory=tempfile.TemporaryDirectory(prefix='asr-readonly-');self.addCleanup(directory.cleanup)
        return write_bundle(Path(directory.name),case or self.normal)
    def run_cli(self,args):
        argv=[sys.executable,*CLI_FLAGS,str(a.HERE/'asr_posterior.py'),'--evidence-root',str(args['root']),'--expected',str(args['expected_path']),'--expected-sha256',args['expected_sha256'],'--custody',str(args['custody_path']),'--custody-sha256',args['custody_sha256']]
        return subprocess.run(argv,capture_output=True,timeout=90,env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'})
    def test_complete_normal_real_cli(self):
        args=self.bundle();r=self.run_cli(args);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        x=a.decode(r.stdout);self.assertEqual(x['status'],'complete_native_verified');self.assertEqual(x['verified_complete_slots'],18)
        self.assertEqual(x['expected_sha256'],args['expected_sha256']);self.assertEqual(x['custody_sha256'],args['custody_sha256'])
    def test_complete_recovered_real_cli_preserves_failure(self):
        args=self.bundle(self.recovered);r=self.run_cli(args);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        x=a.decode(r.stdout);self.assertEqual(x['producer_terminal']['publisher_step_conclusion'],'failure');self.assertEqual(x['recovery_terminal'],{'run':'success','job':'success','step':'success'})
        self.assertFalse(x['original_publisher_reclassified']);self.assertFalse(x['training_admitted'])
    def test_partial_preflight_real_cli(self):
        args=self.bundle(f.normal_case('all_notrun'));r=self.run_cli(args);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        x=a.decode(r.stdout);self.assertEqual(x['status'],'partial_native_verified');self.assertEqual(x['ledger_reported_attempts'],0);self.assertIsNone(x['consumed_attempts']);self.assertFalse(x['input_verified_by_original_host'])
    def test_cli_actual_null_template_blocks(self):
        args=self.bundle();args['expected_path'].write_bytes(a.canonical({'schema':'k1-asr18-offline-acceptance-v1','mode':None,'producer':None,'archive':None,'result':None,'input_authorities':None,'recovery':None}));args['expected_sha256']=a.sha(args['expected_path'].read_bytes());r=self.run_cli(args);self.assertEqual(r.returncode,2);self.assertFalse(a.decode(r.stdout)['native_success_established'])
    def test_pinned_request_and_custody_cannot_drift(self):
        args=self.bundle()
        for key in ('expected_sha256','custody_sha256'):
            wrong=dict(args,**{key:'0'*64})
            with self.subTest(key=key),self.assertRaises(ValueError):a.load_and_accept(**wrong)
    def test_original_body_drift_and_no_body_reject(self):
        args=self.bundle();p=args['root']/'chunk_0000.bin';raw=p.read_bytes();p.write_bytes(b'x'+raw[1:])
        with self.assertRaises(ValueError):a.load_and_accept(**args)
        p.unlink()
        with self.assertRaises(OSError):a.load_and_accept(**args)
    def mutate_index(self,args,fn):
        x=a.decode(args['custody_path'].read_bytes());fn(x);args['custody_path'].write_bytes(a.canonical(x));args['custody_sha256']=a.sha(args['custody_path'].read_bytes())
    def test_index_path_locator_extra_duplicate_role_reject(self):
        for mode in ('traversal','absolute','alias','locator','duplicate','extra'):
            with self.subTest(mode=mode):
                args=self.bundle()
                def change(x):
                    row=x['entries'][0]
                    if mode=='traversal':row['path']='../outer'
                    elif mode=='absolute':row['path']='/tmp/outer'
                    elif mode=='alias':row['path']='folder/../outer'
                    elif mode=='locator':row['locator']='https://example.invalid/private'
                    elif mode=='duplicate':x['entries'].append(copy.deepcopy(row))
                    else:x['entries'].append({**row,'role':'caller_success_boolean'})
                self.mutate_index(args,change)
                with self.assertRaises(ValueError):a.load_and_accept(**args)
    def test_symlink_hardlink_fifo_fail_without_reading_special_file(self):
        for mode in ('symlink','hardlink','fifo'):
            with self.subTest(mode=mode):
                args=self.bundle();path=args['root']/'archive.bin';old=path.read_bytes();path.unlink()
                if mode=='symlink':
                    target=args['root']/'body';target.write_bytes(old);path.symlink_to(target)
                elif mode=='hardlink':
                    target=args['root']/'body';target.write_bytes(old);os.link(target,path)
                else:os.mkfifo(path)
                r=self.run_cli(args);self.assertEqual(r.returncode,2,r.stdout+r.stderr);self.assertEqual(a.decode(r.stdout)['status'],'blocked_or_unconfirmed')
    def test_foreign_nonnull_stdout_rejected_through_cli(self):
        case=copy.deepcopy(self.normal);case.logs['producer']=a.canonical({'schema':'k1-asr18-safe-checkpoint-v1','identity':{**case.expected['producer']['identity'],'run_id':124},'stage':'complete'})+b'\n'
        r=self.run_cli(self.bundle(case));self.assertEqual(r.returncode,2)
    def test_bad_cli_arguments_and_source_pin_fail_safely(self):
        r=subprocess.run([sys.executable,*CLI_FLAGS,str(a.HERE/'asr_posterior.py'),'--SECRET_NEVER_EMIT'],capture_output=True,timeout=30)
        self.assertEqual(r.returncode,2);self.assertNotIn(b'SECRET_NEVER_EMIT',r.stdout+r.stderr);self.assertEqual(a.decode(r.stdout)['status'],'blocked_or_unconfirmed')
        with tempfile.TemporaryDirectory(prefix='bad-validator-pin-') as directory:
            path=Path(directory);(path/'asr_posterior.py').write_bytes((a.HERE/'asr_posterior.py').read_bytes());(path/'validator-dependencies.json').write_bytes(b'{"private":"SECRET_NEVER_EMIT"}')
            r=subprocess.run([sys.executable,*CLI_FLAGS,str(path/'asr_posterior.py')],capture_output=True,timeout=30)
            self.assertEqual(r.returncode,2);self.assertNotIn(b'SECRET_NEVER_EMIT',r.stdout+r.stderr);self.assertEqual(r.stderr,b'');self.assertEqual(a.decode(r.stdout)['status'],'blocked_or_unconfirmed')
    def test_failure_cli_does_not_leak_private_argument_body(self):
        args=self.bundle();args['expected_path'].write_bytes(b'{"private":"TOP_SECRET_NEVER_EMIT"}')
        args['expected_sha256']=a.sha(args['expected_path'].read_bytes());r=self.run_cli(args);self.assertEqual(r.returncode,2);self.assertNotIn(b'TOP_SECRET',r.stdout+r.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
