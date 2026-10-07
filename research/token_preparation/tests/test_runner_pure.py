"""Invented metadata/HTTP/process/FP32 fixtures only; no DSP, Torch or install."""
import ast
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/deps'))
import contracts as c
from geometry import geometry
import common
import fetch_public_inputs as inputs
import launch_once as launch
import public_artifact as gate
from test_public_artifact import source_fixture,payload_fixture


class Proc:
    pid=4321
    def __init__(self,values=(None,)):self.values=list(values)
    def poll(self):
        return self.values.pop(0)if len(self.values)>1 else self.values[0]


def procstate(children='',critical=None,observation='AVAILABLE_AT_SAMPLE'):
    return dict(critical_errors=critical or {},children_observation=observation,children=children,
        VmRSS=123,VmSize=456,Threads=1,io={k:dict(status='AVAILABLE',value=10)for k in ('rchar','wchar','read_bytes','write_bytes')})


class PureRunnerTests(unittest.TestCase):
    def test_safe_failure_receipts_do_not_echo_private_text(self):
        from safe_failure import describe
        obj=describe(ROOT,ValueError('native frontend failure'),'features_control','NATIVE_FEATURES')
        self.assertEqual(obj['error_code'],'NATIVE_FRONTEND_FAILURE')
        hostile='secret=https://private.example/a password=not-for-public /home/private/account'
        obj=describe(ROOT,RuntimeError(hostile),'features_control','CONTROL_FORWARD_ONE')
        self.assertEqual(obj['error_code'],'UNCLASSIFIED_RUNTIME_FAILURE')
        self.assertNotIn(hostile,json.dumps(obj))
        gate.privacy_check(obj)

    def test_input_overflow_has_only_one_sentinel(self):
        class Response:
            status=200
            headers={}
            def __init__(self,raw,url):self.s=io.BytesIO(raw);self.url=url
            def read(self,n):return self.s.read(n)
            def geturl(self):return self.url
        url='https://raw.githubusercontent.com/x/y/'+'a'*40+'/file'
        raw=b'abc';e=dict(raw_url=url,bytes=3,sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob 3\0'+raw).hexdigest())
        response=Response(raw+b'MORE_BYTES_NOT_READ',url)
        with self.assertRaises(ValueError):inputs.receive(response,e)
        self.assertEqual(response.s.tell(),4)


    def test_plan_geometry_and_scope(self):
        rows=common.read_json(ROOT/'metadata/TRAIN32.json')['rows']
        self.assertEqual(len(rows),32)
        from run_features_control import validate_rows
        validate_rows(rows)
        expected=((rows,885760,199,5470,1812),(rows[:20],497920,112,3070,1017))
        for rs,samples,calls,fbank,selected in expected:
            gs=[geometry(r['frames'])for r in rs]
            self.assertEqual(sum(g['frames']for g in gs),samples)
            self.assertEqual(sum(g['callbacks']for g in gs),calls)
            self.assertEqual(sum(g['fbank_rows']for g in gs),fbank)
            self.assertEqual(sum(g['model_rows']for g in gs),selected)
        bad=copy.deepcopy(rows);bad[0]['role']='development_a'
        with self.assertRaises(ValueError):validate_rows(bad)

    def test_ctc_cohorts_repeated_blank_oov(self):
        self.assertEqual(c.minimum_frames([1,1,2]),4)
        self.assertEqual(c.collapse([1,1,0,1,2,2]),[1,1,2])
        with self.assertRaises(ValueError):c.encode('你好啊')
        with self.assertRaises(ValueError):c.validate_target([0,1])
        groups=[c.D20]*20+[c.Q12]*12
        self.assertAlmostEqual(c.cohort_loss([8]*20+[9]*12,[[1,2,3,4]]*20+[[4,3,4]]*12,groups),2.5)
        self.assertEqual(c.ctc_nll([[-math.log(6)]*6]*2,[1,1]),math.inf)
        with self.assertRaises(ValueError):c.cohort_weights([c.D20]*32)

    def test_no_training_route_and_single_control_forward(self):
        tree=ast.parse((ROOT/'src/control.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='initialization_control')
        calls=[n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='model']
        self.assertEqual(len(calls),1)
        for path in (ROOT/'src').glob('*.py'):
            t=ast.parse(path.read_text())
            for n in ast.walk(t):
                if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):
                    self.assertNotIn(n.func.attr,('backward','AdamW','SGD','set_rng_state'))
        self.assertIn('model.train()',ast.get_source_segment((ROOT/'src/control.py').read_text(),fn))
        self.assertNotIn('torch',sys.modules)

    def test_gzip_bounded_exact_single_member(self):
        raw=b'invented fixture';packed=gzip.compress(raw,mtime=0)
        e=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),compressed_bytes=len(packed),compressed_sha256=hashlib.sha256(packed).hexdigest())
        self.assertEqual(inputs.gunzip_exact(packed,e),raw)
        for broken in (dict(e,bytes=len(raw)-1),dict(e,sha256='0'*64)):
            with self.assertRaises(ValueError):inputs.gunzip_exact(packed,broken)
        multiple=packed+packed;e2=dict(e,compressed_bytes=len(multiple),compressed_sha256=hashlib.sha256(multiple).hexdigest())
        with self.assertRaises(ValueError):inputs.gunzip_exact(multiple,e2)

    def test_input_requests_closed(self):
        m=common.read_json(ROOT/'metadata/PUBLIC-INPUT-MAP.json')
        self.assertEqual(len(m['request_allowlist']),155)
        self.assertEqual(sum(e['bytes']for e in m['request_allowlist']),6097691)
        for e in m['request_allowlist']:inputs.validate_request(e)
        bad=copy.deepcopy(m['request_allowlist'][0]);bad['raw_url']=bad['raw_url'].replace('raw.githubusercontent.com','example.com')
        with self.assertRaises(ValueError):inputs.validate_request(bad)
        bad=copy.deepcopy(m['request_allowlist'][0]);bad['commit']='main'
        with self.assertRaises(ValueError):inputs.validate_request(bad)

    def test_read_proc_missing_and_optional_io(self):
        from supervision_reuse import read_proc
        def read(p):
            if p.name=='status':return 'VmRSS:\t123 kB\nThreads:\t1\nVmSize:\t456 kB\nVmHWM:\t130 kB\n'
            if p.name=='children':return ''
            raise PermissionError('invented optional I/O denial')
        x=read_proc(4321,read)
        self.assertFalse(x['critical_errors'])
        self.assertTrue(all(v['value']is None for v in x['io'].values()))
        def missing(p):return ''if p.name!='io'else 'rchar: 1\n'
        self.assertIn('VmRSS',read_proc(4321,missing)['critical_errors'])

    def test_owned_group_single_and_extra_child(self):
        with patch.object(launch.os,'getpgid',return_value=4321),patch.object(launch,'read_proc',return_value=procstate()),patch.object(launch,'cpu_ticks',return_value=.25):
            s=launch.observe_group(Proc(),1)
            self.assertEqual(s['rss'],123*1024);self.assertEqual(s['processes'],1)
        with patch.object(launch.os,'getpgid',return_value=4321),patch.object(launch,'read_proc',return_value=procstate('4322')),patch.object(launch,'cpu_ticks',return_value=.1):
            with self.assertRaisesRegex(ValueError,'PROCESS_LIMIT'):launch.observe_group(Proc(),1)

    def test_missing_child_visibility_fails(self):
        s=procstate(observation='NOT_AVAILABLE');s['children']=None;s['children_error']={'type':'FileNotFoundError'}
        with patch.object(launch.os,'getpgid',return_value=4321),patch.object(launch,'read_proc',return_value=s):
            with self.assertRaisesRegex(ValueError,'CHILD_VISIBILITY_REQUIRED'):launch.observe_group(Proc(),1)

    def test_permission_denial_not_relaxed_by_exit(self):
        s=procstate(critical={'VmRSS':{'type':'PermissionError'}})
        with patch.object(launch.os,'getpgid',return_value=4321),patch.object(launch,'read_proc',return_value=s):
            with self.assertRaisesRegex(ValueError,'MANDATORY_PROC'):launch.observe_group(Proc([None,0]),1)

    def test_confirmed_exit_race_is_unknown_not_zero(self):
        s=procstate(critical={'VmRSS':{'type':'FileNotFoundError'}})
        with patch.object(launch.os,'getpgid',return_value=4321),patch.object(launch,'read_proc',return_value=s):
            self.assertIsNone(launch.observe_group(Proc([None,0]),1))

    def test_github_default_denial_precedes_git(self):
        with patch.object(launch.subprocess,'check_output',side_effect=AssertionError('no git')):
            for env in ({},{'GITHUB_EVENT_NAME':'pull_request'},{'GITHUB_RUN_ATTEMPT':'2'}):
                with self.assertRaises(ValueError):launch.github_admission({},env)

    def test_source_context_denies_unapproved(self):
        with self.assertRaises(ValueError):launch.admit(dict(approved=False))
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);(root/'SOURCE-FREEZE.json').write_text('{}')
            with self.assertRaises(ValueError):common.check_context(root,{'schema':'bad'},'inputs')

    def test_atomic_progress_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'x.json';common.write_json(p,{'n':1})
            with self.assertRaises(FileExistsError):common.write_json(p,{'n':2})
            common.replace_json(p,{'n':2});self.assertEqual(common.read_json(p),{'n':2})
            self.assertFalse(p.with_name('x.json.pending').exists())

    def test_partial_saved_only_repair(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);out=root/'work/artifact';(out/'native').mkdir(parents=True)
            name,rows=gate.GEOMETRY[0];raw=b'\0'*(rows*1600)
            (out/'native'/f'{name}.f32le').write_bytes(raw)
            with patch.object(launch,'ROOT',root):
                rejected=launch.repair_partial_evidence(out)
            self.assertFalse(rejected)
            evidence=common.read_json(out/'native-features.json')
            self.assertEqual(evidence['rows'][0]['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertNotIn('callbacks',evidence['rows'][0])
            self.assertTrue(evidence['no_new_feature_computation'])

    def test_partial_incomplete_write_not_public(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);out=root/'work/artifact';(out/'official').mkdir(parents=True)
            name,_=gate.GEOMETRY[0];(out/'official'/f'{name}.f32le').write_bytes(b'x')
            with patch.object(launch,'ROOT',root):rejected=launch.repair_partial_evidence(out)
            self.assertEqual(len(rejected),1)
            self.assertFalse(list((out/'official').iterdir()))
            self.assertEqual(len(list((root/'work/incomplete-writes').iterdir())),1)

    def test_last_publication_pair_gate(self):
        from verify_publication import verify
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);(root/'work').mkdir();out=root/'publication';out.mkdir()
            raw=b'invented archive bytes';h=hashlib.sha256(raw).hexdigest();name='token-preparation-artifact.tar.gz'
            receipt=h+'  '+name+'\n';(out/name).write_bytes(raw);(out/(name+'.sha256')).write_text(receipt)
            common.write_json(root/'work/PUBLICATION-READY.json',dict(archive=name,sha256_receipt=name+'.sha256',gzip_bytes=len(raw),sha256=h,
                publication_total_bytes=len(raw)+len(receipt),retention_days=1,upload_performed=False))
            self.assertTrue(verify(root))
            (out/name).write_bytes(b'drift')
            with self.assertRaises(ValueError):verify(root)


if __name__=='__main__':unittest.main()
