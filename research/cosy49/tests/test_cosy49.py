"""Stdlib fixtures, fake tensors and AST only; no model/CTC/frontend execution."""
import ast,copy,hashlib,json,math,sys,types,unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import candidate_control as c
import train_once as t
import data_contract as d
import launch_once as launch
from common import read_json

class Tensor:
    def __init__(self,value=0.,shape=(),finite=True):
        self.value=value;self.shape=shape;self.ndim=len(shape);self.finite=finite;self.device=types.SimpleNamespace(type='cpu');self.dtype='float32'
    def item(self):return self.value
    def all(self):return self

def fake_torch():return types.SimpleNamespace(Tensor=Tensor,float32='float32',isfinite=lambda t:Tensor(t.finite))
def fixture_optimizer():
    params=[Tensor(shape=(2,))for _ in range(28)]
    model=types.SimpleNamespace(parameters=lambda:iter(params))
    optimizer=types.SimpleNamespace(state={p:{'step':Tensor(4),'exp_avg':Tensor(shape=p.shape),'exp_avg_sq':Tensor(shape=p.shape)}for p in params})
    return model,optimizer,params

class Cosy49Tests(unittest.TestCase):
    def setUp(self):self.rows=read_json(ROOT/'metadata/TRAIN49.json')['rows']
    def test_source_adapter_exact_author_hash(self):self.assertEqual(hashlib.sha256((ROOT/'src/candidate_control.py').read_bytes()).hexdigest(),'33ac29d8f668fbede4409eb5bdb0990f8c5ed3714664e38aff426b9ec6606336')
    def test_no_scientific_import(self):self.assertFalse({'torch','numpy'}&set(sys.modules))
    def test_exact49_actual_targets(self):
        c.validate_rows(self.rows);cosy={r['alias']:r for r in self.rows[32:]};self.assertEqual(cosy['D']['target_ids'],[3,4]);self.assertEqual(cosy['I']['target_ids'],[3,5]);self.assertNotIn('Q',cosy)
    def test_three_cohort_weights(self):
        w=c.cohort_weights([r['cohort']for r in self.rows]);self.assertEqual(sum(w),1);self.assertEqual([sum(w[:20]),sum(w[20:32]),sum(w[32:])],[Fraction(1,3)]*3)
        self.assertAlmostEqual(c.objective([len(r['target_ids'])for r in self.rows],self.rows),1)
    def test_d_i_length_normalization(self):
        losses=[0.]*49;losses[35]=2.;self.assertAlmostEqual(c.objective(losses,self.rows),1/51)
    def test_math_nonnegative_only_oracle(self):
        with self.assertRaises(ValueError):c.objective([-1.]*49,self.rows)
        cs=[r['cohort']for r in self.rows];result=c.terminal_fit([1.]*49,[-1e-8]*49,cs);self.assertEqual(result['status'],'TRAIN_FIT_IMPROVED');self.assertFalse(result['stop_training'])
    def test_dev_row_and_omission_rejected(self):
        for bad in (self.rows[:-1],copy.deepcopy(self.rows)):
            if len(bad)==49:bad[0]['role']='development'
            with self.assertRaises(ValueError):c.validate_rows(bad)
    def test_live_data_pin_blocked_or_exact(self):
        pin=read_json(ROOT/'metadata/COSY17-INPUTS.json')
        self.assertEqual(pin['feature_manifest']['sha256'],'da6406fd5f2b6c10bf09369a876e92e060acaeb88ce807ea8305d87be89038ed')
        if pin['status']=='VERIFIED_PUBLIC_PIN':
            d.frozen_inputs(ROOT);self.assertEqual(len(pin['requests']),1);self.assertTrue(d.hex_digest(pin['commit'],40))
        else:
            with self.assertRaisesRegex(ValueError,'audited public pin'):d.frozen_inputs(ROOT)
    def pinned_fixture(self):
        pin=copy.deepcopy(read_json(ROOT/'metadata/COSY17-INPUTS.json'));pin.update(status='VERIFIED_PUBLIC_PIN',commit='a'*40,audit_receipt_sha256='b'*64,public_projection_review_sha256='e'*64)
        pin['requests']=[dict(repository=pin['repository'],commit=pin['commit'],path='fixtures/cosy17.zip',raw_url='https://raw.githubusercontent.com/'+pin['repository']+'/'+pin['commit']+'/fixtures/cosy17.zip',bytes=pin['archive']['bytes'],sha256=pin['archive']['sha256'],git_blob_sha1='c'*40)]
        return pin
    def test_mock_public_pin_closed_contract(self):
        pin=self.pinned_fixture();self.assertIs(d.validate_new_pin(pin,self.rows),pin)
    def test_mock_public_pin_changes_rejected(self):
        for kind in ('status','commit','audit','projection','count','url','duplicate','destination','hash','bytes','sha1'):
            p=self.pinned_fixture()
            if kind=='status':p['status']='PENDING'
            elif kind=='commit':p['commit']=None
            elif kind=='audit':p['audit_receipt_sha256']=None
            elif kind=='projection':p['public_projection_review_sha256']=None
            elif kind=='count':p['requests'].pop()
            elif kind=='url':p['requests'][0]['raw_url']+='?mutable=1'
            elif kind=='duplicate':p['archive']['members'][1]['destination']=p['archive']['members'][0]['destination']
            elif kind=='destination':p['archive']['members'][1]['destination']='../audio.wav'
            elif kind=='hash':p['requests'][0]['sha256']='d'*64
            elif kind=='bytes':p['requests'][0]['bytes']+=4
            elif kind=='sha1':p['requests'][0]['git_blob_sha1']=None
            with self.assertRaises(ValueError,msg=kind):d.validate_new_pin(p,self.rows)
    def test_feature_order_cmvn_and_hash_drift_rejected(self):
        good=[dict(r['feature'],recording=r['recording'],pcm_sha256=r['pcm_sha256'],wav_sha256=r['wav_sha256'])for r in self.rows]
        d.validate_feature_entries(self.rows,good)
        for key,value in [('normalization','POST_CMVN'),('sha256','f'*64),('shape',[1,400]),('recording','other')]:
            bad=copy.deepcopy(good);bad[32][key]=value
            with self.assertRaises(ValueError):d.validate_feature_entries(self.rows,bad)
    def public_manifest_fixture(self):
        rows=[dict(recording=r['recording'],cohort=r['cohort'],target_ids=r['target_ids'],frames=r['frames'],wav_sha256=r['wav_sha256'],pcm_sha256=r['pcm_sha256'],complete_wake_target=r['complete_wake_target'],split='train',actual_human_lexical_transcript=r['text'],feature={k:v for k,v in r['feature'].items()if k!='file'}|{'path':'features/'+r['recording']+'.f32le'})for r in self.rows[32:]]
        return dict(schema='cosy17-public-precmvn-features-v1',source_count=17,selected_rows=693,raw_feature_bytes=1108800,dtype='<f4',normalization='PRE_CMVN',feature_width=400,tokens=list(c.TOKENS),development_included=False,inaudible_rows_included=False,original_audio_included=False,source_order=[r['recording']for r in self.rows[32:]],rows=rows)
    def test_public_schema_adapter_preserves_all17_identities(self):
        mapped=d.public_feature_entries(self.public_manifest_fixture(),self.rows)
        self.assertEqual(len(mapped),17);self.assertEqual([f['sha256']for f in mapped],[r['feature']['sha256']for r in self.rows[32:]])
    def test_public_schema_adapter_rejects_private_or_changed_manifest(self):
        for key in ('schema','order','target','normalization','path','hash'):
            p=self.public_manifest_fixture()
            if key=='schema':p['schema']='cosy17-native-feature-evidence-v1'
            elif key=='order':p['source_order'].reverse()
            elif key=='target':p['rows'][3]['target_ids']=[1,2,3,4]
            elif key=='normalization':p['rows'][0]['feature']['normalization']='POST_CMVN'
            elif key=='path':p['rows'][0]['feature']['path']='../outside.f32le'
            elif key=='hash':p['rows'][0]['feature']['sha256']='f'*64
            with self.assertRaises(ValueError,msg=key):d.public_feature_entries(p,self.rows)
    def test_disarmed_release_rejects_before_environment(self):
        release=read_json(ROOT/'EXECUTION-RELEASE.template.json');self.assertFalse(release['approved']);self.assertFalse(release['training_authorized']);self.assertIsNone(release['approved_parent_head'])
        with patch.object(launch,'verify_freeze',side_effect=AssertionError('must not reach source execution')):
            with self.assertRaisesRegex(ValueError,'explicit reviewed'):launch.admit(release)
    def test_optimizer_each_update_finite_accept(self):
        model,opt,params=fixture_optimizer()
        with patch.dict(sys.modules,{'torch':fake_torch()}):self.assertTrue(t.optimizer_state_finite(model,opt,4))
    def test_optimizer_each_key_nan_rejected(self):
        for key in ('step','exp_avg','exp_avg_sq'):
            model,opt,params=fixture_optimizer();opt.state[params[0]][key].finite=False
            with patch.dict(sys.modules,{'torch':fake_torch()}):
                with self.assertRaisesRegex(ValueError,'finite optimizer'):t.optimizer_state_finite(model,opt,4)
    def test_optimizer_wrong_step_shape_missing_rejected(self):
        for case in ('step','shape','missing'):
            model,opt,params=fixture_optimizer();state=opt.state[params[0]]
            if case=='step':state['step'].value=3
            elif case=='shape':state['exp_avg'].shape=(3,)
            else:del state['exp_avg_sq']
            with patch.dict(sys.modules,{'torch':fake_torch()}):
                with self.assertRaises(ValueError):t.optimizer_state_finite(model,opt,4)
    def test_scan_inside_loop_after_update_before_record(self):
        s=(ROOT/'src/train_once.py').read_text();run=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='run');loop=next(n for n in ast.walk(run)if isinstance(n,ast.For)and ast.unparse(n.iter)=='range(1, 301)')
        text=ast.unparse(loop);self.assertLess(text.index('optimizer.step()'),text.index('optimizer_state_finite('));self.assertLess(text.index('optimizer_state_finite('),text.index('history.append('));self.assertNotIn('break',text)
    def test_initial_control_uses_existing_outputs_before_backward(self):
        s=(ROOT/'src/train_once.py').read_text();a=s.index('if step==1:');b=s.index("journal.emit('backward'",a);part=s[a:b]
        self.assertIn('logits.detach()',part);self.assertIn('for v in per',part);self.assertNotIn('tensor_ctc_loss(',part);self.assertNotIn('model(',part);self.assertNotIn('forward(',part)
        self.assertNotIn('initialization_control(',s);self.assertNotIn('a20-initial-logits',s);self.assertIn('optimizer_checks==300',s)
    def test_only_two_ctc_sites_and_two_forward_call_sites(self):
        tree=ast.parse((ROOT/'src/train_once.py').read_text());calls=[n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('tensor_ctc_loss'),2);self.assertEqual(calls.count('forward'),2);self.assertNotIn('cosy_endpoint_gate',calls)
    def test_geometry_and_resource_caps(self):
        p=read_json(ROOT/'PROTOCOL.json');g=p['resource']['geometry'];self.assertEqual(g['padded_feature_bytes'],49*95*400*4);self.assertEqual(g['unpadded_rows'],2505);self.assertEqual(p['output_contract']['terminal_logits_bytes'],49*95*6*4)
        self.assertEqual(p['resource']['guard']['aggregate_CPU_seconds'],300);self.assertEqual(p['resource']['guard']['aggregate_wall_seconds'],600);self.assertIn('finite scans',p['resource']['timing_estimate']['basis'])
    def test_execution_workflow_exact_guard(self):
        s=(ROOT.parents[1]/'.github/workflows/a20-cosy49.yml').read_text();self.assertIn("if: github.repository == 'jiying2007/kws-pipeline' && github.event_name == 'push' && github.run_attempt == 1",s);self.assertIn('timeout-minutes: 15',s)
    def test_public_projection_no_private_locators(self):
        for path in ROOT.rglob('*'):
            if not path.is_file():continue
            text=path.read_text()
            for forbidden in ('/workspace/','/root/.codex/','library_file_id','sediment://','"local_path"','source_references'):
                if path.name=='test_cosy49.py':continue
                self.assertNotIn(forbidden,text,str(path.relative_to(ROOT)))
if __name__=='__main__':unittest.main()
