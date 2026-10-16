import copy,itertools,json,math,unittest
import verify_result as v
class NumericTests(unittest.TestCase):
    def test_bruteforce(self):
        rows=[[math.log(x) for x in r] for r in [[.3,.2,.1,.25,.15],[.4,.1,.2,.2,.1],[.2,.3,.2,.1,.2],[.2,.1,.2,.2,.3]]]
        for target in ([],[1],[1,1],[1,2],[1,2,3],[3,4,3,4]):
            probability=math.fsum(math.exp(sum(rows[i][t] for i,t in enumerate(path))) for path in itertools.product(range(5),repeat=len(rows)) if v.collapse(list(path))==target)
            expected=-math.log(probability) if probability else math.inf
            self.assertAlmostEqual(v.ctc_nll(rows,target),expected)
    def test_empty(self):
        rows=[[math.log(.2)]*5]*83
        self.assertAlmostEqual(v.ctc_nll(rows,[]),-sum(r[0] for r in rows))
    def test_repeat_needs_blank(self):
        self.assertEqual(v.ctc_nll([[math.log(.2)]*5]*2,[1,1]),math.inf)
    def test_first_index_tie(self):self.assertEqual(v.check_log_probs([[math.log(.2)]*5],1),[0])
    def test_collapse(self):self.assertEqual(v.collapse([1,1,0,1,2,2,0]),[1,1,2])
    def test_tolerance_boundary(self):
        for ctc in (False,True):
            t=v.TOLERANCES['ctc_nll' if ctc else 'derived'];bound=t['absolute']+t['relative']*10
            v.close(10+bound*.999,10,'tolerance',ctc)
            with self.assertRaises(v.Invalid):v.close(10+bound*1.001,10,'tolerance',ctc)
    def test_normalization_boundary(self):
        base=math.log(.2)
        v.check_log_probs([[base+1.99e-6]*5],1)
        with self.assertRaises(v.Invalid):v.check_log_probs([[base+2.01e-6]*5],1)
    def test_bad_values(self):
        for x in (True,False,None,'-1',float('nan'),float('inf'),-float('inf')):
            with self.subTest(x=x),self.assertRaises(v.Invalid):v.check_log_probs([[x]*5],1)
    def test_dimension(self):
        for x in ([],[[0]*4],[[0]*6],[[0]*5]*2):
            with self.assertRaises(v.Invalid):v.check_log_probs(x,1)
    def test_json_reject(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.assertRaises(v.Invalid):v.strict_json(raw)
    def test_bool_not_number(self):
        with self.assertRaises(v.Invalid):v.number(True)
    def test_type_exact(self):self.assertFalse(v.exact({'x':True},{'x':1}))
    def test_unknown_fields(self):
        with self.assertRaises(v.Invalid):v.keys({'expected':1,'unknown':2},{'expected'})

import fixtures
class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixtures.fixture()
    def test_complete_negative(self):
        r=v.verify_objects(*self.base);self.assertEqual(r['evidence_validation'],'PASS');self.assertEqual(r['scientific_validation'],'DESCRIPTIVE_ONLY');self.assertEqual(r['successful_step_receipts'],1200);self.assertEqual(r['model_forward_calls_returned'],1218);self.assertEqual(r['statistics']['probe-B-0600']['macro_complete_rate'],0.)
    def test_complete_positive(self):
        r=v.verify_objects(*fixtures.fixture(True));self.assertEqual(r['scientific_validation'],'DESCRIPTIVE_ONLY');self.assertTrue(r['statistics']['probe-B-0600']['fit_success'])
    def reject(self,change,args=None):
        args=copy.deepcopy(args or self.base);change(args)
        with self.assertRaises(v.Invalid):v.verify_objects(*args)
    def test_mutations(self):
        changes={
        'unknown_file':lambda a:a[0].update({'model.pt':{}}),
        'call_gap':lambda a:a[0].pop('call-0001.json'),
        'step_gap':lambda a:a[0].pop('step-0100.json'),
        'forward_gap':lambda a:a[0].pop('forward-0002.json'),
        'return_gap':lambda a:a[0].pop('forward-return-0002.json'),
        'missing_return_still_opt':lambda a:a[0].pop('forward-return-1216.json'),
        'missing_payload':lambda a:a[0].pop('payload-0002.json'),
        'probe_early':lambda a:a[0]['forward-0002.json'].update(kind='probe'),
        'group_leakage':lambda a:a[0]['payload-0001.json']['samples'][0].update(preset_voice_id='qwen-preset:vivian'),
        'wrong_A_weight':lambda a:a[0]['payload-0001.json']['samples'][0].update(loss_weight=.125),
        'wrong_B_weight':lambda a:a[0]['payload-0609.json']['samples'][0].update(loss_weight=1/12),
        'wrong_probe_weight':lambda a:a[0]['payload-1217.json']['samples'][0].update(loss_weight=.25),
        'terminal_select':lambda a:a[0]['terminal.json'].update(probe_used_for_selection=True),
        'terminal_best_step':lambda a:a[0]['terminal.json'].update(primary_endpoint_step=300),
        'fake_macro':lambda a:a[0]['terminal.json']['probe_endpoints']['B'].update(macro_complete_rate=1.),
        'binding_plan_hash':lambda a:a[0]['binding.json'].update(plan_sha256='f'*64),
        'binding_runtime':lambda a:a[0]['binding.json'].update(runtime_lock_sha256='f'*64),
        'binding_cohort':lambda a:a[0]['binding.json'].update(cohort_sha256='f'*64),
        'init_seed':lambda a:a[0]['B-initialization.json'].update(seed=1),
        'init_hash':lambda a:a[0]['B-initialization.json'].update(initial_weight_sha256='f'*64),
        'bool_grad':lambda a:a[0]['step-0001.json'].update(finite_grad_norm=True),
        'wrong_step_hash':lambda a:a[0]['step-0001.json'].update(payload_sha256='f'*64),
        'unknown_step_key':lambda a:a[0]['step-0001.json'].update(retry=True),
        'nonfinite_lp':lambda a:a[0]['payload-0001.json']['samples'][0]['log_probs'][0].__setitem__(0,float('nan')),
        'unnormalized_lp':lambda a:a[0]['payload-0001.json']['samples'][0]['log_probs'][0].__setitem__(0,-1.),
        'wrong_alternate':lambda a:a[0]['payload-1217.json']['samples'][0]['ctc_nll_by_target'].update(KW2=1.),
        'wrong_blank':lambda a:a[0]['payload-1217.json']['samples'][0].update(anchor_nll=1.),
        'wrong_rates':lambda a:a[0]['payload-1217.json']['per_class_complete_rate'].update(KW1=1.),
        'wrong_prefix':lambda a:a[0]['payload-0001.json']['samples'][0].update(prefix_frames=2),
        'zero_prefix_not_null':lambda a:a[0]['payload-0001.json']['samples'][0].update(prefix_blank_probability_min=.96),
        'wrong_pcm':lambda a:a[0]['payload-0001.json']['samples'][0].update(pcm_sha256='f'*64),
        'wrong_claim':lambda a:a.__setitem__(5,a[5].replace(b'1200',b'1201')),
        'extra_budget':lambda a:a[0].update({'call-1201.json':{}}),
        }
        for label,change in changes.items():
            with self.subTest(label=label):self.reject(change)
    def test_failures_and_incomplete(self):
        cases=[(False,False,False,False,'forward'),(True,False,False,False,'forward'),(True,True,False,False,'backward'),(True,True,True,False,'optimizer_step')]
        for returned,payload,call,step,phase in cases:
            for failure in (False,True):
                with self.subTest(phase=phase,returned=returned,failure=failure):
                    a=fixtures.partial(self.base,2,returned,payload,call,step,failure,phase);r=v.verify_objects(*a)
                    self.assertEqual(r['evidence_validation'],'PASS');self.assertEqual(r['scientific_validation'],'NOT_EVALUATED');self.assertEqual(r['successful_step_receipts'],0)
    def test_partial_optimizer_without_return_rejected(self):
        a=fixtures.partial(self.base,2,False,False,True,False,True,'optimizer_step')
        with self.assertRaises(v.Invalid):v.verify_objects(*a)
    def test_continue_after_missing_observation(self):self.reject(lambda a:a[0].pop('A-0050.json'))
    def test_continue_after_failure(self):
        a=fixtures.partial(self.base,2,True,True,True,False,True,'optimizer_step');a[0].update({k:self.base[0][k] for k in ('forward-0003.json','forward-return-0003.json','payload-0003.json')})
        with self.assertRaises(v.Invalid):v.verify_objects(*a)
    def test_timeout_not_science(self):
        a=list(self.base);a[7]=dict(exit_code=-9,wall_timeout=True);r=v.verify_objects(*a);self.assertEqual(r['execution_status'],'incomplete');self.assertEqual(r['scientific_validation'],'NOT_EVALUATED')
    def test_closed_plan(self):
        plan=copy.deepcopy(self.base[1]);v.check_plan(plan)
        for key,value in [('train_weights',{'A':{'KW1':.125,'KW2':.125},'B':{}}),('max_optimizer_calls',1201),('probe_selection',True),('execution_enabled',False),('seed',True),('cohort_sha256','f'*64),('unknown',1)]:
            p=copy.deepcopy(plan);p[key]=value
            with self.subTest(key=key),self.assertRaises(v.Invalid):v.check_plan(p)

from pathlib import Path
import tempfile,os
class SourceTests(unittest.TestCase):
    def test_actual_cohort(self):
        root=Path(__file__).absolute().parent.parent;module=root/'grouped-balance-overlay/research/kws-grouped-balance-v1'
        cohort=v.strict_json(v.safe_read(module/'cohort-prefix-freeze.json'));registry=v.strict_json(v.safe_read(root/'grouped-balance-integrated/research/kws-candidate-registry-v1/candidates.json'));plan=v.strict_json(v.safe_read(module/'plan.json'))
        train,probe=v.check_cohort(cohort,plan,registry);self.assertEqual((len(train),len(probe)),(12,4))
        for key,value in [('preset_voice_id','qwen-preset:vivian'),('pcm_sha256','f'*64),('target_class','KW2'),('input_frames',2),('prefix_frames',2),('role','probe')]:
            c=copy.deepcopy(cohort);c['samples'][0][key]=value
            with self.subTest(key=key),self.assertRaises(v.Invalid):v.check_cohort(c,plan,registry)
    def test_alias_and_changed_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_bytes(b'abc');v.pinned(p,v.sha(b'abc'))
            with self.assertRaises(v.Invalid):v.pinned(p,'f'*64)
            q=Path(d)/'link';q.symlink_to(p)
            with self.assertRaises((OSError,v.Invalid)):v.safe_read(q)
            q.unlink();os.link(p,q)
            with self.assertRaises(v.Invalid):v.safe_read(p)
    def test_no_report_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reports';v.write_reports(p,Path(d)/'result',{},{} )
            with self.assertRaises(FileExistsError):v.write_reports(p,Path(d)/'result',{},{} )


import outer_fixtures
class OuterTests(unittest.TestCase):
    def test_outer_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);repo=base/'grouped-balance-integrated';repo.mkdir();launcher=base/'launch_reviewed_grouped_balance.py';launcher.write_bytes(b'# synthetic, not executable\n')
            freeze=dict(plan_sha256='a'*64,runner_sha256='b'*64,launcher_sha256=v.sha(launcher.read_bytes()),reviewed_source_commit='c'*40,reviewed_source_pins={'source':'d'*64})
            path=outer_fixtures.outer(base,repo,freeze);v.verify_outer(path,freeze['reviewed_source_commit'],repo,freeze)
            intent_path=path.with_name('launch-intent.json');original=intent_path.read_bytes();obj=v.strict_json(original)
            for key,value in [('plan_sha256','e'*64),('runner_sha256','e'*64),('max_optimizer_updates',1201),('max_model_forward_calls',1219),('retry',True),('command_sha256','f'*64),('previous_campaign_budget_reused',True),('cpu_limit_seconds',1800),('runner_evidence_directory_limit_bytes',2**30),('reviewed_source_pins',{})]:
                altered=copy.deepcopy(obj);altered[key]=value;intent_path.write_bytes(v.encoded(altered))
                with self.subTest(key=key),self.assertRaises(v.Invalid):v.verify_outer(path,freeze['reviewed_source_commit'],repo,freeze)
            intent_path.write_bytes(original);path.with_name('stdout.log').write_bytes(b'changed')
            with self.assertRaises(v.Invalid):v.verify_outer(path,freeze['reviewed_source_commit'],repo,freeze)


class SourceClosureTests(unittest.TestCase):
    def test_source_closure_and_plan_pin(self):
        import shutil
        root=Path(__file__).absolute().parent.parent;source=root/'grouped-balance-integrated';overlay=root/'grouped-balance-overlay';prefix='research/kws-grouped-balance-v1'
        plan=v.strict_json(v.safe_read(overlay/prefix/'plan.json'));plan['execution_enabled']=True
        registry_prefix='research/kws-candidate-registry-v1';inputs=v.strict_json(v.safe_read(source/registry_prefix/'input-pins.json'))
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)/'grouped-balance-integrated';repo.mkdir()
            names=set(inputs)|set(plan['code_sha256'])|{registry_prefix+'/'+n for n in ('candidates.json','build_registry.py','input-pins.json')}
            for name in names:
                target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            module=repo/prefix;module.mkdir(parents=True)
            for name in set(plan['diagnostic_code_sha256'])|{'runtime-candidate.lock.json','cohort-prefix-freeze.json'}:shutil.copyfile(overlay/prefix/name,module/name)
            (module/'plan.json').write_bytes(v.encoded(plan));freeze={'plan_sha256':v.sha(v.safe_read(module/'plan.json'))}
            loaded,train,probe,runtime=v.load_sources(repo,freeze);self.assertEqual((len(train),len(probe)),(12,4))
            (module/'run_grouped_balance.py').write_bytes(b'changed source')
            with self.assertRaises(v.Invalid):v.load_sources(repo,freeze)
            (module/'plan.json').write_bytes(b'{}')
            with self.assertRaises(v.Invalid):v.load_sources(repo,freeze)


class FailureBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixtures.fixture()
    def boundary(self,count):
        args=fixtures.partial(self.base,count,True,True,False,False,True,'forward')
        a,k,u=v.forward_events()[count-1];receipt=f'{a}-{u:04d}.json'
        args[0][receipt]=self.base[0][receipt]
        next_a,next_kind,next_u=v.forward_events()[count]
        if next_a=='B':args[0]['B-initialization.json']=self.base[0]['B-initialization.json']
        f=args[0]['failure.json'];f.update(stage=next_a,update=next_u)
        f['context']=dict(phase='forward',kind=next_kind,record_ids=[r['record_id'] for r in (args[3] if next_kind=='probe' else args[2])])
        return args,receipt
    def test_completed_A600_to_B0_reservation_failure(self):
        args,_=self.boundary(608);r=v.verify_objects(*args)
        self.assertEqual((r['successful_step_receipts'],r['model_forward_calls_reserved']),(600,608));self.assertEqual(r['execution_status'],'failed_partial');self.assertEqual(r['scientific_validation'],'NOT_EVALUATED')
    def test_completed_B600_to_probe_A600_reservation_failure(self):
        args,_=self.boundary(1216);r=v.verify_objects(*args)
        self.assertEqual((r['successful_step_receipts'],r['model_forward_calls_reserved']),(1200,1216));self.assertEqual(r['execution_status'],'failed_partial');self.assertEqual(r['scientific_validation'],'NOT_EVALUATED')
    def test_unfinished_prior_event_cannot_cross_stage(self):
        for count in (608,1216):
            args,receipt=self.boundary(count);args[0].pop(receipt)
            if count==608:args[0].pop('B-initialization.json')
            with self.subTest(count=count),self.assertRaises(v.Invalid):v.verify_objects(*args)
    def test_cannot_skip_immediate_next_event(self):
        for count in (608,1216):
            args,_=self.boundary(count);f=args[0]['failure.json']
            if count==608:f.update(stage='B',update=1);f['context']['kind']='training'
            else:f.update(stage='B',update=600);f['context']['kind']='probe'
            with self.subTest(count=count),self.assertRaises(v.Invalid):v.verify_objects(*args)

if __name__=='__main__':unittest.main()
