"""Focused offline saved-evidence tests; explicitly prepared fixtures required."""
from pathlib import Path
import copy, importlib.util, json, os, socket, tempfile, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
V=load('leading_public_verifier',ROOT/'verify_saved.py')
P=load('leading_public_prepare',ROOT/'prepare_inputs.py')
FIXTURE_ROOT=Path(os.environ.get('MELO5_LEADING_FIXTURES',ROOT/'build/fixtures'))
DATA=FIXTURE_ROOT/'data'
DEPS=FIXTURE_ROOT/'dependencies'

class SavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA.is_dir() or not DEPS.is_dir():raise RuntimeError('Explicit preparation required: see README; tests never download inputs')
    def test_full_saved_reproduction_offline(self):
        with patch.object(socket,'socket',side_effect=AssertionError('network prohibited')):
            result=V.verify(DATA,DEPS)
        self.assertEqual(result['status'],'PASS_SAVED_LEADING_CONTEXT')
        self.assertEqual(result['new_model_frontend_native_decoder_calls'],0)
    def test_missing_evidence_fails(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'closed data file set'):V.verify(Path(t),DEPS)
    def test_raw_size_or_hash_drift_fails(self):
        expected={'bytes':3,'sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'size/hash mismatch'):P.verify(b'abc',dict(name='saved',**expected))
    def test_source_delta_overlap_fails(self):
        delta=dict(original=V.descriptor(b'abcd'),result=V.descriptor(b'abcd'),edits=[dict(start=1,end=3,old='bc',new='bc'),dict(start=2,end=4,old='cd',new='cd')])
        with self.assertRaisesRegex(ValueError,'overlapping'):V.apply_delta(b'abcd',delta)
    def test_all_original_pcm_and_two_zero_regions_bound(self):
        with V.staged(DATA,DEPS) as (_,_,_,work,manifest):
            module=V.load('test_byte_binding',work/'src/derived_binding.py')
            row=manifest['rows'][0];p=work/row['local_wav_path'];original=p.read_bytes()
            for position in (44,44+24000*2,len(original)-1):
                damaged=bytearray(original);damaged[position]^=1;p.write_bytes(damaged)
                with self.assertRaises(ValueError):module.verify_derived_inputs(manifest,work)
            p.write_bytes(original)
            self.assertEqual(module.verify_derived_inputs(manifest,work)['status'],'VERIFIED_FIXED_LEADING_AND_POSTROLL_BYTES')
    def test_removed_callback_and_nonfinite_logit_fail(self):
        with V.staged(DATA,DEPS) as (binding,saved,_,work,_):
            scorer=V.load('test_exact_endpoint',work/'src/score_endpoint.py');raw=[json.loads(x) for x in saved['original_A20.raw.jsonl'].splitlines()]
            index=next(i for i,r in enumerate(raw) if r['kind']=='callback')
            removed=copy.deepcopy(raw);removed.pop(index)
            nonfinite=copy.deepcopy(raw);nonfinite[index]['logits'][0][0]=float('nan')
            for rows in (removed,nonfinite):
                p=work/'damaged.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):scorer.score(p,work/'metadata/manifest.json',work/'metadata/geometry.json',binding['acquisition_bindings'])
    def test_scorer_source_drift_fails_before_execution(self):
        pin=next(r for r in V.read(ROOT/'metadata/bindings.json')['dependencies'] if r['name']=='scorer/score_events.py')
        raw=(DEPS/pin['name']).read_bytes()
        with self.assertRaisesRegex(ValueError,'size/hash mismatch'):P.verify(raw+b'\n',pin)
    def test_failed_preparation_does_not_create_empty_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'new'
            with patch.object(P,'fetch',side_effect=ValueError('blocked test GET')):
                with self.assertRaises(ValueError):P.prepare(output)
            self.assertFalse(output.exists())
    def test_existing_prepared_bytes_need_no_network(self):
        with patch.object(P,'fetch',side_effect=AssertionError('network prohibited')):
            result=P.prepare(FIXTURE_ROOT)
        self.assertGreater(result['files'],0)
    def test_only_libm_diagnostic_may_differ(self):
        recorded=V.read(DATA/'M2-token-support.json');computed=copy.deepcopy(recorded)
        computed['numeric_check']['max_binary64_absolute_difference']=0.0
        result=V.compare_token_support(computed,recorded)
        self.assertEqual(result['local_value'],0.0)
        self.assertEqual(result['recorded_value'],recorded['numeric_check']['max_binary64_absolute_difference'])
        for field,value in [('posterior',0.0),('eligible',True)]:
            changed=copy.deepcopy(computed);changed['center72_small_token_comparison'][0][field]=value
            with self.assertRaisesRegex(ValueError,'M2 token support drift'):V.compare_token_support(changed,recorded)
        computed['numeric_check']['binary32_reference_disagreements']=1
        with self.assertRaisesRegex(ValueError,'FP32 reference disagreement'):V.compare_token_support(computed,recorded)

if __name__=='__main__':unittest.main()
