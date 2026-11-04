"""Invented objects only. No model packages, installation or network."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import capture_worker as c

class Config:
    def __init__(self):
        self.__dict__.update(copy.deepcopy(c.EXPECTED))
    def to_dict(self):
        return dict(self.__dict__)

class Thinker:
    def __init__(self):
        self.config = Config()
        self.resolutions = 1
        self.generations = 0
        self.mutate = False
        self.fail = False
    def _prepare_generation_config(self, *args, **kwargs):
        return self.config, {}
    def generate(self, inputs=None, generation_config=None, logits_processor=None,
                 stopping_criteria=None, prefix_allowed_tokens_fn=None,
                 assistant_model=None, custom_generate=None, negative_prompt_ids=None,
                 negative_prompt_attention_mask=None, **kwargs):
        self.generations += 1
        for _ in range(self.resolutions):
            self._prepare_generation_config()
        if self.mutate:
            self.config.num_beams = 2
        if self.fail:
            raise RuntimeError('invented decoder failure')
        return 'invented result'

class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.thinker = Thinker()
        self.runtime = NS(wrapper=NS(model=NS(thinker=self.thinker)))
        self.capture = c.BoundCapture(self.runtime, Path(self.temp.name)/'captures', 2)
        self.bound = NS(opaque_id='clip-000001', descriptor_sha256='d'*64,
                        descriptor={'wav': {'sha256':'a'*64},'raw_pcm16':{'sha256':'b'*64}})
    def infer(self, *args):
        self.thinker.generate()
        return {'raw_text':'invented'}, {'original':'unchanged'}
    def call(self):
        return self.capture.infer(self.infer, self.runtime, self.bound, 'c'*64)
    def test_exact_binding_and_original_evidence(self):
        row, evidence = self.call()
        self.assertEqual(row, {'raw_text':'invented'})
        self.assertEqual(evidence['original'],'unchanged')
        b=evidence['prospective_generation_capture']
        self.assertEqual(b['pcm_sha256'],'b'*64)
        self.assertEqual(b['input_binding_sha256'],'d'*64)
        self.assertEqual(b['decoder_manifest_sha256'],'c'*64)
        self.assertFalse(b['training_admitted'])
        self.assertEqual(b['resolved_generation_config_sha256'], c.sha((self.capture.recorder.output/b['snapshot_file']).read_bytes()))
    def test_zero_capture_fails(self):
        self.thinker.resolutions=0
        with self.assertRaises(ValueError):self.call()
    def test_two_captures_fail(self):
        self.thinker.resolutions=2
        with self.assertRaises(ValueError):self.call()
    def test_duplicate_clip_not_replayed(self):
        self.call()
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.thinker.generations,1)
    def test_failure_poisons_following_clip(self):
        self.thinker.fail=True
        with self.assertRaises(RuntimeError):self.call()
        self.bound.opaque_id='clip-000002'
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.thinker.generations,1)
    def test_post_resolution_mutation_rejected(self):
        self.thinker.mutate=True
        with self.assertRaises(ValueError):self.call()
    def test_outside_claim_generate_rejected(self):
        with self.assertRaises(ValueError):self.thinker.generate()
        self.assertEqual(self.thinker.generations,0)
    def test_custom_controls_rejected_before_native_call(self):
        for key in ('logits_processor','stopping_criteria','prefix_allowed_tokens_fn',
                    'assistant_model','custom_generate','negative_prompt_ids','negative_prompt_attention_mask'):
            self.capture.active='invented';self.capture.generate_calls=0
            with self.assertRaises(ValueError):self.thinker.generate(**{key:object()})
        self.assertEqual(self.thinker.generations,0)
    def test_positional_custom_control_rejected(self):
        self.capture.active='invented'
        with self.assertRaises(ValueError):self.thinker.generate(None,None,object())
    def test_multiple_generate_calls_rejected(self):
        def double(*args):
            self.thinker.generate();self.thinker.generate()
        with self.assertRaises(ValueError):self.capture.infer(double,self.runtime,self.bound,'c'*64)
        self.assertEqual(self.thinker.generations,1)
    def test_runtime_substitution_rejected(self):
        with self.assertRaises(ValueError):self.capture.infer(self.infer,NS(),self.bound,'c'*64)
    def test_missing_required_config_rejected(self):
        del self.thinker.config.num_beams
        with self.assertRaises(ValueError):self.call()
        self.assertTrue(self.capture.dead)
    def test_snapshot_write_failure_cannot_return_success(self):
        with patch.object(c.Path,'read_bytes',side_effect=OSError('invented read failure')):
            with self.assertRaises(OSError):self.call()
        self.assertTrue(self.capture.dead)

    def test_failed_inference_retains_audio_claim(self):
        self.thinker.fail=True
        with self.assertRaises(RuntimeError):self.call()
        claim=json.loads((self.capture.recorder.output/'claim-0001.json').read_text())
        self.assertEqual(claim['opaque_id'], self.bound.opaque_id)
        self.assertEqual(claim['pcm_sha256'], 'b'*64)

    def test_wrapper_restores_legacy_entrypoints(self):
        events=[]
        def load(*args):return self.runtime
        def imports():return NS(infer_qwen_once_after_approval=self.infer),object()
        legacy=NS(load_qwen_after_approval=load,import_helpers=imports)
        def run(*args):
            self.assertIsNot(legacy.load_qwen_after_approval,load)
            self.assertIsNot(legacy.import_helpers,imports)
            events.append('ran')
            raise RuntimeError('invented stage failure')
        legacy.run=run
        with self.assertRaises(RuntimeError):c.run(legacy,'sensevoice','i','o','r')
        self.assertIs(legacy.load_qwen_after_approval,load)
        self.assertIs(legacy.import_helpers,imports)
        self.assertEqual(events,['ran'])

    def test_native_source_mismatch_is_not_relaxed(self):
        with patch.object(c.inspect,'getsourcefile',return_value=__file__):
            with self.assertRaises(ValueError):c.verified_native_source(self.runtime)

    def test_restore_actual_native_resolver_not_remember_wrapper(self):
        original=self.capture.original_resolver
        remembered=self.capture.recorder.original
        self.assertIsNot(original,remembered)
        self.capture.restore()
        self.assertIs(self.thinker._prepare_generation_config,original)
        self.assertIs(self.thinker.generate,self.capture.original_generate)
        self.assertFalse(self.capture.recorder.installed)

    def test_restore_actual_native_resolver_after_failure(self):
        self.thinker.fail=True
        with self.assertRaises(RuntimeError):self.call()
        self.capture.restore()
        self.assertIs(self.thinker._prepare_generation_config,self.capture.original_resolver)

if __name__=='__main__':unittest.main()
