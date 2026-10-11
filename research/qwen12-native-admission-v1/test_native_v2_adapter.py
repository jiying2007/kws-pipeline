#!/usr/bin/env python3
"""Pure invented interfaces; no model packages, inference, weights, or audio.

The real new observer, native-v2 validators, retained SenseVoice binding/parser,
and method-restoration logic run. The scientific call bodies are represented by
mocks at their existing boundary; this is not a real-runtime compatibility pass.
"""
import copy
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = next((parent for parent in Path(__file__).resolve().parents
    if (parent / "tools/asr_native_contracts.py").is_file()), HERE)
BASE = Path(os.environ.get("KWS_NATIVE_BASE", str(DEFAULT_BASE)))
SOURCE = HERE / "source" if (HERE / "source").is_dir() else BASE
for folder in (BASE / "tools", BASE / "research/source-screen32-v1", BASE / "research/qwen6_asr", BASE / "research/qwen6_asr/core"):
    sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location("prospective_native_v2", SOURCE / "research/source-screen32-v1/native_v2_adapter.py")
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
from asr_stage import adapters, runtime_support
import sense_adapter
import sense_binding
from asr_native_contracts import validate_native_capture
from asr_stage.decoding import qwen_transcript
fixture_spec = importlib.util.spec_from_file_location("invented_sense_fixture", BASE / "research/source-screen32-v1/test_sense_adapter.py")
f = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(f)


class Generation:
    def __init__(self):
        self.values = {key: None for key in a.QWEN_CONTROLS}
        self.values.update(num_beams=1, num_return_sequences=1, max_new_tokens=256,
            do_sample=False, eos_token_id=[151645, 151643], return_dict_in_generate=True,
            min_length=0, repetition_penalty=1.0)
    def to_dict(self):
        return copy.deepcopy(self.values)
    def get_generation_mode(self):
        return SimpleNamespace(value="greedy_search")


class Thinker:
    def __init__(self):
        self.config = Generation()
    def _prepare_generation_config(self, config, **kwargs):
        return self.config, {}
    def _get_logits_processor(self, generation_config, logits_processor):
        return logits_processor
    def _get_stopping_criteria(self, generation_config, stopping_criteria):
        return stopping_criteria


class Sequence:
    def __init__(self, ids):
        self.ids = list(ids)
        self.shape = (1, len(ids))
    def __getitem__(self, index):
        row, selected = index
        if row != 0:
            raise IndexError(row)
        return SimpleNamespace(tolist=lambda: self.ids[selected])


class Model:
    training = False
    def parameters(self):
        return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="torch.float32")])


class FakeFeature:
    def __call__(self, audio, **kwargs):
        return {"fictional": "features"}


class WavFrontend:
    __module__ = "funasr.frontends.wav_frontend"
    def __init__(self, path):
        for key, value in dict(fs=16000, window="hamming", n_mels=80, frame_length=25,
            frame_shift=10, filter_length_min=-1, filter_length_max=-1, lfr_m=7, lfr_n=6,
            dither=1.0, snip_edges=True, upsacle_samples=True).items():
            setattr(self, key, value)
        self.cmvn_file = str(path)
        self.cmvn = "invented loaded CMVN tensor"
    def forward(self, input, input_lengths):
        return "invented frontend features"


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        self.bound = SimpleNamespace(opaque_id="clip-000001", descriptor_sha256="1" * 64,
            descriptor={"raw_pcm16": {"frame_count": 160}, "float32_pcm": {"shape": [160], "sha256": "2" * 64}})
        self.pcm = SimpleNamespace(shape=(160,))
        self.calls = 0
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runtime_support, "verify_pcm_array", side_effect=self.verify_pcm))
        self.model = Model(); self.model.thinker = Thinker()
        self.wrapper = SimpleNamespace(model=self.model, processor=SimpleNamespace(feature_extractor=FakeFeature()))
        self.runtime = SimpleNamespace(model_id=a.QWEN, wrapper=self.wrapper, declaration={"batch_size": 1})
        self.token_ids = [17, 18, 151645]
        self.failure = None
        self.post_failure = None
        self.model.generate = self.mock_generate
        self.skip_custom = False
        self.feature_options = {"sampling_rate": 16000, "padding": True, "truncation": False, "return_attention_mask": True}
        self.custom_processors = []
        self.stack.enter_context(patch.object(adapters, "infer_qwen_once_after_approval", side_effect=self.mock_qwen))

    def verify_pcm(self, pcm, expected):
        if pcm is not self.pcm or expected != self.bound.descriptor["float32_pcm"]:
            raise ValueError("invented PCM mismatch")

    def retain(self, name, value):
        self.assertNotIn(name, self.saved)
        self.saved[name] = copy.deepcopy(value)
        return a.canonical_sha256(value)

    def new(self):
        kwargs = {}
        if self.runtime.model_id == a.SENSE:
            _, frontend = a.sense_effective(self.runtime, self.model_root)
            self.retain("native-frontend-config.json", frontend)
            kwargs["frontend_config_bytes"] = json.dumps(frontend, sort_keys=True, separators=(",", ":")).encode()
        return a.NativeCaptureAdapter(self.runtime, getattr(self, "model_root", Path("unused")),
            b'{"invented":"source-lock"}\n', b'{"invented":"asset-lock"}\n', self.retain, **kwargs)

    def mock_generate(self, **kwargs):
        thinker = self.model.thinker
        config, _ = thinker._prepare_generation_config(None, max_new_tokens=256, do_sample=False,
            eos_token_id=[151645, 151643], return_dict_in_generate=True)
        if not self.skip_custom:
            thinker._get_logits_processor(config, self.custom_processors)
            thinker._get_stopping_criteria(config, [])
        self.calls += 1
        if self.failure:
            raise self.failure
        return SimpleNamespace(sequences=Sequence([5, 6] + self.token_ids))

    def mock_qwen(self, runtime, bound, decoder_sha):
        runtime.wrapper.processor.feature_extractor([self.pcm], **self.feature_options)
        runtime.wrapper.model.generate(input_ids=SimpleNamespace(shape=(1, 2)))
        if self.post_failure:
            raise self.post_failure
        row, _ = qwen_transcript("language Chinese<asr_text>invented transcript", self.token_ids)
        return row, {"token_ids": list(self.token_ids), "input_binding": bound.descriptor,
                     "prompt_context": "", "force_language": None, "transcript_repair_applied": False}

    def test_qwen_success_observed_snapshots_and_unknown_acoustics(self):
        adapter = self.new(); row, raw = adapter.infer(self.bound, "3" * 64, {})
        self.assertTrue(raw["native_v2_validated"])
        validate_native_capture(adapter.config, raw["native_capture"])
        self.assertEqual(raw["native_capture"]["acoustic_verbatim_completeness"], "unknown")
        self.assertEqual(raw["native_capture"]["input"]["audio_frames"], 160)
        self.assertEqual(adapter.config["source_lock_sha256"], hashlib.sha256(b'{"invented":"source-lock"}\n').hexdigest())
        self.assertEqual(adapter.config["effective_config"]["resolved_generation_config_sha256"],
                         a.canonical_sha256(self.saved["native-resolved-generation-config.json"]))
        self.assertEqual(self.calls, 1)
        self.assertNotIn("_prepare_generation_config", vars(self.model.thinker))
        self.assertIsInstance(self.wrapper.processor.feature_extractor, FakeFeature)

    def test_qwen_token_limit_is_retained_but_never_valid(self):
        self.token_ids = [17] * 256
        adapter = self.new(); row, raw = adapter.infer(self.bound, "3" * 64, {})
        self.assertFalse(raw["native_v2_validated"])
        self.assertEqual(row["completeness"], "incomplete")
        self.assertEqual(raw["native_capture"]["termination"]["stop_reason"], "token_limit")
        self.assertTrue(raw["native_capture"]["termination"]["truncated"])
        self.assertEqual(raw["native_capture"]["termination"]["generated_token_ids"], self.token_ids)

    def test_qwen_eos_then_tokens_and_short_return_are_invalid(self):
        for ids in ([151645, 17, 151645], [17, 18]):
            with self.subTest(ids=ids):
                self.saved = {}; self.token_ids = ids
                adapter = self.new(); _, raw = adapter.infer(self.bound, "3" * 64, {})
                self.assertFalse(raw["native_v2_validated"])
                self.assertEqual(raw["native_capture"]["termination"]["stop_reason"], "unexplained_return")

    def test_qwen_eos_at_budget_is_valid(self):
        self.token_ids = [17] * 255 + [151643]
        adapter = self.new(); _, raw = adapter.infer(self.bound, "3" * 64, {})
        self.assertTrue(raw["native_v2_validated"])

    def test_missing_inherited_control_fails_before_forward(self):
        del self.model.thinker.config.values["suppress_tokens"]
        with self.assertRaisesRegex(ValueError, "missing observed"):
            self.new()
        self.assertEqual(self.calls, 0)
        self.assertNotIn("native-config.json", self.saved)

    def test_noncanonical_snapshot_retention_is_rejected_before_forward(self):
        with self.assertRaisesRegex(ValueError, "canonical UTF-8"):
            a.NativeCaptureAdapter(self.runtime, Path("unused"), b"{}", b"{}",
                lambda name, value: hashlib.sha256(json.dumps(value, indent=2).encode()).hexdigest())
        self.assertEqual(self.calls, 0)

    def test_nonempty_effective_suppression_is_rejected_before_forward(self):
        self.model.thinker.config.values["suppress_tokens"] = [17]
        with self.assertRaises(ValueError):
            self.new()
        self.assertEqual(self.calls, 0)

    def test_runtime_config_drift_before_forward(self):
        adapter = self.new()
        self.model.thinker.config.values["num_beams"] = 2
        with self.assertRaisesRegex(ValueError, "resolved generation config drift"):
            adapter.infer(self.bound, "3" * 64, {})
        self.assertEqual(self.calls, 0)
        self.assertNotIn("_prepare_generation_config", vars(self.model.thinker))

    def test_missing_observed_custom_controls_never_filled(self):
        self.skip_custom = True
        adapter = self.new()
        with self.assertRaisesRegex(ValueError, "missing native Qwen observations"):
            adapter.infer(self.bound, "3" * 64, {})
        self.assertTrue(self.saved["clip-000001.native-observation.json"]["inference_returned"])
        self.assertIn("clip-000001.native-returned-raw.json", self.saved)

    def test_injected_custom_processor_rejected_before_forward(self):
        self.custom_processors = [object()]
        adapter = self.new()
        with self.assertRaisesRegex(ValueError, "unreviewed Qwen custom controls"):
            adapter.infer(self.bound, "3" * 64, {})
        self.assertEqual(self.calls, 0)

    def test_frontend_missing_control_and_truncation_rejected(self):
        for mutation in ("missing", "truncation"):
            with self.subTest(mutation=mutation):
                self.saved = {}
                self.feature_options = {"sampling_rate": 16000, "padding": True, "truncation": False, "return_attention_mask": True}
                if mutation == "missing":
                    del self.feature_options["truncation"]
                else:
                    self.feature_options["truncation"] = True
                adapter = self.new()
                with self.assertRaises(ValueError):
                    adapter.infer(self.bound, "3" * 64, {})
                self.assertEqual(self.calls, 0)

    def test_qwen_exception_oom_timeout_restores_and_never_retries(self):
        for error in (RuntimeError("invented failure"), MemoryError("invented OOM"), TimeoutError("invented timeout")):
            with self.subTest(error=type(error).__name__):
                self.saved = {}; self.failure = error
                adapter = self.new()
                with self.assertRaises(type(error)):
                    adapter.infer(self.bound, "3" * 64, {})
                observation = self.saved["clip-000001.native-observation.json"]
                self.assertEqual(observation["exception_type"], type(error).__name__)
                self.assertFalse(observation["inference_returned"])
                self.assertNotIn("native_capture", observation)
                for name in ("_prepare_generation_config", "_get_logits_processor", "_get_stopping_criteria"):
                    self.assertNotIn(name, vars(self.model.thinker))
                self.assertIsInstance(self.wrapper.processor.feature_extractor, FakeFeature)
                with self.assertRaisesRegex(ValueError, "duplicate clip attempt"):
                    adapter.infer(self.bound, "3" * 64, {})

    def test_qwen_generation_tokens_survive_postprocessing_failure(self):
        self.post_failure = ValueError("invented detokenizer failure")
        adapter = self.new()
        with self.assertRaisesRegex(ValueError, "detokenizer failure"):
            adapter.infer(self.bound, "3" * 64, {})
        observed = self.saved["clip-000001.native-observation.json"]
        self.assertTrue(observed["generation_returned"])
        self.assertFalse(observed["inference_returned"])
        self.assertEqual(self.saved["clip-000001.native-generation-return.json"]["returned_sequence_ids"], [5, 6] + self.token_ids)
        self.assertEqual(observed["returned_generated_token_ids"], self.token_ids)
        self.assertNotIn("native_capture", observed)

    def setup_sense(self):
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.model_root = Path(temporary) / "models/sensevoice"; self.model_root.mkdir(parents=True)
        tokenizer_path = self.model_root / sense_binding.TOKENIZER_NAME
        tokenizer_path.write_bytes(f.INVENTED_TOKENIZER_BYTES)
        cmvn = self.model_root / "am.mvn"; cmvn.write_bytes(b"invented CMVN")
        self.frontend = WavFrontend(cmvn)
        self.sp = f.InventedTokenizer()
        self.ids = list(f.DEFAULT_IDS)
        self.skip_frontend = False
        self.frontend_frames = 160
        self.wrapper = SimpleNamespace(model=self.model, vad_model=None,
            kwargs={"tokenizer": self.sp, "frontend": self.frontend})
        self.runtime = SimpleNamespace(model_id=a.SENSE, wrapper=self.wrapper, declaration={"batch_size": 1},
            receipt={"assets": {"verified_files": [{"filename": "am.mvn", "bytes": len(b"invented CMVN"),
                "sha256": hashlib.sha256(b"invented CMVN").hexdigest()}]}})
        def inference(**kwargs):
            if not self.skip_frontend:
                kwargs["frontend"].forward(SimpleNamespace(shape=(1, self.frontend_frames)),
                    [self.frontend_frames])
            self.calls += 1
            if self.failure:
                raise self.failure
            text = kwargs["tokenizer"].decode(self.ids)
            return ([{"key": self.bound.opaque_id, "text": text}], {"test_only": True})
        self.model.inference = inference
        self.stack.enter_context(patch.object(sense_binding, "TOKENIZER_SHA256", f.INVENTED_TOKENIZER_SHA256))
        self.stack.enter_context(patch.object(runtime_support, "describe_cpu_tensor", return_value={"shape": [2, 560], "dtype": "torch.float32", "sha256": "8" * 64}))
        self.stack.enter_context(patch.object(sense_adapter, "_begin_clip", return_value=self.pcm))
        self.stack.enter_context(patch.object(sense_adapter, "require_verified_receipt"))
        self.stack.enter_context(patch.object(sense_adapter, "verify_pcm_array", side_effect=self.verify_pcm))
        self.stack.enter_context(patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}))

    def test_sense_success_real_binding_retains_ctc_not_eos(self):
        self.setup_sense(); adapter = self.new(); _, raw = adapter.infer(self.bound, "3" * 64, {})
        self.assertTrue(raw["native_v2_validated"])
        validate_native_capture(adapter.config, raw["native_capture"])
        self.assertEqual(raw["native_capture"]["termination"]["kind"], "full_input_ctc_return")
        self.assertNotIn("eos", raw["native_capture"]["termination"])
        self.assertEqual(raw["native_capture"]["termination"]["raw_token_ids"], f.DEFAULT_IDS)
        self.assertEqual(raw["native_capture"]["acoustic_verbatim_completeness"], "unknown")
        self.assertEqual(adapter.config["effective_config"]["frontend_config_sha256"],
                         a.canonical_sha256(self.saved["native-frontend-config.json"]))
        self.assertNotIn("forward", vars(self.frontend))

    def test_sense_unknown_lexical_and_bad_metadata_preserved_invalid(self):
        self.setup_sense()
        for ids in (f.METADATA_IDS + [0, 1], [1] + f.DEFAULT_IDS, f.METADATA_IDS):
            with self.subTest(ids=ids):
                self.saved = {}; self.ids = ids
                adapter = self.new(); _, raw = adapter.infer(self.bound, "3" * 64, {})
                self.assertFalse(raw["native_v2_validated"])
                self.assertEqual(raw["native_capture"]["termination"]["raw_token_ids"], ids)

    def test_sense_missing_frontend_observation_and_cropped_input_rejected(self):
        self.setup_sense()
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.saved = {}; self.skip_frontend = missing; self.frontend_frames = 80
                adapter = self.new()
                with self.assertRaises(ValueError):
                    adapter.infer(self.bound, "3" * 64, {})

    def test_sense_missing_frontend_attribute_before_inference(self):
        self.setup_sense(); del self.frontend.dither
        with self.assertRaises(AttributeError):
            self.new()
        self.assertEqual(self.calls, 0)

    def test_sense_missing_noncanonical_or_changed_runtime_asset_rejected(self):
        self.setup_sense()
        _, snapshot = a.sense_effective(self.runtime, self.model_root)
        changed = copy.deepcopy(snapshot); changed["attributes"]["dither"] = 0.0
        for raw in (None, json.dumps(snapshot, indent=2).encode(), json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()):
            with self.subTest(raw=raw):
                self.saved = {}
                with self.assertRaises(ValueError):
                    a.NativeCaptureAdapter(self.runtime, self.model_root, b"{}", b"{}", self.retain,
                        frontend_config_bytes=raw)
        self.assertEqual(self.calls, 0)

    def test_sense_runtime_frontend_drift_rejected(self):
        self.setup_sense(); adapter = self.new(); self.frontend.dither = 0.0
        with self.assertRaisesRegex(ValueError, "frontend runtime drift"):
            adapter.infer(self.bound, "3" * 64, {})
        self.assertEqual(self.calls, 0)

    def test_sense_exception_oom_timeout_no_capture_or_retry(self):
        self.setup_sense(); original_inference = self.model.inference
        for error in (RuntimeError("invented"), MemoryError("invented OOM"), TimeoutError("invented timeout")):
            with self.subTest(error=type(error).__name__):
                self.saved = {}; self.failure = error
                adapter = self.new()
                with self.assertRaises(type(error)):
                    adapter.infer(self.bound, "3" * 64, {})
                self.assertFalse(self.saved["clip-000001.native-observation.json"]["inference_returned"])
                self.assertIs(self.model.inference, original_inference)
                self.assertNotIn("forward", vars(self.frontend))
                with self.assertRaisesRegex(ValueError, "duplicate clip attempt"):
                    adapter.infer(self.bound, "3" * 64, {})

    def test_sense_raw_tokens_survive_result_binding_failure(self):
        self.setup_sense(); adapter = self.new()
        with patch.object(sense_adapter, "bound_sense_transcript", side_effect=ValueError("invented binding failure")):
            with self.assertRaisesRegex(ValueError, "binding failure"):
                adapter.infer(self.bound, "3" * 64, {})
        self.assertEqual(self.saved["clip-000001.native-tokenizer-input.json"]["raw_token_ids"], self.ids)
        self.assertIn("clip-000001.native-inference-return.json", self.saved)
        self.assertTrue(self.saved["clip-000001.native-observation.json"]["inference_returned"])

    def test_sense_tokens_survive_tokenizer_exception_without_false_return(self):
        self.setup_sense(); self.sp.decoded_override = object(); adapter = self.new()
        with self.assertRaises(ValueError):
            adapter.infer(self.bound, "3" * 64, {})
        self.assertEqual(self.saved["clip-000001.native-tokenizer-input.json"]["raw_token_ids"], self.ids)
        self.assertFalse(self.saved["clip-000001.native-observation.json"]["inference_returned"])

    def test_import_is_stdlib_and_no_model_packages_loaded(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "qwen_asr", "funasr", "transformers")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
