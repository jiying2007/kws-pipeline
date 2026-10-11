#!/usr/bin/env python3
"""Pure invented interfaces; no model packages, inference, weights, or audio.

The real new observer, native-v2 validators, retained SenseVoice binding/parser,
and method-restoration logic run. The scientific call bodies are represented by
mocks at their existing boundary; this is not a real-runtime compatibility pass.

Set KWS_QWEN_RETAINED_EVIDENCE to the independently resolved output/qwen06
directory to additionally replay hash-pinned Transformers method ASTs against
the actual retained prepared JSON. This remains a zero-model synthetic proof.
"""
import ast
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
            min_length=0, repetition_penalty=1.0, max_length=20,
            bos_token_id=None, pad_token_id=None, decoder_start_token_id=None,
            _from_model_config=True, use_cache=True)
    def to_dict(self):
        return copy.deepcopy(self.values)
    def get_generation_mode(self):
        return SimpleNamespace(value="greedy_search")


class Thinker:
    def __init__(self):
        self.config = Generation()
    def _prepare_generation_config(self, config, **kwargs):
        return copy.deepcopy(self.config), {}
    def _get_logits_processor(self, generation_config, logits_processor, input_ids_seq_length=None):
        return logits_processor
    def _get_stopping_criteria(self, generation_config, stopping_criteria):
        return stopping_criteria


class SyntheticTensor:
    """Synthetic only; patched torch interface, never an actual runtime Tensor."""
    def __init__(self, value, device="cpu", dtype="synthetic-int64"):
        self.value = copy.deepcopy(value)
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.shape = (len(value),) if type(value) is list else ()
        self.ndim = len(self.shape)
    def to(self, device):
        return SyntheticTensor(self.value, device, self.dtype)
    def unsqueeze(self, dim):
        if dim != 0 or self.ndim != 0:
            raise ValueError("invented tensor shape")
        return SyntheticTensor([self.value], self.device.type, self.dtype)
    def __getitem__(self, index):
        return SyntheticTensor(self.value[index], self.device.type, self.dtype)
    def __lt__(self, other):
        values = self.value if type(self.value) is list else [self.value]
        return SimpleNamespace(any=lambda: any(value < other for value in values))
    def tolist(self):
        return copy.deepcopy(self.value)


SYNTHETIC_TORCH = SimpleNamespace(Tensor=SyntheticTensor, int64="synthetic-int64", long="synthetic-int64",
    tensor=lambda value, device, dtype: SyntheticTensor(value, device, dtype), is_floating_point=lambda value: False)
RUNTIME_TOKEN_FIELDS = ("_bos_token_tensor", "_eos_token_tensor", "_pad_token_tensor", "_decoder_start_token_tensor")


def synthetic_active(config, prefix=2):
    config = copy.deepcopy(config)
    config.values.update(_bos_token_tensor=None, _decoder_start_token_tensor=None,
        _eos_token_tensor=SyntheticTensor(config.values["eos_token_id"]),
        _pad_token_tensor=SyntheticTensor(config.values["eos_token_id"][0]), max_length=prefix + 256)
    return config


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


class ActiveProjectionTests(unittest.TestCase):
    def setUp(self):
        self.prepared = a.generation_snapshot(Generation())
        self.active = synthetic_active(Generation())
        self.torch_patch = patch.dict(sys.modules, {"torch": SYNTHETIC_TORCH})
        self.torch_patch.start(); self.addCleanup(self.torch_patch.stop)

    def project(self, config=None, prefix=2):
        return a.active_generation_snapshot(self.active if config is None else config, self.prepared, prefix)

    def test_exact_projection_and_prepared_identity_unchanged(self):
        original_prepared = copy.deepcopy(self.prepared)
        original_hash = a.canonical_sha256(self.prepared)
        config, tokens = self.project()
        expected = copy.deepcopy(self.prepared); expected["max_length"] = 258
        self.assertEqual(config, expected)
        self.assertEqual(tokens, {"_bos_token_tensor": None, "_decoder_start_token_tensor": None,
            "_eos_token_tensor": {"device": "cpu", "dtype": "torch.int64", "shape": [2], "values": [151645, 151643]},
            "_pad_token_tensor": {"device": "cpu", "dtype": "torch.int64", "shape": [], "values": 151645}})
        a.clone(tokens)
        self.assertEqual(self.prepared, original_prepared)
        self.assertEqual(a.canonical_sha256(self.prepared), original_hash)
        self.assertEqual(a.generation_snapshot(Generation()), original_prepared)
        self.assertEqual(set(self.active.values), set(self.prepared) | set(RUNTIME_TOKEN_FIELDS))
        self.assertIs(type(self.active.values["_eos_token_tensor"]), SyntheticTensor)
        with self.assertRaisesRegex(ValueError, "exact JSON types"):
            a.generation_snapshot(self.active)

    def test_projection_detaches_even_if_to_dict_returns_live_mapping(self):
        self.active.to_dict = lambda: self.active.values
        original_keys = set(self.active.values)
        original_eos = self.active.values["_eos_token_tensor"]
        config, tokens = self.project()
        config["eos_token_id"].append(99)
        tokens["_eos_token_tensor"]["values"].append(99)
        self.assertEqual(set(self.active.values), original_keys)
        self.assertIs(self.active.values["_eos_token_tensor"], original_eos)
        self.assertEqual(original_eos.tolist(), [151645, 151643])
        self.assertEqual(self.active.values["eos_token_id"], [151645, 151643])

    def test_all_four_runtime_fields_required(self):
        for key in RUNTIME_TOKEN_FIELDS:
            with self.subTest(key=key):
                config = copy.deepcopy(self.active); del config.values[key]
                with self.assertRaisesRegex(ValueError, "missing Qwen runtime token field"):
                    self.project(config)

    def test_none_fields_and_mandatory_tensors_are_not_interchangeable(self):
        for key in RUNTIME_TOKEN_FIELDS:
            with self.subTest(key=key):
                config = copy.deepcopy(self.active)
                config.values[key] = SyntheticTensor(151645) if config.values[key] is None else None
                with self.assertRaises(ValueError):
                    self.project(config)

    def test_only_exact_tensor_type_not_duck_or_subclass(self):
        class TensorSubclass(SyntheticTensor):
            pass
        duck = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype=SYNTHETIC_TORCH.int64,
            shape=(2,), tolist=lambda: [151645, 151643])
        for value in ([151645, 151643], 151645, duck, TensorSubclass([151645, 151643]), object()):
            with self.subTest(value=type(value).__name__):
                config = copy.deepcopy(self.active); config.values["_eos_token_tensor"] = value
                with self.assertRaisesRegex(ValueError, "exact Tensor"):
                    self.project(config)

    def test_cpu_and_exact_int64_dtype_required(self):
        class EqualDtype:
            def __eq__(self, other):
                return True
        for device, dtype in (("cuda", SYNTHETIC_TORCH.int64), ("meta", SYNTHETIC_TORCH.int64),
                              ("cpu", "synthetic-int32"), ("cpu", "synthetic-float64"), ("cpu", EqualDtype())):
            with self.subTest(device=device, dtype=type(dtype).__name__):
                config = copy.deepcopy(self.active)
                config.values["_eos_token_tensor"] = SyntheticTensor([151645, 151643], device, dtype)
                with self.assertRaisesRegex(ValueError, "CPU int64"):
                    self.project(config)

    def test_exact_bounded_shape_checked_before_values(self):
        for key, shape in (("_eos_token_tensor", ()), ("_eos_token_tensor", (1, 2)),
                           ("_eos_token_tensor", (65536,)), ("_eos_token_tensor", (True,)),
                           ("_pad_token_tensor", (1,))):
            with self.subTest(key=key, shape=shape):
                config = copy.deepcopy(self.active); tensor = config.values[key]; tensor.shape = shape
                tensor.tolist = lambda: self.fail("values read before bounded shape validation")
                with self.assertRaisesRegex(ValueError, "shape drift"):
                    self.project(config)

    def test_exact_integer_token_values_no_bool_float_or_wrong_order(self):
        for key, value in (("_eos_token_tensor", [151643, 151645]), ("_eos_token_tensor", [151645, 151645]),
                           ("_eos_token_tensor", [151645, True]), ("_eos_token_tensor", [151645.0, 151643]),
                           ("_pad_token_tensor", 151643), ("_pad_token_tensor", True),
                           ("_pad_token_tensor", 151645.0)):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.active); config.values[key] = SyntheticTensor(value)
                with self.assertRaisesRegex(ValueError, "value drift"):
                    self.project(config)

    def test_unknown_non_json_and_nonfinite_remain_rejected(self):
        for key in ("unknown_public", "_unknown_private"):
            for value in (SyntheticTensor(7), {"nested": SyntheticTensor(7)}, (7,), object(),
                          float("nan"), float("inf"), float("-inf")):
                with self.subTest(key=key, value=type(value).__name__):
                    config = copy.deepcopy(self.active); config.values[key] = value
                    with self.assertRaises(ValueError):
                        self.project(config)

    def test_entire_remaining_public_and_private_json_config_frozen(self):
        for key, value in (("min_length", 1), ("repetition_penalty", 1.1), ("use_cache", False),
                           ("_from_model_config", False), ("_unknown_private", None),
                           ("unknown_public", 0), ("eos_token_id", [151643, 151645]),
                           ("max_length", 257), ("max_length", 259), ("max_length", 258.0)):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.active); config.values[key] = value
                with self.assertRaisesRegex(ValueError, "active generation config drift"):
                    self.project(config)
        for key in self.prepared:
            with self.subTest(missing=key):
                config = copy.deepcopy(self.active); del config.values[key]
                with self.assertRaisesRegex(ValueError, "active generation config drift"):
                    self.project(config)

    def test_observed_positive_integer_length_required(self):
        for prefix in (None, True, 0, -1, 2.0, "2"):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(ValueError, "generation input length"):
                    self.project(prefix=prefix)
        with self.assertRaisesRegex(ValueError, "active generation config drift"):
            self.project(prefix=3)

    def test_loaded_torch_required_without_import_fallback(self):
        with patch.dict(sys.modules):
            del sys.modules["torch"]
            with self.assertRaisesRegex(ValueError, "missing loaded Qwen torch runtime"):
                self.project()
            self.assertNotIn("torch", sys.modules)


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
        self.active_mutation = lambda config: None
        self.logits_input_length = 2
        self.input_shape = (1, 2)
        self.stopping_first = False
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
        config = synthetic_active(config)
        self.active_mutation(config)
        if not self.skip_custom:
            if self.stopping_first:
                thinker._get_stopping_criteria(config, [])
            thinker._get_logits_processor(config, self.custom_processors, self.logits_input_length)
            thinker._get_stopping_criteria(config, [])
        self.calls += 1
        if self.failure:
            raise self.failure
        return SimpleNamespace(sequences=Sequence([5, 6] + self.token_ids))

    def mock_qwen(self, runtime, bound, decoder_sha):
        runtime.wrapper.processor.feature_extractor([self.pcm], **self.feature_options)
        with patch.dict(sys.modules, {"torch": SYNTHETIC_TORCH}):
            runtime.wrapper.model.generate(input_ids=SimpleNamespace(shape=self.input_shape))
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
        observed = self.saved["clip-000001.native-observation.json"]
        self.assertEqual(observed["generation_input_ids_seq_length"], 2)
        self.assertEqual(observed["logits_input_ids_seq_length"], 2)
        for key in ("custom_logits_processors", "custom_stopping_criteria"):
            self.assertEqual(observed[key + "_resolved_generation_config"]["max_length"], 258)
            self.assertEqual(observed[key + "_runtime_special_tokens"]["_pad_token_tensor"]["values"], 151645)

    def test_active_config_rejections_restore_consume_and_never_forward(self):
        for key, value in (("min_length", 1), ("_from_model_config", False), ("_new_private", 0),
                           ("_other_tensor", SyntheticTensor(7)), ("max_length", 259),
                           ("_pad_token_tensor", SyntheticTensor(151643))):
            with self.subTest(key=key):
                self.saved = {}; self.active_mutation = lambda config: config.values.update({key: value})
                adapter = self.new()
                with self.assertRaises(ValueError):
                    adapter.infer(self.bound, "3" * 64, {})
                self.assertEqual(self.calls, 0)
                observed = self.saved["clip-000001.native-observation.json"]
                self.assertFalse(observed["inference_returned"])
                self.assertNotIn("generation_returned", observed)
                self.assertNotIn("native_capture", observed)
                self.assertNotIn("clip-000001.native-generation-return.json", self.saved)
                for name in ("_prepare_generation_config", "_get_logits_processor", "_get_stopping_criteria"):
                    self.assertNotIn(name, vars(self.model.thinker))
                self.assertIsInstance(self.wrapper.processor.feature_extractor, FakeFeature)
                with self.assertRaisesRegex(ValueError, "duplicate clip attempt"):
                    adapter.infer(self.bound, "3" * 64, {})

    def test_actual_input_length_bound_before_generate_and_matches_logits(self):
        for shape, length in (((1, 2), None), ((1, 2), True), ((1, 2), 2.0), ((1, 2), 3),
                              ((1, 3), 2), ((2, 2), 2), ((1, 0), 0), ((1, True), True)):
            with self.subTest(shape=shape, length=length):
                self.saved = {}; self.input_shape = shape; self.logits_input_length = length
                adapter = self.new()
                with self.assertRaises(ValueError):
                    adapter.infer(self.bound, "3" * 64, {})
                self.assertEqual(self.calls, 0)

    def test_stopping_uses_same_bound_length_and_rechecks_entire_config(self):
        def mutate_after_logits(generation_config, logits_processor, input_ids_seq_length):
            generation_config.values["min_length"] = 1
            return logits_processor
        with patch.object(self.model.thinker, "_get_logits_processor", mutate_after_logits):
            adapter = self.new()
            with self.assertRaisesRegex(ValueError, "active generation config drift"):
                adapter.infer(self.bound, "3" * 64, {})
            observed = self.saved["clip-000001.native-observation.json"]
            self.assertEqual(observed["custom_logits_processors"], [])
            self.assertNotIn("custom_stopping_criteria", observed)
            self.assertEqual(self.calls, 0)

    def test_stopping_before_logits_rejected_without_invented_length(self):
        self.stopping_first = True
        adapter = self.new()
        with self.assertRaisesRegex(ValueError, "missing Qwen logits input length"):
            adapter.infer(self.bound, "3" * 64, {})
        observed = self.saved["clip-000001.native-observation.json"]
        self.assertEqual(observed["generation_input_ids_seq_length"], 2)
        self.assertNotIn("custom_stopping_criteria", observed)
        self.assertEqual(self.calls, 0)

    def test_stopping_explicit_drifted_length_is_rejected(self):
        def stop(generation_config, stopping_criteria, input_ids_seq_length=None):
            return stopping_criteria
        # A default is not an observation. The pinned method has no such
        # parameter; a different caller supplying one must still match.
        original_generate = self.model.generate
        def generate(**kwargs):
            config, _ = self.model.thinker._prepare_generation_config(None)
            config = synthetic_active(config)
            self.model.thinker._get_logits_processor(config, [], 2)
            self.model.thinker._get_stopping_criteria(config, [], 3)
            return original_generate(**kwargs)
        with patch.object(self.model.thinker, "_get_stopping_criteria", stop), \
             patch.object(self.model, "generate", generate):
            adapter = self.new()
            with self.assertRaisesRegex(ValueError, "Qwen stopping input length drift"):
                adapter.infer(self.bound, "3" * 64, {})
            self.assertEqual(self.calls, 0)

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


@unittest.skipUnless(os.environ.get("KWS_QWEN_RETAINED_EVIDENCE"), "explicit retained-evidence path required for AST proof")
class PinnedSourceReplayTests(unittest.TestCase):
    """Exact retained method bodies and prepared bytes; no dependency imports."""
    IDENTITIES = {
        "native-resolved-generation-config.json": "20b7cd2778d4bfd858e2eb051ee96cc8c564657a495a7ae9325da044c2764021",
        "retained-source/site-packages/transformers/generation/utils.py": "55fd5c50aa785e226afa03e4507b8d27c0328714c0bd048bc60af9e8813a1173",
        "retained-source/site-packages/transformers/generation/configuration_utils.py": "d83f2281f939402be1633a29f3c760e29f5d2f284258d8ed99693b873744074b",
    }

    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["KWS_QWEN_RETAINED_EVIDENCE"])
        cls.raw = {name: (cls.root / name).read_bytes() for name in cls.IDENTITIES}
        for name, identity in cls.IDENTITIES.items():
            if hashlib.sha256(cls.raw[name]).hexdigest() != identity:
                raise ValueError("retained AST proof source identity mismatch: " + name)
        cls.prepared = json.loads(cls.raw["native-resolved-generation-config.json"])
        cls.utils = ast.parse(cls.raw["retained-source/site-packages/transformers/generation/utils.py"])
        cls.configuration = ast.parse(cls.raw["retained-source/site-packages/transformers/generation/configuration_utils.py"])
        cls.ns = {"copy": copy, "__version__": "4.57.6", "torch": SYNTHETIC_TORCH,
            "logger": SimpleNamespace(warning=lambda *args: None, warning_once=lambda *args: None),
            "isin_mps_friendly": lambda **kwargs: SimpleNamespace(
                any=lambda: kwargs["test_elements"].value in kwargs["elements"].value)}
        nodes = [cls.method(cls.configuration, "GenerationConfig", name) for name in ("to_dict", "dict_dtype_to_str")]
        nodes += [cls.method(cls.utils, "GenerationMixin", name) for name in ("_prepare_special_tokens", "_prepare_generated_length")]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "retained-transformers-methods-only", "exec"), cls.ns)
        cls.Config = type("ExtractedGenerationConfig", (), {name: cls.ns[name] for name in ("to_dict", "dict_dtype_to_str")})

    @staticmethod
    def method(tree, owner, name):
        body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
        return next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)

    def fresh(self):
        config = self.Config(); config.__dict__.update(copy.deepcopy(self.prepared))
        return config

    def activate(self, config, prefix):
        host = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False), device="cpu")
        self.ns["_prepare_special_tokens"](host, config, kwargs_has_attention_mask=True, device="cpu")
        return self.ns["_prepare_generated_length"](host, config, has_default_max_length=True,
            has_default_min_length=True, model_input_name="input_ids", input_ids_length=prefix,
            inputs_tensor=SimpleNamespace(shape=(1, prefix)))

    def tearDown(self):
        self.assertEqual({name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            for name in self.IDENTITIES}, self.IDENTITIES)
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "qwen_asr", "funasr", "transformers")))

    def test_actual_prepared_bytes_and_identity_remain_unchanged(self):
        self.assertEqual(a.generation_snapshot(self.fresh()), self.prepared)
        self.assertEqual(a.canonical_sha256(self.prepared), self.IDENTITIES["native-resolved-generation-config.json"])
        self.assertEqual(len(self.prepared), 67)

    def test_exact_pinned_mutations_pass_new_projection_and_fail_prepared_clone(self):
        for prefix in (2, 17, 100):
            with self.subTest(prefix=prefix):
                config = self.activate(self.fresh(), prefix)
                self.assertEqual(set(config.to_dict()) - set(self.prepared), set(RUNTIME_TOKEN_FIELDS))
                with self.assertRaisesRegex(ValueError, "exact JSON types"):
                    a.generation_snapshot(config)
                with patch.dict(sys.modules, {"torch": SYNTHETIC_TORCH}):
                    projected, tokens = a.active_generation_snapshot(config, self.prepared, prefix)
                expected = copy.deepcopy(self.prepared); expected["max_length"] = prefix + 256
                self.assertEqual(projected, expected)
                self.assertEqual(tokens["_eos_token_tensor"]["values"], [151645, 151643])
                self.assertEqual(tokens["_pad_token_tensor"]["shape"], [])
                self.assertEqual(tokens["_pad_token_tensor"]["values"], 151645)
                self.assertEqual(config.max_length, prefix + 256)
                self.assertIs(type(config._eos_token_tensor), SyntheticTensor)
                self.assertEqual(a.generation_snapshot(self.fresh()), self.prepared)

    def test_pinned_active_json_drift_and_new_non_json_fields_are_rejected(self):
        for key, value in (("min_length", 1), ("_from_model_config", False), ("max_length", 259),
                           ("new_field", SyntheticTensor(1)), ("_new_private", 1), ("new_nan", float("nan"))):
            with self.subTest(key=key):
                config = self.activate(self.fresh(), 2); setattr(config, key, value)
                with patch.dict(sys.modules, {"torch": SYNTHETIC_TORCH}):
                    with self.assertRaises(ValueError):
                        a.active_generation_snapshot(config, self.prepared, 2)

    def test_pinned_generate_order_and_logits_length_binding(self):
        generate = self.method(self.utils, "GenerationMixin", "generate")
        calls = {name: [node for node in ast.walk(generate) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == name]
            for name in ("_prepare_generation_config", "_prepare_special_tokens", "_prepare_generated_length",
                         "_get_logits_processor", "_get_stopping_criteria")}
        order = [min(node.lineno for node in calls[name]) for name in calls]
        self.assertEqual(order, sorted(order))
        logits = calls["_get_logits_processor"][0]
        length = next(keyword.value for keyword in logits.keywords if keyword.arg == "input_ids_seq_length")
        self.assertIsInstance(length, ast.Name); self.assertEqual(length.id, "input_ids_length")
        stops = calls["_get_stopping_criteria"][0]
        self.assertNotIn("input_ids_seq_length", [keyword.arg for keyword in stops.keywords])
        decode = [node.lineno for node in ast.walk(generate) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "decoding_method"]
        self.assertLess(order[-1], min(decode))


if __name__ == "__main__":
    def zero_model_audit(event, args):
        if event in ("subprocess.Popen", "os.system", "socket.connect", "socket.bind"):
            raise RuntimeError("synthetic adapter tests forbid external execution/network")
        if event == "import" and args[0].split(".")[0] in ("torch", "numpy", "qwen_asr", "funasr", "transformers", "whisper"):
            raise RuntimeError("synthetic adapter tests forbid model-package imports")
    sys.addaudithook(zero_model_audit)
    unittest.main(verbosity=2)
