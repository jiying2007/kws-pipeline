#!/usr/bin/env python3
"""Offline relocated-tokenizer regression tests with explicitly invented bytes.

The fixture is NOT a SentencePiece model, production asset, transcript, or model
result. Only a minimal tokenizer interface and model/PCM/tensor dependencies are
stubbed. The retained verifier, parser-byte hash, corrected parser, capturing
tokenizer and source-specific binding body execute for real. Expected tokenizer
SHA256 is patched only for this invented-byte test fixture; no production file,
verifier or binder is modified or replaced. No packages, weights or Docker run.
"""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RETAINED = HERE.parent / "qwen6_asr"
for folder in (RETAINED, RETAINED / "core", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import sense_binding as retained

spec = importlib.util.spec_from_file_location("screen32_sense_adapter", HERE / "sense_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

INVENTED_TOKENIZER_BYTES = b"INVENTED OFFLINE TEST FIXTURE; NOT A SENTENCEPIECE MODEL\x00\xff\n"
INVENTED_TOKENIZER_SHA256 = hashlib.sha256(INVENTED_TOKENIZER_BYTES).hexdigest()
PRODUCTION_TOKENIZER_SHA256 = retained.TOKENIZER_SHA256
METADATA_IDS = [25001, 25009, 25010, 25011]
DEFAULT_IDS = METADATA_IDS + [1]


class InventedTokenizer:
    """Small deterministic stub; token IDs/pieces are test data, not asset claims."""
    def __init__(self):
        self.vocabulary_size = 25055
        self.unknown_id = 0
        self.pieces = {0: "<unk>", 1: "fictional transcript", 2: "<|literal|>",
                       25001: "<|zh|>", 25009: "<|EMO_UNKNOWN|>",
                       25010: "<|Speech|>", 25011: "<|woitn|>"}
        self.unknown_ids = {0}
        self.decoded_override = None
        self.mutate_decode_ids = False
        self.decode_calls = []
        self.vocab_calls = 0
        self.sp = self

    def vocab_size(self):
        self.vocab_calls += 1
        return self.vocabulary_size

    def unk_id(self):
        return self.unknown_id

    def id_to_piece(self, token_id):
        return self.pieces.get(token_id, "fictional piece")

    def is_unknown(self, token_id):
        return token_id in self.unknown_ids

    def render(self, ids):
        return "".join(self.id_to_piece(token_id) for token_id in ids)

    def decode(self, ids):
        self.decode_calls.append(list(ids))
        text = self.render(ids) if self.decoded_override is None else self.decoded_override
        if self.mutate_decode_ids:
            ids.append(1)
        return text


def function_node(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "relocated-runtime" / "models" / "sensevoice"
        self.root.mkdir(parents=True)
        self.path = self.root / retained.TOKENIZER_NAME
        self.path.write_bytes(INVENTED_TOKENIZER_BYTES)
        self.sp = InventedTokenizer()
        self.sha_patch = patch.object(retained, "TOKENIZER_SHA256", INVENTED_TOKENIZER_SHA256)
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def bind(self, ids=None, decoded=None, capture=None):
        ids = list(DEFAULT_IDS if ids is None else ids)
        decoded = self.sp.render(ids) if decoded is None else decoded
        if capture is None:
            capture = {"token_ids": ids, "decoded": decoded, "token_scope": retained.TOKEN_SCOPE}
        return adapter.bound_sense_transcript(decoded, capture, self.sp, self.path)


class RealBindingTests(FixtureCase):
    def test_historical_default_absent_but_relocated_explicit_path_succeeds(self):
        decoded = self.sp.render(DEFAULT_IDS)
        capture = {"token_ids": list(DEFAULT_IDS), "decoded": decoded, "token_scope": retained.TOKEN_SCOPE}
        # Preserve the old real lock/default-path relationship while exercising
        # its failure; the source checkout deliberately has no historical asset.
        with patch.object(retained, "TOKENIZER_SHA256", PRODUCTION_TOKENIZER_SHA256):
            historical = retained.tokenizer_path()
            self.assertFalse(historical.exists(), "This offline regression requires the historical default to be absent")
            self.assertNotEqual(historical, self.path)
            with self.assertRaisesRegex(ValueError, "Pinned SenseVoice tokenizer bytes changed"):
                retained.bound_sense_transcript(decoded, capture, self.sp)
        row, metadata, binding = adapter.bound_sense_transcript(decoded, capture, self.sp, self.path)
        self.assertEqual(row, {"raw_text": "fictional transcript", "completeness": "complete", "quality_flags": []})
        self.assertEqual(metadata, ["zh", "EMO_UNKNOWN", "Speech", "woitn"])
        self.assertEqual(binding["tokenizer"]["sha256"], INVENTED_TOKENIZER_SHA256)
        self.assertEqual(binding["parser_contract_sha256"], retained.CONTRACT_SHA256)
        self.assertEqual(binding["raw_decoded_sha256"], hashlib.sha256(decoded.encode()).hexdigest())
        self.assertTrue(binding["raw_redecode_matches_capture"])
        self.assertTrue(binding["metadata_boundary_verified"])
        self.assertFalse(binding["lexical_unknown_present"])
        self.assertEqual(binding["raw_token_ids"], DEFAULT_IDS)
        self.assertEqual(binding["metadata_token_ids"], METADATA_IDS)
        self.assertEqual(binding["lexical_token_ids"], [1])
        self.assertEqual(self.sp.decode_calls, [DEFAULT_IDS])

    def test_actual_verifier_is_used_and_production_hash_is_not_accepted_for_fixture(self):
        self.assertIs(adapter.verify_tokenizer, retained.verify_tokenizer)
        with patch.object(retained, "TOKENIZER_SHA256", PRODUCTION_TOKENIZER_SHA256):
            with self.assertRaisesRegex(ValueError, "Pinned SenseVoice tokenizer bytes changed"):
                self.bind()
        self.assertEqual(self.sp.decode_calls, [])

    def test_explicit_path_rejects_missing_symlink_directory_and_wrong_bytes(self):
        for kind in ("missing", "symlink", "directory", "wrong_hash"):
            with self.subTest(kind=kind):
                self.path.unlink()
                target = self.root / "invented-target"
                if kind == "symlink":
                    target.write_bytes(INVENTED_TOKENIZER_BYTES)
                    self.path.symlink_to(target)
                elif kind == "directory":
                    self.path.mkdir()
                elif kind == "wrong_hash":
                    self.path.write_bytes(b"different invented bytes")
                with self.assertRaisesRegex(ValueError, "Pinned SenseVoice tokenizer bytes changed"):
                    self.bind()
                if self.path.is_dir() and not self.path.is_symlink():
                    self.path.rmdir()
                elif self.path.exists() or self.path.is_symlink():
                    self.path.unlink()
                self.path.write_bytes(INVENTED_TOKENIZER_BYTES)
        self.assertEqual(self.sp.decode_calls, [])

    def test_vocabulary_and_unknown_semantics_remain_fail_closed(self):
        mutations = (
            ("wrong vocabulary", lambda sp: setattr(sp, "vocabulary_size", 25054)),
            ("bool vocabulary", lambda sp: setattr(sp, "vocabulary_size", True)),
            ("wrong unknown ID", lambda sp: setattr(sp, "unknown_id", 1)),
            ("bool unknown ID", lambda sp: setattr(sp, "unknown_id", False)),
            ("wrong unknown piece", lambda sp: sp.pieces.update({0: "not-unk"})),
            ("unknown ID not marked unknown", lambda sp: sp.unknown_ids.clear()),
            ("wrong emotion piece", lambda sp: sp.pieces.update({25009: "<|NEUTRAL|>"})),
            ("emotion incorrectly unknown", lambda sp: sp.unknown_ids.add(25009)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                self.sp = InventedTokenizer()
                mutate(self.sp)
                with self.assertRaisesRegex(ValueError, "vocabulary/unknown semantics mismatch"):
                    self.bind()
                self.assertEqual(self.sp.decode_calls, [])

    def test_metadata_emotion_unknown_is_not_lexical_unknown(self):
        row, metadata, binding = self.bind()
        self.assertIn("EMO_UNKNOWN", metadata)
        self.assertEqual(row["quality_flags"], [])
        self.assertFalse(binding["lexical_unknown_present"])
        row, _, binding = self.bind(ids=METADATA_IDS + [0, 1])
        self.assertEqual(row["quality_flags"], ["lexical_unknown"])
        self.assertTrue(binding["lexical_unknown_present"])
        self.assertEqual(binding["lexical_token_ids"], [0, 1])

    def test_redecode_text_mismatch_and_id_mutation_are_rejected(self):
        self.sp.decoded_override = "invented different decode"
        with self.assertRaisesRegex(ValueError, "Exact captured raw token re-decode mismatch"):
            self.bind()
        self.sp.decoded_override = None
        self.sp.mutate_decode_ids = True
        with self.assertRaisesRegex(ValueError, "Exact captured raw token re-decode mismatch"):
            self.bind()

    def test_capture_association_is_not_relaxed(self):
        decoded = self.sp.render(DEFAULT_IDS)
        base = {"token_ids": list(DEFAULT_IDS), "decoded": decoded, "token_scope": retained.TOKEN_SCOPE}
        cases = [dict(base, decoded="different"), dict(base, token_scope="framewise"),
                 dict(base, extra=True), {k: v for k, v in base.items() if k != "decoded"}, []]
        for capture in cases:
            with self.subTest(capture=capture):
                with self.assertRaisesRegex(ValueError, "captured output association mismatch"):
                    self.bind(decoded=decoded, capture=capture)
        self.assertEqual(self.sp.decode_calls, [])

    def test_invalid_or_out_of_vocabulary_tokens_are_rejected(self):
        for ids in ([25055], [-1], [True], ["1"], [1] * 4097):
            with self.subTest(ids=ids[:3]):
                capture = {"token_ids": ids, "decoded": "fictional", "token_scope": retained.TOKEN_SCOPE}
                with self.assertRaises(ValueError):
                    self.bind(decoded="fictional", capture=capture)
        self.assertEqual(self.sp.decode_calls, [])

    def test_invalid_metadata_token_boundary_preserves_complete_raw_decode(self):
        for ids in ([1] + DEFAULT_IDS, [25001, 25010, 25009, 25011, 1], METADATA_IDS[:3], []):
            with self.subTest(ids=ids):
                row, metadata, binding = self.bind(ids=ids)
                self.assertEqual(row, {"raw_text": self.sp.render(ids), "completeness": "unknown",
                                       "quality_flags": ["decoding_warning"]})
                self.assertEqual(metadata, [])
                self.assertFalse(binding["metadata_boundary_verified"])
                self.assertIsNone(binding["lexical_token_ids"])
                self.assertIsNone(binding["lexical_unknown_present"])

    def test_text_prefix_must_agree_with_four_token_pieces(self):
        decoded = "fictional prefix" + self.sp.render(DEFAULT_IDS)
        self.sp.decoded_override = decoded
        row, metadata, binding = self.bind(decoded=decoded)
        self.assertEqual(row["raw_text"], decoded)
        self.assertEqual(row["completeness"], "unknown")
        self.assertEqual(metadata, [])
        self.assertFalse(binding["metadata_boundary_verified"])

    def test_retained_parser_quality_flags_and_empty_lexical_boundary(self):
        row, _, binding = self.bind(ids=METADATA_IDS)
        self.assertEqual(row["quality_flags"], ["empty_transcript"])
        self.assertTrue(binding["metadata_boundary_verified"])
        self.assertEqual(binding["lexical_token_ids"], [])
        row, _, _ = self.bind(ids=METADATA_IDS + [2])
        self.assertEqual(row["raw_text"], "<|literal|>")
        self.assertEqual(row["quality_flags"], ["decoding_warning"])


class RuntimePathTests(FixtureCase):
    def test_valid_explicit_runtime_path(self):
        self.assertEqual(adapter.runtime_tokenizer_path(self.root), self.path)

    def test_invalid_model_roots(self):
        for root in (Path("models/sensevoice"), self.root.parent / "wrong", self.root.parent.parent / "sensevoice"):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "model root mismatch"):
                    adapter.runtime_tokenizer_path(root)

    def test_root_and_parent_symlinks_rejected(self):
        linked_parent = Path(self.temporary.name) / "linked-runtime"
        linked_parent.symlink_to(self.root.parent.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "model root alias"):
            adapter.runtime_tokenizer_path(linked_parent / "models" / "sensevoice")
        alias_parent = Path(self.temporary.name) / "alias-runtime" / "models"
        alias_parent.mkdir(parents=True)
        (alias_parent / "sensevoice").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "model root alias"):
            adapter.runtime_tokenizer_path(alias_parent / "sensevoice")

    def test_missing_symlink_hardlink_and_directory_tokenizer_rejected(self):
        for kind in ("missing", "symlink", "dangling_symlink", "hardlink", "directory"):
            with self.subTest(kind=kind):
                self.path.unlink()
                target = self.root / ("invented-target-" + kind)
                if kind in ("symlink", "hardlink"):
                    target.write_bytes(INVENTED_TOKENIZER_BYTES)
                if kind in ("symlink", "dangling_symlink"):
                    self.path.symlink_to(target)
                elif kind == "hardlink":
                    os.link(target, self.path)
                elif kind == "directory":
                    self.path.mkdir()
                with self.assertRaisesRegex(ValueError, "tokenizer file"):
                    adapter.runtime_tokenizer_path(self.root)
                if self.path.is_dir() and not self.path.is_symlink():
                    self.path.rmdir()
                elif self.path.exists() or self.path.is_symlink():
                    self.path.unlink()
                self.path.write_bytes(INVENTED_TOKENIZER_BYTES)


class ScientificParityTests(unittest.TestCase):
    def test_binding_ast_has_only_explicit_path_parameter_and_verify_argument_delta(self):
        before = function_node(RETAINED / "core/sense_binding.py", "bound_sense_transcript")
        after = function_node(HERE / "sense_adapter.py", "bound_sense_transcript")
        self.assertEqual([arg.arg for arg in after.args.args],
                         [arg.arg for arg in before.args.args] + ["tokenizer_path"])
        after.args.args.pop()
        calls = [node for node in ast.walk(after) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "verify_tokenizer"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.dump(calls[0].args[-1]), ast.dump(ast.Name(id="tokenizer_path", ctx=ast.Load())))
        self.assertEqual(len(calls[0].args), 2)
        self.assertEqual(calls[0].keywords, [])
        calls[0].args.pop()
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_scientific_inference_arguments_exactly_match_retained_source(self):
        before = function_node(RETAINED / "core/asr_stage/adapters.py", "infer_sense_once_after_approval")
        after = function_node(HERE / "sense_adapter.py", "infer_sense_once_after_approval")
        def inference_calls(node):
            return [call for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute) and call.func.attr == "inference"]
        original, relocated = inference_calls(before), inference_calls(after)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(relocated), 1)
        self.assertEqual(ast.dump(original[0], include_attributes=False), ast.dump(relocated[0], include_attributes=False))
        # Output/evidence payload is scientific state too; diagnostics cannot
        # silently introduce text repair, ITN, or change token evidence scope.
        old_return = next(node for node in ast.walk(before) if isinstance(node, ast.Return))
        new_return = next(node for node in ast.walk(after) if isinstance(node, ast.Return))
        self.assertEqual(ast.dump(old_return, include_attributes=False), ast.dump(new_return, include_attributes=False))

    def test_import_does_not_load_model_or_runtime_packages(self):
        self.assertFalse(any(name in sys.modules for name in
                             ("torch", "numpy", "sentencepiece", "funasr", "qwen_asr", "transformers")))


class InferencePlumbingTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.state = {"untrusted_old_state": True, "model_inference_returned": True}
        self.bound = SimpleNamespace(opaque_id="clip-000001", descriptor={"float32_pcm": {"invented": True}})
        self.pcm = object()
        self.frontend = object()
        self.forward_calls = []
        self.forward_error = None
        self.after_forward = None
        self.output_override = None
        self.skip_capture = False
        self.vocab_calls_at_forward = []
        def inference(**kwargs):
            self.forward_calls.append(kwargs)
            self.vocab_calls_at_forward.append(self.sp.vocab_calls)
            if self.forward_error is not None:
                raise self.forward_error
            decoded = self.sp.render(DEFAULT_IDS) if self.skip_capture else kwargs["tokenizer"].decode(list(DEFAULT_IDS))
            if self.after_forward is not None:
                self.after_forward()
            if self.output_override is not None:
                return self.output_override
            return ([{"key": self.bound.opaque_id, "text": decoded}], {"invented": True})
        self.model = SimpleNamespace(inference=inference)
        self.wrapper = SimpleNamespace(model=self.model, kwargs={"tokenizer": self.sp, "frontend": self.frontend})
        self.runtime = SimpleNamespace(model_id=adapter.SENSE, wrapper=self.wrapper)
        self.torch = ModuleType("torch")
        self.torch.inference_mode = nullcontext
        # These stubs isolate CPU/model dependencies. The real explicit path,
        # verifier, capture, parser and token binding remain in the call path.
        self.stack = []
        for guard in (patch.object(adapter, "require_verified_receipt"),
                      patch.object(adapter, "_begin_clip", return_value=self.pcm),
                      patch.object(adapter, "verify_pcm_array"),
                      patch.dict(sys.modules, {"torch": self.torch})):
            self.stack.append(guard.start())
            self.addCleanup(guard.stop)
        self.model_guard, self.begin_clip, self.pcm_postcheck, _ = self.stack

    def infer(self):
        return adapter.infer_sense_once_after_approval(self.runtime, self.bound, "a" * 64, self.root, self.state)

    def test_valid_relocated_path_verified_before_and_after_single_forward(self):
        row, evidence = self.infer()
        self.assertEqual(row["raw_text"], "fictional transcript")
        self.assertEqual(self.state, {"stage": "complete", "model_inference_returned": True})
        self.assertEqual(len(self.forward_calls), 1)
        self.assertGreaterEqual(self.vocab_calls_at_forward[0], 2)
        self.assertGreater(self.sp.vocab_calls, self.vocab_calls_at_forward[0])
        self.assertEqual(self.sp.decode_calls, [DEFAULT_IDS, DEFAULT_IDS])
        self.assertEqual(evidence["token_binding"]["tokenizer"]["sha256"], INVENTED_TOKENIZER_SHA256)
        self.assertFalse(evidence["rich_postprocess_applied"])
        self.assertFalse(evidence["use_itn"])
        self.assertEqual(evidence["token_evidence"]["token_scope"], retained.TOKEN_SCOPE)
        kwargs = self.forward_calls[0]
        self.assertIs(kwargs["data_in"], self.pcm)
        self.assertIs(kwargs["frontend"], self.frontend)
        self.assertEqual({key: value for key, value in kwargs.items() if key not in ("data_in", "frontend", "tokenizer")},
                         {"key": ["clip-000001"], "device": "cpu", "language": "auto", "use_itn": False,
                          "text_norm": "woitn", "fs": 16000, "data_type": "sound",
                          "output_timestamp": False, "ban_emo_unk": False})
        self.begin_clip.assert_called_once_with(self.runtime, self.bound, "a" * 64)
        self.pcm_postcheck.assert_called_once_with(self.pcm, self.bound.descriptor["float32_pcm"])

    def test_missing_wrong_bytes_and_semantics_fail_before_forward(self):
        for kind, code in (("missing", "SENSE_TOKENIZER_MISSING"), ("wrong_hash", "SENSE_TOKENIZER_BYTES_MISMATCH"),
                           ("vocabulary", "SENSE_TOKENIZER_SEMANTICS_MISMATCH")):
            with self.subTest(kind=kind):
                self.path.write_bytes(INVENTED_TOKENIZER_BYTES)
                self.sp.vocabulary_size = 25055
                if kind == "missing":
                    self.path.unlink()
                elif kind == "wrong_hash":
                    self.path.write_bytes(b"mutated invented bytes")
                else:
                    self.sp.vocabulary_size = 7
                with self.assertRaises(ValueError) as caught:
                    self.infer()
                self.assertEqual(self.forward_calls, [])
                self.assertEqual(self.state, {"stage": "tokenizer_verification", "model_inference_returned": False})
                self.assertEqual(adapter.failure_details(caught.exception, self.state)["code"], code)
        self.pcm_postcheck.assert_not_called()

    def test_tokenizer_bytes_changed_after_forward_fail_in_real_binding(self):
        self.after_forward = lambda: self.path.write_bytes(b"post-forward invented mutation")
        with self.assertRaisesRegex(ValueError, "Pinned SenseVoice tokenizer bytes changed") as caught:
            self.infer()
        self.assertEqual(len(self.forward_calls), 1)
        self.assertEqual(self.state, {"stage": "token_binding", "model_inference_returned": True})
        self.assertEqual(adapter.failure_details(caught.exception, self.state)["code"], "SENSE_TOKENIZER_BYTES_MISMATCH")
        self.pcm_postcheck.assert_not_called()

    def test_forward_exception_identity_preserved_and_not_called_complete(self):
        failure = RuntimeError("invented private-looking diagnostic")
        self.forward_error = failure
        with self.assertRaises(RuntimeError) as caught:
            self.infer()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.state, {"stage": "model_inference", "model_inference_returned": False})
        self.assertEqual(len(self.forward_calls), 1)
        self.pcm_postcheck.assert_not_called()

    def test_invalid_output_contract_and_association_are_post_forward(self):
        for output, code in (([], "SENSE_RESULT_SHAPE_MISMATCH"),
                             (([{"key": "wrong", "text": "fictional"}], {}), "SENSE_RESULT_ASSOCIATION_MISMATCH"),
                             (([{"key": self.bound.opaque_id, "text": "fictional", "extra": True}], {}),
                              "SENSE_RESULT_ASSOCIATION_MISMATCH")):
            with self.subTest(code=code):
                self.output_override = output
                with self.assertRaises(RuntimeError) as caught:
                    self.infer()
                self.assertEqual(self.state, {"stage": "result_contract", "model_inference_returned": True})
                self.assertEqual(adapter.failure_details(caught.exception, self.state)["code"], code)
        self.pcm_postcheck.assert_not_called()

    def test_missing_token_capture_is_post_forward_and_pre_binding(self):
        self.skip_capture = True
        with self.assertRaisesRegex(ValueError, "Raw token capture does not match model output"):
            self.infer()
        self.assertEqual(self.state, {"stage": "token_capture", "model_inference_returned": True})
        self.pcm_postcheck.assert_not_called()

    def test_pcm_postcheck_failure_preserves_post_forward_stage(self):
        failure = RuntimeError("invented PCM mismatch")
        self.pcm_postcheck.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            self.infer()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.state, {"stage": "pcm_postcheck", "model_inference_returned": True})

    def test_wrong_runtime_stops_before_guards_or_forward(self):
        self.runtime.model_id = "invented wrong model"
        with self.assertRaisesRegex(RuntimeError, "Wrong runtime") as caught:
            self.infer()
        self.assertEqual(self.state, {"stage": "model_verification", "model_inference_returned": False})
        self.assertEqual(adapter.failure_details(caught.exception, self.state)["code"], "SENSE_WRONG_RUNTIME")
        self.model_guard.assert_not_called()
        self.begin_clip.assert_not_called()
        self.assertEqual(self.forward_calls, [])


class FailureDetailsTests(unittest.TestCase):
    def test_fixed_codes_and_stage_without_raw_exception_text(self):
        message = "Pinned SenseVoice tokenizer bytes changed"
        result = adapter.failure_details(ValueError(message), {"stage": "token_binding", "model_inference_returned": True})
        self.assertEqual(result["code"], "SENSE_TOKENIZER_BYTES_MISMATCH")
        self.assertEqual(result["stage"], "token_binding")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertTrue(result["model_inference_returned"])
        self.assertEqual(result["message"], {"representation": "utf8_rendered_exception_prefix_not_native_stderr",
                                           "characters": len(message), "prefix_limit_characters": 4096,
                                           "prefix_bytes": len(message.encode()),
                                           "prefix_sha256": hashlib.sha256(message.encode()).hexdigest(),
                                           "prefix_truncated": False, "raw_text_omitted": True})
        self.assertNotIn(message, repr(result))

    def test_prefix_digest_is_bounded_and_unicode_safe(self):
        message = "invented sensitive-looking diagnostic " + "\u4e2d" * 5000 + "\ud800"
        result = adapter.failure_details(RuntimeError(message), {"stage": "invented invalid stage", "model_inference_returned": 1})
        self.assertEqual(result["code"], "UNCLASSIFIED")
        self.assertEqual(result["stage"], "adapter_dispatch")
        self.assertIsNone(result["model_inference_returned"])
        self.assertEqual(result["message"]["characters"], len(message))
        self.assertTrue(result["message"]["prefix_truncated"])
        prefix = message[:4096].encode("utf-8", "backslashreplace")
        self.assertEqual(result["message"]["prefix_bytes"], len(prefix))
        self.assertEqual(result["message"]["prefix_sha256"], hashlib.sha256(prefix).hexdigest())
        self.assertNotIn("sensitive-looking", repr(result))
        lone_surrogate = adapter.failure_details(ValueError("\ud800"), {})
        self.assertEqual(lone_surrogate["message"]["prefix_sha256"], hashlib.sha256(b"\\ud800").hexdigest())

    def test_unrecognized_or_unrenderable_exception_does_not_leak_type_or_text(self):
        class InventedPrivateError(Exception):
            def __str__(self):
                raise RuntimeError("invented rendering failure")
        result = adapter.failure_details(InventedPrivateError(), {})
        self.assertEqual(result["exception_type"], "Exception")
        self.assertEqual(result["code"], "UNCLASSIFIED")
        self.assertEqual(result["stage"], "adapter_dispatch")
        self.assertEqual(result["message"]["prefix_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(result["message"]["characters"], 0)


if __name__ == "__main__":
    unittest.main()
