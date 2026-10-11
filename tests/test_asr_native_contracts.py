"""Pure-stdlib invented metadata regressions; never run an ASR or read audio."""
import ast
import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from asr_native_contracts import (ADAPTER_FAMILIES, QWEN_ADAPTER, SENSE_ADAPTER,
    WHISPER_ADAPTER, canonical_sha256, validate_native_capture, validate_native_config)
from asr_native_fixtures import native_fixture, rebind_native_fixture


class NativeASRContractTests(unittest.TestCase):
    def test_all_native_families_have_normal_nonmutating_metadata_path(self):
        for adapter in ADAPTER_FAMILIES:
            with self.subTest(adapter=adapter):
                config, capture = native_fixture(adapter)
                before = copy.deepcopy((config, capture))
                self.assertIsNone(validate_native_config(config))
                self.assertIsNone(validate_native_capture(config, capture))
                self.assertEqual((config, capture), before)
                self.assertEqual(capture["acoustic_verbatim_completeness"], "unknown")

    def test_no_generic_six_parameter_coercion(self):
        for adapter in (QWEN_ADAPTER, SENSE_ADAPTER):
            for key, value in (("language", "zh"), ("task", "transcribe"), ("beam_size", 1),
                               ("temperature", 0.0), ("prompt", ""), ("hotwords", [])):
                with self.subTest(adapter=adapter, key=key):
                    config, _ = native_fixture(adapter)
                    config["native_request"][key] = value
                    with self.assertRaises(ValueError):
                        validate_native_config(config)

    def test_null_missing_unknown_and_unsupported_are_distinct(self):
        for adapter, key in ((QWEN_ADAPTER, "force_language"), (WHISPER_ADAPTER, "beam_size")):
            config, capture = native_fixture(adapter)
            self.assertIsNone(config["native_request"][key])
            validate_native_capture(config, capture)
            for replacement in ("unknown", "not_applicable", 1, False, []):
                altered = copy.deepcopy(config)
                altered["native_request"][key] = replacement
                with self.subTest(adapter=adapter, value=replacement), self.assertRaises(ValueError):
                    validate_native_config(altered)
            del config["native_request"][key]
            with self.assertRaises(ValueError):
                validate_native_config(config)
        config, _ = native_fixture(SENSE_ADAPTER)
        config["native_request"]["eos_token_id"] = None
        with self.assertRaises(ValueError):
            validate_native_config(config)

    def test_no_target_prompt_peer_output_or_postcorrection(self):
        changes = ((QWEN_ADAPTER, "context", "你好小窝"),
                   (QWEN_ADAPTER, "force_language", "Chinese"),
                   (SENSE_ADAPTER, "use_itn", True),
                   (WHISPER_ADAPTER, "prompt", "你好小窝"),
                   (WHISPER_ADAPTER, "prefix", "你好"))
        for adapter, key, value in changes:
            config, _ = native_fixture(adapter)
            config["native_request"][key] = value
            with self.subTest(adapter=adapter, key=key), self.assertRaises(ValueError):
                validate_native_config(config)
        for adapter in ADAPTER_FAMILIES:
            for key in ("transcript_postcorrection_applied", "target_context_received", "peer_transcript_received"):
                config, capture = native_fixture(adapter)
                capture[key] = True
                with self.subTest(adapter=adapter, key=key), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)

    def test_bool_cannot_masquerade_as_native_integer_or_reverse(self):
        for adapter in ADAPTER_FAMILIES:
            mutations = (("adapter_version",), ("execution", "batch_size"), ("input_policy", "channels"))
            for keys in mutations:
                config, _ = native_fixture(adapter)
                target = config
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = True
                with self.subTest(adapter=adapter, keys=keys), self.assertRaises(ValueError):
                    validate_native_config(config)
            for key in ("trimmed_frames", "processed_start_frame", "channels", "audio_frames"):
                config, capture = native_fixture(adapter)
                capture["input"][key] = False if key != "channels" else True
                with self.subTest(adapter=adapter, key=key), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)
            for key in ("target_context_received", "peer_transcript_received"):
                config, capture = native_fixture(adapter)
                capture[key] = 0
                with self.subTest(adapter=adapter, key=key), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)

    def test_float_int_bool_and_nonfinite_are_not_interchangeable(self):
        for temperature in (False, 0, float("nan"), float("inf"), -float("inf")):
            config, _ = native_fixture(WHISPER_ADAPTER)
            config["native_request"]["temperature"] = temperature
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                validate_native_config(config)
        config, _ = native_fixture(QWEN_ADAPTER)
        config["native_request"]["do_sample"] = 0
        with self.assertRaises(ValueError):
            validate_native_config(config)

    def test_unknown_keys_rejected_at_every_contract_boundary(self):
        for adapter in ADAPTER_FAMILIES:
            for path in ((), ("native_request",), ("execution",), ("input_policy",), ("effective_config",)):
                config, _ = native_fixture(adapter)
                target = config
                for key in path:
                    target = target[key]
                target["unreviewed_override"] = None
                with self.subTest(adapter=adapter, path=path), self.assertRaises(ValueError):
                    validate_native_config(config)
            for path in ((), ("input",), ("termination",), ("effective_config",)):
                config, capture = native_fixture(adapter)
                target = capture
                for key in path:
                    target = target[key]
                target["unreviewed_override"] = None
                with self.subTest(adapter=adapter, path=path), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)

    def test_config_version_family_and_lock_binding(self):
        for key, value in (("schema_version", 1), ("adapter_version", 2), ("adapter_id", "generic-asr"),
                           ("model_family", "invented-other-family"), ("source_lock_sha256", "unknown"),
                           ("asset_lock_sha256", None)):
            config, _ = native_fixture(QWEN_ADAPTER)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_config(config)
        for key in ("source_lock_sha256", "asset_lock_sha256", "native_config_sha256"):
            config, capture = native_fixture(QWEN_ADAPTER)
            capture[key] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_capture(config, capture)

    def test_effective_config_drift_is_rejected_even_when_rehashed(self):
        for adapter in ADAPTER_FAMILIES:
            config, capture = native_fixture(adapter)
            capture["effective_config"]["execution"]["device"] = "cuda"
            capture["effective_config_sha256"] = canonical_sha256(capture["effective_config"])
            with self.subTest(adapter=adapter), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
            config, capture = native_fixture(adapter)
            config["effective_config_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                validate_native_config(config)

    def test_qwen_inherited_constraints_cannot_force_termination_or_targets(self):
        for key, value in (("forced_eos_token_id", 151645), ("forced_bos_token_id", 17),
                           ("bad_words_ids", [[1]]), ("sequence_bias", {"1": 9.0}),
                           ("stop_strings", ["小窝"]), ("custom_logits_processors", ["target"]),
                           ("custom_stopping_criteria", ["substring"]), ("num_beams", True)):
            config, capture = native_fixture(QWEN_ADAPTER)
            config["effective_config"][key] = value
            rebind_native_fixture(config, capture)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_capture(config, capture)

    def test_complete_input_and_decoder_completion_do_not_claim_acoustics(self):
        for adapter in ADAPTER_FAMILIES:
            for claim in (True, False, "complete", "verified", None):
                config, capture = native_fixture(adapter)
                capture["acoustic_verbatim_completeness"] = claim
                with self.subTest(adapter=adapter, claim=claim), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)

    def test_partial_source_crop_unknown_and_bad_padding_fail_closed(self):
        for adapter in ADAPTER_FAMILIES:
            for key, value in (("processed_start_frame", 1), ("processed_end_frame", 159),
                               ("trimmed_frames", 1), ("audio_frames", 0),
                               ("audio_frames", None), ("sample_rate", 48000), ("channels", 2)):
                config, capture = native_fixture(adapter)
                capture["input"][key] = value
                with self.subTest(adapter=adapter, key=key), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)
            config, capture = native_fixture(adapter)
            capture["input"]["padding_frames"] += 1
            with self.assertRaises(ValueError):
                validate_native_capture(config, capture)

    def test_whisper_checks_unpadded_length_before_accepting_pad_only_claim(self):
        config, capture = native_fixture(WHISPER_ADAPTER, 480000)
        config["input_policy"]["max_audio_frames"] = 480000
        rebind_native_fixture(config, capture)
        validate_native_capture(config, capture)
        capture["input"]["audio_frames"] = 480001
        capture["input"]["trimmed_frames"] = 1
        with self.assertRaises(ValueError):
            validate_native_capture(config, capture)
        config["input_policy"]["max_audio_frames"] = 480001
        with self.assertRaises(ValueError):
            validate_native_config(config)

    def test_qwen_requires_actual_terminal_eos_and_bounded_count(self):
        for ids in ([1, 2, 3], [151645, 1], [1, 151643, 151645], [1, False, 151645],
                    [1, 151645.0], [], [1] * 256 + [151645]):
            config, capture = native_fixture(QWEN_ADAPTER)
            capture["termination"]["generated_token_ids"] = ids
            capture["termination"]["generated_tokens"] = len(ids)
            with self.subTest(ids=ids[:4]), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        config, capture = native_fixture(QWEN_ADAPTER)
        capture["termination"].update(generated_token_ids=[1] * 255 + [151643], generated_tokens=256)
        validate_native_capture(config, capture)

    def test_missing_unknown_or_truncated_termination_never_passes(self):
        for adapter in ADAPTER_FAMILIES:
            config, capture = native_fixture(adapter)
            del capture["termination"]
            with self.subTest(adapter=adapter), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
            for key, value in (("truncated", True), ("truncated", None)):
                config, capture = native_fixture(adapter)
                capture["termination"][key] = value
                with self.assertRaises(ValueError):
                    validate_native_capture(config, capture)
        for adapter in (QWEN_ADAPTER, WHISPER_ADAPTER):
            for reason in ("unknown", "sample_limit", "context_limit", None):
                config, capture = native_fixture(adapter)
                capture["termination"]["stop_reason"] = reason
                with self.subTest(adapter=adapter, reason=reason), self.assertRaises(ValueError):
                    validate_native_capture(config, capture)

    def test_sense_requires_real_ctc_scope_and_token_boundaries_not_eos(self):
        for key, value in (("inference_returned", False), ("inference_returned", 1),
                           ("token_scope", "framewise_logits"), ("metadata_boundary_verified", False),
                           ("raw_redecode_matches_capture", False), ("lexical_token_ids", None),
                           ("lexical_token_ids", [17]), ("metadata_token_ids", [25001])):
            config, capture = native_fixture(SENSE_ADAPTER)
            capture["termination"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        for ids in ([0, 18], [25055, 18]):
            config, capture = native_fixture(SENSE_ADAPTER)
            capture["termination"]["raw_token_ids"][4:] = ids
            capture["termination"]["lexical_token_ids"] = ids
            with self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        config, capture = native_fixture(SENSE_ADAPTER)
        capture["termination"]["eos_observed"] = True
        with self.assertRaises(ValueError):
            validate_native_capture(config, capture)

    def test_whisper_effective_sample_limit_must_be_observed_and_correct(self):
        for value in (None, "unknown", 225, False):
            config, capture = native_fixture(WHISPER_ADAPTER)
            config["effective_config"]["effective_sample_len"] = value
            rebind_native_fixture(config, capture)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_native_capture(config, capture)

    def test_whisper_recomputes_native_suppression_no_extra_lexical_filter(self):
        for field, value in (("resolved_suppress_token_ids", [17, 101, 102, 103, 104, 105, 106, 200, 201]),
                             ("resolved_suppress_token_ids", [101, 102, 103, 104, 105, 106, 200]),
                             ("initial_blank_suppress_token_ids", [100]),
                             ("non_speech_token_ids", [200, 200]), ("space_token_ids", [])):
            config, capture = native_fixture(WHISPER_ADAPTER)
            config["effective_config"][field] = value
            rebind_native_fixture(config, capture)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        config, capture = native_fixture(WHISPER_ADAPTER)
        config["effective_config"]["control_token_ids"]["no_speech"] = None
        config["effective_config"]["resolved_suppress_token_ids"].remove(106)
        rebind_native_fixture(config, capture)
        validate_native_capture(config, capture)

    def test_whisper_public_result_and_finalize_eot_are_not_a_trace(self):
        for stage in ("DecodingResult", "after_finalize", "returned_tokens", None):
            config, capture = native_fixture(WHISPER_ADAPTER)
            capture["termination"]["capture_stage"] = stage
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        config, capture = native_fixture(WHISPER_ADAPTER)
        capture["termination"] = {"tokens": [17, 18], "text": "你好小窝", "eot": 100}
        with self.assertRaises(ValueError):
            validate_native_capture(config, capture)

    def test_whisper_trace_rejects_missing_eot_early_eot_false_counts_and_suppressed_ids(self):
        for ids in ([17, 18, 19], [100], [17, 100, 18, 100], [20, 17, 100],
                    [17, 200, 100], [False, 17, 100], [17] * 224 + [100]):
            config, capture = native_fixture(WHISPER_ADAPTER)
            capture["termination"].update(generated_token_ids=ids, step_count=len(ids),
                                           pre_finalize_token_count=4 + len(ids))
            with self.subTest(ids=ids[:5]), self.assertRaises(ValueError):
                validate_native_capture(config, capture)
        for key, value in (("step_count", 4), ("pre_finalize_token_count", 8),
                           ("completed", 1), ("initial_token_ids", [103, 17, 101, 108])):
            config, capture = native_fixture(WHISPER_ADAPTER)
            capture["termination"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_capture(config, capture)

    def test_whisper_natural_eot_on_last_allowed_step_is_not_falsely_truncated(self):
        config, capture = native_fixture(WHISPER_ADAPTER)
        capture["termination"].update(generated_token_ids=[17] * 223 + [100],
            step_count=224, pre_finalize_token_count=228)
        validate_native_capture(config, capture)

    def test_module_has_no_model_network_process_or_external_imports(self):
        tree = ast.parse((ROOT / "tools/asr_native_contracts.py").read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertEqual(sorted(imports), ["__future__", "hashlib", "json"])

    def test_public_candidate_sources_are_unactivated_not_download_or_asr_evidence(self):
        sources = json.loads((ROOT / "research/asr-native-v2/whisper-candidate-sources.json").read_text())
        self.assertIs(sources["activated"], False)
        self.assertEqual(sources["family_vote_limit"], 1)
        self.assertEqual(sources["source_license"], "MIT")
        self.assertEqual(sources["native_checkpoint_license"], "MIT")
        self.assertEqual(sources["source_revision"], "86098128c0b4f24f0e2aa2994de830614b474227")
        for candidate in sources["candidates"]:
            self.assertIs(candidate["checkpoint_downloaded"], False)
            self.assertIs(candidate["checkpoint_bytes_verified"], False)
            self.assertRegex(candidate["expected_checkpoint_sha256"], r"^[0-9a-f]{64}$")
        for key in ("cpu_latency_status", "cpu_ram_status", "keyword_accuracy_status"):
            self.assertEqual(sources[key], "not_measured")


if __name__ == "__main__":
    unittest.main()
