"""Source-only regression tests: no downloads, Docker, model or audio inference."""
from pathlib import Path
import copy
import json
import math
import os
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import input_adapter as inputs
import host


class ContractTests(unittest.TestCase):
    def test_fixed_request_and_source_roles(self):
        value, digest = inputs.frozen_request(HERE / "request.json")
        self.assertEqual(value["max_asr_calls"], 192)
        self.assertEqual(value["preset_split"], {"Vivian": "train", "Dylan": "train", "Serena": "dev", "Uncle_Fu": "dev"})
        self.assertEqual(value["nearphone_quarantined_rows"], 48)
        self.assertEqual(value["ordinary_natural_negative_rows"], 0)
        self.assertFalse(value["optimizer_loss_authorized"])
        self.assertFalse(value["training_admitted"])
        self.assertFalse(value["human_gold"])
        self.assertFalse(value["natural_negative_qualified"])
        self.assertEqual(len(digest), 64)

    def test_request_format_cannot_create_second_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.json", Path(directory) / "second.json"
            first.write_bytes(inputs.canonical(inputs.request_value()))
            second.write_text(json.dumps(inputs.request_value(), indent=4))
            _, left = inputs.frozen_request(first)
            _, right = inputs.frozen_request(second)
            self.assertNotEqual(left, right)
            self.assertEqual(host.claim_ref(left), host.claim_ref(right))

    def test_changed_budget_or_permissions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            for name, value in (("max_asr_calls", 193), ("retry", True), ("training_admitted", True),
                                ("source_artifact_id", 1), ("optimizer_loss_authorized", True)):
                with self.subTest(name=name):
                    changed = inputs.request_value()
                    changed[name] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        inputs.frozen_request(path)

    def test_balanced_original_order_has_no_cell_reuse(self):
        self.assertEqual(sorted(inputs.ORDER), list(range(64)))
        for batch in range(4):
            selected = inputs.ORDER[batch * 16:(batch + 1) * 16]
            self.assertEqual([sum(n // 16 == lane for n in selected) for lane in range(4)], [4] * 4)
        self.assertEqual([sum(inputs.SPLITS[inputs.PRESETS[n // 16]] == role for n in inputs.ORDER)
                          for role in ("train", "dev")], [32, 32])

    def test_float_wav_rejects_corruption_and_clipping(self):
        def wav(values, rate=24000):
            pcm = struct.pack("<" + "f" * len(values), *values)
            fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
            chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
            return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
        self.assertEqual(inputs.float_wav(wav([0.1, -0.2]))[1], 2)
        for raw in (wav([math.nan]), wav([math.inf]), wav([1.0]), wav([0.1], rate=16000), wav([0.1])[:-1]):
            with self.assertRaises(ValueError):
                inputs.float_wav(raw)

    def test_json_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            inputs.parse(b'{"head":"first","head":"second"}')

    def test_execution_is_main_attempt_one_without_write_token(self):
        head = "a" * 40
        env = {"GITHUB_REPOSITORY": "jiying2007/kws-pipeline", "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_NUMBER": "37",
            "GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64", "GITHUB_SHA": head, "GITHUB_RUN_ID": "123",
            "GITHUB_WORKFLOW_SHA": head,
            "GITHUB_WORKFLOW_REF": "jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main"}
        event = {"after": head, "ref": "refs/heads/main", "deleted": False,
                 "repository": {"private": False, "full_name": "jiying2007/kws-pipeline"}}
        claim = {"ref": host.claim_ref("b" * 64), "object": {"sha": head, "type": "commit"}}
        self.assertEqual(host.validate_execution(env, event, claim, "b" * 64), (head, "123"))
        for key, value in (("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_RUN_ATTEMPT", "2"),
                           ("GH_TOKEN", "never-pass-this"), ("GITHUB_REF", "refs/heads/research/old"),
                           ("GITHUB_WORKFLOW_SHA", "c" * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                host.validate_execution(dict(env, **{key: value}), event, claim, "b" * 64)
        with self.assertRaises(ValueError):
            host.validate_execution(env, event, dict(claim, object={"sha": "c" * 40, "type": "commit"}), "b" * 64)

    def test_cleanup_must_be_confirmed_before_continuation(self):
        closed = {"created_confirmed": True, "operations": {"cleanup": {"status": "complete"}}}
        self.assertTrue(host.lifecycle_closed(closed))
        self.assertFalse(host.lifecycle_closed({}))
        self.assertFalse(host.lifecycle_closed({"created_confirmed": False, "operations": closed["operations"]}))
        self.assertFalse(host.lifecycle_closed({"created_confirmed": True, "operations": {"cleanup": {"status": "failed_no_retry"}}}))

    def test_initial_terminal_contains_every_unrun_slot(self):
        states = host.initial_states()
        self.assertEqual(set(states), {"qwen06", "sensevoice", "whisper"})
        self.assertEqual(sum(len(rows) for rows in states.values()), 192)
        self.assertTrue(all(row["status"] == "not_run" and row["actual_text"] is None
                            and row["attempt_consumed"] is False for rows in states.values() for row in rows))


class NativeReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("KWS_PUBLIC64_RETAINED")
        if not root:
            raise unittest.SkipTest("set KWS_PUBLIC64_RETAINED to the exact source-only retained checkout")
        cls.root = Path(root)
        host.verify_sources(cls.root)
        cls.readers = host.native_readers(cls.root)

    def qwen(self):
        tokens = [15, 151645]
        decoded = "language Chinese<asr_text>测试"
        expected, metadata = self.readers["qwen06"](decoded, tokens)
        row = {"audio_id": "clip-000001", "wav_sha256": "a" * 64, "pcm_sha256": "b" * 64, "frames": 16000}
        evidence = {"token_ids": tokens, "decoded_skip_special_tokens": decoded, "metadata": metadata,
            "prompt_context": "", "force_language": None, "transcript_repair_applied": False,
            "input_binding": {"wav": {"sha256": row["wav_sha256"]},
                "raw_pcm16": {"sha256": row["pcm_sha256"], "frame_count": row["frames"]},
                "operations": {"mix": False, "resample": False, "trim": False}}}
        return row, {"actual_text": expected["raw_text"], "original_native_decoder_evidence": evidence}, {"termination": {"generated_token_ids": tokens}}

    def test_qwen_raw_text_and_tokens_must_bind(self):
        row, observation, capture = self.qwen()
        identity = {"native_decode_config": {}}
        host.bind_native_text("qwen06", identity, observation, capture, row, Path("unused"), self.readers)
        changed = copy.deepcopy(observation)
        changed["actual_text"] = "替换原文"
        with self.assertRaises(ValueError):
            host.bind_native_text("qwen06", identity, changed, capture, row, Path("unused"), self.readers)
        changed = copy.deepcopy(capture)
        changed["termination"]["generated_token_ids"] = [20, 151645]
        with self.assertRaises(ValueError):
            host.bind_native_text("qwen06", identity, observation, changed, row, Path("unused"), self.readers)

    def test_qwen_non_eos_cannot_be_promoted(self):
        row, observation, capture = self.qwen()
        evidence = observation["original_native_decoder_evidence"]
        evidence["token_ids"] = [15, 16]
        _, evidence["metadata"] = self.readers["qwen06"](evidence["decoded_skip_special_tokens"], evidence["token_ids"])
        capture["termination"]["generated_token_ids"] = evidence["token_ids"]
        with self.assertRaises(ValueError):
            host.bind_native_text("qwen06", {"native_decode_config": {}}, observation, capture, row, Path("unused"), self.readers)

    def test_sense_transcript_and_lexical_unknown_boundary(self):
        row = {"audio_id": "clip-000001", "wav_sha256": "a" * 64, "pcm_sha256": "b" * 64, "frames": 16000}
        decoded = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>测试"
        raw_ids, lexical = [25000, 25001, 25002, 25003, 15], [15]
        expected = self.readers["sensevoice"](decoded, lexical)
        config = {"effective_config": {"parser_contract_sha256": "c" * 64, "tokenizer_sha256": "d" * 64}}
        binding = {"raw_token_ids": raw_ids, "metadata_token_ids": raw_ids[:4], "lexical_token_ids": lexical,
            "raw_decoded_sha256": inputs.sha(decoded.encode()), "metadata_boundary_verified": True,
            "raw_redecode_matches_capture": True, "lexical_unknown_present": False,
            "parser_contract_sha256": "c" * 64, "tokenizer": {"sha256": "d" * 64}}
        evidence = {"raw_result": [[{"key": row["audio_id"], "text": decoded}], {}],
            "token_evidence": {"decoded": decoded, "token_ids": raw_ids}, "token_binding": binding,
            "use_itn": False, "rich_postprocess_applied": False, "metadata": expected["metadata"],
            "input_binding": {"wav": {"sha256": row["wav_sha256"]}, "raw_pcm16": {"sha256": row["pcm_sha256"], "frame_count": row["frames"]},
                              "operations": {"mix": False, "resample": False, "trim": False}}}
        observation = {"actual_text": expected["raw_text"], "original_native_decoder_evidence": evidence}
        capture = {"termination": {key: binding[key] for key in ("raw_token_ids", "metadata_token_ids", "lexical_token_ids")}}
        host.bind_native_text("sensevoice", {"native_decode_config": config}, observation, capture, row, Path("unused"), self.readers)
        for key, value in (("actual_text", "不同原文"),):
            changed = copy.deepcopy(observation); changed[key] = value
            with self.assertRaises(ValueError):
                host.bind_native_text("sensevoice", {"native_decode_config": config}, changed, capture, row, Path("unused"), self.readers)
        evidence["token_binding"]["lexical_unknown_present"] = True
        with self.assertRaises(ValueError):
            host.bind_native_text("sensevoice", {"native_decode_config": config}, observation, capture, row, Path("unused"), self.readers)

    def test_whisper_requires_real_pre_finalize_eot_trace(self):
        effective = {"initial_token_ids": [10], "eot_token_id": 99, "n_text_ctx": 448, "effective_sample_len": 224}
        trace = [{"step": 1, "token_ids": [10, 20], "completed": False},
                 {"step": 2, "token_ids": [10, 20, 99], "completed": True}]
        termination = self.readers["whisper"](trace, effective)
        self.assertEqual(termination["generated_token_ids"], [20, 99])
        changed = copy.deepcopy(trace); changed[-1]["completed"] = False
        with self.assertRaises(ValueError):
            self.readers["whisper"](changed, effective)
        with self.assertRaises(ValueError):
            self.readers["whisper"](trace[:1], effective)

    def test_pure_readers_never_import_frameworks(self):
        self.assertFalse({"torch", "numpy", "scipy", "funasr", "qwen_asr", "whisper"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
