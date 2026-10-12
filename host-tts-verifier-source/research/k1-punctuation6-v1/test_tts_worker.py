"""Offline stdlib tests. Fake tensors/numerics do not exercise inference or SciPy.

Run with python -B, python -B -O, and python -B -OO. Temporary fixtures and
reports live outside the frozen source tree, under phase2-tts/test-results.
"""
import ast
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import wave

import contract
import tts_common as common
import tts_gate as gate
import tts_worker as worker

RESULTS = Path(__file__).resolve().parents[3] / "test-results"
RESULTS.mkdir(exist_ok=True)


def eos_record(n=3):
    return dict(capture_calls=1, iterations=n, last_token_ids=[2150], eos=2150,
                ended_with_eos=True, hit_token_cap_without_eos=False,
                sequence_length=n, earlier_eos_seen=False)


def riff(chunks):
    body = b"WAVE" + b"".join(tag + struct.pack("<I", len(payload)) + payload +
                               (b"\0" if len(payload) % 2 else b"") for tag, payload in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def float_wav(values):
    samples = struct.pack("<%df" % len(values), *values)
    return riff([(b"fmt ", struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)),
                 (b"fact", struct.pack("<I", len(values))), (b"data", samples)])


def pcm_wav(frames=2560, value=7):
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", ""))
        handle.writeframes(struct.pack("<h", value) * frames)
    return out.getvalue()


def pcm_wav_samples(values):
    """Exact synthetic int16 samples; this helper does not model quantization."""
    out=io.BytesIO()
    with wave.open(out,"wb") as handle:
        handle.setparams((1,2,16000,0,"NONE",""))
        handle.writeframes(struct.pack("<%dh" % len(values),*values))
    return out.getvalue()


class Tensor:
    """Only the real shape/index/detach/cpu/tolist interface being intercepted."""
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (len(values), len(values[0]) if values else 0)
        self.ndim = len(self.shape)

    def __getitem__(self, index):
        if index != (slice(None), -1):
            raise IndexError(index)
        return Tensor([row[-1] for row in self.values], (len(self.values),))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return deepcopy(self.values)


def model_fixture(tokens=None, hidden=None, shape=None):
    tokens = [10, 11, 2150] if tokens is None else tokens
    config = NS(codec_eos_token_id=2150, num_code_groups=16, vocab_size=3072,
                spk_id={"dylan": 2878, "uncle_fu": 3010, "vivian": 3065},
                spk_is_dialect={"dylan": "beijing_dialect", "uncle_fu": False, "vivian": False},
                codec_language_id={"chinese": 2055, "beijing_dialect": 2074})
    def owner():
        attention = NS(q_proj=1, k_proj=1, v_proj=1, config=NS(_attn_implementation="eager"))
        p = NS(device="cpu", dtype="torch.float32")
        return NS(training=False, parameters=lambda: iter([p]), buffers=lambda: iter([p]),
                  named_modules=lambda: iter([("attention", attention)]))
    model = owner()
    seq = Tensor([tokens], shape)
    result = NS(sequences=seq, hidden_states=[None] * (len(tokens) if hidden is None else hidden))
    original = mock.Mock(return_value=result)
    model.config = NS(talker_config=config)
    model.tts_model_type = "custom_voice"
    model.tts_model_size = "0b6"
    model.tokenizer_type = "qwen3_tts_tokenizer_12hz"
    model.speech_tokenizer = NS(model=owner())
    model.talker = NS(generate=original, generation_config=NS(num_beams=1,
        num_return_sequences=1, forced_eos_token_id=None, forced_bos_token_id=None,
        stop_strings=None, max_time=None))
    return NS(model=model), original, result


def generation_kwargs(wrapper):
    kwargs = {k: v for k, v in contract.expected_plan()["tts"]["recipe"].items()
              if k not in ("language", "non_streaming_mode")}
    kwargs.update(min_new_tokens=2, eos_token_id=2150, output_hidden_states=True,
                  return_dict_in_generate=True, inputs_embeds=Tensor([], (1, 9, 1024)),
                  attention_mask=Tensor([], (1, 9)), trailing_text_hidden=Tensor([], (1, 2, 1024)),
                  tts_pad_embed=Tensor([], (1, 1, 1024)))
    kwargs["suppress_tokens"] = [i for i in range(wrapper.model.config.talker_config.vocab_size - 1024,
                                                wrapper.model.config.talker_config.vocab_size) if i != 2150]
    return kwargs


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="k1-offline-", dir=RESULTS)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class CommonTests(TempCase):
    def test_write_once_is_exclusive_canonical_private_and_durable(self):
        path = self.root / "claim.json"
        synced = []
        original = os.fsync
        def sync(fd):
            synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            original(fd)
        with mock.patch.object(common.os, "fsync", side_effect=sync):
            identity = common.write_once(path, {"文本": "你好，小窝", "a": 1})
        self.assertEqual(synced, ["file", "dir"])
        self.assertEqual(path.read_bytes(), contract.canonical({"文本": "你好，小窝", "a": 1}))
        self.assertEqual(identity, {"bytes": path.stat().st_size, "sha256": common.sha(path)})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            common.write_once(path, {"a": 2})

    def test_atomic_json_replacement_fsyncs_file_then_directory(self):
        path = self.root / "state.json"
        common.atomic_json(path, {"first": True})
        calls = []
        original = os.fsync
        def sync(fd):
            calls.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            original(fd)
        with mock.patch.object(common.os, "fsync", side_effect=sync):
            common.atomic_json(path, {"second": True})
        self.assertEqual(calls, ["file", "dir"])
        self.assertEqual(json.loads(path.read_bytes()), {"second": True})
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_symlink_and_hardlink_rejected(self):
        original = self.root / "real"
        original.write_bytes(b"x")
        link = self.root / "link"
        link.symlink_to(original)
        with self.assertRaises(ValueError):
            common.unaliased(link, file=True)
        hard = self.root / "hard"
        os.link(original, hard)
        with self.assertRaises(ValueError):
            common.unaliased(hard, file=True)

    def test_symlink_parent_rejected(self):
        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            common.write_once(alias / "new.json", {})

    def test_child_environment_is_constructed_without_host_data(self):
        poisoned = {"PATH": "/arbitrary/private/path", "HOME": "/private/home",
                    "HF_HOME": "/private/cache", "XDG_CACHE_HOME": "/private/cache2",
                    "GITHUB_TOKEN": "not-a-credential", "GH_TOKEN": "sentinel", "KWS_EVIDENCE_TOKEN": "sentinel",
                    "ACTIONS_RUNTIME_TOKEN": "sentinel", "LD_PRELOAD": "sentinel", "ARBITRARY_SECRET": "sentinel"}
        with mock.patch.dict(os.environ, poisoned, clear=True):
            env = common.child_environment(self.root, self.root / "venv/bin/python")
        common.assert_no_host_secrets(env)
        self.assertEqual(env["HOME"], str(self.root / "home"))
        self.assertEqual(env["PATH"], str(self.root / "venv/bin") + ":/usr/bin:/bin")
        self.assertFalse(set(poisoned).difference({"PATH", "HOME"}) & set(env))
        self.assertNotIn("sentinel", env.values())
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_secret_guard_rejects_each_banned_key(self):
        keys = ["GITHUB_TOKEN", "GH_TOKEN", "KWS_EVIDENCE_TOKEN", "KWS_DATA_TOKEN", "ACTIONS_RUNTIME_TOKEN",
                "ACTIONS_UNKNOWN", "LD_PRELOAD", "PYTHONPATH", "SSH_AUTH_SOCK", "AWS_SECRET_ACCESS_KEY",
                "HF_TOKEN", "PASSWORD", "PRIVATE_SECRET", "NETRC", "GIT_ASKPASS"]
        for key in keys:
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.assert_no_host_secrets({key: ""})
        common.assert_no_host_secrets({"TOKENIZERS_PARALLELISM": "false"})

    def test_isolated_command_uses_isolated_no_bytecode_python(self):
        command = common.isolated_command("/venv/python", "/source/entry.py", ["--x", 3])
        self.assertEqual(command[:4], ["/venv/python", "-I", "-B", "-c"])
        self.assertEqual(command[-3:], ["/source/entry.py", "--x", "3"])

    def test_private_exception_class_names_are_projected(self):
        private_error=type("SecretPathOrTokenWouldLeak",(Exception,),{})
        self.assertEqual(common.safe_exception(private_error()),"OtherException")
        self.assertEqual(common.safe_exception(ValueError()),"ValueError")

    def test_no_inference_library_was_imported(self):
        self.assertFalse({"torch","numpy","soundfile","scipy","qwen_tts","transformers"} & set(sys.modules))

    def test_validators_do_not_use_optimization_sensitive_assert(self):
        for name in ("tts_common.py", "tts_worker.py", "tts_gate.py", "tts_semantics.py"):
            tree = ast.parse(Path(__file__).with_name(name).read_text())
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name)


class CaptureTests(unittest.TestCase):
    def test_valid_singleton_returns_original_before_native_decode(self):
        wrapper, original, native = model_fixture()
        observed = worker.capture_talker(wrapper)
        seen = wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertIs(seen, native)
        self.assertEqual(observed, eos_record())
        original.assert_called_once()

    def test_119_iterations_accepted(self):
        wrapper, original, _ = model_fixture([1] * 118 + [2150])
        seen = worker.capture_talker(wrapper)
        wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertEqual(seen["iterations"], 119)
        original.assert_called_once()

    def test_120_iterations_rejected_even_with_eos(self):
        wrapper, original, _ = model_fixture([1] * 119 + [2150])
        seen = worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertEqual(seen["iterations"], 120)
        original.assert_called_once()

    def test_no_eos_rejected_before_decode(self):
        wrapper, _, _ = model_fixture([1, 2, 3])
        seen = worker.capture_talker(wrapper)
        decoded = []
        with self.assertRaises(ValueError):
            result = wrapper.model.talker.generate(**generation_kwargs(wrapper))
            decoded.append(result)
        self.assertEqual(decoded, [])
        self.assertFalse(seen["ended_with_eos"])

    def test_early_eos_rejected(self):
        wrapper, _, _ = model_fixture([2150, 2, 2150])
        seen = worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertTrue(seen["earlier_eos_seen"])

    def test_empty_sequence_rejected(self):
        wrapper, _, _ = model_fixture([])
        worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))

    def test_empty_batch_rejected(self):
        wrapper, _, native = model_fixture()
        native.sequences = Tensor([], (0, 3))
        worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))

    def test_multiple_batch_rejected(self):
        wrapper, _, native = model_fixture()
        native.sequences = Tensor([[1, 2, 2150], [1, 2, 2150]])
        worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))

    def test_sequence_hidden_count_mismatch_rejected(self):
        wrapper, _, _ = model_fixture(hidden=2)
        worker.capture_talker(wrapper)
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))

    def test_token_bool_or_float_rejected(self):
        for tokens in ([True, 2, 2150], [1.0, 2, 2150]):
            with self.subTest(tokens=tokens):
                wrapper, _, _ = model_fixture(tokens)
                worker.capture_talker(wrapper)
                with self.assertRaises(ValueError):
                    wrapper.model.talker.generate(**generation_kwargs(wrapper))

    def test_second_interception_never_calls_original_again(self):
        wrapper, original, _ = model_fixture()
        worker.capture_talker(wrapper)
        wrapper.model.talker.generate(**generation_kwargs(wrapper))
        with self.assertRaises(ValueError):
            wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertEqual(original.call_count, 1)

    def test_capture_resets_explicitly_for_next_cell(self):
        wrapper, original, _ = model_fixture()
        observed = worker.capture_talker(wrapper)
        wrapper.model.talker.generate(**generation_kwargs(wrapper))
        observed.clear()
        wrapper.model.talker.generate(**generation_kwargs(wrapper))
        self.assertEqual(original.call_count, 2)

    def test_extra_missing_and_changed_kwargs_rejected_before_original(self):
        for kind in ("extra", "missing", "temperature", "max_new_tokens", "forced_min", "suppression", "batch", "empty_embeddings", "positional"):
            with self.subTest(kind=kind):
                wrapper, original, _ = model_fixture()
                worker.capture_talker(wrapper)
                kwargs = generation_kwargs(wrapper)
                args = ()
                if kind == "extra": kwargs["unexpected"] = None
                if kind == "missing": del kwargs["trailing_text_hidden"]
                if kind == "temperature": kwargs["temperature"] = 0.7
                if kind == "max_new_tokens": kwargs["max_new_tokens"] = 121
                if kind == "forced_min": kwargs["min_new_tokens"] = 1
                if kind == "suppression": kwargs["suppress_tokens"] = []
                if kind == "batch": kwargs["inputs_embeds"] = Tensor([], (2, 3, 1024))
                if kind == "empty_embeddings": kwargs["inputs_embeds"] = Tensor([], (1, 0, 1024))
                if kind == "positional": args = ("forbidden",)
                with self.assertRaises(ValueError): wrapper.model.talker.generate(*args, **kwargs)
                original.assert_not_called()


class ModelTests(unittest.TestCase):
    def test_exact_model_accepted(self):
        wrapper, _, _ = model_fixture()
        worker.validate_model(wrapper)

    def test_identity_conditioning_and_generation_mutations_rejected(self):
        mutations = [lambda m: setattr(m, "tts_model_type", "voice_design"),
                     lambda m: setattr(m, "tts_model_size", "1b7"),
                     lambda m: setattr(m, "tokenizer_type", "other"),
                     lambda m: setattr(m.config.talker_config, "codec_eos_token_id", 99),
                     lambda m: setattr(m.config.talker_config, "num_code_groups", 8),
                     lambda m: m.config.talker_config.spk_id.update(dylan=2879),
                     lambda m: m.config.talker_config.spk_is_dialect.update(dylan=False),
                     lambda m: m.config.talker_config.codec_language_id.update(chinese=2054)]
        for key, value in (("num_beams", 2), ("num_return_sequences", 2), ("forced_eos_token_id", 2150),
                           ("forced_bos_token_id", 7), ("stop_strings", ["stop"]), ("max_time", 1)):
            mutations.append(lambda m, k=key, v=value: setattr(m.talker.generation_config, k, v))
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                wrapper, _, _ = model_fixture()
                mutate(wrapper.model)
                with self.assertRaises(ValueError): worker.validate_model(wrapper)

    def test_both_models_require_cpu_float32_eval_and_eager(self):
        for target in ("tts", "codec"):
            for kind in ("training", "empty", "cuda", "half", "buffer", "attention", "flash"):
                with self.subTest(target=target, kind=kind):
                    wrapper, _, _ = model_fixture()
                    owner = wrapper.model if target == "tts" else wrapper.model.speech_tokenizer.model
                    if kind == "training": owner.training = True
                    if kind == "empty": owner.parameters = lambda: iter([])
                    if kind == "cuda": owner.parameters = lambda: iter([NS(device="cuda:0", dtype="torch.float32")])
                    if kind == "half": owner.parameters = lambda: iter([NS(device="cpu", dtype="torch.float16")])
                    if kind == "buffer": owner.buffers = lambda: iter([NS(device="cuda:0")])
                    if kind == "attention": owner.named_modules = lambda: iter([])
                    if kind == "flash": owner.named_modules = lambda: iter([("attn", NS(q_proj=1, k_proj=1, v_proj=1, config=NS(_attn_implementation="flash_attention_2")))])
                    with self.assertRaises(ValueError): worker.validate_model(wrapper)


class ExecuteTests(TempCase):
    def run_cells(self, generate=None, write=None, **kwargs):
        return worker.execute_cells(self.root, generate or (lambda row: (object(), eos_record(), 0.25)),
                                    write or (lambda array, cell: {"source_frames": 3840}),
                                    alarm=lambda seconds: None, **kwargs)

    def outcomes(self):
        return json.loads((self.root / "outcomes.json").read_bytes())

    def test_exact_six_fixed_calls_with_durable_claim_before_each_callback(self):
        synced_paths = []
        actual_fsync = os.fsync
        def sync(fd):
            synced_paths.append(os.readlink("/proc/self/fd/%d" % fd))
            actual_fsync(fd)
        called = []
        def generate(row):
            called.append(deepcopy(row))
            path = self.root / "claims" / (row["cell_id"] + ".started.json")
            raw = path.read_bytes()
            started = json.loads(raw)
            self.assertEqual(started["request_sha256"], hashlib.sha256(contract.canonical(row)).hexdigest())
            self.assertIn(str(path), synced_paths)
            self.assertGreater(synced_paths.index(str(path.parent)), synced_paths.index(str(path)))
            current = self.outcomes()
            self.assertEqual(current["consumed_attempts"], len(called))
            self.assertEqual(current["cells"][len(called)-1]["claim"], {"bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()})
            synced_paths.clear()
            return object(), eos_record(), 0.25
        with mock.patch.object(common.os, "fsync", side_effect=sync):
            outcome = self.run_cells(generate)
        self.assertEqual(called, contract.cells())
        self.assertEqual([(x["voice"], x["text"], x["seed"]) for x in called],
            [(voice, text, seed) for voice, seed in (("Dylan",1012001),("Uncle_Fu",1012002),("Vivian",1012003))
             for text in ("你好小窝", "你好，小窝")])
        self.assertEqual(outcome["status"], "completed_six_once")
        self.assertEqual(outcome["consumed_attempts"], 6)
        self.assertEqual(len(list((self.root / "claims").iterdir())), 6)
        self.assertEqual(len(outcome["cells"]), 6)

    def test_failure_at_each_position_consumes_once_and_stops(self):
        for position in range(6):
            with self.subTest(position=position), tempfile.TemporaryDirectory(dir=RESULTS) as td:
                root = Path(td)
                calls = []
                def generate(row):
                    calls.append(row["cell_id"])
                    if len(calls) == position+1: raise TimeoutError("synthetic deadline")
                    return object(), eos_record(), 0.1
                with self.assertRaises(TimeoutError):
                    worker.execute_cells(root, generate, lambda *_: {"source_frames": 3840}, alarm=lambda _: None)
                result = json.loads((root / "outcomes.json").read_bytes())
                self.assertEqual(len(calls), position+1)
                self.assertEqual(result["consumed_attempts"], position+1)
                self.assertEqual([r["status"] for r in result["cells"]], ["complete"]*position + ["failed_no_retry"] + ["not_run"]*(5-position))
                self.assertEqual(result["status"], "stopped_no_retry")
                self.assertEqual(len(result["cells"]), 6)
                self.assertEqual(len(list((root / "claims").iterdir())), position+1)

    def test_no_resume_or_extra_calls_after_completed_run(self):
        self.run_cells()
        call = mock.Mock()
        with self.assertRaises(ValueError): self.run_cells(call)
        call.assert_not_called()

    def test_claim_error_prevents_callback(self):
        call = mock.Mock()
        with mock.patch.object(worker, "write_once", side_effect=OSError("synthetic fsync failure")):
            with self.assertRaises(OSError): self.run_cells(call)
        call.assert_not_called()
        self.assertEqual(self.outcomes()["consumed_attempts"], 0)
        self.assertEqual([r["status"] for r in self.outcomes()["cells"]], ["not_run"]*6)

    def test_real_claim_file_or_directory_fsync_failure_prevents_callback(self):
        for failure in ("file","directory"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(dir=RESULTS) as td:
                root=Path(td);original=os.fsync;callback=mock.Mock()
                def sync(fd):
                    name=os.readlink("/proc/self/fd/%d" % fd)
                    if (failure=="file" and name.endswith(".started.json")) or (failure=="directory" and name==str(root/"claims")):
                        raise OSError("synthetic durable claim failure")
                    original(fd)
                with mock.patch.object(common.os,"fsync",side_effect=sync):
                    with self.assertRaises(OSError):
                        worker.execute_cells(root,callback,mock.Mock(),alarm=lambda _:None)
                callback.assert_not_called()
                self.assertTrue((root/"claims/k1p6-001.started.json").exists())
                with self.assertRaises(ValueError):
                    worker.execute_cells(root,callback,mock.Mock(),alarm=lambda _:None)
                callback.assert_not_called()

    def test_failed_capture_evidence_preserved(self):
        state = {"capture_calls": 1, "iterations": 120, "ended_with_eos": False}
        def generate(_): raise ValueError("synthetic capture failure")
        with self.assertRaises(ValueError): self.run_cells(generate, capture_state=state)
        eos = json.loads((self.root / "eos.json").read_bytes())
        self.assertEqual(eos["rows"], [{"cell_id":"k1p6-001", "failed_capture":state}])
        self.assertEqual(self.outcomes()["cells"][0]["error_class"], "ValueError")

    def test_rejected_eos_never_writes_audio(self):
        bad = eos_record(120)
        write = mock.Mock()
        with self.assertRaises(ValueError): self.run_cells(lambda _: (object(), bad, 0.1), write)
        write.assert_not_called()
        self.assertEqual(self.outcomes()["consumed_attempts"], 1)
        self.assertEqual(self.outcomes()["cells"][0]["status"], "failed_no_retry")

    def test_fixed_eos_identity_rejected_before_audio(self):
        bad = eos_record()
        bad.update(eos=999, last_token_ids=[999])
        write = mock.Mock(return_value={"source_frames":3840})
        with self.assertRaises(ValueError): self.run_cells(lambda _: (object(), bad, 0.1), write)
        write.assert_not_called()

    def test_custom_exception_name_does_not_leak_into_evidence(self):
        private_error=type("SecretPathOrTokenWouldLeak",(Exception,),{})
        def fail(_): raise private_error("private error message")
        with self.assertRaises(private_error):self.run_cells(fail)
        output=self.outcomes()
        self.assertEqual(output["error_class"],"OtherException")
        self.assertEqual(output["cells"][0]["error_class"],"OtherException")
        self.assertNotIn("private error message",(self.root/"outcomes.json").read_text())
        self.assertNotIn("SecretPathOrTokenWouldLeak",(self.root/"outcomes.json").read_text())

    def test_audio_geometry_failure_consumes_and_stops(self):
        with self.assertRaises(ValueError): self.run_cells(write=lambda *_: {"source_frames": 1})
        self.assertEqual(self.outcomes()["consumed_attempts"], 1)
        self.assertEqual(self.outcomes()["cells"][0]["status"], "failed_no_retry")

    def test_keyboard_interrupt_remains_consumed_and_cancels_alarm(self):
        alarms = []
        def fail(_): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            worker.execute_cells(self.root, fail, mock.Mock(), alarm=alarms.append)
        self.assertEqual(alarms, [300, 0, 0])
        self.assertEqual(self.outcomes()["consumed_attempts"], 1)


class GateTests(TempCase):
    def context(self):
        env = {"GITHUB_REPOSITORY":gate.REPOSITORY,"GITHUB_EVENT_NAME":"push",
               "GITHUB_REF":"refs/heads/"+gate.SOURCE_BRANCH,"GITHUB_JOB":"k1-tts",
               "GITHUB_RUN_ATTEMPT":"1","RUNNER_ENVIRONMENT":"github-hosted","RUNNER_OS":"Linux",
               "RUNNER_ARCH":"X64","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"701"}
        event = {"created":True,"deleted":False,"ref":env["GITHUB_REF"],"after":env["GITHUB_SHA"],
                 "repository":{"full_name":gate.REPOSITORY,"private":False}}
        return env, event

    def api(self):
        env, event = self.context()
        identity = gate.fresh_context(env, event)
        base = "https://api.github.com/repos/"+gate.REPOSITORY
        run_url = base+"/actions/runs/701"
        history_url = base+"/actions/workflows/"+gate.WORKFLOW.split("/")[-1]+"/runs?branch="+gate.SOURCE_BRANCH+"&event=push&per_page=100"
        job = {"id":702,"name":"k1-tts","status":"in_progress","conclusion":None,"head_sha":"a"*40,"run_id":701,"labels":["ubuntu-24.04"]}
        run = {"id":701,"run_attempt":1,"head_sha":"a"*40,"head_branch":gate.SOURCE_BRANCH,"event":"push","path":gate.WORKFLOW,"status":"in_progress"}
        replies = {base:{"full_name":gate.REPOSITORY,"private":False},run_url:run,
                   history_url:{"total_count":1,"workflow_runs":[deepcopy(run)]},
                   run_url+"/attempts/1/jobs?per_page=100":{"total_count":1,"jobs":[job]},
                   base+"/git/ref/heads/"+gate.RESULT_BRANCH:None}
        seen = []
        def get(url, expected=200):
            seen.append((url, expected))
            if url not in replies: raise AssertionError("unexpected remote protocol path")
            return deepcopy(replies[url])
        return identity, replies, seen, get

    def test_exact_fresh_context(self):
        identity = gate.fresh_context(*self.context())
        self.assertEqual(identity["run_attempt"], 1)
        self.assertEqual(identity["run_id"], 701)

    def test_environment_variants_fail_closed(self):
        changes = {"GITHUB_REPOSITORY":"other/repo","GITHUB_EVENT_NAME":"workflow_dispatch",
                   "GITHUB_REF":"refs/heads/main","GITHUB_JOB":"other","GITHUB_RUN_ATTEMPT":"2",
                   "RUNNER_ENVIRONMENT":"self-hosted","RUNNER_OS":"Windows","RUNNER_ARCH":"ARM64",
                   "GITHUB_SHA":"a"*39,"GITHUB_RUN_ID":"0701"}
        for key, value in changes.items():
            with self.subTest(key=key):
                env, event = self.context()
                env[key] = value
                with self.assertRaises(ValueError): gate.fresh_context(env, event)

    def test_event_created_exact_true_and_public_exact_false_required(self):
        variants = [{"created":False}, {"created":1}, {"deleted":True}, {"ref":"refs/heads/main"},
                    {"after":"b"*40}, {"repository":{"full_name":gate.REPOSITORY,"private":True}},
                    {"repository":{"full_name":gate.REPOSITORY,"private":0}}]
        for change in variants:
            with self.subTest(change=change):
                env, event = self.context()
                event.update(change)
                with self.assertRaises(ValueError): gate.fresh_context(env, event)

    def test_preflight_uses_only_same_workflow_history_and_exact_five_gets(self):
        identity, replies, seen, get = self.api()
        replies["https://api.github.com/repos/"+gate.REPOSITORY+"/actions/runs"] = {"total_count":5000,"workflow_runs":[]}
        result = gate.read_only_preflight(identity, get)
        self.assertTrue(result["preflight_verified"])
        self.assertEqual(result["numeric_job_id"], 702)
        self.assertEqual(len(seen), 5)
        self.assertIn("/actions/workflows/k1-punctuation6-tts-v1.yml/runs?", seen[2][0])
        self.assertEqual(seen[-1][1], 404)

    def test_preflight_current_run_mutations_rejected(self):
        for key, value in (("run_attempt",2),("id",702),("head_sha","b"*40),("head_branch","main"),
                           ("event","workflow_dispatch"),("path","other.yml"),("status","completed")):
            with self.subTest(key=key):
                identity, replies, seen, get = self.api()
                url = next(k for k in replies if k.endswith("/runs/701"))
                replies[url][key] = value
                with self.assertRaises(ValueError): gate.read_only_preflight(identity, get)

    def test_history_incomplete_or_already_consumed_rejected(self):
        for count, rows in ((0,0),(2,1),(2,2),(101,100),(True,1)):
            with self.subTest(count=count, rows=rows):
                identity, replies, seen, get = self.api()
                url = next(k for k in replies if "/actions/workflows/" in k)
                row = replies[url]["workflow_runs"][0]
                replies[url] = {"total_count":count,"workflow_runs":[deepcopy(row) for _ in range(rows)]}
                with self.assertRaises(ValueError): gate.read_only_preflight(identity, get)

    def test_history_wrong_current_id_path_or_commit_rejected(self):
        for key, value in (("id",702),("path","other.yml"),("head_sha","b"*40)):
            with self.subTest(key=key):
                identity, replies, seen, get = self.api()
                url = next(k for k in replies if "/actions/workflows/" in k)
                replies[url]["workflow_runs"][0][key] = value
                with self.assertRaises(ValueError): gate.read_only_preflight(identity, get)

    def test_jobs_incomplete_nonstandard_or_matrix_rejected(self):
        for kind in ("incomplete", "overflow", "matrix", "selfhosted", "larger", "wrong_head", "wrong_run", "finished"):
            with self.subTest(kind=kind):
                identity, replies, seen, get = self.api()
                url = next(k for k in replies if "/attempts/1/jobs?" in k)
                jobs = replies[url]
                job = jobs["jobs"][0]
                if kind == "incomplete": jobs["total_count"] = 2
                if kind == "overflow": jobs["total_count"] = 101
                if kind == "matrix": jobs.update(total_count=2,jobs=[job,deepcopy(job)])
                if kind == "selfhosted": job["labels"] = ["self-hosted","ubuntu-24.04"]
                if kind == "larger": job["labels"] = ["ubuntu-24.04-8core"]
                if kind == "wrong_head": job["head_sha"] = "b"*40
                if kind == "wrong_run": job["run_id"] = 702
                if kind == "finished": job["status"] = "completed"
                with self.assertRaises(ValueError): gate.read_only_preflight(identity, get)

    def source_root(self):
        k1=self.root/"research/k1-punctuation6-v1";k1.mkdir(parents=True)
        helper=self.root/"research/qwen6_tts";helper.mkdir(parents=True)
        (k1/"plan.proposed.json").write_bytes(contract.canonical(contract.expected_plan()))
        (k1/"entry.py").write_bytes(b"# fixed offline source\n")
        (helper/"helper.py").write_bytes(b"# fixed offline helper\n")
        workflow=self.root/gate.WORKFLOW;workflow.parent.mkdir(parents=True)
        workflow.write_bytes(b"# disabled fixture workflow\n")
        freeze={"schema":"k1-punctuation6-tts-source-freeze-v1","files":{}}
        for name in gate.source_members(self.root):
            raw=(self.root/name).read_bytes()
            freeze["files"][name]={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        (k1/"tts-source-freeze.json").write_bytes(contract.canonical(freeze))
        return k1,helper

    def test_source_freeze_covers_exact_files_bytes_and_plan(self):
        self.source_root()
        result=gate.verify_source(self.root)
        self.assertEqual(result["frozen_files"],4)
        self.assertEqual(result["source_freeze_sha256"],common.sha(self.root/"research/k1-punctuation6-v1/tts-source-freeze.json"))

    def test_source_byte_drift_rejected(self):
        k1,_=self.source_root()
        (k1/"entry.py").write_bytes(b"# drift\n")
        with self.assertRaises(ValueError):gate.verify_source(self.root)

    def test_source_extra_member_or_pycache_rejected(self):
        k1,_=self.source_root()
        extra=k1/"unfrozen.txt";extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):gate.verify_source(self.root)
        extra.unlink()
        (k1/"__pycache__").mkdir()
        with self.assertRaises(ValueError):gate.verify_source(self.root)

    def test_source_alias_rejected(self):
        k1,_=self.source_root()
        (k1/"alias.py").symlink_to(k1/"entry.py")
        with self.assertRaises(ValueError):gate.verify_source(self.root)

    def release_fixture(self):
        """Independent temporary source and metadata; no actual release is changed."""
        k1,_=self.source_root()
        names=("data_transport.py","data_publisher.py","tts_publish.py","tts_common.py","tts_gate.py")
        for name in names:(k1/name).write_bytes(("# offline fixed writer fixture "+name+"\n").encode())
        release=json.loads(Path(__file__).with_name("tts-release.json").read_bytes())
        release.update(enabled=True,fixture_write_readback_verified=True,fixture_source_commit="d"*40,
            fixture_result_commit="e"*40,fixture_archive_sha256="f"*64,
            fixture_publisher_source_hashes={name:common.sha(k1/name) for name in names},
            independent_source_review=True,root_execution_release=True,historical_audio_inventory_frozen=True)
        (k1/"tts-release.json").write_bytes(contract.canonical(release))
        freeze={"schema":"k1-punctuation6-tts-source-freeze-v1","files":{}}
        for name in gate.source_members(self.root):
            path=self.root/name
            freeze["files"][name]={"bytes":path.stat().st_size,"sha256":common.sha(path)}
        (k1/"tts-source-freeze.json").write_bytes(contract.canonical(freeze))
        return k1,release

    def test_enabled_fixture_gate_binds_all_five_writer_files_and_source(self):
        k1,release=self.release_fixture()
        actual,source=gate.verify_release(self.root)
        self.assertEqual(actual,release)
        self.assertEqual(set(actual["fixture_publisher_source_hashes"]),
                         {"data_transport.py","data_publisher.py","tts_publish.py","tts_common.py","tts_gate.py"})
        self.assertEqual(source["source_freeze_sha256"],common.sha(k1/"tts-source-freeze.json"))
        self.assertEqual(source["frozen_files"],10)

    def test_fixture_source_commit_need_not_equal_separate_tts_source_commit(self):
        _,release=self.release_fixture()
        env,event=self.context()
        actual_tts_identity=gate.fresh_context(env,event)
        self.assertNotEqual(release["fixture_source_commit"],actual_tts_identity["executed_head_sha"])
        self.assertNotEqual(release["fixture_result_commit"],actual_tts_identity["executed_head_sha"])
        self.assertEqual(gate.verify_release(self.root)[0],release)

    def test_release_exact_fields_reject_missing_or_extra_metadata(self):
        k1,release=self.release_fixture()
        for kind in ("missing","extra"):
            with self.subTest(kind=kind):
                changed=deepcopy(release)
                if kind=="missing":del changed["fixture_source_commit"]
                else:changed["private_note"]="sentinel"
                (k1/"tts-release.json").write_bytes(contract.canonical(changed))
                with mock.patch.object(gate,"verify_source") as verify:
                    with self.assertRaisesRegex(ValueError,"exact TTS release fields"):gate.verify_release(self.root)
                verify.assert_not_called()

    def test_release_fixture_source_result_archive_require_exact_hex_types(self):
        k1,release=self.release_fixture()
        for key,bad in (("fixture_source_commit",None),("fixture_source_commit","d"*39),("fixture_source_commit","D"*40),
                        ("fixture_result_commit",True),("fixture_result_commit","bad"),
                        ("fixture_archive_sha256",None),("fixture_archive_sha256","f"*63),("fixture_archive_sha256",17)):
            with self.subTest(key=key,bad=bad):
                changed=deepcopy(release);changed[key]=bad
                (k1/"tts-release.json").write_bytes(contract.canonical(changed))
                with mock.patch.object(gate,"verify_source") as verify:
                    with self.assertRaises(ValueError):gate.verify_release(self.root)
                verify.assert_not_called()

    def test_release_writer_hash_map_rejects_missing_extra_or_wrong_entries(self):
        k1,release=self.release_fixture()
        for kind in ("missing","extra","wrong","null","list"):
            with self.subTest(kind=kind):
                changed=deepcopy(release);hashes=changed["fixture_publisher_source_hashes"]
                if kind=="missing":hashes.pop("tts_gate.py")
                if kind=="extra":hashes["unrelated.py"]="a"*64
                if kind=="wrong":hashes["tts_gate.py"]="0"*64
                if kind=="null":changed["fixture_publisher_source_hashes"]=None
                if kind=="list":changed["fixture_publisher_source_hashes"]=list(hashes)
                (k1/"tts-release.json").write_bytes(contract.canonical(changed))
                with mock.patch.object(gate,"verify_source") as verify:
                    with self.assertRaisesRegex(ValueError,"same reviewed writer bytes"):gate.verify_release(self.root)
                verify.assert_not_called()

    def test_release_each_writer_byte_drift_rejected_before_source_verification(self):
        k1,release=self.release_fixture()
        for name in release["fixture_publisher_source_hashes"]:
            with self.subTest(name=name):
                path=k1/name;original=path.read_bytes();path.write_bytes(original+b"# drift\n")
                with mock.patch.object(gate,"verify_source") as verify:
                    with self.assertRaisesRegex(ValueError,"same reviewed writer bytes"):gate.verify_release(self.root)
                verify.assert_not_called();path.write_bytes(original)

    def test_release_scope_review_and_readback_flags_are_exact(self):
        k1,release=self.release_fixture()
        changes=(("enabled",1),("fixture_write_readback_verified",1),("fixture_write_readback_verified",False),
                 ("maximum_tts_attempts",6.0),("maximum_tts_attempts",7),("asr_enabled",True),("training_enabled",True),
                 ("source_branch","main"),("result_branch","main"),("independent_source_review",False),
                 ("root_execution_release",False),("historical_audio_inventory_frozen",False))
        for key,bad in changes:
            with self.subTest(key=key,bad=bad):
                changed=deepcopy(release);changed[key]=bad
                (k1/"tts-release.json").write_bytes(contract.canonical(changed))
                with mock.patch.object(gate,"verify_source") as verify:
                    with self.assertRaises(ValueError):gate.verify_release(self.root)
                verify.assert_not_called()

    def test_release_nonwriter_source_byte_drift_still_rejected(self):
        k1,_=self.release_fixture()
        (k1/"entry.py").write_bytes(b"# nonwriter drift\n")
        with self.assertRaisesRegex(ValueError,"frozen source byte drift"):gate.verify_release(self.root)

    def test_disabled_release_blocks_before_source_verification(self):
        release = json.loads(Path(__file__).with_name("tts-release.json").read_bytes())
        # This negative fixture is independent of the reviewed repository activation.
        release["enabled"] = False
        folder = self.root / "research/k1-punctuation6-v1"
        folder.mkdir(parents=True)
        (folder / "tts-release.json").write_bytes(contract.canonical(release))
        with mock.patch.object(gate, "verify_source") as verify:
            with self.assertRaisesRegex(ValueError, "disabled"): gate.verify_release(self.root)
        verify.assert_not_called()

    def test_real_worker_disabled_gate_precedes_inference_import(self):
        imported = []
        native_import = __import__
        def tracking(name, *args, **kwargs):
            if name.split(".")[0] in {"torch","numpy","soundfile","scipy","qwen_tts","transformers"}:
                imported.append(name)
                raise AssertionError("inference import crossed disabled gate")
            return native_import(name, *args, **kwargs)
        with mock.patch.object(gate, "verify_worker_release", side_effect=ValueError("disabled")), mock.patch("builtins.__import__", side_effect=tracking):
            with self.assertRaisesRegex(ValueError, "disabled"): worker.run_real(self.root, self.root / "publication-members")
        self.assertEqual(imported, [])

    def test_json_duplicate_nonfinite_and_alias_rejected(self):
        path = self.root / "bad.json"
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): gate.read_json(path)


class NumericArray:
    """Small deterministic fake; no NumPy/SciPy numerical correctness claim."""
    def __init__(self, values, dtype="float32", ndim=1):
        self.values, self.dtype, self.ndim = list(values), dtype, ndim
    def __len__(self): return len(self.values)
    def __iter__(self): return iter(self.values)
    def tobytes(self): return struct.pack("<%d%s" % (len(self), "d" if self.dtype in ("<f8", "float64") else "f"), *self.values)
    def astype(self, dtype): return NumericArray(self.values, dtype)
    def all(self): return all(self.values)


class NumericOps:
    float64 = "float64"
    @staticmethod
    def asarray(a, dtype): return NumericArray(a.values, dtype)
    @staticmethod
    def isfinite(a): return NumericArray([math.isfinite(v) for v in a])
    @staticmethod
    def any(a): return any(a)
    @staticmethod
    def abs(a): return NumericArray([abs(v) for v in a])
    @staticmethod
    def max(a): return max(a)


class SoundFileFake:
    def __init__(self,pcm_values=None):
        self.calls,self.raw_values=[],None
        self.pcm_values=None if pcm_values is None else list(pcm_values)
    def write(self, path, array, rate, subtype):
        self.calls.append((Path(path).name, rate, subtype))
        if subtype == "FLOAT":
            self.raw_values = list(array)
            Path(path).write_bytes(float_wav(array.values))
        elif subtype == "PCM_16":
            if self.pcm_values is None:
                Path(path).write_bytes(pcm_wav(len(array)))
            else:
                if len(self.pcm_values)!=len(array):raise ValueError("synthetic PCM fixture geometry")
                Path(path).write_bytes(pcm_wav_samples(self.pcm_values))
        else: raise AssertionError("unexpected subtype")
    def read(self, path, dtype): return NumericArray(self.raw_values), 24000


class WritePairTests(TempCase):
    def setUp(self):
        super().setUp()
        (self.root / "raw").mkdir()
        (self.root / "derived").mkdir()
        self.sf = SoundFileFake()
        self.resampler = mock.Mock(side_effect=lambda a, up, down: NumericArray([0.125] * ((len(a)*2+2)//3), "float64"))
    def write(self, array):
        return worker.write_pair(array, "k1p6-001", self.root, NumericOps, self.sf, self.resampler)

    def test_fixed_write_geometry_readback_hashes_and_fsync_with_fakes(self):
        array = NumericArray([0.25] * 3840)
        calls = []
        original = os.fsync
        def sync(fd):
            calls.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            original(fd)
        with mock.patch.object(common.os, "fsync", side_effect=sync): result = self.write(array)
        self.assertEqual(self.sf.calls, [("k1p6-001.wav",24000,"FLOAT"),("k1p6-001.wav",16000,"PCM_16")])
        self.assertEqual(calls, ["file","dir","file","dir"])
        self.assertEqual(self.resampler.call_args.args[1:], (2,3))
        self.assertEqual(self.resampler.call_args.args[0].dtype, "float64")
        self.assertEqual((result["source_frames"],result["frames"]), (3840,2560))
        self.assertEqual(result["raw_float_sha256"], hashlib.sha256(array.tobytes()).hexdigest())
        self.assertEqual(result["raw_wav_sha256"], common.sha(self.root / result["raw_path"]))
        self.assertEqual(result["wav_sha256"], common.sha(self.root / result["derived_path"]))
        self.assertEqual(result["quality_flags"], [])
        self.assertEqual(result["source_peak"],0.25)
        self.assertEqual(result["derivative_peak"],0.125)
        self.assertEqual(result["derivative_float_sha256"],hashlib.sha256(struct.pack("<d",0.125)*2560).hexdigest())

    def test_input_shape_dtype_empty_duration_and_nonfinite_rejected(self):
        for array in (NumericArray([0.1],ndim=2), NumericArray([0.1],dtype="float64"),
                      NumericArray([]), NumericArray([0.0]*(12*24000)), NumericArray([math.nan]), NumericArray([math.inf])):
            with self.subTest(dtype=array.dtype, ndim=array.ndim, length=len(array)):
                with self.assertRaises(ValueError): self.write(array)
        self.assertEqual(self.sf.calls, [])

    def test_float_readback_mutation_rejected(self):
        self.sf.read = lambda path,dtype: (NumericArray([0.5]*3),24000)
        with self.assertRaises(ValueError): self.write(NumericArray([0.25]*3))
        self.resampler.assert_not_called()

    def test_resample_bad_length_rejected(self):
        self.resampler.side_effect = lambda *_: NumericArray([0.25])
        with self.assertRaises(ValueError): self.write(NumericArray([0.25]*6))
        self.assertEqual(len(self.sf.calls), 1)

    def test_no_overwrite(self):
        self.write(NumericArray([0.25]*6))
        with self.assertRaises(ValueError): self.write(NumericArray([0.25]*6))
        self.assertEqual(len(self.sf.calls), 2)

    def test_derived_pcm_silent_detects_zero_stored_pcm_with_nonzero_tiny_raw(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        self.sf=SoundFileFake([0]*4)
        self.resampler.side_effect=lambda array,*_:NumericArray([tiny]*4,"float64")
        result=self.write(NumericArray([tiny]*6))
        self.assertGreater(result["source_peak"],0)
        self.assertGreater(result["derivative_peak"],0)
        self.assertEqual(result["quality_flags"],["derived_pcm_silent"])
        with wave.open(str(self.root/result["derived_path"]),"rb") as handle:
            self.assertFalse(any(handle.readframes(handle.getnframes())))

    def test_zero_raw_and_zero_stored_pcm_retain_both_silence_flags(self):
        self.sf=SoundFileFake([0]*4)
        self.resampler.side_effect=lambda array,*_:NumericArray([0.0]*4,"float64")
        result=self.write(NumericArray([0.0]*6))
        self.assertEqual(result["quality_flags"],["silent","derived_pcm_silent"])
        self.assertEqual(result["source_peak"],0)
        self.assertEqual(result["derivative_peak"],0)

    def test_any_single_positive_or_negative_pcm_lsb_prevents_silent_flag(self):
        tiny=struct.unpack("<f",struct.pack("<f",1e-10))[0]
        for value in (1,-1):
            for position in (0,1,3):
                with self.subTest(value=value,position=position),tempfile.TemporaryDirectory(dir=RESULTS) as td:
                    root=Path(td);(root/"raw").mkdir();(root/"derived").mkdir()
                    samples=[0]*4;samples[position]=value
                    result=worker.write_pair(NumericArray([tiny]*6),"k1p6-001",root,NumericOps,
                        SoundFileFake(samples),lambda array,*_:NumericArray([tiny]*4,"float64"))
                    self.assertEqual(result["quality_flags"],[])

    def test_quality_flags_preserve_silent_and_out_of_range_samples(self):
        self.resampler.side_effect = lambda a,*_: NumericArray([1.5]*((len(a)*2+2)//3))
        result = self.write(NumericArray([0.0]*6))
        self.assertEqual(result["quality_flags"], ["silent","normalized_peak_ge_one"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
