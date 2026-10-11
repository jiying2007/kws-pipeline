"""Prospective native-v2 observations around the pinned, unchanged ASR calls.

No loader, launcher, download, retry, permission grant, or historical conversion.
The caller verifies immutable source/assets/kernel scope before constructing this
object, retains exact lock bytes, then consumes each blind clip at most once.
All method wrappers are observational and restored, including on exceptions.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from asr_native_contracts import (ADAPTER_FAMILIES, CAPTURE_CLASS, CONDITIONING_POLICY,
    CONFIG_CLASS, QWEN_ADAPTER, SENSE_ADAPTER, QWEN_REQUEST, SENSE_REQUEST,
    canonical_sha256, validate_native_config, validate_native_capture)

QWEN = "Qwen/Qwen3-ASR-0.6B"
SENSE = "FunAudioLLM/SenseVoiceSmall"
QWEN_CONTROLS = ("num_beams", "num_return_sequences", "forced_bos_token_id",
    "forced_eos_token_id", "suppress_tokens", "begin_suppress_tokens", "bad_words_ids",
    "force_words_ids", "sequence_bias", "stop_strings")
FRONTEND_FIELDS = ("fs", "window", "n_mels", "frame_length", "frame_shift",
    "filter_length_min", "filter_length_max", "lfr_m", "lfr_n", "dither",
    "snip_edges", "upsacle_samples")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clone(value):
    # Also rejects non-JSON/nonfinite fields. Missing fields never get defaults.
    canonical_sha256(value)
    return json.loads(json.dumps(value, allow_nan=False))


def same(left, right, message):
    require(canonical_sha256(left) == canonical_sha256(right), message)


def execution(model, declaration):
    require(model.training is False, "native model must be in eval mode")
    count = 0
    for parameter in model.parameters():
        require(parameter.device.type == "cpu" and str(parameter.dtype) == "torch.float32",
                "native live parameters must all be CPU float32")
        count += 1
    require(count > 0 and type(declaration["batch_size"]) is int and declaration["batch_size"] == 1,
            "native parameter/batch observation missing")
    return {"device": "cpu", "dtype": "float32", "batch_size": declaration["batch_size"]}


def generation_snapshot(config):
    snapshot = clone(config.to_dict())
    for key in (*QWEN_CONTROLS, "max_new_tokens", "do_sample", "eos_token_id"):
        require(key in snapshot, "missing observed Qwen generation control: " + key)
    return snapshot


def qwen_effective(config, execution_observed, custom_processors, custom_stops):
    snapshot = generation_snapshot(config)
    require(custom_processors is not None and custom_stops is not None,
            "missing observed Qwen custom control lists")
    require(len(custom_processors) == 0 and len(custom_stops) == 0,
            "unreviewed Qwen custom processors/stopping criteria")
    mode = config.get_generation_mode().value
    return {"execution": clone(execution_observed),
        "native_request": {"context": "", "force_language": None,
            **{key: snapshot[key] for key in ("max_new_tokens", "do_sample", "eos_token_id")}},
        "generation_mode": mode, **{key: snapshot[key] for key in QWEN_CONTROLS},
        "custom_logits_processors": [], "custom_stopping_criteria": [],
        "resolved_generation_config_sha256": canonical_sha256(snapshot)}


def frontend_snapshot(frontend, asset_receipt):
    from asr_stage.runtime_support import describe_cpu_tensor
    require(type(frontend).__module__ == "funasr.frontends.wav_frontend"
            and type(frontend).__name__ == "WavFrontend", "unreviewed SenseVoice frontend")
    values = {name: getattr(frontend, name) for name in FRONTEND_FIELDS}
    files = {row["filename"]: row for row in asset_receipt["verified_files"]}
    cmvn = Path(frontend.cmvn_file)
    require(cmvn.name == "am.mvn" and not cmvn.is_symlink(), "SenseVoice CMVN path")
    raw = cmvn.read_bytes()
    require(len(raw) == files["am.mvn"]["bytes"]
            and hashlib.sha256(raw).hexdigest() == files["am.mvn"]["sha256"], "SenseVoice CMVN drift")
    return clone({"class": type(frontend).__module__ + "." + type(frontend).__name__,
        "attributes": values, "cmvn_sha256": hashlib.sha256(raw).hexdigest(),
        "loaded_cmvn_tensor": describe_cpu_tensor(frontend.cmvn, True)})


def sense_effective(runtime, model_root):
    from sense_adapter import runtime_tokenizer_path, verify_tokenizer
    from sense_binding import _contract, CONTRACT_SHA256
    wrapper = runtime.wrapper
    require(wrapper.vad_model is None, "SenseVoice VAD must be absent")
    identity = verify_tokenizer(wrapper.kwargs["tokenizer"].sp, runtime_tokenizer_path(model_root))
    parser = _contract()  # Retained source bytes/hash actually checked, not a hash-only claim.
    require(len(parser["SENSE_GROUPS"]) == 4, "SenseVoice parser metadata prefix")
    frontend = frontend_snapshot(wrapper.kwargs["frontend"], runtime.receipt["assets"])
    effective = {"execution": execution(wrapper.model, runtime.declaration),
        "native_request": clone(SENSE_REQUEST), "decoding": "full_input_greedy_ctc",
        "vad_used": wrapper.vad_model is not None, "rich_postprocess_applied": False,
        "vocabulary_size": identity["vocabulary_size"], "lexical_unknown_id": identity["lexical_unknown_id"],
        "metadata_prefix_tokens": len(parser["SENSE_GROUPS"]), "tokenizer_sha256": identity["sha256"],
        "parser_contract_sha256": CONTRACT_SHA256, "frontend_config_sha256": canonical_sha256(frontend)}
    return effective, frontend


@contextmanager
def observe_method(owner, name, observer):
    """Restore the precise instance state, not a permanent bound-method override."""
    original = getattr(owner, name)
    was_instance_attribute = name in vars(owner)
    saved = vars(owner).get(name)
    def observed(*args, **kwargs):
        return observer(original, args, kwargs)
    setattr(owner, name, observed)
    try:
        yield
    finally:
        if was_instance_attribute:
            setattr(owner, name, saved)
        else:
            delattr(owner, name)


class FeatureObserver:
    """Qwen ProcessorMixin calls this actual feature-extractor object once."""
    def __init__(self, original, bound, observed):
        self.original, self.bound, self.observed = original, bound, observed

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, audio, **kwargs):
        from asr_stage.runtime_support import verify_pcm_array
        require("input" not in self.observed and type(audio) is list and len(audio) == 1,
                "exactly one Qwen full-wave feature call")
        verify_pcm_array(audio[0], self.bound.descriptor["float32_pcm"])
        for key, expected in (("truncation", False), ("padding", True),
                              ("return_attention_mask", True), ("sampling_rate", 16000)):
            require(key in kwargs, "missing observed Qwen frontend control: " + key)
            same(kwargs[key], expected, "Qwen frontend control drift: " + key)
        self.observed["input"] = full_input(self.bound)
        self.observed["frontend_call"] = {key: kwargs[key] for key in
            ("truncation", "padding", "return_attention_mask", "sampling_rate")}
        return self.original(audio, **kwargs)


def full_input(bound):
    frames = bound.descriptor["raw_pcm16"]["frame_count"]
    require(type(frames) is int and 0 < frames <= 192000, "bounded full-input native frames")
    return {"sample_rate": 16000, "channels": 1, "audio_frames": frames,
        "processed_start_frame": 0, "processed_end_frame": frames, "padding_frames": 0,
        "trimmed_frames": 0, "preprocessing": "full_input", "pcm_format": "float32_pcm16_scale_32768"}


def qwen_termination(ids):
    require(type(ids) is list and all(type(token) is int for token in ids), "observed Qwen token IDs")
    ended = bool(ids) and ids[-1] in QWEN_REQUEST["eos_token_id"]
    clean = ended and not any(token in QWEN_REQUEST["eos_token_id"] for token in ids[:-1])
    return {"kind": "autoregressive_eos", "capture_stage": "returned_generation_sequence",
        "generated_token_ids": list(ids), "generated_tokens": len(ids),
        "stop_reason": "eos_token" if clean else ("token_limit" if len(ids) == 256 and not ended else "unexplained_return"),
        "truncated": not clean}


def sense_termination(raw):
    binding = raw["token_binding"]
    require(binding["token_scope"] == "Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits",
            "SenseVoice collapsed nonblank capture scope")
    return {"kind": "full_input_ctc_return", "inference_returned": True,
        "token_scope": "ctc_collapsed_nonblank_ids_not_framewise_logits",
        **{key: clone(binding[key]) for key in ("raw_token_ids", "metadata_token_ids", "lexical_token_ids",
            "metadata_boundary_verified", "raw_redecode_matches_capture")}, "truncated": False}


class NativeCaptureAdapter:
    """One loaded runtime, one frozen config, only one attempt per bound clip.

    ``retain(name, JSON_value)`` must atomically retain immutable JSON. Exact
    ``source_lock_bytes``/``asset_lock_bytes`` are retained by the caller before
    construction. The first may extend historical package pins with observed
    runtime source pins; it must not replace their existing independent checks.
    """
    def __init__(self, runtime, model_root, source_lock_bytes, asset_lock_bytes, retain, *, frontend_config_bytes=None):
        require(runtime.model_id in (QWEN, SENSE), "unsupported native runtime")
        require(type(source_lock_bytes) is bytes and source_lock_bytes and
                type(asset_lock_bytes) is bytes and asset_lock_bytes, "exact retained native lock bytes required")
        self.runtime, self.model_root, self.retain = runtime, model_root, retain
        self.attempts = set()
        self.frontend_config_bytes = frontend_config_bytes
        if runtime.model_id == QWEN:
            self.adapter = QWEN_ADAPTER
            self.resolved, _ = runtime.wrapper.model.thinker._prepare_generation_config(
                None, max_new_tokens=256, do_sample=False, eos_token_id=[151645, 151643],
                return_dict_in_generate=True)
            # Empty custom lists are the frozen request; the actual native
            # normalized lists MUST independently be observed in every call.
            effective = qwen_effective(self.resolved, execution(runtime.wrapper.model, runtime.declaration), [], [])
            self.snapshot = generation_snapshot(self.resolved)
            require(retain("native-resolved-generation-config.json", self.snapshot) == canonical_sha256(self.snapshot),
                    "retained Qwen snapshot must use canonical UTF-8 JSON bytes")
        else:
            self.adapter = SENSE_ADAPTER
            effective, measured = sense_effective(runtime, model_root)
            # The caller first saves this measured snapshot, registers it in the
            # model inventory as a runtime-validation-config asset, and freezes
            # that lock. Consume its actual bytes, never call it a model weight.
            require(type(frontend_config_bytes) is bytes and bool(frontend_config_bytes),
                    "retained SenseVoice runtime-validation-config bytes required")
            self.snapshot = json.loads(frontend_config_bytes)
            require(hashlib.sha256(frontend_config_bytes).hexdigest() == canonical_sha256(self.snapshot),
                    "retained SenseVoice snapshot must use canonical UTF-8 JSON bytes")
            same(measured, self.snapshot, "SenseVoice loaded runtime-validation-config differs from live frontend")
        self.config = {"schema_version": 2, "evidence_class": CONFIG_CLASS, "adapter_id": self.adapter,
            "adapter_version": 1, "model_family": ADAPTER_FAMILIES[self.adapter],
            "source_lock_sha256": hashlib.sha256(source_lock_bytes).hexdigest(),
            "asset_lock_sha256": hashlib.sha256(asset_lock_bytes).hexdigest(),
            "conditioning_policy": CONDITIONING_POLICY, "execution": clone(effective["execution"]),
            "input_policy": {"sample_rate": 16000, "channels": 1, "max_audio_frames": 192000,
                "pcm_format": "float32_pcm16_scale_32768", "preprocessing": "full_input"},
            "native_request": clone(effective["native_request"]), "effective_config": effective,
            "effective_config_sha256": canonical_sha256(effective)}
        validate_native_config(self.config)
        require(retain("native-config.json", self.config) == canonical_sha256(self.config),
                "retained native config must use canonical UTF-8 JSON bytes")

    def _qwen(self, bound, decoder_sha, observed):
        from asr_stage.adapters import infer_qwen_once_after_approval
        wrapper, thinker = self.runtime.wrapper, self.runtime.wrapper.model.thinker
        def prepared(original, args, kwargs):
            require("prepared" not in observed, "exactly one native generation preparation")
            result = original(*args, **kwargs)
            actual = generation_snapshot(result[0])
            same(actual, self.snapshot, "Qwen resolved generation config drift")
            observed["prepared"] = actual
            observed["generation_mode"] = result[0].get_generation_mode().value
            return result
        def generation_return(original, args, kwargs):
            require("generation_returned" not in observed, "exactly one Qwen generation return")
            output = original(*args, **kwargs)
            observed["generation_returned"] = True
            sequence = output.sequences
            shape = list(sequence.shape)
            require(len(shape) == 2 and shape[0] == 1, "Qwen returned sequence shape")
            # Persist before retained batch_decode/parsing/PCM-postcheck. A
            # malformed huge result keeps a bounded prefix, explicitly marked.
            prefix = kwargs["input_ids"].shape[1]
            ids = sequence[0, :65536].tolist()
            self.retain(bound.opaque_id + ".native-generation-return.json", {
                "sequence_shape": shape, "prompt_token_count": prefix,
                "returned_sequence_ids": ids, "capture_truncated": shape[1] > 65536})
            require(shape[1] <= 65536, "Qwen returned sequence exceeds evidence cap")
            observed["returned_generated_token_ids"] = ids[prefix:]
            return output
        def custom(name, key):
            def capture(original, args, kwargs):
                require(key not in observed, "exactly one Qwen custom-control observation")
                arguments = inspect.signature(original).bind(*args, **kwargs).arguments
                require(name in arguments and arguments[name] is not None,
                        "missing observed Qwen normalized custom controls")
                values = arguments[name]
                require(len(values) == 0, "unreviewed Qwen custom controls")
                active = arguments["generation_config"]
                active_snapshot = generation_snapshot(active)
                # Native generate may resolve max_length from prompt length;
                # retain that actual config too, without changing the pre-audio
                # GenerationConfig identity or inventing inherited controls.
                for field in (*QWEN_CONTROLS, "max_new_tokens", "do_sample", "eos_token_id"):
                    same(active_snapshot[field], self.snapshot[field], "Qwen active control drift: " + field)
                require(active.get_generation_mode().value == observed["generation_mode"], "Qwen active mode drift")
                observed[key + "_resolved_generation_config"] = active_snapshot
                observed[key] = []
                return original(*args, **kwargs)
            return capture
        processor = wrapper.processor
        original_feature = processor.feature_extractor
        processor.feature_extractor = FeatureObserver(original_feature, bound, observed)
        try:
            with ExitStack() as stack:
                stack.enter_context(observe_method(wrapper.model, "generate", generation_return))
                stack.enter_context(observe_method(thinker, "_prepare_generation_config", prepared))
                stack.enter_context(observe_method(thinker, "_get_logits_processor", custom("logits_processor", "custom_logits_processors")))
                stack.enter_context(observe_method(thinker, "_get_stopping_criteria", custom("stopping_criteria", "custom_stopping_criteria")))
                row, raw = infer_qwen_once_after_approval(self.runtime, bound, decoder_sha)
                observed["inference_returned"] = True
                self.retain(bound.opaque_id + ".native-returned-raw.json", raw)
        finally:
            processor.feature_extractor = original_feature
        require(all(key in observed for key in ("prepared", "input", "custom_logits_processors", "custom_stopping_criteria", "returned_generated_token_ids")),
                "missing native Qwen observations")
        same(raw["token_ids"], observed["returned_generated_token_ids"], "Qwen returned sequence/token evidence mismatch")
        effective = qwen_effective(self.resolved, execution(wrapper.model, self.runtime.declaration),
            observed["custom_logits_processors"], observed["custom_stopping_criteria"])
        require(observed["generation_mode"] == effective["generation_mode"], "Qwen observed generation mode drift")
        return row, raw, effective, qwen_termination(raw["token_ids"])

    def _sense(self, bound, decoder_sha, state, observed):
        from sense_adapter import infer_sense_once_after_approval
        from asr_stage.runtime_support import verify_pcm_array
        wrapper = self.runtime.wrapper
        effective, snapshot = sense_effective(self.runtime, self.model_root)
        loaded_snapshot = json.loads(self.frontend_config_bytes)
        same(loaded_snapshot, self.snapshot, "SenseVoice loaded runtime-validation-config drift")
        same(snapshot, loaded_snapshot, "SenseVoice frontend runtime drift")
        def inference(original, args, kwargs):
            require("request" not in observed and not args, "exactly one explicit SenseVoice inference")
            request = {key: kwargs[key] for key in SENSE_REQUEST}
            same(request, SENSE_REQUEST, "SenseVoice actual request drift")
            require(set(kwargs) == set(SENSE_REQUEST) | {"data_in", "key", "tokenizer", "frontend"}
                    and kwargs["key"] == [bound.opaque_id]
                    and kwargs["frontend"] is wrapper.kwargs["frontend"], "SenseVoice unreviewed inference fields")
            verify_pcm_array(kwargs["data_in"], bound.descriptor["float32_pcm"])
            observed["request"] = request
            result = original(*args, **kwargs)
            observed["inference_returned"] = True
            self.retain(bound.opaque_id + ".native-inference-return.json", {
                "representation": "native return serialized through ordinary JSON; tuple becomes list",
                "raw_result": json.loads(json.dumps(result, allow_nan=False))})
            return result
        def tokenizer_decode(original, args, kwargs):
            # CapturingTokenizer owns exact call-count/options checks. Capture
            # its first underlying decode input before SentencePiece can throw.
            if "tokenizer_input_ids" not in observed:
                require(len(args) == 1 and not kwargs, "SenseVoice raw tokenizer call")
                observed["tokenizer_input_ids"] = clone(args[0])
                self.retain(bound.opaque_id + ".native-tokenizer-input.json", {
                    "token_scope": "ctc_collapsed_nonblank_ids_not_framewise_logits",
                    "raw_token_ids": observed["tokenizer_input_ids"]})
            return original(*args, **kwargs)
        def frontend(original, args, kwargs):
            require("input" not in observed, "exactly one SenseVoice full-input frontend call")
            arguments = inspect.signature(original).bind(*args, **kwargs).arguments
            frames = bound.descriptor["raw_pcm16"]["frame_count"]
            lengths = arguments["input_lengths"]
            # Pinned extract_fbank passes Python data_len lists; accept a tensor
            # only by reading its actual values, never inventing absent lengths.
            lengths = lengths if type(lengths) is list else lengths.tolist()
            same(lengths, [frames], "SenseVoice frontend length observation drift")
            require(tuple(arguments["input"].shape) == (1, frames), "SenseVoice frontend lost source frames")
            observed["input"] = full_input(bound)
            return original(*args, **kwargs)
        with observe_method(wrapper.model, "inference", inference), \
             observe_method(wrapper.kwargs["frontend"], "forward", frontend), \
             observe_method(wrapper.kwargs["tokenizer"], "decode", tokenizer_decode):
            row, raw = infer_sense_once_after_approval(self.runtime, bound, decoder_sha, self.model_root, state)
            self.retain(bound.opaque_id + ".native-returned-raw.json", json.loads(json.dumps(raw, allow_nan=False)))
        require(observed.get("inference_returned") is True and "input" in observed,
                "SenseVoice full-input inference did not return")
        same(raw["token_binding"]["raw_token_ids"], observed["tokenizer_input_ids"], "SenseVoice returned token evidence mismatch")
        effective["native_request"] = observed["request"]
        return row, raw, effective, sense_termination(raw)

    def infer(self, bound, decoder_manifest_sha256, state):
        require(bound.opaque_id not in self.attempts, "native duplicate clip attempt")
        self.attempts.add(bound.opaque_id)  # Consume before validation or forward; never retry.
        observed = {"schema": "source-screen-native-v2-observation-v1", "opaque_id": bound.opaque_id,
            "input_binding_sha256": bound.descriptor_sha256, "inference_returned": False,
            "acoustic_verbatim_completeness": "unknown"}
        try:
            result = self._qwen(bound, decoder_manifest_sha256, observed) if self.adapter == QWEN_ADAPTER else \
                self._sense(bound, decoder_manifest_sha256, state, observed)
            row, raw, effective, termination = result
            same(effective, self.config["effective_config"], "native actual effective configuration drift")
            # For Qwen this means the retained infer function returned after the
            # generated sequence and immutable PCM postcheck, never OOM/timeout.
            observed["inference_returned"] = True
            capture = {"schema_version": 2, "evidence_class": CAPTURE_CLASS,
                "native_config_sha256": canonical_sha256(self.config),
                "source_lock_sha256": self.config["source_lock_sha256"],
                "asset_lock_sha256": self.config["asset_lock_sha256"], "effective_config": effective,
                "effective_config_sha256": canonical_sha256(effective), "input": observed["input"],
                "termination": termination, "transcript_postcorrection_applied": False,
                "target_context_received": False, "peer_transcript_received": False,
                "acoustic_verbatim_completeness": "unknown"}
            valid = True
            try:
                validate_native_capture(self.config, capture)
            except ValueError:
                valid = False
            raw = dict(raw, native_capture=capture, native_v2_validated=valid, native_v2_observation=clone(observed))
            # Preserve legacy transcript/completeness values as historical API
            # semantics. Admission MUST use native_v2_validated plus real bytes.
            return row, raw
        except Exception as exc:
            observed["exception_type"] = type(exc).__name__
            raise
        finally:
            self.retain(bound.opaque_id + ".native-observation.json", observed)
