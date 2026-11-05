#!/usr/bin/env python3
"""One local CPU model process; one durable claim for each native decode.

No target phrases, reference transcripts, peer outputs, or online model APIs are
passed to any recognizer. Historical campaign launchers are not invoked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

from active_asr_batch import (BOUNDARY, CONFIGS, MODELS, ROOT, digest, existing_result,
                              model_identity, read_json, utc, verify_inputs, write_once)
from setup_asr_runtime import MODEL_LOCK, WHISPER_SHA, WHISPER_SOURCE, canonical, save, sha_file
import research_runtime


def verify_model_assets(model, runtime):
    if model == "whisper":
        if sha_file(runtime / "models/whisper/small.pt") != WHISPER_SHA:
            raise ValueError("Whisper checkpoint changed")
        source = read_json(WHISPER_SOURCE)
        for row in source["files"]:
            if sha_file(runtime / "whisper-source" / row["path"]) != row["sha256"]:
                raise ValueError("Whisper source changed: " + row["path"])
        return {"checkpoint_sha256": WHISPER_SHA, "source_revision": source["source_revision"]}
    spec = read_json(MODEL_LOCK)[model]
    for row in spec["asset_lock"]["files"]:
        path = runtime / "models" / model / row["filename"]
        if path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            raise ValueError("model asset changed: " + row["filename"])
    for row in spec["source_lock"]:
        distribution = importlib.metadata.distribution(row["distribution"])
        if sha_file(distribution.locate_file(row["relative_path"])) != row["sha256"]:
            raise ValueError("installed model implementation changed: " + row["relative_path"])
    return {"asset_lock_sha256": spec["asset_lock_sha256"], "source_lock_sha256": spec["source_lock_sha256"]}


def load_qwen(runtime, threads):
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from asr_stage.architecture import canonical_sha, qwen06_schema, validate_qwen_geometry
    from asr_stage.runtime_support import assert_effective_qwen_attention, describe_cpu_tensor, describe_storage
    from vendor import qwen_loading_gate_v2 as gate
    root = runtime / "models/qwen06"
    spec = read_json(MODEL_LOCK)["qwen06"]
    schema = qwen06_schema()
    validate_qwen_geometry(read_json(root / "config.json"))
    checkpoint_sha = next(row["sha256"] for row in spec["asset_lock"]["files"] if row["filename"] == "model.safetensors")
    evidence = gate.inspect_safetensors(root / "model.safetensors", schema, canonical_sha(schema), checkpoint_sha, approved_aliases=[])
    config = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    for item in (config, config.thinker_config, config.thinker_config.audio_config, config.thinker_config.text_config):
        item._attn_implementation = "eager"
        item.dtype = torch.float32
    model, loading = AutoModel.from_pretrained(str(root), config=config, dtype=torch.float32, device_map="cpu",
        attn_implementation="eager", local_files_only=True, trust_remote_code=False,
        output_loading_info=True, ignore_mismatched_sizes=False, use_safetensors=True, low_cpu_mem_usage=True)
    gate.validate_loading_info(loading)
    model.eval()
    verification = gate.verify_loaded_model(model, loading, evidence, lambda value: describe_cpu_tensor(value, True), describe_storage)
    attention = assert_effective_qwen_attention(model)
    processor = AutoProcessor.from_pretrained(str(root), fix_mistral_regex=True, local_files_only=True, trust_remote_code=False)
    wrapper = Qwen3ASRModel(backend="transformers", model=model, processor=processor, forced_aligner=None,
                           max_inference_batch_size=1, max_new_tokens=256)
    return wrapper, {"loading_info": loading, "loaded_state_verification": verification,
                     "effective_attention": attention, "dtype": "float32", "device": "cpu"}


def decode_qwen(wrapper, pcm, identifier):
    import torch
    from asr_stage.decoding import qwen_generation_options, qwen_transcript
    prompt = wrapper._build_text_prompt(context="", force_language=None)
    features = wrapper.processor(text=[prompt], audio=[pcm], sampling_rate=16000, return_tensors="pt", padding=True)
    features = features.to(wrapper.model.device).to(wrapper.model.dtype)
    features = {key: value.contiguous() for key, value in features.items()}
    with torch.inference_mode():
        output = wrapper.model.generate(**features, **qwen_generation_options())
    prefix = features["input_ids"].shape[1]
    sequence = output.sequences
    if tuple(sequence.shape[:1]) != (1,) or not torch.equal(sequence[:, :prefix], features["input_ids"]):
        raise ValueError("generated Qwen prefix/shape differs")
    ids = sequence[0, prefix:].tolist()
    decoded = wrapper.processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    full = wrapper.processor.batch_decode([ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    projection, metadata = qwen_transcript(decoded, ids)
    return {"text": projection["raw_text"], "native": {"token_ids": ids, "decoded_with_special_tokens": full,
            "decoded_skip_special_tokens": decoded, "metadata": metadata}, "quality_flags": projection["quality_flags"],
            "transcript_extraction": "remove only native language/asr marker prefix", "decoder_terminated": metadata["eos_terminated"]}


def load_sense(runtime, threads):
    import torch
    import yaml
    from funasr import AutoModel
    from funasr.utils import fbank
    from asr_stage.architecture import canonical_sha, sensevoice_schema, validate_sense_geometry
    from asr_stage.runtime_support import describe_cpu_tensor
    from vendor.checkpoint_guard.exact_apply import exact_apply
    from vendor.checkpoint_guard.checkpoint_contract import SENSEVOICE_METADATA_RECIPE
    if fbank._HAS_TORCHAUDIO or fbank._knf is None:
        raise ValueError("locked native-fbank backend required")
    root = runtime / "models/sensevoice"
    config = yaml.safe_load((root / "config.yaml").read_text())
    validate_sense_geometry(config)
    keys = ("model", "model_conf", "encoder", "encoder_conf", "tokenizer", "tokenizer_conf",
            "frontend", "frontend_conf", "specaug", "specaug_conf")
    kwargs = {key: config[key] for key in keys if key in config}
    kwargs["tokenizer_conf"] = dict(kwargs["tokenizer_conf"], bpemodel=str(root / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
    kwargs["frontend_conf"] = dict(kwargs["frontend_conf"], cmvn_file=str(root / "am.mvn"),
                                    dither=0.0, snip_edges=True, upsacle_samples=True)
    kwargs.update(model_path=str(root), init_param=None, device="cpu", ncpu=threads, batch_size=1,
                  seed=0, disable_update=True, disable_pbar=True, disable_log=True,
                  trust_remote_code=False, fp16=False, bf16=False)
    wrapper = AutoModel(**kwargs)
    checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    schema = sensevoice_schema()
    verification = exact_apply(wrapper.model, checkpoint, lambda value, *args: describe_cpu_tensor(value, *args),
        expected_schema=schema, expected_schema_sha256=canonical_sha(schema), metadata_recipe=SENSEVOICE_METADATA_RECIPE)
    del checkpoint
    wrapper.model.eval()
    return wrapper, {"loaded_state_verification": verification, "dtype": "float32", "device": "cpu",
                     "frontend": "kaldi-native-fbank", "dither": 0.0, "no_vad": True, "use_itn": False}


def decode_sense(wrapper, pcm, identifier):
    import torch
    from asr_stage.decoding import CapturingTokenizer
    capture = CapturingTokenizer(wrapper.kwargs["tokenizer"])
    with torch.inference_mode():
        output = wrapper.model.inference(data_in=pcm, key=[identifier], tokenizer=capture,
            frontend=wrapper.kwargs["frontend"], device="cpu", language="auto", use_itn=False,
            text_norm="woitn", fs=16000, data_type="sound", output_timestamp=False, ban_emo_unk=False)
    if not isinstance(output, tuple) or len(output) != 2 or len(output[0]) != 1 or output[0][0]["key"] != identifier:
        raise ValueError("unexpected SenseVoice native result shape")
    decoded = output[0][0]["text"]
    token_evidence = capture.evidence(decoded)
    text = re.sub(r"<\|[^<>]*\|>", "", decoded)
    return {"text": text, "native": {"result": output[0], "metadata": output[1], "token_evidence": token_evidence},
            "quality_flags": [] if text else ["empty_text"], "transcript_extraction": "remove native control tags only; no ITN or rich postprocess",
            "decoder_terminated": True}


def load_whisper(runtime, threads):
    sys.path.insert(0, str(runtime / "whisper-source"))
    import whisper
    model = whisper.load_model(str(runtime / "models/whisper/small.pt"), device="cpu", in_memory=False)
    model.eval()
    if model.dims.n_audio_state != 768 or model.dims.n_audio_ctx != 1500 or model.dims.n_mels != 80 or not model.is_multilingual:
        raise ValueError("unexpected native multilingual Whisper small architecture")
    return model, {"dimensions": asdict(model.dims), "checkpoint_sha256": WHISPER_SHA,
                   "native_source_revision": read_json(WHISPER_SOURCE)["source_revision"], "dtype": "float32", "device": "cpu"}


def decode_whisper(model, pcm, identifier):
    import torch
    import whisper
    from whisper.decoding import DecodingTask, GreedyDecoder
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(pcm), n_mels=model.dims.n_mels)
    options = whisper.DecodingOptions(**CONFIGS["whisper"])
    task = DecodingTask(model, options)
    if not isinstance(task.decoder, GreedyDecoder):
        raise ValueError("native greedy Whisper decoder required")
    trace = []
    original = task.decoder.update

    def observe(tokens, logits, sum_logprobs):
        values, completed = original(tokens, logits, sum_logprobs)
        trace.append({"token_ids": values[0].detach().cpu().tolist(), "completed": bool(completed)})
        return values, completed

    task.decoder.update = observe
    with torch.inference_mode():
        results = task.run(mel.unsqueeze(0))
    if len(results) != 1 or not trace:
        raise ValueError("one native Whisper result and a nonempty decode trace required")
    result = results[0]
    features = result.audio_features.detach().cpu().contiguous()
    complete = trace[-1]["completed"]
    native = {"text": result.text, "tokens": list(result.tokens), "language": result.language,
              "language_probs": result.language_probs, "avg_logprob": result.avg_logprob,
              "no_speech_prob": result.no_speech_prob, "temperature": result.temperature,
              "compression_ratio": result.compression_ratio, "effective_options": asdict(options),
              "pre_finalize_greedy_trace": trace, "decoder_terminated": complete,
              "audio_features": {"dtype": str(features.dtype), "shape": list(features.shape),
                                 "sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest()}}
    return {"text": result.text, "native": native, "quality_flags": [] if complete else ["incomplete"],
            "transcript_extraction": "unaltered native DecodingResult.text", "decoder_terminated": complete}


LOADERS = {"qwen06": load_qwen, "sensevoice": load_sense, "whisper": load_whisper}
DECODERS = {"qwen06": decode_qwen, "sensevoice": decode_sense, "whisper": decode_whisper}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--ids", required=True)
    args = parser.parse_args()
    research_runtime.check_interpreter(research_runtime.load_manifest())
    research_runtime.check_packages(research_runtime.load_manifest(), "asr")
    runtime, output = args.runtime.resolve(), args.output.resolve()
    manifest = verify_inputs(args.inputs)
    requested = args.ids.split(",")
    indexed = {clip["id"]: clip for clip in manifest["clips"]}
    if len(requested) != len(set(requested)) or not set(requested) <= set(indexed):
        raise ValueError("invalid selected audio IDs")
    identity = model_identity(args.model, runtime, args.inputs, args.threads)
    job = read_json(output / "job.json")
    pending, unresolved = [], []
    for identifier in requested:
        status, key = existing_result(output, args.model, indexed[identifier], identity)
        if status == "pending":
            pending.append((indexed[identifier], key))
        elif status != "complete":
            unresolved.append(identifier)
    record_path = output / "attempts" / args.attempt / (args.model + ".json")
    record = {"schema": "active-asr-worker-v1", "model": args.model, "attempt": args.attempt, "started_at": utc(),
              "identity": identity, "source_sha256": sha_file(__file__), "status": "running", "stage": "preflight",
              "selected": requested, "reused": len(requested) - len(pending) - len(unresolved),
              "unresolved_prior_calls": unresolved, "new_calls": 0, "boundary": BOUNDARY}
    save(record_path, record)
    if not pending:
        record["status"] = "complete" if not unresolved else "incomplete"
        save(record_path, record)
        return 0 if not unresolved else 1
    with (runtime / "heavy-model.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record["assets"] = verify_model_assets(args.model, runtime)
        env_keys = ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")
        for key in env_keys:
            path = runtime / "cache" / key.lower()
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")

        def network_guard(event, arguments):
            if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
                raise RuntimeError("model worker network access is disabled")

        sys.addaudithook(network_guard)
        record["network_boundary"] = "offline configuration and Python socket audit; not kernel isolation"
        sys.path.insert(0, str(ROOT / "research/qwen6_asr/core"))
        import numpy as np
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        record["stage"] = "load_model"
        save(record_path, record)
        loaded_at = time.monotonic()
        try:
            model, load_receipt = LOADERS[args.model](runtime, args.threads)
            save(output / "attempts" / args.attempt / (args.model + "-load.json"), load_receipt)
            record.update(stage="infer", load_seconds=round(time.monotonic() - loaded_at, 3),
                          effective_threads=torch.get_num_threads(), effective_interop_threads=torch.get_num_interop_threads())
            if torch.get_num_threads() > 4 or torch.get_num_interop_threads() > 4:
                raise ValueError("CPU thread budget exceeded")
            save(record_path, record)
            for clip, identity_sha in pending:
                remaining = job["deadline_unix"] - time.time()
                if remaining <= 0:
                    raise TimeoutError("persisted wall-clock budget exhausted")
                if len(list((output / "claims").glob("*/*.json"))) >= job["identity"]["max_calls"]:
                    raise ValueError("original recognition call budget exhausted")
                pcm_path = args.inputs.resolve().parent / clip["derived_pcm"]["path"]
                if sha_file(pcm_path) != clip["derived_pcm"]["sha256"]:
                    raise ValueError("input changed immediately before recognition")
                pcm = np.fromfile(pcm_path, dtype="<f4")
                if not np.isfinite(pcm).all() or len(pcm) != clip["derived_pcm"]["frames"]:
                    raise ValueError("invalid decoder PCM")
                claim = {"schema": "active-asr-call-claim-v1", "identity_sha256": identity_sha,
                         "model": args.model, "clip_id": clip["id"], "attempt": args.attempt,
                         "started_at": utc(), "input": clip, "model_identity": identity}
                write_once(output / "claims" / args.model / (clip["id"] + ".json"), claim)
                record["new_calls"] += 1
                record["current_clip"] = clip["id"]
                save(record_path, record)
                started = time.monotonic()

                def timed_out(signum, frame):
                    raise TimeoutError("bounded recognition deadline")

                signal.signal(signal.SIGALRM, timed_out)
                signal.alarm(max(1, int(min(remaining, 999 * 60))))
                try:
                    decoded = DECODERS[args.model](model, pcm, clip["id"])
                finally:
                    signal.alarm(0)
                raw = {"schema": "active-asr-native-output-v1", "identity_sha256": identity_sha,
                       "model": args.model, "clip_id": clip["id"], "native": decoded["native"]}
                write_once(output / "native" / args.model / (clip["id"] + ".json"), raw)
                result = {"schema": "active-asr-result-v1", "status": "complete", "identity_sha256": identity_sha,
                          "model": args.model, "clip_id": clip["id"], "attempt": args.attempt,
                          "input": clip, "model_identity": identity, "decoder_source_sha256": sha_file(__file__),
                          "text": decoded["text"], "quality_flags": decoded["quality_flags"],
                          "transcript_extraction": decoded["transcript_extraction"], "decoder_terminated": decoded["decoder_terminated"],
                          "native_output_sha256": sha_file(output / "native" / args.model / (clip["id"] + ".json")),
                          "elapsed_seconds": round(time.monotonic() - started, 6), "completed_at": utc(), "boundary": BOUNDARY}
                write_once(output / "results" / args.model / (clip["id"] + ".json"), result)
                print(json.dumps({"clip_id": clip["id"], "status": "complete", "seconds": result["elapsed_seconds"]}), flush=True)
            record.update(status="complete" if not unresolved else "incomplete", stage="complete", finished_at=utc(),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(record_path, record)
            return 0 if not unresolved else 1
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error), finished_at=utc(),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(record_path, record)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
