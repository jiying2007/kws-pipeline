"""Offline worker. Launch ONLY through pilot.py's bounded process-tree supervisor."""
from __future__ import annotations

import argparse
import functools
import importlib
import os
from pathlib import Path
import random
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_common import HERE, load_config, require, sha256, write_json, read_json


def offline_guard():
    attempts = []
    def blocked(*args, **kwargs):
        attempts.append("network connection blocked")
        raise RuntimeError("offline worker forbids network")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    def audit(name, args):
        if name in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"):
            blocked()
    sys.addaudithook(audit)
    return attempts


def event(path, phase, clip_id=None):
    write_json(path, {"phase": phase, "clip_id": clip_id, "monotonic": time.monotonic()})


def cpu_environment():
    import torch
    import onnxruntime as ort
    ort.disable_telemetry_events()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU not hidden")
    require(not torch.cuda.is_available(), "CUDA unexpectedly available")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, "Torch thread gate failed")
    # Both ORT pools must be one; upstream sets only intra_op_num_threads.
    original_options = ort.SessionOptions
    def options():
        value = original_options()
        value.intra_op_num_threads = 1
        value.inter_op_num_threads = 1
        return value
    ort.SessionOptions = options
    return torch, ort


def qualify(args, network_attempts):
    torch, ort = cpu_environment()
    from soundfile_adapter import qualify_adapter
    # A dependency-only qualification may import the classes, never construct
    # weights, ORT sessions, or a CosyVoice model.
    def forbidden(*args, **kwargs):
        raise RuntimeError("model/session construction forbidden in source qualification")
    old_session, old_load = ort.InferenceSession, torch.load
    ort.InferenceSession, torch.load = forbidden, forbidden
    result = qualify_adapter(args.source, Path(args.scratch) / "fixtures")
    manifest = read_json(HERE / "source-allowlist.json")
    for row in manifest["files"]:
        if not row["path"].endswith(".py"):
            continue
        module = row["path"][:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        importlib.import_module(module)
    require(not network_attempts, "qualification attempted network")
    require("ttsfrd" not in sys.modules and "wetext" not in sys.modules, "unexpected text frontend")
    ort.InferenceSession, torch.load = old_session, old_load
    write_json(args.result, {"status": "qualified", "adapter": result, "network_attempts": network_attempts, "torch_threads": 4, "ort_threads": 1, "model_constructed": False})
    event(args.state, "complete")


def strict_local_loaders(model_root):
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    expected = (Path(model_root) / "CosyVoice-BlankEN").resolve()
    for cls in (AutoTokenizer, Qwen2ForCausalLM):
        original = cls.from_pretrained
        def factory(original):
            def local_only(path, *args, **kwargs):
                require(Path(path).resolve() == expected, "unexpected pretrained path")
                require(not kwargs.get("trust_remote_code", False), "remote code forbidden")
                kwargs.update(local_files_only=True, trust_remote_code=False)
                return original(path, *args, **kwargs)
            return local_only
        cls.from_pretrained = factory(original)


def token_contract(frontend, cfg):
    """Validate plain text only; no synthesis or pronunciation-token variants."""
    inner = frontend.tokenizer.tokenizer
    records = []
    for row in cfg["rows"]:
        text = row["intended_text"]
        require(frontend.text_normalize(text, split=True, text_frontend=False) == [text], "frontend changed frozen text")
        ids = frontend.tokenizer.encode(text, allowed_special="all")
        actual, length = frontend._extract_text_token(text)
        require(actual.cpu().tolist() == [ids] and length.item() == len(ids), "frontend tokenization changed")
        require(inner.unk_token_id is None or inner.unk_token_id not in ids, "unknown whole-text token")
        records.append(dict(row, text=text, token_ids=ids))
    require(len(records) == 30, "fixed30 token count changed")
    return {"mode":"plain_only_no_inpainting", "clips":records}


def assert_cpu_model(model, torch, ort):
    require(model.__class__.__name__ == "CosyVoice3" and model.sample_rate == 24000 and not model.fp16, "unexpected model/precision/rate")
    require(str(model.frontend.device) == "cpu" and str(model.model.device) == "cpu", "non-CPU device")
    for network in (model.model.llm, model.model.flow, model.model.hift):
        for tensor in list(network.parameters()) + list(network.buffers()):
            require(tensor.device.type == "cpu", "non-CPU tensor")
            if tensor.is_floating_point():
                require(tensor.dtype == torch.float32, "non-FP32 model tensor")
    for session in (model.frontend.campplus_session, model.frontend.speech_tokenizer_session):
        require(session.get_providers() == ["CPUExecutionProvider"], "non-CPU ORT provider")
        opt = session.get_session_options()
        require(opt.intra_op_num_threads == 1 and opt.inter_op_num_threads == 1, "ORT thread mismatch")
    sampling = model.model.llm.sampling
    require(isinstance(sampling, functools.partial), "unexpected sampling callable")
    require(sampling.func.__name__ == "ras_sampling" and sampling.keywords == {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1}, "sampling contract mismatch")


def generate(args, network_attempts):
    cfg = load_config()
    torch, ort = cpu_environment()
    import numpy as np
    import soundfile as sf
    import torchaudio
    from soundfile_adapter import bind_frontend
    bind_frontend(args.source)  # BEFORE importing the model/front-end importer.
    strict_local_loaders(args.model)
    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=str(Path(args.model).resolve()), load_trt=False, load_vllm=False, fp16=False)
    assert_cpu_model(model, torch, ort)
    require(model.frontend.allowed_special == "all", "allowed_special drift")
    require(model.frontend.spk2info == {}, "unexpected saved speaker data")
    tokens = token_contract(model.frontend, cfg)
    write_json(Path(args.output) / "token-contract.json", tokens)
    references = {r["stock_voice"]: r for r in cfg["references"]}
    prompt_id = None
    write_json(Path(args.output) / "startup.json", {"status":"qualified", "sample_rate":model.sample_rate,
        "cpu":True, "fp32":True, "torch_threads":4, "torch_interop_threads":1, "ort_threads":1,
        "prompt_cache":"one official add_zero_shot_spk in memory per fixed reference", "reference_sha256":{r["stock_voice"]:r["sha256"] for r in cfg["references"]}, "network_attempts":network_attempts})
    outcomes = []
    with torch.inference_mode():
        for row in tokens["clips"]:
            clip = row["clip_id"]
            reference = references[row["stock_reference_voice"]]
            selected_prompt_id = "fixed30-" + row["stock_reference_voice"].lower()
            if prompt_id != selected_prompt_id:
                # Release old prompt tensors; only one voice's prompt is retained.
                # Reference extraction is not a TTS synthesis/warmup call.
                model.frontend.spk2info.clear()
                reference_path = str(Path(args.reference) / reference["reference_file"])
                require(model.add_zero_shot_spk(reference["prompt_text"], reference_path, selected_prompt_id) is True, "fixed reference feature extraction failed")
                prompt_id = selected_prompt_id
                require(set(model.frontend.spk2info) == {prompt_id}, "prompt cache scope changed")
            # Each frozen row gets one claim/call. First Eric K1 is the included
            # technical smoke; no warmup, failed-output replacement or retry.
            with open(Path(args.output) / (clip + ".attempt.json"), "x", encoding="utf-8") as f:
                import json
                json.dump({"clip_id": clip, "seed": row["seed"], "attempt": 1}, f)
            event(args.state, "clip", clip)
            random.seed(row["seed"])
            np.random.seed(row["seed"])
            torch.manual_seed(row["seed"])
            started = time.monotonic()
            chunks = []
            frames = 0
            for result in model.inference_zero_shot(row["text"], reference["prompt_text"], reference_path, zero_shot_spk_id=prompt_id, stream=False, speed=1.0, text_frontend=False):
                wave = result["tts_speech"].detach().cpu()
                require(wave.dtype == torch.float32 and wave.ndim == 2 and wave.shape[0] == 1, "invalid native output shape/dtype")
                frames += wave.shape[1]
                require(frames <= 20 * model.sample_rate, "output exceeds 20 seconds; no trimming")
                require(torch.isfinite(wave).all().item(), "nonfinite audio")
                chunks.append(wave)
            require(len(chunks) == 1 and frames > 0, "non-streaming output must be one complete nonempty waveform")
            wave = chunks[0]
            native = wave.numpy()[0]
            peak = float(np.max(np.abs(native)))
            rms = float(np.sqrt(np.mean(native.astype(np.float64) ** 2)))
            # Preserve raw bytes before rejecting clipping; never normalize.
            np.save(Path(args.output) / (clip + ".native.npy"), native, allow_pickle=False)
            require(peak <= 1.0 and rms > 0, "clipping-range/silence gate failed; raw waveform retained")
            sf.write(Path(args.output) / (clip + ".wav"), native, 24000, subtype="PCM_16")
            pcm16 = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(wave).numpy()[0]
            require(np.isfinite(pcm16).all() and float(np.max(np.abs(pcm16))) <= 1.0, "resampled output invalid; no normalization")
            sf.write(Path(args.output) / (clip + ".16k.wav"), pcm16, 16000, subtype="PCM_16")
            files = {suffix: sha256(Path(args.output) / (clip + suffix)) for suffix in (".native.npy", ".wav", ".16k.wav")}
            outcome = dict(row, frames=frames, frames24k=frames, api_return_completed=True, eos_proved=False, sample_rate=24000, duration_seconds=frames / 24000, peak=peak, rms=rms, elapsed_seconds=time.monotonic() - started, files=files, status="generated")
            outcomes.append(outcome)
            write_json(Path(args.output) / (clip + ".result.json"), outcome)
            write_json(Path(args.output) / "outcomes.json", outcomes)
    require(len(outcomes) == 30 and not network_attempts, "incomplete pilot or network attempt")
    write_json(args.result, {"status": "complete", "generated_clips": 30, "network_attempts": network_attempts, "no_training_admission": True})
    event(args.state, "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("qualify", "generate"))
    for name in ("source", "scratch", "result", "state", "output", "model", "reference"):
        parser.add_argument("--" + name, required=name in ("source", "scratch", "result", "state"))
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    event(args.state, "startup")
    network_attempts = offline_guard()
    try:
        (qualify if args.mode == "qualify" else generate)(args, network_attempts)
    except Exception as e:
        write_json(args.result, {"status": "failed", "error_type": type(e).__name__, "error": str(e)[:2000], "network_attempts": network_attempts})
        raise


if __name__ == "__main__":
    main()
