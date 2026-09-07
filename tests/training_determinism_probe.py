#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import struct
import subprocess
import sys
import wave

# Match training/sitecustomize.py before torch import so this diagnostic process
# exercises the same CPU-dispatch/thread contract as training/train_ctc.py.
_DETERMINISTIC_CPU_ENV = {
    "OMP_NUM_THREADS": "2",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "2",
    "MKL_CBWR": "AVX2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "ATEN_CPU_CAPABILITY": "avx2",
}
for _name, _value in _DETERMINISTIC_CPU_ENV.items():
    os.environ[_name] = _value

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import train_ctc as tc  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.numpy().tobytes()


def tensor_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if tensor is None:
            digest.update(b"<none>")
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_payload(value))
        digest.update(b"\0")
    return digest.hexdigest()


def model_digest(model: nn.Module) -> str:
    return tensor_digest(sorted(model.state_dict().items()))


def gradient_digest(model: nn.Module) -> str:
    return tensor_digest((name, parameter.grad) for name, parameter in sorted(model.named_parameters()))


def cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def make_pcm(seed: int, index: int, frames: int = 24000) -> bytes:
    state = (seed ^ ((index + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    samples: list[int] = []
    period = 53 + (index % 11) * 7
    for position in range(frames):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        noise = ((state >> 16) & 0x7FFF) - 16384
        square = 4200 if ((position // period) & 1) == 0 else -4200
        saw = ((position * (index + 3)) % 2048) - 1024
        sample = max(-32768, min(32767, noise // 3 + square + saw))
        samples.append(sample)
    return struct.pack(f"<{len(samples)}h", *samples)


def write_wav(path: pathlib.Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm)


def build_fixed_corpus(work: pathlib.Path) -> pathlib.Path:
    pcm_dir = work / "pcm"
    rows: list[str] = []
    patterns = (
        "1 2 3 4",
        "3 4 3 4",
        "2 1 3 4",
        "",
    )
    for index in range(32):
        path = pcm_dir / f"example-{index:02d}.wav"
        write_wav(path, make_pcm(0xC0DEC0DE, index))
        rows.append(f"pcm/{path.name}\t{patterns[index % len(patterns)]}")
    manifest = work / "train.tsv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def stable_file_tree_digest(root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_train(manifest: pathlib.Path, output: pathlib.Path) -> dict:
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(manifest),
        "--tokens",
        str(ROOT / "keywords" / "tokens.example.txt"),
        "--keywords",
        str(ROOT / "keywords" / "zh_cn_example.tsv"),
        "--frontend",
        "logmel",
        "--feature-dim",
        "32",
        "--hidden-dim",
        "64",
        "--epochs",
        "8",
        "--batch-size",
        "16",
        "--lr",
        "0.001",
        "--seed",
        "1337",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    return {
        "command": command,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "tensor_digest": tensor_digest(sorted(checkpoint["state_dict"].items())),
        "training_corpus_sha256": checkpoint["training_corpus_identity"]["corpus_sha256"],
        "training_environment": checkpoint["training_environment"],
    }


def one_step_probe(manifest: pathlib.Path) -> dict:
    seed = 1337
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(seed)

    token_map = tc.load_tokens(ROOT / "keywords" / "tokens.example.txt")
    keyword_sequences = tc.load_keyword_sequences(
        ROOT / "keywords" / "zh_cn_example.tsv", token_map
    )
    vocab_size_value = tc.vocab_size(token_map)
    dataset = tc.Manifest([manifest], 32, vocab_size_value, "logmel")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=tc.collate,
        generator=generator,
    )
    model = tc.TinyStreamingRNN(32, 64, vocab_size_value)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=0.001, weight_decay=tc.WEIGHT_DECAY
    )
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    x, y, xlen, ylen = next(iter(loader))

    initial = model_digest(model)
    batch = tensor_digest(
        (("x", x), ("y", y), ("xlen", xlen), ("ylen", ylen))
    )
    log_probs = model(x).log_softmax(dim=2)
    forward = tensor_digest((("log_probs", log_probs),))
    raw_ctc = loss_fn(log_probs, y, xlen, ylen)
    sample_weights = torch.where(
        ylen > 0,
        torch.full_like(ylen, tc.POSITIVE_EXAMPLE_WEIGHT, dtype=torch.float32),
        torch.ones_like(ylen, dtype=torch.float32),
    )
    normalized_ctc = raw_ctc / xlen.to(dtype=raw_ctc.dtype).clamp_min(1.0)
    ctc_loss = (normalized_ctc * sample_weights).sum() / sample_weights.sum()
    ordered_loss, _, _ = tc.ordered_token_loss(log_probs, y, xlen, ylen)
    margin_per_sample = tc.keyword_sequence_margin_loss(
        log_probs=log_probs,
        targets=y,
        input_lengths=xlen,
        target_lengths=ylen,
        true_ctc_nll=raw_ctc,
        keyword_sequences=keyword_sequences,
        blank=0,
        margin=tc.KEYWORD_SEQUENCE_MARGIN,
    )
    margin_loss = (margin_per_sample * sample_weights).sum() / sample_weights.sum()
    release_loss = tc.recurrent_release_loss(log_probs, xlen)
    loss = (
        ctc_loss
        + tc.ORDERED_TOKEN_LOSS_WEIGHT * ordered_loss
        + tc.KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT * margin_loss
        + tc.RECURRENT_RELEASE_LOSS_WEIGHT * release_loss
    )
    losses = tensor_digest(
        (
            ("raw_ctc", raw_ctc),
            ("sample_weights", sample_weights),
            ("ctc_loss", ctc_loss),
            ("ordered_loss", ordered_loss),
            ("margin_per_sample", margin_per_sample),
            ("margin_loss", margin_loss),
            ("release_loss", release_loss),
            ("loss", loss),
        )
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = gradient_digest(model)
    torch.nn.utils.clip_grad_norm_(trainable, tc.GRAD_CLIP_NORM)
    clipped_gradients = gradient_digest(model)
    optimizer.step()
    post_step = model_digest(model)
    return {
        "dataset_corpus_sha256": dataset.corpus_identity["corpus_sha256"],
        "batch_digest": batch,
        "initial_model_digest": initial,
        "forward_digest": forward,
        "loss_digest": losses,
        "gradient_digest": gradients,
        "clipped_gradient_digest": clipped_gradients,
        "post_step_model_digest": post_step,
    }


def torch_environment() -> dict:
    config = torch.__config__.show()
    parallel = torch.__config__.parallel_info()
    cpu_capability = None
    get_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if callable(get_capability):
        cpu_capability = get_capability()
    return {
        "cpu_model": cpu_model(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_config_sha256": _sha256_bytes(config.encode("utf-8")),
        "torch_parallel_info_sha256": _sha256_bytes(parallel.encode("utf-8")),
        "torch_cpu_capability": cpu_capability,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "env": {name: os.environ.get(name) for name in sorted(_DETERMINISTIC_CPU_ENV)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest = build_fixed_corpus(work)
    corpus_tree_sha256 = stable_file_tree_digest(work)
    training = run_train(manifest, work / "model.pt")
    one_step = one_step_probe(manifest)
    report = {
        "schema_version": 1,
        "repository_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "fixed_corpus_tree_sha256": corpus_tree_sha256,
        "environment": torch_environment(),
        "training": training,
        "one_step": one_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
