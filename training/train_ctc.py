#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import subprocess
import sys
import wave

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import corpus_digest, inspect_pcm16_wav  # noqa: E402
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

from frontend import features
from frontend_spec import FRONTEND_IDS, FRONTEND_LOGMEL, frontend_id
from model import TinyStreamingRNN

MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FRONTEND_SPEC_VERSION = 2
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 5.0
POSITIVE_EXAMPLE_WEIGHT = 2.0
ORDERED_TOKEN_LOSS_WEIGHT = 0.35
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: pathlib.Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def repository_sha() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def training_environment() -> dict:
    code_paths = [
        pathlib.Path(__file__).resolve(),
        ROOT / "training" / "frontend.py",
        ROOT / "training" / "frontend_spec.py",
        ROOT / "training" / "model.py",
        ROOT / "tools" / "corpus_identity.py",
    ]
    code = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in code_paths
        if path.is_file()
    }
    image_digest = os.environ.get("KWS_TRAINING_IMAGE_DIGEST")
    if image_digest is not None and IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise ValueError(
            "KWS_TRAINING_IMAGE_DIGEST must be sha256:<64 lowercase hex>"
        )
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": int(cudnn) if cudnn is not None else None,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "repository_sha": repository_sha(),
        "training_image_digest": image_digest,
        "container_declared": os.environ.get("KWS_TRAINING_CONTAINER") == "1",
        "requirements_lock_sha256": optional_sha256(
            ROOT / "training" / "requirements.lock"
        ),
        "dockerfile_sha256": optional_sha256(ROOT / "training" / "Dockerfile"),
        "training_code_sha256": code,
    }


def parse_token_ids(value, label: str) -> list[int]:
    if isinstance(value, str):
        try:
            return [int(item) for item in value.split()]
        except ValueError as exc:
            raise ValueError(f"{label}: token ids must be integers") from exc
    if isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return list(value)
    raise ValueError(f"{label}: expected token id string/list")


def manifest_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            audio = value.get("audio", value.get("path"))
            targets = value.get("tokens", value.get("target_ids"))
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError(f"{path}:{line_no}: audio/path must be non-empty")
            metadata = {}
            for field in IDENTITY_FIELDS:
                item = value.get(field)
                if item is not None:
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(f"{path}:{line_no}: {field} must be non-empty text")
                    metadata[field] = item.strip()
            rows.append(
                {
                    "audio": audio.strip(),
                    "tokens": parse_token_ids(targets, f"{path}:{line_no}"),
                    "metadata": metadata,
                }
            )
    else:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
            audio, token_text = raw.split("\t", 1)
            if not audio.strip():
                raise ValueError(f"{path}:{line_no}: empty WAV path")
            rows.append(
                {
                    "audio": audio.strip(),
                    "tokens": parse_token_ids(token_text, f"{path}:{line_no}"),
                    "metadata": {},
                }
            )
    return rows


class Manifest(Dataset):
    def __init__(
        self,
        paths: list[pathlib.Path],
        feature_dim: int,
        vocab_size_value: int,
        frontend: str,
    ):
        self.rows: list[tuple[pathlib.Path, list[int]]] = []
        self.identity_rows: list[dict] = []
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size_value
        self.frontend = frontend
        for manifest_index, path in enumerate(paths):
            root = path.parent
            for row_index, row in enumerate(manifest_rows(path), 1):
                raw_path = str(row["audio"])
                wav = pathlib.Path(raw_path)
                resolved = (wav if wav.is_absolute() else root / wav).resolve(strict=True)
                tokens = list(row["tokens"])
                if any(token <= 0 or token >= vocab_size_value for token in tokens):
                    raise ValueError(
                        f"{path}:{row_index}: targets must be in 1..{vocab_size_value - 1}"
                    )
                measured = inspect_pcm16_wav(resolved)
                identity = {
                    "recording": f"manifest-{manifest_index}:{row_index}",
                    "manifest": path.name,
                    "path": raw_path,
                    **measured,
                    **row["metadata"],
                }
                self.identity_rows.append(identity)
                self.rows.append((resolved, tokens))
        if not self.rows:
            raise ValueError("training manifests contain no examples")
        self.corpus_identity = {
            "schema_version": 1,
            "corpus_sha256": corpus_digest(self.identity_rows),
            "recordings": self.identity_rows,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, tokens = self.rows[idx]
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getframerate() != 16000
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
            raw = wf.readframes(wf.getnframes())
        pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        acoustic = features(
            pcm,
            self.feature_dim,
            frame_len=FRAME_LENGTH_SAMPLES,
            hop=FRAME_HOP_SAMPLES,
            frontend=self.frontend,
        )
        repeated_neighbors = sum(
            1 for left, right in zip(tokens, tokens[1:]) if left == right
        )
        minimum_ctc_steps = len(tokens) + repeated_neighbors
        if acoustic.shape[0] < minimum_ctc_steps:
            raise ValueError(
                f"{path}: {acoustic.shape[0]} acoustic step(s) cannot align "
                f"CTC target requiring at least {minimum_ctc_steps} step(s)"
            )
        return acoustic, torch.tensor(tokens, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    xlen = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    ylen = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    max_t = int(xlen.max())
    feature_dim = xs[0].shape[1]
    padded = torch.zeros((len(xs), max_t, feature_dim))
    for index, x in enumerate(xs):
        padded[index, : x.shape[0]] = x
    targets = (
        torch.cat(ys)
        if any(y.numel() for y in ys)
        else torch.empty(0, dtype=torch.long)
    )
    return padded, targets, xlen, ylen


def ordered_token_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Encourage each target occurrence to own a chronological region.

    CTC remains the primary sequence objective. This auxiliary term prevents
    blank-only/token-starvation collapse on mixed positive + empty-target
    batches while preserving CTC's freedom to choose the exact alignment.
    """
    losses: list[torch.Tensor] = []
    correct = 0
    total = 0
    offset = 0
    for batch_index, target_length in enumerate(target_lengths.tolist()):
        count = int(target_length)
        sample_targets = targets[offset : offset + count]
        offset += count
        if count == 0:
            continue
        steps = int(input_lengths[batch_index])
        if steps <= 0:
            raise ValueError("input length must be positive")
        sample_losses: list[torch.Tensor] = []
        for occurrence, token_tensor in enumerate(sample_targets):
            token = int(token_tensor)
            start = (occurrence * steps) // count
            stop = max(start + 1, ((occurrence + 1) * steps) // count)
            stop = min(stop, steps)
            region = log_probs[start:stop, batch_index, token]
            sample_losses.append(-region.max())
            best_frame = int(region.argmax()) + start
            predicted = int(log_probs[best_frame, batch_index].argmax())
            correct += int(predicted == token)
            total += 1
        losses.append(torch.stack(sample_losses).mean())
    if offset != int(targets.numel()):
        raise ValueError("flattened CTC targets do not match target lengths")
    if not losses:
        return log_probs.sum() * 0.0, correct, total
    return torch.stack(losses).mean(), correct, total


def validate_warm_start(
    checkpoint: dict,
    args: argparse.Namespace,
    vocab_size_value: int,
    fingerprint: int,
) -> None:
    expected = {
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "vocab_size": vocab_size_value,
        "frame_length_samples": FRAME_LENGTH_SAMPLES,
        "frame_hop_samples": FRAME_HOP_SAMPLES,
        "frontend_spec_version": FRONTEND_SPEC_VERSION,
        "vocab_fingerprint": fingerprint,
        "frontend_kind": frontend_id(args.frontend),
    }
    for key, value in expected.items():
        if int(checkpoint.get(key, -1)) != value:
            raise ValueError(
                f"warm-start {key}={checkpoint.get(key)!r} does not match {value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--frontend", choices=sorted(FRONTEND_IDS), default=FRONTEND_LOGMEL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warm-start", type=pathlib.Path)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--positive-example-weight", type=float, default=POSITIVE_EXAMPLE_WEIGHT)
    parser.add_argument("--ordered-token-loss-weight", type=float, default=ORDERED_TOKEN_LOSS_WEIGHT)
    parser.add_argument(
        "--require-container-digest",
        action="store_true",
        help="fail unless KWS_TRAINING_IMAGE_DIGEST is a pinned sha256 digest",
    )
    args = parser.parse_args()

    token_map = load_tokens(args.tokens)
    vocab_size_value = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    if not 2 <= vocab_size_value <= MAX_VOCAB_SIZE:
        parser.error(f"token vocabulary must contain 2..{MAX_VOCAB_SIZE} entries")
    if not 1 <= args.feature_dim <= MAX_FEATURE_DIM:
        parser.error(f"--feature-dim must be 1..{MAX_FEATURE_DIM}")
    if not 1 <= args.hidden_dim <= MAX_HIDDEN_DIM:
        parser.error(f"--hidden-dim must be 1..{MAX_HIDDEN_DIM}")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and > 0")
    if not math.isfinite(args.positive_example_weight) or args.positive_example_weight <= 0.0:
        parser.error("--positive-example-weight must be finite and > 0")
    if not math.isfinite(args.ordered_token_loss_weight) or args.ordered_token_loss_weight < 0.0:
        parser.error("--ordered-token-loss-weight must be finite and >= 0")
    if args.head_only and not args.warm_start:
        parser.error("--head-only requires --warm-start")

    environment = training_environment()
    if args.require_container_digest and environment["training_image_digest"] is None:
        parser.error(
            "--require-container-digest requires KWS_TRAINING_IMAGE_DIGEST=sha256:<digest>"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)

    dataset = Manifest(args.manifest, args.feature_dim, vocab_size_value, args.frontend)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=shuffle_generator,
    )
    model = TinyStreamingRNN(args.feature_dim, args.hidden_dim, vocab_size_value)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        validate_warm_start(checkpoint, args, vocab_size_value, fingerprint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.head_only:
        for parameter in model.in_proj.parameters():
            parameter.requires_grad = False
        for parameter in model.rec_proj.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        total_ctc = 0.0
        total_ordered = 0.0
        ordered_correct = 0
        ordered_total = 0
        for x, y, xlen, ylen in loader:
            log_probs = model(x).log_softmax(dim=2)
            raw_ctc = loss_fn(log_probs, y, xlen, ylen)
            sample_weights = torch.where(
                ylen > 0,
                torch.full_like(ylen, args.positive_example_weight, dtype=torch.float32),
                torch.ones_like(ylen, dtype=torch.float32),
            )
            normalized_ctc = raw_ctc / xlen.to(dtype=raw_ctc.dtype).clamp_min(1.0)
            ctc_loss = (normalized_ctc * sample_weights).sum() / sample_weights.sum()
            ordered_loss, batch_correct, batch_total = ordered_token_loss(
                log_probs, y, xlen, ylen
            )
            loss = ctc_loss + args.ordered_token_loss_weight * ordered_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach())
            total_ctc += float(ctc_loss.detach())
            total_ordered += float(ordered_loss.detach())
            ordered_correct += batch_correct
            ordered_total += batch_total
        batches = max(1, len(loader))
        ordered_accuracy = ordered_correct / max(1, ordered_total)
        print(
            f"epoch={epoch + 1} loss={total / batches:.6f} "
            f"ctc={total_ctc / batches:.6f} ordered={total_ordered / batches:.6f} "
            f"ordered_token_acc={ordered_accuracy:.6f}"
        )

    manifest_metadata = [
        {"name": path.name, "sha256": sha256_file(path)} for path in args.manifest
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "vocab_size": vocab_size_value,
            "vocab_fingerprint": fingerprint,
            "tokens_sha256": sha256_file(args.tokens),
            "frame_length_samples": FRAME_LENGTH_SAMPLES,
            "frame_hop_samples": FRAME_HOP_SAMPLES,
            "frontend_spec_version": FRONTEND_SPEC_VERSION,
            "frontend_name": args.frontend,
            "frontend_kind": frontend_id(args.frontend),
            "training_examples": len(dataset),
            "training_manifests": manifest_metadata,
            "training_corpus_identity": dataset.corpus_identity,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "ctc_reduction": "per-frame-weighted",
            "positive_example_weight": args.positive_example_weight,
            "ordered_token_loss_weight": args.ordered_token_loss_weight,
            "hard_negative_capable": True,
            "training_environment": environment,
        },
        args.output,
    )
    print(
        f"saved {args.output}: examples={len(dataset)} vocab={vocab_size_value} "
        f"frontend={args.frontend} fingerprint=0x{fingerprint:016x} "
        f"corpus={dataset.corpus_identity['corpus_sha256']} "
        f"repo_sha={environment['repository_sha']} image={environment['training_image_digest']}"
    )


if __name__ == "__main__":
    main()
