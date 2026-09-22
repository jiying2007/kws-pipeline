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

from completion_loss import PREFIX_COMPLETION_TAIL_STEPS, strict_prefix_completion_loss
from frontend import features
from frontend_spec import FRONTEND_IDS, FRONTEND_LOGMEL, frontend_id
from model import TinyStreamingRNN
from sequence_margin import keyword_sequence_margin_loss
from synthetic_audio import UINT32_MAX

MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FRONTEND_SPEC_VERSION = 2
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 5.0
POSITIVE_EXAMPLE_WEIGHT = 2.0
WAKE_EXAMPLE_WEIGHT = 1.0
ORDERED_TOKEN_LOSS_WEIGHT = 0.35
KEYWORD_SEQUENCE_MARGIN = 0.05
KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT = 0.10
PREFIX_COMPLETION_LOSS_WEIGHT = 0.10
RECURRENT_RELEASE_TAIL_STEPS = 25
RECURRENT_RELEASE_WARMUP_STEPS = 8
RECURRENT_RELEASE_CONTEXT_STEPS = 4
RECURRENT_RELEASE_LOSS_WEIGHT = 0.05
SAMPLE_WEIGHT_NORMALIZATION_POLICY = "dataset-mean-sample-weight-v1"
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


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cpu_runtime_identity() -> dict:
    fallback_model = platform.processor().strip() or None
    model_name: str | None = None
    flags: list[str] = []
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for raw in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                key, sep, value = raw.partition(":")
                if not sep:
                    continue
                normalized = key.strip().lower()
                payload = value.strip()
                if normalized in {"model name", "hardware"} and model_name is None and payload:
                    model_name = payload
                elif normalized in {"flags", "features"} and not flags:
                    flags = sorted(set(payload.split()))
        except OSError:
            pass
    flags_text = " ".join(flags)
    return {
        "model": model_name or fallback_model,
        "logical_cpu_count": os.cpu_count(),
        "flags_sha256": _text_sha256(flags_text) if flags_text else None,
        "flag_count": len(flags),
    }


def _torch_runtime_identity() -> dict:
    config_text = str(torch.__config__.show())
    parallel_fn = getattr(torch.__config__, "parallel_info", None)
    parallel_text = str(parallel_fn()) if callable(parallel_fn) else ""
    mkl_backend = getattr(torch.backends, "mkl", None)
    openmp_backend = getattr(torch.backends, "openmp", None)
    mkldnn_backend = getattr(torch.backends, "mkldnn", None)
    return {
        "config_sha256": _text_sha256(config_text),
        "parallel_info_sha256": (
            _text_sha256(parallel_text) if parallel_text else None
        ),
        "mkl_available": (
            bool(mkl_backend.is_available())
            if mkl_backend is not None and hasattr(mkl_backend, "is_available")
            else None
        ),
        "openmp_available": (
            bool(openmp_backend.is_available())
            if openmp_backend is not None and hasattr(openmp_backend, "is_available")
            else None
        ),
        "mkldnn_available": (
            bool(mkldnn_backend.is_available())
            if mkldnn_backend is not None and hasattr(mkldnn_backend, "is_available")
            else None
        ),
        "mkldnn_enabled": (
            bool(mkldnn_backend.enabled)
            if mkldnn_backend is not None and hasattr(mkldnn_backend, "enabled")
            else None
        ),
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def training_environment() -> dict:
    code_paths = [
        pathlib.Path(__file__).resolve(),
        ROOT / "training" / "frontend.py",
        ROOT / "training" / "frontend_spec.py",
        ROOT / "training" / "model.py",
        ROOT / "training" / "sequence_margin.py",
        ROOT / "training" / "synthetic_audio.py",
        ROOT / "training" / "wake_pressure_balance.py",
        ROOT / "training" / "completion_loss.py",
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
        "cpu_runtime": _cpu_runtime_identity(),
        "torch_version": str(torch.__version__),
        "torch_runtime": _torch_runtime_identity(),
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
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    raise ValueError(f"{label}: expected token id string/list")


def _keyword_rows(path: pathlib.Path, token_map: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    seen_sequences: set[tuple[int, ...]] = set()
    seen_ids: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 4:
            raise ValueError(f"{path}:{line_no}: expected keyword TSV with token column")
        keyword_id = int(cols[0])
        if keyword_id < 0 or keyword_id > UINT32_MAX or keyword_id in seen_ids:
            raise ValueError(
                f"{path}:{line_no}: keyword id must be unique and fit uint32"
            )
        threshold = float(cols[2])
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: threshold must be finite and in (0,1)")
        names = cols[3].split()
        if not names:
            raise ValueError(f"{path}:{line_no}: keyword token sequence is empty")
        missing = [name for name in names if name not in token_map]
        if missing:
            raise ValueError(
                f"{path}:{line_no}: keyword tokens missing from vocabulary: {', '.join(missing)}"
            )
        sequence = [int(token_map[name]) for name in names]
        key = tuple(sequence)
        if key in seen_sequences:
            raise ValueError(f"{path}:{line_no}: duplicate keyword token sequence")
        if any(token <= 0 for token in sequence):
            raise ValueError(f"{path}:{line_no}: keyword sequence may not contain blank")
        seen_ids.add(keyword_id)
        seen_sequences.add(key)
        rows.append(
            {
                "id": keyword_id,
                "text": cols[1].strip(),
                "threshold": threshold,
                "sequence": sequence,
            }
        )
    if not rows:
        raise ValueError("keyword TSV contains no wake sequences")
    return rows


def load_keyword_sequences(path: pathlib.Path, token_map: dict[str, int]) -> list[list[int]]:
    return [list(row["sequence"]) for row in _keyword_rows(path, token_map)]


def keyword_margin_profile_path(keywords_path: pathlib.Path) -> pathlib.Path:
    return keywords_path.with_suffix(keywords_path.suffix + ".margin.json")


def load_keyword_operating_points(
    keywords_path: pathlib.Path,
    token_map: dict[str, int],
    *,
    default_margin: float = KEYWORD_SEQUENCE_MARGIN,
) -> tuple[list[list[int]], list[dict], pathlib.Path]:
    rows = _keyword_rows(keywords_path, token_map)
    profile_path = keyword_margin_profile_path(keywords_path)
    profile: dict = {}
    if profile_path.is_file():
        value = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
            raise ValueError("keyword margin profile must be schema_version 1")
        raw_keywords = value.get("keywords")
        if not isinstance(raw_keywords, dict):
            raise ValueError("keyword margin profile must contain keywords object")
        profile = raw_keywords
        known = {str(int(row["id"])) for row in rows}
        unknown = sorted(set(str(key) for key in profile) - known)
        if unknown:
            raise ValueError(
                "keyword margin profile contains unknown keyword id(s): "
                + ", ".join(unknown)
            )

    operating_points: list[dict] = []
    for row in rows:
        raw = profile.get(str(int(row["id"])), {})
        if not isinstance(raw, dict):
            raise ValueError("keyword margin profile entry must be an object")
        if "text" in raw and str(raw["text"]) != str(row["text"]):
            raise ValueError("keyword margin profile text does not match keyword TSV")
        positive_margin = float(raw.get("positive_margin", default_margin))
        negative_margin = float(raw.get("negative_margin", default_margin))
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (positive_margin, negative_margin)
        ):
            raise ValueError("keyword margins must be finite and non-negative")
        threshold = float(row["threshold"])
        if not 0.0 < threshold - negative_margin < threshold <= threshold + positive_margin < 1.0:
            raise ValueError("keyword margin profile produces an invalid operating band")
        operating_points.append(
            {
                "keyword_id": int(row["id"]),
                "text": str(row["text"]),
                "threshold": threshold,
                "positive_margin": positive_margin,
                "negative_margin": negative_margin,
            }
        )
    return (
        [list(row["sequence"]) for row in rows],
        operating_points,
        profile_path,
    )


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
    padded = torch.zeros(
        (len(xs), max_t + RECURRENT_RELEASE_TAIL_STEPS, feature_dim)
    )
    for index, x in enumerate(xs):
        steps = int(x.shape[0])
        padded[index, :steps] = x
        context_steps = min(RECURRENT_RELEASE_CONTEXT_STEPS, steps)
        context = x[steps - context_steps : steps]
        repeats = (RECURRENT_RELEASE_TAIL_STEPS + context_steps - 1) // context_steps
        release = context.repeat((repeats, 1))[:RECURRENT_RELEASE_TAIL_STEPS]
        padded[index, steps : steps + RECURRENT_RELEASE_TAIL_STEPS] = release
    targets = (
        torch.cat(ys)
        if any(y.numel() for y in ys)
        else torch.empty(0, dtype=torch.long)
    )
    return padded, targets, xlen, ylen


def parse_wake_keyword_weights(
    value: str,
    keyword_operating_points: list[dict],
) -> dict[int, float]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--wake-keyword-weights must be a JSON object") from exc
    if not isinstance(raw, dict):
        raise ValueError("--wake-keyword-weights must be a JSON object")
    valid_ids = {int(item["keyword_id"]) for item in keyword_operating_points}
    result: dict[int, float] = {}
    for raw_key, raw_value in raw.items():
        try:
            keyword_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("--wake-keyword-weights keys must be keyword ids") from exc
        if keyword_id not in valid_ids:
            raise ValueError(
                f"--wake-keyword-weights contains unknown keyword id {keyword_id}"
            )
        if isinstance(raw_value, bool):
            raise ValueError("--wake-keyword-weights values must be finite and > 0")
        try:
            weight = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--wake-keyword-weights values must be finite and > 0"
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("--wake-keyword-weights values must be finite and > 0")
        result[keyword_id] = weight
    return result


def wake_example_weights(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    keyword_sequences: list[list[int]],
    keyword_ids: list[int],
    *,
    default_weight: float,
    keyword_weights: dict[int, float],
) -> torch.Tensor:
    if len(keyword_sequences) != len(keyword_ids):
        raise ValueError("keyword ids must align with keyword sequences")
    sequence_to_id = {
        tuple(int(value) for value in sequence): int(keyword_id)
        for sequence, keyword_id in zip(keyword_sequences, keyword_ids)
    }
    result: list[float] = []
    offset = 0
    flat = targets.detach().cpu().tolist()
    for raw_length in target_lengths.detach().cpu().tolist():
        length = int(raw_length)
        row = tuple(int(value) for value in flat[offset : offset + length])
        keyword_id = sequence_to_id.get(row)
        result.append(
            float(keyword_weights.get(keyword_id, default_weight))
            if keyword_id is not None
            else 1.0
        )
        offset += length
    if offset != len(flat):
        raise ValueError("flattened CTC targets do not match target lengths")
    return torch.tensor(result, dtype=torch.float32, device=target_lengths.device)


def exact_keyword_sample_mask(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    keyword_sequences: list[list[int]],
) -> torch.Tensor:
    sequences = {
        tuple(int(value) for value in sequence) for sequence in keyword_sequences
    }
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("exact-keyword mask requires non-empty keyword sequences")
    flat = targets.detach().cpu().tolist()
    result: list[bool] = []
    offset = 0
    for raw_length in target_lengths.detach().cpu().tolist():
        length = int(raw_length)
        row = tuple(int(value) for value in flat[offset : offset + length])
        result.append(row in sequences)
        offset += length
    if offset != len(flat):
        raise ValueError("flattened CTC targets do not match target lengths")
    return torch.tensor(result, dtype=torch.bool, device=target_lengths.device)


def sample_weight_statistics(
    rows: list[tuple[pathlib.Path, list[int]]],
    keyword_sequences: list[list[int]],
    keyword_ids: list[int],
    *,
    positive_example_weight: float,
    default_wake_weight: float,
    keyword_weights: dict[int, float],
) -> dict:
    if len(keyword_sequences) != len(keyword_ids):
        raise ValueError("keyword ids must align with keyword sequences")
    if not rows:
        raise ValueError("sample-weight statistics require non-empty dataset rows")
    if (
        not math.isfinite(positive_example_weight)
        or positive_example_weight <= 0.0
        or not math.isfinite(default_wake_weight)
        or default_wake_weight <= 0.0
    ):
        raise ValueError("sample-weight statistics require finite positive weights")
    sequence_to_id = {
        tuple(int(value) for value in sequence): int(keyword_id)
        for sequence, keyword_id in zip(keyword_sequences, keyword_ids)
    }
    all_weight_sum = 0.0
    nonempty_weight_sum = 0.0
    nonempty_rows = 0
    exact_wake_rows = 0
    exact_wake_weight_sum = 0.0
    for _, raw_tokens in rows:
        tokens = tuple(int(value) for value in raw_tokens)
        weight = positive_example_weight if tokens else 1.0
        keyword_id = sequence_to_id.get(tokens)
        if keyword_id is not None:
            wake_weight = float(keyword_weights.get(keyword_id, default_wake_weight))
            if not math.isfinite(wake_weight) or wake_weight <= 0.0:
                raise ValueError("sample-weight statistics contain invalid wake weight")
            weight *= wake_weight
            exact_wake_rows += 1
            exact_wake_weight_sum += weight
        all_weight_sum += weight
        if tokens:
            nonempty_rows += 1
            nonempty_weight_sum += weight
    if nonempty_rows <= 0:
        raise ValueError("sample-weight statistics require at least one non-empty target")
    if exact_wake_rows <= 0:
        raise ValueError("sample-weight statistics require at least one exact wake target")
    row_count = len(rows)
    return {
        "schema_version": 1,
        "policy": SAMPLE_WEIGHT_NORMALIZATION_POLICY,
        "rows": row_count,
        "nonempty_rows": nonempty_rows,
        "exact_wake_rows": exact_wake_rows,
        "exact_wake_weight_sum": exact_wake_weight_sum,
        "exact_wake_mean_weight": exact_wake_weight_sum / float(exact_wake_rows),
        "all_weight_sum": all_weight_sum,
        "nonempty_weight_sum": nonempty_weight_sum,
        "all_mean_weight": all_weight_sum / float(row_count),
        "nonempty_mean_weight": nonempty_weight_sum / float(nonempty_rows),
    }


def normalized_weighted_mean(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    mean_weight: float,
) -> torch.Tensor:
    if values.ndim != 1 or sample_weights.ndim != 1 or values.shape != sample_weights.shape:
        raise ValueError("weighted mean values/weights must be aligned vectors")
    if values.numel() <= 0:
        raise ValueError("weighted mean requires at least one value")
    if not torch.isfinite(sample_weights).all() or bool((sample_weights <= 0).any()):
        raise ValueError("weighted mean sample weights must be finite and > 0")
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("weighted mean normalization must be finite and > 0")
    denominator = values.new_tensor(float(values.numel()) * mean_weight)
    return (values * sample_weights.to(dtype=values.dtype, device=values.device)).sum() / denominator


def ordered_token_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    normalization_mean_weight: float | None = None,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Encourage each target occurrence to own a chronological region."""
    if sample_weights is not None:
        if sample_weights.ndim != 1 or int(sample_weights.numel()) != int(target_lengths.numel()):
            raise ValueError("ordered-token sample weights must match batch size")
        if not torch.isfinite(sample_weights).all() or bool((sample_weights <= 0).any()):
            raise ValueError("ordered-token sample weights must be finite and > 0")
    if sample_mask is not None:
        if (
            sample_mask.ndim != 1
            or int(sample_mask.numel()) != int(target_lengths.numel())
            or sample_mask.dtype != torch.bool
        ):
            raise ValueError("ordered-token sample mask must be a boolean batch vector")
    if normalization_mean_weight is not None and sample_weights is None:
        raise ValueError("ordered-token normalization requires sample weights")
    if normalization_mean_weight is not None and (
        not math.isfinite(normalization_mean_weight) or normalization_mean_weight <= 0.0
    ):
        raise ValueError("ordered-token normalization must be finite and > 0")

    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    correct = 0
    total = 0
    offset = 0
    for batch_index, target_length in enumerate(target_lengths.tolist()):
        count = int(target_length)
        sample_targets = targets[offset : offset + count]
        offset += count
        if count == 0:
            continue
        if sample_mask is not None and not bool(sample_mask[batch_index]):
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
        weights.append(
            sample_weights[batch_index]
            if sample_weights is not None
            else log_probs.new_tensor(1.0)
        )
    if offset != int(targets.numel()):
        raise ValueError("flattened CTC targets do not match target lengths")
    if not losses:
        return log_probs.sum() * 0.0, correct, total
    loss_values = torch.stack(losses)
    weight_values = torch.stack(weights).to(dtype=loss_values.dtype, device=loss_values.device)
    if normalization_mean_weight is None:
        value = (loss_values * weight_values).sum() / weight_values.sum()
    else:
        value = normalized_weighted_mean(
            loss_values,
            weight_values,
            normalization_mean_weight,
        )
    return value, correct, total


def recurrent_release_loss(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    """Require recurrent acoustic memory to return to blank after an utterance."""
    losses: list[torch.Tensor] = []
    available_steps = int(log_probs.shape[0])
    for batch_index, input_length in enumerate(input_lengths.tolist()):
        steps = int(input_length)
        start = steps + RECURRENT_RELEASE_WARMUP_STEPS
        stop = steps + RECURRENT_RELEASE_TAIL_STEPS
        if steps <= 0 or start >= stop or stop > available_steps:
            raise ValueError("recurrent release tail does not fit model output")
        losses.append(-log_probs[start:stop, batch_index, 0].mean())
    if not losses:
        return log_probs.sum() * 0.0
    return torch.stack(losses).mean()


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
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
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
    parser.add_argument("--wake-example-weight", type=float, default=WAKE_EXAMPLE_WEIGHT)
    parser.add_argument(
        "--wake-keyword-weights",
        default="{}",
        help="JSON object mapping configured keyword id to exact-wake sample-weight override",
    )
    parser.add_argument("--ordered-token-loss-weight", type=float, default=ORDERED_TOKEN_LOSS_WEIGHT)
    parser.add_argument(
        "--ordered-token-exact-wake-only",
        action="store_true",
        help="apply the ordered-token auxiliary only to exact configured wake targets",
    )
    parser.add_argument(
        "--keyword-sequence-margin-loss-weight",
        type=float,
        default=KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--prefix-completion-loss-weight",
        type=float,
        default=PREFIX_COMPLETION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--recurrent-release-loss-weight",
        type=float,
        default=RECURRENT_RELEASE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--require-container-digest",
        action="store_true",
        help="fail unless KWS_TRAINING_IMAGE_DIGEST is a pinned sha256 digest",
    )
    args = parser.parse_args()

    token_map = load_tokens(args.tokens)
    keyword_sequences, keyword_operating_points, margin_profile_path = (
        load_keyword_operating_points(args.keywords, token_map)
    )
    keyword_ids = [int(item["keyword_id"]) for item in keyword_operating_points]
    wake_keyword_weights = parse_wake_keyword_weights(
        args.wake_keyword_weights,
        keyword_operating_points,
    )
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
    if not math.isfinite(args.wake_example_weight) or args.wake_example_weight <= 0.0:
        parser.error("--wake-example-weight must be finite and > 0")
    if not math.isfinite(args.ordered_token_loss_weight) or args.ordered_token_loss_weight < 0.0:
        parser.error("--ordered-token-loss-weight must be finite and >= 0")
    for name in (
        "keyword_sequence_margin_loss_weight",
        "prefix_completion_loss_weight",
        "recurrent_release_loss_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(KEYWORD_SEQUENCE_MARGIN) or KEYWORD_SEQUENCE_MARGIN <= 0.0:
        parser.error("keyword sequence margin must be finite and > 0")
    if (
        not math.isfinite(KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT)
        or KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT <= 0.0
    ):
        parser.error("keyword sequence margin loss weight must be finite and > 0")
    if not math.isfinite(PREFIX_COMPLETION_LOSS_WEIGHT) or PREFIX_COMPLETION_LOSS_WEIGHT <= 0.0:
        parser.error("prefix completion loss weight must be finite and > 0")
    if not 0 <= RECURRENT_RELEASE_WARMUP_STEPS < RECURRENT_RELEASE_TAIL_STEPS:
        parser.error("recurrent release warmup must be inside the release tail")
    if not 1 <= RECURRENT_RELEASE_CONTEXT_STEPS <= RECURRENT_RELEASE_TAIL_STEPS:
        parser.error("recurrent release context must fit inside the release tail")
    if not math.isfinite(RECURRENT_RELEASE_LOSS_WEIGHT) or RECURRENT_RELEASE_LOSS_WEIGHT <= 0.0:
        parser.error("recurrent release loss weight must be finite and > 0")
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
    weight_statistics = sample_weight_statistics(
        dataset.rows,
        keyword_sequences,
        keyword_ids,
        positive_example_weight=args.positive_example_weight,
        default_wake_weight=args.wake_example_weight,
        keyword_weights=wake_keyword_weights,
    )
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
    epoch_history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        total = 0.0
        total_ctc = 0.0
        total_ordered = 0.0
        total_margin = 0.0
        total_completion = 0.0
        total_release = 0.0
        ordered_correct = 0
        ordered_total = 0
        for x, y, xlen, ylen in loader:
            log_probs = model(x).log_softmax(dim=2)
            raw_ctc = loss_fn(log_probs, y, xlen, ylen)
            target_weights = torch.where(
                ylen > 0,
                torch.full_like(ylen, args.positive_example_weight, dtype=torch.float32),
                torch.ones_like(ylen, dtype=torch.float32),
            )
            wake_weights = wake_example_weights(
                y,
                ylen,
                keyword_sequences,
                keyword_ids,
                default_weight=args.wake_example_weight,
                keyword_weights=wake_keyword_weights,
            )
            sample_weights = target_weights * wake_weights
            normalized_ctc = raw_ctc / xlen.to(dtype=raw_ctc.dtype).clamp_min(1.0)
            ctc_loss = normalized_weighted_mean(
                normalized_ctc,
                sample_weights,
                float(weight_statistics["all_mean_weight"]),
            )
            ordered_mask = (
                exact_keyword_sample_mask(y, ylen, keyword_sequences)
                if args.ordered_token_exact_wake_only
                else None
            )
            ordered_loss, batch_correct, batch_total = ordered_token_loss(
                log_probs,
                y,
                xlen,
                ylen,
                sample_weights,
                normalization_mean_weight=float(
                    weight_statistics[
                        "exact_wake_mean_weight"
                        if args.ordered_token_exact_wake_only
                        else "nonempty_mean_weight"
                    ]
                ),
                sample_mask=ordered_mask,
            )
            margin_per_sample = keyword_sequence_margin_loss(
                log_probs=log_probs,
                targets=y,
                input_lengths=xlen,
                target_lengths=ylen,
                true_ctc_nll=raw_ctc,
                keyword_sequences=keyword_sequences,
                blank=0,
                margin=KEYWORD_SEQUENCE_MARGIN,
                keyword_operating_points=keyword_operating_points,
            )
            margin_loss = normalized_weighted_mean(
                margin_per_sample,
                sample_weights,
                float(weight_statistics["all_mean_weight"]),
            )
            completion_per_sample = strict_prefix_completion_loss(
                log_probs=log_probs,
                targets=y,
                input_lengths=xlen,
                target_lengths=ylen,
                keyword_sequences=keyword_sequences,
                keyword_operating_points=keyword_operating_points,
            )
            completion_loss = normalized_weighted_mean(
                completion_per_sample,
                sample_weights,
                float(weight_statistics["all_mean_weight"]),
            )
            release_loss = recurrent_release_loss(log_probs, xlen)
            loss = (
                ctc_loss
                + args.ordered_token_loss_weight * ordered_loss
                + args.keyword_sequence_margin_loss_weight * margin_loss
                + args.prefix_completion_loss_weight * completion_loss
                + args.recurrent_release_loss_weight * release_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach())
            total_ctc += float(ctc_loss.detach())
            total_ordered += float(ordered_loss.detach())
            total_margin += float(margin_loss.detach())
            total_completion += float(completion_loss.detach())
            total_release += float(release_loss.detach())
            ordered_correct += batch_correct
            ordered_total += batch_total
        batches = max(1, len(loader))
        ordered_accuracy = ordered_correct / max(1, ordered_total)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": total / batches,
            "ctc": total_ctc / batches,
            "ordered": total_ordered / batches,
            "margin": total_margin / batches,
            "completion": total_completion / batches,
            "release": total_release / batches,
            "ordered_token_accuracy": ordered_accuracy,
        }
        epoch_history.append(epoch_metrics)
        print(
            f"epoch={epoch + 1} loss={epoch_metrics['loss']:.6f} "
            f"ctc={epoch_metrics['ctc']:.6f} ordered={epoch_metrics['ordered']:.6f} "
            f"margin={epoch_metrics['margin']:.6f} completion={epoch_metrics['completion']:.6f} "
            f"release={epoch_metrics['release']:.6f} "
            f"ordered_token_acc={epoch_metrics['ordered_token_accuracy']:.6f}"
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
            "keywords_sha256": sha256_file(args.keywords),
            "keyword_sequences": keyword_sequences,
            "keyword_operating_points": keyword_operating_points,
            "keyword_margin_profile_sha256": optional_sha256(margin_profile_path),
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
            "epoch_history": epoch_history,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "ctc_reduction": "per-frame-weighted",
            "positive_example_weight": args.positive_example_weight,
            "positive_example_weight_semantics": "non-empty-target-v1",
            "wake_example_weight": args.wake_example_weight,
            "wake_keyword_weights": {
                str(keyword_id): float(weight)
                for keyword_id, weight in sorted(wake_keyword_weights.items())
            },
            "wake_example_weight_semantics": (
                "exact-configured-keyword-target-with-per-keyword-override-v2"
            ),
            "sample_weight_normalization": weight_statistics,
            "ordered_token_loss_weight": args.ordered_token_loss_weight,
            "ordered_token_sample_weighting": "training-sample-weights-v1",
            "ordered_token_scope": (
                "exact-configured-wake-targets-v1"
                if args.ordered_token_exact_wake_only
                else "all-nonempty-targets-v1"
            ),
            "keyword_sequence_margin": KEYWORD_SEQUENCE_MARGIN,
            "keyword_sequence_margin_loss_weight": args.keyword_sequence_margin_loss_weight,
            "prefix_completion_loss_weight": args.prefix_completion_loss_weight,
            "prefix_completion_tail_steps": PREFIX_COMPLETION_TAIL_STEPS,
            "prefix_completion_policy": "strict-prefix-terminal-hinge-v1",
            "recurrent_release_tail_steps": RECURRENT_RELEASE_TAIL_STEPS,
            "recurrent_release_warmup_steps": RECURRENT_RELEASE_WARMUP_STEPS,
            "recurrent_release_context_steps": RECURRENT_RELEASE_CONTEXT_STEPS,
            "recurrent_release_tail_mode": "terminal-context-repeat",
            "recurrent_release_loss_weight": args.recurrent_release_loss_weight,
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
