from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import pathlib
import random
import sys
import tempfile
from collections.abc import Mapping

import torch

POLICY = "named-tensor-raw-bytes-v1"


def state_identity(state: Mapping[str, torch.Tensor]) -> dict:
    """Content identity over sorted names, dtype, shape and exact tensor bytes.

    Not a hash of checkpoint serialization, optimizer state, RNG or a resume
    promise. Require little-endian storage explicitly rather than mislabeling
    native bytes as a portable canonical representation on other hosts.
    """
    if sys.byteorder != "little":
        raise ValueError("state identity requires little-endian tensor storage")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state identity requires a non-empty state mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("state tensor names must be non-empty strings")
    records = []
    digest = hashlib.sha256((POLICY + "\0").encode("ascii"))
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name}: expected a dense strided tensor")
        if tensor.is_quantized:
            raise ValueError(f"{name}: quantized tensors are not float training state")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name}: training state contains non-finite values")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for block in (encoded, raw):
            digest.update(len(block).to_bytes(8, "little"))
            digest.update(block)
        records.append({**header, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"schema_version": 1, "policy": POLICY, "sha256": digest.hexdigest(), "tensors": records}

def load_checkpoint_snapshot(path):
    """Load and identify the same file bytes; never reopen for provenance."""
    raw = pathlib.Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    return checkpoint, hashlib.sha256(raw).hexdigest()


def capture_completion_state(model, optimizer, shuffle_generator, *,
                             completed_epochs: int, optimizer_steps: int) -> dict:
    """Record CPU end-of-epoch state, without claiming a supported resume path."""
    for name, value in (("completed_epochs", completed_epochs),
                        ("optimizer_steps", optimizer_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    parameters = dict(model.named_parameters())
    if (any(parameter.device.type != "cpu" for parameter in parameters.values())
            or shuffle_generator.device.type != "cpu"):
        raise ValueError("completion state requires a CPU model and generator")
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            if parameter.device.type != "cpu" or id(parameter) not in names:
                raise ValueError("completion state requires named CPU parameters")
            if id(parameter) in seen:
                raise ValueError("completion state contains duplicate optimizer parameters")
            seen.add(id(parameter))
            group_names.append(names[id(parameter)])
        groups.append(group_names)
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    def require_cpu_tensors(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("completion state requires CPU optimizer tensors")
        elif isinstance(value, dict):
            for item in value.values():
                require_cpu_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                require_cpu_tensors(item)

    require_cpu_tensors(optimizer_state)
    if (len(optimizer_state["param_groups"]) != len(groups)
            or any(len(group["params"]) != len(group_names)
                   for group, group_names in zip(optimizer_state["param_groups"], groups))):
        raise ValueError("completion state optimizer parameter mapping mismatch")
    return {
        "schema_version": 1,
        "policy": "completed-epoch-cpu-state-v1",
        "resume_supported": False,
        "progress_scope": "current-invocation-only",
        "completed_epochs": completed_epochs,
        "optimizer_steps": optimizer_steps,
        "optimizer_parameter_names": groups,
        "optimizer_state_dict": optimizer_state,
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state().clone(),
            "data_loader": shuffle_generator.get_state().clone(),
        },
    }


def atomic_save_checkpoint(checkpoint: dict, path) -> None:
    """Publish only a fully written, weights-only-readable checkpoint.

    Serialization/readback/fsync/replace failure leaves a previous destination
    untouched. This is a completed-endpoint save, not periodic recovery or a
    promise of directory-entry durability across power loss.
    """
    expected = checkpoint.get("float_state_identity")
    if expected is None or state_identity(checkpoint["state_dict"]) != expected:
        raise ValueError("checkpoint float state identity mismatch before save")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = stream.name
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        saved, _ = load_checkpoint_snapshot(temporary)
        if (saved.get("float_state_identity") != expected
                or state_identity(saved["state_dict"]) != expected):
            raise ValueError("checkpoint float state identity mismatch after save")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
