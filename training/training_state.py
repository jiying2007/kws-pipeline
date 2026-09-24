from __future__ import annotations

import hashlib
import json
import sys
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
