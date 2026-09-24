#!/usr/bin/env python3
"""Bounded, opt-in observer of the REAL CPU micro trainer; no replacement math.

Hooks run in a disposable subprocess. They save detached bytes and return the
original objects. Trace-on/off tensor and epoch-history parity is a regression
requirement. This is diagnosis, not an optimizer-resume or portability promise.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import pathlib
import platform
import sys
from unittest.mock import patch

POLICY = "actual-micro-trainer-numeric-trace-v1"
MAX_ROWS = 32
MAX_STEPS = 12
MAX_BYTES = 32 * 1024 * 1024
ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Recorder:
    def __init__(self, root: pathlib.Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError("numeric trace output must be empty")
        self.events: list[dict] = []
        self.bytes = 0
        self.step = 0
        self.forward_steps = 0
        self.weighted_calls = 0
        self.model = None
        self.environment: dict = {}
        self.finished = False
        self.flush()

    def flush(self) -> None:
        value = {"schema_version": 1, "policy": POLICY, "completed": self.finished,
                 "max_steps": MAX_STEPS, "steps": self.step, "tensor_bytes": self.bytes,
                 "environment": self.environment, "events": self.events}
        tmp = self.root / "trace.json.tmp"
        tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        tmp.replace(self.root / "trace.json")

    def capture(self, phase: str, tensors: dict | None = None, metadata: dict | None = None) -> None:
        from training_state import state_identity
        import torch

        if self.finished or len(self.events) >= 512:
            raise ValueError("numeric trace is finished or exceeds its event bound")
        event = {"ordinal": len(self.events), "step": self.step, "phase": phase,
                 "metadata": metadata or {}, "tensors": []}
        if tensors:
            identity = state_identity(tensors)
            event["state_sha256"] = identity["sha256"]
            for index, row in enumerate(identity["tensors"]):
                tensor = tensors[row["name"]].detach().cpu().contiguous()
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                if self.bytes + len(raw) > MAX_BYTES:
                    raise ValueError("numeric trace exceeds byte budget")
                name = f"{len(self.events):04d}-{index:03d}.bin"
                (self.root / name).write_bytes(raw)
                self.bytes += len(raw)
                event["tensors"].append({**row, "file": name})
        self.events.append(event)
        self.flush()  # A failing trainer still leaves an explicitly incomplete prefix.

    def finish(self) -> None:
        if self.model is None or not 0 < self.step <= MAX_STEPS:
            raise ValueError("numeric trace has no completed training steps")
        self.capture("final-state", self.model.state_dict())
        self.finished = True
        self.flush()


def run_observed(output: pathlib.Path, trainer_args: list[str]) -> None:
    # Match feature_cached_trainer's import ordering: model pins CPU math before
    # installing hooks. Do not import torch or perform a probe forward earlier.
    import train_ctc as base
    import torch
    import feature_cached_trainer as cached

    rec = Recorder(output)
    originals = {
        "manifest_init": base.Manifest.__init__, "getitem": base.Manifest.__getitem__,
        "model_init": base.TinyStreamingRNN.__init__, "forward": base.TinyStreamingRNN.forward,
        "step": base.TinyStreamingRNN.step, "collate": base.collate,
        "ctc": base.nn.CTCLoss.forward, "backward": torch.Tensor.backward,
        "clip": base.nn.utils.clip_grad_norm_, "optimizer": torch.optim.AdamW.step,
        "environment": base.training_environment,
        "weighted": base.normalized_weighted_mean,
    }

    def environment():
        value = originals["environment"]()
        value["training_code_sha256"]["training/reproducibility_trace.py"] = digest(pathlib.Path(__file__))
        rec.environment = {**value, "libc": list(platform.libc_ver()),
                           "byteorder": sys.byteorder,
                           "aten_reported_capability": torch.backends.cpu.get_cpu_capability(),
                           "python_executable_sha256": digest(pathlib.Path(sys.executable))}
        rec.flush()
        return value

    def manifest_init(self, *args, **kwargs):
        originals["manifest_init"](self, *args, **kwargs)
        if not 0 < len(self.rows) <= MAX_ROWS:
            raise ValueError("numeric trace only supports bounded micro datasets")
        rec.capture("corpus", metadata={"rows": [
            {"index": i, "tokens": tokens, "file_sha256": identity["file_sha256"],
             "pcm_sha256": identity["pcm_sha256"]}
            for i, ((_, tokens), identity) in enumerate(zip(self.rows, self.identity_rows))]})

    def getitem(self, index):
        result = originals["getitem"](self, index)
        rec.capture("features", {"features": result[0], "targets": result[1]}, {"row": index})
        return result

    def model_init(self, *args, **kwargs):
        originals["model_init"](self, *args, **kwargs)
        if rec.model is not None:
            raise ValueError("numeric trace expects exactly one training model")
        rec.model = self
        rec.capture("initial-state", {**self.state_dict(), "rng.cpu": torch.get_rng_state()})

    def collate(batch):
        rec.step += 1
        if rec.step > MAX_STEPS:
            raise ValueError("numeric trace only supports bounded micro training")
        rec.weighted_calls = 0
        result = originals["collate"](batch)
        rec.capture("batch-input", dict(zip(("features", "targets", "xlen", "ylen"), result)))
        return result

    def step(self, *args, **kwargs):
        result = originals["step"](self, *args, **kwargs)
        if rec.forward_steps == 0:
            rec.capture("first-recurrent-step", {"logits": result[0], "hidden": result[1]})
        rec.forward_steps += 1
        return result

    def forward(self, *args, **kwargs):
        rec.forward_steps = 0
        result = originals["forward"](self, *args, **kwargs)
        rec.capture("forward-logits", {"logits": result})
        return result

    def ctc(self, log_probs, targets, xlen, ylen):
        rec.capture("log-probabilities", {"log_probs": log_probs})
        result = originals["ctc"](self, log_probs, targets, xlen, ylen)
        rec.capture("raw-ctc", {"ctc": result})
        return result

    def weighted(*args, **kwargs):
        result = originals["weighted"](*args, **kwargs)
        rec.capture("weighted-loss", {"value": result}, {"call": rec.weighted_calls})
        rec.weighted_calls += 1
        return result

    def gradients():
        return {name: param.grad for name, param in rec.model.named_parameters() if param.grad is not None}

    def backward(self, *args, **kwargs):
        rec.capture("total-loss", {"loss": self})
        result = originals["backward"](self, *args, **kwargs)
        rec.capture("gradient-before-clip", gradients())
        return result

    def clip(*args, **kwargs):
        result = originals["clip"](*args, **kwargs)
        rec.capture("gradient-after-clip", {**gradients(), "total_norm": result})
        return result

    def optimizer(self, *args, **kwargs):
        result = originals["optimizer"](self, *args, **kwargs)
        tensors = {"parameter." + n: p for n, p in rec.model.named_parameters()}
        for name, param in rec.model.named_parameters():
            for key, value in self.state[param].items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"unexpected non-tensor optimizer state: {key}")
                tensors[f"optimizer.{name}.{key}"] = value
        tensors["rng.cpu"] = torch.get_rng_state()
        rec.capture("optimizer-step", tensors)
        return result

    def loss_observer(name, function):
        def observed(*args, **kwargs):
            result = function(*args, **kwargs)
            tensor = result[0] if isinstance(result, tuple) else result
            rec.capture(name, {"value": tensor})
            return result
        return observed

    with ExitStack() as stack:
        for obj, name, fn in (
            (base, "training_environment", environment), (base.Manifest, "__init__", manifest_init),
            (base.Manifest, "__getitem__", getitem), (base.TinyStreamingRNN, "__init__", model_init),
            (base.TinyStreamingRNN, "forward", forward), (base.TinyStreamingRNN, "step", step),
            (base, "collate", collate), (base.nn.CTCLoss, "forward", ctc),
            (base, "normalized_weighted_mean", weighted), (torch.Tensor, "backward", backward),
            (base.nn.utils, "clip_grad_norm_", clip), (torch.optim.AdamW, "step", optimizer),
        ):
            stack.enter_context(patch.object(obj, name, fn))
        for name in ("ordered_token_loss", "keyword_sequence_margin_loss", "strict_prefix_completion_loss",
                     "recurrent_release_loss", "ordered_path_purity_loss"):
            stack.enter_context(patch.object(base, name, loss_observer(name, getattr(base, name))))
        # The original cache adapter and original trainer own every operation.
        stack.enter_context(patch.object(sys, "argv", [str(cached.__file__), *trainer_args]))
        # cached._run_trainer replaces these module bindings; restore on exceptions too.
        stack.enter_context(patch.object(base, "Manifest", base.Manifest))
        stack.enter_context(patch.object(base, "training_environment", base.training_environment))
        cached.main()
        rec.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    delegated = args.trainer_args
    if not delegated or delegated[0] != "--" or "--warm-start" in delegated or "--head-only" in delegated:
        parser.error("expected -- followed by cold-start micro trainer arguments")
    if delegated[1:3] != ["--trainer", "rnn"]:
        parser.error("numeric tracing is currently scoped to the RNN micro fixture")
    run_observed(args.output, delegated[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
