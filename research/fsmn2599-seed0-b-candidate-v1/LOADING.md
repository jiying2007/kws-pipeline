# Loading and input/output contract

The checkpoint is not in this repository. These instructions document the exact
candidate for a recipient who already has authorized access to its bytes; they
are not a download link or an assertion of model-promotion approval.

## Verify before loading

The expected file is `model.final.pt`, exactly 3,037,273 bytes, SHA-256
`a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77`.
It is a plain saved state dictionary. Do not treat an arbitrary file with the
same name as trusted. Load only verified bytes, on CPU, with
`torch.load(..., map_location="cpu", weights_only=True)`, then strict state-dict
loading. This avoids intentionally enabling full Python pickle loading; it does
not establish safety for untrusted or substituted files.

Obtain the architecture from the fixed WeKws commit in `source-manifest.json`.
The upstream fixed `config.yaml`, `global_cmvn.kaldi` and `tokens.txt` are public
dependencies, not included files. Check their recorded hashes before use. The
upstream config's CMVN path must point at the verified CMVN file in your own
working directory. Do not download another unpinned model or silently change
its class count or feature configuration.

Illustrative loading code, statically checked but not executed as part of this
metadata preparation:

```python
from pathlib import Path
import hashlib
import yaml
import torch
from wekws.model.kws_model import init_model

def verified(path, expected_sha, expected_bytes):
    path = Path(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError("Unexpected artifact size")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Unexpected artifact hash")
    return path

checkpoint = verified("model.final.pt",
    "a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77", 3037273)
config_path = verified("config.yaml",
    "2755715dfb28693afb8158b805ac5461b2aa4efbb410d4483590da8b6f6bfeae", 1139)
cmvn_path = verified("global_cmvn.kaldi",
    "278d109a9e4e70189e2d657ed52ea846446d5abdd58bcc9cb6b8800a52dac91c", 1711)
tokens_path = verified("tokens.txt",
    "5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa", 24528)
config = yaml.safe_load(config_path.read_text())
config["model"]["cmvn"]["cmvn_file"] = str(cmvn_path)
model = init_model(config["model"])
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

This only constructs and loads the architecture. No forward pass, training,
frontend, decoder or keyword acceptance is executed by the example.
PyTorch, PyYAML and the pinned WeKws dependency environment must already be
available; no new dependency installation is prescribed by this record.

## Inputs, outputs and token mapping

- Audio contract used by the referenced streaming frontend: mono 16-kHz PCM16.
  Kaldi-compatible fbank receives PCM-scale float values without dividing by
  32768; 80 mel bins, 25-ms frames, 10-ms hops, inference dither 0, energy_floor 0.
  The upstream training config's dither 1 is not the deterministic inference
  setting. Frontend differences require their own equivalence check.
- The model consumes float32 features shaped `[batch, time, 400]`, formed by
  splicing two left and two right 80-bin fbank frames and applying the fixed
  frame-skip 3 convention. It performs its stored 400-dimensional CMVN internally.
  Passing raw PCM, unspliced 80-dimensional features, or externally normalizing
  again violates this contract.
- The existing output affine maps 140 hidden values to 2599 logits. `forward`
  returns logits and cache. Apply softmax across all 2599 classes when probabilities
  are required; do not renormalize only the target characters.
- For the fixed token file, output index is numeric token ID minus 1, keeping
  indices 0 through 2598. `<eps>` has raw ID 0 and is excluded; raw ID 1 maps to
  CTC blank 0. `sil` is an alias at that raw ID; use `<blk>` for the canonical
  display. The fixed targets map to `[1027, 1401, 1462, 2093]` for `你好小窝`
  and `[1462, 2093, 1462, 2093]` for `小窝小窝`. Adjacent repeated CTC labels
  need blank separation; a greedy transcript is not itself a wake decision.
- FSMN caches and right-context/lookahead matter. Reset state at a recording
  boundary or discontinuity, preserve the fixed splicing/downsampling/cache
  semantics, and define end-of-stream handling before any continuous test.
  The historical short-clip features used 300-ms zero padding. That diagnostic
  padding choice is not an accepted live end-of-stream or latency policy.

## Numerical and deployment boundary

The 864-clip saved-logit comparison used offline-batch head geometry and a
float64 normalized probability calculation. The eight-session cached comparison
used canonical single-row shape `(1, 1, 140)` to `(1, 1, 2599)` and CPU float32
softmax. They are separately identified evidence chains and are not asserted
bitwise equivalent. An arbitrary upstream demo command is not certified to
reproduce this project's fixed three-window decoder and boundary behavior.

The selected C kernels failed the strict 76-frame old-head logit/cache parity
gate; a complete B-specific C parity result is not established. This
state dictionary is not a validated C/SSC305 deployment artifact and
cannot be loaded by the current KWSP shipping reader. No threshold, timing,
memory, power or physical-board acceptance should be inferred from loading it.
