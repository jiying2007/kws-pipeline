# Model artifacts

Model storage is split by lifecycle.

## Intermediate training state

Failed runs, per-round checkpoints and unselected research candidates remain
ephemeral workflow artifacts. They are **not** committed to Git.

## Historical verified model registry

A historical model that passed the retired `model-promotion` publication verifier was also required to be
mirrored into:

```
models/registry/<model-release-tag>/
```

The registry entry contains the exact immutable release tuple, including:

- `xiaowo-model.kwm` — deployable KWSP model;
- `xiaowo-model.pt` — exact source checkpoint;
- `xiaowo-model-provenance.json`;
- `xiaowo-keywords.kwk` and `xiaowo-keywords.tsv`;
- training, qualification, robustness and continuous-FAR summaries;
- `model-promotion-manifest.json`;
- `MODEL_SHA256SUMS`;
- `registry.json` — Git-mirror identity and storage policy.

The original published historical model may be inspected through its immutable
GitHub Release and the durable Git registry. **Current research models do not
require publication or promotion**: each new candidate needs an immutable source,
model, keyword-pack, data and evaluation identity for scientific replay.

Registry entries are immutable. A new study uses a distinct content-addressed
research model identity and directory; it never overwrites a consumed entry.

Current direct-Git policy limits one registry entry to 5 MB. The present KWS
model tuple is below 2 MB, so Git LFS is deliberately not required. If a future
entry exceeds that bound, the storage policy must be changed explicitly rather
than silently growing the repository.

## Runtime format

The embedded KWS research runtime model format is little-endian **`KWSP` ABI v2** with a
canonical 72-byte header. It stores int8 input/recurrent/output matrices,
float32 biases, fixed 16-kHz / 400-sample / 320-sample geometry, 64-bit
vocabulary fingerprint and a model-bound frontend kind:

- `0`: `logmel`;
- `1`: `pcen-lite`.

`training/export_model.py` is the canonical writer; `kws_model_open()` is
the canonical device-side reader.

## Required export tuple

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens keywords/tokens.zh.txt \
  --frontend logmel \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

Use `--frontend pcen-lite` to train a PCEN-lite model. A warm start cannot
switch frontend identity.

The exporter writes:

- `base.kwm` — deployable model ABI v2;
- `base.kwm.provenance.json` — deterministic model lineage.

Provenance records model/checkpoint SHA256, export/training token identities,
vocabulary fingerprint, frontend kind/name, frontend-spec version, training
manifests, seed/optimizer settings and per-matrix int8 quantization diagnostics.

A release qualification bundle must retain all concrete lineage inputs:

- exported `.kwm` and provenance;
- exact source `.pt` checkpoint;
- exact training token file;
- every training manifest recorded by the checkpoint;
- release token file;
- matching KWKP v3 keyword pack and runtime config.

`qualification_manifest.py` re-hashes these files and
`qualification_gate.py` additionally requires the runtime frontend identity to
match the model lineage. See `docs/RELEASE_QUALIFICATION.md`.
