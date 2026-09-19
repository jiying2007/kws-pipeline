# Model artifacts

Model storage is split by lifecycle.

## Intermediate training state

Failed runs, per-round checkpoints and unselected research candidates remain
ephemeral workflow artifacts. They are **not** committed to Git.

## Promoted model registry

A model that has passed the governed `model-promotion` verifier must also be
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

The immutable GitHub model release remains the distribution/attestation source.
The Git registry is the durable, reviewable mirror so a trained model does not
disappear with Actions artifact retention.

Registry entries are immutable. Re-training creates a new release tag and a new
directory; it never overwrites an existing entry.

Current direct-Git policy limits one registry entry to 5 MB. The present KWS
model tuple is below 2 MB, so Git LFS is deliberately not required. If a future
entry exceeds that bound, the storage policy must be changed explicitly rather
than silently growing the repository.

## Runtime format

The shipping runtime model format is little-endian **`KWSP` ABI v2** with a
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
