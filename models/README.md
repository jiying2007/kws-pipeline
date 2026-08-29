# Model artifacts

Trained `.pt` checkpoints, exported `.kwm` binaries and export-provenance JSON files are build/release artifacts and are intentionally not committed here.

The runtime model format is little-endian **`KWSP` ABI v2** with a canonical 72-byte header. It stores int8 input/recurrent/output matrices, float32 biases, fixed 16-kHz / 400-sample / 320-sample geometry, 64-bit vocabulary fingerprint and a model-bound frontend kind:

- `0`: `logmel`;
- `1`: `pcen-lite`.

`training/export_model.py` is the canonical writer; `kws_model_open()` is the canonical device-side reader.

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

Use `--frontend pcen-lite` to train a PCEN-lite model. A warm start cannot switch frontend identity.

The exporter writes:

- `base.kwm` — deployable model ABI v2;
- `base.kwm.provenance.json` — deterministic model lineage.

Provenance records model/checkpoint SHA256, export/training token identities, vocabulary fingerprint, frontend kind/name, frontend-spec version, training manifests, seed/optimizer settings and per-matrix int8 quantization diagnostics.

A release qualification bundle must retain all concrete lineage inputs:

- exported `.kwm` and provenance;
- exact source `.pt` checkpoint;
- exact training token file;
- every training manifest recorded by the checkpoint;
- release token file;
- matching KWKP v3 keyword pack and runtime config.

`qualification_manifest.py` re-hashes these files and `qualification_gate.py` additionally requires the runtime frontend identity to match the model lineage. See `docs/RELEASE_QUALIFICATION.md`.
