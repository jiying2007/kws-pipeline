# Model artifacts

Trained `.pt` checkpoints, exported `.kwm` binaries and export-provenance JSON files are build/release artifacts and are intentionally not committed here.

The runtime model format is little-endian **`KWSP` ABI v2**. It stores int8 input/recurrent/output matrices, float32 biases, fixed 16-kHz / 400-sample / 320-sample frontend geometry and the 64-bit token-vocabulary fingerprint. `training/export_model.py` is the canonical writer; `kws_model_open()` is the canonical device-side reader.

## Required export tuple

Training requires the exact token vocabulary:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens keywords/tokens.zh.txt \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

The exporter writes two artifacts:

- `base.kwm` — the deployable ABI-v2 model;
- `base.kwm.provenance.json` — deterministic lineage for that model export.

The provenance records the exact `.kwm` SHA256, checkpoint SHA256, export token SHA256, token hash retained by the training checkpoint, vocabulary fingerprint, frontend-spec version, training-manifest hashes, seed/optimizer settings, and per-matrix int8 quantization scale/error/RMSE/SNR statistics.

`training/export_model.py` refuses to bind a checkpoint to a same-sized but differently mapped vocabulary. A release qualification bundle must retain **all concrete lineage inputs**, not only their hashes:

- the exported `.kwm` and `.kwm.provenance.json`;
- the exact source `.pt` checkpoint;
- the exact token file used during training;
- every training manifest recorded by the checkpoint;
- the release token file used for export/keyword compilation.

`tools/qualification_manifest.py` receives these through `--model-provenance`, `--checkpoint`, `--training-tokens`, repeated `--training-manifest`, and `--tokens`. It re-hashes the actual files, verifies the training/release token mappings are identical, and rejects any mismatch with the exporter provenance.

A production release should therefore pin the complete tuple: source SHA, `.kwm`, model provenance, checkpoint, training token/manifests, release token vocabulary, `.kwk`, runtime config, evaluation artifacts, target-board evidence, approved policy, qualification manifest and gate result. See `docs/RELEASE_QUALIFICATION.md`.
