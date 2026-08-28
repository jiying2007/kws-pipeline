# Model artifacts

Trained `.pt` checkpoints and exported `.kwm` binaries are build/release artifacts and are intentionally not committed here.

The runtime format is little-endian `KWSP` version 1. It stores int8 input/recurrent/output matrices and float32 biases. `training/export_model.py` is the canonical writer; `kws_model_open()` is the canonical reader.

A production release should pin the model SHA-256 together with the token vocabulary, keyword manifest, acoustic evaluation report, compiler/toolchain and target-SKU certification record.
