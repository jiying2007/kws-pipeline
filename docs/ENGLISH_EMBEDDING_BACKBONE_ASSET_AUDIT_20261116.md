# English shared speech-embedding asset qualification — 2026-11-16

## Candidate boundary
This is a bounded asset/operator audit of the shared feature extractor described by openWakeWord, not admission of any pretrained wake-word classifier. The upstream source code and shared Google speech-embedding feature model are Apache-2.0. The release's pretrained keyword classifiers are CC BY-NC-SA 4.0 and explicitly excluded.

## Frozen upstream
- Repository: dscripka/openWakeWord
- Source commit: 368c03716d1e92591906a84949bc477f3a834455
- Release: v0.5.1
- Shared assets: mel frontend and embedding model, in both TFLite and ONNX forms.
- Release IDs and expected byte sizes are pinned in `configs/research/english-embedding-backbone-assets-20261116.json`.

## Audit
The one-shot worker downloads only four shared feature assets, verifies release IDs and byte sizes, hashes each file, inspects ONNX operators/tensors/initializer sizes, and runs only synthetic zero-tensor smoke inference. It does not download a wake-word classifier, training data, negative corpus, or real audio.

Expected combined sizes:
- TFLite frontend + embedding: 2,422,828 bytes.
- ONNX frontend + embedding: 2,414,536 bytes.

Both are slightly above the preferred 2-MiB model-weight target. This is a budget warning, not an automatic rejection. No quantization or compression benefit is assumed before numerical and event parity tests.

## Still not qualified
- No streaming parity on real final-IPA positives.
- No custom causal event head or training.
- No ARM/C implementation.
- No SSC305 ABI, CPU, RAM, thermal or long-run measurement.
- No FAR/FRR or model selection.

The next stage, if the exact operators are suitable, is a source-identical causal feature stream implementation and parity check, then a small causal event head compared against a causal FSMN-CTC on the same admitted dataset. The upstream bidirectional LSTM head and all CC BY-NC-SA wake-word classifiers remain excluded.
