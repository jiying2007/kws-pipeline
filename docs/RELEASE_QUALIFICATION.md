# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a separate artifact-bound qualification bundle built from the exact model, model-export provenance, source checkpoint, training vocabulary/manifests, keyword pack, release vocabulary, runtime config, evaluation runner/corpus, target-board benchmark binary/audio and measured board evidence.

## Artifact contract

Shipping artifacts use separate ABIs:

- model: **`KWSP` ABI v2**;
- keyword pack: **`KWKP` ABI v3**;
- model lineage/frontend contract: **frontend-spec v2**, with `frontend_kind=0` logmel or `frontend_kind=1` pcen-lite.

The release manifest must carry runtime frontend identity, and the gate requires it to match model lineage exactly.

```text
checkpoint + training tokens + training manifests
                 |
                 v
        export_model.py -> model.kwm + model.kwm.provenance.json
                                  |
model/provenance + concrete lineage + KWKP v3 + release tokens/config
                                  |
                                  +-> exact kws_wav + held-out references
                                  |      -> detections + provenance + metrics
                                  |
                                  +-> exact target kws_board_bench + board WAV
                                  |      -> board summary
                                  |
                                  +-> target resource/soak/thermal/power evidence
                                                       |
                                                       v
                                            qualification_manifest.py
                                                       |
                                                 SKU policy
                                                       |
                                                       v
                                            qualification_gate.py
```

## 1. Freeze training/model lineage

Train with the exact token vocabulary and frontend used by the candidate:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens release/training-tokens.txt \
  --frontend logmel \
  --output release/base.pt

python3 training/export_model.py \
  --checkpoint release/base.pt \
  --tokens release/tokens.txt \
  --output release/base.kwm
```

Use `--frontend pcen-lite` when that is the selected model frontend. Warm start/export must not change frontend identity silently.

The exporter writes `base.kwm.provenance.json`, binding the `.kwm` to checkpoint hash, training/export vocabulary identities, training-manifest hashes, frontend identity/spec, hyperparameters and per-matrix int8 quantization diagnostics.

Retain the **actual checkpoint**, **actual training token file** and **every training manifest** referenced by the checkpoint. Qualification re-hashes these bytes; provenance hashes are not treated as self-authenticating declarations.

Training and release token files may differ byte-for-byte only when their canonical token→ID mapping/fingerprint is identical.

## 2. Compile and freeze KWKP v3

Compile the exact release keywords:

```bash
python3 tools/compile_keywords.py \
  --tokens release/tokens.txt \
  --keywords release/keywords.tsv \
  --out-pack release/xiaowo.kwk \
  --out-json release/keywords.json
```

The pack stores per-keyword threshold, token path, trailing-blank requirement, priority and prefix policy. An L0 pack change creates a new release tuple even when `.kwm` is unchanged.

## 3. Audit split isolation

Run `training/audit_dataset.py` before training/tuning/final qualification. Decoded mono-16-kHz PCM16 SHA256 must not overlap between independent splits. Speaker/session/device/TTS/noise/RIR grouping should additionally remain disjoint at the dataset-management layer.

The final qualification-heldout corpus must never be mined into replay and then still be called independent heldout.

## 4. Run held-out continuous audio

Use the exact retained `kws_wav` binary:

```bash
python3 eval/run_corpus.py \
  --runner qualification/kws_wav.eval \
  --model release/base.kwm \
  --keywords release/xiaowo.kwk \
  --references qualification/references.jsonl \
  --audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --provenance qualification/detections.provenance.json

python3 eval/score_events.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --summary qualification/eval-summary.json \
  --false-positives qualification/false-positives.jsonl \
  --false-rejects qualification/false-rejects.jsonl
```

When domain metadata exists, also retain `eval/domain_metrics.py` output. A product far-field claim requires real far positives in the held-out corpus; synthetic deterministic far coverage is not a substitute.

## 5. Run the physical target-board benchmark

Cross-build `kws_board_bench` with the target toolchain and run the exact `.kwm/.kwk` with representative post-AEC/post-NS audio:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Retain the exact target executable as `qualification/kws_board_bench.target`. The report binds runner/model/pack/audio hashes and records model/pack/arena bytes, mean/p50/p95/p99/max processing time, RTF and p99 scheduling headroom.

Hosted x86 or cross-build success is not a target-board measurement.

## 6. Record target evidence

Copy `configs/qualification.evidence.example.json` and replace all placeholders with candidate measurements. Required evidence covers target/board revision/SoC/toolchain/compiler flags/governor/audio-front-end identity, soak duration, CPU, RSS, stack high-water mark, maximum temperature and average power.

## 7. Build the byte-complete manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.tsv \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

Repeat `--training-manifest` for every manifest used by the checkpoint.

The verifier independently checks:

- canonical model ABI v2 and keyword-pack ABI v3 layouts;
- fixed 16-kHz/400/320 geometry, dimensions, finite values, token bounds, padding and duplicate paths;
- model/pack/release-token vocabulary identity;
- training-token mapping against release tokens;
- actual checkpoint/training tokens/training-manifest multiset against model provenance;
- model frontend kind/name and frontend-spec v2 lineage;
- runtime frontend kind/name against model lineage;
- actual model/provenance/checkpoint/tokens/manifests/pack/config/eval/board SHA256 relationships;
- reference/detection counts, event bounds, audio hours and FAR/FRR/latency formulas;
- board WAV geometry, block count and benchmark formulas;
- target/toolchain/governor/audio-front-end/resource evidence.

The manifest contains no generated timestamp; identical inputs produce identical JSON.

## 8. Apply an explicit SKU policy

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Exit codes:

- `0`: valid evidence and all policy gates passed;
- `1`: valid evidence but one or more thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy, including runtime/model frontend mismatch.

The policy can gate minimum audio/wake counts, FRR, FAR/hour, p95 wake latency, p99 runtime, RTF/headroom, soak, CPU, RSS, stack, temperature and power. `configs/qualification.policy.example.json` is an example schema, not a shipping commitment.

## 9. Retain the complete release tuple

Retain together:

- source SHA;
- `.kwm` + `.kwm.provenance.json`;
- source checkpoint, training tokens and every training manifest;
- release token vocabulary;
- KWKP v3 `.kwk` + keyword source TSV;
- shipping runtime config and exact frontend identity;
- exact evaluation runner/references/detections/provenance/summary/domain metrics;
- exact target benchmark runner/board WAV/board summary;
- target evidence JSON and approved SKU policy;
- qualification manifest + gate result;
- final bundle SHA256.

## Software baseline vs shipping qualification

A green repository can be a mature **software baseline** without being an acoustically qualified Mandarin SKU. The domain-aware loop and nightly long-FAR provide strong deterministic regression evidence, but shipping claims remain blocked until a real Mandarin model, independent human/device held-out corpus and physical target-board evidence exist. Repository issue #2 tracks that external evidence gate.
