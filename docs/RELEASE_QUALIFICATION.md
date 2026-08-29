# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a separate artifact-bound qualification bundle built from the exact model lineage, exact original audio bytes, final runtime/AFE artifacts and machine/raw physical-target evidence.

`v0.3.x` deliberately hard-cuts the evidence schemas. Older qualification manifests/provenance are not accepted as v0.3 shipping evidence.

## v0.3 contracts

- deployable model: **KWSP ABI v2**;
- deployable keyword pack: **KWKP ABI v3**;
- frontend lineage: **frontend-spec v2**;
- model provenance: **schema v3**;
- evaluation provenance: **schema v2**;
- dataset audit: **schema v3**;
- runtime-soak evidence: **schema v2**;
- target evidence: **schema v2**;
- qualification manifest: **schema v2**;
- qualification policy: **schema v2**;
- qualification gate result: **schema v3**.

The software/API version may change while KWSP/KWKP remain unchanged. v0.3 changes the public discontinuity API and evidence contracts, not the on-device model/keyword binary layouts.

## Evidence graph

```text
immutable training OCI image
 + training manifests + every training WAV/decoded-PCM identity
 + clean dataset audit that includes final references.jsonl
        |
        v
train_ctc.py -> checkpoint with canonical training corpus identity
        |
export_model.py -> KWSP v2 + model provenance schema v3

KWSP + KWKP + exact runtime/AFE
        |
        +-> exact kws_wav + final references.jsonl + every held-out WAV
        |       -> evaluation provenance schema v2 + metrics
        |
        +-> exact target kws_board_bench + representative board WAV
        |       -> timing summary
        |
        +-> collect_runtime_soak.py supervising the real process
        |       -> runtime-soak schema v2 (CPU/RSS/temp/soak)
        |
        +-> raw stack evidence + raw power trace/instrument identity
                |
        collect_target_evidence.py
                -> target evidence schema v2
                         |
                         v
                qualification_manifest.py schema v2
                         |
                  SKU policy schema v2
                         |
                         v
                qualification_gate.py -> schema v3 PASS/FAIL
```

## 1. Freeze the training environment

`training/Dockerfile` performs no network package installation. It requires a prebuilt training base referenced by immutable OCI digest:

```text
registry.example/kws-training-base@sha256:<64 hex>
```

Build the repository wrapper:

```bash
python3 training/build_container.py \
  --base-image "$KWS_TRAINING_BASE_IMAGE" \
  --tag kws-training:v0.3
```

Pass the final image digest into shipping training:

```bash
KWS_TRAINING_IMAGE_DIGEST=sha256:<final-image-digest> \
python3 training/train_ctc.py \
  --manifest data/train.jsonl \
  --tokens release/training-tokens.txt \
  --frontend logmel \
  --require-container-digest \
  --output release/base.pt
```

The checkpoint records repository/training-code hashes, lock/Dockerfile hashes, Python/PyTorch/platform identity and final training image digest. `.github/workflows/training-integration.yml` can periodically exercise the real `torch_ctc` path when a digest-pinned `KWS_TRAINING_BASE_IMAGE` is configured.

## 2. Bind the real training corpus

`train_ctc.py` accepts TSV (`WAV<TAB>token_ids`) or JSONL. Human qualification projects should use JSONL identity metadata:

```json
{"audio":"audio/u001.wav","tokens":[1,2,3],"speaker_id":"spk001","session_id":"s01","source_id":"src001","room_id":"living-room-a","device_id":"robot-a"}
```

At checkpoint save time the trainer reopens every WAV and records file SHA256, decoded mono-16-kHz PCM16 SHA256, frame count/duration, stable path/metadata and canonical whole-corpus SHA256. Replacing a WAV underneath an unchanged manifest therefore changes model lineage.

## 3. Audit the exact final held-out manifest

The final dataset audit must include the **same `qualification/references.jsonl`** later supplied to `qualification_manifest.py`:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=qualification/references.jsonl \
  --audio-root qualification=qualification/audio \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json \
  --fail-within-split
```

This is not interchangeable with a separate “qualification manifest” that merely points to similar files. v0.3 qualification hashes the selected training manifests plus the exact references file and requires dataset-audit coverage of those exact bytes.

Speaker/session/source overlap is a hard violation for final human qualification. Room/device overlap is reported and may be promoted to a hard failure by product policy. A final held-out recording must never be mined into replay/tuning and then reused as unbiased evidence.

## 4. Export and freeze the model

```bash
python3 training/export_model.py \
  --checkpoint release/base.pt \
  --tokens release/tokens.txt \
  --output release/base.kwm
```

Model provenance schema v3 binds the exact KWSP bytes, checkpoint, token mappings, frontend identity/spec, training environment/hyperparameters, quantization diagnostics and canonical real training-corpus identity.

## 5. Compile and freeze KWKP v3

```bash
python3 tools/compile_keywords.py \
  --tokens release/tokens.txt \
  --keywords release/keywords.tsv \
  --out-pack release/xiaowo.kwk \
  --out-json release/keywords.json
```

Any pack change creates a new release qualification tuple even when `.kwm` is unchanged.

## 6. Bind the final AFE

Final thresholds and qualification must use the exact shipping microphones, enclosure and audio-pipeline configuration. When using the command AFE adapter, retain executable/config SHA256 and reported latency. Any change to BF/AEC/RES/NS/AGC/gain policy or the physical sound path creates a new qualification tuple.

## 7. Run held-out continuous audio

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

python3 eval/domain_metrics.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --output qualification/domain-metrics.json
```

Evaluation provenance schema v2 reopens every referenced WAV and binds file SHA256, decoded PCM SHA256, frame count and canonical corpus SHA256. Every declared `duration_s` must equal real WAV duration. FAR exposure therefore comes from actual audio bytes, not a self-reported duration.

## 8. Real far-field coverage

A 3–5 m product claim requires genuine human positives through the final microphones/enclosure/AFE. Synthetic distance rendering, TTS or measured-RIR/TTS remain development evidence only.

At minimum include near/mid/far distance, expected azimuth, RT60/SNR, home conversation, near-homophones, TV/phone/smart-speaker playback, local-device playback through shipping AEC, double-talk/AEC residual, motor/fan/gear/pump/chassis noise, moving/static states and final AGC/AEC/NS modes.

## 9. Physical target-board timing

Run the exact target binary/model/pack:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Retain the benchmark executable and board audio. Hosted x86 or cross-build success is not target timing evidence.

## 10. Measure the real runtime process

Supervise the actual product/KWS qualification process:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Runtime-soak schema v2 derives from the child process:

- actual/requested soak duration and early-exit status;
- max child-process RSS;
- average child-process CPU percentage from `/proc/<pid>/stat` CPU-time deltas;
- thermal samples/max temperature when available;
- raw time-series samples.

If the process exits early, the soak collector fails. This prevents the later evidence collector from measuring its own Python process and calling that product RSS/CPU.

Also retain audio-pipeline XRUN/backpressure counters and KWS discontinuity/telemetry evidence where the SKU requires them.

## 11. Assemble target evidence

Stack high-water remains product-harness specific and must have retained raw evidence. Power requires an external raw instrument trace plus instrument/calibration identity.

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`soak_hours`, CPU, RSS and max temperature are imported from the retained runtime-soak JSON rather than command-line declarations. The target-evidence JSON automatically binds runtime-soak and power raw files and also binds any additional `--raw-evidence` inputs.

## 12. Build the byte-complete manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest data/train.jsonl \
  --dataset-audit qualification/dataset-audit.json \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --eval-audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --evidence-collector qualification/collect_target_evidence.py \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

Repeat `--training-manifest` and `--raw-evidence` for all participating artifacts. The selected raw files must exactly match the evidence JSON’s declared raw-evidence tuple.

The verifier independently checks deployable ABIs, vocabulary/frontend lineage, actual training WAV identity, exact dataset-audit coverage, actual held-out WAV identity/duration, evaluation formulas/provenance, board benchmark formulas/artifacts, evidence collector identity and raw target evidence hashes.

## 13. Apply the SKU policy

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Policy schema v2 gates FAR/FRR point estimates and one-sided confidence bounds plus latency, p99 process time, RTF/headroom, evidence exposure, soak, CPU, RSS, stack, temperature and power.

Exit codes:

- `0`: structurally valid evidence and all policy gates passed;
- `1`: structurally valid evidence but thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy.

Gate result schema v3 binds the exact manifest/policy identity.

## 14. Retain the complete release tuple

Retain source SHA/version/tag, KWSP/model provenance/checkpoint/training image digest, training manifests/original training audio identity, clean dataset audit, KWKP/release vocabulary/config, final AFE identity, exact evaluation runner/references/original held-out WAVs/detections/provenance/metrics, exact board runner/audio/summary, runtime-soak raw trace, stack raw evidence, power raw trace/instrument/calibration identity, exact target evidence collector/JSON, approved SKU policy, qualification manifest/gate result and final distribution signature/attestation.

## Software baseline vs shipping qualification

A green v0.3 repository can be a mature software baseline without being an acoustically qualified Mandarin SKU. CI proves parser/runtime contracts, synthetic regressions, corpus/evidence integrity mechanisms, cross-build compatibility, coverage/static-analysis gates and reproducible SDK output. It cannot prove final Mandarin FAR/FRR, real 3–5 m wake performance or physical Cortex-A32 product behavior.

Issue #2 remains open until independent human/device acoustic evidence and physical target-board evidence exist and pass the approved SKU policy.
