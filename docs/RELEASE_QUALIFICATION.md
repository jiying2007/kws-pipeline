# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a second, artifact-bound qualification bundle built from the exact model, keyword pack, vocabulary, runtime config, evaluation runner, reference annotations, detections, target-board benchmark binary, board audio and measured board evidence.

```text
model + pack + tokens + runtime config
        |
        +-> exact kws_wav + references -> detections + provenance + metrics
        |
        +-> exact target kws_board_bench + board WAV -> board summary
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

## 1. Freeze release artifacts

Pin the source Git SHA, ABI-v2 `.kwm`, ABI-v2 `.kwk`, continuous token vocabulary, shipping runtime config, corpus identity/version and final BF/AEC/RES/NS/AGC configuration. Model, pack and vocabulary must carry the same 64-bit vocabulary fingerprint.

## 2. Run the held-out continuous-audio corpus

Use the exact `kws_wav` binary that will be retained as evaluation evidence:

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
  --false-positives qualification/false-positives.jsonl
```

The provenance sidecar binds evaluation runner, model, keyword pack, references and detections by SHA256. The summary also embeds the reference/detection hashes. The final held-out corpus must remain isolated from hard-negative mining and training.

## 3. Run the real target-board benchmark

`kws_board_bench` loads the real `.kwm/.kwk`, reads mono PCM16 16-kHz WAV and times one 320-sample/20-ms runtime call using `CLOCK_MONOTONIC`.

Cross-build it with the target toolchain, copy it to the target board, select the shipping/qualification governor and DVFS policy, then run representative post-AEC/post-NS audio:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Retain the **exact target executable used for that run** as `qualification/kws_board_bench.target`. The report SHA256-identifies the runner, model, keyword pack and complete board WAV, and records model/pack/arena bytes, block count, mean/p50/p95/p99/max process time, RTF and p99 scheduling headroom.

Hosted x86 output remains a regression signal only and cannot replace target-board evidence.

## 4. Record non-benchmark target evidence

Copy `configs/qualification.evidence.example.json` and replace every placeholder with measurements from the candidate. Required evidence includes target/board revision/SoC/toolchain/compiler flags/governor/audio-front-end identity plus soak duration, CPU, RSS, stack high-water mark, maximum temperature and average power.

The example numbers are schema examples only. `qualification_manifest.py` rejects `REPLACE_ME` placeholders.

## 5. Build the deterministic byte-complete manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
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

The manifest intentionally contains no generated timestamp. Byte-identical inputs produce byte-identical JSON.

The verifier independently checks:

- canonical ABI-v2 `.kwm/.kwk`, fixed 16-kHz/400/320 geometry, dimensions, finite scales/biases/thresholds, token bounds, zero padding and duplicate paths;
- runtime config geometry/dimensions and finite runtime parameters;
- model/pack/token vocabulary identity;
- SHA256 of the **actual** model, pack, token file, config, evaluation runner, references, detections, target benchmark runner and board audio;
- evaluation provenance hashes against those actual files;
- reference JSONL itself: unique recordings, durations, expected-event bounds, recording count, expected-wake count and total audio hours;
- detections JSONL itself: recording membership, finite times/confidences and actual detection count;
- evaluation identities: `matched + FR = expected`, `matched + FA = detections`, plus FRR/FAR formulas and latency ordering;
- board WAV itself: mono 16-kHz PCM16, actual duration and actual 20-ms block count;
- board report hashes, artifact sizes, monotonic percentiles, mean/RTF/headroom formulas;
- required target/toolchain/governor/audio-front-end/resource evidence.

This prevents a matching sidecar/summary pair from being reused after the underlying runner, annotations, detections or board audio has changed.

## 6. Apply an explicit SKU policy

Qualification thresholds live outside the evidence manifest:

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

The policy can gate acoustic data volume, expected wake count, FRR, FAR/hour, p95 wake latency, p99 runtime, RTF, p99 headroom, soak duration, CPU, RSS, stack high-water mark, maximum temperature and average power. The gate also validates manifest-internal artifact cross-links and SHA256-binds its result to the exact manifest and policy.

Exit codes:

- `0`: valid evidence and all approved policy gates passed;
- `1`: valid evidence but one or more thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy.

`configs/qualification.policy.example.json` is deliberately named `example-not-a-shipping-policy`; its numeric values are examples, not product commitments.

## 7. Retain the complete release tuple

For each SKU/model/keyword-pack tuple retain together:

- source SHA, `.kwm`, `.kwk`, token vocabulary and runtime config;
- exact evaluation runner, reference annotations, detections, provenance, evaluation summary and false-positive list;
- exact target benchmark runner, board WAV and board summary;
- target evidence JSON and approved SKU policy;
- qualification manifest and gate result;
- SHA256 of the final retained evidence/distributable bundle.

An L0 keyword-only update creates a new qualification tuple because the `.kwk` bytes changed even if `.kwm` did not.

## Software baseline vs shipping qualification

A green repository can be a **software baseline** without being an acoustically qualified Mandarin SKU. Shipping claims remain blocked until real model/data/target-board evidence exists; repository issue #2 tracks that external evidence gate.
