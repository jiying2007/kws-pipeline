# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a second, artifact-bound qualification bundle built from the exact model, keyword pack, vocabulary, runtime config, continuous-audio corpus, target board, and final audio front-end.

The qualification path is intentionally deterministic and hash-bound:

```text
model.kwm + keywords.kwk + tokens.txt + runtime config
                         |
                         +-> continuous corpus -> detections.jsonl
                         |                    -> detections.provenance.json
                         |                    -> eval-summary.json
                         |
                         +-> target-board WAV -> board-summary.json
                         |
                         +-> target evidence -> evidence.json
                                                |
                                                v
                                    qualification-manifest.json
                                                |
                                     qualification policy
                                                |
                                                v
                                          PASS / FAIL
```

## 1. Freeze release artifacts

A release candidate must pin:

- source Git SHA;
- ABI-v2 `.kwm` model;
- ABI-v2 `.kwk` keyword pack;
- continuous token vocabulary;
- runtime config used by the product;
- corpus identity/version;
- final `audio-pipeline`/AEC/NS/AGC configuration.

The model, pack, and vocabulary must carry the same 64-bit vocabulary fingerprint. The release manifest rechecks this identity and stores SHA256 for every artifact.

## 2. Run the continuous-audio corpus

Build `kws_wav`, then execute the exact release artifacts across the held-out corpus:

```bash
python3 eval/run_corpus.py \
  --runner ./build/kws_wav \
  --model release/base.kwm \
  --keywords release/xiaowo.kwk \
  --references qualification/references.jsonl \
  --audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --provenance qualification/detections.provenance.json
```

The provenance sidecar binds the runner binary, model, keyword pack, reference annotations, and generated detections by SHA256.

Score the same files:

```bash
python3 eval/score_events.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --summary qualification/eval-summary.json \
  --false-positives qualification/false-positives.jsonl
```

The summary stores the reference and detection SHA256 values, so it cannot be substituted silently into another qualification bundle.

Do not mine hard negatives from the final held-out qualification corpus and then report the same corpus as unbiased release evidence.

## 3. Run the real target-board benchmark

`kws_board_bench` is a hosted/target Linux tool, not part of the real-time library. It loads the real `.kwm` and `.kwk`, reads mono PCM16 16-kHz WAV, and times one ABI-v2 hop (320 samples / 20 ms) per call with `CLOCK_MONOTONIC`.

Cross-build it with the same target toolchain used for the SDK, copy it and the release artifacts to the board, pin the intended governor/DVFS state, and run representative post-AEC/post-NS audio:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

The JSON report contains:

- actual model/keyword-pack/engine bytes;
- block count and repeated audio duration;
- mean, p50, p95, p99 and max process time;
- real-time factor (RTF);
- p99 headroom relative to the 20-ms hop deadline.

Hosted x86 output is useful only as a regression signal. It must never be presented as Cortex-A32/A7 performance evidence.

## 4. Record non-benchmark board evidence

Copy `configs/qualification.evidence.example.json` and replace every placeholder with measurements from the release candidate:

```json
{
  "target": "product-sku-board",
  "board_revision": "EVT2",
  "soc": "target SoC",
  "toolchain": "arm-linux-gnueabihf-gcc ...",
  "compiler_flags": "-O3 ...",
  "governor": "performance",
  "audio_frontend": "shipping BF/AEC/RES/NS/AGC config id",
  "soak_hours": 8.0,
  "cpu_percent": 4.2,
  "rss_kib": 640.0,
  "stack_high_water_bytes": 32768.0,
  "max_temp_c": 58.0,
  "average_power_mw": 135.0
}
```

The numbers above illustrate the schema only. They are not project targets or measured results.

## 5. Create the deterministic qualification manifest

```bash
python3 tools/release_manifest.py \
  --model release/base.kwm \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --evidence qualification/evidence.json \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

The manifest is deterministic: it intentionally contains no generated timestamp. Rebuilding it from byte-identical evidence produces byte-identical JSON.

It verifies:

- model/pack ABI and fixed 16-kHz/400/320 geometry;
- model/pack/token vocabulary identity;
- SHA256 binding between evaluation provenance and the selected model/pack;
- SHA256 binding between evaluation summary and reference/detection files;
- board-summary model/pack sizes;
- finite/consistent evaluation and performance values;
- required target/toolchain/governor/audio-front-end/soak/resource evidence.

## 6. Apply an explicit product policy

Qualification thresholds live outside the manifest because different SKUs can have different requirements. Copy `configs/qualification.policy.example.json` and replace it with the approved SKU policy.

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Exit codes:

- `0`: all policy gates passed;
- `1`: valid evidence, but one or more qualification thresholds failed;
- `2`: malformed/inconsistent/tampered evidence or policy.

The repository example policy is deliberately named `example-not-a-shipping-policy`. Its numbers are examples, not product commitments.

## 7. Release evidence retention

For every released SKU/model/keyword-pack tuple retain together:

- source commit SHA;
- `.kwm`, `.kwk`, token vocabulary and runtime config;
- corpus reference annotations and corpus version/identity;
- detections, evaluation provenance, evaluation summary and false-positive list;
- board benchmark WAV and board summary;
- evidence JSON;
- approved qualification policy;
- qualification manifest and gate result;
- SHA256 of the final distributable bundle.

A later L0 keyword-only update is a new qualification tuple because the `.kwk` hash changed, even when the acoustic `.kwm` stays unchanged.

## Software baseline vs shipping qualification

The repository may publish a **software baseline** when source CI is green. That does not mean a Mandarin wake-word SKU is acoustically qualified. Shipping claims remain blocked until real model/data/board evidence exists; repository issue #2 tracks that external evidence gate.
