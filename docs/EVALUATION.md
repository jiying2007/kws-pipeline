# Continuous-audio evaluation

KWS release quality is measured on continuous recordings, not isolated clip accuracy.

## Reference and detection contract

Reference JSONL owns the full duration of each recording:

```json
{"recording":"living-room-001","path":"living-room-001.wav","duration_s":3600.0,"expected":[{"keyword_id":1,"start_s":91.2,"end_s":92.1}],"domain":{"distance_band":"far"}}
```

Negative-only recordings use `"expected": []`. Keep speakers, sessions, TTS voices, source recordings and device sessions disjoint across train/calibration/test/qualification.

Runtime detections are JSONL:

```json
{"recording":"living-room-001","keyword_id":1,"time_s":92.25,"confidence":0.73}
```

## Run the real C runtime

```bash
python3 eval/run_corpus.py \
  --runner ./build/kws_wav \
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

The provenance sidecar binds runner, model, keyword pack, references and detections by SHA256.

The scorer reports:

- FAR/hour;
- FRR;
- per-keyword expected/matched/false-reject/false-accept counts;
- p50/p95 post-end detection latency;
- exact reference/detection SHA256.

Matching is monotonic per keyword: first maximize valid matches, then minimize total distance to annotated keyword end. Invalid NaN/Inf values are rejected.

## Domain metrics

When reference rows contain domain metadata, add:

```bash
python3 eval/domain_metrics.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --output qualification/domain-metrics.json
```

The report includes overall metrics plus available domain buckets such as:

- `distance:near`, `distance:mid`, `distance:far`;
- azimuth;
- RT60 state;
- noise profile;
- local playback state;
- composite domain ids;
- keyword confusion/miss matrix;
- deterministic worst-domain score.

Aggregate FRR must not hide a weak far/side/reverb/motor/playback domain. Product policy should set minimum corpus coverage separately from the scorer.

The synthetic domain renderer guarantees that every calibration/test/qualification split containing positives begins with a far positive and then rotates `far -> mid -> near`. This is only a deterministic CI/data-orchestration contract. A real product corpus must independently contain enough genuine samples in every required bucket.

## Long continuous FAR

`kws_raw_stream` consumes raw PCM16 blocks through the same C engine without per-clip resets. `eval/long_far_stream.py` uses it to exercise continuous negative streams and preserve false-accept evidence.

`.github/workflows/far-nightly.yml` provides a recurring hosted synthetic regression. It is useful for detecting a decoder/frontend regression that only appears after long state accumulation, but its generated/hosted FAR is **not** a shipping FAR/hour measurement.

Shipping FAR needs long real continuous background audio through the final microphone/enclosure/AFE path.

## Required product domains

At minimum include:

- normal home conversation, near-homophones and partial wake phrases;
- repeated syllables and shared-prefix phrases;
- television/phone/smart-speaker playback;
- own-device TTS/music through shipping AEC;
- AEC residual/double-talk;
- robot motor, gearbox, fan, pump and chassis vibration;
- silence/impulses/microphone handling;
- expected angles and distances, including the product far-field requirement;
- representative RT60/room states;
- day/night gain and final AGC/AEC/NS modes.

Positive coverage should be bucketed by speaker, session, distance, angle, SPL/SNR, speaking rate and device state.

## Failure replay

False accepts can feed training:

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root data/calibration \
  --output-dir build/hard-negatives \
  --manifest build/hard-negatives.tsv
```

Retrain with the exact token vocabulary; the removed size-only interface must not be used:

```bash
python3 training/train_ctc.py \
  --manifest data/base-train.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start models/base.pt \
  --head-only \
  --epochs 5 \
  --output build/base-hardneg.pt
```

False rejects may be replayed with `eval/mine_false_rejects.py`. Never mine from the final held-out qualification corpus and then report the same samples as unbiased final evidence.

## Release evidence

For every shipping model/pack tuple retain:

- model ABI v2 and model provenance;
- keyword-pack ABI v3;
- exact checkpoint/training tokens/training manifests;
- release tokens/config and frontend identity;
- exact evaluation runner/references/detections/provenance/summary;
- domain metrics and false-positive/false-reject outputs;
- corpus identity/version;
- target-board benchmark/resource evidence;
- approved qualification policy;
- qualification manifest and gate result.

Hosted CI and cross compilation are software correctness signals, not target-board acoustic qualification.
