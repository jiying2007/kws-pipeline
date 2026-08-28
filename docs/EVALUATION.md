# Continuous-audio evaluation

KWS release quality is measured on continuous recordings, not isolated clip accuracy.

## Inputs

`eval/score_events.py` consumes two JSONL files.

Reference recordings:

```json
{"recording":"living-room-001","path":"living-room-001.wav","duration_s":3600.0,"expected":[{"keyword_id":1,"start_s":91.2,"end_s":92.1}]}
```

Detections produced by the product/runtime harness:

```json
{"recording":"living-room-001","keyword_id":1,"time_s":92.25,"confidence":0.73}
```

Every reference row owns the full duration of one continuous recording. Negative-only recordings use an empty `expected` list. Keep speakers, sessions and source recordings disjoint between train/calibration/evaluation splits.

## Product metrics

```bash
python3 eval/score_events.py \
  --references eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/kws-summary.json \
  --false-positives build/false-positives.jsonl \
  --max-far-per-hour 0.10 \
  --max-frr 0.05 \
  --max-p95-latency-ms 500
```

The scorer reports:

- `FAR/hour`: unmatched detections divided by total continuous audio hours.
- `FRR`: unmatched expected wake events divided by expected wake events.
- per-keyword expected/matched/false-reject/false-accept counts.
- p50/p95 detection delay after the annotated phrase end. Detections before phrase end contribute zero post-end delay.

The numeric gates above are examples only. Product thresholds must be declared by the SKU qualification plan and must not be copied blindly into a release claim.

## Required negative domains

A production corpus should include at least:

- normal home conversation and near-homophones;
- partial wake phrases and repeated syllables;
- television, phone and smart-speaker playback;
- own-device TTS/music playback through the shipping AEC path;
- robot motor, gearbox, fan, pump and chassis vibration;
- silence, impulsive noise and microphone handling;
- far-field speech across expected angles/distances;
- day/night gain states and representative AGC/AEC/NS configurations.

## Hard-negative feedback

False wakes should feed the next calibration/training iteration rather than being manually forgotten.

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root eval/audio \
  --output-dir build/hard-negatives \
  --manifest build/hard-negatives.tsv
```

The generated TSV contains empty CTC targets and can be mixed directly with the normal training manifest:

```bash
python3 training/train_ctc.py \
  --manifest data/base-train.tsv \
  --manifest build/hard-negatives.tsv \
  --vocab-size 420 \
  --warm-start models/base.pt \
  --head-only \
  --epochs 5 \
  --output build/base-hardneg.pt
```

Do not mine from the final held-out release corpus and then report that same corpus as unbiased validation.

## Release evidence

For every shipping model/keyword-pack pair, retain the model SHA256, keyword-pack SHA256, token vocabulary identity, compiler/toolchain identity, final runtime config, corpus version, FAR/hour, FRR, latency distribution and target-board performance/thermal results. Hosted CI and cross compilation are correctness signals, not target-board acoustic qualification.
