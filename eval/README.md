# Continuous-audio evaluation

Release KWS quality is measured on continuous audio, not clip classification accuracy.

The supported flow is:

```text
references.jsonl + WAV corpus
       |
       v
build/kws_wav  (real C runtime, .kwm + .kwk)
       |
       v
detections.jsonl
       |
       v
eval/score_events.py
       |
       +--> summary.json: FAR/hour, FRR, latency, per-keyword metrics
       |
       +--> false-positives.jsonl
                   |
                   v
          eval/mine_hard_negatives.py
                   |
                   v
          empty-target CTC manifest
```

## Reference format

One JSON object per recording:

```json
{"recording":"living_room_001","path":"negative/living_room_001.wav","duration_s":3600.0,"expected":[]}
{"recording":"positive_001","path":"positive/positive_001.wav","duration_s":12.0,"expected":[{"keyword_id":1,"start_s":3.1,"end_s":4.0}]}
```

Recording IDs must be unique. WAV inputs must be uncompressed mono PCM16 at 16 kHz. `expected` may be empty for negative recordings.

## Run the real runtime

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl
```

`kws_wav` is a hosted file-I/O wrapper around the same C engine used by the product. Heap and filesystem use in this executable do not enter the real-time library.

## Score release metrics

```bash
python3 eval/score_events.py \
  --references data/eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/summary.json \
  --false-positives build/false-positives.jsonl \
  --max-far-per-hour 0.2 \
  --max-frr 0.05 \
  --max-p95-latency-ms 500
```

The numeric gates above are examples, not universal product requirements. Define them from the actual use case and acoustic test plan.

The scorer reports:

- total audio hours;
- expected/matched/false-rejected wake events;
- FRR;
- false accepts/hour;
- p50/p95 post-keyword-end latency;
- per-keyword expected/matched/FRR/false-accept counts.

A detection is matched only to an expected event with the same keyword ID and within the configured pre/post tolerance window. Each detection can match at most one expected event.

## Mine hard negatives

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root data/eval \
  --output-dir build/hard-negative-wav \
  --manifest build/hard-negatives.tsv
```

The generated manifest contains empty CTC targets intentionally. Add it as another `--manifest` to `training/train_ctc.py`.

Never mine from the final held-out certification corpus and then reuse the same corpus as unbiased release evidence. Keep at least three distinct pools: training/mining, tuning/calibration and final qualification.

## Recommended corpus buckets

At minimum include:

- clean positives from many speakers;
- near/far field, angle and SPL/SNR variation;
- near-homophones and partial keyword phrases;
- ordinary conversational Mandarin;
- TV/music/podcast playback;
- local TTS/speaker playback through the shipped AEC path;
- AEC residual and double-talk conditions;
- motor, fan, gear and mechanical noise from the real product;
- silence and low-level room ambience;
- long negative recordings to make FAR/hour statistically meaningful.

All qualification audio must pass through the same BF/AEC/RES/NS/AGC composition and gain policy used by the shipping SKU.
