# Continuous-audio evaluation

Release KWS quality is measured on continuous audio, not clip classification accuracy.

The supported flow is:

```text
references.jsonl + WAV corpus
       |
       v
build/kws_wav  (real C runtime, .kwm + .kwk)
       |
       +--> detections.provenance.json
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

## Audit split isolation first

Before tuning or final qualification, check that training/mining, calibration and final evaluation do not contain the same decoded PCM under different filenames or WAV wrappers:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.tsv \
  --split calibration=data/calibration.tsv \
  --split qualification=data/eval/references.jsonl \
  --audio-root qualification=data/eval \
  --report build/dataset-audit.json
```

The auditor validates mono 16-kHz PCM16 and hashes the decoded PCM payload. It also retains container-file hashes for provenance. A copied/renamed recording or a WAV rewrapped with different RIFF metadata is still treated as the same audio.

## Run the real runtime

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl \
  --provenance build/detections.provenance.json
```

`kws_wav` is a hosted file-I/O wrapper around the same C engine used by the product. Heap and filesystem use in this executable do not enter the real-time library. The provenance sidecar binds the exact runner/model/pack/reference/detection bytes.

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

### Evidence coverage gates

`run_corpus.py` rejects reference `duration_s` values that differ from decoded
PCM duration by more than one 16-kHz sample. The scorer requires every reference to explicitly
supply `expected` (an empty list means annotated negative audio); omitted
annotations must not silently become negative exposure.

The scorer additionally supports repeatable `--min-expected-per-keyword ID COUNT`,
`--min-negative-hours`, `--min-continuous-negative-seconds`, and
`--max-negative-far-upper-95-per-hour`. FRR gates fail without positive events;
latency gates fail without matched events. Ungated diagnostic summaries retain
legacy zero-valued empty FRR/latency fields, so those values alone are not proof
of quality.

The negative-only upper bound is a one-sided 95% Poisson rate bound using only
recordings explicitly annotated with no expected events. It fails when negative
exposure is absent. Positive audio cannot dilute that denominator. Zero observed
false accepts still gives a positive upper bound (about 2.996/hour for one hour),
so do not substitute a zero observed-FAR target for this confidence-bound target.
This statistical bound assumes representative exposure consistent with a Poisson
count model; repeated identical audio or correlated channels do not create
independent evidence. Run the split/decoded-PCM audit and retain unique source
identities before making an acoustic claim.
A continuous-duration gate requires one sufficiently long negative recording;
many short clips cannot satisfy it by aggregation.

`training/iterate_domain.py` accepts corresponding optional `domain_gates` keys:
`min_expected_per_keyword` (an object mapping canonical uint32 keyword ID strings
to positive integer counts), `min_negative_hours`,
`min_continuous_negative_seconds`, and `max_negative_far_upper_95_per_hour`.
Configured gates apply through `base_gate` to calibration, test candidate
selection and final qualification; missing or non-finite required evidence fails
closed. Existing configurations remain unchanged until a policy is explicitly
set. All domain qualification paths now also require positive `expected` and
`matched` counts to evaluate FRR and latency meaningfully. The far-distance
slice additionally requires a positive integer expected-event count and a finite
FRR in [0, 1]; an absent or empty far slice cannot qualify. Mixed-corpus `far_per_hour` remains a diagnostic and existing gate; neither
it nor synthetic negative-only evidence establishes real-world shipping quality.

The scorer reports:

- total audio hours;
- expected/matched/false-rejected wake events;
- FRR;
- false accepts/hour;
- p50/p95 post-keyword-end latency;
- per-keyword expected/matched/FRR/false-accept counts;
- SHA256 identity for the exact references and detections files.

A detection is matched only to an expected event with the same keyword ID and within the configured pre/post tolerance window. Matching is monotonic and maximizes match count before minimizing total phrase-end timing error, so overlapping windows cannot reuse one detection or cause a simple greedy misassignment.

## Mine hard negatives

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root data/eval \
  --output-dir build/hard-negative-wav \
  --manifest build/hard-negatives.tsv
```

The generated manifest contains empty CTC targets intentionally. Add it as another `--manifest` to `training/train_ctc.py`, using the **same `--tokens` vocabulary** as the base checkpoint:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start build/base.pt \
  --head-only \
  --output build/base-hardneg.pt
```

Never mine from the final held-out certification corpus and then reuse the same corpus as unbiased release evidence. Keep at least three distinct pools: training/mining, tuning/calibration and final qualification.

## Recommended corpus buckets

At minimum include:

- clean positives from many speakers;
- near/far field, angle and SPL/SNR variation;
- near-homophones and partial keyword phrases;
- repeated-syllable and repeated-token confusables;
- ordinary conversational Mandarin;
- TV/music/podcast playback;
- local TTS/speaker playback through the shipped AEC path;
- AEC residual and double-talk conditions;
- motor, fan, gear and mechanical noise from the real product;
- silence and low-level room ambience;
- long negative recordings to make FAR/hour statistically meaningful.

All qualification audio must pass through the same BF/AEC/RES/NS/AGC composition and gain policy used by the shipping SKU.
