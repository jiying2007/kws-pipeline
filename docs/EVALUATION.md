# Continuous-audio evaluation

KWS release quality is measured on continuous recordings, not isolated clip accuracy.

## Inputs

Reference recordings are JSONL rows that own the full duration of one continuous source:

```json
{"recording":"living-room-001","path":"living-room-001.wav","duration_s":3600.0,"expected":[{"keyword_id":1,"start_s":91.2,"end_s":92.1}]}
```

Negative-only recordings use an empty `expected` list. Keep speakers, sessions and source recordings disjoint between train/calibration/final evaluation splits.

Detections emitted by the real C runtime are JSONL rows:

```json
{"recording":"living-room-001","keyword_id":1,"time_s":92.25,"confidence":0.73}
```

## Run the real runtime and preserve provenance

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

The provenance sidecar contains SHA256 for:

- the `kws_wav` runner binary;
- `.kwm` model;
- `.kwk` keyword pack;
- reference JSONL;
- generated detection JSONL;
- recording and detection counts.

This prevents a metric report from being silently reused with a different model/keyword pack or edited detection file.

## Product metrics

```bash
python3 eval/score_events.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --summary qualification/eval-summary.json \
  --false-positives qualification/false-positives.jsonl
```

The scorer reports:

- `FAR/hour`: unmatched detections divided by total continuous audio hours;
- `FRR`: unmatched expected wake events divided by expected wake events;
- per-keyword expected/matched/false-reject/false-accept counts;
- p50/p95 detection delay after the annotated phrase end; detections before phrase end contribute zero post-end delay;
- SHA256 of the exact reference and detection files used for the report.

All durations, timestamps, confidences and gate values must be finite. Invalid `NaN`/`Inf` input is rejected rather than allowed to bypass a numerical threshold.

### Matching semantics

Expected events and detections are matched independently per keyword with a monotonic dynamic-programming assignment. The scorer first maximizes the number of valid matches and then minimizes the total distance between detection time and annotated keyword end. This avoids greedy misassignment when tolerance windows overlap.

Default tolerance is 150 ms before the annotated start and 500 ms after the annotated end. These are scorer defaults, not a product acceptance policy.

The scorer can apply immediate exploratory gates through `--max-far-per-hour`, `--max-frr` and `--max-p95-latency-ms`. Formal product release uses the separate qualification policy described in `docs/RELEASE_QUALIFICATION.md` so the measured evidence remains immutable.

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

Positive coverage should be bucketed by speaker, distance, angle, SPL/SNR, speaking rate, device state and relevant acoustic-front-end modes so aggregate FRR cannot hide a weak field domain.

## Hard-negative feedback

False wakes should feed the next calibration/training iteration rather than being manually forgotten:

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives qualification/false-positives.jsonl \
  --audio-root qualification/audio \
  --output-dir build/hard-negatives \
  --manifest build/hard-negatives.tsv
```

The generated TSV contains empty CTC targets and can be mixed directly with normal training data:

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

For every shipping model/keyword-pack tuple retain:

- model/keyword-pack/token/config SHA256 and vocabulary fingerprint;
- evaluation runner SHA256;
- references and detections plus their hashes;
- evaluation provenance and summary;
- false-positive output;
- corpus version/identity;
- target-board performance and resource evidence;
- approved product qualification policy;
- qualification manifest and gate result.

Hosted CI and cross compilation are correctness signals, not target-board acoustic qualification.
