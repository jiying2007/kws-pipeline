# Final English requested-intent embedding experiment

This is one finite, unqualified L1 research experiment: frozen speech features and
one small causal head, one seed, exactly 96 optimizer updates. Its targets are
requested synthetic intents, not verified pronunciations. It does not satisfy or
modify the existing pronunciation-specific positive admission gate. All human
gold, formal training admission, qualification and model promotion flags remain
false. No model is executed by the source tests or by merging these sources.

The current requested names are **Hey Vosen /heɪ ˈvoʊzən/** (class 1) and
**Hey Mero /heɪ ˈmiːroʊ/** (class 2); class 0 is background intent. Old-IPA wake
audio and old English63 classifiers are not positive training inputs here.

## Fixed feature assets

The completed asset-only audit is
[run 43810594841](https://github.com/jiying2007/kws-pipeline/actions/runs/43810594841),
head `511ccee46dc2fc01483b0d598cfd7fefb286bf31`, artifact `14199923535`.
Its ZIP SHA-256 is
`8ca111c19daa711695e40ddb81a558866bbb73a13f37194f7bfe6ae9cc872adf`;
member `oww-assets-audit.json` SHA-256 is
`b11299c12d13155772619c7f985e55c755b35169ca473863ee006be1c3aecc0f`.
That audit executed graph inspection and synthetic-tensor smoke checks, not KWS
training, pronunciation evaluation or deployment qualification.

Upstream is dscripka/openWakeWord release `v0.5.1`, source commit
`368c03716d1e92591906a84949bc477f3a834455`.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| melspectrogram.onnx | 1,087,958 | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` |
| embedding_model.onnx | 1,326,578 | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` |

The audit observed float32 `[1,16000]` PCM input to mel `[1,1,97,32]` and embedding
`[1,76,32,1]` to `[1,1,1,96]`. Input stays on PCM16 scale. The fixed feature recipe
takes the last 76 mel frames, applies `/10 + 2`, and obtains 96 values. Each call
uses only the latest 16,000 observed samples at a 1,280-sample hop. Recomputing a
trailing window avoids whole-recording future dependence through mel ReduceMax;
it is not a claim of parity with upstream's separate streaming implementation.

`FrozenOnnxEmbedding` verifies exact assets, then reads actual ONNX kernel,
stride, dilation, padding and weight metadata before inference. The anticipated
two STFT convolutions must be `[257,1,512]`, stride 160, without padding. Only
after that check can the worker report selected mel input support `[3360,15872)`.
The 35,200-sample PCM history envelope alone is not full acoustic coverage. This
metadata preflight has not been run by the source tests.

## Finite data and weak supervision

The data-owned worker must authenticate the complete new 144-cell source roster
and its observer artifacts before calling this library. Generation status alone
does not select a row. `weak_rows` requires explicit `eligible_cell_ids` and, for
each observed cell, `source_pcm_sha256`, `observer_artifact_sha256`,
`observation_sha256`, `evidence_status` and the original `phone_review_status`.
Only `USEFUL_FOR_INTENT_RESEARCH` can enter this experiment's explicit selection;
`STRONG_CONFLICT`, `UNKNOWN` and missing observations are excluded. Quantity or
stress can remain unverified. This is a research-use decision, not pronunciation
admission. Original `GENERATED_UNADMITTED` receipts are never rewritten as failed.

The selection purpose, IDs and observations have one canonical SHA-256. Every
unselected, unavailable, duplicate or geometry-ineligible cell stays in the
original denominator with a reason. No retry, replacement or clip trimming is
performed. The selected and geometrically supported bags must retain all six
train/development × intent 0/1/2 groups.

Synthetic families are fixed: aiden/dylan/eric/serena train, ryan/vivian
development. They are synthetic voice families, not independent human speakers.
Real background reuses only the already exposed LibriSpeech reference from
`research/english63_online_c/librispeech_screen.py`: 517 original utterances,
40 speakers, 58,646,164 samples. Reuse its exact archive, FLAC, transcript and
speaker checks; do not reuse its global concatenation as a split. The library
hash-orders those 40 speakers into 30 train and 10 development speakers, then
takes at most 16 seconds per speaker in original member order. Joins and source
spans remain explicit. This is exposed read speech, not independent natural FAR.

Each synthetic case has a same-split real background prefix, original untrimmed
WAV, and 5,120 samples of same-split real background context. Positive and ordinary
negative intents use the same construction. Source IDs, offsets and PCM hashes
bind every segment; no invented silence or EOF frames are added. The worker uses
verified mel support to select positive clip bags, and retains an additional
view truncated at the original clip EOF. That view is censored when full support
has not arrived. Scores are research clip/event coverage, not word-end FRR.

## Executable library interface

1. `weak_rows(recipe, source_rows, *, eligible_cell_ids, observations, excluded)`
   selects PCM-bound weak intents and records exclusions.
2. `background_tracks(source_rows)` accepts 517 verified records with `id`,
   `speaker`, `frames`, `pcm_sha256`, `pcm16le` bytes.
3. `assemble_audio_cases(rows, wav_by_id, tracks, excluded=...)` constructs the
   finite real-prefix/clip/real-context streams.
4. `FrozenOnnxEmbedding(asset_directory)` and
   `extract_feature_bags(cases, embedding, excluded=...)` return bags, complete
   observed embedding streams, and exact case evidence.
5. `train_cached(bags, new_absolute_output_directory, input_binding)` performs the
   fixed fit and writes `causal-head.npz` plus `RESULT.json`.

The head is oldest-to-newest `16×96 → 32 ReLU → 3`, with 49,283 parameters and
train-only mean96/scale96. Seed is 20261116; AdamW uses learning rate 0.001, weight
decay 0.0001 and one CPU thread. Each of 96 updates cycles one bag per class.
Positive bags use log mean target probability; background uses mean window cross
entropy. Development data never sets normalization, checkpoint or threshold.
Fresh NPZ bytes are reloaded for NumPy versus Torch replay at logit tolerance
`1e-4`; raw per-bag observations are retained.

`portable_head.py` exports those exact NPZ arrays and checks saved-weight C replay.
The C core contains only head/history/event logic, without allocation. Its fixed
research event rule is probability 0.80, three confirmations, then 13 hops of
cooldown. It is uncalibrated. Full reset clears state; discontinuity clears
acoustic/event history while preserving the sample clock. No head output exists
before 16 embeddings, and no fake EOF output exists. Native/ARM parity is not an
SSC305 board result and does not include an embedded mel/embedding runtime.

The ONNX feature pair is 2,414,536 bytes; the portable float head is 197,908 bytes.
Their 2,612,444-byte sum already exceeds the preferred 2 MiB asset budget, before
runtime code. Retain the measurement; do not claim the budget or board gate passed.

## Source checks

The existing asset workflow runs the two tests below in normal, `-O` and `-OO`
modes. They use invented PCM, arrays and metadata only. Actual asset inspection
in that workflow is restricted to its already consumed audit head.

```sh
python3 -I -S -B research/english_final_embedding/test_prepared_embedding.py -v
python3 -I -S -B research/english_final_embedding/test_portable_head.py -v
```

Actual training belongs to a separately bounded data-owned worker with exact
source, observer, selection, code and asset identities. No real training or new
144-cell generation result is claimed by this source integration.
