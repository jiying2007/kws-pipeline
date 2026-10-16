# Same-backbone CTC / direct-head fitting diagnostic

This is a bounded CPU **objective/head isolation** runner. The two arms share
identical initial causal DS-TCN backbone tensors and the same input windows and
row schedule. A has five CTC tokens; B has three direct classes and terminal-window
cross entropy. It is not an architecture ranking, a replacement for the improved
product CTC recipe, or a continuous-KWS accuracy claim. No real corpus training
was performed while implementing this source.

## Inputs and label meaning

There are two explicitly separate consumers:

* `reviewed-manifests` reuses `verify_training_manifests(diagnostic=False)`,
  `Manifest`, original PCM identity and the existing logmel frontend. Actual
  text/tokens/kind/keyword ID are retained before `Manifest` drops those fields.
  The original `ctc-training-only` receipt is saved unchanged; the additional
  purpose is `verified-actual-label-facts-for-exposed-direct-fit-v1`. Existing
  source rights and production admission are not weakened. Only exposed `train`
  rows are allowed. Full original source PCM is used, not rendered augmentations;
  identical original PCM is deduplicated with every original row binding retained.
* `registry-exposed-fit` reuses the fixed
  `research/kws-candidate-registry-v1/build_registry.py:verify_inputs` and its
  source/archive checks. It never calls a production diagnostic bypass and never
  changes a production resolver root. The explicit purpose is
  `same-pcm-registry-exposed-positive-fit-v1`; original admission flags remain
  false. It requires the four original ZIPs and source-specific retained rights
  documents before the verifier consumes any archive. The registry and verifier
  byte hashes are fixed in `registry_data.py`.

Registry eligibility uses three actual normalized transcripts agreeing on a
canonical keyword, with three complete same-PCM recognizer chains. It never
uses `target_text` or `target_class` as labels. The historical thirteen remain
excluded because their ASR inputs are only joined by sample ID. The current
registry therefore yields **three KW1 rows, zero KW2 and zero background rows**
(`kw1s-05`, `kw1s-06`, `kw1s-07`). All retain `human_gold=false`,
`training_admitted=false`, `source_trust_root_admitted=false` and
`history_complete=false`. This is single-keyword positive fitting only, not
three-class discrimination or independent evaluation. No FAR can be measured.
The former 12-train/4-probe partition is not relabeled: only these three rows
from the old train pool qualify under the stronger input-binding requirement;
none of the old four probes qualify for this adapter.

The registry has **no source-rights document hashes**. This is a missing machine-readable source binding, not proof that legal permission is absent. Do not invent hashes from
model names, prior optimization, or `training_admitted=false`. Supply existing
source-rights bytes plus their exact SHA256 in `rights_bindings`, one binding for each
selected record; those bindings may reference the **same real file/hash** when
that file explicitly covers all selected samples (`source_ids` list). Separate
duplicate permission documents are not required. The existing `retained-asr-source-rights-v1` structure must
identify the selected `source_id` (`kw1s-05`, `kw1s-06`, or `kw1s-07`)
or an explicit `source_ids` list covering the selected sources,
`source_family_id=kw1-preset-supplement-20261014-v1`, a nonempty `license_id`,
and `training_use_allowed=true` without a prohibition. The consumer verifies
these declared rights bytes and source scope; it cannot independently adjudicate
vendor licenses or turn a declaration into production admission. Missing bytes,
unknown permission or a conflicting actual source restriction blocks the run.
The consumer does not create rights documents or fetch audio.

For reviewed inputs, background labels come only from admitted actual negative/
confusable speech or complete no-speech background. Unknown/incomplete labels
never become background. Full canonical positive transcripts map to their one
keyword. Repeated or multi-keyword utterances are not assigned an arbitrary class.
`all-three-classes` requires examples of every class. Explicit
`incomplete-coverage-fit-only` permits narrower fitting while retaining the gaps.
The registry path always requires the latter mode.

## Whole-source windows, not fabricated alignments

Every source sample is retained at sample 0. Declared digital-zero context is
appended until exactly `400 + (window_frames-1)*320` samples are available.
There is no speech cropping, alignment inference, resampling or VAD. This covers
source tail samples that an unpadded frontend would otherwise leave at EOF.
Zero context is not counted as a background training example. Both arms consume
identical 16-kHz PCM / 400-sample frame / 320-sample hop / 32-logmel windows.

The final-frame direct head must observe the entire labeled window: its causal
receptive field must cover all frames, and sparse dilation schedules with holes
are rejected. Whole-source presence can then define one *window* label without
inventing word times. CE is applied only at that endpoint, never broadcast across
all interior frames. Streaming interior scores remain untrained as aligned
labels; the saved traces expose them rather than establish continuous-KWS quality.
Sources exceeding a declared window fail instead of being cut. The reviewed
loader caps source length at 30 seconds and aggregate fixed features at 100,000
frames. All topology and optimizer budgets must be explicit.

## Running after inputs and a new bounded diagnostic are approved

From the repository root with the existing numerical runtime:

```sh
python -m research.direct_fit.runner --config /local/config.json \
  --output /local/preflight --preflight-only
python -m research.direct_fit.runner --config /local/config.json \
  --output /local/new-run
```

`registry-inputs.example.json` is an input template, not a selected recipe or
execution authorization. Its placeholder rights hashes must be replaced with
real retained bindings. Paths are relative to the process working directory.
Each output directory must be new. `--preflight-only` verifies inputs and builds
features but performs zero optimizer calls. There is no downloader, ASR call,
resume path, threshold search, old research-budget reuse or automatic retry.

One paired step is one CTC and one direct Adam update, with shared LR, weight
decay `1e-4`, gradient clip 5, batch size 1 and fixed row order. The heads have
different shapes; only shared backbone tensors are identical. CTC uses PyTorch
mean reduction (per-target-length normalization), and CE uses one endpoint;
their raw loss scales are not directly comparable. The fixed seed, topology,
window, schedule and event policies are recorded. The soft wall limit is checked
between pairs, not a hard subprocess deadline. Failure halfway through a pair
retains unequal actual arm counts rather than claiming a complete paired step.

## Persistent outputs and comparison

Each run keeps `ctc.initial.pt`, `direct.initial.pt`, both final checkpoints,
`optimization-trace.jsonl`, full config/data/schedule receipts and `run.json`.
Checkpoint publication reuses PR594 `atomic_save_checkpoint`, `state_identity`
and `capture_completion_state`. Optimizer state, RNG, completed epochs, partial
current-epoch offset, actual optimizer counts and initial/final tensor identities
are retained. The endpoint policy explicitly permits partial epochs; it is not
misreported as a completed epoch. A failed save leaves the previous destination
intact and records the failure. These are initial/final endpoint saves, not
periodic crash recovery or power-loss guarantees.

Direct checkpoints say `model_family=direct-kws`, objective
`terminal-window-cross-entropy`, `product_export_compatible=false`,
`promotion_allowed=false`, and `resume_supported=false`. The product CTC exporter
rejects them explicitly before writing. The direct checkpoint never contains a
five-token projection or CTC label head. `load_model` checks source and tensor
identities and recreates the correct model family.

Inference reloads the saved final checkpoint bytes. It writes actual per-frame
logits/probabilities and actual decoder events, preserving model and decoder state
across chunks and resetting only at a recording boundary. Timing is from sample
0 of the declared padded training window. These are fitted-window diagnostics,
not natural unpadded recordings. Raw continuous inference can reuse `load_model`
with the existing `iter_wav_features` / `stream_events` bridge; its sample-0 clock
and unpadded EOF behavior must remain explicit.

Both arms use the same `eval/score_events.py` references and scorer. Different
CTC/direct decision policies and fixed thresholds are recorded, not described as
identical decoders or matched calibrated operating points. Automatic comparison
uses only clip presence: no word boundaries, FRR, FAR or latency are invented.
For separately retained complete acoustic event annotations:

```sh
python -m research.direct_fit.compare --references /local/references.jsonl \
  --ctc-events /local/ctc.events.jsonl --direct-events /local/direct.events.jsonl \
  --mode event-windows --pre-tolerance-s 0 --post-tolerance-s 0 \
  --output /local/comparison.json
```

Use `--mode clip-presence` for complete `expected_keywords` annotations without
timing. Even event-window output remains exposed-development diagnostic only;
this entry point does not select a winner or qualify a product.

## Software validation and integration boundary

`tests/test_direct_fit_numeric.py` uses tiny artificial feature tensors and
artificial zero-PCM geometry, not recorded speech. One full suite makes exactly
four CTC plus four direct Adam updates; all admission/registry/right checks are
software fixtures or read-only public metadata. Checkpoints are reloaded and
traces compared across chunk sizes. Historical binding escalation, missing rights,
actual-vs-intended labels, partial budgets, failed saves, tampering, three-class
coverage and exporter rejection are covered.

This next-stage candidate does not modify the preceding frozen nine-file patch.
This stage extends the existing candidate numeric CI scope, its two workflow
filters, and the existing objective-math declaration, with a JSON-only software
fixture receipt. Public workflow retention metadata is refreshed. No new CI
framework or real-data training job is added. Full corpus/continuous-event training and matched-threshold
calibration are separate bounded work once adequate actual labels/background and
source-rights inputs exist.
