# Matched CTC research pilot (2026-10-14)

## Status and boundaries

The initial source-only preparation was followed by a successful numerical smoke
run [38552529009](https://github.com/jiying2007/kws-pipeline/actions/runs/38552529009).
Its retained [numeric receipt](../effect-data-model-v2/evidence/numeric-run-38552529009/numeric-receipt.json)
records **actual artificial-PCM fixture training and inference**: two optimizer
updates per arm, complete-stream versus chunked parity at chunk sizes1/2/7/25,
source-binding rejection, and execution of the score interface. This is
`artificial-pcm-numeric-contract-only` evidence, not a speech-quality experiment.

**Matched RNN/C1 quality training and evaluation on newly admitted data, C1 C
export/target parity, and SSC305 measurements remain NOT_RUN.** No quality metrics
or shipping approval are claimed. The invented receipts inside unit tests remain
arithmetic fixtures; they are distinct from the retained actual numeric run.
This addition does not edit a product config, frozen model, registry, or r1–r5
artifact and does not enable a workflow or disabled historical research plan.

The product model is `TinyStreamingRNN`: 32 frontend features, 64 tanh recurrent
units, five CTC token logits. It is not a GRU. C1 already exists in
`research/candidate_encoders`: input Linear/ReLU, causal depthwise/pointwise
residual temporal blocks, output Linear, explicit bounded streaming history.
There is no existing C1 `.kwm` exporter, C kernel, quantization or target parity.

## Smallest useful experiment

1. Train a fresh RNN control and C1 challenger on the **same newly admitted data**.
   The historical model is a regression anchor, not a matched training control.
2. Use the same unpadded batch-size-one CTC objective, Adam, learning rate,
   seed, shuffled sample schedule, sample exposures and CPU time cap. No
   architecture-specific auxiliary objectives are introduced in this pilot.
3. Freeze calibration thresholds before selection. Run identical continuous
   recordings through both arms and the same research decoder. Score each word's
   FRR, FAR events/hour, duration/denominators and latency separately.
4. No winner if either arm exceeds the common FAR/resource limits, misses the
   exposure budget, or regresses either wake word. The receipt comparator has no
   promotion authority. Event counts and uncertainty remain necessary; observing
   zero events on a short stream cannot certify a low population FAR.
5. Synthetic/TTS-only results establish plumbing and at most development
   direction. They do not reopen the paused architecture search as a product
   decision. Fresh human/final-AFE evidence and independent negative streams are
   still needed. Historical two-FA replay remains development regression only.

The provisional fixed C1 is width 32, four kernel-5 blocks at dilations 1,2,4,8.
This is a local reduced design, not a published DS-TC-ResNet reproduction. At
50 frames/s, symbolic dense MACs/frame are 5,920 vs RNN64's 6,464. Float32
parameter bytes are 24,852 vs 26,132; persistent model-state bytes are 7,680 vs
256. C1 has 60 frames of past receptive context (1.2 s), no future context.
These exclude frontend/decoder, transient activations, allocation and memory
traffic. They are not CPU, end-to-end latency, RSS or board measurements.

## Dependencies and execution ownership

Use the existing approved CPU environment. `training/requirements.lock` pins
Python 3.12.11 and Torch 2.13.0; `training/Dockerfile` requires an immutable
`KWS_TRAINING_BASE` OCI digest. The existing objective-math CI also pins
numpy 2.3.2 and CPU Torch 2.13.0. Do not invent an OCI digest or silently upgrade
packages. The runner uses no pretrained download, GPU or paid service.
`model.py` pins one thread and deterministic CPU dispatch before importing Torch.

The root executor owns training/inference. No such commands were executed during the initial
source-only preparation; the later fixture-only numeric run is recorded above. A conservative **pilot limit**, not a speed prediction, is
2 epochs and 600 seconds per arm. Runtime and memory need measurement on actual
input. `process_peak_rss_bytes` is the cumulative process high-water mark, not an
independent per-arm resource comparison; use fresh processes/external accounting
for fair RSS conclusions. One long sample may finish after the wall-clock cap;
the runner checks its cap between optimizer steps.

## Run the real research training path

Copy `pilot.example.json` to a private working directory, replace absolute paths,
and choose approved bounds. Every manifest row must include speaker_id,
session_id, source_id, and a valid actual WAV/target token list. The existing
Manifest loader checks PCM16/mono/16k and CTC-alignable length. Actual PCM hashes
are measured, not trusted from manifest text. Train/calibration/selection must
have disjoint speaker, session, source and PCM identities. Keep augmentations
with their source group. TTS voice IDs should be treated as speaker groups.

`synthetic_diagnostic_only: true` explicitly uses the existing non-promotable
synthetic admission lane. It is not a bypass for reviewed product evidence.
Set false only when training manifests satisfy the repository's actual-label,
lineage and source-authority admission. The runner verifies unchanged manifest,
token and PCM identities again before saving.

```sh
timeout 7500s python research/model_comparison/runner.py --repo "$REPO" train \
  --config "$PRIVATE/pilot.json" --output "$PRIVATE/ab-run-01"
```

Outputs are two `*.research.pt` checkpoints and a training receipt with input,
code, split, schedule, runtime, exposure and timing identities. Output directory
must not already exist. Neither checkpoint is passed to the product exporter,
including C0: this standalone CTC-only recipe does not fabricate required product
checkpoint provenance. Incomplete runs are marked and exit nonzero.

## Continuous Python inference and event scoring

`keywords.example.json` has IDs matching the existing two words and illustrative
thresholds only. Calibrate on calibration streams; save/hash the resulting file
before opening selection labels. Inference reads WAV incrementally, preserves
frontend overlap/model history/decoder state across chunks, resets only at the
recording boundary, and does not pad/flush EOF. Model source hashes are verified.
Preflight inspects RIFF chunk sizes without loading PCM, rejecting odd declared
PCM16 data sizes and payloads outside the file or RIFF extent. The stream reader
also checks actual PCM byte coverage against the WAV header, rejecting truncation
after preflight and incomplete PCM16 samples. A complete short
tail without a full feature window is still valid and is not padded. A failed
stream may leave partial detection output, but no completion receipt is written;
do not score that partial output as a completed evaluation.

```sh
timeout 3600s python research/model_comparison/runner.py --repo "$REPO" infer \
  --checkpoint "$PRIVATE/ab-run-01/control.research.pt" \
  --wav "$PRIVATE/selection-continuous.wav" --recording selection-continuous \
  --keywords "$PRIVATE/control-calibrated-keywords.json" \
  --output "$PRIVATE/control-selection-events.jsonl" --chunk-frames 25
# Repeat with treatment and its calibration-only thresholds, same WAV/recording.
python eval/score_events.py --references "$PRIVATE/selection-references.jsonl" \
  --detections "$PRIVATE/control-selection-events.jsonl" \
  --summary "$PRIVATE/control-selection-summary.json"
```

The reference recording ID must match, and reference annotations must come from
independent known event windows, not inferred text alignment. Existing clip-only
annotations require `--clip-presence`; do not invent event timing from them.

The greedy CTC suffix decoder is explicitly a research comparator. It uses
nonblank collapsed token sequences, minimum token confidence, a bounded 150-frame
history and a 50-frame per-word cooldown. It has no C beam-search, VAD or product
operating-point parity claim. Do not compare its numbers against C-runtime numbers
as if only the architecture changed. Threshold tuning is manual/externally
orchestrated; this code does not use selection outcomes to choose thresholds.

## A: actual product RNN path and one-variable ablation

For the nearest deployable improvement, reuse the product trainer and exporter,
with reviewed data and separate output paths. `rnn-negative-policy-pair.json`
changes only supported `train.sequence_margin_negative_policy` from sparse to
runtime-executable. This is a testable hypothesis, not an asserted improvement.
The repository pair materializer validates the allowed override and preserves
all other effective settings; its verify command was run successfully.

```sh
python training/product_development_pair_spec.py verify \
  --pair-spec research/model_comparison/rnn-negative-policy-pair.json
python training/product_development_pair_spec.py materialize \
  --pair-spec research/model_comparison/rnn-negative-policy-pair.json \
  --effective-config "$PRIVATE/admitted-effective.json" \
  --output-root "$PRIVATE/rnn-policy-pair" --base-sha "$BASE" --head-sha "$HEAD"
```

For direct RNN64 training, use the existing admitted manifest and keep all other
CLI options identical between arms. This example shows the treatment variable;
use `sparse-chronological-v1` for control. Preserve the effective recipe's epochs,
weights and auxiliary settings rather than silently replacing them with defaults.

```sh
python training/train_ctc.py --manifest "$PRIVATE/train.jsonl" \
  --tokens keywords/tokens.example.txt --keywords keywords/zh_cn_example.tsv \
  --feature-dim 32 --hidden-dim 64 --seed 20261014 \
  --sequence-margin-negative-policy runtime-executable-v1 \
  --output "$PRIVATE/rnn-treatment.pt"
python training/export_model.py --checkpoint "$PRIVATE/rnn-treatment.pt" \
  --tokens keywords/tokens.example.txt --output "$PRIVATE/rnn-treatment.kwm"
cmake -S . -B "$PRIVATE/build" -DKWS_BUILD_TOOLS=ON
cmake --build "$PRIVATE/build" -j2
python eval/run_corpus.py --runner "$PRIVATE/build/kws_wav" \
  --model "$PRIVATE/rnn-treatment.kwm" --keywords "$PRIVATE/calibrated.kwk" \
  --references "$PRIVATE/selection-references.jsonl" --audio-root "$PRIVATE" \
  --detections "$PRIVATE/rnn-treatment-events.jsonl"
```

Confirm actual built runner name (`kws_wav` in current CMake) and use the existing
keyword compiler for a candidate-specific calibrated `.kwk`. Existing
`eval/long_far_stream.py` accepts explicit model/keywords/runner/seconds/seed and
can add a negative manifest. Its generated synthetic mix is only development
stress; do not substitute it for independent real continuous negatives.

## Checks

```sh
python -m unittest discover -s tests -p test_model_comparison.py -v
python -m py_compile research/model_comparison/*.py
```

24 stdlib tests passed during preparation: leakage on four independent identity
axes, protected split rejection, seed/data/exposure matching, failed resource/FAR
gates, per-word regression, invalid denominators, and decoder token/blank/age/
cooldown/threshold/reset rules. This does not validate Torch execution, gradients,
convergence, streaming model equivalence, C export or wake quality. Execute the
existing candidate numeric suite and an actual small-data pilot before interpreting
any newly generated checkpoints.

## Opt-in hosted numeric smoke (real Torch, artificial PCM)

`tests/test_model_comparison_numeric.py` executes the real training CLI, with two
optimizer updates per arm on explicitly artificial arithmetic PCM, then loads the
actual saved checkpoints with `weights_only=True`. It asserts parameter updates
and finite parameters, matches whole-sequence logits with the exact shared WAV
streaming feature path at chunk sizes 1/2/7/25, checks 11 complete frames with a
partial final window and zero frames for a 399-sample WAV, runs both inference
CLIs at two chunk sizes, invokes the existing event scorer, and rejects a
checkpoint with a corrupted source hash. The generated receipt is strictly a
numeric/integration contract, never acoustic or model quality evidence.

The orchestrator imports no Torch. Each CLI and the numerical worker run in a
clean subprocess, with one `runtime()` call per process, avoiding repeated
`set_num_interop_threads` configuration. All subprocesses share a 120-second
overall deadline; no training/inference is launched on unittest discovery.
Missing/wrong Torch or numpy fails the explicit test rather than skipping it.
There is no dependency installation, model download, or pretrained weight use.

After the existing pinned objective-math environment is prepared:

```sh
timeout 120s .venv-objective/bin/python tests/test_model_comparison_numeric.py \
  --repo . --receipt build/model-comparison-numeric-receipt.json
```

Use an independent CI job with the existing pinned action/runtime setup and a
job timeout covering dependency installation plus this bounded test. Upload the
numeric receipt only after success. During the initial source-only preparation,
only the 24 stdlib tests and py_compile had run; numeric execution was NOT_RUN
at that historical stage. The later actual run38552529009 and its retained
[exact-source numeric receipt](../effect-data-model-v2/evidence/numeric-run-38552529009/numeric-receipt.json)
now establish fixture-only numeric PASS, as summarized at the top of this page.
This does not change the NOT_RUN status of matched quality A/B on newly admitted
data, C1 Cexport/target parity, or SSC305 measurements.

The pilot deliberately fixes **logmel** in both arms and records it in each
checkpoint. This does not claim equivalence to an arbitrary current shipping
frontend, particularly PCEN (whose smoother would require persistent frontend
state). The inspected model-749187ec1d66 provenance says logmel/frontend kind 0,
but a fixed frontend name alone is not numerical or end-to-end product parity.

## Review hardening

Unknown identity placeholders (case/space variants of unknown, none, pending,
null, n/a and prefixed placeholder forms) are rejected. The split guard now
unions connected components using speaker/session identities, original/rendered
PCM hashes and source/derivation families. The runner preserves these declared
links from raw manifest rows, nested speech_like_provenance and lineage_links;
it no longer discards them through Manifest's narrower projection. source_id,
source_family and derivation_family share a source namespace; source_pcm,
source_pcm_sha256 and measured pcm_sha256 share a PCM namespace. This connects
originals, augmentations, aliases and multi-step derivations across splits.

Current WAV PCM is always measured. When source_path is supplied, its PCM is
also measured and declared source hashes must agree. Claimed source metadata
without a readable source_path is an additional conservative grouping link, not
proof of authenticity or measured origin. Existing reviewed-data admission is
still required outside the explicit synthetic diagnostic lane.

Checkpoint source binding requires exactly all eight fixed repository source
paths plus research_runner and comparison_contract, each a lowercase 64-hex
SHA-256 matching current bytes. Empty maps, omitted keys, extras, malformed
hashes and mismatches fail before model construction/inference output.

Defensive pilot bounds: 100 epochs, 100,000 total exposures, 10,000 rows/split,
16 MiB manifests, 30-second training/calibration/selection clips, 300-second
referenced original source clips, 1–256 inference frames/chunk, 24-hour stream
headers, 64 MiB checkpoint, 64 KiB keyword metadata, and 3,600 seconds/arm.
Topology limits are 65,536 parameters, 16,384 persistent state elements and
16,384 symbolic MACs/frame. These are resource safety limits, not SSC305 budgets.
The deterministic sample schedule is a lazy per-epoch iterator (only one shuffled
index list), hashed incrementally; epochs are not expanded into a giant list.
Run training/inference through the explicit external timeout commands above;
only a supervisor can stop a blocked kernel or input operation mid-step.

Regression coverage adds unknown IDs, source-family augmentation leakage,
original-to-derived PCM/source aliases, transitive/nested provenance links,
measured-PCM mismatch, exact source key/digest checks, schedule/chunk/audio bounds.
The opt-in real-Torch smoke additionally rejects an empty source-hash map.

### Actual data-split module bridge

A's fields are explicitly covered: source_family_id, derivation_family_id and
family_id normalize to the same source-family namespace as the legacy non-_id
aliases. original_pcm_sha256/source_pcm_sha256/reference_pcm_sha256/current PCM
share the PCM namespace. reference_audio_sha256 is a **file** SHA-256 (confirmed
with the data-module contract), grouped with measured file_sha256 and
source_wav_sha256/source_audio_sha256/original_audio_sha256, never conflated with
PCM. Measured current file/PCM claims must match actual bytes.

```sh
python tests/test_model_comparison_split_bridge.py --repo .
```

This no-Torch integration test imports the actual training/effect_dataset_split.py,
executes A's plan(), passes its returned rows through B's projected_identity(),
and deliberately tests unsafe cross-split reassignment against validate_split().
It covers six actual upstream grouping/hash fields and reference-file-to-measured-
file linkage. It asserts the upstream GROUP_FIELDS and PCM_FIELDS vocabulary is
fully supported, so new upstream fields cannot silently disappear. The test was
run against the data worker's actual source and passed; 24 unit tests also pass.

An upstream plan is not an admission receipt or permission to repurpose historical
holdouts. A bundle containing only train and calibration remains blocked until a
legitimately independent selection group is available. Do not duplicate
calibration, consume protected test/qualification, or fabricate speaker/session
identity from a voice ID merely to satisfy the pilot contract.

### Receipt stream identity checks

The comparator requires canonical lowercase SHA-256 digests for calibration and
selection streams. Both arms must match within each role, and the two roles must
have different digests: explicitly reusing calibration as selection is rejected,
even when the receipt claims calibration-only thresholds. These checks reject a
contradictory receipt; different digests do not authenticate measurements or prove
that the streams have independent sources or no partial overlap. Existing split
lineage checks and independent evidence admission remain required.
