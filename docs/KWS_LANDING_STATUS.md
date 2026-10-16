# KWS product landing status

## 2026-10-16 data/model roadmap checkpoint

Read-only evidence audit at `a7015ff61947b423ed0162b692a39f7590255941`.
The seven-part data/model roadmap is **not end-to-end complete**. Source and
archive checks are useful completed work; they do not establish a new deployable
candidate, independent acoustic gains, or product qualification. The dated
historical sections below are retained unchanged. Where an older preparation
README says NOT_RUN, use the later exact result receipt for that specific run.

| Original outcome | Current evidence-backed status | Next outcome needed |
| --- | --- | --- |
| Standard/accent/near-pronunciation/uncertain labels and two product policies | **Partial.** Explicit alternative-pronunciation policy and independent listening template exist in [effect_evidence.py](../eval/effect_evidence.py); a completed four-category benchmark and measured conservative/accent-friendly comparison are not established by the inspected evidence. | Freeze the intended event labels and collect their independent references. Playback of a complete target word remains a product-semantic decision. |
| Source-family/speaker splits; machine labels separate from human references | **Partial.** [Grouped split implementation](../training/effect_dataset_split.py) and the [measured inventory](../research/effect-data-inventory-v2/README.md) preserve lineage and historical use. The 385-file recovery snapshot is all development, with no training or sealed rows and no speaker/session IDs; it is not the complete project corpus. | Recover supported metadata without inventing identities; obtain legitimately independent calibration and selection support. |
| Qwen0.6/Qwen1.7/CosyVoice useful-yield comparison | **Partial.** [Actual Qwen screening](../research/tts-pilot-screening-v1/evidence/asr-run-38573117922/stratified-analysis.json) records 7/16 and 6/16 three-ASR candidates. The [supplement](../research/kw1-preset-supplement-observation-v1/README.md) added three KW1 candidates. CosyVoice3's four admissible reference voices remain unresolved in the [pilot](../research/tts-quality-pilot-v1/README.md). | Human phonetic/naturalness reference and a properly bounded three-source comparison; machine yield is not a quality ranking. |
| Hard negatives, independent continuous background, paired AFE | **Partial.** [AliMeeting intake](../research/alimeeting-continuous-intake-v1/RESULT.md) acquired eight complete sessions / 4.205059722 hours. Its frozen-C pass had zero triggers, but negative-admitted duration remains zero. The [same-tuple machine-positive control](../research/alimeeting-continuous-intake-v1/machine-positive-control-v1/README.md) was 0/13. | Complete transcript/boundary and lineage review before negative admission; obtain difficult negative labels and final-AFE paired evidence. |
| CTC versus direct lightweight KWS A/B; optional second stage | **Core comparison not completed.** The [current paired runner](../research/model_comparison/README.md) has actual artificial-PCM numeric smoke, but compares RNN CTC with reduced causal DS-TCN CTC. It is not a completed DS-TC-ResNet/SVDF direct-KWS comparison. | One matched acoustic comparison with reusable checkpoints, followed by deployment work only for a supported candidate. |
| FRR at matched FAR, continuous FAR/confidence interval, real far field and SSC305 | **Core measurements not completed.** The [event scorer](../eval/README.md) supports denominators, per-keyword misses and negative-rate upper bounds. Current new-candidate independent matched-FAR, real far-field and target-board results are not established. | Independent positive/negative exposure, calibration-only threshold selection, frozen event evaluation and actual SSC305 measurements. |
| Multiple seeds, quantization, sealed acceptance | **Not completed for the current candidate.** [LR600 acceptance](../research/kws-lr-stability-v1/result-20261016/acceptance-summary.json) records one seed, 12 exposed machine-labelled training samples, failed raw-greedy fit and no reusable weight export. | A retained candidate, multi-seed confirmation, export/runtime parity, quantization assessment and fresh final acceptance. |

The [latest saved diagnostics](../research/kws-lr-stability-v1/saved-diagnostics-20261016/README.md)
retain a posthoc 12/12 beam result and a separate 12/12 native-decoder result.
The latter used a synthetic constant-one speech gate and uncalibrated .55
threshold; `production_engine_executed=false`. It does not close full-engine
sensitivity, negative-stream FAR, real VAD, generalization or deployment.

### Minimal data preparation without another model run

### 2026-10-16 12:29 UTC text recovery update

The eight original TextGrid bodies have now been recovered locally from a
[pinned public nonofficial mirror](https://github.com/quietcricket/meeting-transcription-dataset/tree/e9bd23a5e6220da17aa09fe7629cf940bf45453b/alimeeting-eval).
All eight SHA256 values and the 711,382-byte total match the prior official-intake
[audit](../research/alimeeting-continuous-intake-v1/audit/audit.json); the mirror is
only a transport source, not an official publisher. A text-only rerun of the
exact prior v5 parser reproduced every retained per-session result: 25 tier
instances, 6,457 intervals and zero literal flags for the seven screening phrases.
No audio was read or revalidated and no admission label was changed.

The recovered tiers contain 25 distinct anonymous annotation labels with no
label repeated across these eight files. This does not independently establish
25 real people or speaker disjointness from historical training. Combining all
tiers on each session clock gives 1,711 uncovered annotation gaps totaling
1,165.345 seconds, including the previously known tails. These are pointers for
coverage review, not confirmed silence or admitted negative exposure. Detailed
local recovery/hash/interval receipts are delivered separately; corpus bodies
and those reports are not included in this repository status-only change.

#### Earlier 2026-10-16 metadata-only checkpoint

The following three paragraphs preserve the earlier same-day inspection before
text recovery; their missing-body statements are superseded by the update above.

The [existing AliMeeting audit](../research/alimeeting-continuous-intake-v1/audit/audit.json)
pins the eight original TextGrid members and their hashes; their total retained
member size is 711,382 bytes. Recover those eight text files from the already
acquired archive/retention location first; a fresh multi-gigabyte audio download
is not required for this metadata step. The current read-only review did **not**
reopen the original transcripts or audio. It inspected the retained audit and
[v5 text-scan receipt](../research/alimeeting-continuous-intake-v1/textgrid-rescan-v5.json).

The prior scan reports 25 tier instances, 6,457 intervals, and no literal match
for its seven listed target/near-target phrases. These are prior text-scan
findings, not newly verified acoustic negatives. All eight TextGrids end before
their WAV, by 47.384 seconds in total. Sparse per-speaker gaps and those tails
need an explicit review disposition; a gap is not proved silence. Retain complete
sessions and unknown labels rather than trimming unreviewed regions away.

Session IDs and original/derived source hashes can be recovered from this audit.
It retains tier counts but not tier names, so it cannot restore actual speaker
labels or prove speaker independence across meetings. Near-channel filenames,
room-looking prefixes, voice presets and synthesis seeds must not manufacture
speaker/session truth. Even a later original-tier-name recovery would require
a documented cross-session identity interpretation and training-exposure audit.

Use the existing grouping/admission/scoring paths. Prioritize a small independent
development benchmark and one checkpoint-preserving acoustic comparison over
another analysis of the same 12 training LPs. Research software work can proceed
with appropriately scoped authorized synthetic/public data; human/final-AFE and
physical-board product qualification remain separate requirements. No existing
admission labels, protected split, consumed experiment budget or shipping policy
is changed by this status note.


Run the existing source-contract report:

```bash
python3 tools/kws_landing_status.py --verify
```

This command verifies the repository's pinned source contract and local model
registry bytes. It does not query live Actions, Releases, restricted real-human
evidence or physical DUT receipts, run a model, or recalibrate thresholds.
`--verify` checks the status contract; a successful exit is not acoustic or
shipping approval. The output keeps `assessment_scope=repository-source-contract-only`
and `live_qualification_checked=false`.

## Read each evidence scope separately

JSON schema v2 replaces the unscoped v1 `evidence` and `research` objects:

- `historical_release_qualification`: the frozen model release's recorded
  synthetic qualification (`256/256`, `0 FR`, `0 FA`) and robustness pass.
  `applies_to_current_source=false`; this does not qualify changed runtime source
  or establish that later regression passed.
- `current_source`: directly projects `threshold_calibration.recalibration_required`
  and its reason from `configs/shipping.xiaowo.json`. It currently requires
  recalibration. `synthetic_qualification_checked=false`; this command performs
  no acoustic evaluation and cannot clear that requirement.
- `dated_regression_and_research`: links the existing dated evidence snapshot
  with its SHA-256, plus the current research and admission entry points. It does
  not copy outcomes into another status ledger or pretend to query live results.
- `external_qualification`: `real_human_final_afe_passed=false` and
  `physical_target_board_passed=false` describe the source contract's pending
  gates, not a live external qualification lookup.
- `historical_research_closure`: the retained pause and next-lane decision belong
  to that closed research line. They neither authorize nor block current work;
  use the current admission entry below.
- `control_plane`: readiness here means required repository files exist. It does
  not mean private inputs, a controlled runner or a physical DUT are available.

The trained deployable-format model and its pinned Git registry mirror are
available. `product.shipping_approved` remains **false**. The next external product
gate remains **real-human final-AFE acoustic qualification**, followed by
**physical target-board performance and soak**. Source recalibration is a separate
unresolved requirement, not evidence that those external gates have run.

## Retained regression and research observations

The [2026-10-09T02:43:25Z snapshot](../research/consolidation/CURRENT_STATUS_2026-10-09.md)
records distinct outcomes:

- D20 raw-logit numerical gate: **FAIL**; D90: **NOT_RUN**
- Fixed50: **EVAL_INCONCLUSIVE_LABEL_SUPPORT**, with insufficient K2/target support
- Old frozen-model [nightly run 37860130019](https://github.com/jiying2007/kws-pipeline/actions/runs/37860130019):
  **2 FA / 8 hours**, upper95 **0.786974 FA/h**, failing 0 FA / upper95 ≤ 0.40
- Human/final-AFE and physical-board qualification: **deferred**

These are dated observations, not results generated by the status command. The
old model's nightly failure is separate from D20/D90, and finite-sample historical
zero counts establish no zero-risk claim. Older PR/publication state in that
snapshot remains historical; see the [selected archive delivery](../research/consolidation/ARCHIVE_DELIVERY_2026-10-09.md)
for the subsequent delivery decision.

## 2026-10-14 current source entry

The completed [r5 run 38518609834](https://github.com/jiying2007/kws-pipeline/actions/runs/38518609834)
is summarized in the [immutable-source-derived terminal result](../research/k1-punctuation6-asr18-v1/RESULT-38518609834.json).
[PR #570](https://github.com/jiying2007/kws-pipeline/pull/570) and
[PR #571](https://github.com/jiying2007/kws-pipeline/pull/571) retain its publication
and closed-source follow-through. The executed source was
`a1159dc902634e65b1fb2a7bbf219389f84089d5`, distinct from the later closed main.

- All **18 slots** have verified native completion on the original full inputs.
  This verifies decoder/process completion, not acoustic verbatim correctness.
- Target three-model unanimity is **0/3 without comma and 0/3 with comma**.
  The sole unanimous machine candidate is the non-target `你好小屋`; it is not a
  target improvement or human label. The six original cells and failed r4 result
  remain sealed, with no retry or training admission.
- All **nine** current execution/publication/claim approval flags are false.
  **108 planned slots** across sealed campaigns are nonreusable; this count is
  not measured consumption, which remains `null`.

The later frozen-model [nightly run 38482297930](https://github.com/jiying2007/kws-pipeline/actions/runs/38482297930)
also **failed** on source `1c8d9c049170e9547b39807d77498e10c19712bf`:
**2 false accepts / 8 hours**, point estimate **0.25 FA/h**, upper95
**0.786974202734 FA/h** against **0 false accepts / upper95 ≤ 0.40**. It used
`model-749187ec1d66` with model SHA-256
`ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`.
This dated source/model regression is separate from r5 ASR completion and is not
a new acoustic evaluation of current main. The original 2026-10-09 observation
above is retained unchanged. No source-only CI pass clears FAR, recalibration,
real-human/final-AFE, physical-board or shipping gates.

A later [bounded original-prefix diagnostic](../research/known-far-diagnostic-v1/README.md) restored both failed events with the original-equivalent runtime, retained logits/VAD/decoder state, and keeps the original regression failed. It does not qualify this changed main or clear recalibration.

## Current entry points

Use the [research map](../research/README.md) and its
[unique current admission checklist](../research/d20-diagnostic-admission-v1/PLAN-SCHEMA.md#current-admission-checklist)
for current work. Neither source checks nor dated experiment receipts authorize
new numerical execution, model calls, acquisition, training or qualification.
The [earlier roadmap](KWS_RESEARCH_PRODUCT_ROADMAP.md) and
[execution record](KWS_LANDING_EXECUTION.md) retain their historical context.

The current source-research additions have distinct evidence scopes:

- [Completed fourth Whisper run and durable archive](../research/source-screen32-whisper-v1/evidence/run-38126849922/README.md)
  ([PR #535](https://github.com/jiying2007/kws-pipeline/pull/535)): run
  `38126849922` completed all 16 inputs with verified pre-finalization natural EOT.
  The [three-ASR comparison](../research/source-screen32-whisper-v1/evidence/run-38126849922/comparison/README.md)
  records 6/16 punctuation-only agreement, or 7/16 under the secondary
  punctuation-plus-whitespace policy, including two common target mismatches.
  Agreement coverage is not acoustic accuracy; these rows remain screen-only,
  without human gold or training admission. The Whisper root/attempt READMEs
  are historical preparation snapshots, including superseded unexecuted and
  publication-blocker text. The completed archive above supplies the current
  outcome; the [earlier two-ASR result](../research/source-screen32-v1/evidence/run-38021467257/README.md)
  remains a separate historical observation.
- [Near-pronunciation saved-pair source and contracts](../research/near_pronunciation_v1/README.md)
  ([PR #536](https://github.com/jiying2007/kws-pipeline/pull/536)) and its
  [isolated synthetic-logit C verification](../research/near_pronunciation_v1/DECODER.md)
  provide research-only tooling. The public integration contains no actual
  saved-trace replay result, and its prepared/NOT_RUN annotations are scoped to
  source preparation. Contract/synthetic tests do not supply ordinary real-world
  negative exposure or independent accent/far-field positives. Matched-FAR and
  recall improvement remain unestablished; model, shipping decoder, thresholds
  and `shipping_approved=false` are unchanged.

Product authority remains `configs/shipping.xiaowo.json` and
`commercial/*.policy.json`. Phase A requires final product microphones/enclosure,
final AFE, real human speech, 3–5 m coverage, rear direction, playback, double-talk,
robot motion and 24 hours of post-AFE negative exposure. Phase B requires physical
target-board evidence across multiple DUTs, including CPU/RTF, RSS/stack, thermal,
power, audio continuity and long soak. These are deferred external qualification
requirements; their absence does not establish a failure of ordinary software tests.

Private raw human audio remains outside the public Git repository. The promoted
model tuple and non-audio evidence have their own explicit retention policies.
