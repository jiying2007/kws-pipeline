# Fixed C1 source augmentation after the balanced first64

This stage compares the actual completed context52 C1 model with one new C1
trained from the **same serialized zero-step initialization** on the old52 plus
all eligible new train rows from the existing first64 cell declaration. It is a
fixed-budget source augmentation experiment. It does not create an independent
test or isolate data quantity from source, text, label confidence and exposure.

The new public adapter is `training/research_c1_source_augmentation.py`. It
imports the existing frontend, `fulltext_ctc`, `research_c1_known_silence.train`,
durable forward/optimizer accounting, AdamW readback and numerical observation
functions. It does not change those sources or the original frozen artifacts.

## Fixed question and inputs

The preceding saved CTC analysis found low support for the K1 character “好” in
several exposed development examples, including competition with blank. The next
question is whether adding source-balanced generated speech changes complete K1
support, complete-text fit and K1/K2 false occurrences at the same 2000 updates,
without changing the vocabulary, model, loss, initial state or learning rate.
This motivation does not establish an acoustic label or a decoder failure.

The baseline is the actual context52 final checkpoint, SHA256
`92d1d0b3d98e39bb9d5739d5bce80dc4a93bd7eb2e6a97f6fd55308012b19857`.
Its old 52 train and 18 development observations at 0/200/500 ms are reused from
the saved 210 conditions. Its new job only observes the new rows, at most 64 × 3
= 192 new forwards. It performs no training and does not rerun those old210.

The data declaration is the unchanged first64 of the existing 256-cell plan,
canonical SHA256
`8ff19752b173cd8097ad8a612bd3329d44ae679419e8c54d30a69e73ad88ebd9`.
The eight previously completed cells are reused; exactly the next56 are the new
upstream TTS allocation. Intended first64 coverage is 48 train and 16 development
cells from the same eight source families. Each family has K1 ×2, K2 ×2,
ordinary negative ×2, partial negative ×1 and near negative ×1. These are intended
generation categories, not observed labels. The original6/2 source role split is
preserved; development cells never enter optimizer draws.

The original52 adopted training dictionaries remain unchanged in the new recipe.
The original18 development dictionaries and two original failed zero-weight rows
also remain unchanged. Audio relocation in a new output directory is explicit;
the existing row identity deliberately excludes only audio path location.

## Full text, weights and exact C1 geometry

All actual weak rows first pass the existing D1 `aggregate_batches` replay. This
materializes the original loader, including every uncertain and legacy-OOV row,
before any new objective projection. No TTS prompt becomes a target. A transcript
is used only if the original D1 v1 weak decision supplied a confirmed complete
text, mask and confidence weight. For example, an observed “你好小吴” stays exactly
that text; it is not replaced with an intended near-negative string.

The vocabulary is the existing fixed 2601-output map, unchanged since the earlier
D1 task declaration. It already added “沃” and “灯” to the original2599 map. Thus
the old2599 loader's “请把灯打开” OOV is representable without a new vocabulary.
The four-token legacy RNN projection is retained as upstream evidence; it does
not restrict this full-text C1 objective.

The new policy, `observed-D1-weak-fulltext-C1-only-v1`, deliberately differs from
the former shared three-model projection in two documented ways:

- Eligibility uses C1's actual unpadded frame count, `(samples - 400) // 320 + 1`,
  and the complete target length plus adjacent-repeat blanks. A smaller FSMN
  frame count does not exclude an otherwise valid C1 target.
- A generation-intent/observed-weak-label disagreement is retained as a diagnostic
  flag. It does not replace the weak label, quarantine an otherwise valid full
  text, or infer a negative label from an unknown transcript.

The original label is retained separately in `observed_label`. Unsupported full
text or infeasible geometry produces an explicit objective reason, null target
IDs and zero CTC contribution. Its original full text, label mask, confidence and
known keyword labels remain in the full ledger. There is no truncation, character
substitution, mapping to blank, or requirement that every intended category be
successfully recognized. Native failed/unknown/unstarted cells without weak rows
remain in the separate complete64 accounting ledger. Fewer than64 successful
rows do not trigger another TTS/ASR attempt or a replacement source selection.

## Exact training and evaluation

The augmentation arm uses the same C1 architecture, 2601 vocabulary, AdamW
`lr=0.001`, `weight_decay=0.01`, clip norm5, seed20261105, batch4 and 2000 steps:
exactly8000 training forwards. The primary objective remains target-length-
normalized complete-text CTC times the original weak confidence, then mean over
the four batch items. The auxiliary term remains scale1 mean blank NLL only on
fully added digital-zero windows. Original quiet speech is not labeled silence.

Rows cycle in old52 order, then eligible new rows in their original first64 plan
order. Each row's visits rotate 0/200/500 ms; fully added zero frames are0/9/24.
With all48 new train rows eligible there are100 training rows and80 visits each.
If the actual upstream result yields fewer eligible rows, the same8000 draws are
declared over the actual frozen roster with exact per-row/per-offset counts.

Initialization uses the retained zero-step file SHA256
`da7671fe71fef9ecd6822c747e3fe153b9ee563a6784c18eded90d3354486ecf`.
Preparation verifies the serialized file and prior audited float-state identity.
The actual training process loads it, hashes its tensors, loads the model state
and verifies its exact float-state identity **before** optimizer construction or
the first training forward. A matching seed alone is never accepted as proof of
the same initialization. The prior trained92d1 checkpoint is not the augmentation
initialization.

Final augmentation observations cover the old70 plus every actual new weak row,
including unsupported train rows, at three offsets: at most402 forwards. Known
full-target loss and edit denominators remain separate from unsupported targets;
known keyword masks remain independently usable. The final checkpoint is saved
before these observations. Full unquantized float32 features `[frames,32]` and
logits `[frames,1,2601]` are saved for every actual condition, not only aggregates.

The `compare` command joins saved baseline old210 plus new observations to the
augmentation's exact same row/offset roster. It verifies identical feature and
label bindings and the actual initial float identity. It reports old train, new
train, old development18 and new development up to16 separately. It does not pool
three offsets as independent samples, search a threshold, select a checkpoint or
claim human-gold accuracy. Existing development18 and the new development source
roles remain development; the original two new development sources were already
observed in the eight-cell pilot.

Both models' complete logits can feed the existing saved CTC primitives
`research_saved_ctc_diagnostic.compute_arrays`, `variants` and `sequence_score`
under a later exact saved-condition declaration. Preserve all K1/K2 positions and
reference mentions, not just the “好” failure subset. That saved-only work requires
no model forwards. The historical264 diagnostic has a fixed input contract and
must not be repurposed by editing its old recipe or claimed results.

## Runtime, storage and failure accounting

The root's frozen balanced64 budget supplies the actual phase ceilings. Baseline
evaluation has a300-second queue limit and270-second worker deadline; the new
training/final-evaluation job has a900-second queue limit and840-second worker
deadline. Both use4 CPU threads,1 interop thread, the original shared heavy lock,
one model construction and one durable queue claim. There are no automatic model
retries. The separately invoked saved-tensor audit uses the same shared lock and
a240-second internal deadline, but constructs no model and performs no forward
or optimizer update. Its time belongs to the explicit metadata/audit reservation.

Training retains started/completed forward and optimizer ledgers, complete
per-draw weak targets/weights/losses, gradient-norm observations, final state and
AdamW moments. It does not retain all8000 intermediate training logit/gradient
tensors or claim numerical replay of the entire optimization trajectory. A
timeout preserves the partial actual evidence and consumes its unique attempt.

The declaration computes exact raw final payload sizes:
`sum(frames * 2601 * 4)` logit bytes and `sum(frames * 32 * 4)` feature bytes.
After terminal audit, group the original serialized numeric files in deterministic
path order, with each group's uncompressed original-file total at most32MiB.
Save every numeric file exactly once and bind all descriptors to the metadata
capsule. Do not quantize, delete, or duplicate old baseline numeric bodies.

## Metadata request and commands

`probe` and `declare` import no Torch and initialize no queue. The request fields
are `schema`, `baseline_output`, `baseline_audit`, `new_plan`, `packages`,
`view_loader`, `terminal_accounting`, and `global_budget`. Pins are the exact
`{path,bytes,sha256}` records; `packages` uses the original
`{path,manifest_sha256}` D1 declarations.

`terminal_accounting` is null only for a preparation probe. A real declaration
requires a pinned `kws-newsource64-terminal-accounting-v1` record with
`terminal:true`, the original `plan_sha256`, all64 ordered `selected_cell_ids`,
actual `source_pins`, and64 `entries`. Each entry binds `cell_id`, actual
`tts_status` (`completed/failed/unknown/not_started`), `asr_status`
(`completed/partial/not_run`), `weak_row_present`, and actual `evidence` pins.
This full ledger is frozen after the upstream jobs terminate; it is not fabricated
from intended outcomes. A weak row must match its actual terminal state.

From the frozen reviewed source and pinned Python runtime:

```bash
python -I -S -B training/research_c1_source_augmentation.py probe \
  --request /absolute/preparation/probe-request.json --output /absolute/new/probe.json
python -I -S -B training/research_c1_source_augmentation.py declare \
  --request /absolute/actual-terminal-input-request.json --mode baseline-evaluation \
  --output /absolute/new/baseline-recipe.json
python -I -S -B training/research_c1_source_augmentation.py declare \
  --request /absolute/actual-terminal-input-request.json --mode augmentation2000 \
  --output /absolute/new/augmentation-recipe.json
python -I -S -B training/research_c1_source_augmentation.py queue-request \
  --recipe /absolute/new/augmentation-recipe.json --python /absolute/pinned/training/python \
  --output /absolute/new/augmentation-queue-request.json
```

Save the reviewed source, actual input/metadata preparation, source mapping and
provider acknowledgment before the root initializes and runs the unique queue.
The same queue-request CLI supports the baseline recipe. `run` is queue-only;
its inherited job, recipe hash, output path and shared-lock lease are checked.
After a real completed result, run `audit --output ... --manifest-sha256 ...
--receipt ...` with the numerical runtime (without`-S`). `compare --request ...
--output ...` then consumes two `{output,manifest_sha256,audit}` records named
`baseline_evaluation` and `augmentation`, without new inference.

The source tests use invented metadata for opposite-intent/actual-label cases,
uncertain masks, complete OOV retention, C1/FSMN geometry differences, source-role
leaks, fixed budgets and per-row exposures. The separate numeric fixture uses
only tiny invented logits; it constructs no model or optimizer.
