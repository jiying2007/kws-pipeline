# D1 held-out development CTC diagnostics

`training/research_d1_evaluate.py` is a separate, read-only v2 entrypoint for
the two development source families already declared in the frozen D1 plan.
It reads a completed training result and the same sealed preparation used by
that run. It performs no optimizer steps, threshold tuning, model selection,
or relabeling. Adding this entrypoint does not report an actual D1 training or
development run.

The active plan's `weak_labels_train_only: true` boundary remains unchanged.
Machine-generated transcripts are training supervision and, in this explicitly
predeclared development view, engineering diagnostics only. These diagnostics
are not formal effectiveness acceptance, independent human validation, a
shipping gate, or grounds for choosing a winner. The two source families are
held-out synthetic reference-clone groups, not two independent samples of
natural human speech. Repeated cells, recognizer agreement, and multiple
candidate evaluations do not increase the number of independent human sources.

## Immutable inputs and completion-driven execution

All nine candidate/near-percentage jobs use the same prepared package and
ordered development roster. The two family IDs are copied from the frozen
plan's development split before execution, never chosen from recognition or
model results. The evaluator checks every saved projected row, repeats the
family/speaker/PCM/reference-ancestor separation against training, and requires
exactly those two development families. It does not silently reduce the roster
to only evaluable clips. Partial observed D1 coverage remains partial coverage.

The recipe is a new immutable declaration:

```json
{
  "schema": "kws-d1-development-evaluation-job-v2",
  "prepared_from_dependency": "d1-prepare",
  "training_from_dependency": "d1-rnn64-near0",
  "candidate": "rnn64-ctc2601",
  "near_percent": 0,
  "held_out_source_family_ids": ["<first-sorted-development-family>", "<second-sorted-development-family>"],
  "evaluator_source_sha256": "<exact training/research_d1_evaluate.py SHA-256>",
  "timeout_seconds": 600
}
```

These angle-bracket values are documentation placeholders; the actual recipe
requires valid family IDs and an exact source hash. Supported candidates are
`rnn64-ctc2601`, `c1-ctc2601`, and `fsmn-head-ctc2601`; near percentages are
0, 5, or 10. Declare both preparation and the matching training job in the
queue's `depends_on` list. The existing queue may additionally require all
nine training jobs before any evaluation starts.

Use argv equivalent to:

```text
python -I -B training/research_d1_evaluate.py --recipe {recipe} --output {output_dir} --heavy-lock /private/heavy-model.lock
```

The declared queue outputs are `summary.json`, `observations.jsonl`, and
`MANIFEST.json`. The evaluator requires the existing hash-bound supervised
queue lease and shared heavy-model slot. It resolves the actual preparation
manifest and training checkpoint from their completed dependency receipts;
no future hash is guessed and no published recipe is rewritten after training.
It validates the receipt, `final.pt`, training summary, and consumption ledger
before using a checkpoint. The checkpoint must bind the same preparation,
vocabulary, source snapshot, candidate, initial state, exposure recipe,
150 optimizer steps, and 600 draws as the completed training result.

Every evaluation has an explicit per-job limit of at most 999 minutes, also
subject to the queue's remaining actual-time budget. Model construction and
forward passes remain in the supervised process group. Soft deadline checks
surround native operations; the queue supplies the process-group hard limit.
Completed jobs are not recomputed. Failed or interrupted output is retained,
with recorded forwards described as a lower bound and no automatic retry.

## Full-text features and metrics

The evaluator reads the preparation's retained PCM16 mono 16 kHz WAVs and
verifies file bytes, sample count, and PCM identity. It recomputes each
candidate's original unpadded frontend: 32-channel log-mel for fresh RNN64/C1,
or the fixed official FSMN 80-channel features with left/right context two and
frame skip three. Geometry and minimum CTC alignment requirements are checked
before forwarding.

The 2601-output vocabulary remains the predeclared mapping, with all original
2599 outputs preserved and the fixed 沃/灯 extension. Every evaluable target
uses its complete saved transcript and nonblank token sequence. Future OOV,
uncertain labels, failed-native audio, intent/weak-label conflicts, and
insufficient geometry retain explicit non-evaluable reasons. They are not
silently converted to blank targets, negatives, or zero-error observations.

Each evaluable clip records:

- Full target IDs and greedy CTC IDs, collapsing consecutive repeats before
  removing blanks. Blank-separated repeats remain separate tokens.
- CTC sequence negative log likelihood, loss divided by full target length,
  and the same saved-confidence-weighted loss used by the training objective.
- Token substitutions, deletions, insertions, total edit numerator, target
  token denominator, and exact-token-match indicator.
- Source, family, speaker, PCM, weak-label, feature, logit, vocabulary, and
  checkpoint bindings needed to replay the diagnostic.

Token counts and exact matches are unweighted. Saved confidence affects only
the CTC loss diagnostic. Aggregation sums token edits and target tokens before
division; it does not average per-clip token rates or clip rates into a token
rate. Insertions can make the token edit rate exceed one, and the result is
not clipped. Exact-match denominator is the number of evaluable clips.
Non-evaluable clips appear in reason counts but neither metric denominator.
An empty evaluable set produces null rates, not a perfect score.

The summary reports the whole declared development roster and each family
separately, including roster hashes and exclusion counts. All nine runs must
retain the same overall and evaluable roster hashes for a same-data comparison.
These are full-transcript CTC engineering metrics; they are not wake-event
offsets, callback latency, false accepts per hour, or natural-audio FRR at a
common FAR operating point.

## Read-only evidence and tests

The model is strictly loaded from the final checkpoint, put into evaluation
mode, and run with gradients disabled. Every model-state tensor and original
checkpoint byte hash is verified again after evaluation. Per-clip numeric
artifacts retain both features and logits, are reopened with the restricted
tensor loader, and are checked against tensor shape/dtype/content hashes.
The closed output manifest includes these artifacts, observations, source
snapshot, request, and summary. Evaluation never updates the prepared package
or training checkpoint.

`tests/test_research_d1_evaluate.py` provides standard-library contracts for
the two-family boundary, OOV/uncertain/failed exclusions, token alignment,
unweighted aggregation, and actual queue CLI dependency handoff. `--numeric`
adds a small synthetic metric case and the real evaluation path over a
synthetic preparation and explicitly fictional completed-training metadata.
The preparation fixture exercises actual PCM copies, fresh model construction,
32-channel log-mel, feature-cache save/reopen, checkpoint serialization, and
closed-manifest readback. Data replay and the official FSMN donor/frontend are
explicit test interfaces. It loads no official weights, uses no real speech,
and runs zero optimizer steps. Passing these tests establishes software
contracts; it is not an actual D1 training or development result.
