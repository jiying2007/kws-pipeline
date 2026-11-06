# Fixed saved CTC diagnostic

This stage reads existing float32 features and logits. It constructs no model,
performs no model forward or optimizer update, and preserves the original full
reference text, weak weights, source bindings and train/development roles.

## Frozen scope

The plan `configs/research/saved-ctc-diagnostic-20261106.json` was fixed before
this diagnostic opened any saved tensor. It includes all 210 context52 final
conditions and the 54 `known-silence-corrected` development conditions in the
paired archive. The 54 `original-tiny-fit` conditions are excluded upfront.

The 264 conditions retain 3,186 original target positions and 219 keyword
reference mentions. Each of the four original positions of every known positive
K1 and K2 occurrence contributes one deletion alternative: 876 contrasts.
Together with 264 unchanged references and 264 saved greedy hypotheses, these
are 1,404 logical score requests and 1,382 unique target tuples within their
respective condition. Logical references remain distinct after deduplication.
Six old greedy hypotheses are empty and have the exact all-blank-path score.
The 0/200/500 ms conditions are repeated conditions, not independent samples.

Input payload is 550,018,024 serialized bytes and 52,172 frames, with a maximum
of 440 frames per tensor. The worker streams one tensor at a time. The fixed
limits are 540 worker seconds, one 10-minute queue job, four CPU threads, one
interop thread, and the existing canonical shared heavy lock. There is no
automatic retry. Failed claims, partial derived artifacts and real exit codes
are retained. Canonical readback is a separate explicit saved-output audit,
also bounded by the 540-second timer and shared lock; it performs no model work.

## Mathematics and interpretation

For the unchanged complete reference `Y`, the stage uses the existing
`tools/diagnose_keyword_ctc_sequence_competition.py::ctc_forward_backward`
helper for standard CTC forward/backward and a Viterbi path. It reuses
`training/sequence_margin.py::_ctc_token_state_temporal_weights` for detached,
occupancy-normalized weights for every original target position. No new CTC
dynamic program is introduced.

Float64 log-softmax is computed across the full 2601-output vocabulary before
gathering the classes needed by the reference and greedy hypotheses. The
gathered array is never renormalized. Standard CPU CTCLoss checks exact sequence
scores; the original raw NLL is checked against the saved float32 result with
the frozen numerical tolerance. Blank is ID 0. The retained vocabulary JSON,
not the historical base tokens file or a decoder keyword table, owns token IDs.

Every single-deletion alternative keeps all other context and repeated-mention
tokens unchanged. The recorded contrast is raw `log P(Y) - log P(Y minus one
original keyword token)`, without length normalization or weak-weight scaling.
The alternatives differ in length. This is a **pairwise target preference**;
neither sequence need be globally preferred, and it does not establish that a
production decoder failed or that either transcript is acoustically correct.
The saved-greedy contrast has the same limited interpretation. There is no beam
search, decoder replay, threshold sweep or posterior reweighting.

The conditional distribution for target position `j` is the target-state
occupancy under the entire supplied reference, normalized across frames. It
must place every supplied token even when unconditional class support is tiny.
It is not the probability that the token was spoken. Repeated characters retain
separate original target indices, and repeated keyword mentions retain separate
reference spans. A Viterbi result is one deterministic optimum; its windows and
blank gaps are not human phoneme or word timestamps.

Each original position retains conditional weights, raw class probability,
full-vocabulary and nonblank competition ranks, logit gaps against blank and
the strongest other nonblank, and actual argmax occupancy. Ties share the
competition rank; rank 1 does not override the actual argmax tie rule. The fixed
rank-1/rank-3 summaries are descriptive, not tuned decision thresholds. Known
added-zero occupancy is separate from original audio. Tiny probabilities may
underflow when exponentiated; zero is not evidence of acoustic absence.

Synthetic pronunciation quality, weak transcription errors and the physical
presence of a syllable still require listening or other independent evidence.
Previously exposed development remains development. No human-gold,
independent-test, generalization, natural-FAR or hardware acceptance follows.

## Entry points and retained evidence

`training/research_saved_ctc_diagnostic.py` supports:

- `declare --context-output ... --context-audit ... --paired-output ... --paired-audit ... --output ...`
- `queue-request --recipe ... --python ... --output ...`
- `run --recipe ... --output ... --heavy-lock ...` through the standard supervised queue only.
- `audit --output ... --manifest-sha256 ... --receipt ...` after the original queue result is completed.

Declaration imports no Torch. It authenticates the original manifests, actual
completion/audit receipts, vocabulary, every original row and call binding,
the excluded model, precise contrast counts, runtime manifest and analysis
source hashes. `run` rejects an unsupervised entry and installs explicit guards
against module construction/call, backward and optimizer construction/steps.

The private recipe preserves source identifiers and full labels. Each condition
has durable started/completed records, a private scalar record, and derived
arrays at their original float64/int64 precision. The derived payload contains
gathered globally normalized probabilities, conditional weights, ranks and
competition values, not a second copy of the original full logits or features.
All outputs and execution sources are covered by a new manifest. Public
summaries contain only aggregates separated by model, split and offset.

Audit reopens original saved tensors and derived evidence, reproduces array
hashes and scalars, and reconstructs the aggregate report. It first rejects
failed queue results, divergent status, or a retained failure marker. It does
not relaunch the analysis queue, a model, or an optimizer.

## Focused validation

Metadata tests use only stdlib:

```sh
python3.12 -I -S -B tests/test_research_saved_ctc_diagnostic.py
```

Tiny synthetic numeric tests use the pinned core environment:

```sh
python3.12 -I -B tests/test_research_saved_ctc_diagnostic.py --numeric
```

The numeric tests enumerate short CTC label paths independently of the dynamic
program, including mandatory blanks for adjacent repeated labels, repeated
target positions, empty hypotheses, normalization before class gathering,
rank ties, and blank-dominated conditional alignments. They never open campaign
tensors. Metadata tests cover fixed typed budgets, fulltext/label/model/PCM
bindings, repeated-mention deletion provenance, output creation and failed
terminal rejection. No full saved-tensor numerical run is part of declaration.
