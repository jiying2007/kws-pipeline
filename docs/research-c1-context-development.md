# Fixed C1 comparison on the exposed context development set

`training/research_c1_context_development.py` compares the original 2,000-step
tiny-fit checkpoint with its completed known-silence correction. It reads all
18 development rows from the frozen context72 ingestion consumer in their
original order. These rows have already been exposed by an earlier development
readback. This is a descriptive development diagnostic, not an independent
final test, model selection, or acceptance decision.

The declaration authenticates both completed queue receipts and their existing
audits, replays the original context input verifier, and derives the original
checkpoint from the corrected experiment's frozen onset lineage. It pins the
two actual checkpoint SHA256 values, the original 2601-symbol vocabulary, every
development row including its full transcript and keyword known masks, PCM
identities, source families, roles and the execution source hashes. Any overlap
with the original training rows or context ingestion training rows is rejected.
No development row is replaced or dropped after declaration. Ordinary speech
retains its complete nonempty text.

Each original verifier runs from its saved `execution-source` in an isolated
stdlib-only subprocess. Its complete file map is checked before and after
execution. This retains the earlier metadata checkpoint helper and the earlier
runtime manifest's validation annotations without rewriting their sources.
The numerical and input-projection code must still match the current consumer;
interpreter and dependency contracts must match exactly after removing only the
runtime manifest's `validation` metadata. A change to an algorithm, package or
interpreter contract fails before any model is loaded.

| Item | Fixed scope |
| --- | --- |
| Checkpoints | Original tiny-fit, then known-silence correction |
| Audio | All 18 existing development rows |
| Prefixes | 0, 200 and 500 ms of PCM16 digital zeros |
| Model forwards | Exactly one per model, row and prefix; 108 total maximum |
| Optimizer / ASR / TTS / threshold selection | Zero |
| Worker deadline | 540 seconds, with signal and monotonic checks |
| Existing durable queue | One job; 10 minutes total |
| Heavy model slot | Same canonical inherited lock as the original jobs |
| Automatic retry | None; failed or unknown claims stay consumed |

The signal deadline covers validation and the worker's numerical output. The
existing queue provides the outer process deadline. Every forward has a durable
started/completed entry. Each complete float32 logit tensor and feature tensor
is retained without quantization, with a per-observation full label and source
binding. The recipe reports the exact uncompressed logits payload size before
launch so the owner can retain sufficient storage. A failure keeps its partial
status, call ledger and tensors; a new output directory does not authorize a
retry.

For each model and prefix the report includes full-text CTC NLL, complete token
edits, blank occupancy, exact full transcripts, and whether each complete K1/K2
token sequence appears contiguously in greedy CTC. These are clip-level token
observations, not timestamped wake events or a natural-FAR denominator. Unknown
keyword masks never become negatives. Known-zero metrics cover only fully
added-zero windows: 9 frames at 200 ms and 24 at 500 ms. The window crossing
original speech is excluded. The aligned original features must be bitwise
equal across offsets and between the two checkpoints.

## Prepare, review and run once

Declaration and queue-request creation use only metadata and the original
read-only data verifiers. They do not import Torch or launch a model. The owner
supplies previously verified manifest hashes, saves the declaration and its
inputs with the existing checkpoint mechanism, and reviews the fixed job before
launching it. No new scheduler or daemon is introduced.

```bash
python3 -I -S -B training/research_c1_context_development.py declare \
  --context-output "$CONTEXT_OUTPUT" \
  --context-manifest-sha256 "$CONTEXT_MANIFEST_SHA256" \
  --context-audit "$CONTEXT_AUDIT" \
  --correction-output "$CORRECTION_OUTPUT" \
  --correction-manifest-sha256 "$CORRECTION_MANIFEST_SHA256" \
  --correction-audit "$CORRECTION_AUDIT" \
  --output "$NEW_RECIPE"

python3 -I -S -B training/research_c1_context_development.py queue-request \
  --recipe "$NEW_RECIPE" --python "$PINNED_CORE_PYTHON" --output "$QUEUE_REQUEST"
```

After the declaration is saved and the one new run is authorized, use the
existing `tools/research_queue.py init` and `run --max-jobs 1` commands, with the
unchanged active plan and the recipe's canonical heavy-lock path. Direct
unqueued execution is rejected. Once outputs are saved, the same module's
`audit --output ... --manifest-sha256 ... --receipt ...` command authenticates
the saved checkpoints, inputs, tensors and ledger and recomputes every numeric
observation without constructing a model, performing a forward or optimizing.

`tests/test_research_c1_context_development.py` uses invented metadata and one
deliberately invalid queued job. It tests source isolation, full-sequence and
unknown-mask semantics, the complete paired roster, immutable budgets and
failed-claim reentry without importing Torch or running model inference.
