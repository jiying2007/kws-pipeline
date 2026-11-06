# Balanced64 saved posterior adapter

`training/research_c1_posterior_audit.py` adapts the unchanged saved CTC diagnostic
kernel to the balanced64 experiment. Its fixed contract is
`configs/research/c1-saved-posterior-20261106.json`. The contract pins the kernel's
42-file source closure and the original manifests, comparison and prior audit.
The new recipe also pins this adapter, its test and documentation.

## Fixed coverage and budget

| Saved model conditions | Newly computed | Existing result reused | Unscored |
| --- | ---: | ---: | ---: |
| Augmentation, all 402 | 390 | 0 | 12 |
| Baseline, new-source 192 | 180 | 0 | 12 |
| Baseline, original-source 210 | 0 | 210 | 0 |
| Total 804 | 570 | 210 | 24 |

The 570 new conditions contain 77,984 frames, 4,089 original target positions,
348 keyword mentions, 1,392 single-token-deletion contrasts and 2,310 unique
sequence scores. Original numeric files total 822,499,298 bytes; the derived
array payload is 28,872,128 bytes before serialization overhead.

The worker and explicit numerical audit are each limited to 600 seconds,
four CPU threads, one interop thread and 2 GiB RSS, using the existing shared
heavy lock. The worker has one supervised ten-minute queue job. Audit is
separately charged; these passes together have a 20 runner-minute ceiling.
There is no automatic retry, model construction, forward, optimizer, backward,
ASR or TTS call. The old 264-condition job is not rerun.

## Original labels and reuse

Original full observed weak references, weights, masks and split/source
bindings are preserved. Generation intent is not supervision. Four uncertain
sources remain without a CTC target and have false masks and zero loss weights.
One of them retains nonempty unadmitted observed text: it is preserved verbatim
and **is not projected into a target**. Legacy missing `frames` or `ctc_mask`
observation fields are not filled. Across two models and three offsets these
produce 24 explicit unscored coverage entries, never synthetic negatives.

The original 210 baseline derived results are reused only after matching their
original recipe, manifest, successful queue/audit receipts, checkpoint, source,
label binding, PCM/numeric identity, target positions, greedy sequence and
hypothesis provenance to the current comparison. Their raw scalar records are
retained unchanged. Original derived files are byte-hashed, not deserialized;
no old CTC dynamic program or model is executed.

Declaration reads JSON and file metadata, imports no Torch and does not open
numeric tensor bodies. The worker checks each new serialized hash, loads with
`weights_only=True`, and validates float32 features/logits against the original
forward ledger before entering the unchanged kernel. Existing mathematical
checks use full-vocabulary float64 log-softmax, unrenormalized gathered columns,
independent CTC scoring, conditional FB/Viterbi support and saved float32 NLL
parity. Module construction/calls, optimizer operations and autograd operations
are locally guarded throughout all tensor work. This guard is not a sandbox.

## Entry points and artifacts

Use the frozen Python 3.12.16 research runtime. Every actual invocation belongs
in the parent execution ledger with argv, cwd, source hash and real return code.

```text
research_c1_posterior_audit.py declare --comparison COMPARISON_JSON \
  --legacy-output OLD_264_OUTPUT --legacy-audit OLD_264_READBACK --output RECIPE_JSON
research_c1_posterior_audit.py queue-request --recipe RECIPE_JSON \
  --python PINNED_PYTHON --output QUEUE_REQUEST_JSON
research_c1_posterior_audit.py run --recipe RECIPE_JSON --output NEW_OUTPUT \
  --heavy-lock CANONICAL_LOCK
research_c1_posterior_audit.py audit --output NEW_OUTPUT \
  --manifest-sha256 ACTUAL_MANIFEST_SHA256 --receipt NEW_AUDIT_RECEIPT
```

`run` must be invoked by the standard supervised research queue; direct
unsupervised execution is rejected. `audit` requires its successful completed
queue result, matching status, manifest, sources and immutable original inputs.
It recomputes only 570 new results and compares exact derived array hashes and
scalars. It source-verifies the 210 reused results and retains all 24 unknowns.

The private recipe contains complete source/weak-label bindings. Output files
are create-once: 570 full-precision derived arrays, `conditions.jsonl` with
780 scalar wrappers, `coverage.json` with 804 model conditions, durable
per-condition claims, execution sources, summary/status and a manifest.
Reused results carry absolute original derived-artifact references, not copies
of their arrays. Original features/logits and checkpoints are not duplicated.

`public-summary.json` is an explicit aggregate-only export with no source IDs,
transcripts or local paths. It covers every old/new train/development cohort,
0/200/500 ms condition, both keywords and all four token positions, including
all repeated mentions, original greedy hit groups, known-negative controls and
paired model changes. The 0 ms condition is primary; offsets are repeated
conditions of the same sources, not independent observations.

The private `summary.json` interface for downstream source-allocation decisions
is fixed: `schema="kws-c1-posterior-audit-v1"`, `status="completed"`, zero
`new_model_constructions`, `new_model_forwards` and `new_optimizer_steps`;
`inputs.comparison.sha256` binds the original saved comparison and
`inputs.augmentation.manifest_sha256` binds the actual control output. The
workload records 570 new, 210 reused, 24 unscored and 804 total model conditions.
A downstream decision must also retain the independent successful readback;
it must not treat an arbitrary pinned JSON file as a completed diagnostic.

## Interpretation

The deletion contrast is raw full-sequence log P(original weak reference)
minus log P(the same reference with one original keyword token removed).
Candidates differ in length and are analytic alternatives, never relabeling.
Conditional occupancy must place every supplied token; normalized q is not
the probability that a syllable was spoken. Viterbi intervals are conditional
model windows, not human phoneme timestamps.

Low q-weighted raw class support together with a negative deletion contrast
supports a saved-model support deficit against that particular alternative.
It does not prove physical acoustic absence. A positive contrast with a greedy
miss does not prove that a finite general decoder will select the reference.
This stage performs no beam search, decoder replay, width/threshold sweep or
keyword biasing. Any decoder-rescue claim requires a separately fixed,
reference-free replay over complete controls and negatives.

The data remain synthetic and weakly labeled, and development was previously
exposed. No human-gold, independent-test, generalization, natural-FAR or physical
SSC305 acceptance follows from these diagnostic values.

## Focused fixtures

```text
python3.12 -I -S -B tests/test_research_c1_posterior_audit.py
python3.12 -I -B tests/test_research_c1_posterior_audit.py --numeric
```

Metadata fixtures cover source closure, typed budgets, unadmitted nonempty
unknown text, reused source/label/model identities, deferred numeric-byte
verification, all-position/repeated-mention and negative-control aggregates,
privacy, create-once outputs and failed-terminal rejection. Numeric mode uses
only an invented six-frame tensor and the existing guarded kernel, checking
full-precision retention. Neither fixture mode opens campaign tensors.
