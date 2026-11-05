# Verified additive context-v2 inputs for full2601

`training/research_context_v2_inputs.py` is an independent consumer for
`kws-active-context-v2-research-weak-view-v1` packages with source adapter
`kws-active-context-v2-native-source-v1`. It requires the v2 stage and exact
context-v1 parent binding. The unchanged v1, D1 and D2 input schemas are not
accepted by this consumer.

The public API is `verified_projection`. It takes explicit package paths and
`MANIFEST.json` file pins, the v2 loader and source inventory returned by
`load_data_module`, one canonical v2 plan pin, the existing vocabulary file and
its file pin, and the exact original official FSMN token bytes. Optional
`prior_sources` supplies source-role evidence for cross-dataset split checks.

```python
loader, producer = context_v2_inputs.load_data_module(
    data_root / "automation/active_context_v2_weak_view.py"
)
rows, evidence = context_v2_inputs.verified_projection(
    [
        {"path": str(chunk_one_package), "manifest_sha256": chunk_one_manifest},
        {"path": str(chunk_two_package), "manifest_sha256": chunk_two_manifest},
    ],
    loader,
    producer["files"],
    expected_plan_sha256=context_v2_plan_sha256,
    vocabulary_path=existing_vocabulary_path,
    vocabulary_sha256=existing_vocabulary_file_sha256,
    original_tokens_path=official_tokens_path,
    prior_sources=previous_source_roles,
)
```

Each package is fully replayed before filtering or projection. Current verifier
source hashes must match the package's retained producer inventory. The
consumer checks global cell/PCM uniqueness, source splits, exact bounded plan
scope, all three zero historical-reissue counters and consistent vocabulary
and context-parent pins. Packages may cover a subset of the 8/16 planned cells;
the report states observed/planned counts and lists missing cells. Missing
observations never become artificial examples or negatives.

The consumer verifies the declared vocabulary file and reconstructs the fixed
2601 mapping using unchanged `research_d1_vocab.validate_vocabulary`. Its
canonical and file identities must also match the v2 plan's predeclared
vocabulary. The plan's requested-text preflight is retained as metadata and
does not decide training eligibility.

For each observed row, unchanged `fulltext_targets` projects the entire
confirmed transcript. Legacy target IDs are ignored. CTC eligibility also
requires sufficient unpadded C1 frames, including adjacent-repeat cost. The
row consumer is `kws-context-v2-fulltext-projection-v1`.

`full_text`, `selected_raw_text`, all ASR transcripts, original observed
labels/weights, retained-row hashes and measured context positions/counts stay
available. Failed-native, uncertain, observed-OOV and geometrically infeasible
rows have no target IDs, zero CTC weight and zero keyword weights. An ASR
transcript can contain an OOV character even when its request passed the fixed
vocabulary preflight; the observation is preserved and excluded from training.

The returned rows include absolute audio paths, split/source identities,
complete targets, weights and masks. Projection itself does not run a model or
claim optimizer exposure.

## Verification

`tests/test_research_context_v2_inputs.py` checks complete-text projection,
fixed vocabulary identity, legacy/v1/schema rejection, source scope,
failed/unknown/OOV zero weights and CTC geometry. With `KWS_CONTEXT_DATA_ROOT`
pointing to a checkout containing the v2 producer and adapter, it also creates
and replays a real-format native/selective-ASR fixture package, then tests
duplicate, changed-plan and altered-package rejection. Both paths run without
models or network access; the official token bytes are a repository fixture.
