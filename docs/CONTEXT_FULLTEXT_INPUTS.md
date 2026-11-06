# Verified context inputs for the existing full2601 vocabulary

`training/research_context_inputs.py` is a separate context consumer. It accepts
only `kws-active-context-research-weak-view-v1` packages whose source adapter is
`kws-active-context-native-source-v1`. It does not rename a context package into
a D1 or D2 schema, and it does not reuse the legacy 2599 target IDs.

The public entry point is `verified_projection`. Supply:

- Explicit package paths and `MANIFEST.json` SHA256 pins.
- The module and source hash inventory from `load_data_module`, pointing to the
  context producer's `automation/active_context_weak_view.py`.
- One expected context plan SHA256 for this bounded 8/16-cell supplement.
- The existing `vocabulary.json` path and its **file** SHA256, plus the original
  official FSMN token file.
- Any previous source-role evidence needed for cross-dataset split checks.

`load_vocabulary` verifies the supplied file pin and uses the unchanged
`research_d1_vocab.validate_vocabulary` to reconstruct the predeclared 2601 map
from the official token bytes. Observed context data never adds symbols. Both
file and canonical vocabulary hashes are returned for the consumer record.

Before projection, every package is fully replayed by its context verifier.
The consumer checks the current verifier source hashes against the package's
retained producer inventory, its source/plan scope, global cell and native PCM
deduplication, and source-group split compatibility. Legacy filtered JSONL
exports are not a source of truth. Unobserved plan cells remain a coverage gap,
not fabricated rows or negatives.

For every verified observed row, the unchanged `fulltext_targets` maps the
entire confirmed transcript. The consumer also checks unpadded C1 frame
feasibility, including adjacent-repeat CTC cost. Its output schema is
`kws-context-fulltext-projection-v1`.

The result preserves `full_text`, `selected_raw_text`, all `asr_transcripts`,
the original `observed_label` masks/weights, retained-row hashes and observed
context measurements. Failed-native, uncertain, OOV or infeasible rows have
`target_ids=null`, CTC weight zero, and all keyword weights zero. No characters
are removed and unknown labels are never filled with negatives.

## Minimal integration

```python
loader, producer = context_inputs.load_data_module(context_loader_path)
rows, evidence = context_inputs.verified_projection(
    [{"path": str(package), "manifest_sha256": package_manifest_sha256}],
    loader,
    producer["files"],
    expected_plan_sha256=context_plan_sha256,
    vocabulary_path=existing_vocabulary_path,
    vocabulary_sha256=existing_vocabulary_file_sha256,
    original_tokens_path=official_tokens_path,
    prior_sources=previous_source_roles,
)
```

The returned rows contain absolute PCM paths, split/source identities, complete
target IDs, confidence weights and masks for a trainer. The consumer does not
perform an optimizer step or claim training exposure.

## Verification

`tests/test_research_context_inputs.py` uses the repository's exact official
token fixture for full2601, full-transcript OOV and zero-weight checks. Set
`KWS_CONTEXT_DATA_ROOT` to a data checkout containing the context producer to
also run the real cross-repository fixture-package replay, including duplicate,
wrong-plan and altered-package rejection. Both paths are model-free.

Planned templates should be checked against the fixed vocabulary before a new
generation declaration. For example, the current map does not contain `厅` or
`暗`: an accurately transcribed context containing either remains whole-text
OOV. The consumer correctly preserves that observation and assigns no training
weight; this is a source-pool coverage issue, not permission to widen the map.
