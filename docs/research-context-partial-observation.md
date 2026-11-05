# Context timeout observations

`tools/research_context_partial.py` creates a separate observation package from
the actual first context-generation timeout. It does not change or call the
original generator, extend its deadline, rewrite its result, or retry a cell.

The original `research_context_generate.load_stage` and queue verifier check the
frozen stage, producer/reference bytes, execution sources, recipe, queue, claim,
actual duration and `timed_out` / `-9` result. The separately pinned inventory
must remain byte-identical before and after reading. All sixteen selected cells
are classified, including the complete returned prefix and the missing suffix.

The observed outcome is twelve technical completions, one failed-native return,
one consumed unknown and two never-claimed cells. Thirteen complete native
returns are exported. The failed-native audio is retained for recognition and
audit; its later training weight remains zero. The consumed unknown has only
its original call/start evidence, and the never-claimed cells have neither. None
of those three cells receives fabricated audio, text, blank targets or negatives.

The exporter reuses the unchanged data parser to validate original FLOAT WAV
samples, the exact saved numeric array, finite geometry, the start and technical
terminal receipts, and the completed derived WAV. The raw native WAV is exported
without trimming, gain changes or replacement. Model modules are forbidden in
this standard-library observation process.

```sh
python3.12 -I -S -B tools/research_context_partial.py export \
  --observation /private/generation-timeout-observation.json \
  --observation-sha256 ACTUAL_OBSERVATION_SHA \
  --owner-root /checkout/frozen-pipeline-owner \
  --data-weak /checkout/frozen-data/automation/active_asr_weak_view.py \
  --data-weak-sha256 ACTUAL_PARSER_SHA \
  --output /private/new-partial-export

python3.12 -I -S -B tools/research_context_partial.py verify \
  --package /private/new-partial-export --proof-sha256 ACTUAL_PROOF_SHA \
  --receipt /private/new-partial-readback.json
```

The package retains a deterministic native ZIP, full proof, exact request,
exporter source bytes and a manifest. Verification independently recomputes the
whole proof and ZIP from the original frozen bytes. It rejects a completed job
substituted for the timeout, changed owner/recipe, altered native bytes, added
unknown audio, promoted technical failure, symlinks and an existing destination.

Downstream input preparation and selective ASR retain their original protocol.
The context source-map also receives the original stage and owning job-recipe
SHA. The unchanged context weak adapter permits this authenticated observed
subset and reports its unobserved selected cells explicitly. Final delivery must
bind the partial proof, ASR audit and weak-view manifest together; the observed
thirteen must never be described as sixteen returned examples or human truth.
