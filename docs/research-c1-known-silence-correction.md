# One C1 correction using known synthetic silence

The completed onset diagnostic observed the same `你 / 好 / 小` emissions in the
added digital-silence windows of all sixteen padded conditions. Original speech
feature bytes were unchanged. Plain full-transcript CTC can still align target
tokens to these silence frames, so prefix timing variation alone is insufficient
to require blank output there.

This single candidate changes both the input prefix exposure and adds a known
silence objective. It is a combined correction, not a one-variable ablation.
The original model, frontend, full-transcript loss, eight source/label bindings,
zero-step initialization and optimizer settings are preserved. Original helpers
are imported unchanged; a small separate batch loop adds the explicit blank
term without altering the original loop or its historical ledger semantics.

## Frozen experiment

| Item | Fixed setting |
| --- | --- |
| Initialization | Original saved zero-step C1 checkpoint `da7671fe…` |
| Training | 2,000 steps, batch four, 8,000 training forwards |
| Source order | Original row `draw_index % 8`, no replacement |
| Prefix order per row visit | 0, 200, 500 ms, repeated |
| Exposures per row | 334 / 333 / 333 for the three conditions |
| Total exposures | 2,672 / 2,664 / 2,664 |
| AdamW | Learning rate 0.001, weight decay 0.01, clip norm five |
| Full-text loss | Original loss, masks, weak confidence 0.5 and normalizer one |
| Auxiliary | Mean empty-target CTC NLL on only complete added-zero windows |
| Auxiliary coefficient | Exactly 1.0, confidence one, independent of speech weak weight |
| Final observations | All eight rows at each prefix: 24 additional forwards |
| Before/development | Zero new forwards, no development optimization or selection |
| Worker/queue | 540 seconds / ten runner minutes |
| Retries/sweeps | None |

For a pure-zero prefix of `S` frontend frames, an empty CTC target has one valid
path, namely blank at all `S` frames. The auxiliary is therefore
`-mean(log_softmax(logits[:S])[:, blank])`. Its frame-mean normalization avoids
making 500 ms inherently 24/9 times more important than 200 ms. No division by
the entire utterance length or multiplication by speech-label confidence is
applied. This fixed coefficient is a soft penalty; it does not guarantee success.

The waveform receives 3,200/8,000 zero PCM16 samples. With the unchanged
400-sample window and 320-sample hop, only **9/24** complete windows are known
to lie in those added zeros. The one window crossing original-audio onset is
excluded. The 0-ms condition has no such known region and auxiliary zero.
Original low-volume or accented speech is neither inspected by a VAD nor
trimmed, relabeled or assigned a new blank target. No decoder rule is changed.

All 24 features are computed and checked before constructing the model. Their
aligned original suffix hashes must match the original saved feature hash.
The auxiliary reuses each existing training forward's logits; it adds no model
forward. The full-text loss contribution, auxiliary contribution and combined
loss are separately retained in the 2,000-step consumption ledger.

## Evaluation frozen before execution

The minimum mechanism gate requires zero nonblank argmax frames in every known
zero window of all sixteen padded conditions. Each of 0/200/500 ms must also
reach at least the original unpadded 5/8 exact full transcripts and no more than
7/32 token edits. All conditions are reported, including failures.

Strict tiny-fit success is separate: all 24 full transcripts must be exact and
the known-zero boundary must pass. Neither gate is independent-speaker,
real-room or target-device qualification. The old 2,000-step results and onset
diagnostic are reused as controls; they are not rerun. Final onset deltas compare
the new model with its own 0-ms logits; old-versus-new metrics are stored separately.

The final readback authenticates the saved optimizer groups, parameter moments
and step counters, checks ledger arithmetic, and recomputes all final numerical
metrics without constructing a model or performing a new optimization step.
Failure is retained as-is; no coefficient or step count is changed after seeing
results. Blank probability means/minima are diagnostics, not tunable thresholds.

## Execution order

Prepare and review the immutable recipe and queue without model calls. Save the
prelaunch evidence with the existing checkpoint mechanism. The owning root
controller must verify that the **entire acoustic124 campaign is terminal**
before starting this ten-minute job. The frozen root launch wrapper uses the
original acoustic validator to authenticate all sixteen completed queue results,
all sixteen durable chunk closures, and the final aggregate and campaign save.
It takes a nonblocking flock on the existing driver-lock file without rewriting
its bytes, retains that lock during the training queue invocation, and writes
the authenticated prerequisite receipt before launch. A failed check starts no
job. Do not use an inter-chunk lock gap to launch training: that would consume
the ASR driver's 60-second busy budget. The worker uses the same inherited
heavy-model lock and durable queue recipe binding. Save the launch receipt with
the terminal experiment evidence.

The metadata tests use no Torch. Optional tensor fixtures verify empty-target
CTC equivalence and gradients on small invented logits; they never construct
an encoder, load a checkpoint, call a model or instantiate an optimizer.

## Supplemental per-draw arithmetic audit

Independent review found one accounting gap in the original readback: it checks
each primary and auxiliary contribution and their batch totals, but two incorrect
per-draw total contributions can cancel within a batch. The training loop itself
records the correct sum. This finding does not change the training objective,
the experiment's outcome, or the number of permitted model calls.

The frozen execution source remains
`d250a12373f68f937fc7c4864b42a682b08240fa`, with entrypoint
`training/research_c1_known_silence.py` SHA256
`85d7e9a1d1cbc33b103c523455971fd866ede049478f464f44530255ea932d4a`.
Its recipe rebuilds and checks its own source hashes. Editing that source would
invalidate the existing frozen recipe, so the correction is a separate
[`research_c1_silence_ledger_audit.py`](../tools/research_c1_silence_ledger_audit.py)
validator. Neither the original source nor its recipe or result is rewritten.

After the original audit has succeeded, supply its receipt's SHA256 and the
sealed output manifest's SHA256. The supplement verifies the complete manifest,
the original successful audit's matching identity, the frozen execution source,
the recipe's saved sources and each of the 8,000 draw contributions. Every
draw must satisfy `total = primary + auxiliary`, in addition to all three batch
totals. It rejects even a resealed invented counterexample whose two incorrect
row totals cancel. Hash arguments must come from the verified terminal evidence,
not from an untrusted replacement bundle.

```bash
python3 -I -S -B tools/research_c1_silence_ledger_audit.py \
  --output "$C1_OUTPUT" \
  --manifest-sha256 "$C1_MANIFEST_SHA256" \
  --original-audit "$C1_ORIGINAL_AUDIT" \
  --original-audit-sha256 "$C1_ORIGINAL_AUDIT_SHA256" \
  --receipt "$C1_SUPPLEMENT_RECEIPT"
```

The new receipt must be outside the sealed output directory. It records the
frozen execution identity separately from the supplementary auditor and its
standard-library helper hashes. This check imports no training module, reads no
tensor through a tensor loader, constructs no model or optimizer, and performs
no forwards or optimization. It supplements the original numerical audit; it
does not replace that audit or establish model-quality acceptance. Keep both
receipts with the terminal evidence. No supplemental result exists until the
completed experiment and original audit are available.

## Verification accounting

The initial publication checks ran 11 commands on CPython 3.12.16, with 99
metadata/contract cases and no model calls. Those default invocations do not
include the **four** optional small-logit fixtures: empty-target CTC value/gradient
equivalence, zero-prefix zero gradient, frame-mean normalization, and exclusion
of invalid masks. Their test source SHA256 is
`ad63213018a84cfa05da5d91e481a67cada0c782bf339ca0ed42bc3c9feefc9c`;
they are selected separately with `--tensor-fixtures`. This describes the
available mathematical fixtures and does not add their definitions to the
recorded execution counts above.

The supplementary audit's standard-library tests cover per-draw cancellation,
per-component cancellation, complete draw counts/order, nonfinite values,
manifest and original-audit bindings, frozen-source changes and preservation of
sealed outputs. They run under normal, `-O` and `-OO` Python modes in the existing
CI job. These test counts remain separate from both the four tensor fixtures
and any later real experiment audit.
