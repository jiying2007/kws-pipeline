# Bound diagnostic metric inputs (candidate)

**Scope: legacy OFFLINE diagnostic profile only. This is not a completed
continuous-stream scoring integration. No live numerical producer/runner has
yet been connected to this boundary.**

A stdlib-only integrity boundary around the existing offline clip metric. No model,
audio, labels, recording names, private hashes, or weights are distributed here.
Only this new directory has been changed; historical frozen runners are untouched.

## Defect and narrow fix

The reviewed historical `measure(rows, events, model, window)` iterated with `zip`
without validating lengths. Two label rows plus one event-list yielded `clips=2`
but only the first label entered denominators. Its `summarize` wrapper checked
lengths, but actual diagnostic callers bypassed that wrapper. Equal lengths alone
also fail to detect swapped recordings.

`measure_bound` now requires a `kws-bound-events-v1` document. Every input and
result has a nonempty unique `recording` and lowercase 64-character `pcm_sha256`.
Input and output identity sets must match exactly. Results are aligned by ID, so
legitimate reorderings are safe. Hash mismatches, duplicate IDs, missing/extra
results, absent event lists, wrong nested event recording IDs and invalid event
values fail before calling the metric. Empty event lists mean zero detections;
missing event lists do not. Label and frame geometry are checked as well.

The caller supplies the frozen trusted metric function. We deliberately preserve
its arithmetic, offline clock convention and label semantics. Supported windows
are `pre_eof` and `eof_plus300` with the historical `seed0_B` profile.
The exported profile identifier is `seed0_B-offline-16khz-stride3-tail300-v1`:
16 kHz source PCM, 300 ms appended tail, 30 ms output-row stride and the historical
45 ms analytical offset. `samples` counts original unpadded source samples;
`frames` counts the complete padded offline output rows. The caller must verify
both against its frozen frontend geometry; this module deliberately does not
infer or guess the sample-to-frame relation. `pcm_sha256` identifies the original
source PCM, not WAV container bytes, padded PCM, features or model outputs.
Scores must be finite in [0,1], consistent with sqrt(product of node posterior
probabilities); they are NOT calibrated confidence estimates. The callback
is not a sandbox: do not pass arbitrary/untrusted executable code.

Identity is not authenticity. A fabricated manifest and fabricated matching hash
can still lie. The caller must verify actual input PCM against the trusted frozen
manifest. This boundary does not certify labels, human review, speaker novelty,
independence, model identity, threshold provenance, or acoustic deduplication.
Different IDs sharing PCM are not automatically merged: dataset deduplication is
a separate roster admission responsibility.

## Migration without rewriting historical evidence

New producers MUST emit native binding and may not use the legacy adapter as
a substitute. For future producers, save each output envelope with the recording ID and the
verified PCM digest of the actual input, even when events is empty. Retain the
whole fixed roster; do not filter missing results to make a run appear complete.
For each score call, construct rows with recording, pcm_sha256, samples, frames,
and two labels (each 0, 1, or null), then call measure_bound with the bound document.
Do not call the old measure directly on external/raw inputs.

The two reviewed historical layouts are:
- Fixed-head experiment: `result.evaluations[*].recording` and `.grid[*]`.
- Two-model diagnostic: `result.evaluations[*].recording` and
  `.models[model].grid[*]`.

Each grid entry has `threshold` and `events`. The private migration helper `adapt_legacy` supports these layouts
only under the explicit `kws-offline-diagnostic-artifact-v1` contract. It requires
original result bytes, their trusted SHA256, an order-independent identity-roster
digest and a nonempty review reference. The result must say COMPLETE and contain
exactly the expected recording IDs and exactly one requested threshold per row.
A selected model must exist. It never guesses an absent model or threshold.

The review reference must point to already reviewed freeze/receipt provenance
linking that result artifact to that input manifest. Do NOT bless arbitrary data
by computing matching digests at the call site. A missing trusted link blocks
legacy conversion. The helper reconstructs artifact-level binding and labels it
as such in output metadata; it does not falsely claim the old inference emitted
per-row PCM hashes. That is weaker provenance than new native bound output.
The roster digest binds ID+PCM only; freeze/receipt validation must separately
verify labels, geometry and metric source identity.

Historical cohort scoring: first adapt/validate the ENTIRE original roster, then
select the matching rows and matching bound results for that cohort. Do not pass
a subset of rows with the full historical artifact to the legacy adapter.

## Validation and integration

Run `python -m unittest -v test_bound_metrics` in this directory (16 public boundary tests). The separate private migration suite adds 6 tests. All fixtures use
synthetic IDs and placeholder hashes. Integrate module and tests as a small
research utility, not a change to deployment KWS behavior. An eventual public
change must include only the utility, tests, and this explanation; private
recalculation artifacts must remain excluded.

Actual saved-result equivalence is a separate required validation step. The restored saved JSON was compared in 32 historical metric cases, including
all saved metric fields and deliberately reordered output envelopes. All matched.
The original metric source file and original PCM were absent: this was a
reconstruction of audited arithmetic and artifact/roster audit, not a new
verification of native inference binding. No model or training was run.
Private adapters, reconstructed metric and result artifacts are excluded from
the public allowlist. The historical `validation-status.json` is private and is
not included in this public directory; the preceding paragraph is the public
validation summary, not a claim that those private inputs can be reproduced here.
