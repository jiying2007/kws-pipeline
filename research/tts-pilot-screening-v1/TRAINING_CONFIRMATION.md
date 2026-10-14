# Explicit third-ASR training candidate confirmation

This is an inactive source change, not a model execution or training admission.

## Policy and admission

The new `third_policy` field supports exactly:

- `disagreement_only`: compatibility default when absent from an old manifest.
  As before, only two successful but different primary transcripts select a third.
- `training_candidate_confirmation`: the same disagreement review **plus** a
  third recognition for first-pass target candidates. Those candidates require
  both primaries to succeed, match the target after the existing conservative
  punctuation/whitespace normalization, report complete decoding with no quality
  flags, and have technically qualified generated audio.

The activation template remains disabled and explicitly set to the old policy.
Activation/admission must contain a reviewed explicit policy value. The launcher
records it in admitted.json, and finalize_input projects it into the manifest.
A missing activation/admission policy, unknown value or pre-expanded manifest is
rejected. An old standalone manifest without a field keeps the old behavior;
there is no environment-variable or CLI auto-upgrade. Root must set the new field
in the next reviewed ASR activation binding before that first run. This does not
change the current PR, run/source/artifact requirements, or first-run guard.

## Selection and immutable attempt budget

Host selection remains outside recognizer containers. The exact same audio-only
opaque job schema is used; neither targets, selection reasons nor peer transcripts
are mounted into the third recognizer. Recognition stays blind.

The two selection sets share one third slot per sample. An existing third result
of any status (including timeout/error/not_run) prevents rescheduling. Duplicates
and attempt=2 are rejected. Current 32 Qwen slots permit at most 64 primary + 32
third = 96 calls; the unchanged 48-slot design maximum is 144. No calls are
accepted for blocked Cosy lanes or unavailable audio. This policy adds coverage
inside the existing ceiling; it adds no new retry or extra-voice budget.

## Result semantics

- Two matching primaries with no successful third remain two-model-consensus.
  `primary_tier` preserves that first-pass machine candidate status independently.
- Only three actual successful matching transcripts can say
  three-model-consensus. Three-model training-candidate status additionally
  requires target match, all decoder completion/quality checks and technical
  audio pass. Incomplete third text agreement is not a training candidate.
- A contradictory third downgrades the final tier to review and final overall
  qualification to false while preserving the first-pass tier and all three raw
  result rows. Third failure or absence never manufactures a third agreement.
- Original two-model disagreement followed by a 2/3 majority remains review;
  adding a third result cannot erase the original disagreement.

New report fields include third_policy, third_reason, primary_tier,
three_model_training_candidate, training_confirmation_status and label_source.
The confirmed tier is `machine_three_asr_training_candidate`, and its label source
is `three_asr_machine_consensus`. It remains `human_gold=false` and
`training_admitted=false`: this supplies an auditable machine-label candidate for
the separately authorized training stage, not a human-verbatim claim or automatic
held-out/test label. Reported rates retain all 48 design slots and 32 executable
Qwen slots; the new candidate numerator never changes a denominator.

## Verification

`python research/tts-pilot-screening-v1/test_suite.py --require-numeric`
executes 40 tests without skips. Isolated `-I -S -B` normal/-O/-OO executes 36
stdlib tests plus one explicit unavailable-numeric skip. The separate disabled
activation-template suite has 15 tests. New policy regressions cover old-default
rejection, explicit selection, genuine triple agreement, contradictory/failed/
incomplete third, no rescheduling, duplicate attempts, quality failures, wrong
common target, shared disagreement/confirmation budget, fixed 96 ceiling and
explicit admission propagation. No actual model or GitHub operation was run.

API metadata requests use a non-redirecting opener. A synthetic HTTPS regression
exercises 301/302/303/307/308 through urllib handlers and proves the bearer-bearing
request is sent only once to api.github.com, never to the redirect destination.
