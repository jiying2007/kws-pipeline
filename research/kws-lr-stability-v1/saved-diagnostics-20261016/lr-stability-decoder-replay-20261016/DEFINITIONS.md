# Saved-LP native decoder controlled replay

This freezes the previously scoped minimum replay. It is a descriptive diagnostic
on 12 exposed machine-labelled training positives with 1276 saved LP rows.
There is one native C pass through those rows, no model/audio/weight access, no
inference, training, TTS, ASR, parameter sweep, synthetic audio or publication.

## Fixed controls

Use the unmodified src/decoder.c at the source-manifest commit, linked only to
the diagnostic harness. Source src/kws.c is read for suppression semantics, not
linked. The official model-bound replay program is not invoked; no .kwm or
KWTRACE identity is invented. Public generated parameter constants are unchanged.

The two keyword structs are inserted in order KW1 then KW2, with IDs1 and2,
tokens[1,2,3,4] and[3,4,3,4], threshold0.55f, immediate policy, trailing blanks0,
priority0, grace0. Threshold0.55 is the public shipping/example setting, not a
calibrated value for this research model; parameter-contract has no generic
threshold default. state_retention=.94; token_boost=1.5 is ineffective. Native
blank-dominant decay and root/fuzzy/abandonment restrictions remain active even
though speech_active is constantly1. This is not a performance upper bound.

Initialize native decoder and separate suppression mirror once per clip. The
native decoder performs its own reset after every hit; the harness adds no
within-clip manual reset. The mirror accepts a native hit iff end_sample >= its
saved deadline, then sets deadline=end_sample+19200. A suppressed native hit
does not extend the deadline but has already reset the native decoder.

speech_active=1 is synthetic control on every frame, not recovered VAD. Do not
use RMS, prefix_frames or model blank as a replacement VAD. Do not append blank
frames or duplicate, concatenate, trim or reorder clips. End samples are
400+320*k for zero-based frame k, with 16kHz nominal source time. These are not
word-end latency or device timestamps.

Each JSON LP number must exactly round-trip to float32. Serialize its exact
float value in hexadecimal text and parse with C float scanf. Record all5
resulting float bit patterns so lossless transport is independently checkable.
This is saved-LP-input replay; original unnormalized logits were not saved, so
there is no original-logit bitexact equivalence claim.

## Recorded outcomes

Keep native decoder hit, keyword ID and confidence separate from mirror
accepted/suppressed outcome. For every original frame save clip/frame indices,
end_sample, synthetic speech flag, argmax, LP float bits, native output and
mirror before/after deadline. Every successful native hit must leave the native
decoder in its own reset state, which the harness checks immediately.

Report sample/class counts of native hits, mirrored accepted outputs and
suppressed hits, including all event frames and IDs. Count samples having an
intended-keyword hit/output and samples with multiple outputs only as finite
training-trace conditions, not recall or duplicate-wake error rates. An output
of the other full keyword is a label-disagreement event, not an adjudicated
false accept. There are no true negative streams or human event boundaries.

## Before and after the one saved-data pass

Before execution, verify accepted evidence/plan/cohort hashes, actual source
writer identity, current runtime source blob hashes, token mapping, row count,
record ordering, finite normalized LP and exact float32 transport. Generate the
parameter header with the pinned public generator. Build without fast-math or
FMA contraction. Run unmodified test_decoder.c synthetic tests and harness
synthetic tests of suppression first hit, no hit, expiry-1, exact expiry,
suppressed-no-extension, per-clip restart and native-reset-before-suppression.
Freeze code, generated header, binary, tests, compiler command and prepared
transport hashes before saved-data execution.

Create an exclusive execution claim immediately before the one native call.
Never retry a claimed native saved-data run. Validate all output frame indices,
bit patterns, totals and suppression recurrence without rerunning native data.
Hash-check original inputs afterward. Completion is one1276-frame native pass
and evidence verification, regardless of the result. Stop on any mismatch or
missing dependency. No automated optimization, extra run, VAD alternative,
threshold change or model call follows a disappointing result.
