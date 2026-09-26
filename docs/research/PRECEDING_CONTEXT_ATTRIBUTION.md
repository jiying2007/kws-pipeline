# Preceding-context event attribution (development only)

> Historical retained evidence. The executable diagnostic lane and its self-tests were retired during repository cleanup after the result was captured; this document remains as evidence, not as a current runnable entry point.

## Scope and source

Base main: `6e434d40514df25740b67554f851d16ad806c707` (#300).
The two frozen #299 hosted grounded models and the verified development pool are
reused. There is no training, changed weight, threshold search, decoder change or
shipping gate change. #293/#297 and the recorded CPU-vendor failure are untouched.

Pool SHA256: `20346a75b33086d8085ebee7e72eacf4baad55c453a241e4f6babd710e222ce5`.
Model SHA256, seed 1337: `33ff92ad11a77d8d7bcde9b7621d29ddc457bd76a7497df4392198cec87af732`.
Model SHA256, seed 2346: `5441898fb732f88b1aba3d68b060a493185a58dcdc8ceee1725dfcfcfdca74b8`.

## Method

For every original clip, reconstruct #300's same-split, lexically guarded
predecessor. Check the original PCM identity, hop padding, exactly shifted event
windows, and absence of complete/cross-boundary keywords. Run the unchanged C
engine on three fresh arms:

1. prior nonwake clip + padding/200-ms gap + original target;
2. equal-length digital silence + original target;
3. the same prior/gap + equal-length digital silence replacing the target.

All original context/control event metrics must reproduce the retained input
report. A freshly compiled keyword pack must equal the retained fixed-threshold
pack. Model, runtime, pool and input reports must remain unchanged. Canonical
`score_events.py` matching determines false accepts BEFORE temporal partitioning;
a wrong keyword or duplicate event is not accepted merely for being in a window.

Partition by emission timestamp into prior-clip, padding/gap, and target-clip
regions. Those regions are not phoneme attribution. Control correspondence uses
exact keyword/time with multiset capacity, not confidence. Timing shifts and
changed keyword identities are explicitly not exact matches.

Where a false accept occurs in the target region, retain the actual C posterior,
verify authentic decoder replay, then blank only frames wholly before target PCM
onset. Later logits, timestamps and speech flags are preserved. This intervention
is explicitly edited decoder evidence, not an authentic-model/qualification run.
Record all resulting events and total FA counts, not just whether the old event
signature disappeared. Event substitution must not masquerade as a repair.

## Executed results

Final code executed 1,536 new C WAV-runner calls, 11 posterior dumps and 22 decoder
replays across both models, all 256 clips per model, and all three development
splits. Both train splits remain 32/32 matched, zero FA. No new training ran.

|Seed|Split|FA with prior|Prior-region reproduced alone|Target event reproduced by silence control|Not reproduced exactly in either control|
|---|---|---:|---:|---:|---:|
|1337|calibration|2|0|0|2|
|1337|development test|0|0|0|0|
|2346|calibration|9|6|2|1|
|2346|development test|11|5|2|4|

No FA was emitted in the padding/gap region. The 11 isolated-prior events are
repeated exposures to six unique failing prior recordings (two calibration,
four development-test); they are not 11 independent failures. These prior
transcripts are incomplete `ni3 hao3 xiao3` or `xiao3 xiao3 wo1` and must remain
nonwake. The zero-error training result does not cover these development voices.

All seven not-exactly-reproduced target events disappear at their original
keyword/time after decoder-prefix blanking. However, only SIX recordings have
no remaining FA. On seed2346 `test-28`, the combined run falsely accepts keyword1
at 1.725 s; both the silence control and blanked replay instead falsely accept
keyword2 at 1.925 s. That is an identity substitution, not a successful repair.
On `test-60`, the combined run emits a predecessor keyword1 false accept at
0.805 s, while the silence control emits a target keyword2 error at 1.625 s.
Counts are therefore not simply additive; this report does not ascribe that
suppression to a particular runtime state without another controlled experiment.

## Consequence

There are at least two actionable failure classes: isolated incomplete-word
confusion on development voices, and decoder-prefix dependence across nonwake
utterances. For the six disappearing-FA cases, clearing only prior decoder
evidence suffices while preserving the actual later acoustic outputs. This
supports a decoder-history contribution, not a claim that all RNN state problems
are absent. One changed error remains, so a blanket reset is not a validated fix.

Next controlled tests should separate incomplete-word generalization from
boundary-path retention, using train-source-only augmentation/supervision. Any
boundary reset/decay candidate also needs positives with legitimate within-word
pauses and the earlier startup regressions; do not ship an oracle onset reset.
Keep actual C calibration checkpoint selection separate from protected release
qualification. No replay volume increase or longer training timeout follows
from repeated predecessor exposure counts.

## Reproduction

```sh
python training/preceding_context_attribution.py \
  --pool /path/to/frozen-pool --pool-sha 20346a75b33086d8085ebee7e72eacf4baad55c453a241e4f6babd710e222ce5 \
  --model /path/to/model.kwm --runner build/kws_wav \
  --dump build/kws_posterior_dump --replay build/kws_decoder_replay \
  --source /path/to/preceding-context-report --output /new/output-directory
```

Outputs retain raw events, per-record geometry, original input file hashes,
control metrics, counterfactual sidecars/traces and partial completion state.
The 16 new tests cover canonical scoring, boundaries, duplicate capacity, event
substitution, malformed/partial sources, label/input binding and frozen pack
validation. Existing preceding-context14, startup21, frozen-pool13 and training
state18 suites also passed locally. These are software and development-evidence
checks, not generalization or physical far-field qualification.
