# Fixed definitions, declared before trace calculation

This is a post-hoc descriptive audit of the existing new A600 observation only.
It does not revise the fixed greedy endpoint or select a decoder, checkpoint,
threshold, reset policy, or beam width. The 12 exposed machine-labelled training
positives and the previously published sentence-end results are already known.
Only the new per-prefix traces and run statistics are uncomputed at declaration.

## Decoder and counting rules

- Blank is token 0. KW1 is [1,2,3,4], KW2 is [3,4,3,4]. SHORT=[3,4] is a
  diagnostic competitor, never an authorized shipping wake word.
- At every original frame, accumulate raw greedy CTC collapse and a width-32,
  top-1 CTC prefix beam with no language model, lexicon, length reward or bonus.
  Initialize once per clip and never reset, including after a target match.
  Preserve both blank and nonblank prefix mass; ties use token-tuple order.
- Exact match means the entire accumulated hypothesis equals the sequence.
  Record all maximal contiguous frame-index intervals of equality for KW1,
  KW2 and SHORT. First-match frame is the first interval start. A terminal
  stable-match start is the start of an interval reaching the last frame;
  it uses hindsight and is not an online decision rule.
- Extra match entries=max(0, number of equality intervals minus one). These
  are hypothesis revisions/re-entries, not physical duplicate wake events.
  Separately count nonoverlapping occurrences of the intended complete target
  in each accumulated hypothesis, reporting its maximum and final count.
  These also do not measure runtime repeat-trigger behavior.
- Save every top-1 hypothesis, exact-match label, blank argmax flag and trace
  transition. Count frames in each equality class; do not turn counts into
  recall, false-accept rates, or independent observations.
- Independently compute exact CTC log-probabilities for KW1, KW2 and SHORT at
  every frame prefix using the standard blank-expanded forward recurrence.
  For KW2 samples, log(P(KW2)/P(SHORT)) is defined only when both probabilities
  are finite. Record its first strictly positive frame, all positive intervals
  and final value. Zero is sequence-probability ordering, not a tuned threshold.
  Prefix beam rank/score and exact CTC probability are different quantities.
- No extra confidence threshold, blank gate, refractory interval, or sensitivity
  sweep is introduced. Do not translate the product threshold .55 onto exact
  sequence probability or beam probability.

## Time and evidence boundary

For zero-based frame k, nominal complete-window availability is
(400+320*k)/16000 seconds from the source PCM clip origin. This follows the
pinned no-padding 400-sample/320-hop frontend and one-output-per-frame causal
RNN. Report frame indices and optionally this nominal coordinate. It excludes
compute/transport latency and has no human keyword-end reference.

The frozen prefix_frames is an RMS<1e-4 leading prefix, not a human speech or
word boundary and not the runtime VAD. Saved RMS is read only for metadata
binding, never as a new threshold or to infer a word end. No audio is opened.

The inspected product runtime uses a max-path keyword Trie, approximate
normalization, acoustic-only per-token geometric confidence, retention and
root/fuzzy-child restrictions, VAD, optional trailing-blank/prefix arbitration,
decoder reset after hits, and engine refractory suppression. This audit does
not implement that contract or claim its events. Product source at the pinned
commit is retained only to substantiate this separation. LPs plus selected
metadata support prefix timing diagnosis, not faithful full-engine replay:
there is no saved runtime keyword pack/settings/state/VAD/event trace for this
new, unexported research model, nor human word ends or long negative streams.

## Validation and preservation

Verify input SHA256s, accepted manifest identity and observation membership,
plan/cohort/source identity links, record order, lengths, finite normalized rows,
and saved greedy paths. The analyzer uses only Python standard library.
Check exact DP and unpruned beam against exhaustive small CTC fixtures at every
prefix, check repeated-token blank separation, and check interval/occurrence
helpers. Write output with exclusive creation outside every original evidence
directory. Read-only source snapshots must match their Git blob hashes.
