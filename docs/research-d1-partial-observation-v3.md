# D1 partial observation input boundary

`training/research_d1_inputs.py` explicitly accepts the data adapter
`kws-active-d1-partial-observation-source-v3`. The retained data loader must
replay the entire package before the pipeline consumes any row. This addition
does not modify the model, vocabulary, exposure recipe, training loop, or
development evaluator.

## Returned audio and consumed unknown calls

The v3 adapter authenticates returned audio from a partially completed
generation job using its original claim, real generation receipt, native bytes,
and separately retained job outcome. It also authenticates any later returned
audio against its own continuation job. A timed out job is not relabeled as a
completed job, and a new observation package does not overwrite its original
outputs.

Only actual observed audio rows reach `aggregate_batches`. Their
`generation_outcome` remains exactly `completed` or `failed-native`, with the
original `kws-bulk-clone-complete-v1` or `kws-bulk-clone-failure-v1` receipt. A
failed native generation remains uncertain, has no selected full transcript,
and contributes zero CTC and keyword loss even if recognizers agree.

A consumed call with no return is a separate accounted-coverage entry. It has
no invented PCM, weak-view row, observed transcript, blank target, or negative
label. The pipeline rejects `generation_outcome=unobservable-consumed` even if
the proposed row has zero weights. Zero weights do not establish that audio
exists. Nonfailed uncertain rows may retain observed text while all masks and
weights remain zero.

`complete_plan_coverage` in the existing aggregate continues to mean that every
one of the 256 planned cells has an observed row. An external accounted closure
may separately prove where every planned cell ended up, including unknown
calls and returned audio that cannot form a legal input. It must not turn this
aggregate flag true or describe accounted cells as returned audio. Training
preparation still requires the real closed packages and a separate audited
coverage declaration; no future manifest or checkpoint hash is guessed.

## Preserved experiment

The original v1 and v2 adapters retain their existing row projections. The v3
path uses the same global plan plus cell identity checks, full package replay,
original PCM deduplication, and source-family/speaker/ancestor role isolation.
Local clip names may repeat between packages; global cells may not.

The fixed 2601-output vocabulary remains the official 2599 outputs followed by
the two previously declared task characters. New whole-transcript OOV items
remain unsupported. The existing recipe still uses 600 draws, 150 optimizer
steps, 300 full-positive draws, and 0/30/60 near draws for the 0/5/10 arms.
Missing required pools block an arm instead of creating examples. The three
initial float states must match the first64 run after real preparation and
before training resumes from the immutable queue.

The original 16 development rows retain their exact PCM, transcript, masks,
weights, and label bindings. Additional observed development rows are reported
separately. A single development pass per arm produces the two summaries by
read-only grouping. These are predeclared machine-weak engineering diagnostics,
not independent human validation, event FAR, threshold selection, or a winner
selection procedure.

## Validation scope

The existing stdlib contract suite adds mixed v1/v2/v3 package orchestration,
explicit v3 identity retention, whole-text projection, failed-native zero loss,
and rejection of unknown audio rows, promoted failures, duplicate cells, and
cross-role ancestry. Its replay double is an interface test; it is not evidence
of native audio or ASR authenticity. Production acceptance additionally requires
the new data loader's complete source and observation replay. No numeric or
model run is needed for this adapter change.
