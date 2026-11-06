# Bounded ASR for eight new source references

`tools/research_reference8_asr.py` composes the existing native ASR batch controller into one queue-supervised job. It consumes eight independently retained source references, with six train-candidate and two development-only roles, and rejects overlap with the original 32 source identities. Original audio, normalized audio, source bindings and the audio-only decoder manifest remain separate. No reference text or peer transcript is supplied to a decoder.

The fixed three cached models are Qwen-ASR 0.6B, SenseVoice and Whisper small. The existing Whisper recipe explicitly supplies Chinese; this is not language-independent discovery. Results remain weak evidence, with no automatic training admission or human-gold claim.

## One deadline and at most 24 calls

The smoke phase selects the first and last input references: two references times three models, or six calls. The full phase requires those completed results and consumes only the remaining six references per model, or 18 additional calls. Both phases use the same native output directory and the same immutable `job.json`, including its first 540-second absolute deadline. The queue has a 600-second outer limit. Four CPU threads and the inherited, supervised shared heavy-model lease remain required.

The controller rejects an already consumed output directory. A claim without a complete result stays unresolved; the full phase cannot use it as a completed smoke result or resubmit it. Phase and cumulative claim/result counts are retained, and partial or failed execution is reported without automatic retry.

## Preparation and queue binding

Declare the request only after the complete source-input capsule has an actual save acknowledgment. The declaration binds the request inputs, original source exclusions, runtime metadata, cached model identities and exact execution-source bytes. Before `run`, separately save and acknowledge a preparation capsule whose context contains the full pin of the original request.

Queue the `run --request` command with that original request's absolute path. The preparation pin includes its path, so a queue recipe copy with identical bytes is not the same pinned object. The queue's own recipe copy and `KWS_RESEARCH_RECIPE_SHA256` still bind those same request bytes. List the expected individual regular output files in the queue contract; directories are not valid file-hash outputs. Keep any corrected pre-execution queue declarations as earlier records.

The declaration and verification commands use metadata and opaque-byte hashes. They do not load a model. Execute the run only through the existing supervised research queue. Preserve native outputs, claims, results, logs, phase summaries and the sequence result, then perform the existing native readback audit and separate reference-content comparison. The declared budget is a ceiling, not evidence that any call ran.

## Regression scope

`tests/test_research_reference8_asr.py` exercises the real existing batch controller with synthetic workers. It verifies first/last smoke selection, six-result reuse and 18 new full calls, one deadline across all six worker launches, no full phase after deadline exhaustion, refusal to reenter consumed outputs, unresolved-claim preservation, supervised-lease enforcement, and original-request acknowledgment binding. The fixtures use no real models or network and run in normal, `-O` and `-OO` modes.

Actual execution and content-check results must be reported from their own receipts. Source preparation, successful fixtures and a frozen request do not establish successful ASR, correct transcripts, independent generalization, natural false-alarm rate or device performance.
## Reusing a saved pilot for new bounded full batches

`active_asr_batch.py run --phase full` keeps its default rule: the first and last
inputs must already have completed smoke results for all three models in that
same job. A separate, explicit `--completed-pilot PATH
--completed-pilot-sha256 SHA256` option supports subsequent new audio batches
after an actually completed and saved pilot. It does not change the reference8
wrapper, its frozen requests, or any previous output.

The private proof uses schema `active-asr-completed-pilot-v1`. Its exact fields
are `schema`, `inputs`, `output`, `source_root`, `queue_result`, `audit_receipt`,
`checkpoint`, `checkpoint_descriptor`, and `checkpoint_ack`. Each file field
contains its absolute `path`, `bytes`, and `sha256`; the three directory fields
are absolute paths. The canonical checkpoint verifier authenticates the real
provider acknowledgement and every retained byte. The pilot input manifest,
queue completion, real audit command/stdout/stderr/exit status, native job,
complete call/native/result matrix, original audio, inventory, and numerical
sources must be members of that saved checkpoint. A local ZIP alone does not
qualify as persistence.

`verify-pilot` takes the same proof path/SHA plus `--runtime` and `--threads 4`.
It checks only saved bytes and metadata, with zero model imports or calls. The
runtime/model/decoder identities must equal the completed pilot; the new input
manifest is the only model-identity difference permitted. The numerical worker,
its repository dependencies and locks must match the saved source byte for
byte. The batch input preparation, model identity, native cache validation, and
worker launcher functions must also match the saved source as Python syntax
trees. This permits the explicit controller addition without relaxing the
numerical source requirement.

Actual use requires `full`, the fixed order `qwen06 sensevoice whisper`, four
CPU threads, one interop thread, `--budget-minutes 9`, a fresh output directory,
and the inherited supervised shared heavy-model lock. The new job binds the
pilot proof and receives its own `3 × N` call ceiling and one 540-second absolute
deadline. Each model processes the entire new roster once, so there are three
worker loads instead of a new smoke/full pair. An already consumed output is
rejected, including an incomplete or unknown attempt; this option grants no
retry or deadline reset. The default smoke/full path remains unchanged.
Both original and derived PCM hashes must be disjoint from the pilot before
any job or output write. Changing clip IDs, manifest order, or input paths does
not make an old recognition input new. Cross-chunk source-cell and PCM isolation
is also frozen in the stage preparation before either new ASR job starts.

For the first64 continuation, the original plan and seeds select 56 new cells
as two fixed chunks of 28, while the first eight remain reused. Each chunk's
actual ASR input count is the subset with the original TTS technical-completed
status, at most 28. Technical failures and unknown calls remain in the full
generation ledger and their native evidence is retained; the selection never
uses ASR agreement or transcript quality. The existing ASR export can also
contain decodable technical failures, so its eligible count must not be treated
as this stage's technical-success count. Each ASR job retains the same native
`job.json`, `claims`, `results`, `native`, and `attempts/full-001` layout, including
three worker receipts and three model-load receipts. The existing saved-only
auditor and per-chunk D1 weak-view consumer remain the downstream interfaces.
Generated audio and machine agreement do not establish human labels, independent
test performance, false-acceptance rates, or hardware qualification.
