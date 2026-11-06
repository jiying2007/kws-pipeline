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
