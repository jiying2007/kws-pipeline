# Six fixed Qwen stock-voice candidates

Exactly six generation attempts: Ryan and Aiden in the prospective train role,
Ono_Anna in prospective development; each has the two phrases predeclared in
plan.json. Sohee remains ungenerated. No reference audio, cloning, new voices,
new seeds, retries, gain repair, threshold sweep or training is part of this run.

The model is Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, revision
85e237c12c027371202489a0ec509ded67b5e4b5, from the Qwen team. Its pinned model card
identifies Apache-2.0. Model weights are fetched only into the ephemeral runner
and never uploaded in artifacts. Source: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5

CPU FP32/eager4-thread generation uses the explicit recipe in plan.json. The
92-input CPU package lock is newly resolved; this is not an exact reconstruction
of the old0.6B environment or waveforms. Controlled package/model input bytes:
2,933,666,464; response-body cap4GiB, separate from Actions/interpreter transport.
Setup≤20min and generation≤20min under the unchanged outer supervisor, in a50min
public standard-runner job. RSS and disk limits are sampled. Per-clip SIGALRM is
cooperative, with the outer wall/CPU guards enforcing the process limits.

Keep both native24k FLOAT WAV/value hashes and the derived16k PCM16 WAV/value
hashes. Resampling finite/length/peak/rail diagnostics are evidence only; no gain
or waveform repair is applied. Generation results remain unqualified synthetic
candidates. Full generation evidence is frozen and bounded to16MiB; the separate
blind artifact is bounded to4MiB. Only its six content-addressed WAVs plus blind
job.json reach the ASR input contract. No model weights, environments or logs
are uploaded. Artifact retention is1day; preserve results immediately.

Recovery 2 retains the scientific plan and locks. Run 37335852099 remains FAILED_NO_RETRY: setup verified, generation failed before any request marker, ASR skipped; its precise exception was not retained. This preparation adds early stage and bounded safe stderr diagnostics; it does not establish or claim to fix the original model/import failure. TTS setup attempt 2 and ASR setup attempt 1 permit at most 6 TTS and 12 ASR calls cumulatively. Prior controlled transfer was 2,933,666,464 B; the proposed new paired transfer remains 6,274,632,911 B (combined expected 9,208,299,375 B). Original per-job bounds are unchanged.
