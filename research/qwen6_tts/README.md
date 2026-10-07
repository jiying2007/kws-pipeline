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
2,934,954,359; response-body cap4GiB, separate from Actions/interpreter transport.
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

Native recovery 3 retains both failed runs: 37335852099 has no retained precise exception; 37339583157 failed at SoundFile import with OSError before model loading or any TTS/ASR request. The sole package-body repair is the official SoundFile 0.13.1 manylinux x86_64 wheel, with identical Python/core metadata and bundled libsndfile 1.2.2; ASR keeps its original SoundFile 0.14.0 bundled wheel. This is an explicit runtime-environment repair, not bitwise replay of nonexistent prior audio.

Both setups verify installed packages/source pins and run an offline, independent 120-second import/native-I/O preflight before any model asset download. It constructs no model and loads no weights; fixed 12-sample FLOAT32/PCM16 checks are infrastructure fixtures, separate from the six requests. Failures stop before model downloads and retain safe stage/exception evidence. The preflight consumes the existing setup20min budget; all job, CPU, sampled-memory, disk, artifact and scientific limits remain unchanged.

Prior actual controlled downloads total 5,867,332,928 B; this attempt plans 2,934,954,359 B TTS plus 3,340,966,447 B ASR = 6,275,920,806 B, for a cumulative successful-plan total of 12,143,253,734 B. Each job remains capped at 4GiB controlled transfer; platform/interpreter/artifact traffic is separate. TTS setup attempt3 and ASR setup attempt1 still permit cumulative maximum6 TTS/12 ASR calls; Sohee0.
