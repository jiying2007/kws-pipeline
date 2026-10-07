# Generated six-clip ASR source screen

This separately named contract accepts exactly six newly generated, hash-bound
blind PCM16 mono16k WAVs, each at most12s. It does not replay the old fixed30 set.
The two ASR models execute sequentially, once per clip, at most12 calls total.
The124-package lock,13 model assets,12 source-member pins and scientific adapter
functions are reused unchanged from the reviewed CPU recovery source.

SenseVoice uses native-fbank, no VAD/no ITN, CPU FP32 and1 thread, dither1.0;
exact dither RNG repeatability is not established. Qwen keeps CPU FP32/eager,
batch1, max256 tokens, empty context and automatic language. No targets, voices,
reference text or intended phrases are accepted by the blind input contract.

Only the producer's two-file blind artifact is downloaded by the ASR job. Its
archive and freeze hashes come through the same workflow's needs dependency.
All six WAVs are verified before setup. Setup≤20min and each recognizer≤10min,
within a50min standard public-runner job. Kernel CPU limits and outer wall limits
are retained; RSS/disk are sampled, not continuous hard limits. Package/model
response bodies remain capped at4GiB, expected3,340,966,447B. Actions/interpreter
and artifact transport are separate. Output≤20MiB, each file≤8MiB, retention1day.

Each model's raw output/status is frozen, including explicit zero-attempt status
when a preceding stage fails. The final raw freeze precedes every private plan
join. Comparison happens after both artifacts are saved, using the unchanged
punctuation/whitespace normalization and threshold-free signal/tail measurements.
Machine agreement is weak support only, never a human/gold label or training
admission. Failed or ambiguous outputs are quarantined without regeneration,
voice/seed sweeps, threshold tuning or default repeated listening.

Native recovery 3 retains both failed runs: 37335852099 has no retained precise exception; 37339583157 failed at SoundFile import with OSError before model loading or any TTS/ASR request. The sole package-body repair is the official SoundFile 0.13.1 manylinux x86_64 wheel, with identical Python/core metadata and bundled libsndfile 1.2.2; ASR keeps its original SoundFile 0.14.0 bundled wheel. This is an explicit runtime-environment repair, not bitwise replay of nonexistent prior audio.

Both setups verify installed packages/source pins and run an offline, independent 120-second import/native-I/O preflight before any model asset download. It constructs no model and loads no weights; fixed 12-sample FLOAT32/PCM16 checks are infrastructure fixtures, separate from the six requests. Failures stop before model downloads and retain safe stage/exception evidence. The preflight consumes the existing setup20min budget; all job, CPU, sampled-memory, disk, artifact and scientific limits remain unchanged.

Prior actual controlled downloads total 5,867,332,928 B; this attempt plans 2,934,954,359 B TTS plus 3,340,966,447 B ASR = 6,275,920,806 B, for a cumulative successful-plan total of 12,143,253,734 B. Each job remains capped at 4GiB controlled transfer; platform/interpreter/artifact traffic is separate. TTS setup attempt3 and ASR setup attempt1 still permit cumulative maximum6 TTS/12 ASR calls; Sohee0.
