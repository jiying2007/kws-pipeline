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
