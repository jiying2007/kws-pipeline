# Anonymous Base clone and blind dual-ASR pilot

Four train-only reference groups × four requested phrase classes produced 16 complete native-tail PCM outputs. EOS, reference/native-tail boundaries and PCM identities were privately verified. This is a small technical pilot, not independent-speaker testing.

## Blind recognition

Qwen 3-ASR-0.6 B and SenseVoiceSmall each loaded once and recognized each output once:2 loads,32 calls,32 completed,0 unknown,0 retries,0 third-decoder calls. Only Mandarin language conditions were provided, with no target text/hotwords. Original native evidence was retained.

Removing only punctuation and whitespace yielded 14/16 full-text agreements. All 14 also matched the requested phrase; agreement on a different phrase was 0/16. Two disagreements remain quarantined without correction. Strict raw-text agreement was 0/16; 14 pairs differed only in punctuation/whitespace, while two also differed lexically. Agreement is machine weak evidence, not truth or human gold. Nothing was admitted to training.

| Requested class | Agreed and matched | Quarantined |
|---|---:|---:|
| 你好小窝 |4/4|0|
| 小窝小窝 |4/4|0|
| 你好小屋 |2/4|2|
| 小屋小屋 |4/4|0|

Anonymous reference-group coverage was 4/4,3/4,4/4,3/4. No individual reference identities or transcripts are published.

## Resources and cleanup

TTS:262.41 s wall,886.71 s sampled CPU,5,922,041,856 B sampled peak RSS. Qwen ASR:370.73 s wall,368.64 s CPU,6,282,051,584 B peak RSS. SenseVoice:60.78 s wall,59.57 s CPU,2,466,578,432 B peak RSS. Original supervisors confirmed leader reaped and zero live model processes; durable result readbacks were verified. These are sampled process-group observations, not hard cgroup guarantees.

ASR allocation explicitly included shared PCM, per-stage output/logs, control records, reserve/result archives, one readback per archive and one active download temporary:96,128,192 B declared new-local peak, with a 2 GiB free-disk floor. No model weights or second inference-input copy were created.

## Preserved TTS accounting exceptions

The historical generic summary retained an incorrect reserved_calls value of 8; independent evidence establishes 16 completed generations. Original artifacts were not rewritten. The original 32 MiB total incremental allocation omitted archive duplication and was exceeded. A subsequent 82 MiB delivery-peak amendment allowed preserving and delivering the evidence; this does not retroactively establish original-budget compliance.

## Interpretation and privacy

Four references and correlated recognition errors limit conclusions. Reference assets came from a third-party mirror and were not verified byte-for-byte against an official distribution; historical speaker exposure remains UNKNOWN. No performance superiority, product acceptance, independent holdout evaluation or event-timing truth is claimed. No audio, audio hashes, private paths, Library identifiers, individual reference identities, weights or credentials are included here.

Machine-readable companion: [anonymous-base-clone-dual-asr-summary.json](anonymous-base-clone-dual-asr-summary.json).
