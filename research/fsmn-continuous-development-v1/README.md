# Anonymous continuous development evidence, 2026-10-19

This is one completed, bounded host incremental replay experiment: 12 lanes = three inter-clip gaps × two transports × two thresholds. No EOF flush was called. The supplied-sample endpoint clock comes from actual feed boundaries. Six paired transports have identical posterior chains and event structures after excluding delivery fields.

## Results

Both transports give the following counts. The first three event columns are mutually exclusive and sum to all emitted events. Cross-seam is an overlapping annotation and must not be added to them.

| Gap (ms) | Threshold | Target hits / 8 | Unknown events | Other unmatched events | Cross-seam events |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 6 | 6 | 5 | 10 |
| 0 | 0.7 | 5 | 2 | 3 | 5 |
| 500 | 0.1 | 7 | 5 | 5 | 9 |
| 500 | 0.7 | 5 | 2 | 1 | 2 |
| 1500 | 0.1 | 7 | 3 | 2 | 4 |
| 1500 | 0.7 | 6 | 0 | 0 | 0 |

There were no duplicate-target or ambiguous events. Unknown labels remain unknown. All other unmatched events in this run crossed a segment boundary; they are retained, not silently dropped or asserted to be acoustically verified false alarms. The original scorer's `unmatched_events` includes unknown events; this summary separates them.

Eligible false-event counts were zero, but eligible negative speech was only 4.16, 5.56 and 8.28 seconds per lane at the three gaps; eligible synthetic silence was 0, 2.72 and 9 seconds. Target and unknown regions, including 1.5-second guards, are excluded. These restricted zero counts do not mean zero non-hit activations or establish real-world FAR.

## Scope and timing

Only 20 of the 38-source reference roster are represented: 8 target, 7 known-negative and 5 unknown-label clips. All were previously exposed. Unique source audio totals 26.12 seconds. Conditions reuse the same sources; synthetic gaps and repeated transport/threshold conditions are not independent exposure hours or independent tests.

At threshold 0.7, the zero-gap and 500-ms conditions each hit 5/8, but one target was gained and another lost. Gap changes context, scores and boundary attribution; aggregate improvement is not evidence of uniform per-target improvement.

Supplied endpoint minus last decoded node is 50–60 ms for fixed-320 transport and 45.4375–84.8125 ms for variable transport. This comprises a fixed 45-ms required context plus batching delay. It is not a measured word-end latency, wall-clock callback latency or microphone/device latency. Matching uses coarse clip labels and node spans, not word-timing gold.

The host processed 7,447,040 supplied samples, 24,930 feeds, 46,520 fbank frames and 15,504 network rows across the 12 lanes. Summed lane wall time was 12.613 seconds; maximum sampled child RSS was 4,435,968 bytes. These are host fast-replay observations, not ARM or device validation, a real-world FAR result, or a generalization claim.

## Review and publication boundary

`results.json` contains aggregate anonymous fields only. Review checked all frozen source hashes, binary identity, 12 completed ledger entries, bindings, receipts, native terminal counters, actual source PCM hashes, reconstructed stream/chunk hashes, scoring and six transport pairs. Review reran no inference. Source/binary digests identify the reviewed configuration; the full private reproduction archive remains separate.

These two files are intended for a separate research-evidence PR. They contain no audio, transcript, recording identifier, private source path, source PCM digest or model weights. This summary does not grant permission to publish private evidence.
