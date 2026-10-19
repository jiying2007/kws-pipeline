# Balanced40 control: fixed development gate failed

The single balanced40 control completed its prespecified training and all twelve candidate evaluation lanes. It is **not promoted**: at the 1500-ms inter-clip gap and threshold 0.7, both transports lost anonymous target identity S10/K2, reducing hits from 6/8 to 5/8. This is one source/keyword loss observed in paired transports, not two independent failures.

## Complete aggregate comparison

Both transports produced the same event structures and posterior chains within each model. Each row below applies to both transports. No target identity was gained; all other baseline target sets remained intact.

| Gap ms | Threshold | Target hits B→candidate /8 | Unknown B→candidate | Other unmatched B→candidate | Cross-seam B→candidate |
|---:|---:|---:|---:|---:|---:|
| 0 | .1 | 6→6 | 6→6 | 5→4 | 10→9 |
| 0 | .7 | 5→5 | 2→2 | 3→2 | 5→4 |
| 500 | .1 | 7→7 | 5→5 | 5→3 | 9→7 |
| 500 | .7 | 5→5 | 2→2 | 1→1 | 2→2 |
| 1500 | .1 | 7→7 | 3→3 | 2→2 | 4→4 |
| 1500 | .7 | 6→5 | 0→0 | 0→0 | 0→0 |

Hits, unknown and other-unmatched are mutually exclusive here; duplicates and ambiguous events are zero. Cross-seam is an overlapping annotation, not an additional partition. The reductions in unmatched/cross-seam events cannot compensate for a lost baseline target under the fixed policy. All seven guarded event-count categories were non-increasing, but the required baseline hit-set inclusion failed in two lanes.

Eligible weak-negative speech and generated-silence event counts remained zero. Their denominators per lane are only 4.16/5.56/8.28 seconds of speech and 0/2.72/9 seconds of generated silence at the three gaps. Regions around targets and unknown labels are excluded. These zero counts do not establish zero false wakes. Unknown labels were not converted to negatives.

## Training and evidence integrity

Original B initialized a head-only, no-KL recipe: 40 fixed balanced batches, 320 presentations, all 146 historical training clips. Loss was target-token-normalized full CTC plus 0.1 competition, averaged over each full eight-sample batch. Fresh Adam used learning rate 1e-4 and clip norm 5; 39/40 steps clipped. Each step and Adam update completed. Only the two final head tensor slices changed; all 28 other slices were preserved byte-for-byte. The exported head ZIP storage bytes matched the candidate payload's two head slices without loading tensors during review.

The final step40 was fixed in advance. Different steps use different batches, so first/last training losses are not a matched effectiveness comparison. This single arm tests a complete recipe; it does not isolate sampling effects or establish a KL causal result.

Candidate execution consumed exactly 7,447,040 samples, 24,930 feeds, 46,520 fbank frames and 15,504 network rows, with zero EOF finish calls. All twelve receipts, source/template bindings, native received-PCM chains, scores and mechanical comparisons passed saved-result integrity checks. Candidate source and policy hashes remained frozen. Numerical review made no new model calls.

## Scope and publication boundary

Evaluation reuses 20 of the 38-source exposed reference roster: 8 targets, 7 weak negatives and 5 unknowns. Unique source audio is 26.12 seconds. Three gaps, two transports and two thresholds are correlated conditions; they do not create independent audio hours. Host fast replay is not device or ARM qualification. Sample-coordinate delivery differences are descriptive, not measured word-end or hardware latency and not a gate.

The candidate has its own structural contract and binary identity. The unavailable old contract body was not claimed as semantically inherited; the earlier C-versus-Torch numerical FAIL remains. No quality or production qualification follows from the new registration.

`results.json` preserves all twelve lanes, anonymous gained/lost identity sets, improvements and regressions, denominators, delays and training scalars. It contains no private recording IDs, source paths, transcripts, PCM hashes, audio or model weights. Full private training/evaluation evidence, including candidate head/Adam/payload, is archived separately and must not be publicly released. No additional tuning or inference is part of this report.
