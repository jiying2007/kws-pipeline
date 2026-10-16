# Grouped-balance saved-evidence diagnosis (post hoc)

This is a **post-hoc descriptive supplement**, not a preregistered endpoint, new training campaign, decoder change, model selection or production qualification. It preserves the original fixed-600 raw-greedy results: both arms train KW1=1/4 and KW2=0/8; probe A=0/4, B=1/4. The four probe recordings remain exposed, machine-labelled, non-test data. No probe logits were decoded by this supplement.

## Reproduce

The three public files are `analyze.py`, `report.json`, and this README. Python standard library only; no Torch, model, optimizer, ASR, TTS, external executable, network request or weights. Supply the preserved original result directory and its independently accepted hash manifest:

```
python -B analyze.py --result /path/to/original-result --accepted-hash-manifest /path/to/acceptance/hash-manifest.json --output /path/to/new-posthoc-report.json
```

The script pins the exact acceptance manifest SHA256 in `MANIFEST_SHA`, then checks every consumed original file against that manifest before parsing it. It creates output exclusively and rejects output within the original result directory. Reproduction must use a new output filename. The report records the script SHA256, acceptance manifest SHA256, frozen verifier identity, and each consumed input SHA256. Logs/weights/audio are not copied into this supplement.

The embedded log-domain CTC recurrence reproduces the independently reviewed frozen verifier's recurrence (verifier SHA256 `cf974939751107b8e6e6c15d34bff1e5830e173a1a45cb3c34d1642681d36bcc`). It computes sequence probability by summing valid CTC paths, respecting repeated-token blank separation. These derived probabilities do not rerun the acoustic model or independently attest its gradients or hidden states.

Before the first saved-logit beam calculation, widths were fixed in code to **32 and 64**, each returning **top 1**, with all five vocabulary tokens, no lexicon, language model, insertion reward, threshold search or target restriction. Widths were not tuned to improve results. Equal scores use lexicographic prefix ordering. A four-frame synthetic case enumerates all 625 raw paths and verifies the unpruned beam's top sequence/probability before analyzing original data. This is a small correctness check, not proof that a finite beam finds the globally highest-probability sequence on these recordings. The returned beam sequence is also scored by the exact CTC recurrence.

The analyzed beam snapshots are A500, A600 and B600, all training recordings. A500 was inspected because the original saved curve showed deterioration afterward; this is explicitly an outcome-informed diagnostic slice, **never a replacement for A600, a selected checkpoint or an independent evaluation**. Both beam widths are reported, even when they agree. The existing raw-greedy endpoint and decoder remain unchanged.

## Findings

### 1. A suffered late deterioration, rather than simply failing to improve

A500 already had raw-greedy KW1=4/4 and KW2=0/8. Its CTC objective was about 0.00996. By A600 it was about 0.121, while the prefix blank constraints still passed.

Receipts store losses/gradient norms **before** each update; named observations store outputs **after** their update. Do not conflate the timing:

- Pre-update 594 total loss: 0.007822.
- Pre-update 595: 0.007945, gradient norm 0.133266.
- Pre-update 596: 0.028547, gradient norm 9.986142; this is above the fixed clip threshold 5.
- Pre-update 597: 0.621408, gradient norm 2.852263.
- Pre-update 600: 0.166777; after update 600 total loss was about 0.12207.

The first large per-sample loss change at pre-update 596 was qwen06-05 (the 71-frame KW1 record): its weighted loss rose from about 0.00083 to 0.02118. Broad deterioration followed at 597. This temporal localization is evidence of an unstable training trajectory; it does **not** prove the cause was that sample, a specific hidden-state phenomenon, Adam, or learning rate alone. Per-sample gradients, optimizer state and reusable weights were not saved. The source used gradient clipping, so it is incorrect to say clipping was absent or that a reported pre-clipping norm is the actual applied norm.

B's final CTC objective continued decreasing from approximately 0.02457 at step 500 to 0.01720 at 600. B was not completely smooth (earlier spikes exist), but did not exhibit A's late large deterioration. Because A and B use different weighted objectives, their raw scalar losses should not be treated as a common loss scale for ranking models.

### 2. Raw-greedy failure mixes decoding differences with residual fitting failure

At A500, all eight KW2 recordings greedily collapsed to `[3,4]`, but each assigned a higher total CTC probability to the correct `[3,4,3,4]` than to that greedy sequence. At B600 this comparison favored the complete target for five of eight KW2 recordings. A target beating one short sequence is not proof that it beats every possible sequence.

The fixed finite-beam top-1 diagnosis gives:

| Saved training snapshot | Original greedy KW1/KW2 correct | Beam 32 KW1/KW2 | Beam 64 KW1/KW2 |
|---|---:|---:|---:|
| A500 | 4/4, 0/8 | 4/4, 8/8 | 4/4, 8/8 |
| A600 | 1/4, 0/8 | 2/4, 0/8 | 2/4, 0/8 |
| B600 | 1/4, 0/8 | 3/4, 5/8 | 3/4, 5/8 |

Beam does not uniformly improve every item: at A600 kw1s-07 is greedy-correct but beam top-1 adds an extra initial token. These are positive-only recordings; this analysis does not measure negative-audio behavior, false activations, streaming latency or decoder deployment cost.

Thus A600 is not merely an unfavorable greedy collapse: its sequence probabilities also deteriorated. B600 contains a substantial decoding-related difference and four remaining beam errors. It is incorrect to label KW2 as wholly unlearned based only on greedy 0/8; it is equally incorrect to claim a decoder substitution solves the whole problem.

### 3. Source inspection found no evident batching/padding/target error

The pinned runner builds record-aligned features, true lengths, flattened targets and target lengths in the same order. The model initializes independent hidden states for every batch row and uses forward-only recurrence. Padding comes after valid audio; CTCLoss receives true input lengths, and diagnostics/anchors slice valid frames. Prefixes are record-specific, including qwen06-10's zero prefix. The frontend uses frame-local feature centering, not a statistic mixing training and probe recordings.

These facts support the intended computation; static review and saved probabilities cannot independently reconstruct gradients or prove weight-state execution. The 12 training recordings span 63–255 frames. Dividing each sample's CTC NLL by its own frame count gives different raw-NLL coefficients to long and short recordings by design. Class balance balances these normalized sample objectives, not raw unnormalized NLL sums. No evidence currently justifies calling that a label/padding bug.

## Recommended next bounded comparison — proposed, not authorized/executed

Separate optimization stability from decoding. Do not change architecture, sample split, normalization, anchor, weighting, decoder and learning rate in one experiment.

The next most focused numerical proposal is a **new one-arm 600-update uniform-weight campaign** that matches original A exactly through 500, then uses learning rate 0.0003 for updates 501–600 instead of 0.003. Keep every other numerical setting fixed. Reuse the original A evidence as a frozen historical control, not as a new independent run. Exact initial, pre-step scalar/gradient and complete observation comparisons through 500 must fail closed before further updates on any mismatch. This is a deterministic mechanistic contrast at one seed, not a general statistical claim.

Proposed new budget: 600 optimizer calls and 608 full-batch forwards (600 training + the same eight observations); no probe forwards, no new ASR/TTS, no retries, no exported weights, no old-budget reuse. The schedule is fixed before running, not triggered adaptively by loss or gradient. Report fixed-600 raw-greedy class outcomes/prefix constraints plus the fixed-600 objective and last-100 loss/gradient range. Do not substitute a favorable earlier point. Saved-logit sequence diagnostics may remain clearly secondary and must not alter the original decoder.

This proposed lower learning rate may prevent deterioration yet slow improvement or leave greedy repeat omissions; that outcome is informative. A failure or inconclusive result stops at the fixed budget. If a concurrent control rather than exact historical replay is required, that is a different two-arm 1200-update proposal needing its own explicit budget. Any new numerical campaign needs separate approval and reviewed frozen source; **nothing in this supplement authorizes execution**.
