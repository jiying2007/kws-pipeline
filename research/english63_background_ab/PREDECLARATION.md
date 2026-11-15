# Frozen English natural-background continuation comparison

This is a single bounded research experiment, not qualification, checkpoint
selection or permission to promote either model. The previous native C screen
of the already exposed 517 LibriSpeech recordings produced 906 GRU proposals
(117 Hey Vosen, 789 Hey Mero) and 746 DS-TCN proposals (all Hey Mero):
[public run 43675351488](https://github.com/jiying2007/kws-data/actions/runs/43675351488).
Those observations motivate a direct comparison of added background supervision.
They do not establish its causal benefit in advance. Nonzero proposals and
`NO_BENEFIT` are valid successful scientific outcomes; no extra attempt follows.

## Inputs and original identities

The complete request is frozen in `protocol.json`. Only `pipeline_commit` and a
short descriptive `reason` are supplied by the data consumer. Its request ID is
`en63-background48-20261115-v1`; the experiment is
`english63-natural-background-continuation`.

The warm start is the **online** GRU and DS-TCN NPZ pair saved by
[run 43595338774](https://github.com/jiying2007/kws-data/actions/runs/43595338774),
artifact 14114695344. Its 100,653-byte archive SHA-256 is
`bc76805934df8eb158fa4259a75c26a4ea4276c15b83542acad5948a1f954b54`.
The original executed pipeline commit was
`fc59bef7fe108ce6accbf51d8de0bfcaae19e0bc`. Both new arms start from these same
per-kind checkpoints, not the earlier whole-recording classifier weights.

| Kind | Frozen NPZ SHA-256 | Seed for both arms |
| --- | --- | --- |
| GRU | `71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623` | 20261114 |
| DS-TCN | `167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935` | 20261215 |

The original English63 TTS archive, three-evidence review and frozen normalizer
are reused byte for byte. The unchanged label bridge is background 0, Hey Vosen
1, Hey Mero 2. Thirty original training clips retain their existing weak targets;
the twelve development clips and twenty-one ambiguous nearphone clips never
enter an optimizer. The same existing source/split auditor runs before training.

Natural source identities come from artifact 14090383839 of
[run 43538998200](https://github.com/jiying2007/kws-data/actions/runs/43538998200).
Its original transcript-absence rule, selected 517 FLAC members, speaker IDs,
sample counts, FLAC hashes and transcript hashes are checked for every row.
The public OpenSLR dev-clean tar remains 337,926,286 bytes, SHA-256
`76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3`,
licensed CC BY 4.0. Int16 lossless decoding uses NumPy 2.3.5 and SoundFile 0.13.1.
The old negative receipt did not contain PCM hashes; this run records fresh PCM
and crop identities after decoding. It saves neither audio nor raw natural
transcripts in git or its result.

## Deterministic study split and crop plan

Sort all forty decimal speaker IDs by the lowercase SHA-256 of the UTF-8 string
`en63-bg48-speaker-v1/` followed by the exact speaker ID (no newline), then ID to
break a tie. The first 32 speakers provide training candidates; the last eight
provide study development recordings. All forty speakers and all 517 recordings
were already exposed. This is a new optimizer exclusion boundary within an
exposed study, not a new independent holdout or a new recording.

For each training speaker, retain source recordings of at least 48,000 samples,
sort by SHA-256 of `en63-bg48-record-v1/` plus the exact source member (then member
for a tie), and take exactly four. Each center crop is
`start = floor((source_samples - 48000) / 2)`, `end = start + 48000` exclusive.
The resulting 128 crops are ordered lexically by source member. They total
6,144,000 samples, 384 seconds and 38,144 frontend frames (298 per crop).
The background class remains a weak transcript-absence proxy; `trusted_negative`
and `human_gold` remain false.

The eight study development speakers are 1462, 1993, 2412, 251, 3081, 3853, 6241
and 6295. Their 102 recordings are concatenated in lexical member order into a
separate development-only stream: 11,124,244 samples, 695.26525 seconds, 69,525
frames. No training clip enters this stream, and no training stream state crosses
into it. No silence, reset, discontinuity or padding is inserted between its
source members. Source-interval event annotations locate the dependency-end
clock; the rolling readout can depend on earlier sources, so these annotations
are not acoustic word attribution.

The independent metadata roster checksums are:

| Representation | SHA-256 |
| --- | --- |
| Lexical training member plus LF | `e1bef5e07944db2b18424885fb6b0868079d0392533fe6cc6c4d26e8d1b6cc42` |
| Training member TAB start TAB exclusive end LF | `3e2c49be5ffbcd11c134621a5c633b9d968d94761885600c2ac002644e15abce` |
| Lexical development member plus LF | `dd9188fdacdd302f5d41ae723176f09bbee14f3aef3c9a65e06c3ddb2f3d89f0` |

## Training comparison

Each of the four new checkpoints receives exactly 48 optimizer updates. Per-kind
control and treatment use the same warm start, seed, fresh AdamW state,
learning rate 0.001, weight decay 0.01, gradient clipping norm 1.0, CPU thread
count 1 and disabled MKLDNN. There is no best-epoch choice, early stopping,
shuffle, learning-rate change, threshold scan or extra evaluation forward.

Let `L_old30` be the unchanged existing mean cross entropy of the original thirty
weak MIL bags. The reused online readout pools mean and maximum over at most 200
causal encoder frames; the reused bag objective is the existing temperature-0.25
log-mean-exp of target-versus-background margins, excluding the first 49 frames.

```
control:   L_old30
treatment: L_old30 + 0.5 * mean(MIL_CE(new_background_window_i), i=1..128)
```

The original thirty-clip term is not diluted by adding more background rows to
a shared average. There is no new per-frame loss. Each treatment update performs
one old30 backward and four natural batches of 32. Each natural batch mean is
multiplied by `0.5 * 32 / 128 = 0.125`. There is one gradient clipping operation
and one optimizer step after accumulation. Features presented to each worker
contain only the original thirty clips and, for treatment, the fixed 128 crops.
The normalizer remains the original training-only mean/std, not a refit to the
natural backgrounds or development recordings.

The old `train_one`, `evaluate` and `verify_saved` model routines are never called:
they would import a different warm-start path or perform additional forwards.
Instead this entrypoint reuses only verified data loading, the existing frame and
MIL functions, model construction/safe NPZ loading, and NPZ writing.

Every update records old loss, natural loss if applicable, weighted objective,
gradient norm and completed-step count. Every arm records the initial and final
FP32 parameter-payload identities and the number of changed tensors. A ZIP hash
change alone is not accepted as a learned-weight change. Each completed NPZ is
immediately added to one atomically replaced `WEIGHTS.zip`; earlier checkpoints
survive a later arm's failure. The successful ZIP contains exactly four NPZs.

## Frozen C evaluation

Original `en63_stream.c` and `en63_stream.h` are reused unchanged. The original
exporter performs all its existing source-summary and normalizer checks and
must reproduce baseline header SHA-256
`912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16`.
The restricted new exporter changes only the two FP32 parameter arrays and NPZ
identity strings for each arm; all frontend tables and model layout stay intact.
It rereads the saved NPZs, checks the exact tensor roster/shapes/FP32/finite values,
and binds the payload bytes to the C header's literal values and ordering.

Evaluation order is baseline GRU, baseline DS-TCN, control GRU, control DS-TCN,
treatment GRU, treatment DS-TCN. Each child initializes a fresh state for the
development-only natural stream, finishes without padding, then initializes a
fresh state for each of the twelve original TTS development clips in lexical cell
ID order. The first natural stream is continuous across its 102 source files;
the twelve TTS clips are isolated. The frozen baseline is actually evaluated on
these exact new boundaries, not reconstructed by filtering old full517 events.

The threshold remains float32 0.8, minimum observation 50 frames, confirmation 3
frames, cooldown 100 frames, pooling macro 0 and feed chunk 160 samples.

| Counter | Per model child | Six model children |
| --- | ---: | ---: |
| Logical streams / initializations / finishes | 13 | 78 |
| Supplied samples | 11,518,484 | 69,110,904 |
| Native forward frames | 71,965 | 431,790 |
| Feed calls | 71,991 | 431,946 |

Unique evaluation source duration is **719.90525 seconds**, not six times that
independent exposure. The twelve TTS clips supply 394,240 samples and 2,440 frames;
the natural stream supplies the other 69,525 frames. It ends with a feed of 84
samples and four new samples after the last complete frame. EOF produces no
extra frame or proposal.

## Predeclared research gate

Evaluate each model kind independently using baseline B, control C and treatment
T. Every condition below must hold for that kind:

1. Natural total proposals B and C are both positive, and `2*T <= B` and `2*T <= C`.
2. For each natural keyword, `T_k <= min(B_k, C_k)`.
3. For each requested keyword's four TTS positive clips, treatment correct-class
   **unique clip coverage** is at least `max(B, C, 2)`.
4. For each requested keyword, treatment unique clips containing an opposite-word
   proposal number at most `min(B, C)`.
5. Across the four ordinary TTS background clips, treatment total proposal count
   is at most `min(B, C)`.

A zero B or C natural total cannot demonstrate the required relative reduction.
All six fixed evaluations still complete. Repeated proposals within one positive
clip cannot inflate coverage. Per-frame probabilities, peak probabilities after
the minimum observed frames and all events are retained as descriptive evidence;
they create no additional post hoc gate. Overall `RESEARCH_BENEFIT` requires both
kinds to pass; otherwise the result is `NO_BENEFIT`. Neither result permits model
selection, promotion, formal training admission, a household FAR/FRR claim,
acoustic endpoint truth or SSC305 qualification.

## Bounds, evidence and tests

One already-consumed public Actions claim authorizes one entrypoint invocation.
Only run attempt 1 of `jiying2007/kws-data` is accepted. Training workers each have
a hard 180-second wall limit, with 720 seconds for all four. Native children have
120 seconds each and 180 seconds for the entire six-child group. The outer data
consumer bounds the entrypoint at 1,200 seconds and saved verification at 180
seconds. Training and native processes stay in the outer process group. Linux
parent-death SIGKILL guards with explicit parent-PID race checks also terminate
children when their own supervisor disappears. No result extends a budget.

The CLI is `run_ab.py --trained-result-dir ... --tts-zip ... --review-json ...
--negative-evidence-dir ... --source-tar ... --output-dir ...`. The alternate
`--verify-result-dir ...` rejects all runtime inputs and imports no NumPy, Torch
or SoundFile. Its source closure contains the exact 24 paths in
`ab_contract.SOURCES`. The data consumer pins every source byte before running
that reader in an isolated stdlib interpreter.

The successful inner result is exactly seventeen files: PROTOCOL, INPUTS,
TRAINING, BUILD, MEASUREMENTS, SUMMARY and REPLAY JSON; SOURCE, WEIGHTS and RUNTIME
ZIP; `native.stderr`; and six `variant-kind.trace` files. WEIGHTS contains only
`control-gru.npz`, `control-dsc_tcn.npz`, `treatment-gru.npz` and
`treatment-dsc_tcn.npz`. RUNTIME contains `baseline.h`, `control.h`, `treatment.h`
and the corresponding three `.bin` files. PCM, decoded FLAC, feature buffers,
temporary headers and compile files stay outside the result directory.

Each trace has a 16-byte `EN63ABT1` header, version 1 and segment count 13. Each
segment starts with little-endian `<IIQQ>` index, row bytes 32, samples and frames,
followed by `<QQ3fI>` frame index, dependency-end sample, three probabilities and
emitted class. Each complete trace is 2,303,208 bytes. The saved reader replays
the fixed decoder from these probabilities, recomputes events, coverage and all
research gates, checks exact NPZ/header relationships and closes all saved file
hashes. Saved verification executes zero model forwards and zero optimizer steps.

`ab_cases.py`, wired into the existing C source check, uses only invented scalar
losses, NPZs, probability traces, PCM and a constant fake C classifier. It checks
gradient weighting, per-keyword/positive/background gates, safe NPZ decoding,
payload changes, exact export, actual file splices, independent segment clocks,
no EOF padding, rehashed summary/event forgery rejection, failed-run checkpoint
retention and hard timeout/parent-death behavior. No original audio or retained
model is executed locally. This PR creates no workflow or schedule.
