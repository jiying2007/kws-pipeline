# KWS V4 bilingual execution baseline — 2026-11-14

The user approved execution of **KWS 中英文唤醒最终迭代方案与执行计划 V4.0**
on 2026-11-14 (Asia/Shanghai). This is the repository execution baseline for
that plan. It supersedes V3's prospective priorities; it does not rewrite
V3 inputs, consumed experiments, failures, labels or actual measurements.

## Scope and completion

- Mandarin: 你好小窝 and 小窝小窝; the pretrained self-developed FSMN-CTC
  remains the main research candidate. Keep separate full-word IDs and results.
- English: Hey Mero and Hey Vosen remain candidates for the English-speaking
  and European markets. Pronunciation, permissible variants and source evidence
  must be established before treating requested spellings as acoustic labels.
  A missing dictionary entry is not a rejection of a name.
- Use one engineering interface and evaluation contract, with separate language
  packages by default. A simultaneous bilingual deployment is a separately
  measured configuration, including combined false accepts and resource use.
- L1 means reproducible research source, actual weights and inference. L2 means
  independent acoustic confirmation of a frozen complete configuration. L3
  adds actual SSC305 BSP/audio-chain/resource/latency/stability confirmation.
  These are completion levels, not the historical model name R1.
- Missing independent recordings or physical hardware blocks only the relevant
  claim. Research with existing eligible inputs continues.

## Inherited state

Pipeline baseline before V4 execution: `1fe4c947aea69574de1de40ed9410d447ca64117`.
Data baseline: `db85340282df2da8e401012b1df174ca70d8f8e0`.
The already-completed isolated native C resource experiment (#806) was reviewed
and integrated at `d28ad923c83291803b498bb498101f7ce92a9ca8`; it was not rerun.

Preserve the following distinctions:

1. R1 FSMN is a development research model. Its two additional initialization
   confirmations vary two new output rows (282 initial elements), not every
   source of training randomness. Original FP32/MKL disagreement remains a
   failure under its original contract; Reference64 has its own contract.
2. ACTIVE113's measured size/speed benefit is not the full-OTHER correctness
   guarantee. The latter keeps the full competition and exact fallback. ARM
   instruction/parity evidence under QEMU is not SSC305 performance.
3. English 63 original Qwen WAVs and 189 ASR calls exist. Their formal training
   admission, human-gold and trusted-negative flags remain false. Five train
   voice presets, two exposed-development presets and the 21 nearphone rows
   retain their existing roles; presets are not independent human speakers.
4. The retained 6,531-parameter GRU and 4,803-parameter DS-TCN use whole-clip
   mean/max readout. Chunk-consistent feature/encoder output alone does not
   make that readout an online event detector.
5. The official bilingual 3M reference is a Zipformer transducer with INT8
   encoder/joiner and FP32 decoder. The original-length 63-WAV replay has five
   Mero events; the old nine included four dependent on 660 ms appended zero
   PCM. Neither count is human recall. The failed INT8 chunk8 comparison stays
   rejected; do not automatically repeat the same comparison.
6. The 517-source/40-speaker/61.09-minute LibriSpeech concatenation observation
   is one source exposure, even across four order/chunk combinations. It is
   constructed read speech, not independent household continuous-negative gold.
7. #806 measured a fresh host C process: peak RSS 57,720 KiB, model weights
   5,446,114 bytes, host feed p99 3.561917 ms. These are different quantities;
   they establish neither SSC305 memory/CPU nor word-end wake latency.

See [V3 baseline](KWS_V3_EXECUTION_BASELINE_20261107.md),
[R1 scope](KWS_V3_FSMN_R1_DEV_20261107.md),
[initialization scope](KWS_V3_CONFIRMATION_20261110.md),
[original English intake](ENGLISH_63_ORIGINAL_AUDIO_INTAKE_20261113.md), and
the retained result records under `research/english63_weak_pilot/` and
`research/sherpa_pcm/`.

## First execution package

1. Reuse existing Mandarin batches and English63 audio/ASR evidence by exact
   batch/source identity; the various historical Mandarin 64-item batches and
   natural64 windows are different inputs. Produce explicit
   per-keyword present/absent/unknown evidence and concrete unresolved fields,
   without turning ASR non-detection, requested text or forced alignment into
   truth. Evidence strength, permitted use and independence are separate fields.
2. Add a bounded causal online readout to the existing English student lane.
   The first experiment is a declared research system comparison: one fixed
   200-frame trailing window, one seed and 48 warm-start optimizer updates per
   retained GRU/DS-TCN candidate. Consume only the 30 existing train-role weak
   intent rows; the 12 exposed-development and 21 nearphone rows never enter
   the optimizer or parameter/threshold selection. Clip-level multiple-instance
   supervision does not create word boundaries. Fixed proposals remain
   uncalibrated research events, with FRR/FAR/word-end latency NOT_RUN.
3. Save new actual weights, optimizer/update accounting, per-input identities,
   proposal events, bounded-state/chunk checks and a fresh saved-weight replay.
   This is a newly declared successor to the retained clip experiment; never
   replace its source, weights or original results.
4. A reusable public data-owned executor consumes a versioned request and the
   original same-repository Actions artifacts, then checks out one immutable
   pipeline source SHA. Source-only PR checks and an actual research worker are
   separate. Persist terminal outputs before another stage consumes them.
5. Advance Mandarin new-data training only with actual eligible inputs and a
   separately declared finite recipe. Existing R1 remains the retained main
   research model while these inputs are prepared.

The first online experiment deliberately has no acoustic promotion decision.
Its useful result is a genuinely trained, continuously executable, replayable
research successor with truthful limits. A poor outcome is retained, not
hidden by a threshold or seed sweep.

## Data and experiment limits

- Train/development/calibration/locked confirmation are separated by real
  source/voice/clone ancestry. Reordering, augmentation or a new synthetic seed
  cannot create an independent source. Small representative review batches
  resolve shared ambiguities; uncertain rows remain excluded as appropriate.
- D1 planning budget: 32 source identities per language, two words x 20
  realizations (1,280 positives), with separate initial development/calibration
  identities. Generate in 128–256-item batches only after the preceding stage
  is useful. D2 expansion to 128–200 identities is conditional on observed gain.
- Independently recorded confirmation pool: initially 30–50 real speakers and
  500 events per word; later 1,000 per word. These counts refer to the confirmation
  pool, not the combined training/development collection.
- At most three active trainable routes per language and at most two finalists.
  Start with one configuration/seed, then use three complete training seeds for
  project-trained finalists. An immutable official pretrained reference does
  not need pointless retraining to satisfy a seed count.
- At most two heavy training jobs on independent resources; one heavy slot on
  a shared host. A completed/rejected experiment is not automatically repeated.
- Cache audio/model/decoding identities. A normalization-rule change recomputes
  the decision, not ASR. Retry only transient failed stages, at most twice;
  unknown remote execution is reconciled before any replay.

## Acoustic and deployment confirmation

Per actual configuration, target total FAR <=0.1/h and per-word FRR <=5%, both
point estimates and predeclared one-sided 95% upper bounds. Target actual
word-end-to-consumable-callback p95 <=500 ms; also report late, missed, duplicate
and wrong-keyword events, and 500 ms deadline recall. For two simultaneously
enabled keywords, use their combined false accepts in system FAR.

Declare event matching before testing (initial maximum post-word window <=1 s).
Primary fixed-window FRR uses recordings whose complete observation window was
confirmed before inspecting detections. Report observed short-recording hits
and censored nontriggers separately; do not retain only successful short clips.
Do not append silent PCM or reset at every utterance without declaring that
different protocol. Word-end audio timestamps and callback clock are distinct.

Report distance/room/accent/source strata; do not infer each stratum passed from
an overall mean. Development negative screening may start at 5–10 h; prepare
100 unique hours of independently collected continuous background per final
configuration with a fixed scenario mix and stopping horizon. Under the stated
Poisson assumptions, zero false events over 30 h only places the one-sided 95%
upper rate at about 0.1/h. Repeated source audio adds no unique exposure.

Select words/models on development evidence. Freeze quantization, final AFE,
decoder, threshold and deployment implementation before independent L2 testing.
A material L3 adaptation requires renewed confirmation for the affected claims.
Measure 1/3/5 m, accents, motor/charger noise and self-playback separately.

Initial optimization budgets (proposals, not measured SSC305 limits): KWS
single-thread <=0.05 CPU-second/audio-second, preferred weights <=2 MiB,
incremental steady PSS 8–16 MiB and initialization incremental peak <=32 MiB.
Measure AFE separately and the complete audio-to-callback chain under concurrent
robot load. Freeze or explicitly revise budgets after actual BSP baseline.

## Resources and execution state

Public `kws-data` / `kws-pipeline`, Dot and existing authorized/free execution
resources only. No Codex/ChatGPT Work shared quota, private repository, new
periodic/hourly patrol, repeated verified TTS/ASR or unsolicited general cleanup.
Source-only checks can run without ML; actual model work uses the declared
public GitHub Actions worker or an independently verified authorized host.

Use existing source/corpus verification and protected PR checks. Do not replace
them with a plan file, CI status or self-declared receipt. Only a real worker/job
is running; queued, failed, rejected and externally blocked are distinct.
Never claim a physical board, independent acoustic pass or continued background
activity without the corresponding current executor and raw measurements.
