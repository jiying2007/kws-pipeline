# Goal-driven KWS data redesign (2026-11-02)

**Status:** proposed engineering/data policy, not evidence of a new model or
production readiness. This proposal does not modify frozen source records,
training admission, qualification, thresholds, or shipping configuration.

## Decision

Stop treating the number of generated audio clips, number of cloned voices,
or ordinary CI PASS as model quality. The data system should optimize **observed
end-to-end event behavior** on independent audio:

- 两个完整唤醒词：**你好小窝**、**小窝小窝**。单独“小窝”不是正例。
- Real-human/final-AFE FRR at or below the existing Phase A 5% point threshold
  and 6.5% one-sided 95% upper-confidence limit, including per-keyword and
  distance/orientation strata.
- Continuous negative-stream FAR at or below 0.1/h point and 0.13/h
  one-sided upper-confidence limit, using genuinely independent exposure.
- Trusted word-end-to-event p95 latency at or below 500 ms.
- Same frozen candidate, final AFE, SKU and physical SSC305 board qualifies
  the CPU, memory, power, capture integrity and multi-DUT soak contract.

These are **existing product qualification gates**, not synthetic-data sample
acceptance criteria. No change to commercial/*.policy.json is proposed.

## Audit of the current sources

1. kws-data's canonical catalog contains 42 historical Qwen3 development
   WAVs (20 human-reviewed synthetic, 18 ASR-train, four historical
   "holdout" ASR); all are observed research clips. Five preset slots are
   not five independent real speakers. The historical holdout is not a new blind
   acceptance set.
2. kws-pipeline's historical 384-clip speech-like base is diagnostic only
   under the reviewed-label training gate. The existing frozen shipping model
   was trained from a tone backend, not from the latest synthetic R&D.
3. The rights-scoped authorized-voice-clone batch (#688) can plan up to 512
   reference speakers / 32768 attempts and checks source-specific rights,
   file identities and replays. It has **not** demonstrated actual model
   synthesis, intelligibility, label accuracy or acoustic generalization.
4. Strict source-bound three-ASR and human-review receipts already exist.
   Unverified text/ASR strings must not be passed into these authorities.
5. A disjointness plan and data-iteration scorer already exist; the missing
   link is **feedback based on measured useful audio and independent event errors**.

Do not delete the historical negative experiments or alter their identity to
suggest training progress. The 2026-11-02 main snapshot reviewed for this
design was 38fc04146abfb0ac095f1281ca1c8fda1524e1f9.

## Target data architecture

Use distinct physical and authority partitions; a manifest is never itself an
admission permission:

| Partition | Content | Permitted purpose |
| --- | --- | --- |
| source_roster | Verified, rights-scoped originals; actual speaker/family/source IDs | Reproducible source inventory |
| generator_intent | Two wake phrases, explicit phonetic/context near misses, ordinary negatives, fixed seeds | Generate **requests**, no labels |
| candidate_audio | Immutable generated WAV + actual PCM hashes, original speaker lineage, model/seed provenance | Acoustic inspection |
| diagnostic_screen | Three frozen, independently executed ASR families, original text outputs, acoustic flags | Prioritization/quarantine only until authenticated |
| admitted_train | Strict source-bound reviewed/three-ASR receipts; train-only split and ancestor history | New CTC optimization |
| development | Source-family-disjoint, *previously exposed* data; error mining and threshold selection with separate histories | Adaptive development only |
| sealed_acceptance | Independent human-labelled, unobserved source families/sessions; no resampling after viewing results | One-time independent acceptance |
| physical_qualification | Real-speaker final mic/AFE continuous streams + SSC305 board logs | Governed shipping decision only |

**All descendants of the same original person/source recording/source family**
must stay in a single partition. Same voice cloned with different prompts,
model revisions, seeds, background overlays or speed perturbations remains the
same original speaker and derivative family. Keep entire overlap history; once
a sample influenced development, do not relabel it a blind holdout.

## Acoustic coverage to build, in this order

- **Wake positives:** both exact phrases, common natural prosody, quiet/fast
  speech, real inter-syllable pauses, preceding/following normal conversation,
  interrupted and resumed speech. A TTS request is not proof the complete
  phrase was spoken.
- **Contrastive hard negatives:** 你好小屋, 小窝 near other words, partial
  prefixes/suffixes, reversed or incomplete tokens, cross-sentence
  carry-over, near-homophones, wrong-word endings, television, speech from
  other people, repeated phrase fragments, and nonwake statements that contain
  keyword subparts. Do not tag an actually spoken full wake as negative.
- **Ordinary continuous negatives:** distinct household rooms, far-field
  reverberation, overlapping speech/music/TV, home-device noise and silence.
  Whole-stream event scoring and valid negative exposure duration matter;
  hundreds of repeated 1-second negatives are not 24 independent hours.
- **Real acoustic domain:** 3–5 m, orientation, room/reverb, microphones,
  speaker sessions, gain/noise, audio chain and motion/double-talk. Use real
  final-AFE recordings for blind qualification, not simulated RIR alone.

For every source-family × phrase × scene, measure actual successful
pronunciation / authenticating-label yield, exact PCM deduplication,
recognizer disagreement, recall failures, false-accept root causes and
latency. An inventory of reference voices is not an inventory of independent
real-user speakers.

## Immediate bounded pilot and stop/go criteria

1. Pin one legally authorized, diverse small reference roster and one
   offline provider revision. Run dry-run first; do not bulk-download or
   clone unknown open-source speakers without their specific authorization.
2. Generate a fixed *small* pilot covering both positive keywords, short
   near-miss families, ordinary negatives and more than one independent
   source family. Freeze prompt, seed and cost before running it.
3. Run independent unprompted ASR on **actual WAV output**, not requested text.
   Preserve native reports and hashes. Three-family screen matches may be
   diagnostic; actual training admission still requires the source-verifying
   toolchain in SPEECH_BASE_ADMISSION.md.
4. Run this extra offline audit against the private output, measuring WAV
   payloads and source-group/name confusions. A duplicate PCM across original
   speakers is a blocker, not independent acoustic diversity.
5. If positives are repeatedly missed by ASR, repair the generator/prompt,
   pronunciation and sample quality **before** adding more speakers. If
   negatives are transcribed as complete wake words, quarantine and inspect.
   If only one source family succeeds, do not generalize from aggregate yield.
6. Admit only genuinely reviewed train receipts, then run a same-runner,
   same-baseline ablation changing **one data variable**; keep decoder,
   frontend, thresholds, source assignment and evaluation windows pinned.
7. Mine actual FR/FA events from independent development streams into
   source-bound requests for the next iteration. Release only after
   independent human/final-AFE + real SSC305 qualification.

A pilot is a **no-go** for bulk expansion if it has unusable generated speech,
identity collisions, source-rights problems, inadequate yield, or an
unresolved systematic near-word false accept. These are engineering decisions,
not arbitrary counts advertised as a statistical guarantee.

## New read-only effect-screen auditor

Implementation: tools/voice_clone_effect_audit.py, with offline tests in
tests/test_authorized_voice_cloning.py. It is deliberately separate from the
existing admission and qualification authorities.

After the authorized batch completed with private manifest/summary:

    python3 tools/voice_clone_effect_audit.py \
      --campaign-root /restricted/voice-clone-output \
      --output /restricted/reports/effect-screen-without-asr.json

Optionally supply a local **diagnostic-only** JSONL result from real ASR runs:

    python3 tools/voice_clone_effect_audit.py \
      --campaign-root /restricted/voice-clone-output \
      --asr-screen /restricted/asr/screen.jsonl \
      --output /restricted/reports/effect-screen-with-asr.json

Screen JSONL row:

    {"item_id":"speaker-k1-s0","file_sha256":"<exact lowercase SHA-256>","observations":[{"model_family":"A","raw_text":"你好小窝"},{"model_family":"B","raw_text":"你好小窝"},{"model_family":"C","raw_text":"你好小窝"}]}

**Never produce the above by copying a requested target text.** It is a schema
example, not a source authentication or recognizer-policy verification.
Each ASR observation must come from an actually executed, independently
identified recognizer. Three differently named model families in JSON do
not prove actual recognizer independence or execution.

The auditor checks:

- The already-sealed batch summary/manifest SHA and count; the exact WAV and
  decoded PCM for each row; non-symlink audio paths under the private root.
- Original speaker/family/reference lineage; identical PCM across original
  speakers; repeated PCM from one speaker, which does not increase diversity.
- Both requested complete wake keywords; per-intent, per-text and
  per-source-family screening yield; negative wake confusions and ASR misses.
- Real human, continuous FAR/FRR and physical SSC305 evidence as **not
  evaluated**, never as an implicit zero or PASS.

**The auditor always returns training_admitted=false,
label_authentication_verified=false, qualification_allowed=false and
shipping_approved=false.** Its output is not a new downstream training format.
Actual materialization must remain behind the existing reviewed source,
lineage, ancestry, source-verifying ASR and training-admission validators.
Keep private audio/transcripts/results off public GitHub.

## Subsequent implementation (not yet executed)

- Connect a pinned offline Qwen3-TTS Base/CosyVoice-compatible adapter to a
  verified source roster, with prompt cache and per-call budget receipts. No
  paid API and no unlicensed reference voices.
- Build actual ASR replay and authenticated-source identity receipts, paired
  with acoustic completeness and duplicate-content checks.
- Feed false-reject and false-accept classes to a bounded source-family
  acquisition queue, instead of blindly multiplying phrase × voice × seed.
- Publish immutable per-candidate *measured* dataset/version cards:
  positive yield, negative subtype coverage, real audio hours, independent
  source families, FA/FR with confidence bounds, exact code/model/AFE/config
  identities and external hardware receipts.
- Keep qualification audio/label authority separate from training/dev and do
  not weaken shipping gates.

**Completion is demonstrated by observed improvement on a fresh, independent,
controlled data/model comparison, followed by real-world Phase A/B/C evidence.
A successful generation script or CI regression alone is not completion.**


## 2026-11-02 implementation: typed intent-level coverage

The authorized batch now accepts an optional, narrowly validated per-phrase
\`challenge_family\`, and the offline audit measures output and three-ASR
screening counts by \`intent:challenge_family\`. See
[\`AUTHORIZED_VOICE_CLONING.md\`](AUTHORIZED_VOICE_CLONING.md#typed-challenge-families-optional-backwards-compatible).

This closes the first gap between a large request matrix and
diagnosable near-miss categories. It **does not** verify that TTS pronounced
the intended words, and it does not prove that a candidate survived the
source-authenticated training gate. Earlier v1 campaigns remain readable
without retagging or rewriting any historical recordings.
