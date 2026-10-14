# Data and model effect iteration v2

Base: `b768b2a0d0dd467b7d863da93c272948b7204b9b` (2026-10-14).

This phase implements data audits and bounded experiments. It does not certify
an acoustic improvement or change the shipped model, thresholds, or decoder.

## Evidence and next executable steps

1. `../effect-data-inventory-v2/`: 385 recovered PCM16 files, including six
   original TTS outputs. File duration is not independent negative exposure.
   These recovered records remain development-only. The separately declared
   384-recording public speech-like base is not counted as locally recovered.
2. `../tts-quality-pilot-v1/`: matched Qwen 0.6B / 1.7B public preset pilot,
   initially one resource probe per model. Four voices, two keywords and two
   seeds give 16 planned cells per engine, including the probe. CosyVoice3's
   additional 16 cells are blocked pending an admissible reference-voice source.
3. `../tts-pilot-screening-v1/`: preserve native synthesis, explicitly derive
   16 kHz PCM, run two blind ASR families first and the third on disagreements.
   Two-model agreement, three-model agreement, review and human reference are
   separate evidence tiers. No homophone normalization maps 屋 to 窝.
4. `../model_comparison/`: current RNN and causal TCN research training and
   continuous streaming inference, with a common research decoder. Numeric CI
   uses explicitly artificial PCM and proves execution contracts only. A TCN
   deployment exporter/kernel is not provided by this research runner.

## Evaluation rules

- Preserve historical split and all shared original/augmentation identities.
  Previously inspected or trained material cannot become a sealed test set.
- Compare on the same data, feature frontend, decoder, operating-point search
  budget and event scorer. Select thresholds on development/calibration only.
- Report each keyword's misses, continuous false alarms per hour, exposure and
  uncertainty. Short repeated source schedules are not independent exposure.
- Keep conservative and accent-friendly policies explicit. A near-pronunciation
  tolerance policy is a product choice, not retrospective relabeling of errors.
- Resource probes and numeric smoke do not establish real speech performance,
  SSC305 runtime costs, or deployment qualification. Those need their own runs.
- Old r4/r5 results, execution controls and consumed-attempt evidence stay sealed.
  New probes use a new explicitly named first-push campaign, not old approvals.

## Completion boundary

At source publication, new TTS/ASR/model quality results remain pending. Actual
runs must retain source SHA, audio/checkpoint hashes, runtime limits, transcripts
and raw event scores. Failed or blocked cells stay visible in fixed denominators.
A comparison without admissible independent evaluation data cannot select a
qualified product winner.
