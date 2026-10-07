# Core subset: source, model and data scope

This notice applies only to the independently verifiable model/training-core
checkpoint. The full target's additional fixed12 and Serena18 audio and their
numerical/benchmark evidence are not included here. Overall full-archive
completion remains false; `CORE_STATUS.json` pins the intended full target.

## Source

Archived WeKws FSMN, CMVN and streaming decoder source retains Apache-2.0
copyright/attribution. Torchaudio's Kaldi-compatible frontend retains its
BSD-2-Clause notice. Full license texts are included. Native adaptations select
a six-class head, adjust streaming state/decoder behavior, and use explicit
mixed-precision boundaries; individual original/public hashes disclose modified
files. Sanitized trainer/contract excerpts omit private orchestration and are
historical scientific source, not untouched originals. No third-party source
or model contribution is relabeled self-authored or blanket relicensed.

## Models

The A20 and F20 checkpoints are trained derivatives of the separately retained
iic donor. The available recorded donor provenance declares Apache-2.0; its
published source is the
[donor identity record](https://github.com/jiying2007/kws-data/blob/main/research/2026-09-30-compact-head-distillation/donor-reference.json).
Model-card revision: `7b61475f2b7d6b0348f624f0853303a3a374f7bc`;
README SHA-256: `8488305cd82bfa76faa1af65770b314a549b4b6afbe82b41582e7c2cb228ee6c`;
weight revision: `68e1625545621d1dfb921866c4bc6b6a811b2685`.
This is recorded provenance, not a fresh model-card byte verification or a
warranty about rights in unknown original pretraining audio. Training
checkpoints include optimizer/RNG state and are not inference-only packages.

## Twenty D20 source WAVs

Exactly twenty originally used, weak-labeled D20 training WAVs are retained with
their original WAV/PCM digests and frame counts. They are Qwen3-TTS stock-preset
synthetic outputs: thirteen Vivian and seven Uncle_Fu, with one 0.6B and nineteen
1.7B generations. They are not real-person recordings, human-gold labels, or
product-qualified data. Detailed pinned generator/source identities and the
bounded distribution assessment are in `training/F-A-D20` after materialization.

The reviewed primary source includes the
[pinned Qwen3-TTS CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/README.md).
Source/model Apache-2.0 declarations do not create a universal output license.
The full-target review did not identify a separate preset-specific redistribution
or impersonation restriction for these exact synthetic research inputs.
`commercial_output_license` remains `not-established`; the bounded research
distribution assessment is not a new commercial-output rights grant or
real-person endorsement. Historical training use does not grant current default
catalog admission, fresh holdout status, qualification or shipping permission.
