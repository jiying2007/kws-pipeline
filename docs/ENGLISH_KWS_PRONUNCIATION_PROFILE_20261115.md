# English wake-word pronunciation policy — updated 2026-11-17

Status: active future-target policy. The latest owner instruction requires Hey Mero to be calibrated from actual American English evidence; the 2026-11-16 image and earlier requested IPA no longer establish its final target. Native-US blind-reading evidence, pronunciation distribution and independent KWS evaluation have not yet established that target.

## Current target status

| Phrase | Current reading field | Interpretation |
|---|---|---|
| Hey Mero | `ipa: null`, `raw_espeak_ipa: null` | Unresolved. Do not choose a spelling-based, historical or synthetic reading as final. |
| Hey Vosen | /heɪ ˈvoʊzən/ | Preserved existing research request, not independently confirmed acoustic gold. This update makes no /z/ to /s/ substitution. |

The active machine-readable policy is `configs/research/english-pronunciation-profile-20261115.json`, now schema v2. The original image-based profile is preserved byte for byte at `configs/research/history/english-pronunciation-profile-image-request-20261116.json`. Its American/British IPA, raw eSpeak strings and stress hints remain historical generation metadata.

For future target-positive generation or admission, an absent active IPA or `new_target_positive_admission_allowed: false` means stop that positive path; do not fall back to the historical profile. Explicit pronunciation-discovery comparisons may continue as unadmitted research. This does not stop already frozen weak-intent experiments or reuse of existing audio.

## Data and annotation rules

- Preserve the original 63 English WAVs, their PCM identities, the three ASR observations per clip, the 21 quarantined nearphone records, and all consumed experiment artifacts unchanged.
- Do not regenerate or overwrite the 63 clips.
- Requested TTS text and ASR agreement are not human acoustic gold. Do not promote a clip to positive, absent, trusted negative, or training-admitted solely because its prompt or transcript matches this table.
- For any future new recording or generated sample, retain source/voice/generator ancestry, raw PCM hash, sample rate, channel count, duration, exact spoken phrase, and an independent pronunciation review status.
- Pronunciation variants must be explicit and separately annotated. No automatic phoneme substitution or broadening of the target set is allowed.
- Keep calibration, development selection, and locked confirmation source-disjoint. Previously exposed LibriSpeech and English63 material remains exposed research data.

## Phonetic acceptance before model training

A sample can be used as a positive only after its actual pronunciation is reviewed against the selected reading and its permitted-variant policy. The review must distinguish:
- phrase onset and the pronunciation of "Hey";
- the actual vowel, rhotic sequence and stress realized in Mero, without forcing the historical /iː/ request;
- the observed realization of Vosen, retaining its existing /voʊzən/ research request as a comparison label rather than acoustic truth;
- omissions, insertions, truncation, disfluency, and competing nearphones.

Obtain blind readings from actual American English speakers by presenting the phrase without IPA, respelling, stress cues or example audio. Record the observed pronunciation distribution before choosing a main reading and reasonable variants. Keep calibration, development and confirmation speakers separate. Borderline accents remain uncertain rather than automatically rejected; a synthetic demonstration or ASR consensus does not establish a speaker's pronunciation.

## Candidate architecture research

The primary model track remains strong pretrained causal FSMN-CTC with training, quantization and target-resource optimization. The saved English embedding plus `1536→32→3` head is a research challenger and useful software-delivery baseline; its progress does not complete the FSMN-CTC track. The retained GRU and DS-TCN remain negative research comparators; their clip-level mean/max models and subsequent event-head experiments did not pass the joint positive/background gate.

The openWakeWord project is a useful implementation reference for:
- a shared pretrained speech-embedding backbone;
- a small DNN or two-layer RNN classifier;
- explicit continuous-stream inference and negative-data evaluation;
- separation of preprocessing, embedding extraction, and keyword head.

Licensing boundary:
- openWakeWord source code is Apache-2.0.
- Its README identifies the Google Speech Embedding feature extractor as Apache-2.0.
- The distributed openWakeWord pretrained wake-word models are CC BY-NC-SA 4.0 and are not eligible for a commercial product path.
- Any training data/feature corpus must be audited independently; the openWakeWord negative corpus and any NC-licensed derivatives must not enter a commercially intended model.
- No third-party model, feature archive, dataset, or runtime is admitted by this document. Exact commit, weight digest, model card, license text, dependency closure, feature geometry, streaming state, C implementation feasibility, and target resource measurements are required before adoption.

References:
- https://github.com/dscripka/openWakeWord
- https://github.com/dscripka/openWakeWord/blob/main/LICENSE
- https://github.com/dscripka/openWakeWord/blob/main/README.md
- https://arxiv.org/abs/2002.01322

## Fixed completion boundary

A research candidate is not a landed English model until all of the following exist for the same frozen configuration:
1. reproducible source, exact weights, training recipe, and provenance;
2. independent positive pronunciation evidence for both phrases and source-disjoint development/locked confirmation;
3. continuous natural-negative exposure with a valid FAR/hour denominator;
4. per-word FRR, combined FAR, confidence bounds, and word-end callback latency;
5. streaming PCM frontend/model/decoder parity and tested reset/discontinuity behavior;
6. C/ARM export and actual SSC305 BSP/audio-chain CPU, RAM, latency, thermal, and long-run measurements;
7. a reproducible release package and model card with all rights and limitations.

V4 numeric targets remain unchanged: combined FAR <= 0.1/h, per-word FRR <= 5%, and word-end-to-consumable-callback P95 <= 500 ms, with one-sided 95% upper bounds reported. These are target gates, not claims already achieved.

## Historical pronunciation records and model identities

Historical requested readings include Hey Mero /heɪ ˈmɪəroʊ/ and /heɪ ˈmiːroʊ/, and Hey Vosen /heɪ ˈvoʊsən/ and /heɪ ˈvoʊzən/. Keep those original request labels and every associated audio, source, weight, trace and experiment identity unchanged. This policy update neither verifies nor rejects a recording based only on which historical request produced it.

The original 63 seven-voice Qwen WAVs, the Qwen/Piper pronunciation probes, the final144/snapshot160 weak-intent data and already saved heads remain historical research evidence under their existing exact contracts. A new target or variant decision must be an additive acoustic annotation; it must not rewrite their hashes or silently promote them into a new independent test set.

The old raw sequences `heɪ mˈiːɹoʊ` and `heɪ vˈoʊzən` belong to the image-based generation request, not an instruction to future Mero target generation. Do not feed them to blind-reading participants. The owner's nine-window “没有唤醒词” review supplies negative evidence only and does not freeze a positive pronunciation.
