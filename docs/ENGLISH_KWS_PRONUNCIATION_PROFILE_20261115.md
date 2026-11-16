# English wake-word pronunciation profile — 2026-11-15

Status: owner-confirmed final target readings transcribed from the user's 2026-11-16 pronunciation-standard image; research metadata only. This file does not establish acoustic gold, training admission, a trained model, model selection, FAR/FRR, or deployment qualification.

## Frozen target readings

| Phrase | Owner-confirmed IPA | Scope |
|---|---|---|
| Hey Mero | /heɪ ˈmiːroʊ/ | American target; MEE-roh, long /iː/, stress on first syllable |
| Hey Vosen | /heɪ ˈvoʊzən/ | American target; VOH-zən, voiced medial /z/, stress on first syllable |

The user-provided image is the controlling source. American targets are Hey Mero /heɪ ˈmiːroʊ/ and Hey Vosen /heɪ ˈvoʊzən/. The same image lists British forms /heɪ ˈmiːrəʊ/ and /heɪ ˈvəʊzən/. Preserve accent variants as explicit metadata; do not merge them into one ambiguous phoneme target. Do not silently replace these readings with spelling-based G2P output or a TTS engine's default pronunciation. This target specification does not prove that any existing WAV contains that pronunciation.

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
- the long stressed /iː/ vowel and following rhotic /r/ sequence in Mero (MEE-roh);
- the initial voiced labiodental fricative /v/, stressed /oʊ/, and voiced medial /z/ in Vosen (VOH-zən);
- omissions, insertions, truncation, disfluency, and competing nearphones.

Until an owner-approved review protocol defines acceptable acoustic variation, borderline accents remain uncertain rather than automatically rejected. The present IPA choices do not by themselves define every accent's allophonic tolerance.

## Candidate architecture research

The next English model comparison should evaluate a lightweight causal event head over an independently license-qualified speech embedding backbone, against a causal FSMN-CTC research candidate. The retained GRU and DS-TCN remain negative research comparators; their clip-level mean/max models and subsequent event-head experiments did not pass the joint positive/background gate.

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

## Superseded pronunciation records (immutable, not relabelled)

The earlier research profile used Hey Mero /heɪ ˈmɪəroʊ/ and Hey Vosen /heɪ ˈvoʊsən/. Those readings are superseded by the user's 2026-11-16 image. All audio and experiment artifacts generated against the earlier readings remain immutable historical research evidence, but are not positive samples for the current target. In particular:
- the original 63 seven-voice Qwen 0.6B WAVs remain exposed and are not relabelled;
- the four Qwen 1.7B instruction-controlled WAVs remain unqualified for the new target;
- the four Piper raw-phoneme WAVs generated against the superseded IPA remain unqualified for the new target;
- no prior ASR agreement, raw eSpeak string, or requested text establishes acoustic gold.

Current target-specific generation must use raw phoneme inputs equivalent to eSpeak hˈeɪ mˈiːɹoʊ and hˈeɪ vˈoʊzən, after confirming the pinned phonemizer maps the symbols exactly. The user's American IPA is the primary target; British variants are tracked separately, not silently mixed into the primary training label.
