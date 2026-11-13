# Original English 63-WAV research intake (non-admitted)

This is the next source-bound step after the merged English weak-intent
bridge. It **does not** train, label, certify, tune, export or promote a model.
Its sole purpose is to reuse the **exact original 63 audio files** for future
reviewed research, without regenerating TTS or re-running any ASR/IPA models.

Original upstream source: kws-data Actions run 43242278962, artifact
13954744587, original ZIP SHA256
`f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc`.
The data-repo evidence join is run 43346200939; the exact previously joined
JSON SHA256 is
`89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12`.
These are historical byte identities, **not newly generated audio**.

With a previously downloaded and locally restricted original Actions ZIP and
the exact already joined JSON report:

```sh
python3 -I -S -B training/english_research_audio_intake.py \
  --tts-zip /restricted/original-english63-artifact.zip \
  --source-review /restricted/english63-three-evidence-review.json \
  --output-dir /restricted/english63-weak-audio-intake
```

The original artifact contains **126 WAVs per 63 recording cells**:
63 `audio-16k.wav` (16-kHz mono PCM16, the ONLY files consumed by the
intake) and 63 `native-f32.wav` (native floating-point companions, NOT
PCM16 trainer inputs). The intake requires exactly one of each path per
cell, rejects any unexpected WAV layout, and requires the exact 63 saved
cell/WAV/PCM identities. Both native and PCM16 members are covered by
the unchanged original ZIP SHA256; there is no new model conversion.

The program checks the original ZIP bytes, all 63 unique PCM16 WAV/PCM hashes,
16-kHz mono PCM16 geometry, exact cell IDs, five train-role and two
development-only synthetic voice-family roles, and all 21 nearphone
quarantines **before creating any output**. It copies bytes unchanged to
three deliberately nontraining directories:

- `weak_diagnostic_only/`: 30 requested-intent diagnostic rows, not gold.
- `development_no_tune/`: 12 exposed development-only rows.
- `nearphone_quarantine/`: 21 potentially confusable utterances.

Every row remains `training_admitted=false`,
`formal_training_eligible=false`, `human_gold=false`,
`trusted_negative=false` and `ctc_target_ids=null`.
The 3-class whole-recording English weak head is kept separate from
Mandarin CTC. An ordinary text request never authenticates what was spoken.
The private, create-only result contains 63 byte-identical WAV copies and
one source-linked MANIFEST.json; it does not publish the audio or any model.

To reach a **new English FSMN research model**, the following are still
necessary: independently verifiable English-acoustic label authority,
speaker/source-family isolation, approved train-only input, a bounded
source-pinned training experiment, same-source evaluation and independent
accent/nearphone/continuous background tests. The existing owner-held R0
has unacceptable exposed nearphone behavior (5/6 wake proposals), so this
intake cannot promote R0. No SSC305 qualification is implied.

Original GitHub Actions audio artifacts have limited retention; verify the
saved ZIP SHA256 before expiry rather than regenerating the 63 original WAVs.
All tests here use invented PCM only, and run in the existing normal/-O/-OO
training-admission CI inventory. No new workflow or scheduled task is added.

## Original-artifact readback correction

The first imported version only tested a mock ZIP with one WAV per cell;
it rejected the authentic source ZIP because it incorrectly counted the
63 native float companions as additional PCM16 WAVs. The corrected
archive-member policy is keyed by **exact path, cell and variant**, not
filename extension alone. Fixed source analysis independently confirmed
63/63 PCM16 file digests, 63/63 matching source cell IDs, 63/63 PCM
digests, 63 companion native WAV cell IDs, and 16-kHz mono PCM16 geometry.
These checks alone are not formal label authority or new model training.
