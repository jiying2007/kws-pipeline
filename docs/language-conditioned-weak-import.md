# Offline source-language-conditioned weak import

This is a separate candidate delta against the reviewed maintained speech-data
queue. It does not edit or replace the prior anonymous/auto execution protocol,
old failed results, model/voice budgets, or production admission gates.

## What this adds

- An explicit `source-language-conditioned-asr-v1` dispatch. Qwen uses the
  language-only `Chinese` prefix, SenseVoice uses `language=zh`. Both are blind
  to target text, but are not language-blind. `detected_language` stays null and
  `language_is_model_independent_observation` stays false.
- A stdlib-only importer retaining each original result ZIP, every ZIP member,
  optional frozen delivery/readback sidecars, and failed-import evidence before
  interpreting outcomes. Fresh output directories only, no overwrite or retry.
- Frozen archive SHA-256 identities and item/opaque-slot associations supplied
  in the caller's **private plan**. Public source contains protocol and public
  source identities only. Do not publish private plans, PCM, result bodies,
  Library identities, or per-item execution artifacts in this repository.
- Historical original plans/events/artifacts captured independently from current
  events. Historical generation receipts can prove existing TTS/PCM association,
  without consuming a new generation budget or rewriting an old ASR attempt.

## Verification boundary

The importer checks the full ZIP roster, size/digests, duplicate/traversing members,
actual archived PCM16 and its normalized float32 digest, model/revision/asset and
source-lock declarations, archived successful worker/process completion, and
parsed/raw hash bindings. Qwen's actual input IDs, preserved prefix, returned
sequence, generated tail IDs, prompt template, EOS, and dual captured decodes
must agree. Sense's original result, captured decode calls/attempts, raw IDs,
metadata/lexical boundary, tokenizer identity, and frozen corrected parser must
agree. The bounded metadata variants supported here are Chinese speech without
ITN and neutral/unknown emotion; other variants are rejected for review.

No tokenizer is imported. Token-to-text mappings are not independently decoded
again: the reviewed native capture and exact frozen archived bytes are retained
and cross-checked. This proves saved-byte consistency, not cryptographic run
attestation or acoustic transcription accuracy. Complete decoder termination is
not proof that every spoken sound was transcribed. Supplied Library readback
sidecars remain parent-attested evidence, not an execution signature.

Current execution facts and offline validation have separate fields. A new
import rejection cannot turn an already observed native forward into an
uncalled slot, erase its original successful completion, or authorize a retry.

## Source rights and lineage

Conditioned TTS/clone candidates require a bounded source-rights review, not a
free-form `license` string or `research_use=allowed` alone. This candidate's
review route is narrowly pinned to the official Qwen CustomVoice model/card
revision and built-in Sohee voice. It checks complete model asset inventory,
generation plan bytes, intended request/seed, original generation receipt and
archive, and the exact resulting WAV/PCM. It does not claim a real speaker's
identity/consent, public audio redistribution rights, or commercial eligibility.
A clone cannot pass this stock-voice route unless its evidence honestly meets
the same source contract; a new source needs its own reviewed protocol.

The prior `rights=unknown` record stays unchanged in historical evidence. A new
bounded internal-noncommercial-project review may be supplied independently.
All original intent, recipe, source identity, speaker/reference roots and split
stay bound to the historical plan. Shared lineage must stay in one split. An
observed development sample is not an independent evaluation holdout.

## Research output

Punctuation normalization is unchanged; it never merges 窝 and 屋. Actual machine
text supplies CTC targets. Requested TTS intent stays separate. Disagreement,
missing/tampered evidence, unsupported speech, missing rights, and invalid CTC
lengths quarantine the item. Two agreeing machine recognizers do not establish
human gold or production quality.

The reused `training/weak_speech_dataset.py` consumes the generated
`weak-train.jsonl`, checks the package manifest and screened report, and returns
frontend feature matrices plus nonblank CTC targets. The conditioned route also
cross-checks intent, protocol and language-provenance fields against the screened
report; missing or changed fields fail even if package checksums are refreshed.
This is a research weak
consumer only: no optimizer, training run, model export, product acceptance, or
independent evaluation is performed or authorized.

## Run offline tests

From the candidate/repository root:

```
python3 -S -B -m unittest discover -s tests -p 'test_*speech_data_queue.py'
python3 -S -B tests/test_language_conditioned_queue.py
python3 -S -B -O tests/test_language_conditioned_queue.py
python3 -S -B -OO tests/test_language_conditioned_queue.py
```

The new fixtures are hand-built synthetic archived observations, explicitly not
actual model executions. No model package, tokenizer, weight, network service,
ASR, TTS, or training is loaded/called by the implementation or tests. Prior queue
regressions and the reused loader are tested alongside the new protocol.
