# SenseVoice relocated tokenizer repair · 2026-10-10

**Software repair only; runtime not executed and continuation not admitted.**
The [actual v3 result](evidence/run-38018029787/README.md) remains overall FAILURE:
16 generated clips, Qwen-ASR 16 successes, SenseVoice one failed attempt and 15
unrun. No TTS, ASR, Docker or model call was made to prepare this change.

## Proven defect and evidence boundary

The unchanged loader verifies the tokenizer at
`/runtime/models/sensevoice/chn_jpn_yue_eng_ko_spectok.bpe.model`. The retained
post-decode binder omitted the path argument, so its default instead points
under `/code/research/qwen6_asr/runtime/models/sensevoice/`, which is absent from
the staged source profile. If reached, that check necessarily rejects the path.
This is a reproducible source defect. The v3 failure receipt retained only
`ValueError`, so it cannot prove that this check was the first exception; model,
capture or earlier result validation could also have failed.

## Narrow implementation

- New `sense_adapter.py` passes the worker's validated runtime model root to the
  exact retained tokenizer verifier before inference and again during captured
  token binding. The path must be absolute, unaliased, inside the fixed
  `models/sensevoice` layout, and a regular single-link file.
- The token-binding body changes only its explicit path parameter. Tokenizer
  SHA-256, vocabulary/unknown semantics, corrected parser hash, captured raw IDs,
  re-decode equality and rich metadata boundary checks remain intact.
- The source-specific inference wrapper preserves the same PCM binding, model
  receipt check, scientific inference arguments, exact-once consumption and
  post-call PCM validation. No retained qwen6 file or module global is modified.
- Fixed diagnostic stage and error-code enums identify validation checkpoints.
  Unknown messages are marked UNCLASSIFIED. Only a bounded UTF-8-rendered
  exception prefix digest/count is retained, never raw message text, paths from
  exceptions, URLs, locals or credentials. This digest is explicitly not native
  stderr. Exceptions retain their original identity.
- `diagnostic.model_inference_returned` is set immediately after the SenseVoice
  inference call returns and remains available if later validation fails. The
  existing v1 receipt's `model_forward_completed`/`forward_completion_confirmed`
  fields retain their historical full-adapter-completion contract. A false value
  there alone never proves that no acoustic forward occurred. Uninstrumented
  Qwen call diagnostics leave the new checkpoint null, not falsely observed.

## Validation and historical source identity

The stdlib fixtures use a stub tokenizer and explicitly invented bytes with a
scoped test-only expected-hash value. They run the real tokenizer hash/vocabulary
verifier and real captured-token binding; they do not replace either function.
They test the absent historical path, valid relocated path, missing/aliased/bad
bytes, semantic/re-decode errors and post-forward failure diagnostics. AST checks
bind copied scientific call arguments and the token-binding body to retained
source. These fixtures do not establish real model success or acoustic quality.

The exact v3 [execution freeze](history/executed-v3-621ba90/execution-freeze.json)
remains SHA-256
`abfc1121ed82367eb8d18fb79ad28911a8453b867be49298ac3cfd93e5ebf2f2`.
The [snapshot map](history/executed-v3-621ba90/snapshot.json) retains only changed
original source files and identifies unchanged paths, allowing all 66 original
hashes to be reconstructed over the current tree. The whole original tree also
remains available at immutable executed head
[`621ba90acd2d78976f4dcf07a385cd4e615ac358`](https://github.com/jiying2007/kws-pipeline/tree/621ba90acd2d78976f4dcf07a385cd4e615ac358).
Raw v3 audio, outcomes, receipts and derived result metadata remain unchanged.
The current source freeze describes this inert repair, not a rewritten v3 run.

## Any future continuation

A separately approved continuation would reuse the exact 16 v3 WAVs and original
blind-job hash, retain the completed Qwen-ASR results, and make at most one new
SenseVoice call per clip. It must record the first clip as a second cross-run
attempt and the remaining 15 as their first attempts. It must not regenerate TTS,
rerun Qwen-ASR, invoke Whisper or overwrite v3 evidence. New identity, immutable
input verification, equivalent hard resource/security scope and explicit
activation would be required. No such continuation is active here; both probe
slots and the v3 workflow identity are consumed.

Human review remains pending; human gold, training admission and shipping
approval remain false. D20 FAIL and D90 NOT_RUN are unchanged.
