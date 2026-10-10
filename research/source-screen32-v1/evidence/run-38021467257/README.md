# Completed machine screen · 2026-10-10

**Automated continuation succeeded. This batch is NOT training-ready.**
SenseVoice completed 16 new one-attempt decodes of the existing 16 WAVs. With the
unchanged original Qwen-ASR results, **13/16 normalized transcripts agree**:
**8/16 pairs match the intended phrase**, **5/16 agree on a different phrase**,
and **3/16 disagree**. Every clip still needs independent human actual-word review.

[Continuation run 38021467257](https://github.com/jiying2007/kws-pipeline/actions/runs/38021467257)
executed immutable head
[`4835c51e7d42e6b2cde8465327db136d2ff69c18`](https://github.com/jiying2007/kws-pipeline/tree/4835c51e7d42e6b2cde8465327db136d2ff69c18).
TTS and Qwen-ASR come from the separately retained
[original v3 run 38018029787](../run-38018029787/README.md), whose overall failure
and first SenseVoice failure remain unchanged.

## Exact attempt accounting

- Original TTS: 16 Qwen clips, each generated once; all 16 planned FireRed cells
  remain NOT_RUN. This continuation made **zero new TTS calls**.
- Original Qwen-ASR: 16 successful decodes, reused byte-for-byte; **zero new calls**.
- New SenseVoice: **16 successful decodes, one new attempt per clip**. Opaque
  clip-000001 has two cumulative consumed attempts (old failure + this success);
  the other 15 have one each. Total SenseVoice consumed attempts across both runs:
  **17**. These counts retain failed attempts rather than relabelling them fresh.
- No new probe or Whisper execution. No automatic restart, audio repair,
  regeneration, selection of only successful clips, or model tuning occurred.
- Human actual text is null/PENDING for all 16; human gold, training admission and
  shipping approval are false. D20 remains FAIL; D90 remains NOT_RUN.

## Full machine comparison

The two raw decoder outputs were frozen before joining intended text. The join
uses original TTS cell → exact WAV hash → opaque blind ID → original Qwen and new
SenseVoice outcomes. Both recognizers consumed the same hash-bound 16 kHz PCM16
WAVs; intended text, voice design and prior ASR answers were not in the SenseVoice
container's audio-only mount.

The existing
[`quality_gates.normalized_actual`](../../../experiment_quality_guards/quality_gates.py)
function removes whitespace and Unicode punctuation only. Its frozen SHA-256 is
`e9a0a838e134746bbd0fdd60aeff48ca9ad2963b721e3ea45bd0d8b45690b15d`.
No homophone folding, script conversion, transcript repair or edit-distance
threshold is used: 窝, 屋 and 吴 remain different characters.

| Cell / retained audio | Design | Intended phrase | Raw Qwen-ASR | Raw SenseVoice | Machine observation | Human actual words |
|---|---|---|---|---|---|---|
| [001](../run-38018029787/generation/tts/screen32-001.wav) | design-a | 你好小窝 | 你好，小窝。 | 你好小窝 | both match intent | PENDING |
| [002](../run-38018029787/generation/tts/screen32-002.wav) | design-a | 小窝小窝 | 小窝，小窝。 | 小窝小窝 | both match intent | PENDING |
| [003](../run-38018029787/generation/tts/screen32-003.wav) | design-a | 你好小屋 | 你好，小屋。 | 你好小屋 | both match intent | PENDING |
| [004](../run-38018029787/generation/tts/screen32-004.wav) | design-a | 小屋小屋 | 小吴，小吴。 | 小吴小吴 | agree; differ from intent | PENDING |
| [005](../run-38018029787/generation/tts/screen32-005.wav) | design-a | 小窝小屋 | 小屋，小屋。 | 小屋小屋 | agree; differ from intent | PENDING |
| [006](../run-38018029787/generation/tts/screen32-006.wav) | design-a | 小屋小窝 | 小窝，小窝。 | 小屋小屋 | disagree; unadjudicated | PENDING |
| [007](../run-38018029787/generation/tts/screen32-007.wav) | design-a | 你好你好 | 你好，你好。 | 你好你好 | both match intent | PENDING |
| [008](../run-38018029787/generation/tts/screen32-008.wav) | design-a | 小窝 | 小屋。 | 小屋 | agree; differ from intent | PENDING |
| [009](../run-38018029787/generation/tts/screen32-009.wav) | design-b | 你好小窝 | 你好，小窝。 | 你好小屋 | disagree; unadjudicated | PENDING |
| [010](../run-38018029787/generation/tts/screen32-010.wav) | design-b | 小窝小窝 | 小窝，小窝。 | 小窝小窝 | both match intent | PENDING |
| [011](../run-38018029787/generation/tts/screen32-011.wav) | design-b | 你好小屋 | 你好，小吴。 | 你好小吴 | agree; differ from intent | PENDING |
| [012](../run-38018029787/generation/tts/screen32-012.wav) | design-b | 小屋小屋 | 小屋，小屋。 | 小屋小屋 | both match intent | PENDING |
| [013](../run-38018029787/generation/tts/screen32-013.wav) | design-b | 小窝小屋 | 小窝，小屋。 | 小窝小屋 | both match intent | PENDING |
| [014](../run-38018029787/generation/tts/screen32-014.wav) | design-b | 小屋小窝 | 小屋，小屋。 | 小屋小屋 | agree; differ from intent | PENDING |
| [015](../run-38018029787/generation/tts/screen32-015.wav) | design-b | 你好你好 | 你好，你好。 | 你好你好 | both match intent | PENDING |
| [016](../run-38018029787/generation/tts/screen32-016.wav) | design-b | 小窝 | 小吴。 | 小屋 | disagree; unadjudicated | PENDING |

All lexical denominators are **16**, including all eight cells lacking dual
intended-text agreement. Qwen alone matches 9/16 intended phrases; SenseVoice
alone matches 8/16. Each design has 4/8 dual intended-text matches. The five shared
deviations are cells 004, 005, 008, 011 and 014. The three disagreements are cells
006, 009 and 016 (opaque IDs 004, 007 and 016).

These are machine lexical observations, not human labels. A shared deviation
cannot determine whether the synthesis, either recognizer, or both recognizers
are wrong. Even a dual intended-text match does not establish acoustic truth or
complete speech. No source, voice, target or negative clip is admitted to training
from this table. Independent listening review of all 16 actual utterances is
still required; do not prefill those labels from these transcripts.

## Dispute field semantics

The original [frozen-disputes.json](asr/frozen-disputes.json) is retained unchanged.
Its `unresolved_audio_ids: []` means no missing/failed/flagged primary decoder
output under the frozen schema. It **does not mean every semantic question is
resolved**. Its three `whisper_audio_ids` are disabled dispute candidates, not
Whisper results or authorization to run Whisper. Those **three textual
disagreements remain unadjudicated**, and human truth remains pending for all 16.

All 16 new SenseVoice rows reported complete decoder status, no decoder flags,
verified raw-token re-decode and metadata boundaries, with the diagnostic showing
`model.inference` returned and the adapter completed. Acoustic completeness remains
**UNKNOWN**. The successful corrected route does not retroactively establish the
first cause of the original exception-class-only ValueError.

## Retained execution and resource evidence

Both new setup and SenseVoice containers completed with exit 0, no reported OOM
and completed cleanup. The retained inference kernel readback reports 12 GiB
memory.max, zero swap, four-CPU quota, PID limit 256, unprivileged UID, no
capabilities, no-new-privileges, seccomp, private cgroup, read-only root and only
loopback networking. These are enforced limits/configuration observations, **not
measured peak memory or CPU usage**.

Setup verified 3,340,966,447 input bytes in 148.98 s; the SenseVoice container took
57.39 s; the complete host phase took 214.95 s. The exact official Python image
completed with a 47,711,592 B default-interface receive observation, below its
128 MiB reservation. That observation includes protocol/background traffic and is
not an exact image transfer count or hard Docker-daemon cutoff. No weights,
installed runtime or credentials are included in the public artifact.

## Durable bytes and source identity

All **169 original artifact members** (1,524,376 bytes unpacked) are retained
byte-for-byte under [asr/](asr/). The original
[artifact freeze](asr/artifact-freeze.json) binds its other 168 members and has SHA-256
`38bb5d17795bdeeee2ed6a94b05d9b6275ac1a27ed118914274151b7903ac9f7`.
[Provenance](provenance.json) binds the archive ID, 370,705-byte ZIP digest,
run/job/head and every member. The ZIP wrapper itself is not retained or recreated.

The 79 copied Qwen files (806,896 bytes) are intentionally present in this complete
artifact and verified byte-identical to original v3. Their source-run identity is
recorded in [continuation provenance](asr/continuation-provenance.json), and the
[cross-run ledger](asr/continuation-ledger.json) distinguishes prior and new
consumed attempts. Original native 24 kHz float and derived 16 kHz PCM16 audio
remain at their old paths without duplicate generation; the
[derived joint summary](result-summary.json) binds both audio/PCM hashes and every
comparison row.

The executed [74-file source freeze](../../execution-freeze.json) is unchanged,
SHA-256 `242565d3dec5174c57572eafef68f5ea49a2ddd42d213b0debdd7f7b44d33079`.
The [continuation preparation document](../../SENSE_CONTINUATION_2026-10-10.md),
[source README](../../README.md), plan and readiness remain historical frozen
pre-execution snapshots. This dated result page is the current status and sits
outside that freeze. All main admission files remain false and this retention
adds no active workflow. The approved continuation is now consumed; this record
permits no further model run or training.
