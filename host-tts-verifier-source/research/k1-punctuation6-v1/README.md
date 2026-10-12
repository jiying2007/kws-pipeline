> Phase-two candidate: see `IMPLEMENTATION.md` for the current disabled TTS/workflow implementation. The remainder records the original design proposal; its original byte snapshot is preserved separately.

# K1 punctuation supplement: disabled proposal

Nothing in this package downloads, loads or executes TTS/ASR. There is no workflow,
remote writer, source-root registration, audio output or execution release.
`entry.py` always refuses. The Python contract is stdlib-only and checks metadata;
passing its tests is not evidence of real inference, durable storage or admission.

## Fixed experiment

Dylan, Uncle_Fu and Vivian each receive exactly `你好小窝` and `你好，小窝` in that
order. The UTF-8 strings are not interchangeable with ASCII comma variants.
Paired seeds are respectively 1012001, 1012002 and 1012003, with no seed search.
There are six predeclared TTS cells and eighteen predeclared ASR slots. Each
eligible unique new waveform gets one Qwen, one SenseVoice and one Whisper call.
All results, failures, quarantines and unstarted rows remain in the denominators.
No new identification, retries, replacement seeds, best-of, warmup generation or
extra audio probes can replenish an attempted slot.

The generator is pinned Qwen3-TTS-12Hz-0.6B-CustomVoice revision
85e237c12c027371202489a0ec509ded67b5e4b5. Its API clears instruction on 0.6B;
there is no supported phoneme-control interface in the reviewed entry. This is
exploratory punctuation conditioning, with no guaranteed pronunciation gain,
statistical inference, generalization or product qualification claim.

## Review sequence and one-use boundary

1. Freeze plan, exact reviewed source closure, runtime/model locks, resources,
   historical duplicate-comparison inventory and publication destinations.
2. Separately release one create-event TTS ref in public `jiying2007/kws-data`.
   Require run_attempt=1, exact source commit and prior-claim absence. Each cell's
   exclusive started record must be file-and-directory fsynced before its sole
   top-level generate_custom_voice call. Local exclusive files alone do not
   establish global one-use authority across fresh runners.
3. Stop without retry on failure, timeout, resource violation, missing natural
   EOS or reaching token/output/duration limits. Keep raw float24k and derived
   mono PCM16/16k WAVs, their file/value hashes and truthful terminal records.
4. Before ASR, apply the frozen identical rule to all six cells. Any same-batch
   WAV/PCM collision quarantines every member symmetrically. Historical
   consumed/selection-audio collisions quarantine matching cells. Successful
   signal-quality-flagged cells also quarantine. No regenerated replacement and
   no reused old transcription. These rules are independent of future ASR text.
5. Bind all six outcomes and screening decisions, but only eligible unique
   waveforms in the blind ASR inventory. Quarantined slots have attempts=0 and
   explicit not_run_duplicate/not_run_quality reasons; all eighteen remain.
6. Independently review the actual TTS run/source/audio readback, freeze the new
   WAV/PCM/frame inventory, then separately release one pipeline ASR create-event
   ref. All native policies/configurations and resource/evidence closures must
   be bound before any audio forward. No eligible cells means no ASR run.
7. Apply existing conservative three-ASR/native/source/rights/history admission.
   Agreement is not majority vote; 沃/屋/我 cannot be rewritten as 窝. Human gold
   and acoustic verbatim completeness are not invented from machine consensus.

## Resource and transport boundaries

The TTS lock contains 2,934,954,359 controlled input bytes and a 4GiB response-body
cap, not a total-wire cap. Existing bounded setup is 1200s and generation 1200s;
CPU limits are per-process 2400/3600s, RSS thresholds sampled at 4/7GiB, affinity
four cores. Each cooperative generation alarm is 300s, maximum waveform 12s;
reaching the duration/token bound blocks continuation. Six clips imply at most
72s, and eighteen ASR inputs at most 216s. These are limits, not duration promises.
The original 1MiB ASR audio transfer reservation is insufficient for the new
worst case. Six derived PCM WAVs consume about 2.30MB, but ASR must download the
complete data ZIP containing native sources and receipts as well. The new
proposal reserves 16MiB of data response bodies, replacing the old 1MiB item;
the resulting mixed-scope reservation is 6,056,574,976B within the unchanged
6GiB acquisition cap, leaving 385,875,968B. This is not an all-wire hard cap.

Public data baseline observed read-only on 2026-10-12:
`jiying2007/kws-data` main 3c49ece03650c68ee81c2a943df27c2c34a3f128,
tree 64603f78decf1a39c52ee5bcb49bcb4cda97038c; public, Actions active, current
connection has push permission. Existing eight workflows are integrity checks;
the future TTS workflow and same-repository write have not been created/tested.

Put TTS source and its one-use workflow in kws-data. Its short-lived repository-
scoped GITHUB_TOKEN may publish only there, using a reviewed create-only output
ref. No persistent credential or cross-repository token is introduced. Publish
six derived WAVs, corresponding native float WAVs and allowlisted generation,
source, termination and resource receipts in kws-data. Any ZIP/chunk packaging
must retain both archive and individual member hashes. Read the complete <=12MiB
ZIP as raw-content chunks, plus <=16KiB manifest, from one immutable data commit;
the 16MiB body reservation is not inferred from the smaller derived audio total.
The reader permits at most 49 GETs: one manifest and up to 48 chunks, once per
fixed path, without retries or redirects. Both manifest and chunk bodies count;
no additional data-body request is implied by the unused reservation slack.
No implicit base64/API retrieval path is permitted without a new accounting
review. Each raw/derived pair is bound to the same single generation receipt
with fixed 2:3 resampling and float/WAV/PCM/frame identities, with no gain or trim.
Pipeline ASR rechecks
every WAV's original bytes, PCM, frame geometry and source binding, not just ZIP.
The writer/readback proposal retains <=12MiB compressed archive, <=48 chunks of
256KiB and <=16KiB outer manifest. Do not enlarge the existing pipeline cap.
The native audio bound is <9.22MB for six raw and six derived WAV payloads plus
small headers; the exact receipts/member-size closure still needs review.

Pipeline ASR reads the immutable verified data commit and publishes only
audio-free ASR/resource/source evidence into `research/k1-punctuation6-v1/evidence/`.
Never place WAVs, WAV-bearing ZIPs, weights, private human-history bodies,
credentials, environments or raw logs in pipeline chunks or logs. No Actions
artifact storage is used. A future real writer success must be read back with
the same actual once-only generation identity; failure never authorizes another
generation to test transport.

References:
- https://github.com/jiying2007/kws-data/blob/3c49ece03650c68ee81c2a943df27c2c34a3f128/docs/RIGHTS.md
- https://github.com/jiying2007/kws-data/actions/runs/37915077642
- https://docs.github.com/en/actions/concepts/security/github_token
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#create
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/README.md

The official model card identifies Apache-2.0. It is not a blanket commercial
license for generated audio. Current publication is authorized research and
development; commercial output rights remain not-established.

## Training remains disabled

The new coverage proposal needs at least one actually admitted new K1 plus one
actually admitted member of each old K2/repeat-nihao/suffix-kw2 group. The six old
machine candidates are merely candidates and train-only; real source/rights/
history/native admission and registered roots remain necessary. Include all
legally admitted candidates; no ASR-dependent selection.

Old twelve-cell coverage remains insufficient and its thirty-six slots exhausted.
Neither old coverage nor the old frozen training recipe changes bytes. If new
coverage passes, create a separately reviewed derived recipe binding that changes
only data/coverage/source/analytic workload and resource proposal. Preserve the
learning settings including seed9101101, single fit/1000 updates, objective,
model, thresholds, no augmentation and last-only checkpoint. The old 24.24s
audio/static workload bound cannot describe new audio. At most twelve training
rows still allow N+24 <=36 additional neural readback passes, but new geometry
and resources must be recalculated. Training has its own future execution review.

Run metadata-only tests with `python -B test_contract.py`, and repeat with `-O`
and `-OO`. The source-reuse lock lists reviewed landmarks, not a complete runnable
dependency closure; no historical identity-bearing runner/release is reused.
