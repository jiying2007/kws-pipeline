# English KWS research labels — separate fail-closed bridge

Research scope only. This does **not** alter the Mandarin syllable/CTC
mapping, default model, runtime, decoder, or current training approvals.

## Why this bridge exists

The owner-held English FSMN-R0 used a three-class *whole-recording* weak
intent head: background=0, Hey Vosen=1, Hey Mero=2. This is **not** a
Mandarin four-phone CTC head. The existing training label code currently
maps only ni3/hao3/xiao3/wo1. English inputs must not silently be mapped
through that Chinese path.

Original research evidence (all preserved, no reruns in this PR):
- 63/63 Qwen TTS English WAVs: kws-data run 43242278962.
- 189/189 native Qwen3-ASR/Whisper/Parakeet calls: run 43246104418.
- 63-row source-linked three-ASR review: run 43267739514.
- Same-synthetic-voice Log-Mel/DTW: run 43334622072.
- Independent wav2vec2 CTC phone observations, 63/63: run 43340235981.
- A source-linked, *machine-only* three-evidence join from those three
  saved reports produced 63 rows and 14 wake pronunciation review pairs.

The machine outputs do not establish phonetic ground truth,
independent human FAR/FRR, or training approval.

## Frozen input and exact source roles

The training/english_research_label_bridge.py program reads a local,
original three-evidence report with exact raw byte SHA256
89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12.
The program also checks each original ZIP/payload digest, all 63
cell IDs and unique original WAV/PCM digests, the full three-ASR family
roster, the seven original Qwen preset identities and exact 14 wake
variant pairs.

Five synthetic preset families are train-candidate role: Dylan, Eric,
Ono_Anna, Sohee, Vivian. Two remain *exposed development-only*:
Serena and Uncle_Fu. These are synthetic presets, not 7 genuine humans.

## Fixed 3-class weak intent vocabulary (NOT CTC)

| Weak whole-recording intent | ID |
| --- | ---: |
| background | 0 |
| Hey Vosen | 1 |
| Hey Mero | 2 |

The requested text never authenticates the actual spoken sound.
The 63 original rows partition into:
- 30 TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY
  (20 requested wake and 10 ordinary conversations, train-role voices).
- 12 EXPOSED_DEVELOPMENT_NO_TUNE
  (8 requested wake, 4 ordinary, exposed development voices).
- 21 QUARANTINE_AMBIGUOUS_NEARPHONE
  (Wosen, Miro, Rosen; 15 train-role + 6 development-role).

Each row retains training_admitted=false, human_gold=false,
trusted_negative=false and ctc_target_ids=null.
The overall report has admitted_training_rows=0,
optimizer_steps=0. The code **does not start training**.

Use the saved original join in a private folder:

    python3 -I -S -B training/english_research_label_bridge.py \
      --source-review /restricted/english63_existing_join.json \
      --output /restricted/new_weak_diagnostic.json

Both paths should remain private. Output is create-only. The module
rejects source-byte edits, positive/negative flag forgery, split leakage,
duplicate PCM, missing ASR observation, unauthorized nearphone promotion
and source-class swaps. Its synthetic-only regression test runs in
normal, optimized and stripped modes via the existing project CI.

## Remaining gate before a new model

Reliable pronunciation review must establish independent, source-bound
audio/phoneme evidence for the invented names and accented positives.
A machine label is not human truth; a self-reported listening action
alone is not a cryptographic proof either.

A future approved English training pathway needs distinct acoustic
review authority, train/eval leakage controls and an English-specific
token policy if using CTC; it must never use Mandarin's four syllables.
The owner-held R0 remains NOT PROMOTED (5/6 development nearphones
triggered proposals). Genuine independent human FAR/FRR and SSC305
hardware resource/latency tests remain NOT_RUN.
