# English KWS V6 — B11 hard-negative generalization

Status: PREREGISTERED / NOT TRAINED. No model promotion is authorized. B0 remains the installed C11 baseline; B9 is a research parent; B10 is a trained research pair with no measured gain on its small new DEV cohort.

## Evidence
- B9 DEV528 contained two independently machine-screened ordinary originals (Piper groups p6288 and p4731) that produced complete or partial Hey Mero CTC events. They are exposed and may be used only for read-only diagnosis, never selection or a fresh denominator.
- B10 added six Silver ordinary TRAIN originals, all “What time is it?”, from six already-qualified TRAIN groups. Matched control and hard-negative arms each received 192 updates from the same frozen B9 parent.
- B10 DEV540: 8 originals (2 positive, 6 ordinary) × 3 models × 3 gains = 72 forwards. B9, B10 control and B10 hard-negative scores were identical: positives 2/2 at 1.0×/0.75× and 1/2 at 0.5×; ordinary false events 0/6 at all gains.
- Thus B10 did not establish an acoustic advantage. Its new DEV negatives were easy for every arm and do not demonstrate transfer to the earlier false-wake examples.

## P0 — evidence handoff
1. B10 evidence is on protected main at `76e5c1e0a068db503603b386bc897cdec68fee35` (#859). The live `kws-main-terminal` ruleset requires a squash PR, resolved threads and six current passing checks, with zero required approving reviews.
2. Preserve B10 weights/results exactly. Do not rerun B10 training or regenerate its 18 WAVs.
3. B0 remains the shipping comparator.

## P1 — read-only B10 transfer diagnostic
Run frozen B9 parent, B10 matched control and B10 hard-negative weights on the already-saved original DEV528 PCM for known ordinary false-wake IDs 535/537 plus the existing positive controls. This is exposed diagnostic only.
- Authenticate original DEV528 source/screen, B9 model and B10 model archives by artifact ID, run, bytes, SHA256, member CRC and per-file SHA256.
- Preserve roles, source groups, PCM bytes and labels. No recropping, relabeling, TTS/ASR/phone calls or derived-gain denominator inflation.
- Evaluate fixed gains 1.0×, 0.75×, 0.5× without threshold search. Save full per-frame logits, CTC tokens, events, frame/sample clocks and checksums.
- Report source IDs 535 and 537 separately; no FAR/hour claim.
- This cannot select/promote a model because DEV528 is exposed.

## P2 — new B11 source cohort
Do not train until source groups and roles are independently reviewed.
- TRAIN: only original source-qualified TRAIN rows still authorized for B11, plus newly generated originals if required. Build diverse ordinary speech, not repetitions of one sentence. Vary sentence form, duration, rate, stress and phonetic context. Exclude nearphone/keyword-like requests as ordinary negatives.
- Admit only independently reviewed Silver/Weak class-0 rows with complete qualified source groups. UNKNOWN remains excluded.
- Before adding an optimizer, score previously admitted ordinary TRAIN originals with the frozen B9 parent at fixed gains. Count a hard family only for an actual full-word false event at 1.0×. If fewer than two independent families qualify, allow one supplemental batch of at most 24 new diverse TRAIN originals in the six prequalified TRAIN groups. Freeze texts, rate, roles and the admission rule before synthesis. If the combined pool still has fewer than two hard families, stop this B11 training hypothesis. Never use DEV to mine training examples.
- DEV: new speaker identities and original PCM, with multiple ordinary sentence families and canonical Hey Mero positives at fast/slow delivery and low amplitude. Speaker group is the statistical unit.
- Require at least 6 independent TRAIN groups and 4 independent DEV groups. CAL and LOCKED remain untouched; minimum 2 groups each. If gates fail, STOP rather than weaken them.

## P3 — paired single-factor training
- Parent: byte-authenticated frozen B9 research checkpoint; B0 is a frozen inference comparator.
- Arm A: matched continuation control.
- Arm B: same initialization, optimizer, seed, steps, old TRAIN, batch schedule and losses, adding only a source-balanced diverse hard-negative loss (control coefficient 0, challenger 0.08). Freeze the eligible group list and the shared draw trace before either optimizer starts.
- Same optimizer update count, initially bounded to 192 per arm. No DEV/CAL/LOCKED gradients, no gain-copy independence, no threshold tuning.
- Save PT/NPZ/ONNX, optimizer trace, exact input manifest, PCM/WAV hashes, parent SHA, source-group draw trace and STOP receipts.

## P4 — frozen evaluation
Compare B0, B9, B11-A and B11-B on the same untouched new DEV originals and fixed gains. Save full frame logits and per-original token/event traces.
Research advancement requires:
1. B11-B reduces false/extra events on at least two distinct new DEV ordinary source groups versus both B11-A and B9.
2. No new ordinary false event is introduced.
3. Positive source-group recall does not regress against B0 at 1.0×, 0.75× or 0.5×.
4. Benefit is confirmed on a second independent DEV cohort.
5. No selection from exposed DEV528/DEV540.
A tie, zero-error ordinary cohort, fewer than two informative hard-negative groups, or mixed regression is INCONCLUSIVE / NO PROMOTION.

## P5 — product release
After acoustic research qualification:
- Exact candidate PT/ONNX ↔ GCC C11 ↔ Clang C11 ↔ ARMv7/Cortex-A32 logits and event parity.
- Integrate only as a separate candidate pack; preserve B0 rollback.
- Physical SSC305 testing on actual board and final AFE PCM: CPU/RSS/power, event latency, 24-hour stability.
- Human 3–5 m evaluation: FRR ≤5%, continuous natural FAR ≤0.1/hour, word-end-to-callback P95 ≤500 ms, with confidence bounds and recorded conditions.
Synthetic TTS, QEMU or green CI cannot establish real far-field or physical SSC305 qualification.

## Constraints and current state
Public repositories/Actions only; no private repos, paid model calls, Work/Codex quota, scheduled patrols, duplicate audio generation, role reassignment, label rewriting or early CAL/LOCKED access. Each run binds source/audio/code/model/decoder/role/seed/budget hashes.
Terminal states: PASS, FAIL, INCONCLUSIVE, STOP, NOT_RUN.
Current: P0 evidence exists; P1 NOT_RUN; P2 NOT_READY; B11 training NOT_RUN; product promotion NOT_ALLOWED.
