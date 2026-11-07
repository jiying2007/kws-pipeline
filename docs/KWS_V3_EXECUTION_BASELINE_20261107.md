# KWS V3 execution baseline and first implementation — 2026-11-07

This records the user-approved V3 research work, its inherited evidence and the
actual first evaluation implementation. A plan, successful CI, or a label
hash is not model training, independent acoustic success, or SSC305 evidence.

## Current scope and correction

The active V3 lanes are **C1-CTC, pretrained FSMN-CTC, and the existing direct-KWS
prototype**. RNN64 remains a low-cost reference where its evidence is applicable.
Use one feasibility configuration/seed per eligible lane, then confirm at most
two finalists with three declared seeds. These are research choices, not an
unbounded sweep or permission to rerun consumed experiments.

The disabled October 10 architecture-only plan remains historical evidence.
It does NOT override V3, reimpose its old eight-fit/1,000-update plan, require
commercial/shipping files, demote direct-KWS from the user-approved research
matrix, or require every C backend before acoustic screening. Its old results
and source identity must not be rewritten. Every new V3 run still needs an
exact, finite budget, authenticated inputs, split isolation and a real executor.

This corrects the previous revision of this document, which incorrectly made
the old disabled plan a new prerequisite. It does not bypass existing data
verification, source rights, queue ownership or mandatory repository checks.

## Frozen inherited evidence

- Pipeline main inspected: `f097f6667fb08fc16c4c0a4c7a3ee1518421f5f8`.
- Data main inspected: `d4c2a39e5c358bdd0770c8d2f8695898c0f54110`.
- Actual research runtime: existing CPython 3.12.16 and core/TTS/ASR role locks.
  A different interpreter may run declared metadata-only regression tests but
  cannot silently stand in for the pinned training/runtime environment.
- Balanced64 C1 augmentation is completed: 96 eligible train rows, 34 development
  rows, four uncertain zero-loss rows, 2,000 optimizer updates, 8,000 training
  draws and 402 final observations. Reuse its actual checkpoint and observations.
- The fixed old/new 3:1 sampler follow-up completed and was not promoted. Preserve
  its negative decision; do not replay it or convert it into a quota sweep.
- Saved C1 posterior diagnostics do not prove all K1 misses are decoder-only.
- The 73 natural windows retain 219 observations (216 quality-usable), zero
  frozen full-text quorum and zero accepted natural-negative seconds. They stay
  development. Their existing TTS/ASR observations must not be repeated.
- Historical D2 unknown calls stay unknown. Never automatically replay them.
- The Local38 archive in data PR #70 is separate historical evidence, not the
  balanced64 checkpoint. Its retention failure and runtime difference affect
  that archive's use, not every other qualified V3 lane.

Exact existing reports: [balanced64 posterior](KWS_C1_POSTERIOR_20261106.md),
[3:1 decision](KWS_C1_ALLOCATION_20261106.md),
[C1 source allocation identity](research-c1-source-allocation.md), and the
`kws-data/docs/NATURAL_ASR_EVIDENCE_20261107.md` record.

## Implemented in this change: per-keyword known-label scoring

`eval/score_keyword_presence.py` consumes explicit **development-only** labels
for each of the two keywords: `present`, `absent`, or `unknown`. One word can be
scored while the other remains unknown. It accepts named saved event streams
from C1, FSMN, direct-KWS or another declared reference without imposing equal
numeric thresholds or pretending their context is already comparable.

It never derives labels from ASR non-detection, requested TTS text, forced
alignment or vocabulary membership. It does not modify the natural full-text
consumer, create a training manifest, establish independent truth, or promote a
model. PCM/label hashes are recorded declarations, not authenticated evidence.

The counts report known-positive hits/misses, known-negative clips/events,
unknown clips/events and duplicate events separately. Repeated PCM recordings
must be evaluated as separate conditions, not counted twice in one denominator.
All-unknown inputs yield `UNSCORED`, not a zero-error PASS. There are no natural
FAR/FRR, event timing, latency or automatic winner claims. Natural-negative
exposure remains zero even for a declared absent-keyword clip.

Reference JSONL has exactly these fields (digest strings are actual input
identities, not the placeholder strings shown here):

```json
{"recording":"sample-a","duration_s":2.0,"pcm_sha256":"<actual PCM SHA-256>","label_evidence_sha256":"<actual label evidence SHA-256>","split":"development","keyword_states":{"1":"present","2":"unknown"}}
```

The existing event format is retained:

```json
{"recording":"sample-a","keyword_id":1,"time_s":1.0,"confidence":0.8}
```

```sh
python -I -S -B eval/score_keyword_presence.py \
  --references /evidence/development-keywords.jsonl \
  --events c1=/evidence/c1-events.jsonl \
  --events fsmn=/evidence/fsmn-events.jsonl \
  --events direct=/evidence/direct-events.jsonl \
  --output /evidence/new-keyword-presence-report.json
```

Use only streams actually produced by the named model and independently verify
their model/source/audio/context identities. The scorer records input-file
hashes and refuses output replacement. It cannot authenticate upstream labels
or certify models by itself. Missing arms should be omitted, never fabricated.

Twenty-four new fictional-metadata regressions are integrated into the existing
`test_existing_boundaries.py` suite, already executed by required CI and the
existing offline quality-guard workflow in normal, `-O` and `-OO` modes. There
is no added workflow, schedule, model fixture, ASR/TTS call or training job.

## Execution order and dependency scope

1. Restore and verify the exact retained model/data assets available to the
   selected executor. Existing qualified data may support research while a new
   label protocol is being developed; the entire M1 audit is not a blanket
   prerequisite for unrelated already-admitted inputs.
2. Reconcile PID/lock/unknown status only for the host/campaign being resumed.
   An inaccessible Dot host does not block source fixes or an independently
   budgeted, isolated public-hosted task with verified inputs. It does prohibit
   guessing that a Dot job is idle or stealing its task/lock.
3. Run one genuinely new, preregistered experiment on the best-supported eligible
   lane. Preserve old/new cohort and per-keyword outcomes. Conditional direct
   training still requires valid window/event supervision and class coverage.
4. Calibrate each viable system on calibration data; compare valid continuous
   events under common external context and scoring rules. This new clip
   diagnostic does not replace that continuous evaluator.
5. Confirm at most two finalists, then export/quantize the supported candidate.
   Complete clean-directory inference replay and explicitly declared retraining
   reproduction before calling the research package reproducible.
6. Independent human/final-AFE and physical SSC305 evidence stays `NOT_RUN`
   until actually measured; absence blocks only the relevant verification claim.

Use public repositories, existing authorized/free compute and the shared heavy
slot on any shared host. No Private repositories, paid model APIs, Work/Codex
shared quota, new hourly/Scheduled polling, or duplicate completed generation.
Persist terminal evidence before advancing. Only an actual worker/job is
`in_progress`; an immutable declaration without a worker is `queued`.

## Completion accounting

This change implements a scoring capability and regression tests. It does not
claim M0 asset recovery, blinded M1 label validation, new training, a selected R1,
canonical-runtime loading of Local38, natural FAR, or physical-board success.
Those require their own real terminal receipts and retained outputs.
