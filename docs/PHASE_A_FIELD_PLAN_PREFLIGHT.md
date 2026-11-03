# Phase-A field-capture planning coverage (source-only)

**Objective:** turn the current *private* `kws-data` field collection pilot into
an actionable, **policy-derived acquisition plan** for Xiaowo KWS without
mistaking scheduled prompts for real speech, human identity, consent or final
AFE/SSC305 qualification.

Entry point: `tools/phase_a_field_plan_preflight.py`. Its only inputs are
the existing `research/field-collection-pilot-20261102/plan.py` **v1 planned**
artifact(s), exact independently frozen SHA-256 for each, and the CURRENT
`commercial/real-human-qualification.policy.json`. The caller-provided policy
must match the currently checked-out repository policy **exactly** after
canonical parsing: keeping the same policy ID while lowering quotas is rejected.
It executes no microphone, TTS, ASR, training, scoring or hardware command.

## Concrete uncovered product requirements

The current immutable Phase-A policy requires:
- ≥40 independent real Mandarin speakers, ≥2 real sessions/speaker.
- ≥1,200 *actual* expected wake events: ≥600 per **你好小窝** and
  **小窝小窝**. Each distance/orientation/playback/double-talk/robot-motion
  slice also has its own required actual support.
- ≥24 **valid post-final-AFE** negative audio hours, including the policy's
  household, speech-confusion, playback and robot-motion exposures.
- FRR ≤5% with 95% upper bound ≤6.5%; FAR ≤0.1/hour with 95% upper
  bound ≤0.13/hour; p95 actual word-end-to-event latency ≤500 ms.
- Future same deployment/AFE/board Phase B and production Phase C.

The v1 field pilot plans **37 prompt tasks per declared session**:
24 wake-intent slots (two keywords × three distances × front/rear × two
repeats), 12 near-homophone confusables (six strings × two distances), and
one continuous-background instruction. It has **no verified audio**, no
acoustic word-end annotation, no verified critical tag such as `playback`,
`double_talk` or `robot_motion`, and no measured negative duration.

At 40 speakers × 2 sessions, the *planned* 80 sessions imply 2,960 task
slots and 960 planned wake requests per word. This clears the raw
**planned-count** minimum, not the actual event, person-independence or
audio-exposure gates. Even a "rear" azimuth is only a **candidate orientation**
until independent capture/annotation supplies the policy's required tag.
All tag-dependent qualification support stays 0 here.

## Important scale boundary: sharding, not bypassing source checks

The original `kws-data` field-pilot source reader has an explicit
**1-MiB per-input-manifest limit**. A full 80-session plan using the
current 37-slot recipe is roughly 1.38 MB in compact UTF-8 JSON, so a
single-file full Phase-A plan would exceed that upstream reader.

**Do not raise the old source budget, weaken its tests, or silently truncate.**
Split a future authorized field campaign into independently frozen smaller
plans, e.g. 40 session-1 records and 40 session-2 records for the same
40 pseudonymous speakers. Keep the original speaker and source-family IDs
consistent, but use distinct real session IDs. Each shard remains under
the v1 manifest byte cap and must pass its own `kws-data` source/rights/audio
verification. The new consumer audits up to eight such shards **together**:
a duplicate session, cross-shard train/dev/sealed family leakage, a single
speaker silently changing source family between sessions, or a repeated source
slot is rejected. If multiple declared speakers share one source family, the
planner reports **both counts** and keeps the source-family quota short; it
cannot authenticate them as independent persons. Each supplied freeze digest must be independently
recorded at plan creation, not generated from an edited input later.

The aggregate digest is a derived comparison key; it is **not** a new
source-authorized/training/qualification root.

## Example operator-only commands

Generate each local plan in an authorized, network-isolated restricted
environment using the already merged `kws-data` pilot:

```bash
python3 -B research/field-collection-pilot-20261102/plan.py plan \
  --participants /restricted/cohort-a-participants.json \
  --output /restricted/field-plan-a.json

python3 -B research/field-collection-pilot-20261102/plan.py plan \
  --participants /restricted/cohort-b-participants.json \
  --output /restricted/field-plan-b.json
```

Record both emitted `plan_sha256` values in the access-controlled planning
record, then on a pinned `kws-pipeline` commit:

```bash
python3 tools/phase_a_field_plan_preflight.py \
  --field-plan /restricted/field-plan-a.json \
  --expected-plan-sha256 REPLACE_WITH_ACTUALLY_FROZEN_A_HASH \
  --field-plan /restricted/field-plan-b.json \
  --expected-plan-sha256 REPLACE_WITH_ACTUALLY_FROZEN_B_HASH \
  --phase-a-policy commercial/real-human-qualification.policy.json \
  --output /restricted/reports/new-field-plan-coverage.json
```

Hashes shown are placeholders and **must fail** until real pinned plans
exist. The tool opens original input files read-only, rejects oversized
individual plan files, creates a new private report with mode `0600`,
and cannot write a report in the public repository or overwrite a result.

The report includes aggregate declared role/session counts, **requested**
keyword/distance support, tag-dependent **missing** support, source-family
split constraints, remaining policy quota counts and the entire current
negative exposure-hour deficit. It intentionally excludes pseudonymous
speaker and session IDs from the output.

## What must still happen before "done"

1. Actual legitimate consenting real people, genuine distinct sessions,
   final microphone/enclosure/AFE capture, appropriate authorization and
   independently reviewed spoken text/keyword event boundaries.
2. Strong byte-level local WAV/PCM checks through the original `kws-data`
   field pilot and existing KWS real-corpus intake/sealing. The plan is
   only the acquisition worksheet; it does not fabricate `recordings`.
3. An independently fresh sealed cohort, actual Phase-A validated positive
   support and ≥24 valid negative hours with policy-specific critical slices.
4. Real measured FRR/FAR confidence bounds and word-end latency on the
   frozen deployment/AFE, followed by physical SSC305 Phase B and terminal
   Phase C. Current shipping remains unapproved.

Every report from this tool has `verified_real_speakers=0`,
`verified_actual_keyword_events=0`, `verified_negative_audio_hours=0`,
`training_admitted=false`, `qualification_allowed=false` and
`shipping_approved=false`. Successful software tests cannot remove any of
these blockers. This preflight is not a substitute for product data collection.
