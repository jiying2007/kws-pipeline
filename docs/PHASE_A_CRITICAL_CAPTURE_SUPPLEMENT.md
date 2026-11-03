# Supplemental real-human Phase-A capture requests (private field worksheet)

The existing upstream \`kws-data\` field pilot prepares 37 fixed tasks per
declared session. Those are valuable for wake words, near-phonetic confusions,
front/rear orientation and a general background prompt, but **do not directly
schedule** Phase-A critical playback, double-talk, robot-motion positive
examples or 24 hours of measured negative exposure.

This is a **collection worksheet**, not a new recognizer, training dataset or
qualification record. It fills that practical acquisition gap without
modifying the immutable upstream field-pilot schema.

## Generate a bounded, policy-derived private acquisition schedule

Start with an independently frozen set of sealed-evaluation field-pilot v1
plans as required by [Phase-A field-plan preflight](PHASE_A_FIELD_PLAN_PREFLIGHT.md).
Each upstream plan must be within its existing 1-MiB original input limit.
The same pseudonymous speaker may appear in different real sessions but
the same speaker and source family cannot cross roles or silently change.

\`\`\`bash
python3 tools/plan_phase_a_capture_supplement.py \
  --field-plan /restricted/field-plan-a.json \
  --expected-plan-sha256 ACTUAL_EXTERNALLY_FROZEN_SHA_OF_A \
  --field-plan /restricted/field-plan-b.json \
  --expected-plan-sha256 ACTUAL_EXTERNALLY_FROZEN_SHA_OF_B \
  --phase-a-policy commercial/real-human-qualification.policy.json \
  --output /restricted/reports/new-phase-a-capture-requests.json
\`\`\`

The output must be a *new file in a private pre-existing directory*. The
script never overwrites one, writes it mode 0600 and refuses public-repository
paths. Source hashes are independently captured **before** collection.
The policy must canonically equal the checked-out source file, not a
caller-authored weaker copy bearing the same ID.

### Required declared cohort

At least 40 unique **declared sealed-evaluation speaker IDs**, at least two
different declared sessions per person, and at least 40 distinct source
families. A synthetic voice roster or 80 sessions all from one person
cannot satisfy this check. This is still only a declaration and does not
authenticate physical persons, their consent, or independence.

Training and development speaker/session rows may be present in original
plans, but are **not** scheduled as sealed qualification audio here. The
aggregate source preflight rejects cross-shard source-family/speaker role
leakage before any worksheet is written.

## Exact supplemental task design

For each distinct declared sealed session, request BOTH exact complete wake
words in four independently annotated acoustic scenes:

| Requested scene | Geometry | Per keyword requirement |
| --- | --- | --- |
| Rear | 5 m / 180 degrees | >=100 actual events |
| Playback | 5 m / front | >=100 actual events |
| Double-talk | 3 m / front | >=80 actual events |
| Robot motion | 3 m / front | >=80 actual events |

Repeats per session/scene/keyword are determined from the *actual current
commercial Phase-A policy* with a 10% planning margin, bounded to 4 repeats.
A 40-person ×2-session pilot requests two repetitions per keyword in each
scene, or **1,280 additional positive-intent requests**. All tags are
\`tags_requested\`, **never verified acoustic tags**. An actual positive
must contain a truly spoken complete wake phrase, verified with human
annotations and independent source/session evidence. Playback of a generated
voice does not automatically count as a real-human positive.

For independent continuous negative exposure, the current commercial
policy requests >=24 hours total with critical strata household 8 h,
speech-confusion 4 h, playback 4 h and robot-motion 4 h.
To avoid double counting requested blocks, this worksheet assigns the
remaining 4-hour global difference to household scenes, then adds 10%
planning margin and divides by the number of sealed sessions.
Each private source WAV is bounded to a **10-minute request**.

For 80 sessions, the exact per-session plan is:

| Negative scene | Requested time/session | Aggregate requested |
| --- | ---: | ---: |
| Household | 594 sec | 13.2 hours |
| Speech confusion | 198 sec | 4.4 hours |
| Playback | 198 sec | 4.4 hours |
| Robot motion | 198 sec | 4.4 hours |
| **Total** | **1,188 sec** | **26.4 hours** |

These are **requested recording durations**, not proof of captured or
usable post-AFE audio. Hidden complete wake phrases must be annotated and
excluded from negative exposure as required by the qualification scorer.
Real post-AFE valid duration may be lower than requested; the 10% margin is
not a quality guarantee and does not change the 24-hour product gate.

At 80 sealed sessions the worksheet has **1,600 capture requests**:
1,280 wake-scene requests and 320 negative-scene requests. Each request
has its original pseudonymous speaker/session/source family and room,
device, AFE profile declaration, deterministic \`request_id\`,
\`prompt_requested\`, \`tags_requested\` and, for negative blocks,
\`duration_s_requested\`.

## After physical recording

The audio operator must use the real microphones/enclosure and final AFE,
record the actual sample clocks/route, rights/consent and session provenance,
and independently annotate actual speech, critical tags, event boundaries
and negative exposure. Keep raw and post-AFE WAV private. Actual acquisition
may use a capture tool compatible with the current real-human corpus
contract; this worksheet does not execute it.

The original \`kws-data\` v1 37 tasks per session are **unchanged**. These
additional planned task IDs are NOT accepted as historical v1 \`slot_id\`
values; they must be ingested through a separately reviewed real-human
corpus intake, without forging v1 receipts.

Phase A still requires the original
\`commercial/real-human-qualification.policy.json\` real-human validation
and scoring, fresh sealed exposure and independent confidence bounds.
Physical SSC305 Phase B and terminal Phase C remain mandatory.

**Every worksheet unconditionally reports zero observed real speakers,
zero actual wake events, zero observed final-AFE negative hours, and all
training/qualification/shipping flags false.** This planning step does not
authorize fresh holdout consumption, training or model promotion.
