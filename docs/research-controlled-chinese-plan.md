# Controlled Chinese input projection

`training/research_controlled_chinese_plan.py` turns the existing owner-held
increment metadata and original R1 source recipe into a deterministic,
individually bound input plan. It uses only the standard library and the
existing pure `research_d1_vocab` and `research_d1_recipe.row_identity` helpers.
It does not run the old training recipe, load a model, decode audio, make an ASR
or TTS call, contact an executor, or select a training budget.

## Exact populations

The restored preparation contains these populations. File names belong to its
existing schema; the implementation derives and checks their actual row counts
against the original summary and independent R1 evidence.

| Cohort | Rows | Preserved use |
| --- | ---: | --- |
| Increment non-near train candidates | 140 | Candidate pool for a separately declared weak research recipe |
| Increment development | 57 | Development only; no optimizer mask |
| Increment nearphone quarantine | 14 | Original label evidence retained; no optimizer mask |
| Increment unknown | 44 | Zero loss, including rows that retain unconfirmed text |
| Original R1 train reference | 96 | Original train roster; no new replay authorized |
| Original R1 development reference | 34 | Original development roster; no optimizer mask |
| Original R1 unknown | 4 | Original zero-loss roster |

The plan therefore binds **389 existing audio rows**, with **255 increment
rows** and **134 original R1 rows** kept distinct. One separately retained
consumed TTS terminal has no audio; it stays outside this denominator and may
not be retried by this preparation. The original R1 source recipe also retains
its own excluded-row references outside the 134-row R1 population.

The 96-row historical R1 train roster already includes five `near-negative`
rows. They retain their historical role and label binding. This does not remove
the separate quarantine on the 14 incremental nearphone rows or authorize a
new replay of the historical rows. Likewise, four nearphone-category rows
inside the 57-row incremental development cohort remain development data.

The older 243-row summary, its 152-row train count, 483.76 seconds, and proposed
128 optimizer updates do not define this plan. No optimizer count, batch size,
seed, replay selection, or sampling schedule is inherited from another run.

## Labels and loss are separate

Every entry retains its complete original observation and exact source row
binding. `labels` preserves the original `full_text`, complete `target_ids`,
CTC mask and weight, and per-keyword targets, masks and weights. The existing
2601-output vocabulary is authenticated by its original bytes; the planner
does not extend it or use a requested phrase as an observed transcript.

The pure full-text helper recomputes the target sequence from the already
confirmed text and requires equality with the saved target IDs. R1 native
frontend geometry is checked using the existing 25 ms window, 10 ms stride and
three-frame skip, including additional CTC frames for adjacent repeated target
IDs. It rejects an inconsistent target rather than shortening a transcript,
dropping an unsupported character, or mapping a character to blank.

`scope.candidate_ctc_mask` and `candidate_ctc_weight` describe possible train
pool membership only. They are false/zero for development, incremental
nearphone quarantine and unknown rows. The original nearphone/development
labels remain available for their permitted descriptive or development uses;
zeroing a training mask does not rewrite those labels. An unknown row with
retained text still has no target IDs and no invented absent-keyword labels.

For every row, `optimizer_authorized`, `effective_optimizer_mask` and
`keyword_loss_authorized` are false, with zero effective optimizer weight.
Top-level formal admission, human gold and independent-test claims remain
false. A later finite recipe must explicitly bind this plan and the permitted
cohorts before a controlled training consumer can use the candidate pool.

## Source and R1 bindings

The supplied increment ZIP hash is mandatory. The planner checks the complete
ZIP/member manifest, all original package manifests, and both original and
normalized audio locators against the exact original archive names, archive
hashes and member hashes. Historical local extraction paths are preserved only
as observation data; they are never used to open files on the current machine.

The separately supplied isolation-manifest hash is also mandatory. Saved
isolation files are read and hash-checked, without executing their auditor.
The increment's 211 known rows must match its saved 54 identity comparisons
exactly. The 44 unknown rows were outside that saved comparison scope and are
identified as such. The new input consumer additionally rejects duplicate
source or PCM identities and any train/development alias collision across all
389 projected rows, including zero-loss rows. Neither check claims independent
human speakers or acoustic independence.

The exact original R1 recipe is authenticated against the already audited
capsule member and the actual R1 binding records. All 134 evaluation rows must
retain their order, source/PCM identities, role, and independently saved
label-binding hash. Each normalized WAV is resolved from `recipe.audio_files`
by its exact historical path to one capsule member with matching bytes and
SHA256. A filename suffix alone is insufficient. The retained R1 checkpoint
and actual plan identities come from the R1 evidence; a checkpoint or optimizer
setting mentioned in the older source-data recipe is not used as a new model
initialization or training recipe.

The output is **private metadata**. It includes original labels, source
identities and archive locators, so it must not be committed or uploaded to a
public repository. The current source-archive bytes are not downloaded or
redecoded by this tool. A real consumer must verify the exact archive/member
and PCM bytes on an authorized controlled execution route before inference or
training. Public audio upload and public-executor readiness remain false.

## Prepare and verify

Use explicit paths and pinned hashes from the existing private evidence. The
destination parent must exist. Preparation refuses output inside this source
repository or any supplied input evidence directory, rejects symlinks in the
output path, creates the file with mode `0600`, and never overwrites a plan.

```sh
python -B training/research_controlled_chinese_plan.py prepare \
  --audit "$CONTROLLED_AUDIT_ZIP" \
  --audit-sha256 "$CONTROLLED_AUDIT_SHA256" \
  --isolation-dir "$R1_ISOLATION_DIR" \
  --isolation-manifest-sha256 "$R1_ISOLATION_MANIFEST_SHA256" \
  --r1-recipe "$R1_SOURCE_RECIPE" \
  --capsule-manifest "$R1_CAPSULE_MANIFEST" \
  --plan "$PRIVATE_INPUT_PLAN"

python -B training/research_controlled_chinese_plan.py verify \
  --audit "$CONTROLLED_AUDIT_ZIP" \
  --audit-sha256 "$CONTROLLED_AUDIT_SHA256" \
  --isolation-dir "$R1_ISOLATION_DIR" \
  --isolation-manifest-sha256 "$R1_ISOLATION_MANIFEST_SHA256" \
  --r1-recipe "$R1_SOURCE_RECIPE" \
  --capsule-manifest "$R1_CAPSULE_MANIFEST" \
  --plan "$PRIVATE_INPUT_PLAN"
```

Verification rebuilds the expected plan from the same pinned inputs and
requires equality, including the implementation's source-byte identities. It
does not trust counts or permission flags copied into an arbitrary output
file. Input paths may move without changing the plan's content identities.

The Python API is `build_plan(...)`, followed by `summarize(plan)` for a
text-free aggregate. The aggregate is not a replacement for the private plan.

## Source-only verification

`tests/test_research_controlled_chinese_plan.py` creates fictional metadata
archives and R1 records. It exercises closed member sets, original labels,
CTC geometry, uncertain text, train/development separation, both nearphone
roles, R1 label/PCM/path bindings, consumed no-audio terminals, and immutable
plan creation/readback. No real private row, audio or model is embedded in the
public tests. Validation uses explicit exceptions and remains active under
Python `-O` and `-OO`.

```sh
python -B tests/test_research_controlled_chinese_plan.py
python -O -B tests/test_research_controlled_chinese_plan.py
python -OO -B tests/test_research_controlled_chinese_plan.py
```

This preparation closes the exact input-projection gap. It does not claim a
controlled executor exists, that an optimizer ran, or that a new checkpoint or
acoustic qualification was obtained.
