# Registered but disabled ASR workflow

The reviewed workflow is now present at
`.github/workflows/tts-pilot-asr-20261014-v1.yml`, byte-identical to workflow.yml
in this folder. Its sole push trigger is the dedicated research execution branch;
main does not match and registration does not consume an execution run.
`activation.json` stays disabled, with null input/source/workflow/numeric bindings.
Disabled or incomplete evidence is rejected before any network or inference. No
run, artifact, source SHA or success receipt has been fabricated.

## Actual evidence required before root enables

1. Bind two engines × two phase records. A real execution has state
   `recorded_run`, plus run_id, head_sha, head_branch, workflow_path, run_attempt=1,
   event=push, actual terminal conclusion, probe lane_outcome, artifact_id, artifact_name, zip_sha256,
   and exact ZIP members. Each member is its real archive path mapped to
   `{sha256, bytes, restore_name}`. The only restored files are corresponding
   engine receipts, planned preset native WAVs and host-receipt.json. Other
   individually reviewed JSON/claim members have restore_name=null: their actual
   bytes are hash-verified and then discarded, never used for model input.
2. A failed probe binds its actual terminal run, actual artifact and failed host receipt
   with matching engine/source/campaign/status. Its remaining phase must be
   exactly `{engine, phase:"remaining", state:"not_run",
   reason:"verified_probe_failed"}`. It has no fictional run/artifact IDs, no
   download and no generation. The failed lane's first design slot is labelled
   upstream pipeline failure, the other 15 not_run. This is not a claim that a
   synthesis attempt occurred. An available other lane can proceed independently.
   Workflow conclusion and lane_outcome are separate: a shared matrix run can
   fail overall while one lane succeeded. Only the bound per-lane host status
   determines whether that lane is blocked.
3. A successful probe requires an actual remaining-run artifact. Remaining may
   have concluded failure and preserved partial output; failed/untouched slots
   remain in the denominator. Its cumulative receipt must retain the exact probe
   outcome at row zero; duplicate probe WAV bytes must be identical. No second
   synthesis or replacement probe is allowed. If both probes failed/no audio is
   eligible, the workflow produces empty screening results without model setup.
4. Bind a reviewed numeric-only NumPy/SciPy requirements file with actual wheel
   hashes and include it in source_files. Include the installed workflow path,
   all new module code/data, launcher/finalizer/tests, and retained container
   lock. Activation JSON itself is excluded to avoid a self-referential hash.
   Retained model/helper source hashes stay enforced by source-lock.json.
5. After this registered-but-disabled source lands on the default branch,
   discover the real workflow_id through the public official endpoint
   `https://api.github.com/repos/jiying2007/kws-pipeline/actions/workflows/tts-pilot-asr-20261014-v1.yml`
   without a token or workflow run. This preparation has not made that request
   and does not guess an ID. Bind the verified response only in a later reviewed
   activation change, then root enables and pushes
   the exact new branch once after reviewing all actual evidence. A null workflow
   ID cannot activate. Deleting/recreating the workflow changes its ID and fails
   the frozen identity check; never change identity to replenish attempts.

## One-shot execution and budgets

Only the exact new branch creation in the public repository, run_number=1 and
run_attempt=1 can enter. Both current run API and the workflow's lifetime history
must identify the same source/head/branch/path/attempt/number and contain exactly
one run for the frozen workflow_id. A second run, rerun, or changed workflow ID
is rejected. No repository tag/ref, credential or persistent permission is
created. Permissions are contents:read and actions:read. Only the read-only
artifact admission step receives the token; model containers receive none.

Execution: actual preset restoration → fixed 48-slot derivation → isolated host →
positive-list public export. Limits stay design 144 / current Qwen 96 ASR slots,
two primaries and third only on their successful disagreement. No sampled
consensus QA or old r5 attempts. Cosy16 remain blocked. A fresh directory is not
new authorization. Later failures do not permit restarting this workflow.

Only the public projection directory is uploaded. Original FLOAT archives and
ignored evidence remain in ephemeral runner staging; runtime, weights, code,
cache, setup logs and unreviewed decoder payloads are not uploaded. The isolated
export_public entrypoint needs the separately prepared sys.path fix before this
template is deployed; that fix has not been applied to root's current PR source.
Actual hosted readiness and budget feasibility remain unverified.

The activation binding now explicitly includes third_policy. Its checked-in value
is disagreement_only and enabled remains false. To confirm first-pass machine
training candidates with the third ASR, root must explicitly review and bind
training_candidate_confirmation before the first execution. No run/artifact
identity or budget guard changes. See ../TRAINING_CONFIRMATION.md for selection,
quality and tier semantics; missing admission policy does not enable expansion.
