# Outcome-based KWS data feedback priorities

This is the first closed-loop **decision layer**, *not* a training/admission
gate. Input is the private, independently checked
\`kws-data\` bulk development report from
[the cross-repository effect boundary](KWS_DATA_BULK_EFFECT_BOUNDARY.md).
The only entry point is
\`tools/prioritize_kws_data_feedback.py\`. It invokes no TTS, ASR, speech
synthesis, model training, or model scoring.

## Why this exists

Blindly generating 16 utterances × 2 repetitions × 512 voice references can
create many near-duplicates and unknown pronunciations without moving the
product FAR/FRR. An honest automated loop must identify *which evidence or
acoustic problem must be resolved next*, preferably before spending another
batch of generation calls.

The planner never interprets target/requested text as actually spoken. It
accepts only the specific \`kws-data-bulk-effect-consumer-v1\` report, with
complete byte/source/ASR-screen membership and derived-count consistency,
and rejects forged source authority or stale totals.

It sorts the following work deterministically: unresolved potential full-wake
in a supposed negative **P0/quarantine**, no machine-supported positive for
keyword 1/2 **P0/pronunciation**, unfinished source or ASR **P1/inspect consumed
receipts**, disagreements **P1/acoustic adjudication**, a supported positive
**P1/authenticate exact actual-label training source**, repeated PCM/single
source-family **P1/diversity diagnosis**, and the mandatory external real
human/final AFE and SSC305 qualification **P0/blocker**.

Field-capture priorities can be checked using [Phase-A field-plan coverage preflight](PHASE_A_FIELD_PLAN_PREFLIGHT.md), which counts only requested tasks and leaves every actual qualification metric unavailable.

The ordered plan is **advice**, not automatically generated additional data.
No one-off confidence threshold, scorecard, or TTS voice count can grant an
unverified source a product-training identity.

## Read-only invocation

\`\`\`bash
# Run after the existing trusted source producer and the independent effect
# boundary. These examples refer to files within one restricted workstation.
python3 tools/kws_data_bulk_effect_boundary.py \
  --plan /restricted/bulk/plan.json \
  --screen /restricted/bulk/screen.json \
  --generated-root /restricted/bulk/generated \
  --output /restricted/reports/effect.json

python3 tools/prioritize_kws_data_feedback.py \
  --effect /restricted/reports/effect.json \
  --output /restricted/reports/feedback-priorities.json
\`\`\`

The output directory must already exist and be private. The output file
must be new, is never overwritten, and has mode 0600. Do not copy WAV,
raw transcripts, original human identity, source-rights files or private
per-sample record bodies into public source-control.

## Action authority (intentionally absent)

Every action contains \`new_model_or_generator_calls_authorized=0\` and both
\`training_admitted=false\` and \`qualification_allowed=false\`. The complete
plan has \`next_campaign_authorized=false\`,
\`source_rights_authenticated=false\`,
\`asr_source_authenticated=false\`,
\`human_gold=false\`, \`shipping_approved=false\`.

The feedback can inform the next independently authorized bounded
\`kws-data\` campaign (after acquiring legitimate voice rights and pinned
provider assets) and the existing training/dev comparison. It cannot
update the canonical source pool, retag a consumed attempt, open a training
request, pull private human speech into GitHub or waive any Phase A/B/C gate.

## Verification and limits

Unit fixtures only use temporary invented PCM and mocked source screens.
They verify honest preference ordering, tampered outcome count rejection and
private-only write behavior. They are **not** evidence of fresh Mandarin
pronunciation, independent speaker generalization, real household FAR,
final-AFE latency, or physical SSC305 performance.

No automatic retries, model/hyperparameter sweeps or promotion are
implemented. The product target remains both complete keywords, actual
per-word FRR/FAR and independent 3–5 m target-board evidence.
