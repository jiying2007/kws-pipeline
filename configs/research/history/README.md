# Historical research comparators (read-only)

The two JSON records in this directory preserve **byte-identical** original
Phase-A real-human and Phase-B physical-target policy inputs previously stored
under `commercial/`. They are frozen *historical scientific reproducibility*
references, **not** current KWS training, model selection, independent corpus
admission, product approval, shipping eligibility or new device evaluation.

- `real-human-phase-a.policy.json`: original 40 independent speakers,
  two sessions, ≥1,200 wakes, ≥24 h real negatives, FRR/FAR/confidence/latency
  and critical-slice thresholds. Includes original historical
  `deployment_tag` and `shipping_approved=false` metadata only because
  previous archived receipts bind those names.
- `target-dut-phase-b.policy.json`: original DUT ≥24 h, cohort ≥3 devices,
  optional historical 72 h source benchmark, source/model/AFE/resource hash
  contracts and fail-closed CPU/RSS/thermal/power/continuity constraints.
  Historical release/approval fields are not new research permissions.

New human field-planning uses
[`configs/research/real-human-coverage-v1.json`](../real-human-coverage-v1.json).
Physical research uses the independently verified
[`configs/research/physical-target-resource-budget.schema.json`](../physical-target-resource-budget.schema.json)
with real source/rights and actual board measurements. No live GitHub Actions
workflow may consume these historical policy files. All original policy source
history remains traceable via Git blobs, prior commits, and Actions receipts.
