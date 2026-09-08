# Phase B physical target qualification and Phase C shipping approval

This document defines the product gates that follow a successful Phase-A `human-qualified-*` Release. It does not substitute hosted fixtures for physical measurements and it does not authorize model retraining.

## Frozen chain

Every Phase-B and Phase-C receipt must remain bound to the same tuple:

- immutable deployment `deployment-c20f3eb88e43`;
- immutable accepted model `model-749187ec1d66`;
- the exact Phase-A `human-qualified-*` Release and real-human corpus identity;
- the exact Phase-A final-AFE identity;
- one shipping SKU and board revision.

Changing the deployment, model, AFE executable/config, SKU or board revision starts a new qualification tuple.

All Phase-A/B/C evidence workflows must be dispatched from the exact current protected `main`. `governance/require_current_main.sh` rejects branch refs, stale main SHAs and weakened live rulesets before evidence promotion.

## Phase B1 — one physical DUT

Use the existing physical evidence collectors documented in `TARGET_EVIDENCE.md`. Raw lab evidence is produced and retained by the controlled qualification station; GitHub receives aggregate metrics, hashes and attestation-bound identities only.

The controlled target runner must carry:

```text
self-hosted
kws-target-board
```

and define `KWS_TARGET_QUALIFICATION_ROOT`. Dispatch `target-dut-qualification` with a pseudonymous `target_profile`; the bundle is resolved as:

```text
$KWS_TARGET_QUALIFICATION_ROOT/<target_profile>/
```

Required fixed bundle files:

```text
target-profile.json
target-evidence.json
board-summary.json
board-runner
board-audio.wav
evidence-raw.jsonl
attestation-verification.json
raw/runtime-soak.json
raw/power.csv
raw/audio-continuity.json
... any additional raw evidence named by evidence-raw.jsonl
```

`board-audio.wav` must be `non-human-public-safe`. Human or Phase-A post-AFE recordings are forbidden as a public target benchmark fixture.

`target-evidence.json` is produced by `tools/collect_target_evidence.py`. It must carry both the final-AFE executable SHA and the **full Phase-A final-AFE identity SHA**. The external trust-layer verification must independently bind that full identity together with the raw-evidence manifest, collector, board runner, model and keyword pack; `target-dut-qualification` parses and re-verifies those fields instead of trusting the summary alone. `board-summary.json` is produced by the exact shipping target `kws_board_bench`.

Per-DUT policy is `commercial/target-qualification.policy.json`. Current gates require at least 24 h soak and bound p99 processing, RTF/headroom, CPU, RSS, stack high-water, temperature, average power, XRUN, lost-sample, discontinuity and backpressure evidence.

A PASS publishes immutable:

```text
target-qualified-<target-evidence-sha16>
```

with `shipping_approved=false`. Raw lab traces remain in the controlled evidence store; public receipts retain their hashes and external-verifier bindings.

## Phase B2 — multi-DUT cohort

One DUT cannot authorize commercial shipping. Dispatch `target-cohort-promotion` only after multiple immutable `target-qualified-*` Releases exist.

The machine policy requires:

- at least 3 unique DUT IDs;
- unique physical target-evidence hashes for every DUT;
- every DUT bound to the same deployment, Phase-A corpus/final-AFE identity, SKU and board revision;
- every DUT already passing the per-DUT policy;
- at least 24 h soak per DUT;
- at least one DUT with >=72 h soak.

A PASS publishes immutable:

```text
target-cohort-qualified-<cohort-summary-sha16>
```

and still leaves `shipping_approved=false`.

## Phase C — terminal shipping approval

`shipping-approval` is the only workflow allowed to emit a product receipt with `shipping_approved=true`.

It is manual-only and runs only after:

1. the workflow source is the exact current protected `main` and live governance passes `governance/verify_live_main_ruleset.py`;
2. the immutable commercial deployment is valid;
3. the supplied immutable Phase-A Release is `qualified=true`;
4. the supplied immutable Phase-B multi-DUT cohort is `qualified=true`;
5. A and B bind the exact same deployment target, human corpus and final-AFE identity;
6. the deployment still contains exactly `你好小窝` and `小窝小窝` at threshold `0.55`.

The workflow creates `shipping-approval-manifest.json` with:

```json
{
  "status": "shipping-approved",
  "shipping_approved": true,
  "shipping_approval_blockers": []
}
```

plus exact deployment/model/wake-word, Phase-A, Phase-B and repository-governance identities. It freezes checksums, produces Sigstore provenance and publishes immutable:

```text
shipping-approved-<approval-manifest-sha16>
```

The approval Release is the shipping authority for that immutable tuple. It does not alter model weights, thresholds, decoder behavior, training state or the fail-closed candidate contract in source control.

## Failure / change rules

- A failed target DUT may be re-measured only as new physical evidence; do not edit the old evidence bytes.
- A copied evidence object cannot represent another DUT.
- A board revision or final-AFE change creates a new cohort.
- Real-human failures do not automatically authorize model training.
- Formal seed `271838` remains consumed/frozen; `271839` remains reserved until an intentional new model candidate is justified.
