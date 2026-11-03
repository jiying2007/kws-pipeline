# Source-bound multilingual KWS development candidate tournament

A deterministic **development-only** model selector for the existing
`dataset-iteration-v4` result cards. This is the evaluation/selection
layer after genuine KWS training and measured same-runner development
replay. It is *not* a new training runner, TTS/ASR provider or model
promotion mechanism. Run it as often as new frozen candidate scorecards
are available; it never calls models or consumes formal holdouts.

Entry point: `tools/select_multilingual_kws_models.py`.

## Why this matters

- The frozen product still has two exact Mandarin words:
  **你好小窝** and **小窝小窝**, with standalone **小窝** forbidden.
- The 2026-11-03 frozen synthetic nightly regression remains
  **2 false accepts / 8 hours = 0.25/h, 95% upper ≈0.787/h**.
  Lowering nightly FAR thresholds or modifying labels cannot fix this.
- Current `kws-data` contains infrastructure for authorized multi-speaker
  Qwen TTS / local 3-ASR machine screening; generated clip count, 3/3 ASR
  strings, and replay CI PASS are not product error metrics.
- The existing governed `tools/compare_dataset_iterations.py` already
  enforces one variable, pinned scorecard protocol and exposure-aware FAR.
  Reuse that calculator rather than invent another FAR estimate.

This addition selects among up to **8 already-evaluated model candidates**,
all measured against the **same independent frozen development dataset**,
same original references digest, code SHA, protocol, seed, positive count
and negative exposure. The only variable allowed to change is `model_id`
(the hash includes the keyword pack/threshold). If dataset **and** model
both change, stop and run an explicit paired rebaseline.

## Selection dimensions

The selector verifies `schema_version=4` v4 scorecards, the producer's
`negative-only-recordings-iteration-approx95-v1` FAR identity/point/bound
math, exact file SHA256, and internal positive `expected/matched/FRR`
consistency. From expected-wake count it also computes a **one-sided 95%
Wilson upper confidence bound** on the *development* false-reject
proportion. This is not independent true label authentication.


Every resource number must be read back from a private, **SHA-bound
measurement receipt**. The minimum receipt schema is:

```json
{
  "schema_version": 1,
  "policy": "kws-host-development-resource-v1",
  "model_id": "ACTUAL_64_HEX",
  "code_sha": "ACTUAL_40_HEX",
  "measurement_class": "host-development-only",
  "simulated": false,
  "metric_values": {"cpu_ms_per_s": 90, "rss_kib": 1900, "p99_block_ms": 3},
  "training_admitted": false,
  "qualification_allowed": false
}
```

The receipt must be an actual local file with the declared SHA, same model
and code identity, and exact matching resource numbers. Editing the numbers
in a tournament plan cannot make a resource-heavy model look fast. This
still does **not** independently prove the truth of a measurement, the
environment's provenance, or physical SSC305 behavior; the source producer
must retain original host measurement logs and attestations. Do not use
synthetic fixtures as real resource measurements.

Candidate gate requirements are declared upfront in a private immutable
tournament plan: minimal positive count and negative hours; maximum
FRR upper bound, FAR upper bound, p95 post-end latency, CPU per audio
second, RSS and p99 block time. Each resource record has a hash-bound
**host-development-only** measurement receipt. The selector does not
claim the host is an actual SSC305 board.

Reject a candidate for insufficient evidence, invalid source/pins,
worse point FRR or FAR confidence than baseline, or any declared
development resource budget failure. Then compute a Pareto frontier over
FRR confidence, FAR confidence, latency and three resource metrics.
Choose a deterministic *research leader* from the non-dominated set.

If there is no eligible model, output `research_leader_id=null`;
**never** choose a winner merely because some experiments finished.
A development model cannot be automatically shipped based on this
ranking. Missing per-keyword true labels / source family separation and
actual independent long-stream/AFE evidence remain required product
blockers, even if an aggregate score looks good.


The \`language_binding\` is also a *private, hash-bound* original cohort
declaration: \`policy=kws-development-language-cohort-binding-v1\`,
\`schema_version=1\`, \`locale=zh-CN\` or \`en-US\`, the **same** pinned
\`dataset_id\`, \`references_sha256\`, \`code_sha\`, \`protocol\`, and
\`seed\` from the v4 baseline, with \`source_role=development-only\`,
\`independent_language_verified=false\`,
\`sealed_or_formal_data_used=false\`, \`training_admitted=false\` and
\`qualification_allowed=false\`. The selector verifies its exact bytes
before ranking, so changing \`locale\` in a tournament plan alone does not
turn a Mandarin dataset into English evidence. This note **cannot prove**
what language a human actually spoke; later source-authenticated language
annotations must be verified through the true data admission chain.

## Input contract (private)

```json
{
  "schema_version": 1,
  "policy": "source-bound-kws-multilingual-development-tournament-v1",
  "locale": "zh-CN",
  "role": "development-only",
  "language_binding": {
    "path": "/restricted/eval/development-language-cohort.json",
    "sha256": "ACTUAL_64_HEX"
  },
  "sealed_or_formal_data_used": false,
  "model_calls_authorized": false,
  "baseline": {
    "id": "control",
    "scorecard_path": "/restricted/eval/control.json",
    "scorecard_sha256": "ACTUAL_64_HEX",
    "resource": {
      "measurement_class": "host-development-only",
      "measurement_receipt_path": "/restricted/eval/control.resource.json",
      "measurement_receipt_sha256": "ACTUAL_64_HEX",
      "cpu_ms_per_s": 100,
      "rss_kib": 2000,
      "p99_block_ms": 4
    }
  },
  "candidates": [
    {
      "id": "candidate-a",
      "scorecard_path": "/restricted/eval/candidate-a.json",
      "scorecard_sha256": "ACTUAL_64_HEX",
      "resource": {
        "measurement_class": "host-development-only",
        "measurement_receipt_path": "/restricted/eval/candidate-a.resource.json",
        "measurement_receipt_sha256": "ACTUAL_64_HEX",
        "cpu_ms_per_s": 90,
        "rss_kib": 1900,
        "p99_block_ms": 3
      }
    }
  ],
  "budgets": {
    "min_expected_wakes": 100,
    "min_negative_hours": 10,
    "max_frr_upper_95": 0.15,
    "max_far_upper_95_per_hour": 0.4,
    "max_latency_ms": 500,
    "max_cpu_ms_per_s": 500,
    "max_rss_kib": 3000,
    "max_p99_block_ms": 10,
    "frr_point_tolerance": 0
  }
}
```

Values are illustrative *research-development* budgets, not new
commercial shipping limits. Placeholder hashes deliberately fail.
`locale=en-US` is also accepted but cannot enable an English
shipping product or invent its wake phrase. Each language runs a
**separate** tournament. English and Mandarin scorecards are never
compared across language populations.

Use the original producer to generate cards on a restricted workstation:
`tools/run_dataset_iteration.py` against the **same frozen development
references** for baseline and candidates. The input cards must be real
execution results with exact model/keyword hashes, not mocked metadata.

```bash
python3 tools/select_multilingual_kws_models.py \
  --plan /restricted/eval/frozen-tournament-plan.json \
  --output /restricted/eval/new-development-ranking.json
```

Output is a private mode-0600 create-only report. It contains a per-candidate
reject reason list, Pareto IDs, deterministic research leader (or none),
underlying development scorecards and referenced SHA. It cannot write into
the public Git repository. The planner itself has **zero training,
model inference, network, license, source admission or deployment calls**.

## Automatic iteration architecture (when actual data exists)

1. `kws-data` privately synthesizes bounded positive/near-miss/ordinary
   negatives with exact generator/source/rights receipts, *then* truly runs
   independently pinned ASR and routes conflicts to quarantine.
2. Original reviewed train-only speech passes existing strict
   `speech_like_corpus_plan`, `asr_source_resolver`,
   `training_admission` and family split checks. No generated
   expected text is treated as gold.
3. Pre-registered control/candidate models train under bounded public/free
   runner budgets, no sweep over sealed results. Both are evaluated under
   the **same** frozen development corpus/code/AFE.
4. This selector automatically rejects regressions and reports the Pareto
   research frontier. The next training round can be proposed from its
   retained report, only if independently authorized and with a new
   causal data/loss hypothesis. Keep earlier unsuccessful runs unchanged.
5. The final product acceptance remains the existing **Phase A** real-human
   40 speaker / 2 session / 1200 wake / 24h negative policy and its
   confidence bounds, followed by **Phase B** same-device physical SSC305
   constraints and **Phase C** reviewed immutable release. A
   development Pareto winner **never changes** shipping approval.

Observed resource improvements on x86 are *not* device certification.
The user requires low-compute on the SSC305, so final deployment decisions
must use actual ARMv7 build, target p99, CPU, memory, power, thermal
and final AFE measured end-to-end word-end-to-event latency.
