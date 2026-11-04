# Git-registered frozen-model FAR research

**Research-only status.** The scheduled `far-nightly.yml` measures reproducible
synthetic longstream false accepts for a *historical comparator*. It does not
qualify a product, estimate real-room FAR, prove 3–5 m wake reliability, or
approve release/installation of a model.

## Execution and source authority

- The model and two-keyword pack are read from
  `models/registry/model-749187ec1d66/` in the exact Actions checkout;
  `tools/verify_model_registry.py`, `MODEL_SHA256SUMS`, and
  `configs/research-baseline.xiaowo.json` must agree on model bytes, source
  SHA, consumed seed and frozen historical interpretation.
- No `gh release download`, model promotion, shipping config, commercial
  approval, private voice/model service or new model training is required.
- The real C runner is built from that run's source; parameter limits and
  research baseline must match their Git-source-bound hashes.
- Four independent deterministic 7,200-second negative streams run with
  seeds 1103/2207/3301/4409, checked injected-source coverage and two
  known historically false-triggering WAV identities. Retain the original
  zero-FAR point-estimate criterion, 0.40/h one-sided 95% upper limit,
  800 minimum injections and 800 seconds of hard-negative audio; do
  not weaken or skip failure when a known false accept recurs.

A source-only CI PASS or a frozen synthetic model summary is **not** actual
real-human, device-embedded, target-AFE or independent longterm FAR evidence.
The original historical Release assets and past receipts remain in GitHub
and Git history; new research outputs are separate v2 evidence, not a rewrite
of those records.

## Read-only third-party recheck

`tools/far_regression_receipt.py` produces
`research-frozen-far-regression-v2` receipts with all saved WAV/source/runner/
model/keyword/config SHA values. Its `fetch` mode accepts
`--research-request <json-path>` with:

```json
{
  "schema": "research-far-verification-request-v1",
  "repository": "jiying2007/kws-pipeline",
  "source_sha": "EXACT_40_HEX_CURRENT_MAIN",
  "research_baseline_sha256": "EXACT_64_HEX",
  "nightly_contract_sha256": "EXACT_64_HEX",
  "parameter_contract_sha256": "EXACT_64_HEX",
  "model_sha256": "EXACT_64_HEX",
  "keyword_pack_sha256": "EXACT_64_HEX"
}
```

Fill only with independently verified live values; placeholders above are
documentation, not a signed or admissible request. The verifier rejects an
older green run when the current matching-source run failed, missing input
or changed version, and does not issue product or scientific performance
approval. Saved old v1 deployment-bound receipts remain historical and are
not silently accepted as v2.

## Migration verification

The mainline corpus/effect regression tests use fabricated offline bytes to
exercise source binding, false/unknown outcomes, model/keyword drift, digest
tampering, failed-current-run precedence and archive extraction safety.
They are **negative/test-contract evidence only**. Preserve Python 3.8 in
research tools; do not add a redundant compatibility import path to the
retired shipping keyword validator.
