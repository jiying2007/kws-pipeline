# Immutable KWS research model registry

This directory retains exact model bytes and training-source provenance for
reproducible acoustic experiments. A GitHub tag and historical metadata fields
are evidence identifiers, **not** permission to ship or a claim of model quality.

Every model directory contains `registry.json` and its hash-bound assets.
Verify a saved model without a deployment, approval, or publication config:

```sh
python3 -B tools/verify_model_registry.py \
  --registry-dir models/registry/model-749187ec1d66
```

The default research comparator is pinned independently by
[`configs/research-baseline.xiaowo.json`](../../configs/research-baseline.xiaowo.json).
Its model and keyword-pack SHA-256 values are checked against actual registry
assets by `.github/workflows/model-registry-contract.yml`.

Do not rewrite an existing registry entry. New research candidates must retain
their own immutable training, data, model and source-family identities.
Retained historical records, model tags and previous measurements are not
blind validation evidence for a new experiment.
