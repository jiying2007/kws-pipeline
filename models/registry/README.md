# Promoted model registry

Each child directory is named by the immutable GitHub model release tag.

A valid entry is a byte-for-byte mirror of the promoted release plus
`registry.json`.

Validate any entry with:

```bash
python3 tools/verify_model_registry.py \
  --registry-dir models/registry/model-<id>
```

For the currently pinned product model, also bind the repository shipping
contract:

```bash
python3 tools/verify_model_registry.py \
  --registry-dir models/registry/"$(jq -r '.model.release_tag' configs/shipping.xiaowo.json)" \
  --config configs/shipping.xiaowo.json
```

Do not manually replace bytes in an existing entry. A new trained/promoted model
gets a new immutable release tag and a new registry directory.
