# Narrow Sense frontend resolver patch (not applied)

`sense-frontend-resolver-review.patch` proposes exactly two edits to the resolver plus adding `tools/sense_frontend_runtime.py`. The base source SHA256 is recorded in resolver-patch-base.json. No existing resolver or source root is edited by this preparation.

1. SenseVoice native-v2 recognizer provenance rows additionally require `frontend_runtime_binding`, an archive/member reference resolved by the existing trusted `read` function.
2. Only `frontend_config_sha256` under adapter `sensevoice-full-input-greedy-ctc` enters the new verifier. Other native hashes retain their original membership checks. Source-policy, rights, source audio, source roots, started/completed/raw-output identity and hash checks remain unchanged.

The new retained binding has exact keys:

    {"schema":"reviewed-sense-frontend-dependencies-v1",
     "snapshot_sha256":"<actual observed frontend bytes hash>",
     "dependencies":{
       "frontend_source":{"reference":{"archive":"...","member":"..."},"bytes":123,"sha256":"..."},
       "fbank_source":{...}, "backend_package_source":{...}, "backend_binary":{...},
       "backend_core_library":{...},
       "cmvn_file":{...}, "tokenizer_file":{...}, "parser_source":{...}
     }}

Every nested original dependency must uniquely match the actual loaded asset/executed source inventory by member path, hash and size. Its original bytes must be present in independently authenticated provenance. The runtime snapshot itself is not inserted into either inventory. Model assets and executed package/module code are distinct from this derived snapshot.

Validation rejects duplicate JSON keys, extra/missing fields, nonfinite numbers, bool/int coercion, incorrect field type labels, dependency aliases/traversal, missing or ambiguous inventory members, raw-byte drift, wrong snapshot binding, CMVN shape/value/hash errors and CMVN source/tensor disagreements. CMVN values must be exactly representable finite float32 observations. The two vectors are independently parsed from the retained original CMVN file according to the already reviewed AddShift/Rescale format and compared to the recorded tensor bytes. No missing value is defaulted.

## External evidence still required

- A genuinely pre-recognition frozen three-model policy and source/request map.
- Actual zero-forward preparation and per-call observations from the fixed runtime; mocked objects are not evidence of execution.
- Independently reviewed eight-dependency manifest, original source/package/model dependency bodies, effective snapshot, and valid pre-forward start/completion chains.
- The frontend binding must itself be a member of the independently pinned provenance, not caller-supplied JSON accepted on its own.
- A reviewed training-eligible source policy, rights and trust root. This patch deliberately creates none.
- Full integration tests against the complete resolver and all three actual capture paths. Current checks are adversarial standard-library verifier tests and patch application/AST integrity checks, not a formal admission pass.

The patch retains the resolver's existing 32MiB member ceiling. A larger backend dependency must remain blocked pending separate review; it cannot be silently truncated or bypassed.

Native binary mapping evidence is mandatory for both the Python extension and its DT_NEEDED core library; true values come only from the reviewed producer checking its own executable mappings against original file inode/device/path. The exact core library bytes are independently required in the reviewed binding and executed inventory. The snapshot exemption does not exempt core library membership.
