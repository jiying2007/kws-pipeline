# Offline candidate identity registry

This is a reproducible **candidate inventory**, not an admission receipt, source
trust root, human gold set, evaluation dataset, or authorization to run training.
It contains exactly the historical thirteen plus kw1s-05/06/07: **16 clips,
KW1=5 and KW2=11**. No historical frozen source pins or receipts are modified.

## Evidence boundary

All sixteen native FLOAT WAV hashes and decoded FLOAT values, and all sixteen
actual 16 kHz mono s16le PCM byte hashes, are verified against pinned archives.
The old thirteen PCM hashes are computed from decoded WAV sample bytes; the WAV
container hash is retained separately. Historical PEAK chunks are omitted only
in memory to reuse the existing strict FLOAT decoder; sample bytes are unchanged.
The new three derived WAV hashes are calculated using the existing canonical
header function in memory. Those WAV containers are not claimed to be archived.
No audio files are created, resampled, normalized, trimmed, or regenerated.

The old ASR ZIP contains 32 WAVs, generation receipts, a report, and results.
Its ASR rows bind by sample ID, **not by per-recognizer PCM digest**. Consequently,
all thirteen are explicitly `historical_id_join_only` with a null ASR PCM binding.
Computing a PCM hash now cannot retroactively prove what a decoder consumed.
The new three have verified started/completed/decoder evidence chains bound to
the same archived PCM and canonical WAV hash across all three recognizers. Their
`kw1-blind-*` receipts are still not the production `retained-asr-*` schema.
Neither campaign is silently promoted into the production source trust root.
Native-to-derived resampling is not rerun, so that transformation remains an
archived provenance claim rather than a newly reproduced numerical derivation.

The existing screening normalize function performs Unicode NFKC, removes
whitespace and Unicode punctuation, and never rewrites homophones to a target.
Each raw transcript, status/completeness, and exact target agreement is checked.
Preset voices are explicitly not real speaker identities. The same named Qwen
preset shares a conservative group across engines and campaigns; independence
is not proven. Known screening and old thirteen development-fit exposure is
recorded, with `history_complete=false`; missing old fit receipts are not rebuilt.
Calibration, selection, and sealed-test eligibility remain false for every row.

## Offline reproduction

From the repository root, with the four original ZIPs in a locally retained artifact directory:

```sh
python research/kws-candidate-registry-v1/build_registry.py \
  --repo . --artifacts /local/artifacts \
  --verify research/kws-candidate-registry-v1/candidates.json
```

Use `--output /local/regenerated.json` to emit metadata only. `input-pins.json`
binds the exact public repository inputs and reused source functions. Its own
raw-byte SHA256 is fixed in the builder. All seven inputs are verified before any
dynamic source executes; execution uses those verified in-memory bytes. Registry
comparison is canonical and type-sensitive, including boolean admission flags. ZIP SHA256
is checked before parsing. The reader bounds compressed bytes, members, member
sizes and total uncompressed bytes; rejects aliases, duplicate/path-traversal
members, symlinks, encrypted entries, duplicate JSON keys and nonfinite JSON.
No archive is extracted and no inference package is imported.

The optional research-only `verify_inputs(registry, sources)` returns verified
PCM bytes in memory after exact whole-registry regeneration. `sources` has exactly
`repo` and `artifacts` keys. It creates no training/admission/fit receipt and does
not authorize any optimizer execution. Callers must retain the evidence limits.

## Tests

```sh
python research/kws-candidate-registry-v1/test_registry.py
python -O research/kws-candidate-registry-v1/test_registry.py
python -OO research/kws-candidate-registry-v1/test_registry.py
```

The default suite runs seven metadata/offline tests without requiring artifact
ZIPs. Add `--require-live` and set `KWS_REGISTRY_REPO` and
`KWS_REGISTRY_ARTIFACTS` to require the eighth live-byte regeneration/adapter
test. Missing locally retained artifact ZIPs or environment variables then fail
closed; no expected skips are counted as live verification. All modes support
`python -I -S -B` and `-O` / `-OO`.
Adversarial cases cover admission and evaluation escalation, fake human speaker
identity, duplicate membership, promotion of historical ASR binding, WAV hash
mislabelled as PCM, swapped ASR binding, malformed JSON, unsafe ZIP entries,
wrong ZIP hashes, symlink archives, truncated PCM, changed raw transcript,
boolean/integer type confusion, and altered pin files or executable source that
must be rejected before any dynamic module runs.
All checks use explicit exceptions rather than optimizer-removable assertions.
