# Saved paired near-pronunciation mechanism experiment

Status: **prepared; real paired replay NOT_RUN in this source change**. The
committed manifest is historical synthetic evidence. Actual ordinary real-world
negative hours, independent accent positives, and independent far-field positives
are all **zero**. Matched-FAR, accent, and far-field conclusions remain
**INSUFFICIENT_EVIDENCE**; future qualified evaluation is **NOT_RUN**.

## Narrow question and fixed controls

Compare the same saved acoustic logits through two research-only decoder modes:

- `strict_children`: top-1 child transitions only.
- `single_fuzzy_wo`: retain the strict lane and permit one non-top-1 token-4
  (`wo1`) transition in a separate lane per complete keyword path. A path cannot
  consume two fuzzy transitions. Existing `-8.25` fuzzy cost and `-16` retention
  budget still apply.

**Both research arms are narrower than the production default fuzzy-child policy.**
An improvement of B over A does not establish improvement over production.

The pinned model is the 6812-byte, F32/H64/V5 int8-weight tanh recurrent model,
SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`.
Vocabulary fingerprint is `0x316aea1a94b303bf`: `<blk>, ni3, hao3, xiao3, wo1`.
There is **no `wu` token**. This experiment cannot demonstrate literal `wu`
recognition or create a new acoustic category. Full keyword paths stay
`[1,2,3,4]` and `[3,4,3,4]`; a short suffix is not a keyword.

The only current threshold is **0.55**. Both arms share root margin 0.5, VAD,
blank/repeat rules, state retention, refractory behavior, keyword pack and fresh
decoder initialization. No threshold sweep, acoustic forward pass, training,
package installation, remote action, shipping-config change or CI change is
performed by these Python tools. The separate research C overlay is described
in `DECODER.md`; production source remains untouched.

## Saved evidence and limits

`data_manifest.json` pins all 13 original KWTR files from diagnostic run
36689641752, artifact 11085476672, original ZIP 23,590,873 bytes with SHA256
`dafaa468c0d9d6ffb323dca2845df74128ea128f0fc3870ce2d4224fcfe26672`.
Public retention source and original archive/member identities are linked in
`../diagnostic-readiness-2026-10-09/DURABLE-TRACE.json` and
`../diagnostic-readiness-2026-10-09/PUBLIC-READINESS-SUMMARY.json`.

The historical source traces cover a synthetic seed-1103 two-hour stream,
overlapping full/prefix windows, isolated source fragments and cold context
controls, plus seed-3301 short cold context. The primary non-overlapping source
coverage is 7,200 + 12 synthetic PCM seconds (360,598 saved frames). Posterior
grid duration is 7,211.96 seconds; this is not an ordinary real-negative FAR
exposure. Seed-3301's full stream is absent. Do not add every trace duration or
count these alternate conditions as independent tests.

The source-case metadata describes synthetic negatives `[3,3,3,4]` and
`[3,4,3,1]`. It is not positive-wake trial coverage or human annotation. Expected
human event truth, speaker/session identities and original-nightly state
equivalence are explicitly unknown. Each new replay resets the decoder. Saved
posterior slices retain prior acoustic history; cold controls have different
acoustic initialization. KWTR has no trie/refractory/RNN-state snapshot. No
claim of exact original or newer-nightly session-state restoration is allowed.

The 16 TTS items remain screen-only, outside this replay manifest. Neither TTS
screening nor raw ASR text is gold truth or training admission. Never normalize
`窝` and `屋` into an apparent ASR consensus.

## Verify existing bytes, without replay

From the repository root, with an already materialized archive member directory:

```sh
python3 research/near_pronunciation_v1/compare_saved.py \
  --trace-root /path/to/extracted \
  --verify-inputs-only
```

The supplied directory must contain the original `build/...kwtr` paths.
Validation checks the repository-pinned model/pack, exact 13-member identity set,
SHA256, size, KWTR header, frame count, finite logits, speech flags, timestamps and
frame grid. A missing cache entry or hash mismatch is a hard stop. There is no
`eval/run_corpus.ensure_cached_trace` import or acoustic-forward fallback.
The verification result explicitly says `result_status=NOT_RUN`.

## Compare saved replay outputs

The comparator does **not** invoke the C runner. After separately authorized
full-trace executions, retain a JSONL stdout file for each trace/arm and one arm
receipt per mode. Each successful invocation is exactly:

```text
RESEARCH_REPLAY MODEL PACK TRACE RECORDING --child-admission ARM
```

The derived research runner emits the original JSONL detection format:
`recording`, `keyword_id`, `time_s`, `confidence`. Empty stdout is a valid zero-hit
result only with a successful complete replay receipt. Do not fabricate an empty
file for a missing or failed run. Keep the build receipt from `verify_decoder.py`.

Each arm receipt has this structure (uppercase values below are placeholders):

```json
{
  "schema": "kws-saved-paired-arm-v1",
  "arm": "strict_children",
  "threshold": 0.55,
  "input_binding": "REPLACE_WITH_contracts.input_binding(plan, manifest)_OBJECT",
  "decoder_executable_sha256": "SHA256_OF_THE_COMMON_RESEARCH_EXECUTABLE",
  "replay_kind": "full_decoder_replay",
  "acoustic_forward_calls": 0,
  "replays": [
    {
      "recording": "EXACT_MANIFEST_RECORDING_ID",
      "trace_sha256": "EXACT_MANIFEST_TRACE_SHA256",
      "initial_state": "COPY_THE_RECORDING_initial_state_OBJECT",
      "returncode": 0,
      "complete_trace_replay": true,
      "argv": ["RESEARCH_REPLAY", "MODEL", "PACK", "TRACE", "EXACT_MANIFEST_RECORDING_ID", "--child-admission", "strict_children"],
      "invocation_input_hashes": {
        "executable": "SHA256_OF_THE_COMMON_RESEARCH_EXECUTABLE",
        "model": "PINNED_MODEL_SHA256",
        "keywords": "PINNED_PACK_SHA256",
        "trace": "EXACT_MANIFEST_TRACE_SHA256"
      },
      "observed_invocation_inputs": [
        {"path": "RESEARCH_REPLAY", "sha256": "SHA256_OF_THE_COMMON_RESEARCH_EXECUTABLE", "bytes": 1},
        {"path": "MODEL", "sha256": "PINNED_MODEL_SHA256", "bytes": 6812},
        {"path": "PACK", "sha256": "PINNED_PACK_SHA256", "bytes": 1},
        {"path": "TRACE", "sha256": "EXACT_MANIFEST_TRACE_SHA256", "bytes": 1}
      ],
      "stdout": {"path": "strict/recording.jsonl", "sha256": "STDOUT_SHA256", "bytes": 0}
    }
  ]
}
```

Replace every placeholder, including the illustrative `bytes=1`, with observed values.
Use `contracts.input_binding(plan, manifest)` to obtain the exact object. It
binds canonical complete plan and manifest content, including truth denominators,
source roles, overlap and initial state. Include every trace once in manifest
order. Paths in `stdout` are relative to the supplied results root. Both arms
must bind the same executable; mode is the only invocation option. Hash the
actual invoked files before execution, retaining path/hash/size descriptors in
`observed_invocation_inputs` in exact argv order. Record actual exit status and stdout,
then hash the outputs. A receipt binds the declared invocation and retained
output; it is not an independent execution attestation. Build/source provenance
must be separately checked against the C-overlay verification receipt. Emitted
times must lie on the exact saved sample grid and confidences must meet 0.55;
only half a six-decimal output unit (0.5 microseconds / 0.0000005 confidence)
is tolerated for C JSONL serialization.

```sh
python3 research/near_pronunciation_v1/compare_saved.py \
  --trace-root /path/to/extracted \
  --results-root /path/to/saved-results \
  --strict-receipt /path/to/saved-results/strict.receipt.json \
  --fuzzy-receipt /path/to/saved-results/fuzzy.receipt.json \
  --output /path/to/new-comparison.json
```

Existing output files are not overwritten. Reports contain per-trace genuine
detection counts, keyword IDs, times, confidences and count changes. Aggregate
counts/exposure and FRR/FAR remain null because the controls overlap and positive
truth is absent. Historical observations are never substituted for fresh
strict/fuzzy outputs.

### Optional controlled production reference

For separately authorized fresh replay with the unmodified production decoder,
pass `--reference-receipt production.receipt.json`. This is a reference baseline,
not a third candidate, and never participates in matched-FAR selection. Use the
same input binding, model/pack, traces, fresh decoder initialization and fixed
0.55 threshold. Record its own executable SHA and build provenance.

Its receipt uses `schema=kws-saved-production-reference-v1`,
`arm=production_reference`, `reference_only=true`, and
`source_kind=unmodified_production`. All other receipt fields and checks are as
above, except every exact argv is the original five-element form:
`[PRODUCTION_REPLAY, MODEL, PACK, TRACE, RECORDING]`, with no tuning flags. Reports
show its per-trace counts/events separately. Historical detections cannot replace
this controlled reference because original session-state equivalence is unknown.

## Future qualified development/test evaluation

The legacy `score_qualified` helper now accepts only explicitly fabricated
`qualified_event_evaluation_fixture` inputs and returns `FIXTURE_ONLY_MEASUREMENT`
(or `INSUFFICIENT_EVIDENCE`), with `fixture_only=true` and
`actual_evidence_verified=false`. It never reports a qualified measurement.
An arbitrary receipt hash and self-consistent metadata do not authenticate
source audio, trace coverage, output bytes or actual execution. In particular,
changing a declared duration and its source interval cannot establish FAR exposure.

The real-data `select_dev_thresholds` and `evaluate_frozen_test` entry points are
fail-closed with `UNVERIFIED_QUALIFIED_EVIDENCE` until actual byte-bound evidence
verification is implemented. Callers cannot unlock them by changing status or
verified flags. The separate `select_fixture_dev_thresholds` and
`evaluate_fixture_frozen_test` helpers exercise only fabricated contracts and
return explicitly `FIXTURE_...` states. They cannot freeze real thresholds.

This tightens the previously unconnected future-helper API, not the saved-only
CLI or historical paired outputs. The frozen plan, manifest and saved results
are unchanged. Explicit machine-consensus labels, nested weak/screen provenance
and declarations forbidding gold/qualification are rejected as independent gold;
weak-label screening remains available under its original role. Even fixture
detections must meet their declared threshold, allowing only half a six-decimal
serialization unit. The CLI cannot trigger this future path.

Future admission requires independent real-human recordings, explicit complete
event truth (`[]` only for a documented negative), truth-source hashes, known
source-family/speaker/session/initial-state metadata, disjoint source intervals
and group-disjoint dev/test splits. Exact trace hashes must be unique, and
`source_recording_sha256` binds intervals to stable original source content even
when an overlap-group label changes. Source-content identity cannot cross dev/test.
These source declarations need trustworthy provenance; this contract cannot
discover undocumented transformations of a recording. Protected historical trace hashes cannot be
relabeled gold, training, ordinary negatives or independent test evidence.

The fixture helpers reuse `eval/score_events.py` for maximum-cardinality event matching,
negative-only FAR and the Poisson 95% upper bound. Missing truth or zero negative
hours cannot become a zero FAR. No-positive FRR is null. Every threshold must
have a separately retained full decoder replay contract/receipt; confidence
filtering old detections cannot reproduce threshold-dependent decoder state.

Fixture reports retain their scoring inputs and are recomputed before selection;
a mutated summary cannot silently change the selection. Both arms and all
thresholds must bind one decoder executable and unchanged shared controls, in
addition to the same model, keyword pack, data and truth denominators. This is
a future evaluation contract; the current fixed-threshold C runner does not
implement a threshold sweep.

In fixture-only selection, the caller must specify the same maximum negative-only upper95 FAR/hour budget
for both arms. Only development may select thresholds, over the same fully
replayed grid and same denominators, minimizing FRR among feasible thresholds.
Ties use lower upper95 FAR, then lower threshold. No feasible threshold is a
reported failure, not permission to relax the budget. The frozen selection retains its development reports and original budget; test
evaluation recomputes that choice before accepting exactly one already-frozen
threshold per arm. Reusing a replay receipt across distinct arm/threshold/split
contracts and test-set selection are rejected. Acoustic,
accent/far-field and commercial qualification still require their own evidence.

Do not substitute `compare_dataset_iterations.py` (which rejects same-model,
same-data/different-decoder pairs), or `iterate_domain` calibration (which is not
this matched-FAR selector).

## Local tests

All fixtures are fabricated contract data. No test calls an acoustic model or
performs a C decoder replay.

```sh
python3 -m unittest discover -s research/near_pronunciation_v1/tests -v
python3 -O -m unittest discover -s research/near_pronunciation_v1/tests -q
python3 -OO -m unittest discover -s research/near_pronunciation_v1/tests -q
```

These focused tests do not constitute the full repository build/CI suite. Root
integration owns any later aggregate test registration and actual replay.
