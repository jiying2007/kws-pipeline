# Runtime configuration

This document is the operator-facing companion to `configs/parameter-contract.json`.
That JSON file is the single source of truth for every tunable parameter, its unit
and its legal range; this document explains what each parameter does, how to tune
it, and what a change invalidates.

## How the contract is consumed

| Consumer | Mechanism |
| --- | --- |
| `src/kws.c`, `src/decoder.c`, `src/frontend.c`, `src/keyword_pack.c` | include the private header `build/generated/kws_parameter_limits.h`, generated at CMake configure time by `tools/gen_parameter_limits.py` |
| `tools/compile_keywords.py` | reads the JSON directly; `--contract` overrides the path |
| `tools/qualification_common.py` | reads the JSON directly when validating a runtime config |
| `tests/test_parameter_contract.py` | re-derives every bound from the JSON and asserts the generated header matches |

Consequences:

- A bound is a compile-time constant in C, so `kws_engine_init()` returns
  `KWS_EINVAL` and `kws_keyword_pack_open()` returns `KWS_EFORMAT` for any value
  outside the contract. There is no second copy of the range to drift.
- The generated header records `KWS_PARAMETER_CONTRACT_SHA256`, so a shipped
  binary can be tied back to the exact contract revision that produced it.
- `configs/shipping.xiaowo.json` pins the same digest in its
  `parameter_contract.sha256` and
  `threshold_calibration.required_parameter_contract_sha256` fields. The separate
  `threshold_calibration.parameter_contract_sha256` retains the contract identity
  recorded for the historical calibration; metadata maintenance must not silently
  rewrite that evidence binding. While the historical and required digests differ,
  `recalibration_required` must remain true and `shipping_approved` false. Editing
  the contract without revisiting these bindings fails
  `tests/test_parameter_contract.py`.

Regenerate the header manually with:

```sh
python3 tools/gen_parameter_limits.py configs/parameter-contract.json build/generated
python3 tools/gen_parameter_limits.py configs/parameter-contract.json build/generated --check
python3 tools/gen_parameter_limits.py configs/parameter-contract.json --json
```

## Layers

| Layer | Meaning | Change cost |
| --- | --- | --- |
| L0 | model-bound: packaged in the `.kwm` blob | new model release |
| L1 | firmware constant: compile-time macro | rebuild plus full regression |
| L2 | product config: `kws_config_t` field | rebuild of the caller plus revalidation of calibrated thresholds |
| L3 | field policy: per-keyword KWKP v3 record field | recompile the keyword pack and rerun calibration |

## L1 decoder boundary policy

`KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES` is the compile-time limit for
consecutive frames whose speech gate is inactive while a keyword prefix may be
carried across an utterance boundary. The default is `12` frames, or `240 ms` at
the current 20 ms frame hop. After the twelfth inactive frame the decoder clears
only trie-prefix and pending-keyword history. It does **not** reset the frontend,
RNN hidden state, refractory accounting, or calibrated keyword thresholds.

While the stream remains speech-inactive, partial paths are cleared at the end
of every frame so nonblank posterior noise in the gap cannot seed the next
utterance. Speech resumption starts decoder matching from an empty trie history.
This is deliberately separate from `state_retention`: retention controls gradual
path decay; the boundary policy prevents a sufficiently long non-speech interval
from joining two otherwise valid phrase fragments.

Because this policy changes which decoder event paths can survive, changing the
boundary value invalidates calibrated keyword thresholds even though it does not
modify the threshold numbers themselves. Recalibrate on the applicable acoustic
evidence before promotion; current shipping approval remains blocked on the
real-human final-AFE gate.

## L2 runtime parameters

These are the fields of `kws_config_t`. `kws_default_config()` returns exactly the
contract defaults below.

| Field | Type | Default | Range | Unit | Effective | Invalidates thresholds |
| --- | --- | --- | --- | --- | --- | --- |
| `min_speech_dbfs` | float | `-55.0` | `[-120.0, 0.0]` | dBFS | yes | yes |
| `token_boost` | float | `1.5` | finite, `>= 0.0` | nat | **no — deprecated** | no |
| `state_retention` | float | `0.94` | `(0.0, 1.0)` exclusive | ratio per frame | yes | yes |
| `refractory_ms` | uint32 | `1200` | `[0, 10000]` | ms | yes | no |
| `external_vad_threshold` | float | `0.45` | `(0.0, 1.0)` exclusive | probability | yes | yes |

### `min_speech_dbfs`

Speech gate applied to the 400-sample frame energy of the post-AFE PCM. It is
only consulted when the caller does **not** supply an external VAD probability in
frame metadata. Raising it makes the engine treat more frames as non-speech,
which accelerates prefix decay through `KWS_SILENCE_RETENTION_LOG` and reduces
false accepts at the cost of recall in quiet far-field conditions.

The frontend resets `last_dbfs` to the contract minimum (`-120.0`), so a freshly
reset frontend never reports speech. Any value below that floor is rejected.

### `state_retention`

Per-frame retention factor applied to a live keyword prefix on a speech frame.
The decoder stores `log(state_retention)`; the value must be strictly inside
`(0, 1)`. Lower values make the decoder less tolerant of slow or disfluent
speech and suppress more continuous-background false starts. The default `0.94`
keeps about 94% of a prefix score per 20 ms speech frame.

### `refractory_ms`

Suppression window after an emitted detection. It is measured in processed
samples, so it is unaffected by wall-clock jitter. Detections landing inside the
window are counted in `kws_engine_stats_t.refractory_suppressed` rather than
emitted. It does not invalidate thresholds because it only removes duplicate
events after the confidence gate has already fired.

### `external_vad_threshold`

Decision threshold applied to `kws_frame_metadata_t.external_vad_probability`
when the caller sets `KWS_FRAME_EXTERNAL_VAD_VALID`. When the flag is clear the
engine falls back to `min_speech_dbfs`.

Before this field existed the comparison was a hard-coded `0.45`, so a product
could not align the gate with its own VAD. The default preserves the previous
behaviour exactly; any other value is a deliberate change and invalidates
calibrated thresholds.

### `token_boost` — deprecated and ineffective

The field is retained for source and ABI compatibility, including its finite,
nonnegative validation and default `1.5`. Its value no longer participates in
search, retention or confidence arithmetic. Search scores accumulate acoustic
log-probability and retention/fuzzy-child costs; the terminal retention gate uses
`retention_log = terminal_score - terminal_acoustic`. Confidence remains
`exp(acoustic_score / depth)`.

The older implementation added `token_boost` once per depth, then subtracted
`token_boost * depth` at the retention gate. Although these offsets cancel in
real arithmetic, they do not cancel exactly in floating point: a large finite
boost can round away fuzzy-child costs, and a maximum finite boost can overflow
and bypass the retention gate through NaN. Removing this unused offset makes
all accepted boost values use identical decoder arithmetic.

The historical `0.0` to `10.0` sweep over 12 positive clips found byte-identical
output, but that limited result never established equivalence for all inputs or
all finite boosts. The regression suite now checks ordinary and extreme values
(`0.0`, `1.5`, `10.0`, `1e9`, `FLT_MAX`), retention rejection and identical search
states/confidence on deterministic replay. These are software contract tests,
not acoustic qualification. Removing an offset can still change rounding and
path selection near decision boundaries versus an older source revision,
including at default boost `1.5`; qualification must bind the new source and
rerun affected replay/calibration and product gates. Existing thresholds and
shipping defaults are unchanged, and older evidence must retain its original
source identity.

Do not use this field to tune recall. Use `state_retention`,
`min_speech_dbfs`, or per-keyword `threshold`.

## L3 per-keyword fields

These are fields of `kws_keyword_t`, emitted by `tools/compile_keywords.py` into a
KWKP v3 pack and revalidated by `src/keyword_pack.c` on load.

| Field | Type | Default | Range | Unit |
| --- | --- | --- | --- | --- |
| `threshold` | float | none — mandatory | `(0.0, 1.0)` exclusive | probability |
| `min_trailing_blanks` | uint8 | `0` | `[0, 8]` | frames |
| `priority` | uint8 | `0` | `[0, 15]` | rank |
| `grace_frames` | uint8 | `0` | `[0, 32]` | frames |

`threshold` has no default: every keyword must state its acceptance threshold
explicitly in the TSV, so an uncalibrated keyword cannot slip through.

`min_trailing_blanks` is the number of consecutive blank-dominant frames
required after a terminal qualifies and before it may fire. It applies to every
prefix policy. The qualifying frame itself does not count as a trailing blank,
even if blank is dominant while a fuzzy terminal qualifies. A nonblank frame
restarts the consecutive-blank count. The default is
`policy_defaults.longest.min_trailing_blanks = 1` for `longest` and
`policy_defaults.grace.grace_frames = 3` for `grace`; the compiler applies those
only when the TSV leaves the field empty and the value would otherwise be `0`.

`priority` is the arbitration rank; higher values win at the priority comparison.
Immediate candidates compare priority, trie depth, then confidence. Pending
`longest`/`grace` candidates compare depth, priority, then confidence. Both resolve
an exact tie by the smaller unsigned 32-bit keyword ID, including boundary IDs
`0` and `UINT32_MAX`, independently of TSV/pack order. Ranks above 15 are rejected,
because the pack record is a single byte and unbounded ranks are a configuration
error rather than a capability.

## Prefix policies

| Policy | Value | Behaviour |
| --- | --- | --- |
| `immediate` | 0 | fire as soon as the terminal confidence clears `threshold` and `min_trailing_blanks` |
| `longest` | 1 | hold the terminal and keep extending while longer keywords still match; requires `min_trailing_blanks >= 1` |
| `grace` | 2 | hold the terminal for `grace_frames` after it first qualifies; requires `grace_frames >= 1` |

`immediate` with zero trailing blanks keeps same-frame emission. A nonzero
blank requirement uses fixed-size per-keyword waiting state, separate from
`longest`/`grace` arbitration. A nonblank frame cancels the previous immediate
wait; a currently qualifying terminal can start a fresh wait. A dead or
retention-exhausted terminal also cancels it, so unrelated speech followed by
blanks cannot release an old immediate detection. New qualification requires
speech activity; an already-qualified terminal may finish its blank wait on
inactive frames, subject to the existing utterance-boundary reset. Ready
immediate terminals compete by priority, then depth, then confidence, then the
smaller unsigned keyword ID. Reset,
discontinuity and keyword replacement clear these waits.

Use `longest` when one wake word is a prefix of another (for example `小窝`
against `小窝小窝`). Use `grace` when the longer variant may arrive late.

## Change checklist

1. Edit `configs/parameter-contract.json` and rerun `python3 tools/gen_parameter_limits.py
   configs/parameter-contract.json build/generated`.
2. If the parameter is `effective` and `invalidates_thresholds`, re-run
   threshold calibration and update `threshold_calibration` in
   `configs/shipping.xiaowo.json`, including its `parameter_contract_sha256`.
3. Update `parameter_contract.sha256` and
   `threshold_calibration.required_parameter_contract_sha256` in
   `configs/shipping.xiaowo.json`. Preserve the historical calibration digest
   unless a new source-bound calibration has actually run. Even a documentation-
   only contract edit changes its byte identity; identify it as metadata
   maintenance, retain `recalibration_required`, and never infer new qualification.
4. Run `python3 tests/test_parameter_contract.py`, then `cmake --build` and
   `ctest` to confirm the C side accepts the new range.
5. If a frontend L1 constant changed, re-run the frontend parity test
   (`python3 tests/test_frontend_parity.py ./build/kws_feature_dump`) because
   feature bytes may differ. Decoder-only L1 changes instead require decoder
   replay/regression evidence against the affected event paths.

## What is not covered

The contract fixes legal ranges, not product performance. Two evidence classes
remain open and are tracked in `configs/shipping.xiaowo.json`:
`real-human-final-afe-acoustic-qualification` and
`physical-target-board-performance-and-soak`. Hosted x86 timing and synthetic
corpus results must never be promoted into either.
