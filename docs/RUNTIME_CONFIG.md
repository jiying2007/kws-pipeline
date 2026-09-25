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
  `parameter_contract.sha256` and `threshold_calibration.parameter_contract_sha256`
  fields. Editing the contract without revisiting the shipping contract fails
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

## L2 runtime parameters

These are the fields of `kws_config_t`. `kws_default_config()` returns exactly the
contract defaults below.

| Field | Type | Default | Range | Unit | Effective | Invalidates thresholds |
| --- | --- | --- | --- | --- | --- | --- |
| `min_speech_dbfs` | float | `-55.0` | `[-120.0, 0.0]` | dBFS | yes | yes |
| `token_boost` | float | `1.5` | `[0.0, +inf)` | nat | **no — deprecated** | no |
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

The field is retained for source and ABI compatibility. It is added exactly once
per trie depth to the search score, so it is a constant offset; the emitted
confidence is `exp(acoustic_score / depth)` and the retention gate computes
`retention_log = terminal_score - terminal_acoustic - token_boost * depth`, which
cancels the offset exactly.

Measured evidence: sweeping `token_boost` from `0.0` to `10.0` over a 12-clip
positive set produced byte-identical output for every clip, while the control
parameters (`state_retention` `0.94 -> 0.50`, `min_speech_dbfs` `-55 -> -20`)
both changed the results. The contract marks the parameter `"effective": false`
and the generated header emits `KWS_PARAM_TOKEN_BOOST_VALID` but no behavioural
claim.

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

`min_trailing_blanks` is the number of blank-dominant frames required before a
held terminal may fire. It applies to every prefix policy. The default is
`policy_defaults.longest.min_trailing_blanks = 1` for `longest` and
`policy_defaults.grace.grace_frames = 3` for `grace`; the compiler applies those
only when the TSV leaves the field empty and the value would otherwise be `0`.

`priority` is the arbitration rank. Higher wins. Ties are resolved by trie depth
and then by confidence, so duplicate ranks remain deterministic. Ranks above 15
are rejected, because the pack record is a single byte and unbounded ranks are a
configuration error rather than a capability.

## Prefix policies

| Policy | Value | Behaviour |
| --- | --- | --- |
| `immediate` | 0 | fire as soon as the terminal confidence clears `threshold` and `min_trailing_blanks` |
| `longest` | 1 | hold the terminal and keep extending while longer keywords still match; requires `min_trailing_blanks >= 1` |
| `grace` | 2 | hold the terminal for `grace_frames` after it first qualifies; requires `grace_frames >= 1` |

Use `longest` when one wake word is a prefix of another (for example `小窝`
against `小窝小窝`). Use `grace` when the longer variant may arrive late.

## Change checklist

1. Edit `configs/parameter-contract.json` and rerun `python3 tools/gen_parameter_limits.py
   configs/parameter-contract.json build/generated`.
2. If the parameter is `effective` and `invalidates_thresholds`, re-run
   threshold calibration and update `threshold_calibration` in
   `configs/shipping.xiaowo.json`, including its `parameter_contract_sha256`.
3. Update `parameter_contract.sha256` in `configs/shipping.xiaowo.json`.
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
