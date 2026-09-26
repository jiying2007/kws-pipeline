# Architecture

`kws-pipeline` separates the offline/control plane from the bounded always-on C runtime.

## Real-time data path

```text
mono PCM16 @ 16 kHz
 -> fixed 25-ms Hann / 20-ms hop / FFT512 / mel bank
 -> model-selected frontend: logmel or pcen-lite
 -> 32-d feature normalization
 -> int8-weight tiny recurrent acoustic model
 -> blank + pinyin token logits
 -> dominant-token CTC admission
 -> shared-prefix keyword trie
 -> prefix-policy arbitration
 -> speech / threshold / refractory gates
 -> detection {keyword_id, confidence, end_sample}
```

The model ABI fixes 400-sample analysis frames and 320-sample acoustic hops. `kws_engine_accept_pcm16()` accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples per call, so one call can produce at most one acoustic step and one detection. A 10-ms/160-sample upstream audio block is therefore safe without another resampler.

The runtime is C11 + libm, owns no worker thread, performs no heap allocation or filesystem I/O and does no text/pinyin conversion. The caller owns one aligned arena and keeps the model blob alive for the engine lifetime. That dependency surface is enforced rather than documented: `tools/check_runtime_purity.py` asserts that every undefined symbol of `libkws_pipeline.a` stays inside an explicit allowlist.

## Artifact ABIs

The two deployable binary contracts are intentionally versioned separately:

- **`KWSP` model ABI v2**, 72-byte canonical header. It binds feature/hidden/vocabulary dimensions, frontend kind, 16-kHz/400/320 geometry, quantization scales, 64-bit vocabulary fingerprint and tensor offsets.
- **`KWKP` keyword-pack ABI v3**, 24-byte header + 48-byte records. Each record binds keyword id, threshold, token sequence, `min_trailing_blanks`, priority, prefix policy and grace frames to the same vocabulary fingerprint.

A pack with a mismatched vocabulary cannot attach to a model. Unsupported versions, non-canonical sizes/layouts, duplicate ids/paths, invalid token ids, NaN/Inf values or invalid prefix-policy metadata are rejected before the active trie is replaced.

## Frontend contract

`training/frontend_spec.py` is the dependency-free reference for both supported frontends:

- `frontend_kind=0`: `logmel`;
- `frontend_kind=1`: `pcen-lite`.

Both share sample rate, frame/hop, window, FFT and mel geometry. PCEN-lite keeps a fixed-size per-mel smoothing state and applies bounded gain normalization before the common feature normalization. No dynamic allocation is introduced.

The model header selects the frontend. Training checkpoints and export provenance record the same frontend identity and frontend-spec version. `qualification_manifest.py` carries runtime frontend identity into the release manifest and `qualification_gate.py` requires it to match model lineage. A PCEN model therefore cannot silently be released with a logmel runtime configuration, or vice versa.

CI runs the actual C frontend through `kws_feature_dump` and compares both modes against the reference implementation.

## Open-token KWS

The acoustic network learns reusable pinyin-token acoustics. Configured wake phrases are bounded token paths, not dedicated binary classifiers. A normal L0 phrase change can therefore update `.kwk` only when the acoustic model already separates the required tokens.

The default 32-feature / 48-hidden / ~420-token geometry is about 1.2 MMAC/s and about 26 KB of model weights+biases. These are design calculations, not physical target-board measurements.

## Decoder state and CTC admission

The decoder is a bounded shared-prefix Trie with a lightweight Viterbi-style scorer; it is not a general CTC prefix-beam search.

Each frame admits at most one nonblank structural label: the highest-logit nonblank token only when it also beats blank. Non-dominant token posteriors still contribute to normalization/confidence but cannot fabricate Trie transitions.

Each non-root Trie node retains two independent states:

- `score`: best path whose latest token has not subsequently observed a blank;
- `blank_score`: best path after at least one blank-dominant frame since the latest token.

A repeated child token can advance only from `blank_score`. Non-repeated children may advance from the better state. This preserves the structural CTC requirement for adjacent identical labels without carrying a vocabulary-wide beam.

## Shared-prefix arbitration

`KWKP` v3 makes prefix conflicts explicit instead of depending on keyword order.

Per keyword:

- `immediate`: qualifying terminal can emit in the current step. If several immediate terminals compete, higher priority wins, then deeper path, then confidence.
- `longest`: terminal is offered to bounded pending state and releases only after `min_trailing_blanks`; a deeper shared-prefix candidate can replace it before release.
- `grace`: terminal is held for at least `grace_frames` and its trailing-blank condition; deeper/higher-priority pending candidates can replace it during the grace window.

The pending state is fixed-size. Emission resets decoder state, preserving one-detection-per-call semantics.

## Domain-aware offline loop

`training/iterate_domain.py` is an offline orchestration layer around the same deployable runtime:

```text
base synthetic examples
 -> acoustic scene renderer
 -> split leakage audit
 -> frontend candidate fit/train
 -> quantize/export KWSP
 -> threshold search + KWKP v3 compile
 -> real C runtime on calibration/test
 -> domain metrics + worst-domain objective
 -> adaptive distance curriculum
 -> candidate selection
 -> untouched qualification render/evaluation
 -> frozen best bundle
```

`configs/training/xiaowo.domain.json` models nominal near/mid/far distance bands (0.3–1 m, 1–3 m, 3–5 m), azimuth, RT60, SNR, noise profile and optional playback/AEC residual. The AFE layer can be the repository proxy or an external command wrapper around the shipping audio pipeline.

Training domains are weighted stochastic samples and may be reweighted by worst-domain curriculum. Evaluation positives are deterministically rotated `far -> mid -> near`; this guarantees that a far-field FRR gate is backed by an actual far positive rather than by random domain presence/absence. Negative scenes remain independently sampled.

The dependency-free domain prototype has a 98.5% post-quantization token-core validation floor before it participates. That internal floor is not the product metric: complete rendered utterances still run through the real C runtime for calibration/test/qualification and domain FAR/FRR/latency scoring.

The older generic synthetic prototype uses its own stricter 99.5% token-fit floor. These are two distinct offline smoke contracts, not shipping acoustic thresholds.

## Long continuous background path

`kws_raw_stream` feeds raw PCM16 blocks through the same runtime without WAV/clip resets. `eval/long_far_stream.py` uses it to exercise long continuous negative streams and record false accepts. The nightly workflow is a hosted/synthetic regression watch, not evidence for a production FAR/hour claim.

## Provenance and qualification

Training requires the actual token vocabulary. Checkpoints retain vocabulary fingerprint/token hash, manifest hashes, frontend identity/spec, seed and optimizer settings. Export refuses incompatible vocabulary/frontend metadata and emits deterministic model provenance with quantization diagnostics.

Shipping qualification then re-hashes the concrete checkpoint, training tokens, complete manifest multiset, exported model, pack, release tokens/config, evaluation artifacts and target-board artifacts. Runtime frontend identity must equal model-lineage frontend identity. The SKU policy is applied only after these byte-complete consistency checks.

## Ownership and hard bounds

- model blob: caller-owned, read-only, stable address;
- engine arena: caller-owned, exclusive, aligned;
- parsed keyword pack: needed only until `kws_engine_set_keyword_pack()` returns;
- engine: single-thread owner; serialize externally;
- no global mutable state, runtime plugins, heap or locks.

Current limits are 16 keywords, 16 tokens per keyword, 40 features, 64 recurrent units and 512 acoustic tokens. Model ABI v2 fixes the acoustic geometry; keyword-pack ABI v3 fixes the prefix-policy record contract.

Parser attack-surface hardening is covered by deterministic tests and Clang libFuzzer/ASan/UBSan smoke for `.kwm`/`.kwk`. It does not replace signing/authentication of production update artifacts.

## Parameter governance

Every tunable lives in exactly one place: `configs/parameter-contract.json`. The contract declares, per parameter, its layer, type, default, unit and inclusive or exclusive bounds. Two consumers read it and cannot drift apart:

- `src/*.c` include `build/generated/kws_parameter_limits.h`, produced from the contract at CMake configure time by `tools/gen_parameter_limits.py`. A bound is therefore a compile-time constant, and `kws_engine_init()`/`kws_keyword_pack_open()` reject anything outside it with `KWS_EINVAL`/`KWS_EFORMAT`.
- the python tools (`tools/compile_keywords.py`, `tools/qualification_common.py`) load the same JSON directly, so the compiler and the qualification recorder apply the identical range.

`tests/test_parameter_contract.py` asserts that the generated header round-trips back to the contract values, that regeneration is byte-identical, and that an inconsistent contract fails generation instead of shipping.

### Layers

| Layer | Meaning | Where it lives |
| --- | --- | --- |
| L0 | model-bound | inside the `.kwm` blob; a product cannot change it |
| L1 | firmware constant | compile-time macro; a change needs a rebuild and a regression |
| L2 | product config | a `kws_config_t` field; a change invalidates calibrated thresholds |
| L3 | field policy | a per-keyword KWKP v3 record field; a change needs recalibration |

### L1 constant conversion table

The decoder and the frontend work in log/nat space. These are the human-unit equivalents at 16 kHz with a 400-sample frame and a 320-sample hop (20 ms). The values come from `configs/parameter-contract.json`.

| Constant | Value | Human unit |
| --- | --- | --- |
| `KWS_SILENCE_RETENTION_LOG` | `-0.3566749439` | `ln(0.70)` — a live prefix keeps 70% of its score per non-speech frame |
| `KWS_MIN_PATH_RETENTION_LOG` | `-16.0` nat | abandonment budget: about 45 non-speech frames (0.90 s) or 259 speech frames (5.17 s) |
| `KWS_ROOT_START_LOGIT_MARGIN` | `0.5` nat | a near-tied competing keyword root may start a path within a 1.65x probability ratio |
| `KWS_FUZZY_CHILD_RETENTION_COST_LOG` | `-8.25` nat | 51.6% of the abandonment budget, so two fuzzy child advances can never complete a keyword |
| `KWS_PCEN_SMOOTHING` | `0.025` | PCEN-lite smoothing ratio |
| `KWS_PCEN_ALPHA` | `0.98` | PCEN-lite exponent |
| `KWS_PCEN_DELTA` | `2.0` | PCEN-lite bias under the square root |
| `KWS_PCEN_EPSILON` | `1e-06` | PCEN-lite smoothing regulariser |
| `KWS_LOGMEL_COMPRESSION` | `32.0` | `log1p(32 * energy)` |
| `KWS_FEATURE_NORMALIZATION` | `0.25` | scale applied to the mean-normalised feature vector |
| `KWS_MEL_LOW_HZ` / `KWS_MEL_HIGH_HZ` | `80` / `7600` Hz | mel filterbank edges |

The 512-point FFT and its bin mapping are not tunable: they are fixed by the frontend contract that the training-side feature spec implements, and `tests/test_frontend_parity.py` holds the two implementations together. The mapping follows this project's convention `bin = floor((N + 1) * f / fs)`, with `N = 512` bins for the real-input transform and `fs = 16000 Hz`, clamped to `N / 2 = 256`. The `N + 1` is deliberate, not an off-by-one: changing the convention shifts every mel edge and invalidates every calibrated threshold.

## Confidence is an approximation

`kws_detection_t.confidence` is computed with `expf(acoustic_score / keyword_depth)` in `src/decoder.c`. The per-frame log probabilities feeding `acoustic_score` use `approx_logsumexp()`, whose normalization uses the bounded `fast_exp_nonpos()` approximation. Confidence therefore is not an independently calibrated probability. Re-measure an absolute acceptance threshold with the exact runtime and model that will ship.

`docs/RUNTIME_CONFIG.md` is the operator-facing companion: what each L2/L3 field means, how to tune it, and what a change invalidates.
