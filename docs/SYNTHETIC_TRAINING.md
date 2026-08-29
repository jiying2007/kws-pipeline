# Synthetic and domain-aware self-training loops

These workflows close the **software, data and model-control loop before complete real product data is available**. They deliberately separate synthetic evidence from shipping acoustic evidence.

A successful synthetic loop can prove deterministic split isolation, model fitting/export, keyword-pack compilation, C-runtime evaluation, threshold calibration, failure replay, candidate selection and held-out synthetic qualification. It cannot prove Mandarin human-speech quality, real-room 3–5 m performance or physical Cortex-A32 performance.

Repository issue #2 remains the real-evidence gate.

## Generic synthetic loop

```bash
python3 training/iterate.py \
  --config configs/training/xiaowo.synthetic.json \
  --runner build/kws_wav
```

The four data pools are `train`, `calibration`, `test` and untouched `qualification`. `training/audit_dataset.py` checks decoded PCM SHA256 isolation.

The dependency-free generic prototype is a real deterministic softmax learning path. It extracts train-only token/background frames, trains and int8-quantizes the acoustic head, then requires **99.5% post-quantization held-out token-fit accuracy** before the candidate can enter end-to-end synthetic qualification. Complete candidates still run through the C runtime.

The alternative `torch_ctc` backend uses `train_ctc.py`/`export_model.py` and supports head-only replay in later rounds.

## Domain-aware multi-frontend loop

Build `kws_wav`, then run:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The default domain config models nominal:

- near: 0.3–1.0 m;
- mid: 1.0–3.0 m;
- far: 3.0–5.0 m;
- azimuth from front/side/back directions;
- RT60 0.15–0.80 s;
- SNR -5 to 30 dB;
- white/fan/motor/media noise;
- optional local playback and AEC residual;
- 60-mm nominal dual-mic spacing;
- proxy AFE or a command adapter for an external shipping audio pipeline.

These are simulation parameters, not measured product coverage.

### Split semantics

The domain renderer first creates independent base splits and then renders acoustic scenes. Decoded PCM leakage is audited across all four splits.

Training scenes remain weighted stochastic samples and can be reweighted by adaptive curriculum. Evaluation positives are deterministic:

```text
far -> mid -> near -> far -> ...
```

The first positive in calibration, test and qualification is therefore always far. This prevents a far-field gate from passing/failing only because weighted random sampling happened to omit the far positive class. The domain summary records overall and per-split distance histograms plus the evaluation positive order.

### Frontend A/B

The domain loop can evaluate both model-bound frontends:

- `logmel` (`frontend_kind=0`);
- `pcen-lite` (`frontend_kind=1`).

Every candidate receives its own model, calibration thresholds, KWKP v3 pack, calibration/test metrics and domain report. Candidate ranking is based on the same gate-aware objective; qualification remains untouched until the best candidate is frozen.

### Dependency-free domain prototype

The hosted domain prototype uses deterministic synthetic token scenes. For logmel it supervises energetic token frames. PCEN-lite is stateful, so the prototype uses one stable discriminative token-core frame per synthetic token scene for this **internal frame-classifier fit only**; transition frames are not mislabeled as blank.

The quantized domain prototype must achieve at least **98.5% token-core validation accuracy** before it can participate.

That 98.5% value is not a product FRR target. Every complete rendered calibration/test/qualification utterance—including all transition frames—still runs through the real C runtime, decoder and keyword pack. The final domain metrics therefore remain sequence/runtime metrics rather than frame-fit metrics.

### Adaptive curriculum

After a round, `domain_curriculum.py` converts worst-domain results into bounded distance-band weights. The next **training** render may emphasize weak distance bands. Calibration/test/qualification are not reweighted from their results.

### Evidence class

The domain loop emits `evidence_class: synthetic-domain-qualified` only when the frozen best candidate passes the synthetic qualification gates. Its manifest also lists explicit limitations:

- no real human speech;
- simulated acoustic scenes are not production far-field qualification;
- the shipping AFE must be evaluated through the command adapter or real recordings;
- physical target-board and independent human held-out evidence remain issue #2 gates.

## Long-FAR synthetic regression

`eval/long_far_stream.py` drives `kws_raw_stream` over long continuous negative PCM without resetting runtime state between clips. The nightly workflow uses this as a regression watch for state accumulation, false accepts and parser/runtime changes.

Do not convert hosted/generated hours into a product FAR claim. Production FAR requires long real recordings through the final microphones/enclosure/AEC/NS/AGC configuration.

## Failure replay

Calibration failures can feed the next learning round:

```text
false accept -> mine_hard_negatives.py -> empty-target replay
false reject -> mine_false_rejects.py -> positive replay
```

The final qualification pool is never a mining source. Once a held-out sample is deliberately fed back into training/calibration, it is no longer eligible as independent held-out evidence.

## Promotion to real evidence

Keep the same orchestration when product data arrives, but replace synthetic-only claims with real split identities:

1. real train data;
2. real calibration data for threshold/replay;
3. independent real test data for candidate comparison;
4. frozen real qualification-heldout data;
5. final `audio-pipeline` configuration;
6. physical target-board benchmark/resource evidence;
7. byte-complete `qualification_manifest.py` + `qualification_gate.py`.

Only that release path can close issue #2.
