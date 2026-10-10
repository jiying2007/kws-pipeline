# Domain-aware synthetic self-training

These workflows close the **software, data and model-control loop before complete real product data is available**. They deliberately separate synthetic evidence from shipping acoustic evidence.

A successful synthetic loop can prove deterministic split isolation, model fitting/export, keyword-pack compilation, C-runtime evaluation, threshold calibration, failure replay, candidate selection and held-out synthetic qualification. It cannot prove Mandarin human-speech quality, real-room 3–5 m performance or physical Cortex-A32 performance.

Repository issue #2 remains the real-evidence gate.

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

The domain renderer audits all four base splits before creating any acoustic
scene, even when a call renders only a subset. It then audits the rendered subset.
Both stages are mandatory and their report hashes are retained in
`domain-summary.json`. Different split seeds/noise cannot hide a shared original
waveform or a known shared source family, speaker, reference audio, session or
derivation family. The development and refinement callers recheck the rendered
manifest immediately before consumption.

Trainer TSV files keep their two-column format. Each rendered TSV has a
`<manifest>.lineage.json` sidecar containing the exact manifest SHA256 and full,
ordered source rows, including original labels and admission receipts. Auditing
checks row count, order, paths and target IDs, then rereads both the rendered WAV
and its pre-render source WAV to verify file and decoded-PCM identity. Source
metadata and receipts also survive in `domain-index.jsonl` and evaluation
references. Deleting, swapping or reusing a stale sidecar fails when
`training/audit_dataset.py --require-lineage` is used.

The standard-library helper
`audit_dataset.verify_manifest_lineage(manifest, audio_root=None)` returns the
full verified rows for training admission. It verifies the byte/manifest chain;
the training consumer must separately validate review history, actual labels,
reviewer/revision binding and permitted admission purpose.

A clean audit means no violation among the identities observed. Per-field
`identity_disjointness` and per-split `identity_coverage` explicitly report
missing historical speaker/reference/session/derivation evidence as `unknown`.
Provider names, synthesis engines and voice settings are not silently promoted
to person or recording identity. A known `source_id` explicitly supplied as
recording metadata remains an isolation requirement. These checks do not turn
synthetic fixtures or historical diagnostic inputs into training-admitted or
product-qualification evidence.

Only deliberately synthetic algorithm tests may pass
`--synthetic-contract-test-only` to `iterate_domain.py` or
`adversarial_refinement.py`. The flag is forwarded explicitly to each CTC
invocation, including repair; no config value or absent review receipt enables
it automatically. The resulting checkpoints and descendants are non-promotable.
Normal development/model-training runs retain reviewed admission by default.

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
