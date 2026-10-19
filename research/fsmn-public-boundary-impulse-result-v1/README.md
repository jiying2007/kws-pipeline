# Public boundary-impulse single-frame diagnostic

Status: **NEW_ENVIRONMENT_DIAGNOSTIC_ONLY**. The controlled execution completed successfully; this is not a numerical qualification or promotion.

The new C endpoint exactly matches the saved historical C endpoint. The new Torch endpoint does **not** match the saved historical Torch endpoint. Consequently, historical attribution is stopped. This run cannot establish the cause of the historical 36 fbank failures, replace the historical oracle, or qualify any model. Historical failure counts and thresholds remain unchanged; B remains EXPORTED_NOT_NUMERICALLY_VALIDATED.

## Scope and observations

Exactly one C frontend call and one Torch frontend call were made for the public synthetic frame with 399 zero PCM16 samples and a final +28000 sample. No model, training, repeated trial or tolerance adjustment was used. In this new environment, DC removal, preemphasis and windowed input are byte-identical; the first observed difference is in the complex FFT. The final logfbank differs beyond the frozen tolerance in 34 of 80 entries, with maximum absolute difference 0.009482383728027344. These observations are restricted to the new environment and do not establish a historical FFT root cause.

The per-stage rule is diagnostic localization only. Zero violations in power or linear mel do not establish exact equality or a new stage acceptance contract.

## Resource and reproducibility limits

Torch 2.11.0+cpu and torchaudio 2.11.0+cpu used two intra-op threads and one inter-op thread. The runtime reported MKL 2024.2 Build 20240605 and AVX512 capability. Historical runtime dispatch was not recovered.

Total wall time was 2.314 seconds. Each child had a 1 GiB hard address-space limit; RSS values are sampled process-group observations, not guaranteed hard peaks. The 8 MiB retained-output budget covers logs, products and receipts, not a hard aggregate quota on transient compiler scratch. No hard aggregate CPU guarantee is claimed.

This candidate contains only public synthetic-input metadata, public-source identities and diagnostic summaries. It contains no private filesystem paths, original archive bodies, recordings, model parameters or credentials. The supplied source summary stated that it had not been committed or published; this directory retains that summary for public research review.

## Subsequent analysis of saved values

In all 34 failing logfbank bins from this new environment, the saved linear mel values on both sides are above epsilon. Their maximum linear absolute difference is approximately 2.9726e-8, about 0.003 times the original linear tolerance, while their maximum relative difference is approximately 0.9437%. Applying float64 log to the saved mel values diagnostically predicts the same 34 failures; the residual between that prediction and the observed log differences is at most 8.3833e-7. This analysis used saved values and added no frontend or FFT calls.

This supports only an explanation of low-energy relative error amplified by log in this new environment's saved values. Float64 log is not a certified oracle and does not establish the cause of historical failures. Epsilon, tolerances, historical failure conclusions and model status remain unchanged. No per-bin arrays are included here.
