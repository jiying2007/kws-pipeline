# Saved-window independent direct-DFT diagnostic

Status: **CONVERGED_HIGH_PRECISION_DIAGNOSTIC_ONLY**. This is a numerical research observation, not a certified oracle, historical reconstruction or model qualification.

## Fixed scope and result

One bounded process evaluated an explicit direct DFT of the existing identical 512-point binary32 windowed input. The input was decoded as its exact stored binary values; no waveform, window or frontend was regenerated. The program used mpmath 1.3.0 with the pure Python backend, fixed 60 and 100 decimal digits, 32 fixed standard tests and exact rational DC/Nyquist anchors. No FFT library, original C/Torch frontend, model, training, power, mel or log stage was run. No retry occurred.

All specified tests, convergence checks and rational anchors passed. Under the preregistered squared-error margin, C was closer in 147 bins, Torch in 106, and 4 were indeterminate. Aggregate complex L2 error was approximately 3.64444e-5 for C and 4.12322e-5 for Torch; the recorded global direction was C_CLOSER. This does not mean C is closer in every bin.

## Evidence and interpretation limits

The 60/100-digit references use the **same implementation and arithmetic library**. Agreement across precision and passing standard tests support numerical stability; they are not a rigorous interval proof, correct-rounding proof or independent certification of mathematical truth. The full 60-digit DFT array was **not retained**. The saved result retains convergence differences, stable-direction decisions and summaries, so complete separate replay of every convergence decision would require a new numerical operation. No such replay was performed for this publication.

The conclusion applies only to the saved windowed input and complex FFT comparison. It cannot establish downstream logfbank accuracy, whole-chain correctness, production benefit or model eligibility. The preceding new Torch endpoint did not match its historical endpoint, so historical attribution remains stopped. Historical failures, epsilon, tolerances and acceptance gates are unchanged. B remains EXPORTED_NOT_NUMERICALLY_VALIDATED.

## Resource scope

Total wall time was approximately 1.074 seconds, within the 300-second total and 30-second single-command caps. Per-child address space had a 1 GiB hard cap; sampled process-group RSS peaked at 17256448 bytes. RSS is a sampling observation, not a guaranteed hard peak, and no hard aggregate CPU claim is made. Retained outputs had an 8 MiB cap.

This publication contains only anonymous synthetic-input metadata, source identities and aggregate diagnostic results. It contains no per-bin arrays, private filesystem paths, private Library identifiers, recordings, model parameters or credentials. Publishing it must not rerun the DFT or any diagnostic; ordinary existing public CI regressions remain allowed.
