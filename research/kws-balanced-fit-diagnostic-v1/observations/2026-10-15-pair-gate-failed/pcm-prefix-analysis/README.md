# Pair PCM energy analysis, 2026-10-15

Read-only standard-library analysis of the two original PCM streams. WAV and PCM hashes both match the verified registry. No audio generated or resampled; no Torch, model, optimizer, ASR or TTS calls. The closed training budget is unchanged.

## Fixed method

PCM16 divided by 32768. Windows are 400 samples (25 ms), hop 320 samples (20 ms), complete windows only. Thresholds fixed before reading PCM: RMS >= 1e-4, and separately RMS >= 1% of each recording's maximum window RMS. Threshold crossings describe window intervals, not exact speech onset. Prefix summaries use exactly 1680 and 3200 samples (105 and 200 ms).

## Findings

- KW1 qwen06-05: first 105 ms and 200 ms are entirely exact zeros. Whole-clip RMS 0.05559346; peak 0.30914307. Both thresholds first crossed at window 23, 460–485 ms.
- KW2 qwen06-06: first 105 ms RMS 0.0000506952, peak 0.0001525879, integer range [-5,5], exact zeros 25.7143%. First 200 ms RMS 0.0000464454, peak 0.0003051758, integer range [-10,5], exact zeros 38.5%. Whole-clip RMS 0.07933634; peak 0.43618774. Absolute threshold first crossed at window 10, 200–225 ms; relative threshold at window 11, 220–245 ms.
- Combined with the already-saved step-300 raw argmax paths, all token transitions for both samples occur within the first five frontend windows (ending by 105 ms), before either fixed energy threshold is crossed. KW1 is exactly silent there; KW2 has very low amplitude, not exact silence.

## Interpretation and limits

This directly supports an early low-energy emission/alignment failure in the observed greedy paths. For KW1, the zero-input logmel frames are mathematically zero under the inspected frontend, and the causal recurrent model cannot use future audio to emit those initial labels. Therefore its complete greedy match on KW1 is not evidence that it recognized those syllables acoustically at their actual positions.

The analysis does not reveal actual spoken text or exact phonetic boundaries. It does not prove that all CTC probability mass is on the greedy path, that full logits are equal, or that later acoustic inputs have no effect. Target sequence loss is still improving; finite-budget nonconvergence and other mapping/optimization possibilities remain open.

The two description thresholds are not validated production VAD thresholds. Do not turn them into a product gate without separate evidence. No new experiment, budget, audio, labels, or model weights are created by this report.

Artifacts: summary.json contains compact statistics; pcm-energy.json retains every window and its true range; the original local implementation is retained separately; analyze_pcm_portable.py is the public reproduction CLI.

## Portable reproduction

Run `python analyze_pcm_portable.py --archive asr-11899008382.zip --output NEW_DIRECTORY`. Omit `--output` for a JSON bundle on stdout. The output directory must not already exist. No audio is extracted or generated.

Before opening ZIP contents, the script checks exactly 1,382,028 archive bytes and SHA256 f8bf70a0539d6ca7bd11dfed9229a087c0c7d27b943550f73136cd765f54ef0c. It reads only the two fixed WAV members with exact declared and actual size limits, checks their WAV/PCM hashes and geometry, and repeats the frozen arithmetic. It contains no workspace paths and reports no local environment information.

Actual recomputation produced byte-identical summary.json and pcm-energy.json. The CLI enforces the frozen report hashes; across a different Python/platform numeric implementation it fails closed if serialization or floating-point bytes differ, rather than claiming approximate equality. Stdout mode wraps the same verified report objects in a bundle. `test_portable.py --archive asr-11899008382.zip` checks exact bytes, stdout object equality, refusal to overwrite an existing directory, and rejection of wrong archive size/hash. All checks use explicit exceptions and remain active under normal Python, -O and -OO. Results for all three modes are retained in portable-test-result.json.

For public reproduction use analyze_pcm_portable.py.
