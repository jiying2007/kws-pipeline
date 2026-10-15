# KW1 supplementary run: factual observation v1

This is an additive, text-only observation archive of the one authorized supplementary run. It changes no frozen source pin, previous experiment, model, or call budget. No model was rerun to prepare or check this archive.

## Identity and retention

- Repository: `jiying2007/kws-pipeline`
- Source commit: [`1ef2b7e9b9d236c13f33b1bdee78d8ae754dc8fa`](https://github.com/jiying2007/kws-pipeline/tree/1ef2b7e9b9d236c13f33b1bdee78d8ae754dc8fa)
- [Run 38664865348](https://github.com/jiying2007/kws-pipeline/actions/runs/38664865348), attempt 1
- Artifact `11942325284`, `kw1-preset-supplement-20261014-v1-public`
- ZIP: 2,424,250 bytes; SHA-256 `e9cb187fbc0c0233f95e95efa7ca064359d7b00e3e6205041c7da06b3dedc5fe`
- Config SHA-256: `110494b07d3fb84c7b2537889e364ea9349b8f45e92774c3a033b398b6bbe916`
- Artifact created 2026-10-15 01:35:43 UTC, expires 2026-10-29 01:35:42 UTC (14-day retention). GitHub artifact access may require authentication. This directory does **not** provide a permanent public full-ZIP download. An out-of-band backup does not establish a permanent public download.

`report.json`, `ledger.json`, and `manifest.json` are exact original ZIP bytes, not reformatted copies. `source-retention.json` hashes all other local files and the separate observation-only CI workflow; it intentionally has no self-hash. Repository commit identity is the external trust anchor. Hashes alone do not authenticate a maliciously rewritten archive or establish model truth.

## Observed outcome

All 16 TTS slots and all 48 ASR slots completed, each with one attempt and no retry. All 16 audios passed the reported technical audio gate. Three-recognizer text agreement occurred for 6/16 slots, but only 3/16 matched the target and qualified as machine train-only candidates: `kw1s-05`, `kw1s-06`, `kw1s-07`. All three are Qwen17 / Uncle_Fu.

| Engine | Voice | Instruction | Slots | Target candidates |
|---|---|---|---:|---:|
| qwen06 | Uncle_Fu | empty | 2 | 0 |
| qwen06 | Eric | empty | 2 | 0 |
| qwen17 | Uncle_Fu | empty | 2 | 1 |
| qwen17 | Uncle_Fu | clear standard Mandarin / final first tone | 2 | 2 |
| qwen17 | Eric | empty | 2 | 0 |
| qwen17 | Eric | clear standard Mandarin / final first tone | 2 | 0 |
| qwen17 | Dylan | empty | 2 | 0 |
| qwen17 | Dylan | clear standard Mandarin / final first tone | 2 | 0 |

`observation.json` retains every slot, each of the 48 raw ASR results, seeds, generation settings, candidate flags, full call ledger, voice/engine/instruction aggregates, resource evidence, and source hashes. Instruction results are descriptive observations in this tiny sample; differing voices and unbalanced engine denominators preclude claims of model superiority or general instruction benefit.

## Resources and independent acceptance

The prior independent verification accepted all 235 ZIP files and exactly reproduced all 16 derived PCM payloads byte-for-byte with the pinned NumPy/SciPy versions. Its numeric rows and explicit trust limitations are recorded in `observation.json`; they are an attestation of that completed check, not checks performed by this directory's stdlib script. Admission workflow/history predicates were inferred from successful pinned admission code at that step, not claimed as directly observed original API snapshots.

All 10 recorded stages completed. Nine worker containers exited zero without an OOM flag; the image preparation stage is not a worker container. Four owned runtimes were removed. Their 17,792,079,998 logical regular-file bytes are not measured reclaimed filesystem capacity. Stage wall times sum to approximately 975.642 seconds, not whole-job duration. Memory limits (12 GiB) are configuration, not measured usage: peak process RSS and cgroup memory were not sampled and remain unknown. Full per-stage wall times, host free-space/memory observations, limits, and cleanup records are retained in the original report and observation. `resources.model_calls_made=0` describes offline recovery/report generation; actual campaign consumption is 16 TTS plus 48 ASR calls.

## Limits and next scientific gates

The existing 13 plus these 3 candidates yield 16 machine train-only candidates (KW1=5, KW2=11). Common KW1/KW2 voice labels now include Serena, Uncle_Fu, and Vivian. This is a necessary three-group coverage condition, not proof of independent talkers or permission to assign calibration, selection, and test roles to these three labels.

`training_admitted=false`, `shipping=false`, and `BLOCKED_NO_CALIBRATION_SELECTION_GOLD` remain unchanged. No human gold, independent calibration/selection gold, role assignment, or fresh held-out evaluation is established. The earlier 0/13 KWS result is not an evaluation of the new 16-candidate pool. The present 3/16 is neither KWS recall nor FAR.

A future stage would first need a frozen role assignment and leakage-safe split contract, independent calibration/selection data, real-scenario evaluation, and on-device resource measurements. These are requirements, not authorization for new work or model-call budget. This archive does not claim a production-ready closed loop or that human labeling is the only remaining obstacle.

## Reproduce the text checks

From the repository root:

```sh
python3 -I -S -B research/kw1-preset-supplement-observation-v1/verify_observation.py
python3 -I -S -B research/kw1-preset-supplement-observation-v1/test_verify_observation.py
```

Repeat both with `-O` and `-OO`. Checks use only Python stdlib, no network, no model imports, no subprocesses, and no writes by the verifier. Tamper tests use temporary copies. The separate path-filtered CI runs all three interpreter modes and fails closed on any error; frozen CI and source bindings remain untouched.

If the exact original ZIP is separately available, add `--zip /path/to/artifact.zip` to the verifier. That additionally checks its digest, complete inventory and member hashes; it still does **not** recompute audio numerics or authenticate the original run. Reproducing numeric audio derivation requires the ZIP plus the pinned scientific dependencies and original verification procedure. Without the ZIP, these files allow only offline text consistency checks and inspection of the recorded independent acceptance.
