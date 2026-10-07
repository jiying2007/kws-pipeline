# Whole-N1 saved A20 diagnostic

This is an append-only source companion to `kws-data/research/2026-10-07-n1-wholeclip-leading`. Read that data appendix for measurements and limitations. It does not replace the earlier generation, human review, or heuristic records.

The collector and generated input fixture are exact acquisition source. The strict saved-trace validator is retained with its one-input manifest/geometry bindings. Runtime, model, frontend, decoder, thresholds, and the PR485 scorer are unchanged; `metadata/bindings.json` pins their existing public sources and identities. No model, native library, audio, or acquisition launcher is duplicated here. This package reproduces saved evidence, not a new acquisition.

Use Python's standard library with locally available, hash-verified public dependencies:

```
python -B verify_saved.py --data-dir PATH_TO_NEW_DATA_APPENDIX --n1-source-dir PATH_TO_ORIGINAL_N1_DATA --scorer-dir PATH_TO_PR485_EVAL
```

The original N1 data is pinned to kws-data commit `3856a7f07ad6fd571260103ce27469aabb6f4ea5`, directory `research/2026-10-07-single-k1-followed-wu`. Its PCM16 WAV and human-adjudication JSON are the only source files needed for verification. The unchanged PR485 `eval/score_events.py` and `eval/context_policy.py` are pinned to pipeline commit `6f2461ff11cddf0a1264f5e2a04282c40b5dc4ce`. These files must be supplied explicitly; the verifier never downloads them, loads a model, plays audio, compiles a collector, or runs native code.

Verification reconstructs the specified zero-context WAV in memory, checks the full original PCM remains unchanged, validates every saved callback/logit/event and integer geometry, and reproduces the retained observations after the explicitly declared raw-hash normalization. It runs the unchanged clip-presence scorer on N1 only. Its CONTEXT_UNVERIFIED result is retained because the panel contains a positive and no negative; no saved controls are inserted and no matched-panel claim is made.

Public raw bytes remove only the process ID in the first record. Resource projection removes host-global monotonic clocks and internal release inventories, while retaining measurements and caveats. The report removes only an absolute host command. Original and projected byte hashes and the exact transformations are declared in bindings. Existing component rights and source restrictions remain unchanged.
