# Known FAR failures: original-prefix diagnostic

This is a diagnostic of frozen-model synthetic regression run [38482297930](https://github.com/jiying2007/kws-pipeline/actions/runs/38482297930), not a new qualification or a passed FAR run. The original 2 FA / 8 h result remains failed. No model, threshold, ASR label or shipping approval was changed.

## Reconstructed inputs and results

The original source was `1c8d9c049170e9547b39807d77498e10c19712bf`. The runtime and generator source bodies used from `2e4cc45be0714df060ee8f4865d96341f8cd1677` were compared against its immutable Git tree. All 240 restored domain rows matched except local paths; all 224 negative WAV hashes and original source-WAV hashes matched. This is the existing deterministic tone/proxy generator, not human speech.

Keeping the original 7200-second scheduling horizon is essential: shortening `--seconds` changes mandatory injection placement. Only the output loop was stopped at 7194 seconds (seed 1103) and 2375 seconds (seed 3301), after each original event plus context. Every injection entry before the cutoff matched the original timeline except filesystem paths (339 and 120 entries). PCM over each original four-second capture interval matched exactly. The 9569 seconds of prefixes are overlapping diagnostic replays, not new independent exposure.

Both keyword 2 events occurred at the original timestamps: 7191.345 and 2372.645 seconds. The first direct-GCC confidence was 0.603047 versus the original 0.603046; the second was 0.962665 in both. We do not claim bitwise equivalence with the original hosted compiler. In separate four-second cold starts, seed 1103 did not fire, while seed 3301 fired at 2.005 seconds with confidence 0.993992. A cold clip is therefore not a substitute for original continuous state.

## Saved state, not a guessed cause

The two compressed traces retain 1051 frames around the original events, with current logits/VAD and decoder state before/after each input block. Node tuples are token,parent,depth,terminal-keyword-index,score,blank-score,acoustic-score,blank-acoustic-score. The event frame's after-state is reset by the normal detection path. Existing debug accessors are read-only; parsed detection objects matched the local uninstrumented run. JSON key ordering differs, so byte-equality was correctly replaced by parsed-object comparison after an initial diagnostic assertion failed.

At the seed 1103 event the dominant-token sequence over the previous two seconds is blank→xiao3→wo1, while a deeper xiao3/wo1/xiao3 prefix already exists in decoder state. At seed 3301 the model's dominant sequence includes xiao3→wo1→blank→xiao3→wo1. These observations distinguish a search-path question from acoustic confusability; they do not establish a human transcript or prove an optimal fix. Both events use immediate/zero-trailing-blanks, so the separate longest/grace pending-expiry repair does not explain or resolve them.

Do not promote a higher threshold based on these two negatives. Any search/retention change needs paired positive-FRR and continuous-negative evaluation, frozen threshold selection and independent data. Real-human/final-AFE and SSC305 gates remain open.

## Evidence and reproduction boundary

`result.json` binds source, model, pack, compilers, restored prefix hashes, events and compressed/uncompressed trace hashes. The C diagnostic source is retained exactly as executed; set `KWS_TRACE_START_SAMPLE` before invoking it. It links the original core and tool I/O sources, calls the existing read-only debug API, writes normal detections to stdout and bounded diagnostic JSON to stderr. It is not an installed SDK component.

Reconstruction uses the original `training/render_domains.py` with `configs/nightly.xiaowo-frozen-model.json`, all original negative rows in original order, and `eval/long_far_stream.py` with seeds 1103/3301, duration 7200, injection rate 2/min, one mandatory injection per clip and gain 0.90. The original API artifact 11858837112 (SHA256 `1c1a7921f2114fad9df4bd1298f32331c06886bd80d5bda888a916776b6e058a`) is the timeline/domain/capture authority; no missing original body was fabricated. Original full-stream PCM was not retained by that run, so exactness is proven for the restored source hashes, schedule and saved capture intervals, not by comparing an unavailable original full-stream hash.

Local direct-GCC diagnostics do not replace the repository's CMake/hosted build identity, hardware measurements, full regression or release gates. No real-user audio or credentials are included here.

The exact original domain-index and two injection timelines are retained as compressed metadata, with compressed and original byte hashes in `result.json`. They contain public synthetic provenance and original hosted paths, not waveform bodies. This avoids relying solely on an expiring Actions artifact for future input/schedule reconstruction.
