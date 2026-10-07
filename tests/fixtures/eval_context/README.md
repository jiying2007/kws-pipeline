# Saved Melo context regressions

`sources.json` pins three public numeric logs in `jiying2007/kws-data` commit
`9307956e16ed93dc6c858611f3c35ae17768a011`. Raw scientific data stays in that
repository. Run `python3 tools/prepare_eval_context_fixtures.py` explicitly to
download and verify the three logs into ignored `build/eval-context-fixtures`.
Use `--source-dir DIR` instead to verify and copy already downloaded named logs
without network access. Preparation fails for unavailable or corrupt data.

Then run `python3 tests/test_eval.py`. The tests are fully offline, fail if any
fixture is missing or corrupt, and never silently skip the Melo regressions.
The public logs exclude the acquisition PID; no log numbers are changed during
preparation. `tests/test_eval.py` independently pins the complete SHA-256 of each
file and extracts only recording identity/duration and emitted callback events.
There are no audio or model artifacts, inference, decoder replay, TTS, ASR,
training or playback in this workflow.

| File | SHA-256 | Public source |
| --- | --- | --- |
| original.raw.jsonl | b93d9068eba66a8a744d9e706b0d16a763240df566942714029fa3f053b6f4ff | [Original short EOF](https://github.com/jiying2007/kws-data/blob/9307956e16ed93dc6c858611f3c35ae17768a011/research/2026-10-06-melo5-a20-diagnostic/original_A20.raw.jsonl) |
| positive300.raw.jsonl | 90196e0730db17fbae25d604d7e57d51dec4bb1b6ef1192b3c36d7f2fae29cf2 | [M1/M2 plus 4,800 zeros](https://github.com/jiying2007/kws-data/blob/9307956e16ed93dc6c858611f3c35ae17768a011/research/2026-10-06-melo2-tail-context/original_A20.raw.jsonl) |
| negative300.raw.jsonl | 743e97c86540a1917c938170fc21b511fc6a4b75e61f5877a13bce8cd226b98d | [M3/M4/M5 plus 4,800 zeros](https://github.com/jiying2007/kws-data/blob/9307956e16ed93dc6c858611f3c35ae17768a011/research/2026-10-06-melo3-negative-context/original_A20.raw.jsonl) |

Human words are M1 你好小窝 (K1), M2 小窝小窝 (K2), M3 你好小屋,
M4 小屋小屋 and M5 你好你好 (three nonwake clips). The source labels and
limitations are retained in the public
[human adjudication](https://github.com/jiying2007/kws-data/blob/9307956e16ed93dc6c858611f3c35ae17768a011/research/2026-10-06-melo6-human-adjudication/adjudication-result.json)
and [original manifest](https://github.com/jiying2007/kws-pipeline/blob/ea1c554a82c89797023365243a827883b94e0254/research/melo5-a20-diagnostic-v1/metadata/manifest.json).
M6 remains excluded. Human listening was on native 44.1-kHz audio after
hypothesis exposure; KWS used bound 16-kHz derivatives, which were not separately
heard. Zero-appended copies are changed contexts, not fresh sources.

The original condition has no appended input; both modified groups have exactly
4,800 digital-zero samples appended. All use 16 kHz, 4,800-sample feed chunks,
per-clip reset of frontend/model/decoder/clocks, processing retained final input,
no EOF padding and no EOF flush. These declarations come from the frozen
[original plan](https://github.com/jiying2007/kws-pipeline/blob/ea1c554a82c89797023365243a827883b94e0254/research/melo5-a20-diagnostic-v1/metadata/plan.json)
and context annexes, not from inferring silence or reset behavior from event
counts. Tests report input-availability coordinates only and create no keyword
time alignment. Original: 0/2 target-positive clips, 0/3 negative clips with
events. Matched 300 ms: 1/2 target-positive clips, 1/3 negatives with an event
(M3 falsely activates K1). No population rate, latency or qualification claim is
made by this regression.
