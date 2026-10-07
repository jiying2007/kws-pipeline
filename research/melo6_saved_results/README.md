# Melo6 saved-result join

This offline adapter reproduces the saved result from run
[37414956708](https://github.com/jiying2007/kws-pipeline/actions/runs/37414956708).
The original all-six exact-text gate **fails**: 2 both-match rows, 3 recognizer
disagreements, and 1 shared off-plan reading. All six remain quarantined; human
pronunciation and acoustic truth are UNKNOWN, CTC targets are null, and training
admission is zero. No homophone or diagnostic category changes that result.

The adapter validates saved source, generation, input, raw-result and receipt
bindings before joining intent. It imports no speech model, performs no
inference, playback, download or generation, and never rewrites an input.
Every input location is supplied explicitly. Use Python 3.10 or newer.

## Reproduce from public saved files

Use the source checkout at
`dd859c5b9feeaeedcff937cf94ac0858a9793ad1` for `SOURCE`, the data checkout at
`fdf579cc43622cb5a752e7f5f3c191a149b7956c` for `DATA`, and the separate result
directory `research/2026-10-06-melo6-blind-asr-results` for `RESULTS`. Those exact
source/data commits are linked in the result provenance. `JOIN` is the directory
containing this README in the source supplement checkout. Extract the reviewed
original result ZIP to an empty directory named by `ASR`; extraction is a local
file operation, not a new ASR run. Supply absolute paths for these shell variables.

```sh
python -B "$JOIN/join_saved.py" \
  --asr-root "$ASR" --source-root "$SOURCE" \
  --source-head dd859c5b9feeaeedcff937cf94ac0858a9793ad1 \
  --release-sha256 7601dcd9d10ae3d776565bdb2104c7a05b8969746c1310335206f6fb95ea1661 \
  --tts-root "$DATA/research/2026-10-05-melo6-source-screen/recovery" \
  --plan "$SOURCE/research/melo6_tts/execution-plan.json" \
  --tts-audit "$DATA/research/2026-10-05-melo6-source-screen/independent-recovery-audit-result.json" \
  --out "$NEW_OUTPUT"
```

`NEW_OUTPUT` must not exist and must be outside the supplied input roots. The
expected output is 24,485 bytes with SHA256
`7af81332103cf8174d8cbda9deda00e988027493e84f4643518993f20cdf280e`.
The source head is caller provenance; the adapter verifies the local matching
release/component bytes and does not query GitHub to establish the head.

For the four public saved-artifact regression tests, run `test_join_saved.py`
with the same seven input arguments, replace `--out` with
`--expected-comparison "$RESULTS/comparison-result.json"`, and append `-v`.
Tests reproduce the exact comparison, reject coherent aggregate/receipt text
contradiction, check freeze-before-intent order, and refuse output overwrite.
Temporary tampered copies are test inputs only; original saved files stay intact.

## Reused code and correction

`compare_melo6.py` and `comparison/audio_review.py` are unchanged from the
published TTS comparison: SHA256
`427ffce6fd334461d9c2c79662a96d7dbdd54624256ef11272e63e70720114de` and
`9c5c1e4d31238f3e5e2daa6a5d102638f916183daee56d63735a11a37a60e891`.
`saved_verifier.py` retains the six previously reviewed pure saved-verifier
functions. The join additionally requires recovered `raw_text` to equal the
unchanged completion receipt text. Valid freeze hashes cannot hide a
contradiction between those records.

The earlier development suite had nine tests, including temporary synthetic
records and a historical Qwen integration fixture. That historical suite and
its before/after correction evidence are separate from the four public tests
here; Qwen results are never relabeled as Melo results. This directory requires
only the stated public Melo source/data and existing result files.

The standalone `diagnostic.py` helper and its six tests describe reviewed
text-reading differences only. Run `python -B -m unittest discover -s "$JOIN"
-p test_diagnostic.py -v`. It is not imported by the comparator or runner and
does not provide an acceptance gate. See the result directory's `ANNEX.txt`.

All resource values are saved observations: sampled RSS/disk are not continuous
peaks or hard caps; per-process CPU limits are not aggregate enforcement; stage
times include model loading and checks. None measures A20/KWS performance.

This source supplement does not alter `melo6_asr`, the frozen release, workflow,
or runtime. It belongs on a separate branch based on the original source
commit, so publishing it does not initiate another scientific run.
