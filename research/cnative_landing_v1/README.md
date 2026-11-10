# C-native complete-keyword metric correction

This source integrates the saved-output metric repair delivered on 2026-11-10.
It does not run training, ASR/TTS, or promote any model. Original experiment
archives and default runtime remain unchanged.

## Correction to the 2026-11-09 causal DS-TCN screen

The archived `exact_recall` was computed from `bool(greedy_local_ids)`, not from
a complete target sequence. The previously quoted 60%/70% median complete-word
recall is withdrawn. The corrected complete-keyword containment counts on the
same 34 exposed weak synthetic development recordings are:

| Seed | K1 true positives / positives | K2 true positives / positives |
| --- | --- | --- |
| 20261109 | 0/10 | 0/10 |
| 20261110 | 1/10 | 0/10 |
| 20261111 | 0/10 | 1/10 |

Whole-collapsed-sequence equality gives K1 0/10 for all three seeds and K2
0/10, 0/10, 1/10. Complete-keyword false-positive counts are 0/24 per keyword
and seed, but zero errors with poor recall is not a useful model selection.
The repeated seeds do not multiply independent recording exposure. None of
these row counts is continuous-stream FAR, independent human accuracy, or
physical SSC305 evidence.

`metrics.py` separates exact sequence, contiguous complete-keyword containment,
and nonempty diagnostic counts. It preserves CTC repetition/blank ordering,
unknown masks, keyword identity and separate positive/negative denominators.
The field named `gold_present` is an old input field name, not a human-gold
claim. Only explicit boolean labels are scored. This low-level scorer does
not authenticate source rights, labels or the declared recording roster.

The 28 invented-input regressions in `test_metrics.py` are imported by the
existing `research/experiment_quality_guards/tests/test_existing_boundaries.py`
entry, already run by required CI in normal, -O and -OO modes. No workflow or
schedule is added. The existing tests and historical retention pins are unchanged.

```sh
python3 -S -B -m unittest research.cnative_landing_v1.test_metrics -v
python3 -S -B -O -m unittest research.cnative_landing_v1.test_metrics -v
python3 -S -B -OO -m unittest research.cnative_landing_v1.test_metrics -v
python3 -B research/cnative_landing_v1/metrics.py \
  --model-dir /controlled/cnative-kws-causal-ln-v1 --output new-audit.json
```

The archive audit reinterprets saved rows only and refuses output overwrite.
Input data/role/label and closed-roster verification must be performed before
using rows for a new experiment. Local tests passed on CPython 3.12.16; full
repository CI is reported separately and must pass before protected merge.

The old causal checkpoint archive does not retain the executed forward graph.
It must not be guessed from weight shapes and claimed as original-model parity.
A new model requires a distinct graph/source/recipe identity, correct complete
keyword scoring, and its own bounded experiment. C/INT8, independent acoustic
and physical-board results are not conferred by this metric correction.
