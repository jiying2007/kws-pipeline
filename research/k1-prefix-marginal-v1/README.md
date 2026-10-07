# K1 first-prefix CTC: failed research candidate

The fixed saved-output gate **failed**. On the confuser M3 (`你好小屋`), the
candidate falsely accepted K1 with A=0.7411511568874126, B=0.19759056685088203,
R=0.06125827626170545. N1 (`你好小窝，屋里有人`) accepted with
A=0.872928816141001. The candidate is stopped without tuning; this is not an
improvement or deployment claim.

This independent research directory preserves a useful mathematical component:
a constant-memory ten-state CTC first-prefix marginalizer, a fixed-precision
integer enclosure that prevents floating rounding from accepting an exact tie,
and independent raw-path/lifecycle/numeric tests. It has no runtime integration.

## Fixed meaning and result

Every six-column row uses the declared order blank, 你, 好, 小, 窝, 屋. CTC merges
adjacent raw repeats before removing blank. A comprises collapsed transcripts
starting with `你好小窝`; B starts with `你好小屋`; R is every other path,
including incomplete prefixes. The binary EOF rule is A > B+R, equivalently
A > 1/2, with ties rejecting. A three-class plurality is not substituted.

| Observation | Human whole-clip text | A | K1 action | Fixed gate |
| --- | --- | ---: | --- | --- |
| M1 | 你好小窝 | 0.9980094357009468 | accept | pass |
| M2 | 小窝小窝 | 4.996303452138876e-10 | reject | pass |
| M3 | 你好小屋 | 0.7411511568874126 | false accept | **fail** |
| M4 | 小屋小屋 | 4.235591324033869e-19 | reject | pass |
| M5 | 你好你好 | 2.8051710085160566e-06 | reject | pass |
| N1 | 你好小窝，屋里有人 | 0.872928816141001 | accept | pass |

M2's K2 behavior is outside this K1 candidate. Its K1 rejection establishes no
K2 improvement. All six results are numerically certified for the rounded
binary64 input-weight law; certification does not establish acoustic truth.

## Preserved execution history

The original saved runner failed on a binary64 exponential underflow before
the first EOF. The source and original failed ledger are retained. The failure
coordinate reconstructs 68 computed DP transitions, 59 committed rows and zero
EOF decisions; all 537 saved rows had already been loaded. Its measured CPU,
wall time and peak memory are unknown. A separate conversion-only diagnosis
examined 69 rows and made zero DP transitions.

A separately frozen numerical recovery permits only actually computed zero
terms/weights whose exact recovered-FP32 difference from the maximum is at
most -512. It never prunes a positive result. The recovery consumed the same
537 rows, 57 callbacks and six whole-observation DP instances, producing six
EOF decisions. Across both attempts: 605 computed and 596 committed DP rows,
six EOFs. The recovery recorded 0.038824465 CPU seconds and
0.042778524999448564 wall seconds; these are bookkeeping, not product
performance measurements. There were exactly zero new model, frontend,
decoder, audio, TTS, ASR or training calls in this saved-row evaluation stage.

The underflow-tail bound is ideal omitted-tail total variation <= 6*T*2^-512.
It excludes ordinary exp/subtraction/fsum/division rounding. The module's
certificate concerns exact normalized ratios of the supplied rounded binary64
weights only. These two statements do not certify ideal real-valued softmax.

## Scope and remaining bottleneck

Absorbed A or B mass cannot be retracted by any suffix. This preserves a true
first wake followed by an independent later 屋, but also preserves an erroneous
early 窝 branch. Identical posterior inputs cannot identify whether later
evidence belongs to the same acoustic occurrence or a separate word. The M3
failure leaves the acoustic 窝/屋 distinction and occurrence boundary unresolved;
suffix protection alone does not solve that bottleneck.

The caller supplies whole-observation EOF and the reset boundaries. This is
not keyword-anywhere or continuous detection, and EOF input availability is
not word latency. The six observations were already exposed and use one stock
voice; they provide no held-out generalization, FAR/FRR, calibration, physical
board or production qualification. No threshold, normalization, crop, reset,
terminal slot, class grouping or model parameter was fitted after the failure.

## Archive layout and evidence

- `frozen/` contains the exact reviewed module, original runner and recovery
  runner source bytes, protocols, manifests, tests and disabled release requests
- `review/` preserves the initial tie failure, corrected 612-case oracle,
  5,523,157 visited raw paths, 11 edge groups, eight underflow groups, and
  independent mathematical/numerical review
- `check_synthetic.py` reruns only the independent bounded synthetic checks
  without replacing any archived report
- [kws-data PR20](https://github.com/jiying2007/kws-data/pull/20) contains the
  append-only numerical evidence at `research/2026-10-07-k1-prefix-marginal`

Historical documents in `frozen/` and `review/` describe their preparation or
review-time state. Their `NOT_EXECUTED`/`NOT_RELEASED` wording is preserved,
not a statement that the later reported evaluation never happened. Included
release templates remain disabled; publication does not release another run.
Actual release/approval records are not distributed. Hash references to those
records remain as historical provenance, without reconstructing their contents.
Public manifest projections remove internal path metadata and explicitly bind
the original and projected identities; scientific source/evidence bytes remain
unchanged. Original pre-certificate source bytes are not included; its retained
initial-oracle report documents the observed tie failure and original hashes.

The raw inputs are referenced, not duplicated:

- [M1–M5 raw rows at dc90c3aa](https://github.com/jiying2007/kws-data/blob/dc90c3aa325700fdf17da449452437c90373818e/research/2026-10-06-melo5-leading-context/original_A20.raw.jsonl)
- [N1 raw rows at 8c301d7a](https://github.com/jiying2007/kws-data/blob/8c301d7a4709b066633ebfd9316159281ee320ea/research/2026-10-07-n1-wholeclip-leading/original_A20.raw.jsonl)

Result bytes: 11,283; SHA256:
`76e35a4f16a004f6e3724b282c00775c350e3d02a7d833c4e8e5856bf68fc54a`.

## Offline synthetic verification

Requires standard-library Python 3.10+ on Linux. From this directory:

```sh
(cd frozen/k1-prefix-marginal-v1 && python3 -B -m unittest discover -s tests -p test_k1_prefix_marginal.py -v)
(cd frozen/k1-prefix-saved-evaluation-v1 && python3 -B -m unittest -v test_runner.py)
(cd frozen/k1-prefix-saved-evaluation-recovery-v2 && python3 -B -m unittest -v test_runner.py)
python3 -B check_synthetic.py
```

The scoped workflow runs the same 30 + 7 + 12 tests and independent checks,
using the repository's already pinned checkout action. It installs no
dependencies, fetches no data, passes no execution release and opens no saved
raw inputs. No new research model workflow is dispatched by this package.
