# KWS product landing status

The repository distinguishes **model availability** from **product shipping
approval**.

Run:

```bash
python3 tools/kws_landing_status.py --verify
```

The current status is intentionally:

- trained deployable-format model: **available**;
- immutable model release: **available**;
- Git model registry mirror: **available**;
- synthetic qualification: **passed**;
- shipping approval: **false**;
- synthetic architecture search: **paused**.

The next product gate is **real-human final-AFE acoustic qualification**, followed
by **physical target-board performance and soak**.

## Why architecture search is paused

The research line has already tested margin tuning, PCEN64, GRU128 capacity,
stacked GRUs and a zero-initialized local-context residual adapter. Larger
capacity showed mechanism value, but the deployable-shape and efficient
architecture variants did not produce a stable calibration operating region.

At this point another synthetic architecture change is lower-value than measuring
the existing deployable model on the actual product distribution. Architecture
search reopens only if product data shows a repeatable failure that implicates
model capacity rather than AFE, acoustic coverage, threshold calibration or
runtime integration.

## What is still needed for product landing

Phase A uses the frozen model with final product microphones/enclosure and final
AFE. The policy requires real human speech, 3-5 m coverage, rear direction,
playback, double-talk, robot motion and 24 hours of post-AFE negative exposure.

Phase B requires physical target-board evidence across multiple DUTs, including
CPU/RTF, RSS/stack, thermal, power, audio continuity and long soak.

Private raw human audio remains outside the public Git repository. Only the model
tuple and non-audio evidence are versioned here.
