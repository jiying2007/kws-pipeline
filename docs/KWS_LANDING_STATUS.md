# KWS product landing status

The repository distinguishes **model availability** from **product shipping
approval**.

Run:

```bash
python3 tools/kws_landing_status.py --verify
```

This command verifies the repository's pinned source contract and local model
registry bytes. It does not query live Actions, Releases, restricted real-human
evidence or physical DUT receipts. `real_human_final_afe_passed=false` and
`physical_target_board_passed=false` describe this pinned source candidate's
pending evidence; they are not a live external qualification lookup. The JSON
reports `assessment_scope=repository-source-contract-only` and
`live_qualification_checked=false` for that boundary.

The current status is intentionally:

- trained deployable-format model: **available**;
- immutable model release: **available**;
- Git model registry mirror: **available**;
- synthetic qualification: **passed**;
- shipping approval: **false**;
- historical synthetic architecture-search line: **paused in its retained closure record**.

The next product gate is **real-human final-AFE acoustic qualification**, followed
by **physical SSC305/target-board performance and soak**. These are product
qualification gates, not entry conditions for software research.

## Research can replace the architecture now

The historical line tested margin tuning, PCEN64, GRU128, stacked GRUs and a local
context adapter without establishing a stable calibration operating region. Its
closure record and the status command's pause field remain historical facts about
that line; they do not prohibit the newly authorized software research program.
This documentation does not change the closure JSON, machine gates or shipping tuple.

Current research may replace the acoustic encoder, frontend, objective, decoder
and model ABI. Real-human recordings and physical boards are not prerequisites
for lawful, bounded software experiments. Fixed data identity, causal/reset/chunk
checks, honest labels, limited experiments and explicit stopping conditions still
apply. Select one supported, maintainable deployment path rather than accumulating
unused compatibility layers or multiple permanent runtimes.

The target specified by the user is **SSC305 with dual Cortex-A32 cores**. Report
CPU budgets normalized to **one core**, alongside actual thread count; do not halve
a single-thread cost merely because two cores exist. Synthetic and authorized open
data support full software implementation, ARM32 builds, streaming/parity checks
and integration packages now. Board work is primarily later acceptance testing,
not a preliminary delivery gate. Missing real-human evidence limits the associated
product claims and qualification, rather than blocking software development.

CPU and I/O are the primary resource priorities. Larger models and higher RAM are
allowed when the quality/resource tradeoff supports them. Separate cold/warm
startup reads from resident-model steady-state CPU, 20 ms block percentiles, file
I/O and allocation. Preloading or intending resident weights does not guarantee
zero physical I/O or page faults; measure those counters as well.
Model bytes, causal cache, C arena and whole-framework RSS are
different measurements. Hosted x86 timing and PyTorch/ONNX RSS do not establish
SSC305 performance or embedded workspace requirements; freeze product budgets on
the actual target under its AFE and competing workloads.

The [retained Qwen42 comparison](https://github.com/jiying2007/kws-data/tree/bbc5c5d4eabef4b39b90e49352cb722996031d34/research/2026-09-30-qwen42-baselines),
merged in [kws-data #2](https://github.com/jiying2007/kws-data/pull/2), reports frozen
C target presence in 0/20 positive clips versus 19/20 for the external sherpa-onnx
reference. Neither emitted events on the 22 confusables. Fixed 500 ms tail and
500 ms head+tail padding controls did not change either result. These observed
short-clip counts are not event FRR, continuous FAR, endpoint latency, a fair
market ranking or product qualification. Frontends, decoder APIs, compute budgets
and threshold scales differ. Zero confusable events cannot compensate for missing
all positives. See the [current research roadmap](KWS_RESEARCH_PRODUCT_ROADMAP.md)
for the bounded replacement program and evidence limits.

## What is still needed for product landing

Phase A uses the frozen model with final product microphones/enclosure and final
AFE. The policy requires real human speech, 3-5 m coverage, rear direction,
playback, double-talk, robot motion and 24 hours of post-AFE negative exposure.

Phase B requires physical target-board evidence across multiple DUTs, including
CPU/RTF, RSS/stack, thermal, power, audio continuity and long soak.

Private raw human audio remains outside the public Git repository. Only the model
tuple and non-audio evidence are versioned here.
