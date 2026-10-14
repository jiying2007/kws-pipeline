# Remaining fifteen, executable but not activated

No workflow or inference has been started by this package. Install workflow.yml at
.github/workflows/tts-quality-remaining-20261014-v1.yml only after source review;
create research/tts-quality-remaining-20261014-v1 once at the reviewed commit.
Both engines run serially, only their remaining 15 cells. Cell zero is restored,
never generated again. Successful output has cumulative 16 outcomes and 16 native
WAV files, with human pronunciation/naturalness review still pending.

## Actual input bindings

Bindings use root-authenticated public evidence from source commit
89936529b86ec48f335accefc96b155025b684c9, run 38554125374, first run/attempt.
Qwen06 job 115818916334 / artifact 11887374758 and Qwen17 job 115818916445 /
artifact 11888856280 each completed successfully. The exact immutable archives
are retained in research/effect-data-model-v2/evidence/probe-run-38554125374/
qwen06.zip and qwen17.zip. Both archive SHA256 values match authenticated GitHub
artifact digest and local bytes. Source manifest binds 78 original files, including
the probe workflow; none of those files is modified by this package.

The admission step alone gets the ephemeral built-in GitHub token with read-only
contents/actions permissions. It rechecks fixed run/job/artifact metadata through
GitHub API, including exact job identity, run/source association, success and digest.
Overall run failure is acceptable only if the target lane job and all target-lane
local evidence succeed. Failed/skipped/missing/cancelled/OOM target lanes stop.
It reads the fixed archived ZIP, no redownload or new credentials. ZIP members are
flat allowlisted files; duplicate/path/link/encrypted/size violations are rejected,
and all six required evidence members match extracted bytes and locked hashes.

The host receives no GH_TOKEN. Docker command environment is a positive allowlist
and contains no token. Source and inputs are mounted read-only under /code. A separate
/code/probe-source copy preserves the exact original source while the existing
reviewed setup binder selects its engine lock in /code/research. The loader remains
the original verified offline CPU FP32/eager code. The new worker invokes only
run_verified_loaded → run_loaded(..., phase='remaining'). Persistent claims prevent
retry, even when loading or generation fails. First global workflow run_number=1,
attempt=1, public first-created branch, exact push/source metadata are checked.

## Measurements and budget

Zero paid services; public standard ubuntu-24.04 only, serial 2-lane matrix.
Each job <=110min; host <=6000s; setup <=1200s; inference <=4800s; per cell <=300s.
The existing supervisor enforces model-load <=1200s, output cap, and Docker limits:
4CPU, 12GiB/no-swap, 256 PIDs, unprivileged, readonly root, dropped capabilities,
seccomp/no-new-privileges; inference network none. Model/runtime download/install
caps are unchanged from probe. No ASR or old r5 calls occur here.

resource_receipt.py reads, but never resets, current-container memory.peak/cpu.stat
and process getrusage(RUSAGE_SELF).ru_maxrss. Linux RSS is explicitly converted from
KiB to bytes and identified as one-process high water; cgroup peak covers the
container process tree. Start/final observations distinguish cumulative counters.
Worker finally is best effort on success/failure without masking original errors.
The host writes missing/invalid worker measurements as unknown, including when a
hard kill prevents finally. Unknown never becomes zero or an inferred peak. Original
probe peakRSS was not measured and is not backfilled.

Public output is a positive list of safe JSON resource/status/binding evidence,
expected claims and authorized preset native WAVs; no setup stderr, weights,
packages, environment, tokens or signed URLs. Total cap 64MiB, 8MiB/file, retention
14 days. A failed lane preserves attempted outcomes and cannot automatically resume.

## Tests and verification limit

From repository root, run these self-contained stdlib test entrypoints:

python -B research/tts-quality-remaining-v1/run_tests.py
python -I -B research/tts-quality-remaining-v1/run_tests.py
python -I -S -B research/tts-quality-remaining-v1/run_tests.py

Each mode passes 30 tests: 6 host/admission tests, 18 archive/source/remaining
boundary tests, and 6 resource-receipt tests. No external template fixtures are
needed; the tests use retained original source/ZIPs already in the repository,
plus synthetic unit fixtures generated in temporary directories. For standalone
review outside a checkout only, TQP_TEST_SOURCE_ROOT can point at that checkout.
Resource availability/unknown behavior uses mock/kernel-record fixtures and does
not manufacture measurements for the old probe. Tests call no network, Docker or
inference runtimes. No site packages are required, including under -I -S.

Actual archived metadata + ZIP + native waveform + 78-source bindings passed for
both engines. An independent review also executed the frozen pilot through mocked
generation and verified exactly 15 new calls per lane, cumulative 16 outcomes,
unchanged cell-zero SHA, and rejected reentry. These are adapter tests, not real
remaining inference. Only the later authorized workflow can establish that result.
