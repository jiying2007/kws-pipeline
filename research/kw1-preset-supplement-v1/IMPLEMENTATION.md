# KW1 reconstructed implementation revision 2026-10-15

Status: SOURCE_REVIEWED_PENDING_FIRST_EXECUTION. Fresh offline and independent
source review passed for the reconstructed revision. Execution admission is
enabled for the sole first-created campaign branch after protected-main
integration and its required CI checks. No actual campaign execution is
established by this source state.
This is a new implementation over verified main d957ad7606a5acc1e26cb031df3133cdb9937dcd
(tree 1c761d317b3f364201a1c8015094b08e22891838), not restoration of the lost final
candidate tree d5c88213a559de4bb1698bf6e373eff9112f68b8 or its prior PASS claims.
Recovered source blobs remain separately immutable. Missing plan, TTS worker,
public evidence/export/verification, host launcher/common and tests were rebuilt.
Recovery provenance is documented in RECONSTRUCTION.md.

The original scientific design is unchanged: sixteen TTS slots (four 0.6B, twelve
1.7B), then all three fixed recognizers on all sixteen unique anonymous PCM clips
(maximum 48 ASR calls). No model execution, new campaign branch or training was
performed during reconstruction. The campaign's real budget remains 0/16 TTS and
0/48 ASR used. Synthetic test fixtures are not campaign output.

Admission requires a public first-created research branch push and exact source,
server run identity, first lifetime run, run_number=1/run_attempt=1. No replay,
resume, dispatch or automatic retry exists. Configuration excludes only its own
self-hash; the claim separately binds it. The launcher binds original wall and
monotonic admission clocks to a host boot and never renews the 18000-second budget.
Its execution claim is exclusive. Finalization only reconstructs durable evidence;
an existing public package is verified read-only against the admitted identity.

The container backend retains 4 CPUs/12 GiB/no swap/256 PIDs/non-root/read-only,
network-free model execution with no credentials; 14 GiB available host memory and
25 GiB disk gates precede each stage. Exact runtime-only cleanup follows verified
container teardown. Setup/load 1200 seconds, call 300 seconds, global 18000 seconds,
workflow 330 minutes. Public Ubuntu Actions budget is $0, with no paid fallback.

Native samples remain 24 kHz FLOAT mono with exact source-value identity, finite
nonclipping/nonsilent duration <=12s and natural EOS strictly before token cap.
Host derivation uses resample_poly(2,3), Kaiser 5.0 constant padding and float64
floor-to-int16. No normalization, trimming, repaired audio or regenerated slot.
Blind mounts expose only positive-allowlisted code and anonymous hash-named PCM,
never target, plan, voice, seed, instruction or peer transcript context.

Public export authenticates complete durable parent chains. Invalid/missing/empty/
truncated/schema-invalid authoritative receipts expose fixed names/hash/size only,
with UNVERIFIED state and unknown attempts reserved, never a free retry. Unknown
JSON bodies, model caches, weights, logs, credentials, environment dumps and signed
URLs are not exported. Scientific raw metadata is represented by a reviewed
identity projection and its digest; exact decoder transcripts/status are retained.
Reports and all 64 ledger slots are reconstructed with strict JSON scalar types.
An independently verified temporary package is atomically published NOREPLACE.
Public caps are 64 MiB total and 8 MiB/file, retention 14 days.

Machine consensus remains train-only candidate evidence, never human_gold or
calibration/selection gold. Three common preset groups satisfy only a necessary
coverage condition. Previous C13 observations are not fresh heldout. Admission is
BLOCKED_NO_CALIBRATION_SELECTION_GOLD; no training or shipping follows.

Offline entry: python3 -I -S -B test_suite.py, also -O and -OO. Numeric entry with
hash-locked NumPy/SciPy: python -I -B test_suite.py --require-numeric. Missing tests,
zero tests, skips and missing numeric dependencies fail. Actual test results are
reported separately only after execution; no historical PASS is inherited.
