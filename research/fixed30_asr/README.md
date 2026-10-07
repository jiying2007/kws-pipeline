# Fixed30 setup recovery v2

Prepared source candidate, not an executed result. Scope is exactly30 existing
synthetic recordings, each decoded at most once by SenseVoiceSmall and
Qwen3-ASR-0.6B in sequential fresh CPU processes. All30 are exposed human-reviewed
rule-development recordings. This is not heldout accuracy or generalization
validation. No KWS, training, synthesis, threshold tuning or automatic relabeling.

The only audio inputs are fixed30-blind-inputs.zip:30 WAVs and blind job.json.
The artifact contains raw ASR/status/resource JSON and hashes, no human labels,
alias map, model weights, tensors or package bodies. See NOTICE.md for the
SenseVoice model-license distinction and unchanged native-fbank frontend.

Publication mapping: this directory goes to research/fixed30_asr/ in the chosen
public repository; copy workflow.yml byte-for-byte to
.github/workflows/fixed30-asr-regression.yml. Keep the template here too. The
runner verifies both its source freeze and deployed workflow template identity.
It scans only its own research directory, not the containing repository.

The target is jiying2007/kws-pipeline, branch
research/fixed30-asr-setup-recovery-v2. Only its first branch-creation push can run,
with public-repository and run_attempt1 guards; later pushes and reruns cannot.
Checkout binds the triggering event SHA. Before that authorized first push, the
parent must approve the exact source and adjacent research/fixed30-asr-release.json
with approved:true, the experiment ID and reviewed candidate-freeze.json SHA256.
The release lives outside the source freeze to avoid a self-referential hash.
Its prepared template is unapproved; source preparation is not execution GO.
A public standard ubuntu-24.04 runner is used, without secrets or dependency
caches. Checkout and exact Python3.12.14 provisioning share5min; controlled
setup≤20min, each model≤10min, and5min is reserved for finalization/upload inside
the50min job. A platform kill can still prevent artifact persistence.

The explicit downloader's package/model response-body budget is4GiB. Hosted
checkout/action/interpreter transfer is separate, unmeasured infrastructure;
this is not a kernel-wide network cap. Disk and RSS limits are sampled. Per-model
kernel RLIMIT_CPU600s and an outer wall600s deadline are enforced. Per-clip120s
SIGALRM is cooperative only. Resource-limit failure never raises a limit or
retries a model. Five source distributions are built once with locked tools;
their actual wheel/METADATA hashes remain future setup results.

Output artifact retention is1day. The operator must immediately download and
durably persist the result, then verify its hashes before any private label join.

This is setup attempt2 after run37285275142 failed before any model call. That
run remains FAILED_NO_RETRY. recovery-control.json binds its measured transfer
and zero ASR consumption. Expected cumulative controlled download is6,681,932,894B
across the two attempts; maximum is7,635,933,743B including the already-used
3,340,966,447B. Total model-call scope remains the first30x2 calls, at most60.

The sox1.5.0 metadata declaration is completed from its exact pinned source;
all8 docs/tests-only entries remain inactive. No package/asset pin or model
recipe changes. Models are fetched after all five source-wheel checks/install.
New setup diagnostics publish stage/exit codes and bounded stderr-only
whitelisted codes/classes; unexpected metadata text is hashed, not disclosed.
No arbitrary argv, paths, error messages or raw stderr are public.
