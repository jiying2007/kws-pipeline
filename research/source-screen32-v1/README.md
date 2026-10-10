# Bounded source-screen preparation: 32 planned, zero generated

This reuses retained Qwen TTS/ASR and experiment-quality tools. It now includes an
unexecuted source-specific container launcher, with admission still disabled and
its workflow only a research template. Tests use fictional objects and kernel
records; they validate contracts, not actual hosted or model execution.
`readiness.json` is explicitly **not execution ready**.

## Frozen design

`plan.json` fixes two sources, two adult voice-design descriptions per source,
eight exact phrases and one generation attempt per cell (seeds 101001–101032):

- Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- FireRedTeam/FireRedTTS3, **Instruct** variant
- 你好小窝、 小窝小窝、 你好小屋、 小屋小屋、 小窝小屋、 小屋小窝、 你好你好、 小窝

No reference audio, person imitation, cloning, paid API, punctuation repair,
extra seeds, regeneration or quality-based selection. A failed attempt is consumed;
remaining cells in that source stay NOT_RUN. The ceiling is 32, not a requirement
to manufacture 32 valid clips. First admitted generation is itself one of the 32
cells, not an extra speech smoke test. No historical source gets renamed fresh.
A description repeated across eight phrases is a prompt group, not proof of a
stable voice identity, eight independent people, age coverage or source independence.

All prospective outputs remain `source_screen_only`, `human_gold=false`,
`training_admitted=false`. Preserve native output and derived PCM hashes,
finite/rail/silence/termination flags and failures; do not repair waveforms to
obtain a pass. The ledger helper checks declarations only; a later execution
adapter must bind actual bytes with the existing PCM/audio guards before ASR.

## Reuse, not a replacement framework

`reuse-pins.json` binds the existing recipes and safety sources without changing
them. `request_preview` prepares call arguments as data. Qwen reuses the existing
120-token generation recipe, changing the API from stock voice to VoiceDesign.
It never imports qwen-tts. FireRed stays blocked. Historical six-clip commands and
approvals cannot execute this plan, and the six-clip ASR contract is not weakened.

The primary readback is the existing pinned Qwen3-ASR-0.6B plus SenseVoiceSmall:
sequential one-pass, empty context, no hotwords, no intended text or voice metadata.
An audio-only projection sorts content hashes before assigning opaque IDs. A
separate ASR environment must receive only these inputs, not this producer plan.
Both raw primary outputs are frozen before private target joins. The pure
`freeze_disputes` helper selects only clean, complete two-ASR disagreements for
one frozen Whisper pass; missing/flagged outputs stay unresolved. Whisper's exact
model/runtime admission is still missing. Maximums are 64 primary and 32 dispute
calls, with no repeated voting. Agreement never creates a human label.

The helper reuses existing punctuation/whitespace normalization. It performs no
homophone folding, 窝/屋 replacement, fuzzy acceptance or gold-label assignment.
Actual-label coverage and PCM/source/exposure lineage must pass the existing
quality gates before any separately authorized training or comparison. Historical
Cosy6 actual words, including Z3 小挖, remain unchanged.

## Free CPU feasibility and precise blockers

The [official Qwen card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/blob/5ecdb67327fd37bb2e042aab12ff7391903235d3/README.md)
declares Apache-2.0. Official metadata binds its two weights to **4,515,695,644 B**. The eleven exact
sidecar bodies have now been acquired and hashed, totaling **4,468,188 B**. The
complete model snapshot is **4,520,163,832 B**; adding the retained candidate
runtime yields **4,956,729,799 B**. This exceeds the old 4 GiB controlled-transfer
ceiling. No weight body was downloaded or verified in this task.

A CPU FP32/eager implementation is plausible using the retained Qwen guards,
but no 1.7B capacity/latency pass exists. The proposed free standard public
ubuntu-24.04 runner is documented at [4 vCPU, 16 GB RAM and 14 GB storage](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories).
Those are specifications, not observed available memory/disk. `readiness.json`
proposes 12 GiB **hard** memory, zero swap, 6 GiB transfer, 8 GiB initial free disk,
2 GiB host-memory reserve, 300 s per clip, 4,800 s generation and 7,200 s whole job.
These are unadmitted limits, not measured performance. A model-size label alone
cannot prove FP32 residency or fit. Old sampled RSS supervision is not a hard
sandbox. Hard isolation/readback must precede imports; failed capacity stops,
without switching precision, hardware, source or expanding the budget silently.

The [official FireRed code](https://github.com/FireRedTeam/FireRedTTS3/blob/7a1f3a7282ff184cc1c7f070556baaf5f08b5216/fireredtts3/llm/fireredtts3_instruct.py#L300-L317)
hardcodes CUDA and uses CUDA autocast. Its required Instruct and RedAE directories
show about 8.48 GB + 3.78 GB, before tokenizer/dependencies. The [pinned model card](https://huggingface.co/FireRedTeam/FireRedTTS3/blob/dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610/README.md)
declares Apache-2.0. Exact asset hashes/runtime closure and a verified free CPU
route are absent. No CUDA rewrite, GPU rental or fallback has been attempted.
`research/firered_cpu_preflight` is **FireRed AED ASR**, not FireRedTTS3; its
results supply no TTS qualification. Both TTS model licenses remain distinct
from rights in outputs or third-party data. SenseVoice retains its separate
FunASR model license and [notice](../qwen6_asr/NOTICE.md).

## Next safe commands and gates

From the repository root:

```sh
python3 -B research/source-screen32-v1/contract.py
python3 -B research/source-screen32-v1/contract.py --preview-cell screen32-001
python3 -B research/source-screen32-v1/test_contract.py
python3 -B -O research/source-screen32-v1/test_contract.py
python3 -B -OO research/source-screen32-v1/test_contract.py
```

There is no valid generation command yet. Qwen asset metadata is complete.
The exact locked-wheel VoiceDesign API is now verified by source inspection, and
the narrow adapter is mock-tested. The runtime
orchestration and actual hard-scope readback are now implemented with offline
fixtures. Independently review and admit the frozen implementation, then verify
the live container/native-runtime gates before generation. A staged start may leave FireRed's
16 cells NOT_RUN, preserving denominator and blocker rather than substituting a
source. Never invoke the historical `--execute-reviewed-six` or FireRed ASR
preflight as a substitute. This preparation does not reopen unrelated denied
archive-codec or CPU-qualification counterexamples.

## Unchanged evidence boundaries

`source-roles.json` distinguishes fixed50 evaluation, the older native A20 50-asset
inventory, Serena18, exposed Cosy6 Z1–Z6 and the separate CosyVoice SFT six.
Fixed50 remains EVAL_INCONCLUSIVE_LABEL_SUPPORT. D20 remains FAIL; D90 NOT_RUN;
shipping_approved=false. Final human/AFE and physical-device qualification remain
deferred. No model accuracy, FRR, FAR, human-label or shipping improvement is
claimed by planning or its fictional unit fixtures.

### Reused actual hosted observations, checked 2026-10-10

The [2026-10-01 probe](https://github.com/jiying2007/kws-pipeline/actions/runs/36915231606/job/110547462206)
reported 15,265,845,248 B available RAM and 92,425,285,632 B free disk, four CPUs,
Python 3.12.3 and image 20260927.320.1. Its allowlisted 1,925-byte JSON is retained
as `hosted-resource-observation.json`, re-extracted from the official log.
At that moment, 12 GiB plus a 2 GiB host reserve left 233,459,712 B of margin.
This supports a plausible free capacity route, not a current reservation or a
model-fit result. Effective cgroup limits were unknown. Reuse the existing
resource-probe workflow for current availability; hard enforcement still needs
separate verification. This task did not dispatch it.

[Qwen6's completed smaller-model run](https://github.com/jiying2007/kws-pipeline/pull/482)
used the repaired native-I/O setup and completed six TTS/twelve ASR calls on free
CPU, with generation taking 301.404 s. The panel itself was rejected: only Aiden
K2 had weak support; no voice supported both wakes, and all six are now exposed.
This is useful infrastructure history, not 1.7B model fit, fresh data or training
admission. No prior scientific run is reopened.

The eleven exact new sidecars are now byte-bound in `qwen-model-lock.json`,
including SHA256, Git blob SHA1, official URL and reported repository revision.
Only metadata is committed; the sidecar bodies are not published here. The existing
92-input runtime already has complete pins; no package-version refresh is proposed.
The retained static JSON checker passed all eight model JSON files without importing
model dependencies. Config bytes confirm VoiceDesign/1b7 and Transformers4.57.3.
The remaining work is review/admission of the implemented orchestration and
actual hosted proof of its hard scope, then cells 001–016. The larger declared transfer/resource budget
is distinct from a hardware blocker; historical small-run limits are not universal
resource limits. FireRed's sixteen cells remain visibly NOT_RUN. No valid command
for actual generation is manufactured before these gates are implemented.


### Exact VoiceDesign API review and narrow adapter

The exact 113,529-byte `qwen-tts==0.1.1` wheel from the pinned official PyPI URL
matches SHA256 `11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d`.
All 17 Python members match the retained source lock. Source inspection confirms
`generate_voice_design(text, instruct, language, **recipe)` and the device-generic
CPU candidate path. The exact wheel and member evidence is in `source-review.json`.
No third-party module was imported or installed for this review.

`qwen_adapter.py` accepts only the frozen Qwen cells in order, verifies the loaded
VoiceDesign/1b7/12Hz identity, absent speaker encoder/tables, CPU FP32 parameters,
eager attention in model and codec, and evaluation mode. It fixes all three random
seeds before each single generation call. It rejects invalid native geometry or
nonfinite values, preserves native float values and records their hash plus silence
and peak flags without waveform repair. An exception consumes the in-memory
attempt and stops that source. The caller still needs durable once-only claims;
reconstructing this object is never permission for another attempt.

The adapter does not load a model or create an execution scope. The new scope checker reads actual private cgroup-v2 memory/swap/CPU/PID limits,
read-only mounts, dropped capabilities, seccomp/no-new-privileges and network
interfaces. There is no caller boolean or callback bypass. Unit tests supply
fictional kernel records; no actual container scope has been executed here. Its native receipt
keeps termination unverified and acoustic completeness UNKNOWN. The future admitted
launcher must establish and verify hard limits before imports; then enforce current
capacity, exact installed/source/asset identity, local-only loading, persistent
attempts, deadlines, EOS capture, native and derived file hashes, and blind packing.
The old six-cell setup has a fixed 0.6B model identity and 4 GiB transfer cap; the
old runner and ASR/packing code fix six clips. They cannot be invoked unchanged or
silently monkey-patched into this new sixteen-cell lane. Reuse their audited helpers
through a separately reviewed, bounded orchestration delta, leaving old evidence
and source pins intact. CPU fit and runtime compatibility remain NOT_RUN.


## Reviewed next execution path, still unexecuted

The source-specific files now fit together: `hosted_run.py`, `runtime_scope.py`,
`setup_adapter.py`, `tts_worker.py` and `asr_worker.py`. The setup adapter binds
only the exact new model identity, locks, budgets, cache paths and fixed child
roles to retained helpers. It does not invoke historical runners or modify their
source files. ASR scientific loader bodies remain AST-identical; only the new
bounded audio-only preloader replaces their six-clip entry contract.

The official Python 3.12.14 slim-bookworm amd64 image is pinned in
`container-lock.json`; its verified manifest declares 45,449,534 compressed
layer/config bytes. A single image-pull client is monitored for 180 s; all host receive-counter deltas
during it conservatively count against a 64 MiB reservation. This is an observed
budget with an abort-before-setup rule, not a hard Docker-daemon transfer cutoff.
Polling can overshoot; cancelling the client does not prove daemon fetching stops.
A failed, timed-out or over-budget pull retains daemon completion and cumulative
bytes as unverified and starts no setup or generation. No daemon shutdown or host
security change is used. TTS package
and model acquisition gets 6 GiB minus that reservation. ASR preserves its 4 GiB
input cap, plus the declared 64 MiB image allowance. Each fresh job requires at
least 16 GiB free disk, with a 12 GiB runtime workspace cap; these are explicit
new-lane budgets, not new hosted hardware claims.

Every container has 12 GiB RAM, zero swap, four CPU quota, 256 PIDs, no capabilities,
no-new-privileges, read-only root/code and no privileged/host namespace/socket
mount. Inference has `--network none`, read-only runtime/model inputs and only its
own output/scratch. Kernel values are read back inside the scope before runtime
imports; launcher flags alone never qualify it. The host enforces setup 1200 s,
inference-stage 4800 s, load/startup 1200 s, each clip 300 s and whole job 7200 s.
A kill removes the entire dedicated container; no automatic restart exists.
These ceilings may stop a slow run before all cells finish.

Exclusive durable TTS/ASR claims consume an attempt before a call. Terminalizers
reconcile claims with exact receipts after normal exit, interruption or OOM;
unstarted cells stay NOT_RUN. All 32 producer cells remain visible. Native float
WAVs, float-value hashes, fixed 16k PCM16 derivatives, decoded-PCM hashes, EOS and
quality flags are retained without repair or selection. Human actual labels stay
PENDING; no candidate is training-admitted.

The ASR container source allowlist excludes producer contracts, plans, prompts and
TTS outputs. Its only data mount is `job.json` plus content-hash named WAVs. Qwen
and SenseVoice have separate processes/output mounts and cannot read each other's
answers. Raw individual receipts and immutable freezes precede terminal summaries.
Whisper is not part of this execution path. A TTS failure preserves its valid
prefix evidence but does not automatically start downstream ASR.

`workflow.yml` is deliberately only a template. `execution-release.json` remains
`approved=false`. After root review of `execution-freeze.json` and immutable source,
activation must copy the exact template to
`.github/workflows/source-screen32-run.yml`, bind the reviewed freeze in the release,
and create the unique `research/qwen16-voicedesign-once-v1` branch once. Both workflow
run number and attempt must be 1. No dispatch/retry path or admission from CI exists.
The intended TTS entry is:

```sh
python3 -I -B research/source-screen32-v1/hosted_run.py --phase tts
```

Running it from this unapproved preparation fails before acquisition. Local tests
use only stdlib fixtures and mocks; no image layers, installed runtime, exact native
libraries, model fit, latency, speech quality or ASR result is claimed.
