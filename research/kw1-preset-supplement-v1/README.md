# KW1 preset supplement: prepared planning/source draft

Status: PREPARED_NOT_EXECUTED. TTS calls 0; ASR calls 0. This is an independent
small overlay, not an executable campaign, a change to old campaigns, or a claim
that source review has authorized inference. The inactive workflow draft is
outside `.github/workflows` and its execution gate always raises. It has no
model-loading, generation, ASR, uploading or GitHub-write implementation.

## Narrow objective and reason

Existing machine-only candidates contain 2 KW1 clips in Serena/Vivian groups and
11 KW2 clips across Serena/Vivian/Uncle_Fu/Dylan. Two KW1 groups cannot populate
three group-disjoint train/calibration/selection roles. This proposed campaign
seeks additional KW1 machine-consensus voice conditions, not a quality winner,
human pronunciation truth, product promotion, or a statistical independence claim.

All 16 slots have the literal input `你好小窝`, Chinese, native 24 kHz FLOAT mono.
No punctuation variants, phoneme substitution, Romanized replacement, added words,
or mapping 窝/屋/沃/我 into one label. New IDs and seeds; no old audio is submitted
to TTS or ASR, no old one-time branch/claim/attempt/budget is reused. Existing
candidates are only a frozen coverage reference in this draft, never auto-assigned
or replayed. Previously inspected C13 audio cannot silently become fresh heldout data.

## Official fixed support and limits

Use the two existing CustomVoice checkpoints and the unchanged qwen-tts 0.1.1
runtime. Exact model revisions, official URLs and source SHA256 values are in
`official-support.json`; existing local source references are in `source-bindings.json`.
The two relevant official source bodies match the retained runtime source-lock hashes.
The planning audit did not download/unpack a wheel or model weights.

- [0.6B config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/config.json)
  and [1.7B config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/config.json)
  both contain Uncle_Fu 3010, Eric 2875, Dylan 2878.
- The [fixed wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/6cafe5582caea83df269c36b1ce62d953a9cc66b/qwen_tts/inference/qwen3_tts_model.py)
  exposes generate_custom_voice with text/speaker/language/instruct and generation
  kwargs. It explicitly discards instructions for 0b6. Therefore only 1.7B gets an
  instruction arm. The 0.6B card's contrary generic statement is not adopted.
- The [fixed model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/6cafe5582caea83df269c36b1ce62d953a9cc66b/qwen_tts/core/models/modeling_qwen3_tts.py)
  maps Chinese/Auto to the preset's dialect for Eric (Sichuan) and Dylan (Beijing).
  A Mandarin instruction competes with this conditioning; it is not proven to
  override it. Uncle_Fu is the non-dialect primary route.
- The instruction is ordinary natural-language conditioning. There is no verified
  forced-tone, phoneme or pinyin argument. Its pronunciation efficacy is unknown.
- Official model cards declare Apache-2.0; retain notices. Use only named presets,
  no personal recordings/real-person cloning/reference uploads. Code/model license
  is not a general personality-rights clearance. Runtime must reverify the actual
  supported-speaker list and frozen config IDs before any call.

## Exact 16 slots

Two new seeds, 2026101401 and 2026101402, are used as fixed stochastic replicates.
Seed/engine/instruction changes never create a new independent speaker group.
The fixed 1.7B instruction is:

请用标准普通话清晰自然地朗读提供的文字，最后一个字读第一声；不要改字、增字或漏字。

| Slot | Model | Preset group | Seed | Instruction |
|---|---|---|---|---|
| kw1s-00 | qwen06 | Uncle_Fu | 2026101401 | empty |
| kw1s-01 | qwen06 | Uncle_Fu | 2026101402 | empty |
| kw1s-02 | qwen06 | Eric | 2026101401 | empty |
| kw1s-03 | qwen06 | Eric | 2026101402 | empty |
| kw1s-04 | qwen17 | Uncle_Fu | 2026101401 | empty |
| kw1s-05 | qwen17 | Uncle_Fu | 2026101402 | empty |
| kw1s-06 | qwen17 | Uncle_Fu | 2026101401 | fixed request |
| kw1s-07 | qwen17 | Uncle_Fu | 2026101402 | fixed request |
| kw1s-08 | qwen17 | Eric | 2026101401 | empty |
| kw1s-09 | qwen17 | Eric | 2026101402 | empty |
| kw1s-10 | qwen17 | Eric | 2026101401 | fixed request |
| kw1s-11 | qwen17 | Eric | 2026101402 | fixed request |
| kw1s-12 | qwen17 | Dylan | 2026101401 | empty |
| kw1s-13 | qwen17 | Dylan | 2026101402 | empty |
| kw1s-14 | qwen17 | Dylan | 2026101401 | fixed request |
| kw1s-15 | qwen17 | Dylan | 2026101402 | fixed request |

The 0.6B block has four slots; the 1.7B block has twelve. Within each 1.7B
speaker/seed pair only instruct changes. Within the Uncle_Fu/Eric empty-instruct
pairs only the checkpoint changes. These are separate contrasts, not an equal-size
model ranking. Do not adapt instructions, seeds or parameters after seeing results.

Everything else stays fixed: do_sample=true, top_k=50, top_p=1.0, temperature=0.9,
repetition_penalty=1.05; subtalker do_sample=true/top_k=50/top_p=1.0/temperature=0.9;
max_new_tokens=120, non_streaming_mode=true, CPU FP32/eager/eval/inference_mode.
Reset Python/NumPy/Torch RNGs to the cell seed before each call; record exact source,
platform and versions. Seeds support repeatable settings, not a cross-platform
bitwise determinism guarantee. No post-generation trimming, gain repair, resampling
replacement or hidden selection. Preserve native FLOAT values and file hashes.

## One future first-run chain, one final artifact

Proposed new identity: `kw1-preset-supplement-20261014-v1`. One independent workflow,
one `research/kw1-preset-supplement-20261014-v1` branch, one first-created public
push, run_number=1, run_attempt=1. Both names are proposed, not verified registered
or absent on GitHub. Never rename an attempted campaign to replenish budget.
No workflow_dispatch, schedule, rerun, resume, matrix retries or paid fallback.

The future reviewed host performs these serial stages in that same run:

1. Authenticate current run/checkout/source-lock/workflow/first-created-branch
   identities. Root creates the branch only at the final reviewed merged source.
   The new source manifest binds exact implementation and model/runtime inputs;
   no self-referential commit hash is invented. Check run history and a durable
   campaign claim before model load. Consume the campaign once, including load or
   setup failure. Keep built-in read-only API credentials in this guard alone.
2. Check current host RAM/disk, verified immutable images, exact model/package bytes
   and runtime scope. Generate the four 0.6B slots, destroy its container/runtime
   working state, then generate the twelve 1.7B slots. Model caches may only be
   reused after byte verification, never generation outputs or old attempt claims.
3. Require all 16 technical completions before ASR. Check natural EOS before token
   cap, finite mono samples, 24 kHz, 0<duration<=12s, nonclipping/nonsilent, actual
   FLOAT sample-value SHA256 and native WAV SHA256. Reject duplicate native or
   derived audio identities rather than crediting a duplicate as another voice.
   Derive 16 kHz s16le PCM with the existing pinned NumPy/SciPy numerical policy;
   save the derivation source/settings and both native/derived hashes. No repair.
4. Create a separate positive-allowlist ASR staging directory. It contains only
   minimal reviewed recognizer code/runtime and anonymous hash-named PCM plus
   geometry/hash job records. Do not mount the repository, this plan, TTS source
   metadata, target strings, speaker/seed/instruct/engine fields, mappings, prompts,
   hotwords or peer transcripts. Even read-only target metadata is forbidden in
   recognizer containers. `blind_projection` defines the input record interface;
   complete mount/content verification still requires implementation review.
5. Run Qwen3-ASR-0.6B, SenseVoiceSmall, Whisper-small sequentially on all sixteen
   slots once each, using their already fixed immutable model/runtime settings.
   Maximum 16+16+16=48 calls, no conditional replacement, majority shortcut or
   target-biased decoder. A failure consumes its attempt and stops the campaign;
   unfinished calls stay NOT_RUN. Keep raw text/completion/status per recognizer.
6. Rejoin results on the host by opaque identity and hash, recompute technical and
   exact target agreement under the unchanged NFKC + punctuation/whitespace rule.
   Three real successful complete unflagged transcripts must all match the target;
   2/3 agreement, absent third, homophone mismatch or decoder truncation fails.
   The report has fixed denominator16, per-slot/voice/model/condition counts and
   actual consumed-call counters; missing and failed slots never disappear.
7. Export one allowlisted artifact, including terminal failure records when safe.
   Include exact source/run metadata, claims/attempt ledger, model identities,
   native authorized preset WAVs, anonymous PCM, all raw ASR result rows, technical
   metrics, resource receipts and aggregate/group reports. Exclude weights,
   packages, cache, arbitrary logs, environment dumps, credentials and signed URLs.
   A single terminal upload uses that sanitized directory, not the host directory.
   No downstream artifact ID is needed before execution; record the assigned ID
   after publication. Do not make real results out of empty placeholders.

`workflow.yml` shows the first-run routing and offline tests, but deliberately
stops before these unimplemented host stages. It must not be registered or enabled
as if it were a finished runner. The pure validators are interface/contract drafts,
not proof of artifact authentication, actual mounts, audio quality or resource use.

## Bounded free CPU and terminal failure

Money cap $0; standard public ubuntu-24.04 only. One model container at a time,
4 CPU, 12 GiB RAM, no swap, 256 PIDs, nonroot, readonly root/code/runtime,
capabilities dropped, seccomp/no-new-privileges, inference network=none. No token
in model containers. Require >=14 GiB host available memory and >=25 GiB free disk
at each stage. Keep unchanged per-stage acquisition/install byte caps from pinned
adapters; never accumulate five expanded runtimes. Stage cleanup cannot erase claims
or results. Recheck capacity before every next stage.

Setup<=1200s/stage; load<=1200s/stage; each model call<=300s; global wall<=18000s;
workflow timeout330min. The global limit is binding even if individual maxima would
sum to more, so completion is not guaranteed. Native clip<=12s. Sanitized artifact
<=64MiB, each file<=8MiB, retention14days. No free-host availability promise.

Wrong source/model/runtime/voice/route, unsafe path or archive, capacity failure,
load failure, timeout, OOM, cancellation, nonfinite/clipping/silence/EOS failure,
identity collision or malformed/failed ASR completion is terminal. Preserve the
started attempt and known metrics, mark untouched slots NOT_RUN, no retry/resample,
no paid upgrade, no alternate model. RSS is per-process high water; cgroup peak is
container-tree memory. Record both with scope and unknown when unavailable.

## Group/label admission remains blocked

The same named preset is one group across engines, seeds and instructions. Distinct
preset IDs are conservative design groups, not empirically validated independent
human speakers. Never credit repeats, identical waveform bytes or prompt variants
as new groups. Existing machine candidates retain human_gold=false and
training_admitted=false. New three-ASR agreement has those same flags.

A new Uncle_Fu or Dylan KW1 candidate can improve overlap with the four existing
KW2 voice groups. Eric alone can make KW1 total three groups while the intersection
covering both keywords remains only Serena/Vivian; that still cannot fill three
roles with both keywords. `coverage_only` explicitly exercises this counterexample.

Even three common groups are only a necessary condition. Audit each keyword in
train/calibration/selection, global voice/family/duplicate leakage, waveform and
source identities, sample counts, prior inspection/use, split freeze and all
quality/admission gates before any training. All samples of one group stay in one
role. No role assignment or heldout borrowing occurs here; no human recall/FAR,
naturalness, real-far corpus or product qualification follows from this campaign.

## Current files, tests and implementation work still required

Only this new directory is proposed. No existing workflow, 30-ASR source binding,
old control, model/config, scientific code, registry, retention JSON or corpus is
changed. The independent root CHANGESET records exactly these new files.

32 standard-library synthetic contracts run without imports of Torch/NumPy/SciPy,
network or models in normal/-O/-OO. They test fixed slots, factor pairing, 0.6B
instruction exclusion, budgets/terminal stop, first-run route, opaque schema,
unchanged homophone distinction, three-result candidate semantics and group blockers.
`plan_contract.py --base BASE` separately checks the 15 explicit reference files.
`--check-execution` is expected to fail, proving the prepared draft cannot execute.

Before any execution decision: implement/review the actual once-only host guard,
serial isolated TTS+ASR workers, complete source/mount allowlists, identity bridge,
byte-bounded exporter, resource/timeout enforcement, and failure recovery accounting;
add negative tests and mocked end-to-end 16+48 scheduling without model calls.
A later authorized integration may add exactly one new workflow, its precise
retention allowlist/CI metadata, and necessary independent source files in one PR.
Do not edit old frozen host/expected-plan/activation or r5 authorities. Existing
48-slot validate_manifest is not a validator for this new 16-slot campaign.

## Separate completed diagnostic archive carried in this overlay

The sibling `../alimeeting-continuous-intake-v1/float-c-parity-observation-v1/`
contains exactly the nine previously reviewed publication-allowlist files (48,239
bytes). This is completed historical diagnostic evidence, not new campaign output.
It retains 13 float forward-call receipts and no new C acoustic call: float/C top1
matches on 1372/1375 frames, but logits are not numerically identical (maximum
absolute difference about 2.19759, aggregate RMSE about 0.129911). Float itself has
zero xiao/token3 top1 frames. These data cannot attribute the failure solely to
quantization, nor isolate quantization from activation/accumulation/frontend effects.
The archive contains no weights/audio/raw logits. `compare.py` is an archived
execution script and must not be run by source CI; only `verify_package.py` is the
read-only receipt/hash checker. Nothing is added to the currently running PR579.
