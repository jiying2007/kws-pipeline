# TTS quality pilot, new campaign 2026-10-14

## Result and readiness

Prepared, NOT executed. No model loaded, no workflow launched, no old r5 allowance used.
This directory is new; historical code, controls and evidence are unchanged.
The executable first step is two independent Qwen resource probes, one per model.
The first probe is cell 0 of each 16-cell lane, not a 17th generation. A successful
probe does not automatically launch the remaining fifteen. Root reviews actual
resources and audio first. The current workflow deliberately only calls phase probe.

CPU availability is a distinction between code support and measured execution:
- Qwen0.6 CustomVoice has retained CPU FP32/eager setup and execution evidence in
  qwen6_tts; this campaign is still unrun.
- Historical source-screen32 produced 16 CPU FP32/eager Qwen1.7 VoiceDesign clips.
  This demonstrates that family’s route, not this CustomVoice checkpoint’s quality.
- CosyVoice3 official implementation selects CPU and CPUExecutionProvider without
  CUDA. GPU is not a hard requirement for the unaccelerated path. No local run is
  claimed. Disable fp16, TensorRT and vLLM for a CPU probe.
- Current assistant executor has no Docker or Torch and ~10GB physical RAM;
  it cannot satisfy the retained 12GiB container + 2GiB host-reserve requirement.
  Use the existing public GitHub ubuntu-24.04 runner path, not a paid runner.

## Exact identities, voices, controls and rights

Qwen0.6: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice,
85e237c12c027371202489a0ec509ded67b5e4b5 (retained model lock).
Qwen1.7: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice,
0c0e3051f131929182e2c023b9537f8b1c68adfe (official metadata observed 2026-10-14).
Both lanes use Vivian, Serena, Uncle_Fu, Dylan; the latter is Beijing-dialect tagged.
They are model-provided named presets, not personal reference uploads. Their
speaker lists must also be verified from the actually loaded model before calls.
Use the exact Hanzi 你好小窝 / 小窝小窝, Chinese, no punctuation, empty instruction,
matched sampling settings and two seeds. 0.6B lacks the 1.7B instruction-control
feature. No supported Qwen phoneme/pinyin API was found in official documentation;
do not treat Romanized text as guaranteed pronunciation control.

CosyVoice3 candidate: FunAudioLLM/Fun-CosyVoice3-0.5B-2512,
29e01c4e8d000f4bcd70751be16fa94bf3d85a18. Its official snapshot contains NO
spk2info.pt; frontend returns an empty speaker dictionary when absent. The base
model is not the older CosyVoice-300M-SFT four-preset model. The official example
uses asset/zero_shot_prompt.wav and a transcript; four distinct licensed voices
have NOT been established. Cosy lane is blocked, not filled using invented voices,
user audio or four copies of the same speaker. The source does support Chinese
pinyin pronunciation inpainting (official example uses [j][ǐ]); do not introduce
this extra factor into the baseline comparison. A separate matched intervention
would require a revised within-budget plan after reference rights are established.

Official model cards declare Apache-2.0. Preserve notices for code/model distribution.
A code/model license does not itself establish ownership/consent for arbitrary
reference recordings or confer voice/personality rights. Use only authorized
presets or specifically verified official reference assets. No personal audio,
paid APIs, cloud endpoint subscriptions, or user data uploads.

Sources:
- https://github.com/QwenLM/Qwen3-TTS (Custom Voice Generate / speaker table)
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/cosyvoice.py
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/frontend.py
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/example.py

## Executable route

1. Review this new directory and retained dependency hashes. Install workflow.yml
   as .github/workflows/tts-quality-probe-20261014-v1.yml through the existing
   protected-branch review/merge route. Do not create its trigger branch yet.
2. Once reviewed source is present, create research/tts-quality-probe-20261014-v1
   at that exact commit using the existing connector branch-creation route.
   Only a first public-branch push, workflow run number 1, first run attempt is accepted. No workflow_dispatch
   or secret is required. Public ubuntu-24.04 only; no paid fallback.
3. Matrix runs qwen06 then qwen17 (max-parallel 1), one setup and one real call each.
   Allowlisted receipt artifacts upload on failure and retain for 14 days. No repeat of a failed lane.
4. Review resource receipts, raw audio, native EOS, clipping/silence, exact syllables
   and naturalness. A root-reviewed continuation must import the same durable probe
   receipt/claim and use phase remaining to call cells 1..15, never regenerate cell 0.
   A continuation workflow and artifact transfer are intentionally not activated here.

run_probe.py reuses source-screen32 host_capacity, pinned image pull, container_args,
container inspection and supervisor. setup_adapter.py reuses its setup binder and
qwen6_tts 92 exact package inputs / 17 source member hashes; only new checkpoint
binding changes. qwen06-model-lock.json and qwen17-model-lock.json contain all 13
exact filename/size/SHA256/revision URLs. Small official text assets were hashed;
large weight hashes came from official LFS metadata, not downloaded weights.
Runtime verifies all bytes before load. Source versions remain intact.

## Bounded free execution budget

Monetary cap: zero. Probe campaign: 2 model calls, 0 ASR, 0 retry; max 45min/job,
2 jobs serial. Each container: 4 CPU, 12GiB memory, no swap, 256 PIDs, unprivileged,
read-only root/code, no capabilities, seccomp and no-new-privileges; inference
network none. Setup max 1200s; model load max 1200s; each generation max 300s.
Per-engine job wall budget inside launcher 2400s (workflow outer 2700s).
Acquisition uses retained ~6GiB input limit per engine including image reservation;
package + checkpoint declarations are checked before transfer. Qwen0.6 snapshot
2,498,388,392 bytes; Qwen1.7 4,520,218,951 bytes. 16GiB free disk required for setup.
Setup/working tree cap remains 12GiB; installed 3GiB; build temporary 2GiB.
Public artifact cap 32MiB/lane. Artifacts contain only authorized synthesized
preset audio and allowlisted evidence; no model weights or setup stderr.

Full contingent design remains 48 slots maximum = 3 engines ×4 voices×2 texts×2 seeds.
Currently 32 slots are concretely specified, Cosy16 blocked. Any probes count inside
these limits. ASR budget belongs to a new experiment and must not reuse old r5 controls.
Do not start ASR from this probe workflow.

## Quality endpoint

Preserve raw 24kHz float output without trimming/gain repair. Keep its SHA256,
wall seconds, EOS, duration, peak/RMS, clipping fraction and flags. ASR handoff
must derive 16kHz separately and bind both hashes. Technical success is not qualified
speech. Qualified yield requires blind human confirmation of exact target phonetics,
no added/missing syllables, natural delivery and no tail artifact. Report qualified
count / attempted count, failed-slot count and per-engine/per-voice denominators;
ASR agreement is a secondary diagnostic. With 16 per engine this is exploratory,
not a statistically established ranking. Human review defaults PENDING, qualified
false, training_admitted false. Keep duplicate/family speaker groups out of both
training and heldout test splits.

## Tests and limits

Run: python -m unittest discover -s research/tts-quality-pilot-v1 -p 'test_*.py' -v
Eight static adapter/matrix/quality gates/model-lock/retained-pin and clean-environment subprocess tests pass. Compilation
passes. Docker, dependency installation and real inference are NOT locally tested.
No performance, quality or free-host availability guarantee follows from these tests.
