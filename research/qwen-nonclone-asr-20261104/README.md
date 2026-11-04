# Four-file blind three-ASR screen

This fresh campaign follows the successful public-preset Qwen TTS run
41803323981 at adc2047807289dcba1ff7774c4340797fa0ae57e. It never generates more
speech. The four exact native WAVs total 8.08 seconds; their SHA256, geometry,
finite float samples and complete artifact index were independently checked.

## Inputs and isolation

The recorded derive.py uses existing NumPy 2.3.5 / SciPy 1.17.0, resample_poly
2/3, Kaiser 5, constant padding, then floor(float64 * 32768) to PCM16. No gain
normalization, crop or repair. Original and derived hashes, PCM hashes, full
frame counts and all four mapping rows are retained. Requested text remains
host-only. Workers receive only anonymous IDs, content-addressed audio paths
and digests, with no target text or other recognizer's output.

All four clips go to all three recognizers, maximum 12 one-shot calls. No
result-dependent replacement or retry. A source-verified archive projection
restores three historical qwen6_asr files to the retained worker's exact pins.
The current repository files and historical locks are not changed. The
source-screen Sense adapter already explicitly binds its runtime tokenizer;
it does not use the formerly defective default-path generic adapter. The old
wrapper retains less postprocessing failure evidence than the newer generic
adapter, and such failures are unresolved, never counted as successful labels.

## Actual reusable execution evidence

Run https://github.com/jiying2007/kws-pipeline/actions/runs/38573117922 on
2026-10-14 succeeded with 32 Qwen, 32 SenseVoice and 30 Whisper outputs. Its
variable-subset Whisper wrapper and setup are reused, not the earlier failed
fixed-16 campaign, admission, input binding or branch. New host, branch and run
identity are independent; first branch creation, run 1 and attempt 1 only.

Qwen3-ASR-0.6B revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 is
Apache-2.0 per the fixed publisher card:
https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md
SenseVoiceSmall revision 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 is by
Alibaba/FunAudioLLM, governed by FunASR Model Open Source License v1.1,
not the code project's MIT license. Retain qwen6_asr/NOTICE.md and
FUNASR_MODEL_LICENSE. This is reference/learning research, not deployment.
https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
Whisper source 86098128c0b4f24f0e2aa2994de830614b474227 and multilingual small
checkpoint use the official MIT route. The exact source, weights and wheels
are bound by the retained independent source/runtime locks.
https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/LICENSE

## Budget

One standard public Ubuntu 24.04 CPU job, at most 85 minutes. Inner deadline
4800 seconds includes reserved launch/cleanup time; no stage starts with less
than 120 seconds remaining. Each setup at most 1200 seconds, each recognizer
stage at most 1500 seconds and clipped to the absolute deadline minus cleanup
reserve; retained workers bound load to1200 and each clip to300 seconds.
Every setup or inference failure stops the entire campaign, so uncertain
cleanup cannot overlap a later container. No paid APIs, Codex or Work.

Retained isolation: nonroot, 12 GiB RAM/no swap,4CPU,256PID,private cgroups,
readonly code/root/runtime and audio, network-none inference, no worker
credentials. Primary and Whisper runtimes are separate. Require25GiB free
host disk and14GiB available RAM. Primary setup response budget4GiB; Whisper
1536MiB. These are not total network limits including Actions, Docker/protocol
traffic. The locked image uses its existing monitored128MiB reservation.
Public artifact cap32MiB of structured evidence only, no weights, packages,
raw setup logs or private inputs. Four-by-three denominator always retained.

## Meaning

Exact punctuation/space-normalized agreement is a machine screening tier,
never calibrated confidence, human gold, full acoustic coverage, training
admission or held-out FAR/FRR. All three raw outputs remain immutable before
host-only comparison. Positive/near-miss/ordinary-negative intents from TTS
are not assumed to be the words spoken. No training takes place in this run.
