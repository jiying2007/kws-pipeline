# Melo6 output-signature recovery source

This is the pre-execution source checkpoint for a narrowly reviewed guard repair.
The original load returned, then the output metadata equality guard failed; all
six synthesis calls remained unused. Its actual loaded dimensions were not saved.
The recovery records them before validation and permits only the exported batch
N and channel S to specialize to integer 1. Output name, float dtype, rank,
other dimensions, inputs, metadata, and actual output (1,1,T) remain strict.
The pinned final Conv weight has shape [1,16,7], followed by Tanh. No voice,
seed, tone, noise controls, model, dependency, call limit, or ASR decoder changed.

The historical projection below describes the preserved earlier checkpoint.
Current hashes and scope are in public-projection.json. User permission has been
received; the ASR release is armed only when all six actual blind files and hashes
are verified. asr-release-template.json stays deliberately disarmed for tests.
Actual execution outcomes, including the first failure, are separate evidence.

## Preserved preparation documentation

# Prepared Melo six-phrase source

This is a source-only, unapproved diagnostic candidate. The adapter defaults to
disabled; the paired release is `approved=false`, both blind-input hashes are
null, and no blind inputs, model weights, or generated speech are included.
The local workflow files are retained, with execution blocked by the absent
inputs and unapproved release. Preparing this tree does not authorize
execution or publication.

## Exact source and identity

`run_melo6.py`, `conversion_recipe.py`, `pack_blind.py`, the comparison helper,
all existing tests, input arrays, dependency locks, and the complete ASR tree
are byte-identical to the reviewed candidate. One TTS provenance value was
replaced with its public source URL, causing the public adapter freeze and its
dependent plan binding to acquire new hashes. `public-projection.json` records
each original-to-public hash mapping. This is not a claim of byte identity for
the changed JSON files.

The diagnostic-only v2 candidate additionally retains a private exception chain
in `runtime/private-raw-cause.json` (serialized size at most 64 KiB, file mode
0600, no captured frame locals, within existing receipt/evidence budgets).
This file is excluded from every public source/data/artifact allowlist. Only
safe retention status, size, hash and truncation flags may be public; raw
exception messages and paths must remain private. Pure fake-exception tests
cover this behavior. No scientific control or call budget changes.

The public `execution-plan.json` is a source projection with valid bindings to
this tree. Its hash is
`d3cd5e4a1c5c816e02c42b915b7930868ebe686ce4b8d57d9c721225af37cc21`.
The retained original v2 preparation plan is
`e2bafa18888149fcb5ef7f6a8758df8ab7ee321d26e2a3623f4ab922856da35c`.
The proposed binding choice is to use the reproducible public plan above for
the eventual single local execution. The unapproved paired release therefore
points to that public plan and public adapter freeze. This choice is subject
to final binding review and explicit execution/publication permission; it is
not approval or an executed state. The original v1/v2 plans remain intact as
unexecuted preparation evidence, with explicit original-to-public mappings.
No generation receipts exist, and none are relabeled. If this binding choice
is accepted, every future generation, blind-export and comparison receipt
must use the public plan hash consistently. This projection does not
authorize any additional generation.

The ASR/supervisor origin is public
[PR 482](https://github.com/jiying2007/kws-pipeline/pull/482), exact head
[`16ba0941bd7d7d6705d1ee94bb433dbd4fc6aa51`](https://github.com/jiying2007/kws-pipeline/tree/16ba0941bd7d7d6705d1ee94bb433dbd4fc6aa51/research/qwen6_asr).
The retained source hashes are in `adapter-freeze.json` and
`../melo6_asr/candidate-freeze.json`; the projection manifest records the
original ASR contract hash and its already reviewed Melo adaptation. Legacy
Qwen wire-schema labels remain compatibility identifiers.

## Fixed diagnostic

The exact six inputs are in `frozen-six-inputs.json`, in this order:
你好小窝; 小窝小窝; 你好小屋; 小屋小屋; 你好你好; 今天天气很好.
Keep dictionary tones including 你好 3+3, every phone/tone/blank, `sid=[1]`,
`noise_scale=0.6`, `length_scale=1.0`, `noise_scale_w=0.8`, seed 0 once before
the one CPU session, batch 1, intra-op 2/inter-op 1 threads. The two stochastic
graph nodes mean waveform bit determinism is unproven. No other voice, tone,
seed, speed, repeatability call, retry, or fallback is included.

TTS has at most six calls, one attempt per phrase, 60 seconds per call,
600 seconds overall, a sampled 2-GiB process-group RSS stop threshold, and
20 MiB of total evidence including logs (4 MiB maximum) and the blind pack.
Setup is separate: 300 seconds, 32-MiB response bodies, 256-MiB expanded
wheels, 1-GiB task-local disk, three exact wheels totaling 23,783,002 bytes,
and no retry. `SETUP-AFTER-APPROVAL.txt` retains the full reviewed procedure;
native loader and session compatibility have not been established.

Keep native mono float32 44.1-kHz samples unchanged, at most 10 seconds each.
The only derivative uses SciPy 1.17.0 `resample_poly` on float64 values with
160/441, Kaiser 5.0, constant-zero extension; PCM16 is floor(x*32768) then
saturation and little-endian encoding. No gain, trim, denoise, or duration
padding is permitted. Invalid or partial generation stops before ASR.

ASR retains its exact 124 package inputs, 13 model assets, scientific decoder
code and CPU controls. The two recognizers receive only six opaque 16-kHz
PCM WAVs, at most 12 calls total, without intended text, phone/tone arrays or
voice identity. The job is bounded to 50 minutes; controlled bodies total
3,340,966,447 bytes with a 4-GiB cap. No private-local ASR workaround or GPU
wheel is included. See `../melo6_asr/README.md` and its exact locks.

Freeze raw outcomes before any intent join. `compare_melo6.py` requires all
six cases and both complete exact normalized transcripts for weak lexical
support only. It does not itself verify all saved files; verify generation,
blind-input and both raw-result freezes independently first. A failed,
missing, incomplete, or conflicting row stays quarantined and stops the
screen. Even a complete lexical pass stops for separate human review.
Human pronunciation and acoustic tail truth remain UNKNOWN; CTC targets
remain null, with no training admission, speaker split, source advancement,
KWS-improvement claim, or automatic retry.

## Public model and dependency pointers

The exact ONNX model is
[csukuangfj/vits-melo-tts-zh_en, revision a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/model.onnx](https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/model.onnx):
170,429,550 bytes, SHA256
`bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d`.
`onnx-static-metadata.json` records the inspected graph identity; it is not a
model body. The already verified model is reused without a new download.

The upstream [MeloTTS-Chinese card](https://huggingface.co/myshell-ai/MeloTTS-Chinese/blob/af5d207a364ea4208c6f589c89f57f88414bdd16/README.md)
and [configuration](https://huggingface.co/myshell-ai/MeloTTS-Chinese/blob/af5d207a364ea4208c6f589c89f57f88414bdd16/config.json)
identify the source model. Fixed controls follow the
[pinned sherpa example](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/scripts/melo-tts/test.py).
Exact CPU wheel URLs, sizes, versions and hashes are in `dependency-lock.json`
and `requirements-runtime.txt`; ASR dependencies/models are separately pinned
in `../melo6_asr/runtime-lock.json` and `model-locks.json`. No resolver, install,
download, runtime import, session, or inference is part of source verification.

Retained notices: `MeloTTS-LICENSE` (MIT), `Sherpa-onnx-LICENSE` (Apache-2.0),
and `../melo6_asr/FUNASR_MODEL_LICENSE` with its `NOTICE.md`. These have
different scopes and do not establish training-corpus rights, participant
consent, voice lineage, speaker novelty, or a license for the whole harness.
Package licenses remain in their upstream distributions; no dependency or
model body is redistributed here.

## Offline checks

From `research/melo6_tts`, with the already available NumPy/SciPy used by the
reviewed pure fixtures, run:

```sh
python -B -m unittest -v test_conversion_recipe test_adapter_mock test_compare_melo6 test_private_raw_cause test_public_source
```

These tests use fake sessions and in-memory numeric fixtures, never speech
models. To run the standard-library-only source checks, use
`python -B -m unittest -v test_public_source`. Do not install dependencies to
run these preparation checks.

## Remaining release inputs after permission and successful generation

1. Resolve the proposed public-plan binding during final review. Only after
   approval, execute that exact plan once and use its hash in all six
   write-ahead claims, generation receipt/freeze, blind export receipt and
   later comparison bindings. Record actual bytes and hashes; none can be
   supplied by this source-only preparation.
2. Validate every saved generation file and successful six-row receipt, then
   pack the six canonical PCM derivatives into `blind-inputs.zip` plus
   `blind-input-freeze.json`. Each `job.json` clip needs its opaque ID,
   content-addressed audio path and actual WAV hash; the freeze needs actual
   job hash and every member's size/hash. Keep intended text outside this pack.
3. Independently verify the exact two-file `research/melo6_blind/` handoff and
   set `blind_archive_sha256` and `blind_freeze_sha256` in the release from
   those actual bytes. Recheck the approved public `plan_sha256`, projected
   `tts_candidate_sha256`, unchanged `asr_candidate_sha256`, and fixed 6/12
   call limits. Set `approved=true` only under explicit approval covering
   execution and this exact public scope.
4. Verify the retained `../melo6_asr/workflow.yml` exactly matches
   `.github/workflows/melo6-source-screen.yml` in the source tree.
   Recheck the full source/input membership, hashes, safe disclosure, and
   2-MiB source cap before the one first-branch push. No branch, workflow
   deployment, or external write has been performed by this preparation.

Proposed destinations remain public `jiying2007/kws-pipeline`, branch
`research/melo6-source-screen-v1`, and the bounded evidence archive in public
`jiying2007/kws-data`, branch `research/melo6-results-archive-v1`, path
`research/2026-10-05-melo6-source-screen/`, each via draft PR without merge.
After a completed ASR job, the archive still needs actual raw-result freezes,
verified comparison bindings/results and safe resource/failure receipts.
The private raw-cause receipt is always excluded. Raw child logs require
independent disclosure review. The source projection
does not include approval, generated-result placeholders, or that archive.
