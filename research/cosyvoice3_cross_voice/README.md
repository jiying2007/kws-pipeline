# CosyVoice3 fixed30 cross-voice research batch

One synthetic-data experiment, not a trained model or production qualification.
Five existing public Qwen stock-synthetic references condition plain CosyVoice3
zero-shot synthesis: Eric/Serena/Vivian train18, Uncle_Fu development6, Dylan
sealed-recording holdout6. Each receives K1你好小窝, K2小窝小窝, repeated你好你好,
single小窝, 你好小屋 and小屋小屋, exactly once with fixed per-row seed.
All source voices and historical examples are already exposed; the proposed
holdout can only be fresh recordings, never unseen speakers. No natural-person
reference, ASR, KWS baseline, training, threshold sweep or inpainting is included.

The first Eric K1 is the included technical smoke. There is no hidden warmup
synthesis, retry, reseed, replacement reference or second generation batch.
A successful complete batch preserves all30 outcomes before the user listens to
only four Eric/Serena K1/K2 clips. These short1.2–1.92s references are structurally
allowed by the inspected API but new conditioning quality is not guaranteed.
Wrong/uncertain wakes cannot become assumed-correct supervision. Four accepted
clips do not label the other20 train/dev recordings.

## Execution controls

Default arm.json is disabled. The push-only workflow first runs stdlib tests.
The only allowed activation changes arm.json from the reviewed disarmed parent,
binds all other source/config/runtime-lock bytes by SHA256, and runs once on the
fixed public research branch. No PR event, workflow_dispatch, rerun or previous
non-skipped heavy job can admit another batch. Source/runtime/model verification
and the historical dependency/source qualification remain before any TTS call.

Offline worker enforcement uses the historical Python socket monkey-patches and audit hook, Hugging Face offline flags/local-only loaders, and empty-attempt receipts; it is not OS/firewall or native-subprocess egress isolation. No system/network security setting is changed.

Same historical pinned model5,427,029,103B and package closure3,007,556,262B;
Torch intra-op4/inter-op1, ORT1/1, CPUFP32. Standard ubuntu-24.04 at most4vCPU,
12GiB sampled process-tree RSS,2GiB host-memory reserve,14GB effective whole-job
storage with1GiB disk reserve,128MiB combined retained/staged/upload-output envelope,
20s audio/clip,300s/call,900s startup. The controller work deadline is55min,
reserving cleanup/packaging time before the workflow's60min job deadline.
60min×4vCPU gives4core-hours theoretical capacity; no aggregate CPU-seconds
accounting/cap is claimed. The old12clip run observed591.37s full controller,
8.33GB RSS and11.56GB job storage;30calls are estimated15–25min, not guaranteed.

Two public artifacts are allowlisted separately: train-dev24 and sealed-six.
The latter receives identity/header QC only in the packager and no semantic
inspection. Opaque storage is procedural experimental custody, not private or
encrypted publication. Preserve its ZIP without extraction until original and
candidate checkpoint plus decoder configuration identities are frozen. No such
candidate training/evaluation is part of this generation batch.

The exact official source/model revisions, reference hashes and fixed rows are
in pilot-config.json. Historical runtime lock is reused byte-for-byte. See
RIGHTS_AND_COST.md; no credentials, private input paths, chat transcripts,
models or dependency packages belong in the artifacts.
