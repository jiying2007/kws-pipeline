# Qwen 0.6B non-clone 64-cell expansion (source preparation)

This is a new, one-shot, controlled noncommercial research campaign. It does not retry, resume, or substitute cells from the consumed four-call pilot. Preparing these files does not activate a workflow or run any model.

## Frozen intended design

- Model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, revision `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Four preset voices: Vivian, Serena, Uncle_Fu, Dylan. Each worker checks that all four appear in the loaded model's supported-speaker list before claiming any call. No reference voice, cloning, speaker substitution, or instruction-control claim.
- Eight texts, in order: 你好小窝, 小窝小窝, 你好小好, 你好小你, 小窝你好, 小好小窝, 你好你好, 小小窝窝.
- Two independent seed values for each voice/text pair, `2026110400 + global_ordinal`, ordinal 0 through 63, voice then text then repeat order.
- 16 positive-intent cells, 48 near-miss-intent cells. The negative requested strings contain neither complete wake-token sequence. All requested characters are in 你/好/小/窝. These near misses do not solve OOV-negative coverage.
- Exact retained generation kwargs, including empty instruct and max_new_tokens=120. No retries, resampling, or substitutions. One exclusive durable attempt claim per cell, including calls that fail.
- Intended use is machine-weak-label, train-only research. Requested text is intent, never proof of the speech actually produced. Every admission field stays false; no evaluation/production admission or human-gold claim.

## Isolation and budgets

A single Ubuntu 24.04 GitHub job has a 90-minute outer timeout. The launcher starts its approximately 4800-second inner budget before image acquisition, reserves the last 60 seconds for cleanup, and uses the unchanged retained supervisor. It performs setup once and then four sequential fresh containers with 16 cells each. Every lane has separate output and claims and its own 0..16 progress index. Runtime/code are readonly during inference; network is disabled; no host credentials are forwarded. Container/model/runtime pins are unchanged.

Any setup, creation, configuration, generation, or cleanup failure stops further lanes. An ambiguous create/cleanup failure is not retried. Global deadline exhaustion may leave later cells not_run, and is a campaign failure, not a smaller successful sample. Container and Docker-operation deadlines are bounded independently, so the inner threshold is approximate rather than a real-time OS hard kill; the 90-minute job timeout is the final outer bound. Image-transfer accounting retains the old monitored reservation, not a hard daemon transfer cap.

Terminal accounting always intends all 64 cells, with claimed failed attempts and unrun cells retained. Untrusted/corrupt receipt reconciliation itself fails closed rather than silently inventing successful outcomes. Public export is an explicit filename allowlist of preset synthetic WAV and JSON evidence. It excludes claims, arbitrary JSON, private input, raw logs, package/model trees, and weights. WAV files are bounded to native 12-second/24-kHz mono float geometry plus header reserve; the whole exported artifact including index is capped at 96 MiB. Public outputs are not proof of an independent trust root.

## One-shot launch identity (not activated here)

Only first creation of `research/qwen-nonclone64-20261104-v1` in public `jiying2007/kws-pipeline`, push event, run_number=1 and run_attempt=1 is eligible. There is no manual-dispatch, schedule, or pull-request model trigger. The workflow has read-only contents permission and non-persisted checkout credentials. Do not create this branch until the complete TTS source freeze has been reviewed. A consumed run must not be rerun.

The source lock includes the new launcher, worker, tests, documentation, workflow, and byte-identical retained helpers. It also records the retained pilot source-lock digest. The new source lock is a reviewed-source checksum manifest, not an independent provenance authority.

## Offline verification

Run from repository root:

    for opt in '' -O -OO; do
      python3 -I -B $opt research/qwen-nonclone64-20261104/test_campaign.py
    done

Tests use invented waveform-like values and mock model/container calls. They never import Torch or install packages, download weights, contact services, or produce real speech. Python 3.8 grammar compatibility is checked using ast.parse; this is not a real Python 3.8 interpreter/runtime test. Tests cover exact plan/seeds, preset preflight, call claims, no retry, fixed denominator, source hashes, public allowlists, lanes/mounts, cleanup-stop behavior, and deadline exhaustion.

## Review caveats and unresolved admission

The separately prepared native capture concerns the ASR thinker's effective GenerationConfig, not TTS generation. It is not an integration prerequisite for this TTS-only workflow: once the TTS source freeze is reviewed, the new audio can be generated once and independently screened later. The TTS worker retains the pilot's source-values hash and WAV metadata. Missing ASR native-runtime evidence, independent trust root, independent audio-only labeling, and OOV-negative coverage remain downstream training-admission blockers. Generation does not imply training admission. There are no new ASR calls or training runs in this workflow.
