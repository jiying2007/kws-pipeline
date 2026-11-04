# Prospective guarded-metadata preparation overlay

This is a new source-only campaign draft, not permission to launch. Identity:
`research/native-prepare-fence-v2-20261104-v1`. Consumed campaigns and all 88
original source identities remain immutable. The host explicitly verifies the
old preparation lock and permits only seven fixed stage replacements plus one
addition. New source bytes live here; staging projects them to the old import
paths without editing any retained repository file.

Replacements: prepare_worker.py and zero_forward.py; Sense worker.py and
sense_capture.py; Whisper whisper_capture.py; and the two prospective Qwen/Sense
source manifests. Addition: safe_diagnostic.py alongside the staged preparation
worker. The diagnostic implementation is bound in the actual Qwen manifest,
Sense reviewed source closure, and Whisper source snapshot. All unchanged
scientific loader bodies, model/runtime locks and setup guards remain fixed.

The metadata-only fence capability is independently reviewed and does not call
forward/generate/inference. Existing model-call traps and their stated coverage
limits remain. Whisper's optional on_loaded callback runs after the actual
retained loader and its byte checks, immediately before native config creation;
it never announces that stage before a failed load or after config creation.

The worker's outer diagnostic context covers guard restoration and proof writes.
Only fixed ordered stages and exact built-in exception-class enums are retained.
No messages, exception attributes, paths, traceback, environment values or raw
model payload enter the public summary. Unknown classes map to OtherException.
The host strictly reads these bounded terminal records. A missing or invalid
terminal diagnostic cannot grant preparation success. Any primary failure stops
later work; optional export failure cannot replace that primary exception.

The public artifact allowlist has exactly status.json, capacity.json and
safe-summary.json. Full native configuration/source/model evidence remains
runner-private. There is no audio, ASR inference, training admission, or change
to the unconditional full-ASR launch block.

Same-job initial free-space threshold remains 25 GiB. Original setup thresholds
remain 16 GiB primary and 8 GiB Whisper. Later capacity checkpoints are observations
only. Preserve the 1,800-second absolute inner budget, 120-second launch reserve,
40-minute single-job outer limit, first-created branch/run-1/attempt-1 identity,
and no retries. No new runner-data cleanup is introduced; retained task temporary
file and created-container cleanup remain unchanged.

All tests are source-only invented fixtures. They do not establish a successful
real preparation. Publish or run only after separate parent review and approval.
