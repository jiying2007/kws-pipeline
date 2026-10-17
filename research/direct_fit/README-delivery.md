# Bounded experiment result handoff

This POSIX/Linux stdlib helper (descriptor flags are not Windows-portable) accompanies the public direct-fit runner at
`62176f79c11edac7e0417c27ad74f44f7d4152fc` (PR #596). It changes no training,
source-admission, model, optimizer, existing budget or product claim. No actual
training, TTS, ASR, model/dependency download or publication was performed here.

## What it establishes

One wrapper invocation owns subprocess launch, observation of the actual exit
code/signal, collection, hashing, ZIP construction, and stdout framing. There is
no next-tool-call gap between successful child exit and result collection.
The child remains a subprocess, so its real return code/signal is preserved even
when packaging fails. A command launch failure is not described as an exit.

Only named files are read; no directory recursion or environment/command dump.
Symlinks (including ancestors), traversal, FIFOs and existing output directories
are rejected. Original file bytes, including `run.json`, are retained unchanged.
If present, `run.json` checkpoint/trace/artifact hashes must match collected bytes.
Child stdout/stderr are drained and hashed without displaying their contents.
The summary exposes only six named run fields. Their values and selected files
must still be reviewed as appropriate for the private destination; no redactor
can guarantee removal of secrets from arbitrary caller-selected files.

The returned stdout line has a schema, run ID, end marker, archive length and
SHA256, manifest SHA256, summary, and base64 archive. Default ZIP limit: 256 KiB;
maximum configurable ZIP limit: 700,000 bytes; hard frame ceiling: 1 MiB;
uncompressed input ceiling: 32 MiB. Larger results yield manifest-only evidence
and a failure, never a claim of full recovery. If even the manifest exceeds the
ZIP limit, a terminal-only manifest retains its inventory manifest hash. The
per-file hash cannot prove recoverability when the bytes are not retained.

## Integration contract

A future authorized run is launched using `result_delivery.py run --output FRESH
--file run.json --file checkpoint.pt ... -- COMMAND {output}`. Every required
output must be named explicitly. For #596 include its four checkpoint files,
config/data-receipt/schedule/run/comparison JSON, optimization trace, clip
references, and each arm's frame-trace/events JSONL. No actual run is authorized
by this documentation. The intended caller must request enough tool-output
budget for the whole frame and capture the exact line, not a summarized excerpt.

The parent must, without relying on the previous workspace still existing:

1. Capture the complete stdout frame. A truncated/missing/corrupt frame is a
   failed handoff. Never rerun a spent experiment budget to replace it.
2. Recover using `result_delivery.py recover --frame FRAME --destination FRESH`.
   This verifies the archive and every member before writing. It returns
   `verified_local_only`, with original terminal state and recovery scope.
3. Through the parent's existing authorized private file service, upload the
   reconstructed `handoff.zip`; obtain a real file identity/version and read it
   back. Compare exact bytes/length/SHA256 to the observed frame.
4. Only the parent can record delivery success after actual matching readback.
   Keep original experiment failure separate from evidence-delivery success.
   A manifest-only archive can preserve a failure witness but is not delivery of
   the complete experiment outputs. The helper cannot accept or validate a
   fabricated service receipt and has no service-specific network integration.

Public repository/CI must contain only source and artificial test evidence.
Do not publish private frames, checkpoints, logs, audio, credentials or inputs.

## Explicit limitations

Hashes establish byte identity and consistency, not authenticated experiment
provenance or Library identity. A malicious actor can fabricate a self-consistent
frame; the parent must retain its trusted tool-call context.

This is a bounded handoff, not a durable-storage backend. Local writes, ZIP
creation, exit 0, and complete stdout emission do not establish cross-environment
persistence. If the wrapper is SIGKILLed, the environment disappears before the
frame is emitted/captured, or transport truncates stdout, no recovery guarantee
exists. The terminal report is only established when captured. No periodic
checkpointing, crash resume or real-run recovery is claimed. The helper neither
reconstructs the already-lost four checkpoints nor authorizes repeating that run.

After the main child exits, pipe draining has a two-second grace period.
Detached descendants retaining pipes cause an explicit incomplete-stream failure;
partial log hashes are not described as complete logs. The wrapper does not impose a new
experiment timeout or silently retry the command. The allowlist is a caller
contract, not a proof that no other output matters. Full semantic validation of
arbitrary experiments is outside its scope. Input files must be quiescent after
command exit; descriptor checks detect common concurrent writes but do not
constitute a hostile-process snapshot guarantee. Local reconstruction is not
atomic across all files; an interrupted extraction remains unconfirmed.

## Software tests

Run `python -B tests/test_direct_fit_delivery.py --report build/direct-fit-delivery-fixture.json`.
The 19 stdlib cases generate tiny byte fixtures and Python exit/signal actions,
including a rollback test that deletes the producer directory before restoration.
No torch, audio, model, service or network operation is invoked. The existing
objective-math job runs this separately from numeric model tests. Its retained
JSON-only software receipt records executed case counts and exact source hashes.
