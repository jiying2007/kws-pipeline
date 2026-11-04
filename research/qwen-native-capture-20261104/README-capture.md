# Prospective primary worker integration

This wrapper imports the unchanged `research/source-screen32-v1/asr_worker.py`
from the sibling staged source closure. That worker retains its original source
pins, scientific loader bodies, single-attempt PCM claims, 300-second per-clip
limit, receipt hashing and stop-on-failure ledger. No historical run is edited.

The initial observer used the following invocation shape:

`python capture_worker.py --model qwen06 --input /input --output /output --runtime /runtime`

Qwen now requires an in-process strict native sink factory and audio-only request
map. The CLI does not provide them and fails before worker execution. See
ADMISSION.md for the preparation and strict prospective schema integration.
The same entry accepts `sensevoice` and leaves that inference path unchanged.
For Qwen, immediately after the verified loader returns, it checks the actual
unwrapped thinker generation source against the reviewed Transformers 4.57.6
file hash, then installs the observer on `loaded.wrapper.model.thinker`.
The official pinned Qwen model delegates its top-level generation call to that
instance. The hook never loads a model itself or replaces the native decoder.

The actual thinker.generate boundary rejects caller-supplied custom processors,
stopping criteria, prefix constraints, assistant models, alternate generation
code and negative prompts. A successful clip requires exactly one native call
and one immutable resolver snapshot. The snapshot binding records opaque ID,
WAV/PCM/input descriptor identity and decoder manifest identity inside the raw
decoder evidence, whose hash is already included in the old completion receipt.
Any missing/duplicate capture, control mutation or failed write poisons the new
wrapper and cannot be used to issue a successful result.

The source review records later native special-token, length and cache changes.
The snapshot is honestly labeled as the resolver-return stage, not final full
configuration. Required greedy controls are rechecked on the same actual object
after generation. The full native-v2 contract and train-root review are separate.

Integration requirements:

- Freeze this directory's sources plus the unchanged archived primary closure.
- Include `native-capture/resolved-generation-config-NNNN.json` and
  `native-capture/resolution-NNNN.json` and `native-capture/claim-NNNN.json`
  in the host's exact private evidence
  whitelist and archive hash inventory. The old raw-freeze hashes top-level
  files only; it does not cover these new nested members.
- Retain original failed snapshots without granting success. Preserve all old
  terminal slots. No retries, no extra model calls and no fresh TTS are implied.
- Do not publish raw capture, transcripts or the prior archive without approval.
- Source-file hash mismatch is a setup failure, not permission to relax the pin.
- This adds the missing Qwen observation only. It does not supply Sense native-v2
  frontend evidence, a reviewed train trust root, rights or independent splits.

`test_capture_worker.py` uses only invented objects. It does not import model
libraries, install dependencies, download assets or run recognition.
