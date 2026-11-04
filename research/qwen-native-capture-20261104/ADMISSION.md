# Prospective native evidence, no launch approval

The observation-only wrapper is now insufficient for its Qwen `run` entry:
`native_sink_factory` is required before any worker call. There is intentionally
no standalone CLI path that supplies that factory yet. No new ASR workflow is
enabled by these files.

## Zero-forward preparation

After the unchanged scientific loader returns a verified live model, call
`prepare_strict_identity(loaded, output, executed_head, source_bodies)`.
It derives model identity from the actual verified loaded asset inventory, hashes
the supplied executed source bytes and invokes the pinned native thinker's
`_prepare_generation_config` with the exact locked top-level dispatch kwargs.
It never invokes generate or forward. It records the actual complete resolver
return and native greedy mode. Source closure completeness remains a reviewed
host responsibility; do not substitute a caller-authored truncated inventory.

Sense and Whisper must also prepare their actual configs before any recognition.
Freeze the complete three-model recognizer policy from those preparations.
Runtime preparation is not model inference, but it still requires authorized
loading/setup and is not performed by source tests.

## Shared sink

`NativeEvidenceSink(root, policy_raw, model_raw, source_raw, recognizer_id,
source_bodies, verified_assets)` validates the existing native policy validator
and retained lock byte bindings. It never grants trust-root admission.

`begin(bound, source_id, request_id)` consumes one opaque audio identity.
`start_native(actual_config)` checks the actual config against the frozen policy
and writes immutable `retained-asr-started-v1` before forward.
`complete_native(actual_text, actual_capture, original_evidence)` validates the
existing native capture contract and persists `retained-asr-raw-transcript-v1`
and `retained-asr-completed-v1`, including original start/output byte hashes,
input frame bounds, model/source inventories and native capture.
`fail()` prevents any later success. Source/asset authentication and chronology
are provided by the verified prospective wrapper, not arbitrary JSON inputs.

Qwen's BoundCapture optionally accepts this sink and an audio-only request map:
`{opaque_id: {source_id: ..., request_id: ...}}`. It calls `begin` before inference,
then its actual native resolver return hook checks the complete snapshot against
the prefrozen policy and writes the real start before returning control to
generate. Its successful completion uses the real bound PCM, returned token IDs,
EOS metadata and decoded text; target and peer transcripts are never supplied.

## Unresolved launch gates

- The host must integrate all three zero-forward preparations, freeze their
  policy, reload for bounded inference and enforce exact matching configs.
- Actual 64 generated audio bytes and derivation/source identities must be
  finalized and reviewed; invented fixtures are not that input lock.
- Sense's observed frontend snapshot is runtime-derived, not model or source
  code. Existing resolver membership requires a separate reviewed extension;
  it must not be falsely listed as an original asset or executed source.
- Retain source bodies and native dependencies at their exact inventory member
  paths in controlled evidence for the resolver. Weights need not be copied.
- Full capture and raw transcript persistence beyond the runner is unresolved.
  Do not upload them publicly or invent another destination. A future public
  artifact must contain only approved aggregate status, never recursive payload.
- A reviewed train root, lawful source policy/rights, all three complete native
  evidence chains and split checks remain necessary. No test grants admission.

The previous legacy observer records remain original screening evidence. No
historical run is relabeled, reconstructed, retried or silently promoted.
