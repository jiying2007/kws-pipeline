# Prospective SenseVoice effective frontend capture

Local source preparation only. No model/package imports, downloads, execution, remote writes, or trust-root edits are performed by the tests. Original loaders, runtime locks and model pins remain unchanged. These records are additional observations, not a renamed native-v2 admission certificate.

## Integration API

`run_wrapped(legacy, sense_adapter, input_root, output_root, runtime_root, parser_path, reviewed)` wraps the caller-supplied, source-verified retained worker and Sense adapter. The caller must ensure `sense_adapter` is the same module the retained worker imports, and freeze this module plus integration source before execution. The wrapper restores the exact loader, adapter and native inference references on success/failure. The fixed-container CLI is described below; no enabled workflow is provided.

The `reviewed` argument is a pre-reviewed exact mapping for these eight keys, each containing only `sha256` and integer `bytes`:

- frontend_source: actual loaded WavFrontend.forward source file.
- fbank_source: actual loaded kaldi.fbank function source file, resolved from forward's globals.
- backend_package_source: the actual kaldi-native-fbank package entrypoint.
- backend_binary: the module binary owning the actual OnlineFbank class. Resolution uses its loaded module identity, never a guessed filename/glob.
- cmvn_file: actual loaded frontend.cmvn_file bytes.
- tokenizer_file: the retained runtime's tokenizer_file, also byte-compared against the loaded SentencePiece object's serialized model.
- parser_source: caller-supplied source-verified parser path.

This repository intentionally does not invent the reviewed manifest. Integration must obtain/review these identities from the exact pinned source/package/asset bytes before execution. The runtime's own observations cannot grant trust to themselves. Missing dependency identity, unexpected backend layout, or an unavailable actual file is a hard failure, not a fallback to defaults. Package/model bytes are not copied by this module.

## Observations and chain

At the retained adapter entry, a durable exclusive claim binds opaque ID, WAV SHA256, PCM16 SHA256, float32 PCM SHA256, frame count, input-binding SHA256 and decoder-manifest SHA256. Attempts are never retried; any capture/inference/persistence failure poisons this runtime observer.

The observer intercepts the actual loaded model.inference call. Before passing control through, it checks the exact request arguments, input PCM shape and bytes, the actual frontend identity, and the retained CapturingTokenizer's underlying tokenizer identity. It reads these actual frontend attributes without supplying defaults: fs, window, n_mels, frame_length, frame_shift, filter_length_min, filter_length_max, lfr_m, lfr_n, dither, snip_edges and upsacle_samples.

The loaded CPU float32 CMVN tensor's shape, values and canonical little-endian float bytes hash are recorded alongside the actual CMVN file identity. No CMVN values are reconstructed from assumptions or old evidence. Dependency identity and frontend configuration are rechecked after inference; drift fails closed. The dither random-state reproducibility remains explicitly unestablished.

Each lane writes under `output_root/sensevoice/native-capture/`:

- claim-NNNN.json: written before the retained adapter/native call.
- frontend-NNNN.json: immutable observed snapshot, fsynced immediately before native inference.
- completed-NNNN.json: successful snapshot/claim/evidence linkage.

The wrapper attaches `prospective_frontend_capture` to the decoder evidence returned to the unchanged worker. That worker's ordinary decoder-evidence hash and completion receipt then bind this observation. The completion file additionally hashes the original decoder evidence before adding the capture field. Every admission flag remains false. Keep these runner-local payloads private unless separately authorized for retention/publication.

## Limits

No ASR campaign has been activated, and the fixed-container worker has not been tested against a live loaded model. It does not establish speech correctness, acoustic completeness, independent source authority, training admission, or reproducible dither RNG. It captures the frontend's effective object attributes and CMVN, not every transient internal native-fbank options object. The exact backend source/binary identities are necessary to interpret inherited native behavior. A subsequent integration review must decide whether additional call-time backend-options observation is required by its admission contract.

Native-v2's resolver also requires retained dependency bodies and source/asset inventory references; hashes alone are insufficient. The wrapper does not change that resolver. With an explicitly supplied reviewed strict sink, new original start/completion records are written prospectively, never reconstructed from past runs. A future strict source policy must truthfully bind these new original observations, retain necessary dependency bytes, and separately review source rights. Historical SenseVoice records remain unchanged.

## Offline tests

Run test_sense_capture.py with Python normal, -O and -OO. The 28 tests use invented scalar/tensor/PCM fixtures and mocked runtime objects, plus source-file/module-identity fixtures. They cover snapshot-before-call, complete receipt linkage, real reference restoration, missing fields, dependency/CMVN/tokenizer drift, source aliases, changed requests/PCM, duplicate and missing calls, failures/poisoning, and Python3.8 grammar. No Torch, NumPy, FunASR or model import occurs. Grammar compatibility is not a real Python3.8 runtime test.

## Zero-forward preparation and strict sink

`prepare_loaded(loaded, output, parser_path, reviewed, source_lock_bytes, asset_lock_bytes, max_audio_frames, metadata_groups)` reads an already-loaded runtime without calling model/frontend forward. It checks observed CPU/float32 parameters, actual batch size and tokenizer semantics, writes private reviewed dependency bytes and an immutable frontend snapshot, and emits `proposed-native-config.json`. `metadata_groups` must come from the already source-verified pinned parser contract. This is not a substitute for a fresh runtime loader or an authorization to run one.

`frontend-observation.schema.json` freezes the snapshot's exact structure, actual value type labels and origin. The loaded tensor's values, shape and float32 hash are observed, never filled from historical defaults. `frontend_config_sha256` binds these runtime-derived bytes. They are explicitly not an original loaded model asset or an executed source file.

The optional `BoundSenseCapture(..., prepared=..., strict_sink=..., identity_map=...)` integration uses the shared reviewed NativeEvidenceSink generic API. The identity map must be pre-frozen, audio-only, with explicit source_id and request_id for each opaque clip. It calls begin before the adapter, start_native only after the actual input/config checks and before the native model call, and complete_native only after actual returned CTC/token/PCM evidence validates. Failure calls sink.fail and prevents further successful use. `run_wrapped` accepts and forwards these three optional parameters.

The shared sink owns genuine retained-asr-started-v1/completed-v1/raw-transcript-v1 serialization and full three-recognizer-policy binding. It must receive the entire pre-frozen policy before any recognition call. This module supplies the observed native_capture, not an after-the-fact policy. Tests use an invented sink to verify call order and the unchanged retained native-v2 contract to validate emitted fields. They do not prove end-to-end resolver admission.

## Proposed narrow resolver review (not implemented)

Add a specifically named Sense runtime frontend observation dependency exception, analogous to Qwen's resolved GenerationConfig exception, without broadening existing source roots. For frontend_config_sha256 only, require exact private retained bytes satisfying the frozen runtime schema; verify every nested dependency hash/size against retained source/loaded-asset/runtime-package identities and require each referenced body. Verify CMVN shape, finite float32 values and canonical tensor hash. Bind the observation's config hash to the pre-recognition three-ASR policy and the started/completed capture chain, and compare the actual pre-forward observation unchanged. Continue rejecting arbitrary runtime JSON, missing bodies, invented membership, post-recognition policy freezes and unsupported old records.

Until that extension and an independent source policy/root receive review, the frontend snapshot remains an honestly captured runtime dependency with strict resolver membership pending. The old resolver and trust root are untouched. No snapshot is inserted into source files or loaded assets to bypass the membership check.

## Two-phase fixed-container worker (source-only)

`worker.py --phase prepare` or `worker.py --phase infer` uses fixed `/input`, `/output`, `/runtime` and `/code` paths. It has no arbitrary command or remote action interface. `/input` remains the blind job/audio directory only. Read-only policy files are under `/code/runtime-policy/`, avoiding an unreviewed new mount.

Before importing/loading the legacy model path, `sense-review.json` is mandatory: exact schema `reviewed-sense-runtime-capture-v1`, executed_head_sha, dependencies (the eight pre-reviewed identities) and sources (each exact path/member/bytes/sha256). Source paths must be in `/code/` or `/runtime/` and must match the pre-reviewed hashes. Missing or malformed dependency review fails before the loader. Review manifests are not generated from runtime observations.

Preparation runs the unchanged verified loader, observes actual loaded configuration, writes private strict identity/model/source lock bytes and dependency observations, then uses an internal BaseException sentinel to stop before any forward. The retained worker may write its ordinary stopped/no-attempt summary during cleanup; `phase-summary.json` distinguishes successful zero-forward preparation. Inference performs fresh zero-forward preparation again, then the shared strict sink must match the fully frozen three-ASR policy before forward. `/code/runtime-policy/recognizer-policy.json` and `identity-map.json` are mandatory for inference. The map uses schema `blind-asr-source-request-map-v1`, exact clips entries audio_id/source_id/request_id/wav_sha256, with complete current batch coverage and actual WAV association.

The legacy supervisor progress path, 300-second per-cell alarm, one-attempt claims and stop-on-first-failure behavior remain intact. The new worker also fails if the retained inference summary is not complete. Outputs include `/output/sensevoice/prepare/identity.json`, model-lock.json, source-lock.json, proposed-native-config.json and private dependency snapshots; strict inference adds the shared sink's original records. There is no public payload export here.

The current full local test set is 28 capture tests, 25 adversarial verifier tests, 15 worker orchestration tests and 11 independent-review regression tests, each passing normal/-O/-OO. This is 79 distinct mock/static tests, not a real model-run result or complete integration test of the independently evolving shared sink/host.

Independent review fixes: the producer, schema, worker review and verifier uniformly enforce the existing 32MiB backend member ceiling. The proposed resolver guard rejects non-string recognizer IDs with ValueError before dictionary lookup. Strict inference now persists the original private source_bodies at their exact source-lock member paths under strict-native, with no overwrite or path traversal. Recovered dependency identities are documented separately; this is not a completed review manifest.

The eighth dependency, backend_core_library, is the actual kaldi_native_fbank/lib/libkaldi-native-fbank-core.so. Static ELF review confirmed the extension DT_NEEDED edge. Both native binaries must be executable mappings in the current process with matching original path/inode/device; only booleans are retained, not raw process maps. No other process is inspected. System ABI libraries remain within the existing pinned-container boundary, not newly claimed per-file measurements. reviewed-dependency-identities.json now contains all eight exact reviewed identities.

Inference privately preserves tokenizer and CMVN under their original model inventory names, and parser/package/native code under exact executed source inventory member paths. native-dependencies.json records member-only assembly metadata with archive_id=null until authenticated host provenance supplies an authorized archive identity. The runtime frontend snapshot stays separate and is not inserted into either original inventory.

Final source-closure hardening: before loading, source_closure verifies the already-reviewed original model-locks.json and its canonical Sense source_lock hash, resolves all seven originally pinned FunASR package members via installed distribution metadata without importing those packages, and retains their exact verified original bytes. Existing reviewed runtime paths are deduplicated only when bytes agree. Missing lock, installed-source drift or ambiguous aliases fail closed. CLI SystemExit is preserved so isolated --help exits successfully without a false failure receipt.
