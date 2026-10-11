# Offline child-admission decoder overlay

This is a source-preparation and synthetic-logit test component, not an acoustic result. It makes no claim that `wo1` recognizes `wu1`, 屋, an accent, or any new pronunciation.

## Isolation and build contract

- All tracked shipping files, public headers, CMake targets, installed SDK, model bytes, defaults, and archived experiments are untouched.
- `decoder_overlay.json` binds the original source/dependencies to SHA-256 values from source tree `fafd0721c71f619f8af6327b22746b2598557b09`. It stores surgical replacement hunks, not a second complete decoder implementation.
- `prepare_decoder.py` requires every original hash and each unique patch context to match, verifies derived hashes, and emits five derived files into a fresh output directory. Source drift, ambiguous contexts, derived drift, and existing output fail closed. There is no automatic fallback, fetch, compilation, model loading, or inference.
- The derived files require explicit `-DKWS_RESEARCH_CHILD_ADMISSION=1` to enable research behavior. Defining the guard to another value is an error. Omitting it retains the original layout and behavior.
- `verify_decoder.py` resolves each translation unit's local include dependencies using the same compiler before building. Every local dependency, including the generated parameter header, must be in the exact hash-bound allowlist. Untracked quote-include shadows are rejected, and all bound input hashes are rechecked at the end. Installed system/compiler headers are outside this local-source closure.
- `verify_decoder.py` uses the installed C compiler and Python standard library. It does not install packages, read a model file, run frontend/model inference, invoke a data campaign, or publish anything. Engine integration tests provide synthetic logits directly to the existing replay bridge.

Prepare source only:

```sh
python3 research/near_pronunciation_v1/prepare_decoder.py \
  --source-root . --output-dir build-near-pronunciation/derived
```

Run the separate research compiler/unit verifier in a fresh directory:

```sh
python3 research/near_pronunciation_v1/verify_decoder.py \
  --source-root . --output-dir build-near-pronunciation-verify
```

The verifier builds `kws_decoder_replay_research` and writes `verification-receipt.json`, including all compiler commands, original/derived source hashes, overlay/preparer/verifier/test hashes, executable SHA-256, tested object-parity hashes, synthetic-state/event parity hashes, and host type sizes. No default shipping build is enabled or altered. Keep this receipt distinct from the acoustic trace producer's build identity.

## Two arms

The guarded internal enum contains only:

- `strict_children`: every non-root child advance requires the child's token to be the decoder's deterministic top-1 token.
- `single_fuzzy_wo`: the same rule, with at most one non-top-1 child advance for token 4 (`wo1`) per candidate path. That advance charges the existing `-8.25` search-only cost against the existing inclusive `-16` total retention budget.

The existing root rule is shared unchanged: top-1 root, or a root within the existing `0.5` logit margin of dominant blank/another configured root. Root ambiguity does not consume the child exception. CTC repeated labels still require a blank-separated prefix. Token duration, blank/speech retention, VAD, boundary resets, acoustic confidence, terminal arbitration, and the engine's refractory/event path are shared.

Lane 0 is the original score/acoustic-score state and has spent no exception. A separate lane 1 has spent exactly one exception. Each lane keeps independent nonblank, blank-separated, next, and next-blank score/acoustic pairs. A stronger spent path cannot erase an unspent hypothesis that has different legal future moves. Top-1 advances preserve their lane; only the permitted non-top-1 `wo1` advance moves from lane 0 to lane 1. Lane 1 cannot seed a root or take a second fuzzy advance. Every path reset clears both lanes. No global counter, text mapping, ASR admission, new token, or new acoustic model is introduced.

The internal debug copy API additionally exposes lane-1 scores and the research mode in guarded builds. These headers remain uninstalled. The replay CLI never permits policy changes within a recording.

## Research replay interface

The executable accepts the original four positional arguments and exactly one required mode:

```text
kws_decoder_replay_research model.kwm keywords.kwk trace.kwtr recording-id --child-admission strict_children
kws_decoder_replay_research model.kwm keywords.kwk trace.kwtr recording-id --child-admission single_fuzzy_wo
```

This describes an interface; preparation/verification does not execute either command on user data. Actual saved-trace replay remains subject to the paired evaluation's input admission and execution authorization.

The research CLI rejects missing, unknown, or duplicate modes and all other tuning flags before input access. It requires:

- F32/H64/V5, vocabulary fingerprint `0x316aea1a94b303bf`
- Model SHA-256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`
- Exactly the full keywords ID 1 `[1,2,3,4]` and ID 2 `[3,4,3,4]`
- Threshold `0.55`, immediate policy, and zero trailing-blank gate, priority, and grace values for both keywords
- The existing model/trace geometry, model hash, and vocabulary identity checks

The saved-pair harness must additionally bind the exact keyword-pack hash and identical trace bytes, recording boundaries, initial state, model, controls, and recording ID in both arms. The replay tool does not acquire missing inputs or run acoustic inference on a cache miss.

Stdout remains JSONL detections with `recording`, `keyword_id`, `time_s`, and `confidence`. No hits means no detection lines. Capture exit status and stderr: partial stdout from a rejected/truncated trace is not an accepted replay result. Mode and executable identity belong in the enclosing per-arm receipt, not in a fabricated acoustic producer identity.

## Verification scope

The verifier checks:

- Original vs guard-off object bytes for decoder, engine bridge, and replay CLI at `-O0`, `-O2`, and `-Os`
- Original, guard-off, and guard-on default-mode complete original decoder-state/event bytes over 20,000 deterministic synthetic frames
- Existing decoder and retention C tests in those three builds
- Canonical K1/K2, one weak `wo1`, two weak `wo1` rejection, non-`wo1` rejection, partial/isolated/reordered phrases, root ambiguity and exact margin boundary
- Independent spent/unspent hypotheses, CTC repeated-label separation, exact retention boundary, cold/emission/VAD-boundary resets, invalid-mode atomicity, and blocked search-cost mutation
- Why a legacy fuzzy-child cost of `-16` is not strict at the inclusive boundary
- Exact research input allowlist, shared engine refractory behavior, and reset-state debug exposure
- Source drift under Python normal, `-O`, and `-OO`; seeded-output refusal; untracked quote-header shadow rejection before compilation
- AddressSanitizer and UndefinedBehaviorSanitizer for the focused decoder tests

LeakSanitizer is explicitly disabled because the hosted verification environment runs under ptrace; no leak-check claim is made. No real acoustic, FAR/FRR, CPU timing, board, or release qualification follows from these source/unit checks.

On the verified x86-64 GCC host, shipping and guard-off engine size is 23,792 bytes. The research engine is 32,024 bytes (+8,232); decoder size is 11,648 vs 19,876 bytes. Public struct sizes and original trie-node size are unchanged. These are hosted type-layout observations, not board RAM/stack/latency evidence.
