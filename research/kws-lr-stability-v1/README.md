# Completed LR-stability diagnostic: budget consumed

This campaign completed once on 2026-10-16. The 600 optimizer updates and 608 full-batch forwards (600 training plus eight observations) are fully consumed. The plan is now disabled. Retry, resume, extension and reuse of this budget are forbidden; any new numerical experiment requires a separately reviewed plan and budget.

Independent saved-evidence verification passed for all 608 payloads and 600 successful step receipts. The saved visible trajectory through step 500 matched the historical A trajectory exactly. This does not independently establish equality of unsaved internal parameter or Adam states.

At the fixed step-600 endpoint, KW1 complete match was 4/4 and KW2 was 0/8, with macro complete rate 0.5 and prefix constraints passing. The fit criterion still failed. Historical fixed A600 had KW1 1/4, KW2 0/8 and macro 0.125. These are descriptive results on 12 already exposed machine-labelled training records with one seed and no probe. They establish neither generalization, independent evaluation, human gold, recall, false-activation performance nor production readiness. No reusable weights were exported.

closure.json records the consumed budget and source/result identities. The result-20261016 archive candidate retains complete raw numerical evidence, original acceptance outputs, frozen source references and saved-logit posthoc analysis. Its README describes reconstruction and the omitted private outer-process evidence. Preparing this candidate does not itself publish it or authorize publication.

## Historical reviewed source and preregistration

The original preregistration below is preserved verbatim. Its enabling or launch language describes historical preparation and grants no new execution permission.

# Fixed late-learning-rate stability diagnostic

Execution is gated by plan.json and the separately reviewed parent launcher. Only the parent executor may freeze reviewed merged source pins, enable the exact plan and launch the new campaign once. The previous grouped-balance budget is exhausted and cannot be reused. Creating this source package used no model forward, optimizer, TTS or ASR calls.

## Single scientific change

Fresh uniform-weight arm A, seed 20261015, same TinyStreamingRNN initialization, 12 training records in exactly the original order, original full-batch frontend, frame-normalized CTC plus lambda=1 record-specific prefix-blank anchor, and fresh Adam at lr=.003. All model, frontend, registry, source archives, RMS prefix and runtime wheel identities remain pinned to the original grouped campaign. No warm start or old weights are loaded.

Updates 1–500 use lr=.003. After the step-500 training observation has matched the historical A evidence, update 501 changes the existing Adam parameter-group learning rate to .0003. Adam moments are retained; the optimizer is not recreated. Updates 501–600 keep that fixed rate. No other numerical optimization changes. The lower learning rate is fixed in advance, never triggered by observed loss or gradient. A finite loss rise or poor decode is not an early-stop condition.

New campaign: kws-lr-stability-20261016-v1. Maximum 600 optimizer calls and 608 full-batch model forwards: 600 training plus observations at 0/50/100/200/300/400/500/600. There are no probe features, forwards or probe-based choices. The original registry and prefix file include all 16 historical records for source identity verification; only the fixed 12 training records enter feature preparation and model calls. No new TTS, ASR, augmentation, VAD, retry, resume, extension, decoder tuning, model selection or weight export.

## Historical replay is a mandatory gate

historical-input-pins.json binds 1017 existing files to the already accepted grouped-balance hash manifest. Every required historical file is verified before model import. Initialization must equal SHA256 0eafd98dd20f8c799d1c4d733a47314018ecfdd942c2769fed49cf8ae4fc7b2d. Each training forward through update 500 must match the saved historical A payload's complete ordered sample dictionaries, including every log probability, target, prefix, loss component, weight and raw decode, and aggregate scientific fields exactly. Before each of these optimizer updates, pre-update loss components and pre-clipping gradient norm must also match exactly. The new campaign schema/identity is not compared as though it were historical evidence.

All observations through post-step 500 must exactly match their historical samples and aggregate scientific fields. The first 500-step sequence must pass before update 501 is allowed. Payload comparison runs before backward/optimizer; gradient comparison runs before optimizer reservation. A mismatch saves failure evidence and stops with no retry. Losses/gradients are pre-update; named observations are post-update (step 0 is initial). This verifies the recorded numerical trajectory, not independently retained per-step internal weights or Adam moments, which the old campaign did not save.

The historical A600 observation is pinned as the fixed control. The new fixed-600 raw-greedy per-class/macro outcomes, prefix constraints and objective are the primary descriptive endpoint. Offline analysis may compare last-100 loss/gradient ranges without additional model calls. No earlier checkpoint may replace this endpoint. This is a one-seed mechanistic contrast; it does not establish generalization, independent evaluation or production qualification. All recordings remain exposed machine-labelled positives; no false-activation, negative-audio, streaming-latency or deployment-cost claims are supported.

## Reuse and bounded execution

This module deliberately reuses the unchanged, hash-pinned research/kws-grouped-balance-v1 contract utilities and file/runtime helpers. It does not run that module's experiment or reuse its Budget. New Budget has its own exact 600/608 sequence and new exclusive campaign claim. The two modules must reside next to each other in an assembled repository tree. Source-manifest.json lists new files and the exact existing shared dependencies; the parent launcher must pin the full source closure, including model/frontend and candidate-registry dependencies.

The same durable forward reservation/return and optimizer reservation/success receipts are retained. All raw LP and per-sample losses remain available in payload-NNNN.json; step receipts hash-bind payloads. Added step fields record learning_rate and historical_replay_matched (true only through 500). learning-rate-transition.json records the sole fixed transition. Failure is terminal; partial receipts remain authoritative if the process dies.

The wrapper defaults to disabled, unknown merged commit/pins, and unconfirmed CLI. It validates source/history before spawning, uses only free local CPU, and bounds CPU and wall time to 3600 seconds, virtual address space to 8 GiB, OS per-file size to 16 MiB, and core dumps to zero. Runner JSON output is capped at 1 MiB/file and 512 MiB cumulatively, reserving 1 MiB for a failure receipt. Full 600-step LP evidence is expected below approximately 180 MB uncompressed, not a guaranteed compression ratio. Abort/timeout terminates and waits for the process group; it never relaunches. Numerical, identity, runtime, budget or evidence failures stop immediately.

## Synthetic-only validation

In an assembled repository layout, run python -B -I research/kws-lr-stability-v1/test_contract.py, adding -O or -OO before the script for optimization-mode coverage. Run numeric_preflight.py to verify static plan/source pins. Tests use mock receipt sequences and synthetic dictionaries, not Torch or real models. Both enabled and disabled plan shapes must pass the test suite; passing tests alone never authorizes execution. Independent saved-evidence verification is required after any later approved execution. Source testing does not consume or prove success of the numerical experiment.
