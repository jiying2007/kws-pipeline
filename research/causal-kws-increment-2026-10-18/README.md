# Small causal-KWS research increment

Offline publication candidate, 2026-10-18. This directory contains reusable research cores, anonymous synthetic contracts, aggregate evidence and public-source identity metadata. Integration and remote CI are still pending. No weights, feature caches, audio, individual training labels or transcripts are included. It cannot reproduce the private training runs offline.

## Retained assets

- `core/joint_branch.py`: byte-identical live 140-dimensional teacher-hook consumer. It preserves the stateful auxiliary head and the native per-keyword event state, and passes only auxiliary KW2 into the OR merger. Torch and a validated checkpoint/config are supplied by a caller; this is not a standalone inference command.
- `core/online_merge.py`: byte-identical deterministic merger. Compare actual supplied-sample cursors, prefer B at the same cursor, and suppress another event of the same keyword within an inclusive 8000-sample interval from the last kept event. Different keywords are independent. Branch-internal hysteresis/refractory semantics remain separate.
- `core/event_state.py`: exact class-source span for the auxiliary native trigger state. The original per-keyword 1000 ms cooldown and low-probability rearming are retained.
- `core/timing_auxiliary.py`: byte-identical weak teacher-relative auxiliary-loss core. Scale 0.1; completion offsets 0..3 after the fourth teacher node; prefix offsets -1..1 around the first three; hinge margin 0.5. The denominator stays the original batch's two-label supervised confidence sum. Teacher nodes are weak timing proxies, not verified acoustic word boundaries.
- `policy/research-tiers.json`: declarative post-evidence strict-dual / majority / abstention policy, with anonymous schema/vote fixtures. It deliberately does not add an automatic label-admission implementation.

The joint source expects the existing event-head architecture through `cfg['core_path']`, the event-state module through `cfg['state_path']`, and caller-owned model/resource identity checks. Its original constructor/wiring and tensor operations are not exercised by the standard-library tests. Those tests inject a tiny fake model only to verify call order, persistent state, startup handling and event wiring. Original runtime evidence is summarized separately.

## Two fixed research operating points

On the already-exposed 26-clip weak/historical development reference (5 KW1 and 6 KW2 positives):

- **B 0.70 conservative:** 8/11 target clips found and zero weak-negative-reference events.
- **B 0.10 effect-priority:** 10/11 before real EOF; 11/11 with the EOF+300 ms proxy window. One gain uses 105 ms synthetic tail. There are three near-reference triggers, with per-keyword counts 1 and 2; ordinary and incomplete references contribute zero. This point does not meet the conservative zero-trigger constraint.
- **Post-hoc OR:** B 0.70 plus the step-240 auxiliary head's KW2 threshold 0.85 gives 9/11 on the same reference with zero weak-negative events. It was selected after examining those results. The added hit is a development complement, not an independent validation gain. Auxiliary KW1 is not in the OR.

The auxiliary head uses native thresholds [0.95, 0.85]. Its step-120/180 matched comparisons improve KW2 from 0/6 to 2/6 while KW1 remains 0/5; step 240 reaches 3/6 with extra training exposure. Eleven of 17 weak positives have usable teacher-relative timing; six remain base-loss-only. These differences in exposure and evidence strength must stay visible.

## Continuous evidence and clocks

The exposed L 60-second joint pilot exactly matched 1999 old feature/logit prefix rows and 60 posterior segments. Both branches and the OR emitted zero events. This checks numerical continuity on a zero-event prefix; the anonymous positive wiring fixtures are not raw-positive end-to-end tests.

The previously model-unread S 1800-second mirror excerpt has completed independent result review. B 0.70 and B 0.10 use two independent decoder states sharing one float32 posterior stream; the lower-threshold result is not obtained by filtering higher-threshold events. Both have zero full duration-valid candidates and zero final events, as do the auxiliary head and OR. This says nothing about empty partial token paths. The source is mirror audio with official transcript metadata; it is not official full-archive audio verification.

All S branches run without synthetic EOF padding, and state continues through intervals excluded from scoring. Clock labels use actual 320-sample delivery availability, never nominal acoustic centers substituted for trigger time. Primary and unknown-sensitivity views have different denominators; `evidence/continuous-S-aggregate.json` preserves every native/common view. Shared primary exposure is 1799.7 seconds, unknown-isolated 1479.68 seconds, and unknown-plus-complete-window 1474.88 seconds. Model-row/whole-window masks do not certify independence from all prior refractory/deduplication state.

Short-clip reference posterior arithmetic is float64, while the authoritative continuous chain retains float32. No cross-path bitwise claim is made. Earlier L 1800-second B evidence used threshold 0.60; it is not a B 0.70 continuous rerun. S thresholds/checkpoints were fixed before its execution and were not tuned on S.

## Evidence and source boundaries

All reference scores are research observations. Weak negative counts are conditional clip counts, not natural false alarms per hour. Unknown content is a sensitivity layer, not a negative label. Two meeting excerpts and the historical development reference are not gold or independent device acceptance. The strict C numerical-parity failure remains unresolved and was not used as the authoritative continuous model. No SSC305, single-core, INT8, accuracy/FAR release or best-model qualification is established.

The research tiers retain 8 original strict-dual weak admissions, 3 majority-only low-confidence candidates, and 1 pronunciation-conflict abstention. The majority weight 0.1 is a proposed cautious heuristic, not a calibrated probability; the original quarantine and voice-lineage split are retained. The policy was defined after observing recognition evidence. No actual training or relabeling is authorized by publishing it.

Corpus terms, author-code licensing and mirror metadata are distinct. `evidence/public-sources.json` records their scope and exact evidence hashes; no corpus or source license is changed by this candidate. Reused project code follows the receiving repository's licensing and attribution review.

## Local checks and integration

From this directory, run `PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_*.py'`. Repeat with `-O` and `-OO`. These tests use only the Python standard library; there is no numerical model, real decoder, recognizer, audio or tensor load. `LOCAL-VALIDATION.json` records the actual local checks. Remote CI must be run on the eventual integration commit and recorded separately.

Use only the explicit public manifest payload. The suggested destination is `research/causal-kws-increment-2026-10-18/`; a publisher may adapt imports/navigation to the then-current main branch without copying one-off experiment directories. No remote branch or pull request is changed by this candidate.

## Controlled next-eleven increment

[Final bounded comparison and retention decision](NEXT11-DECISION.md) records the completed matched-step control/treatment experiment and links to its [anonymous aggregate](evidence/next11-controlled-increment-aggregate.json). Neither new head is promoted; retain the original CTC model and existing conservative OR research candidate. Training-fit gains, development limits and the historical positive regression remain explicit.
