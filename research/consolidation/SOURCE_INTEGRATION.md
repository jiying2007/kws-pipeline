# Public research source retention, 2026-10-07

This source-only follow-up retains reusable tools and historical conclusions on
main `c44dcf6fadb60b4a084200f4565b8e5c2503eac4`. It changes no shipping model,
runtime, configuration, threshold, training implementation, qualification rule,
or existing workflow. The existing 61-branch cleanup-retention guard is intact.

## Safety and identity contract

- 718 byte-exact public source/archive paths are bound to 834 original
  commit/path provenance records in [the retention manifest](source-retention-2026-10-07.json)
- Latest sources come from #460, #461, #462, #466, #467, #468, #470, #471,
  #472, #473, #475, #478, #482, #483 → #484, #487 and #488. Earlier #469,
  #477, #480 and #481 differing files remain separately archived; equal files
  share one exact retained location
- All 17 latest historical workflow definitions, including saved-only ones, are
  retained outside `.github/workflows`. Four earlier sibling workflow variants
  are likewise non-active. Original path → archive path, commit, blob SHA-1,
  SHA-256 and byte count are explicit. No old acquisition workflow is deployed
- Cosy30's original armed `arm.json` is retained as historical evidence. The
  usable source directory instead has the existing admission protocol's
  `disabled` mode, null activation identities and unchanged batch identity
- The offline runner reconstructs 17 original workflow paths plus the original
  Cosy30 arming record only in a temporary non-Git checkout for exact static
  checks. This is an explicitly different verification projection, never the
  deployed tree and never a new source-freeze or execution authorization
- Old `approved: true`, `training_authorized`, admission files, README commands
  and branch names record past decisions. They do not authorize new work
- The fixed30 blind audio ZIP is deliberately pointer-only. Its two historical
  source records and exact Git blob remain indexed; this source consolidation
  does not claim the audio input archive is locally retained
- No private natural recordings, private derived features/heads, other unpublished private research artifacts or current acquisition outputs were inspected or copied

## Source map and scientific boundaries

| Retained entry | Useful content | Current boundary |
|---|---|---|
| [ASR runtime](../asr_runtime/README.md), [FireRed CPU preflight](../firered_cpu_preflight/README.md) | Dependency/admission, resource and failure contracts; mocked tests | Historical qualification records; no installation or live qualification authority |
| [CosyVoice3 pilot](../cosyvoice3_pilot/README.md), [Cosy30](../cosyvoice3_cross_voice/README.md) | Bounded generation source, locking, admission, output packaging | Historical generation success is not human-label acceptance or KWS qualification; Cosy30 outputs were unavailable in the audit |
| [Qwen20](../native_a20_qwen20/README.md), [N0](../native_a20_n0/README.md), [domain diagnostics](../native_a20_domain_diagnostics/README.md) | Immutable saved-evidence schemas, scoring and adversarial checks | Original data pins remain unchanged; no independent acoustic or shipping claim |
| [CPU metadata probe](../cpu_torch_metadata_probe/README.md), [token preparation](../token_preparation/README.md) | Pure metadata/network mocks and bounded preparation designs | Live probe and preparation launchers are not run |
| [fixed300](../fixed300/PROTOCOL.json), [Cosy49](../cosy49/README.md), [terminal archive](../cosy49-results-v1/README.md) | Training-source history, pure tests, rejected/terminated outcome | No candidate promotion, retraining, new data admission or sweep |
| [fixed30 ASR](../fixed30_asr/README.md), [Qwen6](../qwen6_tts/README.md) | Original and recovery source/freeze histories | Syntax-only validation for these modules; no bundled runnable unit suite and no live ASR/TTS |
| [Melo6](../melo6-source-screen.md), [saved results](../melo6_saved_results/README.md) | Original/recovery source, human-adjudication and context diagnostics | Machine text gate remains failed; human and acoustic/context conclusions stay distinct |
| [N1 generation](../single_k1_followed_wu/README.md), [whole-clip appendix](../n1-wholeclip-leading-v1/README.md) | Mocked generation contracts, geometry/scoring source and full saved verifier | One exposed positive clip; `CONTEXT_UNVERIFIED`; no word boundaries or negative denominator |
| [First-prefix marginal CTC](../k1-prefix-marginal-v1/README.md) | Mathematical source, independent oracle, original/recovery mocks | Failed candidate retained: M3 false K1 while N1 remains accepted; no qualification |

## Offline verification

The new workflow runs only the explicit synthetic/mock/static allowlist, with
NumPy 2.3.5 and SciPy 1.17.0 as numeric test dependencies. Dependency installation
is a separate ordinary CI setup step; the checks themselves do not download
fixtures, models or runtime packages. No broad test discovery or acquisition
launcher is invoked. N1's synthetic companion has its own explicit entry, fixing
the original #487 workflow's missing N1 coverage.

```sh
python3 -B tools/verify_research_sources.py
python3 -B research/consolidation/source_tests/test_source_retention.py
python3 -B tools/run_research_source_checks.py --suite synthetic
```

[The allowlist](source-offline-allowlist.json) records exact commands, working
directories, dependencies and scope. Each file runs in its own process to avoid
cross-contamination from repeated historical module names. The runner never
writes archived workflows into the actual checkout and verifies retained bytes
both before and after the checks.

Saved-evidence checks require explicit prepared fixtures. The runner does not
silently download, invent, replace or skip missing evidence. For N1, the
[closed fixture manifest](source_fixtures/n1-saved-inputs.json) binds 19 already
public files totaling 137,742 bytes. Only the 47,600-byte derivative WAV is used;
the larger native WAV is unnecessary. The preparation helper only copies an
explicit local cache and verifies all hashes, sizes and closed-set identities.
The full verifier performs saved numeric scoring and in-memory byte reconstruction,
with no model, frontend, decoder, compilation, playback or network calls.

```sh
python3 -B research/consolidation/prepare_n1_saved_fixtures.py \
  --source-dir /path/to/exact-n1-cache --output-dir /path/to/prepared-n1
python3 -B research/n1-wholeclip-leading-v1/verify_saved.py \
  --data-dir /path/to/prepared-n1/data \
  --n1-source-dir /path/to/prepared-n1/source \
  --scorer-dir /path/to/prepared-n1/scorer
```

The source-only #475 historical endpoint tests are explicitly
[excluded from execution](source-excluded-checks.json): their mixed98 natural-input
metadata was intentionally omitted from the public archive. Their exact source
is retained for evidence; this integration does not seek private inputs to make
them runnable. [The verification report](SOURCE_TEST_REPORT.md) distinguishes
passed, not-run and syntax-only checks.

## Historical findings and deferred designs

[The branch-outcome index](history/BRANCH_OUTCOMES.md) records all 32 dormant
branches and [the branch manifest](history/branch-retention-manifest.json) records
180 changed-path dispositions. These are audit-time records against the earlier
`55a4e23379ad7072db507dbe419b38d8898e17fa` baseline; an audit-time "missing"
status is not a claim that the file is still missing from this integration.
Twenty-seven missing public September narrative
notes are preserved byte-for-byte under `history/notes/`. Those notes are
historical, may contain superseded claims, and grant no current execution authority.
Use the outcome index and current research map for corrected conclusions.

In particular, #450 corrects the earlier long-history explanation: the phase-
aligned cold crop reproduced the event, but changing phase also added 80 earlier
samples, so phase alone was not isolated as causal. Label-prior #431 and decoder
counterfactual failures remain negative conclusions, not product improvements.
The restricted-development design and both keyword-set alternatives remain
retained pointers requiring a fresh current-contract review; no stale product
code is restored. No old branch or PR is deleted or closed by this batch.
