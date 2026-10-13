# Read-only ASR evidence acceptance

This fixed, offline command validates already available public ASR evidence.
It does not run a recognizer, fetch data, upload files, retry a transfer, change a
ref, create a claim, or activate an execution control. No actual ASR acceptance
request is supplied. The historical ASR-r2 archive and transcripts remain
unavailable; this tool cannot reconstruct missing bodies from hashes.

## Reused validators and source authority

`asr_posterior.py` imports the existing repository modules only after verifying
`validator-dependencies.json` against its embedded SHA-256 and checking every
listed body. Import origins are also fixed. The dependency manifest lists the
explicit reviewed public source closure, including the v2 issuer and contract, excluding only the
non-runtime consolidation retention index. That index records this new tool and
cannot also be its own transitive content pin. No validator tree is copied here.

The unchanged `parse_manifest`, `reassemble`, `validate_archive`, and
`validate_source_binding` validate canonical ZIP/chunk/member bodies, original
native completion/start/raw outputs, source/input authorities, and fixed18
ledger accounting. The fixed result Git tree and single-parent commit are
recomputed. Source commit metadata is compared with externally pinned API
identity/tree/date; this is not reconstruction of a signed source commit.

## Two custody modes

- `normal_publication` requires the original completed run/job/history and fixed
  host/publisher steps. The original publisher must have succeeded. A failed host
  may carry a valid partial package. A successful host requires the complete
  host receipt.
- `recovered_checkpoint` retains the original producer identity and terminal
  failure. A separate completed/success recovery run, job and pinned step must
  bind the same checkpoint, final commit and original ZIP. Recovery's source
  release and shared runtime bodies are checked. The recovery receipt must
  appear once in the original log. Recovery success is transport evidence and
  does not reclassify the original publisher or establish native success alone.

The command does not call the recovery writer or use its in-progress execution
predicate as a terminal-success predicate. Failed, unknown or unfinished
recovery yields observation-only output with no native completion claim.

## Source-bound v2 owner custody

A v2 request pins the original producer owner identity, claim ref and owner commit. Independent original ref/commit bodies are required alongside the source, run/job/history, package and publication evidence. The proof binds actual release/control member bytes and the same source freeze. Full producer history may truthfully contain later ordinals; only API-confirmed ordinal one/attempt one may own this v2 claim. A caller cannot turn a legacy request into v2 by supplying a success flag, or use v1 to bypass a source-selected v2 contract. The historical v1 branch retains its original history-one predicate. The current v2 producer selectors derive from the single compiled campaign scope; full archive binding rejects a detached future control/proof paired with either a future ref or the old r3 ref. Future source materialization must update that scope and the matching frozen authority together. The explicit legacy release/history predicates do not drift with future producer constants. Outer package paths and result-ref custody remain fixed to this verifier revision's compiled scope; old v1 complete custody requires its original-scope frozen verifier after a future scope change. No universal cross-version archive compatibility is claimed.

Both normal and recovered custody use the same strict archive/native validators. Recovery retains the producer's failed publisher status and separately checks the recovery run's completed-success terminal contract. Its own history remains singular. This command is read-only and does not call the issuer, consume a private handoff, or call transfer recovery. Actual authority remains absent until independently pinned original evidence is provided.

## Phase-aware scientific and resource completion

The adapter adds explicit completion checks above the existing archive schemas:

1. Any started slot requires the original predecode policy and budget proof.
2. Full success requires CPU, cleanup, supervision and budget receipts. The
   original `CpuLedger` thresholds and arithmetic remain authoritative.
3. The seven supervised setup/prepare/decode workers must match original
   producer-run names, modes and CPU/cleanup stages. Setup and prepares use the
   setup bucket; each decode uses its family bucket. These are the names and
   stages written by `runtime_session.py` and `hosted_run.py`.
4. Those same seven workers require the original successful cleanup conditions:
   acknowledged pause, no host deadline expiry/overrun, zero cleanup overrun,
   and duration within the original15s budget. A remove error followed by a
   successful exact-ID absence proof remains admissible.

These thin-layer checks are additional completion gates; they were not all
provided by the original archive schema. They do not impose a full-success
roster on a legitimate preflight or first-slot failure. Partial results retain
available evidence and the original18-row denominator. Missing required native
bodies still fail the original strict validators.

Verified raw transcripts and normalized transcripts are separate fields.
Normalization calls the unchanged `tools/speech_asr_consensus` punctuation-only
function. No homophone, case, script conversion or Unicode folding is added.
Three distinct complete family outputs with identical nonempty normalized text
may produce a machine candidate. `training_admitted` and `human_gold` always
remain false; acoustic verbatim completeness remains unknown. Ledger-reported
attempts are observations, while `consumed_attempts` remains null. The tool never
authorizes retries or new attempts.

## Local custody and command

Supply the original local files, a custody index, and an expected-input document
whose SHA-256 values were approved independently of this command. The adapter
cannot grant itself that authority. Null actual-input fields are rejected.

    python3 -I -S -B research/k1-asr18-offline-posterior-v1/asr_posterior.py \
      --evidence-root /approved/evidence \
      --expected /approved/expected.json --expected-sha256 <approved-sha256> \
      --custody /approved/custody.json --custody-sha256 <approved-sha256>

Every custody row binds a role, relative file path, length, SHA-256 and original
resource locator. Original API JSON, connector logs, immutable bytes and locally
reassembled ZIP are distinct roles. No original writer receipt, HTTP status,
request count or wire telemetry is reconstructed. Complete branch-scoped push
history can support the narrower workflow predicate only as an explicitly
identified semantic adaptation of that original route.

Local reads reject links, special files, path aliases, duplicate roles, wrong
locators and inconsistent bodies. Caps are12MiB ZIP, the existing32MiB expanded
archive limit,8MiB per log,80 indexed roles and80MiB aggregate. These are offline
reader limits, not execution-budget changes. The command writes no files and
emits fixed safe failure metadata on rejection.

Null-identity stdout diagnostics are isolated as CI/precontext observations.
Recognized nonnull foreign identities are rejected. Exact-identity diagnostic
records and envelope hashes never substitute for native bodies.

## Self-contained model-free tests

`fixture_cases.py` generates synthetic evidence in temporary directories from
existing public test helpers. It does not read a historical ZIP, private source
checkout or external service. The original collector, policy/native receipt
writers, canonical packer and strict validators still run. Model constructors,
external setup and service observations are explicit synthetic substitutes.
This is a minimal native-orchestration test layout, not a reproduction of every
real installation or model-constructor output. Image transfer, model downloads,
Whisper setup/broker and model-load sidecars are synthetic or omitted by the
explicit setup/constructor substitutes. The native worker receipt/start/return
path and original resource writers are exercised; their generated sidecar roster
is checked without pruning unknown members. The six WAVs are generated public
synthetic test inputs, not new TTS or recordings. Packages contain no WAVs,
weights, credentials or private text.

The three base cases are cached immutably and returned as deep copies; mutations
cannot contaminate later cases. Repeated generation is checked for deterministic
ZIP hashes. Each optimization mode runs in a separate process.

Run `test_asr_posterior.py`, `test_offline_custody.py` and
`test_claim_posterior_integration.py` in normal, `-O` and `-OO`
with `-I -S -B`. The A–I matrix covers complete ordinary and recovered evidence;
preflight, first-slot and started-unknown partials; missing or hash-only bodies;
identity/source/input/Git drift; native/claim/resource omissions; uncertain
publication without retry; stdout attribution; and failed recovery. Real
isolated CLI invocations cover custody paths, null requests and safe failures.
All generated run/job identities and native results are labeled synthetic.
