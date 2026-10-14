# Remaining runtime and retention migration boundaries

## Preserve historical execution and posterior identities

The r4/r5 executed source, source freeze, releases, claims, original archives,
receipts, and posterior results are immutable records. The current closed
mainline snapshot is not a replacement for either executed source. This module
adds a future preparation entrypoint; it does not rewrite those histories.

The retained posterior has 182 pinned dependencies and calls the revision's
`campaign_contract.current_scope()`. Updating that contract and regenerating the
posterior's manifest would produce a new verifier identity. It would not be the
original r4/r5 verifier. Original evidence continues to require its original
source and independently pinned verifier.

## Concrete next-version dependencies

An execution-capable capsule consumer must be a new version, not an `enabled`
field added to the closed schema. Before it can replace the old entrypoints it
must address all of these dependencies together:

1. `campaign_contract.current_scope`, `require_current_scope`, source-bound
   identity, release and consumed-claim history must take a caller-selected,
   source-pinned immutable capsule. An untrusted claim cannot choose its scope.
   The global prior-history builder must become an explicit immutable history
   reference without changing how unknown consumption is reported.
2. `claim_admission`, `hosted_run`, `native_worker` and `runtime_session` must
   bind that same capsule and exact independently approved release bytes before
   setup or any worker starts. Changing only a control parser is insufficient.
3. `evidence_transport`, `evidence_publisher` and `transfer_recovery` must use
   the selected producer scope for package paths, Git custody and publication.
   Recovery must retain the original producer's terminal failure and require its
   own bounded authority; it cannot inherit fresh native-call permission.
4. A new source-closure schema must distinguish stable library, scientific
   policy, input authority and mutable release controls while still pinning all
   of them for execution. The existing 47-member scientific identity must stay
   comparable; old source-freeze schemas must remain byte-preserving readers.
5. A new posterior dispatcher must select the exact archived validator by an
   independently pinned verifier/policy identity, never by arbitrary import
   paths from an archive. Keep old v1/v2 parsing explicitly revision-bound and
   require actual bodies. Reuse a fixed validator library only after showing
   identical results on the original r4/r5 evidence and the full negative matrix.
6. Workflow names, resource budgets, source retention and explicit dependency
   rosters must consume the new version together. Workflow generation is not
   dispatch or approval. Only then can the observed activation/closeout file
   count be measured again and a full reduction claimed.

None of these execution migrations is implemented by the closed preparation
CLI. Its existing integration is deliberately limited to new offline preparation
and deterministic drift verification.

## Retention split without rewriting history

The pre-integration 2026-10-14 manifest contains 718 historical file rows, 834
provenance rows and 72 maintenance records, alongside eight current added
workflow identities. These serve different purposes but currently live in one
large document. The existing `verify_research_sources.verify` and
`run_research_source_checks.selected_checks` read that legacy structure. A
sidecar with no consumer would not solve this coupling.

A safe next format is:

- Keep the original historical rows and original provenance bytes unchanged.
  Select the historical manifest by an immutable source commit and exact digest.
  Do not overwrite archived file identities with current mainline file hashes.
- Put only current source/workflow identities, the historical-manifest reference,
  and the latest maintenance-record digest into a small active index.
- Store one immutable maintenance record per change. Each record binds its
  predecessor digest, before/after public file identities, the exact source
  commit, and observed CI metadata. Append a new record instead of editing an
  old one. Do not invent unavailable CI status, run identity, or original bytes.
- Make an explicit v2 retention reader verify the old manifest plus the chained
  records and current small index. Compare its effective selected source roster
  and checks against the legacy reader before switching maintenance writers.
  Unknown branches, missing bodies, duplicate identities, forks in the event
  chain, current hashes substituted for history, and self-pinning cycles must
  remain rejected.

The public metadata custom rule covers relevant future publication; it does not
make a sidecar self-authenticating. The current integration owner still updates
required legacy metadata until a reviewed v2 reader/writer migration actually
lands. This change does not create a duplicate full historical manifest, modify
the existing 1.3 MB retention file, or pretend the split is already active.
