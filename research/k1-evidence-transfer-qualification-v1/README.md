# Hosted zero-model transfer qualification

This is a fixed two-source-phase qualification of the existing bounded publisher
HTTP, blob, tree, commit, checkpoint-ref and immutable-byte readback path. It is
not ASR evidence, native inference acceptance or recovery of a lost ASR archive.
The unchanged ASR `policy`, `identity`, archive validator and source-binding
validator cannot admit this synthetic fixture. ASR execution/publication controls
must remain disabled before this entry reads the short-lived Actions token.

## Immutable input and output

Input is the already retained public fixture at commit
`8b805105810518f849a42fcc6db5d2fd331b8e33`, tree
`aeea9ae50c067b11b1d2cabf5bb9000a31d2d1f2`. The fixed input manifest and 23 chunks
are under `evidence/k1-transfer-qualification-v1/`. The ZIP is 5,884,370 bytes,
SHA-256 `5a75f40a03388dffa6366aa5a39b740582f4b6ea9f36f0d7212d17a1ad8ab901`.
Only `fixture-lock.json` metadata is included here. Runtime has no fixture
generator, model imports, corpus acquisition or producer-directory input.

The new source-preserving checkpoint contains only 24 additional paths:
`evidence/k1-hosted-transfer-qualification-v1/manifest.json` and
`chunks/0000.bin` through `chunks/0022.bin` under that directory. The outer
manifest retains the actual first hosted producer source/run/attempt/job and
numeric job identity. Its archive is explicitly synthetic and remains byte-for-
byte identical to the original fixture.

Fixed refs:

- Source phase 1: `research/k1-hosted-transfer-checkpoint-v1`
- Source phase 2: `research/k1-hosted-transfer-recovery-v1`
- Checkpoint: `evidence/k1-hosted-transfer-qualification-v1-checkpoint`
- Final: `evidence/k1-hosted-transfer-qualification-v1-recovered`

Only the exact public repository, push-created source ref, zero `before`,
non-forced push, run attempt 1, exact workflow/source SHA and one actual run/job
are admitted. The workflow's fixed job is `transfer_qualification`. No dispatch,
schedule, workflow-run trigger, matrix, rerun or branch update is authority.

## Two separately reviewed source phases

The checked-in first-phase control enables only checkpoint qualification. Its
recovery flag remains false and its producer pin is null. Twelve model, decode,
retry, storage and credential flags are literally false. The independently
reviewed `source-freeze.json` closes an explicit public source roster, including
the workflow, controls and shared publisher. Its own bytes are excluded.

1. Read and verify all 24 original immutable fixture bodies. Using the production
   bounded writer, verify both output refs absent and create the 24 output blobs,
   source-preserving tree, deterministic commit and checkpoint ref.
2. Independently read the exact checkpoint ref, commit, complete recursive tree
   and all 24 output bodies. Only after full verification emit a bounded public
   metadata receipt and exit with the fixed injected-failure code 86. Do not
   create the final ref. The run and its original failure remain failure.
3. The root independently verifies the receipt, reachable checkpoint and actual
   failed producer termination. A second source PR must explicitly pin the
   original producer identity, numeric job, source tree/freeze, commit timestamp,
   checkpoint commit/tree, manifest hash and archive hash/size. It disables the
   producer and enables recovery only. Unknown pins cannot become dynamic
   authority from workflow outputs, a mutable ref, local sidecar or dispatch.
4. The independent recovery run verifies its own source, the original failed
   producer run/history/job and original source, then reads only the pinned new
   checkpoint. It validates all 24 bodies, the whole source-preserving tree and
   original producer identity, checks the final ref absent, and creates that ref
   pointing at the exact same checkpoint commit. It creates no new blobs, tree,
   commit, model call or replacement producer identity. A final failure leaves
   the checkpoint intact.

Source PR/main CI is offline only; it never invokes `qualification.py` as a live
entry. The new one-use source refs are created only after source review and CI.
The two phases are independently bounded executions; the review gap is not
misrepresented as one uninterrupted 600-second deadline.

## Exact bounds

First phase uses 38 planned API calls: 6 repository/current-run/history/jobs/
source-commit/source-tree GETs; 2 destination-absence GETs; 24 blob POSTs; 3
tree/commit/checkpoint-ref POSTs; 3 checkpoint-ref/commit/tree GETs. The 40-call
hard cap permits only the shared client's one global uncertain-blob
reconciliation GET and, only following a confirmed 404, one identical blob POST.
There is no generic retry or repeated uncertain tree/commit/ref POST.

There are exactly 48 planned raw requests: 24 original input bodies and 24 new
checkpoint bodies. Requests are once per immutable path. Across both passes the
hard raw-body cap is 16 MiB, reserved before each read. API requests are at most
512 KiB each; API responses at most 1 MiB each. Aggregate API upload JSON is at
most 9 MiB; aggregate API response bodies at most 8 MiB. Chunks are at most
256 KiB and the new manifest at most 16 KiB. One shared monotonic deadline and
outer process alarm bound all live qualification transport to 600 seconds,
including the initial input download. No client can reset that phase deadline.

Recovery uses exactly 17 planned API calls: 6 current-run/source checks, 3
original producer run/history/jobs checks, 2 original source checks, 3 checkpoint
ref/commit/tree checks, 1 final-absence GET, 1 final-ref POST and 1 final-ref GET.
Its API cap is 17 and its sole allowed POST route/body creates that fixed final
ref at the pinned checkpoint commit. It reads exactly 24 raw checkpoint bodies,
under the same byte caps and a fresh, separately authorized 600-second deadline.
It cannot access the old fixture raw route or a producer filesystem directory.

The workflow uses only the fixed `ubuntu-24.04` standard hosted label and
requires a public repository both in the event gate and actual API response.
Public repositories' standard hosted runner minutes are free; larger runners
are billable. This entry uses no Actions artifacts, caches, custom images,
payload/log escrow, paid service or new credential. Only the existing ephemeral
same-repository `GITHUB_TOKEN` reaches the transfer process. Repository settings,
spending limits and permissions are not changed. See GitHub's
[runner choices](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job)
and [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Before a complete checkpoint is reachable and independently verified, the bytes
are not established as preserved. Hashes or successful orphan blob POSTs alone
are insufficient. Reachability is not a perpetual-retention or cross-service
backup guarantee.

## Offline checks

`python3 -I -S -B research/k1-evidence-transfer-qualification-v1/test_qualification.py`
runs only invented local service fixtures. Repeat with `-O` and `-OO`. Tests
exercise the actual shared writer; they do not replace it with a success stub.
They cover exact first-phase API/raw counts, a fresh recovery with all local
file reads forbidden, exact original identity retention, final-ref-only writes,
disabled-before-token admission, strict source events, synthetic-ASR rejection,
corrupt/missing/reordered bytes, substituted identity/hash pins, preexisting refs,
deadline rejection and checkpoint retention after final failure.

Unexpected failures also attempt one bounded metadata-only envelope: the fixed
phase/stage, independently admitted identity or null, exact publication-error
HTTP status when known, measured API/raw counters, and checkpoint commit/tree/
confirmation flags when known. Unknown observations remain null. Error strings,
HTTP bodies, payloads and credentials are never projected. A failed diagnostic
print cannot replace the original exception. The deliberately injected stop
after successful checkpoint verification retains its separate exact receipt.
Raw-body accounting includes bytes returned by the underlying bounded HTTP read
even when its status or size then causes failure; successful reads are counted
once. Each reservation includes the one excess-byte cap probe. Socket partial
bytes not returned by that read are unknown, and counts exclude HTTP framing.
