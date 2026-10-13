# Fixed Git create-only claim qualification

This is a zero-model qualification of a specific Git reference primitive. It does not admit an ASR campaign, change the production history-one rule, or restore the sealed ASR results. The existing ASR, prepare, transfer-recovery, and earlier qualification controls stay closed.

## Scope and identity

Only a new push creating `research/k1-atomic-claim-qualification-v1` can trigger the new workflow. Ordinary source branches, pull requests and main do not trigger it. The fixed job is `qualify` on the public standard `ubuntu-24.04` runner, with a ten-minute job ceiling. Checkout is SHA-pinned with credential persistence disabled. The existing `contents: write` and `actions: read` scopes are confined to this qualification job; only its final qualification step explicitly receives the short-lived token. No new credential, permission, external service, model, weights, audio, Artifact, cache, or payload-log transport is used.

The workflow-level concurrency group has `cancel-in-progress: false`. This is scheduling protection, not the claim primitive. The qualification budget gate requires both the environment and the actual API run to identify ordinal `run_number=1`, attempt one, the fixed workflow, exact source, branch and job. Complete branch/workflow history is retained and may contain multiple source events; duplicate run IDs or contradictory duplicate workflow ordinals are rejected. Later ordinals or attempts cannot write. The new workflow must have zero prior runs before root releases its source ref. If ordinal one is cancelled before it reaches the write gate, this experiment may make no progress; a later ordinal does not take over. The ordinary CI invokes only its offline test script and cannot consume this ordinal.

The fixed target objects already exist and contain the earlier public synthetic transfer fixture, not ASR recognition results:

- Owner A: `8b805105810518f849a42fcc6db5d2fd331b8e33`, sole parent `276e56062b8d9eeac11ea93348a792b6bd008c8c`.
- Owner B: `1fd1b76275b22c0501d8be5bf1087ead7b1fca60`, sole parent Owner A.
- Both trees: `aeea9ae50c067b11b1d2cabf5bb9000a31d2d1f2`.

Only two new refs may be created:

1. `refs/heads/evidence/k1-atomic-claim-qualification-v1-race`
2. `refs/heads/evidence/k1-atomic-claim-qualification-v1-lost-response`

No new Git blob, tree or commit is created. No PATCH, DELETE, ref reset, alternate key or automatic retry is allowed. If either ref already exists, preflight stops. These fixed target commits distinguish synthetic contenders; they do not encode or prove a future production claim owner's run/job/body construction.

## Two scenarios

For the race, two independent spawned OS processes meet a barrier and each sends one direct REST `POST /git/refs` for the same race ref, using different fixed targets. Each child alone records its own permit and synthetic downstream witness only after receiving and validating exact HTTP 201, ref and target. The parent never chooses one of two successful children to hide duplication. Passing requires one 201, one non-authorizing 409 or 422 response, one permit and one witness, and an independent GET of the exact winner target. The two client call intervals, measured with the same-host monotonic clock, must overlap. The interval includes client connection and response handling; it does not reveal server linearization times. If scheduling yields non-overlapping intervals, this attempt is only a limited sequential observation and fails qualification without another POST. A 409 or 422 is not, by itself, proof of why the service rejected the request. New processes then read the existing ref as the same owner and as the other owner; neither may gain admission or send a POST.

For injected loss, one separate child creates the lost-response ref for A. It first retains a validated original 201 observation, then deliberately discards the application acknowledgement and exits with 86 before producing a permit or witness. This is injected response loss plus process exit, not an observed network timeout. Only the exact retained 201 and zero permit/witness establish the intended injection. A new process may GET the ref but must reject re-entry even for the same owner. A real timeout, rate/permission denial, malformed response, wrong target, unexpected exit or budget failure stops later phases and preserves whatever may already exist.

## Budgets and accounting

The eligible qualification invocation has maximum 20 API calls: 17 GET and three POST. The three writes are exactly the two race contenders and the one loss contender. There is no retry budget.

The GET roster is workflow metadata, current run, attempt-one jobs, complete filtered history, actual source commit, both target commits, both absent refs (nine); race result, same-owner race re-entry, other-owner race re-entry (three); loss result and loss re-entry (two); final complete history and both final refs (three).

Cumulative POST JSON bodies are at most 1,536 bytes. This is not HTTP wire traffic. Each received response body is capped at 1 MiB and total observed received bodies at 4 MiB; concurrent responses share the accounting bound, including complete failure responses. Partial bytes not observable after a socket failure are not presented as exact wire accounting. The internal monotonic total window is 300 seconds, with 285 seconds for normal work and 15 seconds reserved for bounded child cleanup. Each request has a bounded remaining-time timeout, and no further request is dispatched after a failure or cap. The ten-minute job timeout remains an outer ceiling, not a promise that a killed or delayed job will complete its internal window.

The conditional global write bound relies on the documented unique workflow ordinal, attempt-one gate, one fixed trusted workflow entry, and exclusive local invocation directory. Fixed refs alone only bound successful ref objects; they cannot bound arbitrary external callers or malicious repeated invocations. No such external attacker guarantee is claimed. The qualification records safe public identities, complete history fields, statuses, byte counts, body hashes, and each child’s independent permit/witness outcome within a 128 KiB diagnostic-output cap. These are small public metadata, not fixture or recognition payload bodies. The token is not written to files, stdout, child command lines, receipts, ref bodies or fixtures.

## Local and later actual evidence

Offline tests use actual spawned processes and an atomic local fake service. They cover concurrent and duplicate events, source and owner mismatch, existing refs, read failure, non-authorizing status codes, response loss, crash, entry witnesses, token boundaries and bounds. These tests do not prove Git service semantics. Normal, `-O` and `-OO` run from ordinary CI; actual Git qualification requires a separately released source ref after source review and main CI.

Actual acceptance must preserve the exact source/run/job/ordinal/attempt, complete histories, both children and their independently recorded witnesses, the injected child's original 201 and exit 86, bounded request receipts, and both refs' immutable targets. The recoverable HTTP evidence in the job log is a safe projection produced by the frozen client, including the hash of the actual received body. Complete response JSON left only in RUNNER_TEMP is not claimed to have been externally retained. Independent ref GET observations must be collected separately. Original failures remain failures. A passed fixture does not qualify production token handoff, real two-run scheduling, model admission, or ASR completion. A future history-two ASR policy and one-use token-free host handoff require separate source review and authorization.

Official reference semantics:

- [Run number and attempt](https://docs.github.com/en/actions/reference/workflows-and-actions/variables#default-environment-variables): run number starts at one for a particular workflow and is unchanged by rerun; attempt increments.
- [Concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency): scheduling scope is a repository concurrency group.
- [Create a reference](https://docs.github.com/en/rest/git/refs#create-a-reference): 201 is creation; 409 and 422 are non-authorizing error statuses. Documentation does not alone prove the service's concurrent behavior for this fixture.
