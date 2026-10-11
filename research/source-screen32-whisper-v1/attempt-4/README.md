# Fourth one-shot campaign admission (disabled source)

This is a new admission gate for the fourth separately authorized Whisper
campaign. The public source release remains `approved: false`; its workflow is
an inert template outside `.github/workflows`. Preparing or merging this source
cannot run a campaign. Activation requires a separately reviewed change on the
unique branch `research/qwen16-whisper-small-third-attempt-4-v1`, with the exact
workflow template installed and the fourth release approved. Only a fresh
branch-created push, workflow run number 1 and run attempt 1 can pass admission.
No workflow dispatch, schedule, automatic retry or fallback is provided.

## Source and evidence identity

The source baseline is public main commit
`021e940fa231e0a8625e967a5b01ac2483f18a56`, tree
`6f49c9e37e71666735f8b922fdf5ef46593899ce` (the local source-equivalent commit is
`344f0d29cd9410a3cf39f645b1e5fe092c19f5c9`). It contains the corrected,
source-audited pip compatibility policy. No prior execution gate, release or
freeze is regenerated. Their historical hashes remain unchanged.

The fourth freeze includes the entire historical 234-file closure, current
runtime modules and tests, both pip compatibility audit files, the new fourth
admission source and all three retained failure archives. This binds exact
source bytes, original sixteen audio identities, 22 runtime wheel hashes,
22 official source-file hashes and the original model checkpoint hash.
The release binds the new freeze; admission also requires the checked-out Git
head, push head and fresh branch event to agree exactly. All older campaigns
must remain disabled, with no older active workflow present.

Previous campaigns consumed their one-shot authorization and remain separate:

- Run 38105471896: setup failure, zero audio attempts, 16 NOT_RUN
- Run 38107706348: setup failure, zero audio attempts, 16 NOT_RUN
- Run 38112558973 at `f17670af7da490bfe2d292661fbe3f5114d5eb4e`:
  pip 25.0.1 capability/help-probe failure, zero audio attempts, 16 NOT_RUN;
  source freeze `16901d0d47d10486a8754568cb03020a3a155a9034ccc4e522fb796929a48ad4`

The retained third receipt does not establish the help exit code, individual
missing options or HTTP status. The corrected policy accepts missing
`--resume-retries` only for pip 25.0.1 whose eleven audited source files match
exactly. Other versions must expose the complete explicit-zero-resume contract.
The new source does not prove network acquisition will succeed.

## Runtime and isolation

The new gate reuses current `attempt-3/campaign_host.py`, `bounded_host.py`,
`bounded_connect.py`, `pip_acquire.py`, `pip_compatibility.py` and
`phased_setup.py`; there is no copied/forked transport implementation.

The setup mount contains exactly 13 files, including `pip_compatibility.py`.
The inference mount remains exactly the original eight file paths. It excludes
all attempt-specific modules, compatibility audits, tests, prior-run context,
peer transcripts, human labels and network transport. Current setup helpers
are validated before any image or container work; inference starts only after
complete setup and network teardown, using the original offline constructor.

The model, decoder and original sixteen anonymous clips are unchanged. The
budget remains one free GitHub-hosted CPU job, at most 45 workflow minutes,
2400 host seconds and one absolute 1200-second setup deadline. Broker plus
worker sum to 4 CPU, 12 GiB without swap and 256 PIDs. Approved HTTPS hosts and
TLS verification remain fixed; aggregate acquisition remains 1536 MiB with
zero connection/resume retries. Offline install precedes source/model
acquisition. Any failure stops; setup failure preserves 16 NOT_RUN and zero
new audio attempts. There is no alternate model, extra clip or hidden rerun.

## Publication boundary and verification

The third archive was copied byte-for-byte into this local preparation from
commit `7017fd4f1805737347d02cbead9ed3aa7fc079bf`. Publication of that runtime
telemetry is separately blocked pending explicit permission. This local
freeze depends on those bytes; do not publish this preparation or activate its
workflow until that publication boundary is cleared. This source preparation
itself runs no pip command, install, Docker, model or network acquisition.
The future workflow template contains an evidence-artifact upload step, which
also needs the applicable publication permission before activation.

Run source-only checks through the existing CI entrypoint in all three modes:

```
python3 -I -B research/source-screen32-whisper-v1/test_contract.py
python3 -I -B -O research/source-screen32-whisper-v1/test_contract.py
python3 -I -B -OO research/source-screen32-whisper-v1/test_contract.py
```

The fourth suite tests exact source closure, compatibility audit inclusion,
all three immutable zero-audio failures, disarmed historical gates, workflow
identity, fresh-event/run rejection, freeze/source drift, bundle separation,
and fail-before-runtime behavior. The reused bounded-acquisition and campaign
suites exercise the unchanged resource, network and phase contracts using
stdlib fixtures and mocks. Actual hosted pip/image/egress behavior remains
unverified until the separately admitted single campaign executes.
This work remains exploratory source-screen evidence, never human gold,
training admission or shipping approval.
