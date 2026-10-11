# Third Whisper campaign attempt: retained setup failure

[Run 38112558973](https://github.com/jiying2007/kws-pipeline/actions/runs/38112558973)
was campaign attempt **3**, following runs **38105471896** and **38107706348**.
GitHub run number and run attempt were both **1**. The executed head was
`f17670af7da490bfe2d292661fbe3f5114d5eb4e`, tree
`67fa8ba025959b20649992248078062de545219a`.

## First established failure and limits

Setup stopped without retry on **wheel 01**, before dependency acquisition.
`wheel-01-worker-result.json` records **pip 25.0.1**, stage **pip_help_probe**,
and reason **pip_capability_probe**. This fixed reason does not distinguish a
nonzero help-command exit from a missing required option. The artifact contains
no retained help output, stdout or stderr that would establish which occurred.
No HTTP status is established; the earlier campaign's HTTP 403 is not evidence
of this campaign's cause.

Docker image pull completed, with its separate receive observation within the
reservation. Broker and worker Docker inspection and actual in-container kernel
isolation receipts were retained. They show read-only roots, private read-only
cgroups, no capabilities, no-new-privileges and seccomp; the worker had only
loopback and network-none scope. Broker readiness and its owned Unix socket check
are implied by the executed control flow reaching the worker, rather than a
standalone final broker receipt.

The pip effective-configuration probe, wheel download and exact output checks,
offline dependency installation, official source acquisition, model acquisition,
model loading, audio inference and native natural-EOS validation were **not run**.
All **16** audio rows remain **not_run**, with **0** attempts and **0** completions.
No transcription-quality conclusion follows from this setup failure.

The budget ledger charged **1,188,299 bytes** as a conservative reserved allowance
because there is no final broker receipt. This is **not observed TLS traffic**;
observed ciphertext bytes are unknown. Image-transfer observations are separate,
include protocol and potentially unrelated host traffic, and do not establish
exact daemon accounting or total network/wire bytes. Both broker and worker
containers were removed, with no recorded deadline overrun.

## Durable public evidence

The **14 JSON artifact members** here are byte-identical copies from artifact
**11692750251**, `qwen16-whisper-small-third-attempt-3-terminal-v1`.
The retained ZIP was **7,468 bytes**, SHA-256
`96be915aea519f2ab3126168c8d19994c37f962b9797cb352ddd574d1774cf2f`.
`artifact-freeze.json` verifies the other **13** members. `provenance.json`
records all 14 member hashes and sizes, execution identifiers and interpretation
limits. The existing retained artifact was reused without another download.

All **234** entries in the executed source freeze, SHA-256
`16901d0d47d10486a8754568cb03020a3a155a9034ccc4e522fb796929a48ad4`,
were independently verified against the executed tree and rechecked unchanged in
the archival base. The archival commit is not the executed remote head.

This archive adds only public source-screen evidence and these provenance notes.
It changes no existing frozen source, locks, workflow gates or release approvals.
Preparation READMEs remain historical source-frozen instructions. No further
campaign is authorized or activated here. No arbitrary job logs, API response
bodies, model bytes, wheels or private human records are included. All evidence
remains `source_screen_only`; training admission, shipping approval and
human-gold flags remain false.
