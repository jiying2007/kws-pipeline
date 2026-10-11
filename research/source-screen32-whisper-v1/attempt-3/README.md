# Bounded official-pip acquisition and third campaign

This is a reusable repository mechanism. Its containers, private Unix sockets,
relay processes and scratch directories exist only for one admitted job and are
removed afterward. It does not establish a permanent proxy, credential grant,
network service or trust exception.

## Fixed experiment

The third campaign remains one free GitHub CPU job, at most 45 workflow minutes,
with the same Whisper small model, original 22 runtime wheels, 22 source files,
fixed decoder and 16 retained anonymous clips. The host ceiling is 2400 seconds;
setup has one absolute 1200-second ceiling. Any failure stops the campaign. There
is no retry, model/version fallback, index lookup or transcript-based selection.
The two previous setup failures and all historical freezes remain unchanged.
Public source has a disabled third release; no active third workflow belongs on
main. A reviewed dedicated branch activation is a separate action.

## Acquisition and isolation

For each wheel, official pip runs in a network-none worker. Its loopback shim
can reach only a narrowly mounted Unix socket. A second hardened container owns
CONNECT egress to the unchanged approved HTTPS hostnames on port 443. No host TCP
listener, Docker socket, host networking, custom CA, TLS interception, firewall
change or extra capability is used. TLS certificate verification stays intact.
The broker cannot see encrypted paths, URL userinfo or HTTP status; generic pip
failure is never labeled HTTP 403 without actual structured evidence.

Concurrent broker and worker limits sum to 4 CPU, 12 GiB memory without swap and
256 PIDs: broker 1 CPU/256 MiB/16 PIDs; worker 3 CPU/11.75 GiB/240 PIDs. Docker
inspection and in-container kernel evidence must agree before runtime work.
Worker transient storage is a 512 MiB tmpfs, with an exact-wheel per-file limit.
Pip uses an empty configuration/credential context, direct locked requirements,
required hashes, no dependencies/index/cache, and zero connection/resume retries.
Unsupported bootstrap-pip capabilities stop before acquisition.

A trusted host ledger charges both directions of broker-read TLS stream bytes
once across all assets. It never resets on reconnect or process replacement.
The shared ceiling remains 1536 MiB. Wheel allowance is ceil(expected size * 1.02)
plus 1 MiB, also bounded by the remaining shared ceiling. Source/model use the
same aggregate ledger and retain the old downloader's exact plaintext checks.
A missing final broker receipt reserves its allowance and aborts, rather than
inventing a measured byte count.

These counters are not total wire traffic and do not measure decoded HTTP body
bytes. They exclude kernel buffering, TCP/IP overhead and retransmissions.
Ciphertext overhead allowance replaces the former pip-wheel early body-size
rejection. Exact final filename, size and SHA256 are checked independently before
wheelhouse admission; disk and memory ceilings bound transient decoded content.
No artifact is installed merely because it fits a transfer quota.

## Phase order and inference boundary

1. Acquire and independently verify all 22 locked wheels, one at a time.
2. Offline install, pip check and installed-byte inventory.
3. Acquire the locked official source group, then the same model checkpoint.
4. Offline finalize and verify runtime asset records.
5. Only after complete setup and network teardown, run the unchanged inference
   worker on the original 16 clips, in its original eight-file blind code mount.

Inference has no broker socket, transport implementation, campaign context, peer
transcripts or human labels. Failures retain the full 16-row denominator and zero
new audio attempts when setup did not finish. This remains exploratory
source-screen evidence, not training admission, human gold or shipping approval.

## Verification boundaries

All source-only tests run through the existing test_contract.py normal/-O/-OO CI
entrypoint. Real local Unix/loopback fixtures may be unavailable in a restricted
editing environment; they are mandatory on GitHub Actions and cannot silently
skip there. No source test downloads packages/models or performs inference.
Actual Docker limits, the pinned image's pip capabilities and official TLS
acquisition are verified inside the separately admitted campaign. A failure in
those checks consumes that campaign attempt and stops before model/audio work.
This design does not establish that pip will resolve the earlier HTTP 403.
