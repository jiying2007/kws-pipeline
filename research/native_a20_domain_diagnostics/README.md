# Offline validators for A20 domain and decoder evidence

This directory contains newly authored, stdlib-only saved-record validation code
and invented adversarial tests. It consumes the corresponding scientific record
in kws-data at `research/2026-10-04-domain-decoder-diagnostics`.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/verify_saved.py --data-root /path/to/the/data-record
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

No dependency installation, network, subprocess, native build/run, model, audio,
feature extraction, softmax, decoder replay or live probe is used. The exact CTC
checker independently enumerates the already frozen six invented tables; it is
not a changed decoder. The 28 fake tests (27 original tests plus one publication-gate test) exercise the offline validator only.

The record preserves UNKNOWN/unadjudicated domain results, conditional Qwen
reconstruction, all six witness outcomes, and the earlier Qwen20/N0 failures.
A stronger joint-path metadata contract is not evidence of an original C-port
bug or a quality gain. The existing A20 decoder remains unchanged. This source
is additive to the frozen A20 baseline and does not touch shipping, training,
model assets, qualification seeds or existing workflows.

See DATA-REFERENCE.json for the paired record identity, and the data record's
RIGHTS-SOURCES-PRIVACY.md and REPRODUCIBILITY.json for the full boundary.
New code is under the existing kws-pipeline Apache-2.0 license. Original source
recordings and third-party models retain their separate terms; none is included.

## Saved-only CI publication gate

The one path-filtered source workflow, `.github/workflows/native-a20-domain-saved.yml`,
uses read-only repository permissions and an immutable checkout action. It reads
DATA-REFERENCE.json, rejects NOT_PUBLISHED or a closed publication_ready gate,
and checks out exactly the 40-hex data commit. The validator's --require-published
mode independently binds that actual checkout HEAD and the complete manifest
SHA256/byte count. It then validates saved evidence, runs the 28 pure tests, and
requires tracked source and data to remain unchanged. No dependency installation,
artifact upload, cache, audio/model access, live probe or native execution is added.

Local candidate validation may omit --require-published. That does not satisfy CI.
The source publication gate must remain closed until the data commit is published
and verified, the pin/manifests are explicitly refreshed, and that updated freeze
has passed separate review. Updating a pin creates a new recorded source identity;
it never silently changes an old freeze or authorizes publication by itself.
