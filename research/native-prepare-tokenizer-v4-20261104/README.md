# Prospective tokenizer-binding preparation v4

New candidate identity: `research/native-prepare-tokenizer-v4-20261104-v1`.
All118 consumed scan-v3 source identities remain immutable, along with every
older lock and workflow. No previous run is retried.

Only three stage mappings change: Sense capture and the Qwen/Sense source
manifests which bind its original bytes. The Sense implementation is exactly the
independently reviewed SHA256
`f9069a14142185cba54eafa97dc53a9d71fe1a2dab5329b6e6bb671999fcd3a7`.
It obtains the tokenizer path from the loaded native tokenizer's pinned bpemodel
binding, verifies the model/revision and measured receipt identities, then checks
exact original file bytes and serialized processor bytes. It does not invent a
tokenizer_file attribute on the real retained PreparedRuntime dataclass.

The tests use that retained dataclass definition with invented bytes. Earlier
safe diagnostics established only an exception class and stage; they did not
prove the exact missing attribute name in a past real run. The defect mechanism
and correction are established through source inspection and offline fixtures,
not by claiming a more detailed historical trace than was retained.

The Qwen and Sense executed-source manifests bind the new Sense capture bytes
at the unchanged staged pathname. Whisper does not execute this Sense module,
so its executed source bytes and worker remain byte-identical to scan-v3. A
future preparation still records its own new executed commit identity. No
unexecuted dependency or new model setting is fabricated for Whisper.

The host retains scan-v3's hash-pinned, cache-isolated boundary/scanner imports,
ROOT checks,64MiB output polling threshold, bounded whole-scan retries and final
post-exit accounting. Scientific loader bodies, native metadata capability,
no-audio guard coverage, all model/runtime locks, and three-model scope are
unchanged. Full ASR remains unconditionally blocked.

Same-job25GiB initial reserve, original16GiB primary/8GiB Whisper setup reserves,
40-minute single job,1800-second inner absolute budget,120-second launch reserve,
first-created branch/run-1/attempt-1, no retries and no new cleanup remain intact.
Public artifacts remain exactly status.json,capacity.json,safe-summary.json.
Native payload remains runner-private. No audio or inference is added.

This is a local candidate only. Tests invoke no ML framework, model, container or
remote service and do not establish a successful real preparation. Review and CI
must precede any separately authorized new execution.

## Preserved tokenizer filesystem boundary

The inherited regular() helper checks resolved path, type, links and size before
read_bytes. It is not an atomic fd-based snapshot or a hard read bound if a
concurrently writable file grows. This correction relies on the existing
read-only /runtime mount and completed setup process, followed by exact original
byte length/hash and serialized tokenizer equality. It does not claim separate
TOCTOU protection for concurrently writable paths. That different boundary would
require separate review; consumed regular() remains unchanged.
