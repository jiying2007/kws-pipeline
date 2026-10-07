# Source-consolidation verification report

Date: 2026-10-07. Base: `c44dcf6fadb60b4a084200f4565b8e5c2503eac4`,
base tree: `a9a07847b3c778a8385762b6bbf3ef2087db7e82`.

## Passed locally

- Retention verification: **718 exact retained paths**, **834 original source
  provenance rows**, **2 pointer-only historical audio ZIP records**
- All **54 pre-existing active workflow files** match the main base exactly.
  The only new active workflow runs the explicit offline source checks
- Original source-freeze metadata validates through the archive mapping:
  token preparation **41 files**, fixed300 **37 files**, Cosy49 **36 files**.
  The historical Cosy30 payload SHA-256 matches its original armed record;
  the usable tree is explicitly disarmed
- **55/55 synthetic/mock/static command groups PASS**, including **820 unittest
  executions**. Every group runs separately in the temporary original-path
  source projection. NumPy **2.3.5**, SciPy **1.17.0**, Python **3.12**
- Independent first-prefix oracle PASS: **612 cases / 5,523,157 paths**, plus
  **11 edge checks** and **8 mocked underflow checks**. This does not change
  the failed saved-candidate conclusion
- fixed30 source: **23 files parsed**, Qwen6 source: **36 files parsed**. These
  are syntax checks only, not runtime, package, model or acoustic validation
- New source-retention/runner guard suite: **32 tests PASS**, including changed
  bytes/modes, provenance and projection errors, unexpected active workflows,
  rearming, pointer-only audio, symlinks, path traversal, generic discovery,
  execution flags, environment shell escapes and report-filename escapes
- N1 companion: **12 synthetic tests PASS** (included in the 820 above). These
  cover geometry and malformed numeric traces; they make no historical PCM claim
- N1 full saved-only preparation and verification: **2/2 explicit commands PASS**.
  The closed **19-file / 137,742-byte** fixture set was hash/size checked. It uses
  the existing **47,600-byte public derivative WAV**, not the native WAV.
  Numeric observations and unchanged PCM were reproduced; the result remains
  **CONTEXT_UNVERIFIED**, with **zero model/frontend/decoder/network calls**
- Core regression guards remain intact: original **102 retained source files**
  match; core retention **10 tests**, cleanup retention **3 tests**, and the
  original core workflow's exact test inventory pass

The synthetic suite does not install dependencies or acquire fixtures. The new
CI has a separate version-pinned, wheel-only NumPy/SciPy installation step; no
model/runtime packages or training are installed by this workflow. N1's full
saved fixture preparation is local-only and is not silently included in CI.

## Additional public saved-evidence validation: passed

All nine remaining saved-evidence suites were supplied their exact, already-
public fixture sets and passed: Qwen20 public-record validation **17 tests**,
both N0 saved suites **13 + 3**, Melo2 tail context **8**, Melo3 negative context
**9**, Melo5 A20 saved verification **12**, Melo5 leading context **10**, Melo6
human adjudication **9**, and Melo6 saved join **4**. These **85 additional test
executions** bring the source total to **905 unittest executions**.

The final wrapper ran **11/11 saved command groups PASS**, including the nine
suites and the two N1 preparation/verifier commands. Together with the synthetic
suite, **all 66 explicit command groups PASS**. Original source bytes and data
pins remain unchanged; fixtures live outside the publication tree.

Bounded fixture accounting:

- Qwen20 and N0: **33 exact files / 2,143,488 bytes**, including their original
  closed-set evidence manifests. Compressed numeric records are bounded and
  hash-checked before decompression; no audio/model input is involved
- Melo: **105 fixture destinations / 2,752,266 bytes**, representing **98 unique
  objects / 2,519,291 bytes**. Reusing existing public source leaves **58 objects /
  1,974,266 bytes** to retrieve. The largest object is **274,118 bytes**
- N1: **19 exact files / 137,742 bytes**, as described above

Text fixture reads used the GitHub connector. Its UTF-8-only format cannot
return the required small binary objects, so those used credential-free,
size-bounded reads of the same exact immutable official raw URLs with SHA-256
and Git blob verification. No bulk repository/archive download was used.

Melo leading and saved-join checks read already-public synthetic ZIP/WAV bytes
to verify recorded hashes and PCM correspondence. They do not play audio,
generate clips, replay conversion or invoke a model/frontend/decoder. The
historical machine result remains **failed/inconclusive quarantine, 2/6 dual
exact intent matches, zero training admission**, and human M6 remains
**partial/UNKNOWN**. Passing saved checks does not improve those conclusions.

## Deliberately non-executable historical tests

The **35-test #475 historical endpoint suite is source evidence only**. A
preliminary invocation stopped at import because its deliberately omitted
mixed98 metadata was absent. The archive's own contract confirms these tests
are not public runnable checks. They are excluded from the executable allowlist;
no private missing inputs were sought and no source hashes were relaxed. The
14 invented archive-transport tests do pass. See `source-excluded-checks.json`.

No historical candidate launcher, model frontend, decoder, training, inference,
package/runtime acquisition, old workflow dispatch, old label activation,
branch recreation or scientific rerun was performed. No private artifacts or
active acquisition outputs were inspected. The original fixed30 blind input ZIP
was not materialized or added to this source batch.

## Publication boundary

This report describes local verification of the prepared source tree. Exact
remote-tree readback, draft PR review and hosted CI must be checked against the
published head before merge. Software tests and saved reconstruction do not
provide independent acoustic, physical-board or shipping qualification.
