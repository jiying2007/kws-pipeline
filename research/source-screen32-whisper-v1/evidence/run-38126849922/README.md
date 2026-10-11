# Fourth Whisper run: durable public evidence

Run [38126849922](https://github.com/jiying2007/kws-pipeline/actions/runs/38126849922),
activation `1986acbeb790833a92f5014fccb88a931c0f3e7f`, tree
`8c0088d93e4b1e9fd8d28e92dc0ea48794f4a015`, completed its one-shot fourth campaign.
All 16 original anonymous inputs had exactly one attempt and a validated natural
Whisper EOT in the actual decoder-update trace before finalization. The three
previous setup failures and their zero-audio outcomes remain immutable.

This is exploratory source-screen evidence. Acoustic verbatim completeness is
unknown. It supplies neither human gold, training admission nor shipping approval.
Natural decoder termination does not establish that every spoken word was heard.

The separate [16-row comparison](comparison/README.md) retains punctuation-only
agreement of 6/16 and explicitly labeled punctuation-plus-whitespace agreement
of 7/16. It does not merge homophones or change any original target or transcript.
The [Chinese report](comparison/report.zh.txt) contains all 16 rows.

## Durable carrier and exact provenance

`chunks/terminal.zip.part-000` and `part-001` are ordered, unmodified pieces of the
original Actions artifact ZIP, not recompressed copies. Concatenating the chunks
restores exactly 1,843,987 bytes with SHA256
`b938916aeee60210863fb2f86bf7d1de89663dc9a1775e7910d702a64af14cf4`.
`ARCHIVE.json` binds both chunks and every member by exact path, size and SHA256.
`provenance.json` binds that manifest, run, execution source freeze and independent
verification receipts. The Git tree, rather than an expiring Actions download,
is the durable carrier. No signed download URL or API response is retained here.

There are 431 members totaling 4,636,835 decoded bytes: 410 JSON files and 22
locked official Whisper source/support files, with one JSON overlap. This
includes tokenizer tables and mel filters under the upstream MIT license, whose
exact LICENSE is inside the archive. Wheel bodies, model weights, audio files,
private human review material, credentials and raw CI logs are excluded. Model
and wheel hashes in receipts are metadata, not retained model/package bodies.

The compact layout avoids creating 431 loose files in the repository, including
the 2,455,235-byte installed-file inventory. Its tradeoff is that raw evidence
requires reconstruction or ZIP-aware access rather than individual GitHub text
views. All summary/verification files remain readable JSON. Existing repository
chunked-archive conventions are followed; each chunk is at most 1 MiB.

## Recheck without runtime execution

From the repository root, using only the Python standard library:

```sh
python3 -S -B research/source-screen32-whisper-v1/evidence/verify_fourth_archive.py
python3 -S -B research/source-screen32-whisper-v1/evidence/test_fourth_archive.py
python3 -S -B -O research/source-screen32-whisper-v1/evidence/test_fourth_archive.py
python3 -S -B -OO research/source-screen32-whisper-v1/evidence/test_fourth_archive.py
python3 -S -B research/source-screen32-whisper-v1/evidence/test_saved_comparison.py
```

Optionally append `--write-zip /new/path/fourth-terminal.zip` to the verifier.
It creates a new file only after all checks succeed, refuses an existing file,
and does not extract or run any archived source. No pip, installation, model
load, inference, network acquisition, activation or rerun is performed.

The verifier checks bounded canonical paths, regular unencrypted ZIP members,
closed membership, JSON duplicate/nonfinite rejection, all chunk/member hashes,
the original artifact freeze, all 262 frozen source entries, original WAV/PCM
and zero-padded float32 identities, native configuration and durable trace
bindings, independent tokenizer decoding, pip compatibility outcomes, exact
locked download records, resource scopes, transfer-ledger arithmetic and cleanup.
The independent receipts additionally preserve the terminal API identity checks
performed before archival. CI runs the saved-only/adversarial suite in normal,
`-O` and `-OO` modes, separately from immutable campaign source tests. The saved
comparison check authenticates the closed 11-file package and original retained
sources, then recomputes all 16 rows/counts/rates and the Chinese report directly
from the in-memory verified archive.

## Observed bounds and remaining uncertainty

- Host elapsed time: 243.723451168 seconds, below the 2400-second host ceiling.
  There is no separate setup elapsed receipt. Setup occurred within this entire
  successful host run, below its 1200-second ceiling.
- Setup accounted for 774,057,552 of 1,610,612,736 allowed bytes. This is
  bidirectional tunnel socket receive accounting before forwarding, excluding
  CONNECT headers, kernel buffering and network overhead. It is not total wire
  traffic. Image transfer was separately observed, not exact daemon accounting
  or a proven hard cumulative wire-byte cap.
- All 22 wheel results record pip 25.0.1, successful help, only
  `--resume-retries` absent, and `audited-25.0.1-no-resume` with the frozen
  eleven-file source audit reporting a match. The runtime image's pip source
  bodies are not retained, so these remain executed-source attestations.
- Exact identities of 22 wheels, 22 source/support files and the original model
  are bound to locked download records. Official source bytes are retained and
  checked directly. Model and wheel bodies were neither retained nor downloaded
  again for verification.
- All 50 setup container inspect scopes and 46 retained broker/worker kernel
  scopes pass. Standalone source/model worker kernel records are absent; their
  inspect scopes and frozen phase control flow are the available evidence.
- All 50 setup containers were removed across 26 clean cleanup receipts, without
  deadline overruns. Inference exited zero, recorded no OOM, and cleanup completed.

The archive changes no decoder, input, model, frozen execution gate, historical
source freeze or past run. It grants no additional experiment authorization.
