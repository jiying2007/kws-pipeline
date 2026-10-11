# Original 16-clip three-ASR comparison

This offline, stdlib-only comparison joins the retained Qwen0.6 and SenseVoice
continuation evidence to verified Whisper fourth run 38126849922. No new model
execution occurs here. Every row retains the original WAV and PCM identity.

## Results

- Native EOT and valid triple coverage: 16/16; fixed and observed denominators 16.
- Literal unanimity: 0/16.
- Historical punctuation-only normalization: 6/16 (37.5% agreement coverage),
  comprising 4 target matches and 2 common target mismatches; 10 disagreements.
- Explicit punctuation-plus-whitespace secondary normalization: 7/16 (43.75%),
  comprising 5 target matches and 2 common mismatches; 9 disagreements.
- The sole additional secondary agreement is clip-000005 (Whisper: `你好 你好`).
- Common mismatches are clip-000006 (`小屋小屋`, target `小窝小屋`) and
  clip-000014 (`小屋`, target `小窝`).
- Neither normalization merges homophones, converts scripts, changes case,
  removes symbols/OOV, or substitutes generated target text.

These are agreement/target-reference statistics, not acoustic accuracy. Native
EOT does not prove verbatim acoustic completeness, which remains unknown.
Three independent ASR unanimous text can be a working-truth candidate under
provenance, rights, role and OOV gates. Every row here remains source_screen_only,
human_gold=false, training_admitted=false, shipping_approved=false. This
retrospective comparison is not a preregistered three-ASR training gate.

## Files and integrity

- `comparison.json`: complete 16-row raw texts, both normalization policies,
  categories, actual/fixed denominators, target-reference matches and hashes.
- `report.zh.txt`: concise Chinese summary and all 16 rows.
- `verification.json`: independent row/count/hash checks; 24 unit tests passed.
- `comparison-input.json` and `authentication.json`: byte-identical copies of
  historical prepared public inputs and their authentication record. Their old
  NOT_RUN annotations describe preparation time, not this completed comparison.
- `compare.py` and `test_compare.py`: unchanged original comparator and tests.
- `compare_verified_run.py` and `test_compare_verified_run.py`: additive verified
  comparison and explicit normalization/denominator tests.
- `SHA256SUMS.json`: exact SHA-256 and byte length of every other package file.

The new result binds `terminal-outcomes.json`, a logical member of the exact
archive in the parent directory's `ARCHIVE.json`, to SHA-256
115773be9a64ee04cb0d5bb5002268528e9a4573ef094d754565187f95942a10.
The parent `verification-receipt.json` independently validates native evidence
and has SHA-256 4e40bff753892fef4cd5754f786fa35a759f6f1dbfdb59243ceacf0dbe09845e.
Comparison input SHA-256 is
e45b6620c8491b204234ddb9e506ec93b947807e43e1b7fc1cf17ecb95bee51e.

No historical outputs were edited. Original SenseVoice still retains one
failed_no_retry and fifteen not_run separately from its 16-success continuation.
Private human observations were not read, copied, or changed.

## Rechecking

Run `python -m unittest discover -s PATH_TO_THIS_DIRECTORY -p 'test_compare*.py'`.
The 24 unit tests use in-memory synthetic terminal fixtures only; they are not
model outputs. Actual outputs were checked independently across all 16 rows.

For full source reauthentication, run `compare_verified_run.py --help` and
supply an extracted verified terminal file, its SHA above, the parent verified
receipt and its SHA above, the repository source root, run ID 38126849922 and
head SHA 1986acbeb790833a92f5014fccb88a931c0f3e7f. Choose new output paths; the
script refuses to overwrite files. The published JSON uses repository-relative
provenance paths and a logical archive member, whereas a local re-run records
its own local input locations. Analysis content is unchanged by that rewrite.
