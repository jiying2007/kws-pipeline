# Offline dual-ASR evidence import

Standard-library-only, model/vendor-neutral research tooling. Python 3.10+ on
Linux; no pip install, model download, inference, audio decoding, or network call.
The SSC305 runtime is unaffected: this is an offline research tool, not a device
ASR, KWS decoder, TTS implementation, or shipping qualification gate.

## What the result means

The CLI validates an **external declaration**: two output files claim a particular
recording ID, WAV SHA256, model identity/revision/run, manifest, and rules. It
checks exact association; it does **not prove that either model ran, or that a
model actually consumed those WAV bytes**. No WAV file is opened. Immutable
revision syntax and distinct model IDs do not authenticate models or prove their
independence. The output explicitly records these limits.

Keep full decoder raw tokens/output and execution receipts in the outer runner,
bound to recording ID, original WAV SHA256, model/revision/run, decoding options,
and transcript extraction. Adapters and launchers are intentionally outside this
module. `raw_text` must be the minimally extracted transcript, without spelling,
homophone, repetition, or punctuation repair. Only an independently audited
adapter may remove decoder metadata. Residual `<asr_text>`, `<|...|>` or leading
`language ...` makes the text evidence unknown.

Human labels are also predeclared input assertions, not independently certified
here. The tool preserves them exactly, including unknown. A positive or negative
machine consensus is separately reported as weak evidence and never becomes
human gold. Both models can be jointly wrong; paired errors are listed explicitly.
Selected probes do not establish population performance, acoustic correctness,
full-sentence accuracy, CER/WER, occurrence counts, word times, FAR/hour, or
release readiness.

## Predeclare, then import

Before collecting model output, freeze the manifest, rules, and licensed external
lexicons, and record the manifest/rules **byte SHA256 values independently** in
your experiment plan. Supply those values to `--manifest-sha256` and
`--rules-sha256`. Do not derive fresh expected hashes from changed files during
import. This tool checks supplied hashes; it cannot establish when predeclaration
occurred or prevent an operator from replacing the entire plan.

The manifest declares the exact two models, including their run IDs. Each model
output must match its declared slot. Every manifest recording must occur exactly
once in each output, including explicit failed or not-run rows. Row order may
change; there is no filename/content-based remapping. Unknown IDs, duplicate IDs,
missing/extra rows, WAV SHA drift, model drift, rules drift, and execution-kind
mixing fail closed.

All six input paths are explicit. Rules contain no paths and cannot read an
arbitrary file indirectly. Input files must be distinct regular files; final
component symlinks and input hard-link aliases are rejected. Choose a fresh
output path outside the input file set. Existing outputs, directories, symlinks,
and hard links are never overwritten. Fully serialized output is installed with
an exclusive hard link in the output directory; an output appearing during the
run also causes rejection. The parent output directory must already exist and
support hard links. Use a trusted, stable directory; this is not a sandbox against
a malicious actor replacing ancestor directories during execution.

Success writes one JSON readout and a JSON receipt to stdout, both carrying the
predeclared manifest/rules SHA, rule version, exact model declarations and all
six input byte hashes. The receipt also hashes the readout. Exit 2 means rejected
input or output I/O failure; stderr carries the reason, without transcript text
or private input paths. Validation failure creates no result file. No partial
accepted readout is published on serialization/write failure.

## Synthetic example

All fixtures are invented, including the eight IDs, WAV hashes and model names.
There are no corresponding WAV files, real listening labels, bundled third-party
lexicons, model weights, inference adapters, or execution receipts. The tiny
character/phrase dictionaries were authored solely for this test. They are not
coverage-complete or suitable for production Chinese transcription.

From the repository root, with `/tmp/asr-example.json` absent:

```bash
python3 research/asr_evidence/calibrate.py \
  --manifest research/asr_evidence/tests/fixtures/manifest.json \
  --manifest-sha256 5b02185af92acd1ff2597c477da6560c27448f594dff9810efb1951115d93da3 \
  --rules research/asr_evidence/tests/fixtures/rules.json \
  --rules-sha256 a16a6218e7102179ecf41d8095eca3d96227965dd44c0e4429e42a664e05d1ac \
  --characters research/asr_evidence/tests/fixtures/characters.json \
  --phrases research/asr_evidence/tests/fixtures/phrases.json \
  --model-a research/asr_evidence/tests/fixtures/model_a.json \
  --model-b research/asr_evidence/tests/fixtures/model_b.json \
  --output /tmp/asr-example.json
```

## Exact input contract

`calibrate.py` is the authoritative validator. Unsupported schemas/fields are
rejected rather than supported through compatibility branches. All objects below
have **exact keys**. Input is strict UTF-8 JSON: duplicate keys, NaN/Infinity,
float literals (including overflowing exponents), unpaired Unicode surrogates,
NUL and excessive nesting are rejected. Booleans do not satisfy integer fields.
IDs are nonempty, at most 256 Unicode scalars, with no whitespace/control code
points. SHA256 is exactly 64 lowercase hex characters.

### Manifest: `kws-asr-evidence-manifest-v1`

- `schema_version`: the schema string above
- `probe_set_id`: a declared identity for this probe collection
- `execution_kind`: `synthetic_fixture`, `not_run_template`, or
  `declared_model_output`
- `target_order`: 1–4 unique NFC contiguous-letter targets, each 2–16 scalars
- `models`: exactly two objects, in model-a/model-b order, each with
  `model_id`, `revision` and `run_id`; revisions are immutable 40/64 lowercase
  hex strings, not floating branches. Model IDs must differ
- `records`: 1–256 objects, each with `recording_id`, `wav_sha256` and
  `human_target_presence`; labels are a target-order array of `positive`,
  `negative`, or `unknown`

`fixture-only-` model IDs are mandatory for synthetic fixtures and forbidden for
other execution kinds. This prevents accidental evidence-level mixing; it is
not an authenticity or anti-forgery mechanism.

### Rules: `kws-asr-evidence-rules-v1`

- `schema_version`: the schema string above
- `rule_version`: exactly `conservative-text-evidence-v1`
- `target_order`: exactly the manifest's ordered targets
- `lexicons`: exactly `characters` and `phrases`, each an object with `sha256`
  and integer `entries` (1–100,000)
- `uncertainty_markers`: 1–32 unique, normalized, case-insensitive marker
  strings, each at most 64 scalars. The baseline markers from `REQUIRED_MARKERS`
  must all be present; a rules file may add blockers but cannot remove them

Character input is a JSON object from canonical decimal Unicode codepoint strings
to comma-separated readings, for example `{"26143":"xīng"}`. Phrase input maps
2–128-scalar NFC letter sequences to one reading-array per character, for example
`{"星河":[["xīng"],["hé"]]}`. Readings must be NFC, nonempty, at most 32 scalars,
with letters and optional tone digits/colon; up to 16 distinct readings per
character are retained. Dictionary files must match their declared byte hashes
and entry counts. Each target must resolve to a complete unambiguous reading.

External full lexicons are deliberately not redistributed. Obtain permission or
a suitable license separately, export to these exact formats, hash the exports,
and preregister those hashes in rules. No pypinyin package is imported, installed,
or presumed licensed by this module. The small test dictionaries do not imply
complete lexical coverage. Missing readings and polyphonic ambiguity cause
abstention for unmatched targets.

### Each model output: `kws-asr-evidence-input-v1`

- `schema_version`: the schema string above
- `execution_kind`: exactly the manifest declaration
- `manifest_sha256`, `rules_sha256`: exact predeclared byte hashes
- `model`: exactly the manifest's corresponding `model_id/revision/run_id`
- `records`: exactly one object per manifest ID with these fields:
  - `recording_id`, `wav_sha256`: exact ID/hash association
  - `status`: `success`, `error`, `timeout`, or `not_run`
  - `raw_text`: string or null; success requires a string, even when empty
  - `completeness`: `complete`, `incomplete`, or `unknown`; success alone is
    not a completeness declaration
  - `quality_flags`: a unique array from `missing_characters`, `incomplete`,
    `ambiguous`, `non_speech`, `decoding_warning`

A not-run row must contain null text, unknown completeness and no quality flags.
Every row in `not_run_template` must be not-run. An entirely not-run envelope
cannot claim `declared_model_output`. Error/timeout text may be retained, but
cannot produce a definite target state. A declared output with only failures is
valid evidence of attempted execution **as asserted by its producer**; coverage
is zero and no model quality claim follows.

## Conservative text rules

1. Keep raw text. Apply NFC and collapse whitespace only. Never apply NFKC,
   delete punctuation, repair words/homophones, or suppress repetitions
2. Non-success, empty/punctuation-only text, incomplete/unknown completeness,
   quality flags, mandatory uncertainty markers, decoder metadata, or unsupported
   controls make every target unknown
3. Otherwise, contiguous literal target presence is positive **text evidence**
4. For an unmatched target, unknown/OOV or ambiguous dictionary readings, a
   one-character edit/deletion/insertion, boundary-separated target, or a
   dictionary phonetic-confusable candidate make its state unknown
5. Only remaining successful, complete, nonempty, unflagged unmatched text is
   negative text evidence

Dictionary matching uses greedy longest phrase lookup, then all character
readings; whitespace and punctuation remain boundaries. Tone-mark removal
preserves u/ü distinction (tone-number suffixes are also ignored for toneless
comparison). Candidate matching allows up to one toneless syllable mismatch.
These are conservative text heuristics, never measured tone/pronunciation or an
acoustic verifier.

## Readout and bounded resources

Per-model, per-target counts report known human positives/negatives, TP/FP/FN/TN,
unknown-on-positive/negative, human-unknown bits, error identities, and known-bit
coverage as numerator/denominator/fraction. Unknown is excluded from definite
errors, but **remains in the known-label coverage denominator**. A zero denominator
has null fraction. Human-unknown bits never contribute to known-label metrics.

Paired readouts report jointly covered known bits, common errors, definite
mismatches, one-unknown disagreements, and both-unknown bits. Per-record output
keeps normalized/raw transcript, conservative derivation blockers, dictionary
candidates, human labels, weak machine consensus, and unchanged final labels.
There is no pooled accuracy or model-reliability score.

Hard limits are fixed in code, not operator-overridable: manifest 512 KiB; rules
64 KiB; each model output 2 MiB; each dictionary 8 MiB; each raw transcript 1,024
scalars; total raw transcript per model 32,768 scalars; JSON depth 8; final readout
32 MiB. Limits apply before semantic use; all file parsing and hashing uses the
same bounded bytes. Exceeding a limit rejects the import rather than truncating
text or silently omitting rows.

## Checks

```bash
python3 research/asr_evidence/tests/test_calibrate.py
python3 -m py_compile research/asr_evidence/calibrate.py research/asr_evidence/tests/test_calibrate.py
python3 tools/test_inventory.py --root research/asr_evidence/tests --workflow .github/workflows/research-asr-evidence.yml
```

The path-scoped `research-asr-evidence` workflow runs these offline checks without
installing dependencies. Tests exercise CLI success/failure, predeclaration and
model drift, synthetic/template state separation, immutable human unknown,
manual metric totals, jointly wrong consensus, Unicode, lexical ambiguity,
strict JSON/type/length/file bounds, ID/SHA joins, and exclusive output publication.
No fixture is evidence of real ASR execution or acoustic model performance.
