# Independent-data metadata preparation (draft)

Standard-library-only metadata preparation. The frozen prior revision was independently
reviewed on eight private TextGrids and five private JSON ledgers. This revision
extends finite unknown-marker screening and adds explicit production identity and
partition checks. It has synthetic tests only; the prior executed wrapper and
its receipts remain unchanged. No model, audio, network, training or TTS path exists.

## Priority 1: candidate contact history

For a separately authorized private preparation:

1. Verify the supplied ZIP byte lengths and SHA-256 values, then every member
   against the private MANIFEST before using contracts or ledger locators.
2. Keep the handoff, raw transcripts, ledger contents, and derived per-session
   outputs outside this public repository. Read source ledgers without changing
   them. Never upload private inputs or their plaintext excerpts.
3. Adapt the candidate contract to a private JSON array. Official Test membership
   is not established by mirror IDs. The official/mirror license discrepancy
   must remain unresolved until source-specific evidence settles it.
4. Scan the located train146/eval38 inventory, actual20 stream inputs, and 984-row
   split/group/source_metadata ledger. Record each input hash and its scope.
   L60/S1800 AISHELL4 evidence must not be treated as AliMeeting Test membership.
5. `exposure` emits exact-ID token clues tied to input hashes. It is only a first
   pass, not a semantic audit of split/group/sourcefamily/session relationships.
   Review those fields explicitly under the private contract; leave incomplete
   history, speaker identity, sourcefamily and pretraining exposure `unknown`.
   Zero string matches never establish independence or admission.

Example (all paths are user-supplied private files):

```sh
python3 metadata.py exposure --candidates /private/candidates.json \
  --ledger /private/inventory.json --output /private/exposure-clues.json
```

## Priority 2: one old-development mask preparation

The draft adapter requires `sample_rate`, integer `num_samples`, explicit
`review_terms`, and unchanged `historical_pending`, plus `session_id`,
`source_family`, and expected lowercase `textgrid_sha256`. The CLI rejects a
filename other than `<session_id>.TextGrid` or a hash mismatch before masking.
This binds the caller-reviewed contract, not independent proof of audio identity. Validate these fields against
the private contract and historical audit before producing real masks. The
sample extent must come from provenance-backed metadata, not guessed TextGrid
length. This run does not verify audio.

```sh
python3 metadata.py mask --textgrid /private/session.TextGrid \
  --contract /private/adapted-contract.json --output /private/session-mask.json
```

The strict long-text IntervalTier parser handles UTF-8/BOM, UTF-16, escaped
quotes, and multiline strings without confusing quoted content with structure.
Unsupported syntax, invalid extents, overlapping same-tier intervals, and
trailing garbage fail closed. Numeric input uses Decimal with bounded precision.
Support starts round up and ends round down; review starts round down and ends
round up. Review wins over support, then unknown fills the entire remaining
sample timeline including transcript gaps and the audio tail. Intervals are
mutually exclusive `[start_sample, end_sample)` and cover `[0, num_samples)`.

A supported transcript region is only a candidate, not verified speech absence
or a negative. Tier union never proves all sound sources were annotated. Empty
and annotation-only markers supply no speech support. Review terms are literal
per-interval matches, including inside markers; there is no normalization of
窝/屋, no concatenation across tiers, and no cross-segment or phonetic search in
this draft. Such searches, if later added under the contract, may only produce
review hints. `screening_complete=false` remains explicit.

All masks fix development, new independent hours 0, human gold false, negative
admitted false, accepted negative seconds 0, expected null, human review pending,
speaker overlap unknown and audio verified this run false. Historical pending
content is copied under `historical_pending` without changing old artifacts.

## Validation and remaining work

```sh
python3 -m unittest discover -s research/independent-metadata-prep -p 'test_metadata.py' -v
```

For this revision, only public synthetic tests run. The suite exercises exact Decimal boundaries,
conservative interval partitioning, priority, parser rejection, escaped syntax,
unknown gaps/tails, no cross-tier/segment joins, distinct characters, pending
preservation, and unknown exposure after either positive or absent ID clues.
Repository-wide model/audio/replay suites are not run for this draft scope.

Before any further real preparation, verify private contracts, source identities,
and output scope. Keep private material outside the repository. Production
identity and complete interval-partition validation use explicit exceptions;
they remain enabled under Python -O and -OO. The historical one-off wrapper
is archived unchanged and is not the reusable CLI.

## Independent review correction (2026-10-19)

The original nine fixtures and two new regressions pass. Session contact clues now
recognize underscore channel suffixes without confusing longer numeric IDs.
Mixed bracket/annotation syntax conservatively blocks transcript support, including
when another tier overlaps it; target review terms still take precedence.
These changes do not establish complete phonetic screening or acoustic labels.
The reviewed code subsequently parsed eight hash-bound private TextGrids in a
metadata-only task. Aggregate annotation candidate coverage was 13972.870 seconds
and unknown coverage was 1165.345 seconds. Accepted negative duration and new
independent exposure remain zero. No audio or model was read or executed.
Private per-session masks, source identities and ledger contents are not included.

## Finite unknown-marker rules in this revision

Any occurrence of [] () （） 【】 <> {} ［］ 〈〉 in an interval blocks
support conservatively, including mixed spoken text and overlapping tiers.
Case-insensitive ASCII tokens UNK or UNKNOWN with no adjacent ASCII letter,
digit or underscore also block support. An entire trimmed label equal to 未知,
不清, or 听不清 blocks support. Target-review terms still have higher priority.
These are finite screening rules, not an exhaustive annotation grammar; other
unknown spellings and arbitrary embedded uncertainty phrases are not detected.
Normal spoken sentences containing these Chinese words are not automatically
classified unknown unless they meet the exact-label or delimiter rule.

The independent reviewer found no added-marker instances in the eight frozen
inputs. No historical result was regenerated or relabeled. Historical aggregate:
eight development sessions, accepted negative seconds 0, new independent hours 0;
twenty candidate sessions retain unknown exposure/independence after five scoped
ledger scans. No-match is not independence. Literal, phonetic, or segmented
screening can only flag review candidates and never proves target absence.
