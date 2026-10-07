# Experiment quality guards

Small standard-library gates for actual-label coverage, cross-generator identity,
exposure, independent event changes, timing availability and saved-observation
interpretation. The public adapter supplies portable examples and public historical
outcomes. This candidate adds an opt-in original-plan consistency check and observed
ASR decision regressions to the existing core and dataset-admission CLI.

This is an isolated research addition. It does not integrate a production builder,
run an acoustic model, grant training authority or qualify a product. No default,
model, ABI, release or existing research branch changes are included.

## Run

From the repository root, using Python 3.9 or newer:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s research/experiment_quality_guards/tests -v
python3 -B research/experiment_quality_guards/review_saved.py
python3 -B research/experiment_quality_guards/admit_dataset.py research/experiment_quality_guards/examples/balanced-input.json
```

The `experiment-quality-guards` workflow runs these 77 tests using system
Python and checks every test file against the workflow inventory. It triggers only
on this research directory or its own workflow file, has read-only repository
permissions and a three-minute timeout, and installs no dependencies. Existing
general CI does not discover this directory; a green unrelated job is not evidence
that these tests ran. This candidate's workflow commands have been checked locally; the candidate has not
been run on GitHub.

The tests and CLIs need only the files in this directory and Python's standard
library. The twelve admission CLI tests start only the pure Python JSON gate. The
supervision tests use fake process objects and fake text readers; they do not
launch a collector or read live process telemetry. The example data is invented.

## Call the admission gate

A builder can import `assess_admission` from `admit_dataset.py`, supply a JSON-shaped
payload and stop on a nonzero return code:

```python
report, exit_code = assess_admission(payload)
if exit_code:
    raise ValueError(report)
```

Input fields are `requested_qualification`, `rows`, complete prior `history`,
`declarations`, `policy`, `frozen_policy_sha256` and optional boolean
`require_unseen_voice`. Use canonical `train`, `dev`, `heldout`, `background` or
`regression` splits. Persist the returned identity ledger without dropping prior
history. Freeze the policy digest before examining results, not after them.

- Exit 0 allows only the explicitly requested qualification
- Exit 1 rejects that qualification
- Exit 2 means invalid input, a changed policy pin or an I/O failure

`balanced_source_groups` requires every declared balanced group to meet frozen
absolute K1/K2 and nonwake minima and have no identity/exposure conflicts. Only
complete independent human actual words count. Generation intent, ASR agreement,
training greedy text and repeated audio cannot fill a missing category. An unseen
voice claim additionally requires verified declared lineage absent from history.

`exposed_regression` requires explicit EXPOSED regression use and no identity
conflicts. It permits historical report/code regression despite insufficient
coverage; `coverage_eligible` stays false even if nested counts meet the minima.
Neither mode grants training, product or fresh-validation authorization. Lineage
attestations are conflict-checked, not acoustically authenticated. Input manifests
still need independent source, rights and completeness review.

## Optional original-plan consistency check

Set `requested_label_basis` to `original_intended` to require the extra label check
before the existing requested qualification can pass. Each row supplies separate
`actual_text`, `intended_text`, `review` and exactly two saved `asr_results` objects
with `status` and `raw_text` (a missing observation may be null). No ASR is run.

`label_preparation` reports two distinct results:

- `planned_lexical_support` is SUPPORTED only when both completed, nonempty
  normalized transcripts equal the intended text. Disagreement or consensus on
  another string is REJECTED. A missing plan or unresolved observation is UNKNOWN.
  Missing intent is never filled from a human or machine transcript
- `original_complete_label_eligible` is the extra plan-consistency check only. It
  also requires the existing clean, independent, complete human actual-word review,
  human actual text equal to the plan and no saved ASR quality flags. It does not
  certify CTC encodability, independent evaluation, dataset admission or training.
  `human_actual_review_complete` reports the inherited actual-word review gate;
  `acoustic_completeness` stays UNKNOWN because this text-only helper does not
  independently adjudicate acoustic endpoints

The gate returns exit 1 when an `original_intended` request has any failed label
check, even if the human actual labels satisfy the coverage minima. Explicit
`human_actual` reports the same diagnostics but preserves human actual-word
coverage: ASR errors cannot veto or rewrite that truth. Omitting
`requested_label_basis` preserves the earlier API behavior. The existing
actual-label helper remains responsible for CTC vocabulary eligibility; this
package never emits CTC targets. The existing identity/exposure gate and scope
restrictions still apply to both modes. This is not unattended gold labeling.

`fixtures/observed_asr_decisions.json` retains a small text-only selection from the
completed, previously exposed fixed30 comparison. E/X/L/I/P/D are observed common
ASR errors, but all six fail the original-plan lexical rule. They are not six
observed automatic false admissions. D/I both produced 小五, while human actual
labels remain 小窝/小屋 and original repeated plans remain separate. W has matching
text but UNKNOWN acoustic completeness; Q/T retain inaudible/null actual labels;
Z1 retains actual 你好 separately from its verified original plan 你好你好. Z3
retains actual 小挖 separately from intended 小窝 and keeps its OOV CTC exclusion.
A/B/Z2/Z4 protect genuine lexical support. All six Z original plans are a new
sanitized projection of the independently verified generation/recording join;
the frozen comparison's absent plans are unchanged. The original public generation
config hash is retained in the fixture and bound by `SOURCE_PINS.json`.
Counterfactual plan-completion tests are explicitly marked as invented cases.
All selected recordings remain calibration/code-regression evidence, never
independent accuracy evidence. No probabilities, thresholds, text normalization,
historical outputs, model calls or production builder are changed by this candidate.

The broader metadata/rights and continuous-evaluation contract in [PR465's quality
tools](https://github.com/jiying2007/kws-pipeline/tree/d8453ae8b2b2c620716b32c96d9647834ce96746/research/native_a20_quality)
remains separate. These narrower gates do not replace it. This addition has no
runtime dependency on that draft or the historical archive.

## Public historical evidence and invented examples

`fixtures/public_regressions.json` explicitly labels each evidence basis:

- The six exposed synthetic recordings retain their public actual labels, audio
  hashes and projected keyword/input-availability events. Original detection was
  1/2 wake recordings versus candidate 0/2; false events affected 1/4 versus 2/4
  nonwake recordings. Z5 becomes a miss and Z6 gains false K1 even though both arms
  have two total events. These are descriptive clip counts, not FAR or FRR
- Public Qwen training metadata and Cosy reference metadata identify the same Dylan
  reference WAV/PCM hashes. New waveform bytes do not establish an unseen voice.
  All six remain exposed regression-only evidence
- Reviewed synthetic coverage aggregates retain the missing positives: dev
  Uncle_Fu K1/K2 0/0, training Eric 1/0 and Vivian 0/1. These aggregates are not a
  replayable row-level review. The four anonymous coverage rows are explicitly
  invented examples of those failure patterns, with invented hashes; they are
  not renamed historical recordings or reconstructed labels
- The published old98 aggregate remains 28 retained matches: 25 unchanged, three
  later, none earlier. The saved costs are +300/+180/+120 ms. The 28 neutral timing
  rows use invented identifiers, coordinates and a 1-kHz rate solely to exercise
  those deltas; they are not restored old98 records. Word-tail and wall/service
  latency remain NOT_MEASURED
- Old98 lifetime RSS was 13,848/58,892 KiB; sampled RSS was 4,079,616 bytes in both
  arms. These scopes differ, with n=1 per arm and no model-memory improvement or
  regression established. Telemetry fixtures are invented; no historical
  process identifiers or per-source natural observations are included

`SOURCE_PINS.json` records immutable public URLs, byte counts and SHA256 values for
the old6 labels/results, generation references, training metadata and old98 summary.
The old98 logical summary was checked against its public archive-index member pin;
no archive restoration or model/audio loading is needed to run this package.
The public fixture contains no natural-source row identifiers, natural audio hashes,
raw natural records, storage locators or private-to-public source mapping.

Both historical candidates were terminated and not adopted. All reviewed synthetic
labels remain calibration/regression evidence. This package demonstrates rejection
of misleading claims, not improved acoustics, ASR accuracy, unseen-voice performance,
continuous false activations per hour, physical-board behavior or shipping readiness.

## Portable test helpers

`tests/vendor/actual_label_policy.py` is unchanged reviewed source. Its sibling
`audio_review.py` contains only the exact `ReviewError`, `require`, `valid_text` and
`normalize` definitions plus their two required standard-library/constant bindings.
`supervision_observation.py` contains six exact existing definitions plus their
required bindings; it contains no executor or launch function. Definition hashes
are recorded in `SOURCE_PINS.json`. Helpers are for tests, not newly adopted runtime
limits. No historical runtime, source tree, model or audio snapshot is bundled.

The actual-label helper keeps 小挖 as known nonwake while refusing it as an
encodable CTC label. Whitespace/punctuation are comparison-only normalization;
spelling, homophones and repetition are never repaired. `ctc_target` remains null.

See `TEST_REPORT.md` for the precise tested scope. Existing source licensing applies;
public synthetic provenance does not establish commercial voice/output clearance.
