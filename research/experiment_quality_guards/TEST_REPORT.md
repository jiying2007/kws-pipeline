# Validation report

2026-10-05, Python 3.12.14. The ASR decision candidate passes 77 standard-library
unit/CLI tests, including 59 inherited tests and 18 added regressions. The four
existing test files remain the workflow inventory: 41 quality-gate, 22
counterexample, 12 admission-CLI and 2 public-report tests. No dependency or workflow
change is needed. Each workflow test command also passes independently.

The complete tree was copied into a new temporary directory with no sibling
research packages and no PYTHONPATH. All 77 tests, the public saved-report CLI and
the balanced example admission CLI passed there. The new CLI integration regression
wraps the observed text-only fixtures in explicitly invented group/hash metadata,
then verifies exposed report-only acceptance under human_actual and rejection
under original_intended. Its wrappers are not historical recording identities.

## Added decision boundaries

- Observed E/X/L/I/P/D consensus errors fail original-plan lexical support while
  preserving human actual text. All six are off-plan errors; they are not evidence
  of six false admissions by the original-plan equality rule
- D/I retain actual 小窝/小屋 separately from repeated original plans. The actual
  saved ASRs both returned 小五. A separately marked counterfactual tests both ASRs
  completing the plan and still failing the conflicting human-label check
- Z1 has independently verified original plan 你好你好, actual 你好 and both ASRs 你好.
  Actual-label agreement is not original-plan support. Null/missing plans stay
  UNKNOWN and cannot be filled from actual text
- W preserves machine lexical support while acoustic completeness stays UNKNOWN;
  a complete original-label check fails. ASR execution completeness cannot certify
  the end of a syllable. Acoustic completeness remains UNKNOWN for every text-only
  decision; the inherited complete actual-word review flag is reported separately
- Q/T retain inaudible/null actual text, never blank-CTC or inferred silence truth.
  Missing, empty, failed and one-sided outputs stay unresolved
- Z3 preserves actual 小挖 separately from original plan 小窝; unchanged actual-label
  policy reports OOV 挖 and refuses a CTC label. No homophone or repetition repair
- A/B/Z2/Z4 protect genuine lexical support. Z5/Z6 machine disagreement does not
  erase correct human actual text. ASR quality flags remain separate from lexical
  equality and block the optional complete-label check
- The optional original_intended mode adds plan consistency and requires complete
  human review. human_actual keeps the previous actual-word coverage authority;
  omitting the option keeps the prior API behavior. Scope reports distinguish the
  extra check. Neither mode grants training, product or fresh-validation authority

## Source and scope checks

All 12 pre-existing quality_gates definitions remain byte-identical, including
normalized_actual, human_truth, coverage and identity gates. The new function is
label_preparation. Vendored actual-label/text/supervision helpers, the prior public
fixture, SOURCE_PINS, existing report adapter and workflow are unchanged. All Python
files parse; every test file is explicitly covered by the unchanged workflow.

The new fixture has 17 selected, previously exposed synthetic text cases from the
completed fixed30 saved comparison. It contains only aliases, intended/actual text,
existing review state, two saved text/status/quality-flag observations and use restrictions.
It contains no waveform, private source records, recording hashes, private paths,
chat identifiers, storage locators or raw execution receipts. Six original plans
were independently joined to the original generation configuration and recording
identities before making this projection. The configuration SHA256 is recorded in
the fixture and bound to its public source in SOURCE_PINS.json. Original comparison
null plan fields and all historical evidence stay unchanged.

These are deterministic code regressions, not a new ASR experiment or independent
accuracy evaluation. The selected fixture is not the full30 statistics table.
No original-plan counts are inferred from it. No threshold/confidence tuning,
new normalization, acoustic measurement, training, synthesis, model/ASR call,
download or dependency installation occurred. Caller-supplied metadata does not
prove ASR family independence, human review or recording identity; source binding
is a separate preparation responsibility.

Only the standalone research API and JSON CLI are integrated here. Production
builders, repository C/CMake/CTest, physical-board execution and GitHub CI were not
run for this candidate. No external publication or rollout is included.
