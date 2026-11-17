# R1 consecutive-blank expiry research candidate

This directory archives a rejected opt-in proposal policy based on the current
R1 reference64 SDK. **Canonical `deploy/*` and the default SDK are unchanged.**
The independently named `candidate_sdk/` contains the candidate C/header and
the exact original mock fixtures/runner needed to reproduce its behavior.
It changes no acoustic weights, vocabulary, frontend, thresholds, source roles
or label admission. Within the archived candidate, the original constructor
and caller-allocated `r1_proposal_decoder` keep their API/ABI, and the original
proposal step body is unchanged. Its baseline constructor selects policy 0.

## Frozen hypothesis and result

Before reading the restored R1 frame-gap statistics, `frozen_candidate.json`
declared exactly one candidate: expire the retained four-token prefix on the
**twentieth consecutive blank output row**, with the original maximum span of
150 frames and same-keyword cooldown of 50 frames. The declaration SHA256 is
`493625580bc5bcb4f3c320aff9fe95898d8f8a23422e84f135df182d55b72504`.
No threshold/gap sweep or model execution is part of this comparison.

**Gap20 fails the no-positive-loss development check and must not replace the
default.** On the fixed 34 exposed weak-development records, K1 retained 10/10
positive-row proposals, while K2 declined from 10/10 to 9/10. Both words retained
0/24 weak-negative proposal-positive rows. On the 96 known training records,
K1 retained 27/27 while K2 declined from 28/28 to 24/28. Four unknown records
remain unscored. Across all 134 records, baseline/candidate events were 88/82,
and 128/134 full event lists matched. Fewer events do not establish fewer false
alarms: this candidate loses known weak-positive proposals.

The comparison authenticated both restored numeric archives and all 134 saved
numeric objects, verified their saved feature/logit hashes and original greedy
sequences, and postprocessed the 12,534 existing argmax frames once per policy.
It made zero acoustic-model, audio, TTS, ASR or optimizer calls. Per-record
restarts were preserved. This is not continuous-audio replay, natural FAR,
independent FRR, real-speaker qualification, or acoustic word-end latency.
The original 130 known records contribute 86 baseline events; the four unknown
records contribute two more. Thus the 88-event denominator is consistent with
the older 130-record compact-head comparison's 86 events. No former result is
rewritten by this analysis.

## API and exact semantics

```c
r1_sdk *sdk = r1_sdk_create_with_blank_gap(
    weights, weights_bytes, tables, tables_bytes,
    20, on_proposal, user); /* rejected quality candidate, research only */
```

The API above exists only in the archived candidate, not canonical `deploy/`.
`max_blank_gap=0` disables expiry. Values 1..150 configure an independent
research policy; values above 150 are rejected before creating the core. Only
the declared value 20 was assessed here. Configuration availability is not
evidence that another value was evaluated or is suitable for deployment.

`r1_proposal_decoder_v2`, `r1_proposal_init_v2` and `r1_proposal_step_v2` expose
the same pure-token operation for diagnostics. A blank is class ID 0, not an
acoustically verified silence interval. Repeated nonblank frames terminate a
blank run even when CTC collapse emits no new token. The count saturates at the
configured limit. Expiry clears the retained prefix IDs and frame positions;
it never resets the event count, frame clock, per-word cooldown or acoustic
state. Explicit SDK gap/reset clears both original and v2 decoder state while
retaining the selected policy. Finish adds no frames.

The archived candidate's opaque handle contains the small v2 bookkeeping addition. Query
`r1_sdk_state_bytes()` for the built implementation; the historical 249,384-byte
observation is not a new measurement of this revision. Feed/gap/finish still
perform no allocations or file I/O. Callback errors and numeric rejection keep
the existing fail-stop behavior. The old `r1_proposal` event ABI is unchanged;
record `r1_sdk_blank_gap_limit()` in the caller's run metadata to identify the
chosen policy.

## Saved-output analysis

The standard-library reader parses only the fixed saved tensor ZIP schema.
It interprets an explicit opcode allowlist as inert storage/tensor metadata;
it never calls `pickle.load`, resolves a serialized Python callable, imports
Torch, or loads a neural model. It verifies byte order, tensor layout,
nonfinite values, payload length and the original tensor-content hashes.

```sh
python3 -B research/r1_stream_c/proposal_gap_v1/analyze_saved.py \
  --root /owner/V3-R1-DEV-20261107 \
  --output /new/private-gap20-analysis
```

The input is the existing R1 research record package, recovery index and two
`fsmn-tail-numeric` ZIPs. The tool pins their existing published identities and
refuses overwrite, symlink paths and output within the source/input tree.
Outputs include private per-record source, label and token information and
must remain in the owner's private delivery. Public fixtures contain only
fictional data. No private waveform, label, source identifier or numeric array
is included in this source directory.

## Validation

```sh
python3 -B research/r1_stream_c/proposal_gap_v1/test_proposal.py
python3 -B research/r1_stream_c/proposal_gap_v1/test_saved_numeric.py
python3 -B research/r1_stream_c/proposal_gap_v1/run_contracts.py --cc gcc --sanitize
python3 -B research/r1_stream_c/proposal_gap_v1/run_contracts.py --cc clang
```

The C runner executes the unchanged 21 legacy fictional-token/mock-core groups
plus 15 focused new groups. These cover 19/20/21 blank boundaries, sustained
nonblank tokens, separated blank runs, span/cooldown preservation, disabled
policy parity, invalid arguments, the actual opt-in SDK callback route, explicit
gap policy retention, hot-path allocation checks, callback failure/re-entry and
nonfinite rejection. `--compile-only` compiles without executing the fixture.
Hosted fixture results verify software behavior, not acoustic quality.

The failed candidate remains a reproducible research option. A subsequent
quality experiment needs a separately stated hypothesis and paired positive
and continuous-context evidence; this result does not authorize tuning gap
values against the same development records until a passing value appears.
