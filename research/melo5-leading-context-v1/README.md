# Fixed leading-context saved diagnostic

This appendix verifies a completed, exposed five-clip A20 diagnostic. The one
predeclared condition prepended exactly 24,000 digital-zero samples to every
complete original 16 kHz PCM stream and retained the same 4,800-sample postroll.
M6 is excluded. All numeric observations, human word labels, logs and token
support results live in `jiying2007/kws-data`, under
`research/2026-10-06-melo5-leading-context`.

The prefix was selected from architecture before execution: the model's largest
causal lag is 44 selected rows; four complete feeds produce 39 rows and five
produce 49. Five feeds are the smallest whole-feed prefix covering that lag.
This modifies frontend, model and decoder history together. It is not a
model-cache-only intervention, an algorithm change, or a threshold/duration sweep.

## Public dependencies and exact provenance

`metadata/bindings.json` binds the original acquisition hashes, the independently
audited result, public projections, and exact source reconstructions. The source
base is pipeline commit `ea1c554a82c89797023365243a827883b94e0254` (PR #484).
The actual scorer is separately retrieved from PR #485 commit
`6f2461ff11cddf0a1264f5e2a04282c40b5dc4ce`; that scorer is not claimed to exist
in the older source base. Original and tail controls are read at data commit
`9307956e16ed93dc6c858611f3c35ae17768a011`, without changing them.

The original public audio archive is pinned by full commit, byte count and SHA256.
The verifier recreates only WAV bytes in temporary storage: preserve the original
44-byte header and all PCM; insert 48,000 zero bytes before PCM and append 9,600
zero bytes after it; update the RIFF and data lengths at byte offsets 4 and 40.
It verifies full WAV, PCM and zero-region hashes and the unchanged middle PCM.
No WAVs are reuploaded or played, no model/library is loaded, and no native
collector, frontend or decoder is run.

The original trace validator and collector source are reconstructed byte for
byte from the earlier public core using checked text deltas. The collector is
only checked as source, never compiled or executed. The stateless validator
reproduces the historical observation bytes; only the declared public-to-original
raw SHA is normalized for that comparison. Public raw logs remove only the PID
field. Resource projections remove absolute host clocks and release-file records.
Measurements, failures, events and scientific limitations remain available.

## Reproduce saved evidence

Use Python 3.12 and the standard library. Run from this directory:

```sh
python -B prepare_inputs.py --output-dir build/fixtures
python -B verify_saved.py --data-dir build/fixtures/data --dependencies build/fixtures/dependencies
MELO5_LEADING_FIXTURES="$PWD/build/fixtures" python -B -m unittest discover -s tests -v
```

Preparation is the only network step. It fetches exact immutable public paths,
checks sizes and SHA256 before writing, and does not use credentials or redirects.
For disconnected use, `--source-dir DIR` accepts the same predownloaded relative
file layout. Tests and verification require these prepared inputs and never
download them implicitly. The dedicated normal CI workflow performs preparation
explicitly before the saved tests. No research acquisition is part of CI.

Each of the three conditions is passed independently through the unchanged
PR #485 clip-presence CLI. Its `eval-context-v1` validates appended-context,
reset/EOF declarations and input bindings; it does not define leading context or
attest execution. Leading context is verified separately against reconstructed
WAV bytes. Every saved event is counted, including any that would occur during
the prefix. Source-relative availability subtracts 24,000 from new input sample
availability only; it is not acoustic word-end latency. Model/decoder coordinates
remain in their original stream coordinate system.

The M2 helper recomputes row-local FP32 probabilities and source-equivalent top-3
eligibility from saved logits and checks them against 80-digit arithmetic. It
does not replay CTC or beams, assign tokens to spoken occurrences, or prove a
unique acoustic cause. Read the data report for observations and limitations.

All compact token-support fields must reproduce exactly except
`numeric_check.max_binary64_absolute_difference`. This one scalar measures
host-libm binary64 roundoff and can differ across platforms. The verifier reports
both local and recorded values without replacing either or claiming that scalar
was reproduced. Every FP32 result must still agree with the 80-digit reference;
recorded probabilities, eligibility, ordering and event fields remain exact.

## Scope and rights

These are exposed clips from one voice and synthetic zero-context modifications,
not FRR/FAR estimates, independent-speaker validation, continuous-field testing,
or embedded qualification. The original machine text gate and later human
adjudication remain separate. Audio/model rights and training admission remain
those of the original sources; this appendix grants no additional clearance.
No default model, production runtime, deployment or promotion is changed.
