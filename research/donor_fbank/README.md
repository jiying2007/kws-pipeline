# Isolated donor80 Kaldi/Hamming frontend prototype

[简体中文](README.zh-CN.md)

Local research scope:80-bin raw logfbank streaming plus a fixed400-dimensional affine CMVN component. This directory does not modify the existing product32-dimensional frontend, model validator, model registry, runtime or release gates. No FSMN, decoder, model export, C context-splice or skip3 implementation is included. No target-board performance or end-to-end donor compatibility claim is made.

## Current status: v2 final-feature PASS; original v1 strict gate FAIL

The first frozen v1 layered gate is retained unchanged. After explicit authorization and independent pre-run review, a separate v2 final-feature contract passed. This does not rewrite v1.39 synthetic cases/159 frames give:

- DC removal, preemphasis, Hamming-windowed samples and FFT input exactly equal to the official float32 oracle
- Maximum logfbank absolute difference0.0001926422 against frozen0.001 limit
- Test-only true-context/skip3 plus C400-CMVN maximum difference0.00003004075 against frozen0.0002 limit
- C400-affine applied to golden400 inputs is bit-identical in these cases
- Only full-scale alternating PCM violates the intermediate power/mel gates:20 power scalars,10 mel scalars. Worst tolerance ratios5.2365 and1.7793. Thus overall v1 conformance remains failed, even though final-feature gates pass
- Four frame/stream tests and seven CMVN tests pass, including partition parity, boundary frame counts, no EOF padding, reset, atomic invalid-input/overflow rejection and in-place normalization

The initial product-derived rotating-twiddle FFT had additional multitone/narrow-tone intermediate failures. An isolated direct-root table removed accumulated float rotation drift; both versions/results are preserved outside the source checkout. Thresholds were never widened. Independently, before seeing C results, the oracle author found official float32-versus-float64 power/mel discrepancies that exceed the same v1 allowance on extreme alternating PCM. This is diagnostic evidence, not an excuse to relabel v1 as passing. The separate v2 contract below received explicit, versioned justification and pre-run review after disclosure of existing outputs.

## Fixed numerical contract

Signed PCM16 is converted to float32 without division by32768. Frames are400 samples at160-sample hop, phase0, snip_edges true, dither0. Each frame subtracts its mean; preemphasis0.97 replicates the first sample; a nonperiodic Hamming window follows. Pad to512, compute unnormalized FFT power, apply80 continuous-mel triangular filters from20Hz to8kHz with zero Nyquist weight, then ln(max(power,2^-23)). No energy coefficient, per-frame mel centering, PCEN or log1p.

Hamming/mel coefficients are exact float32 values from pinned torchaudio Kaldi source, stored sparsely as501 nonzero mel weights. The radix2 skeleton is adapted locally from this Apache2 repository's existing FFT; it does not link or alter product math. Generated direct float32 roots replace recurrence. Source references and hashes are in `spec.json`; upstream torchaudio's BSD2 license is retained (trailing whitespace normalized locally; exact original also inside the SHA-pinned fixture). This target is the provider-declared Hamming route reproduced by pinned torchaudio source, not proof of the donor's unobserved original training configuration. WeKws's default Povey route differs.

The caller-owned state is initialized/reset once, accepts0..16000samples per call, emits the first frame at400 samples and subsequent frames every160. Incomplete EOF frames are discarded; no reflected or zero padding. Callback rows are borrowed for the callback only. Single-threaded/non-reentrant use is required. No heap allocation or explicit file I/O exists in this C source; observed non-CRT undefined symbols are math/memory functions. This source property does not establish target realtime deadlines, OS paging behavior or complete SDK readiness.

## CMVN and context scope

`donor_cmvn400` expects an already correctly concatenated400-vector, computes `(x-mean)*inverse_std`, and does not infer stats or context. It rejects nonfinite inputs, nonpositive inverse_std, nonfinite computed outputs and invalid overlap before writing. Exact input/output in-place is supported; partial overlap or output overlapping statistics is rejected.

The pinned donor's five80-element CMVN slices are bit-identical, machine-verified in the golden package. Nevertheless, the C API remains400-dimensional and implements no80-bin shortcut. Golden chain order is chronological[-2,-1,0,+1,+2], left-edge replication only, two complete right-context frames, then centers0,3,6... and400-CMVN. Context/skip are test-only Python operations. The upstream demo additionally waits for800 buffered PCM samples and uses300ms calls; a semantic complete context at720 samples does not imply that demo emitted then. Frame availability and model event latency are distinct.

## Reproduce locally

```
python3 research/donor_fbank/build.py --output /absolute/validation
python3 research/donor_fbank/tests/test_stream_contract.py /absolute/validation/libdonor_fbank.so
python3 research/donor_fbank/tests/test_cmvn_contract.py /absolute/validation/libdonor_fbank.so
python3 research/donor_fbank/tests/compare_goldens.py \
  --library /absolute/validation/libdonor_fbank.so \
  --goldens /absolute/kws-donor-frontend-goldens \
  --output /absolute/validation/layer-result.json
```

Build needs installed GCC and C11/libm, no extra native runtime. First two tests use Python standard library; layered comparison needs NumPy. Golden regeneration additionally needs the already-installed Torch CPU and exact source files documented in the golden package. No installation/download happens in these tools. The last command intentionally exits nonzero for current v1 failures. Do not suppress that failure in an integration or release gate.

Golden package archive SHA256 `ef729d8a4e0170bddeeec67d2095f38f53e67b363fc9d5a56ce8af2fea8e6f54`; tolerance SHA256 `b1f898b183916b0bd1cef789cbb75be02e7e5052256b4a74f00f411dc5946ab1`. The package contains only deterministic synthetic input/reference arrays and source provenance, no private recordings or acoustic-model weight file. It includes39 cases,480 layer arrays and34 selected true-context rows;1059 files regenerate identically in the recorded environment.

## Explicit v2 acceptance and chronology

`v2-contract.json` is a separately named final-feature contract, designed after the v1 results were known. Before its first run, an independent numerical review approved the frozen proposal SHA256 `21529eaca9bd49d0332a78700e7feab3a3964d1e5a1daa675a3141bfdf5226ce`; only status changed to produce approved SHA256 `3bf49445ae77bef6397673379c892885d7d8a406d66b34bf76831a1b1b919019`. The original final log/CMVN, coefficient, frame, finite-value and zero-input hard gates remain unchanged. Original per-bin power/mel comparisons remain reported diagnostics, including their failures.

Independent FFT hard gates add191 checks:30 frozen zero/DC/Nyquist/impulse/sinusoid/noise inputs, all159 original golden FFTwindow inputs, and exact binary scaling by2 and0.5. The reference is a direct DFT using separately generated80-digit Decimal roots, modulo512 indexing and double faithful summation. Operation-count bounds use gamma(n)=n·2^-24/(1−n·2^-24), amplitude budget76, stored-power rounding budget4, Parseval budget148 with double sums+1e-12 bookkeeping, and impulse budget148. They follow nine-stage radix2 rounding analysis, not observed C error multiples. Assumptions include finite normal arithmetic, independently rounded roots and no fast-math reassociation; they are engineering bounds rather than a universal formal proof.

The formal v2 run passed all191 independent FFT checks. Maximum bin error/bound ratio0.0562321 and maximum Parseval relative residual1.18131e-7 were recorded. The same run still records `v1_overall_passed:false`, all original20power/10mel scalar exceedances, and no unexpected final-feature hard failure. A passing final-feature contract is not pointwise intermediate equivalence.

```
python3 research/donor_fbank/tests/accept_final_features.py \
  --library /absolute/validation/libdonor_fbank.so \
  --goldens /absolute/kws-donor-frontend-goldens \
  --contract research/donor_fbank/v2-contract.json \
  --output-dir /absolute/new-v2-evidence-directory
```

The output directory must be new; retained evidence is not overwritten. This v2 scope is C11fbank80 and an independent400affine component. The chronological splice/skip chain remains test-only Python. Still required before a full donor path: implement and verify Ccontext/skip, then separately validate model/cache/logits and event behavior. The current caller-owned state is4,928B and table payload6,816B, excluding code/stack/host memory. These are component sizes, not target memory/performance qualification. This prototype is outside the default build and has not been published.

## Test fixture and dependency separation

The versioned `fixtures/parts/` files losslessly represent the555,581B archive of deterministic synthetic inputs, intermediate/final golden arrays, source provenance and licenses. It contains no private recording and no full acoustic-model weight file. `tests/unpack_goldens.py` validates12 ordered parts (each≤48KiB), exact directory membership, per-part path/size/SHA and the original complete archive SHA before assembling in memory and rejecting unsafe archive members. The repository contains no duplicate whole archive. Ten negative/positive tests cover part tampering, omission, extras, reordering, traversal, size drift, symlinks, rebound part hashes and output overwrite. CMVN statistics and their model-specific Apache2 declaration are retained for reproducibility, separate from any product model.

CI consumes only the fixture and a CPython3.12 Linux x86_64 NumPy2.2.6 wheel from its exact official PyPI URL/SHA in `test-requirements.txt`; it uses no implicit dependency resolution or Torch/ONNX download. NumPy is a test dependency, not a deployment dependency. Both existing local NumPy2.5.2 and the pinned2.2.6 harness runs passed the same reviewed v2 contract. Regenerating the oracle from source is a separate development step requiring the documented pinned source and Torch CPU environment; ordinary CI reads and hash-verifies its reproduced vectors.

The dedicated workflow regenerates source coefficient tables, runs all lifecycle/CMVN/FFT/final-feature gates, and retains machine-readable v1/v2 evidence. Root product build/tests are unaffected. Current publication is a research draft, not product promotion.

An isolated C smoke test additionally covers full16000-sample irregular ingress, oversized-input atomic rejection, reset/EOF, FFT/trace buffers and CMVN overflow. Local ASan/UBSan passed with LeakSanitizer disabled because the local ptrace environment does not support it; dedicated CI runs the sanitizer binary with its normal configuration. No leak-clean claim is made from the local run.

A32 long-run timeline checks directly seed the existing caller-owned state, without adding a test-only API. Samples and frame indices cross2^31 and2^32 in bounded tests; callbacks retain full64-bit index/end values. UINT64 overflow is rejected before any state mutation. size_t is used only for bounded single-call/buffer indices, never as the sustained timeline or its endpoint multiplication.
