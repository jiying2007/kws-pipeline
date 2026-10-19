# Optional profiling and order-preserving affine blocking

Both switches are disabled by default. This is a research runtime, not a production SDK/model promotion. The historical strict C/Torch atol=rtol=1e-5 failure remains unchanged. The portable-contract document is a proposal, not a replacement acceptance rule.

## Build switches

- KWS_B_PROFILE: compile/link b_profile.c and enable five exclusive scopes: frontend, FSMN body, final 2599 head, full softmax and decoder. Default exports stay at 36; profiling adds b_profile_reset, b_profile_size and b_profile_get (39). Snapshot ABI is versioned and size-checked. Statistics are thread-local, not per handle; reset/get on the inference worker thread. Masked traps are required. Timer uses CLOCK_MONOTONIC, preserves errno and the floating-point environment, saturates counters and exposes timer errors. If environment restoration fails on a platform, the measurement is invalid; no platform-independent recovery promise is made.
- KWS_AFFINE_BLOCK2: process two output rows together, retaining each output's four ascending-input accumulation chains and exactly the original final addition order. Row-major weights are unchanged; no packing, quantization, class pruning, softmax change or new weight memory. Odd output and partial-input tails are handled. Logical accumulator scratch increases by 16 bytes; actual registers/spills are compiler/target dependent.
- Both switches can be used together or independently. Neither changes the registered payload, default model identity, full-class normalization or decoder top-3 semantics.

Strict flags remain -fno-fast-math -ffp-contract=off -fexcess-precision=standard with IEEE binary32, RNE, gradual underflow and masked traps. Independent outputs are interleaved differently under block2, so unmasked trap timing is outside the equivalence promise. Finite gates and failure behavior remain in place. No new tolerance has been introduced.

## Existing host test entry

Run python3 -S -B tests/test_research_fsmn_stream.py --execute. It retains the original five-compilation adapter/capture/C++ test and adds the public opt-in matrix via research/fsmn2599-stream-c/check_profile.py. No new workflow is needed.

The matrix performs 19 bounded compilations: four complete B DSOs, four C++ link-only smokes, four invalid-argument checks, four mocked adapter groups, plus profile unit, timer overhead and affine toy fixtures. It executes eleven tiny test programs, never a model or audio example. Three profiler fixtures test API/fenv behavior, fake-clock failure/backwards-time/saturation and 1000 empty timer scopes. The affine fixture extracts the optimized function body directly from production source and compares it against the original four-chain reference on 648 synthetic cases. Random, signed-zero, subnormal, overflow and tail shapes are covered, including bias/no-bias. Toy times are diagnostic, not model benchmarks.

Compilation has a shared 60-second CPU/wall ceiling; each tiny test has 10 seconds and each symbol check 5 seconds, with the existing subprocess address-space and output guards. C++ smokes are not executed. Exact source-binding.json includes helpers/fixtures and is verified before and after. Default/profile-only host DSO bytes match the prior measured runtime in this environment; artifact byte equality is not a cross-toolchain guarantee.

Default-off preprocessing against main 0f01c477 was identical for FSMN and differed only in layout whitespace around conditional wrapper signatures for frontend/softmax/decoder. The complete default host DSO was byte-identical (SHA d0f698b2ccfe7c93b8013386b34e63a738ca0e27f97ce9a8e7071382a8ab92e6).

## Existing ARM compile/link entry

Run python3 -S -B tests/run_research_fsmn_stream_arm_build.py --execute. The existing CI step now covers four modes: default, profile, block2 and both. Each builds the entire frontend/network/full-softmax/decoder DSO and links the C++ smoke, then requires exactly 36/39/36/39 exports. Profile-enabled C++ additionally references b_profile_size. Eight compilations share 60-second CPU/wall limits. Target remains cortex-a32/neon-vfpv4/hard with strict FP flags.

Missing cross tools are an explicit failure, never skip/pass. No target executable or model is run. During local preparation the ARM compiler, C++ compiler and nm were absent; ARM compilation is therefore NOT RUN locally. Passing this future CI matrix will still not establish ARM numerical equivalence, runtime performance or SSC305 deployment readiness. Those need separately controlled target tests before choosing a new default.

## Anonymous host evidence

host-optin-results-20261019.json retains an instrumentation pair, an initial block2 pair and six balanced-order repeated host pairs. All posterior-row byte chains and complete event fields/score-grid bits matched on one previously exposed development stream. Repeating the same stream adds zero independent acoustic hours. No audio, learned weights, corpus IDs, PCM/posterior hashes or transcript is published.

The six fixed pairs used AB/BA/AB/BA/AB/BA, each process starting fresh. Every run is retained. Off/on wall medians were approximately 0.989869/0.358001 seconds; the median paired on/off ratio was 0.366176 (range 0.352798–0.370843). FSMN/body and head median paired scope ratios were 0.175892/0.165348; other scopes stayed near one. This supports a repeatable improvement on this particular cloud host and stream. It is not ARM/cross-host performance, detection-quality qualification or proof over all legal inputs. Full arrays were not retained; cryptographic rolling byte chains cover all posterior rows. The optimization stays opt-in.
