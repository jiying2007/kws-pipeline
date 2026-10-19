# Opt-in B operator profile: one development-stream pair

Public anonymous result; no audio, weights, clip identities, event times or private input hashes are included.

One 35.62-second exposed development stream was run once with profiling off and once with profiling on. The input byte chains, all 1,186 posterior-row float32 byte chains, and eight complete event structures matched. Actual event recording identity was validated before normalizing the process-specific identity. Event score and grid binary64 bits also matched. Dense posterior arrays were not retained; equality is evidenced by the cryptographic rolling byte chain. No new independent test hours were added.

## Exclusive operator timing

- frontend: 80.326 ms, 9.190%
- fsmn_body: 387.630 ms, 44.350%
- head2599: 349.282 ms, 39.963%
- softmax2599: 19.963 ms, 2.284%
- decoder: 36.820 ms, 4.213%

The denominator is only the sum of five non-overlapping scopes, not total process time. No failed scopes, timer errors or counter saturation occurred. This single noisy cloud x86_64 pair is useful for locating computational cost, not estimating a reliable speedup, timer overhead, ARM/SSC305 consumption or release quality. Historical C/Torch strict 1e-5 failure remains unchanged. Keep the full 2599-class softmax; these results do not authorize column pruning.

## Minimal repository integration

The reviewed opt-in overlay can be integrated independently after the runtime-hardening base lands. Apply only the seven paths declared in PROFILE-SOURCE-FREEZE.json: b_profile.c/h, guarded frontend/FSMN/softmax/decoder instrumentation and updated source-binding.json. With KWS_B_PROFILE undefined, default DSO bytes and its 36 exports remain unchanged in the verified host build. Opt-in adds three profiling exports and five exclusive counters. Counters are thread-local, not per-handle.

The preparation check_profile.py assumes a candidate/ staging tree, so it is not a ready-to-copy repository CI entry point. Move its three tiny mock/unit/overhead C fixtures beside the research module, adapt its root/manifest lookup to repository paths, and invoke that bounded compile/mock check through the existing research stream test entry point. Keep model/audio execution out of CI. Preserve the 60-second aggregate compilation budget, 10-second per tiny-program limit, 5-second symbol-inspection limit and strict floating-point flags. Do not add another workflow. The adapted repository paths and updated source-binding must receive a new source freeze and normal independent review; do not rewrite the historical staging freeze.

Public result.json is suitable for the next evidence PR after PR #613 lands. Private paired manifests, stdout and ledger receipts remain separate retained evidence. Target ARM compile and numeric checks remain unperformed.
