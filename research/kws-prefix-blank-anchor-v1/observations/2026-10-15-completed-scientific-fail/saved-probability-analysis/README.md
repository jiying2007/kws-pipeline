# Saved-probability analysis: missing initial ni

These three files (`analysis.py`, `summary.json`, and this README) are a post-hoc
read-only description of the completed prefix-blank-anchor experiment. They do not
change its prespecified endpoint, decoder, tolerance, or scientific FAIL outcome.
The script imports only Python standard-library modules and never runs a model,
optimizer, training, audio processing, or a new forward pass.

## Reproduce

Use the original result directory and independent acceptance hash-manifest.json:

    python -B analysis.py --result-dir RESULT --hash-manifest hash-manifest.json > reproduced.json

The manifest's exact SHA256 is hardcoded in analysis.py. Every consumed observation
is hash checked against that manifest before parsing. summary.json is deterministic;
compare its bytes with reproduced.json. No private absolute paths appear in output.

Token IDs: 0 blank, 1 ni, 2 hao, 3 xiao, 4 wo. Frame j uses the analysis window
[20*j, 20*j+25] ms. These are model frame locations, not validated phonetic boundaries.
The output includes all raw frame argmax IDs and their lossless run-length summaries.

## Main findings

Frozen acceptance: evidence PASS; scientific FAIL. C reproduces all 300 historical
losses/gradient norms and five observations exactly. The new experiment used exactly
600 reservations and 600 successful optimizer steps; no retry is authorized.

T300 KW1 decodes [2,3,4], missing ni; KW2 decodes [3,4,3,4] correctly. Both fixed
prefixes (23/10 frames) have zero nonblank argmax frames. At T200 the two greedy
sequences were already the same as T300. Intermediate results are descriptive,
not alternative endpoints.

KW1 ni never wins frame argmax. Its largest posterior is 0.2890984429 at frame47
(window940–965ms), where hao=0.4441478024 and blank=0.2450604808. The raw argmax
contains no ni to be removed by CTC collapse.

Independent float64 log-domain CTC sums over alignments give:
- Full KW1 [1,2,3,4]: NLL1.4231285522, probability0.2409589837
- Missing-ni [2,3,4]: NLL1.4013491862, probability0.2462644831
- Probability ratio missing/full=1.0220182676 (about2.20% higher)

This is a comparison of two specified sequences, not an exhaustive decoder search.
The missing-ni sequence is slightly favored even after summing its CTC alignments;
therefore the observed miss cannot be attributed solely to greedy versus sequence
scoring. It also does not prove the global best sequence or that a beam decoder
would repair the output. No decoder change is counted as experimental success.

The correct-keyword versus other-keyword NLL margin improves from C300 to T300:
KW1: +0.7065433 to +7.2489905; KW2: -0.6863868 to +94.5608698 (positive favors the
correct keyword). These margins compare only the two known keywords and do not
establish open-set recognition or rejection quality.

T300 KW1 raw runs: blank frames0–46; hao47–48; xiao49–50; wo51–63; blank64–70.
T300 KW2: blank0–24; xiao25–26; wo27–40; blank41–48; xiao49–50; wo51–70; blank71–82.

An additional analytical path restriction finds79.0% of full-KW1 alignment mass
has all first23 frames blank. This is conditioned on the full target sequence,
not an unconditional accuracy or error probability. The complementary21.0% shows
that blank greedy prefixes do not prove zero nonblank CTC alignment mass there.
This restriction only scores saved probabilities; it does not modify actual inference.

## Interpretation and next decision

The prefix goal is achieved on these two exposed training examples, while joint
success remains unmet. KW1 ni is present as a subdominant posterior near the first
emitted hao and remains in substantial full-target alignment mass. The evidence
supports a remaining emission/sequence-probability competition, not an identified
causal explanation of why the model failed to learn that token strongly enough.

Preserve this negative endpoint and close this campaign. Any further training,
seed change, objective change, decoding experiment, or extension needs a separate
plan and approval. These results support no real recall, FRR, FAR/FAh, generalization,
human-gold, or product-readiness claim.
