# Prepared Melo six-phrase ASR reuse

This is a disabled, separately named adaptation of the successful Qwen6 CPU
recognizer runner. No Qwen TTS model is invoked. The scientific ASR runtime,
124 package-input lock, 13 model assets, source pins, inference controls and
raw-before-label-freeze procedure are unchanged.

Changes are limited to the new experiment/release/workflow identity, binding
the new six-clip handoff, and tighter 10-second/2-MiB input limits. Legacy
schema names are compatibility labels, not assertions that this is the old
Qwen experiment. The private plan body is absent from the ASR input contract.

The release is currently approved=false with absent generation hashes, so
publication or execution now would fail closed. Exactly one first branch push
is proposed only after explicit new publication approval, successful private
generation, byte validation, and a fully bound release. There is no retry.

Proposed public destination: jiying2007/kws-pipeline, new branch
research/melo6-source-screen-v1, standard free ubuntu-24.04 runner. One ASR job
has a 50-minute limit; controlled package/model transfer is 3,340,966,447 bytes,
with a 4-GiB cap. Official local CPU Torch access previously returned403;
private-local ASR is not represented as ready and no GPU-wheel workaround is
allowed. Runtime/input verification failure stops before decoding.

SenseVoiceSmall and Qwen3-ASR-0.6B each decode the six opaque clips once,
12 calls maximum. Recognizers receive audio only, no intended text or tone
arrays. Raw status, failures and resource records are frozen before comparison.
ASR agreement is weak machine evidence. Human/acoustic truth remains unknown.
No KWS forward pass, threshold search, training, or speaker split is included.
