# A20 research numerical semantics

This is a public technical summary of the adopted replacement-validation
contract. It is not the historical strict official-FP32 contract, and it does
not erase that contract's recorded failures.

## Fixed input and state

Signed PCM16 mono 16 kHz, unscaled; 400-sample frames, 160-sample hop, 80 mel
channels, snip edges, zero dither and the original 2^-23 energy clamp. Ingress
uses 4,800-sample canonical feeds, an 800-sample initial gate, actual short
tails, reset per stream and no EOF padding/flush. Splice is true +/-2 context
with skip 3 and the pinned tail/index behavior. Six competing classes are
blank, 你, 好, 小, 窝, 屋.

Preprocessing uses binary32 round-to-nearest-even at every operation boundary:
exact integer DC sum, division by 400, subtraction, preemphasis multiply and
subtract using stored binary32 0.97, then stored Hamming multiplication. No
contraction/FMA or flush-to-zero is allowed. The native frontend rejects
non-RNE and FTZ/DAZ modes rather than changing the process environment.

## Independent mathematical authority

The fixed binary32 window is lifted exactly into an outward Decimal80 DFT,
followed by power, positive mel accumulation and clamped logarithm using the
fixed binary32 coefficients. Bounds are fixed independently of candidate
results: absolute window <=65,536, root error 1e-78, complex-component error
1e-60, power radius <1.1e-52, mel radius <5.6e-50 and log radius <5e-43.

The fbank interface must have one uniquely certified binary32 RNE value across
the complete outward interval. Ambiguous rounding fails. Exact splice follows;
CMVN is two separate binary32-RNE subtraction/multiplication operations.
The mathematical FSMN uses exact binary32 weights and these canonical CMVN
values with unrounded high-precision state/cache. Its final uncertainty ceiling
is 1e-20. Float64 diagnostic conversions cannot replace the unrounded authority.

Stable six-class softmax is evaluated from unrounded mathematical logits with
outward intervals. Only the decoder interface is uniquely rounded to binary32.
Decoder path/score arithmetic is binary64: beams 3/20, prune strictly >0.05,
threshold 0, duration 5..250, interval 50, frame scale 3, pinned tie semantics,
shared-node aliasing, rejected-hit score retention, inclusive final-start suffix
search and full model/cache consumption before chunk-decoder first-hit stop.

## Hard comparisons

- Fbank and splice absolute error <=1e-3 against canonical binary32 interfaces
- CMVN absolute error <=2e-4 against its canonical binary32 interface
- Raw logits: worst endpoint distance to the unrounded interval
  <=1e-4 + 1e-5 times the minimum absolute reference interval value
- Probabilities: worst endpoint absolute distance <=1e-5; values within [0,1]
  and row-sum absolute error <=1e-5
- Required decoder non-score fields exact; score absolute error <=1e-5
- Shapes, finite values, cache/index/reset/history, ReLU, feed/tail/EOF,
  consumption and clock behavior exact
- Whole/ragged native tensors, final cache and event records bit-exact

Distance/tolerance calculations are outward/downward conservative. There is no
uncertainty addition to a tolerance. Existing same-input analytical accumulation
bounds remain local hard checks but do not prove propagated final accuracy.
Original saved official-FP32 outputs remain an independent compatibility
record. Missing original tensors are not computed by substituting the new
mathematical authority. Exposed actual-clip behavior is also separately checked
against original available records, including score <=1e-5.

## Native arithmetic

Weights, CMVN, intermediate storage, cache and interfaces are binary32.
Affine/FSMN reductions use sequential binary64 accumulation with a binary32
output cast. FFT retains the original radix-2/bit-reversal schedule using two
512-double work arrays and independently certified binary64 roots; complex
output is explicitly rounded to binary32 before unchanged power/mel/log.

No quantization/training is part of this port. A previous lost validation run
remains missing. Replacement fixtures are numerical validation, and previously
exposed clips are regression; neither is fresh acoustic or FAR/FRR qualification.
