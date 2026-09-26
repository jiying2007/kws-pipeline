# Reproducibility levels

`kws-pipeline` distinguishes three claims:

1. **Traceable training**: the checkpoint records the exact training environment and corpus identity actually used.
2. **Rebuildable training**: the base container image and all Python dependencies are digest/hash pinned so the environment can be reconstructed from retained inputs.
3. **Bit-reproducible SDK**: two independent release builds produce byte-identical installed SDK trees.

Deterministic tar metadata alone satisfies none of the compiled-binary claims. Release documentation must use the strongest claim actually proved by the current gates.


## CPU training reproducibility boundary

The bounded RNN micro-training fixture records exact tensor bytes on independent
hosted runners. Input identity, training code, Torch version, thread topology and
numeric trace completeness are always fail-closed.

Exact model-byte equality is required when the pair is not an observed
AMD-versus-Intel cross-vendor pair. For an AMD/Intel pair, the repository retains
and reports the first numeric divergence but does not require universal bit
equality. Retained #298/#146 evidence already established that cross-vendor
trajectories can diverge from sub-nanoscopic optimizer differences even under the
same pinned CPU training contract.

This exception is diagnostic only. It does not relax dataset identity, objective
math, product-development preflight, formal qualification, robustness,
continuous-FAR, model promotion, or shipping evidence gates. A random matching
hosted CPU pair must not be used to erase a retained cross-vendor failure.
