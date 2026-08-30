# Reproducibility levels

`kws-pipeline` distinguishes three claims:

1. **Traceable training**: the checkpoint records the exact training environment and corpus identity actually used.
2. **Rebuildable training**: the base container image and all Python dependencies are digest/hash pinned so the environment can be reconstructed from retained inputs.
3. **Bit-reproducible SDK**: two independent release builds produce byte-identical installed SDK trees.

Deterministic tar metadata alone satisfies none of the compiled-binary claims. Release documentation must use the strongest claim actually proved by the current gates.
