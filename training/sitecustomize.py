from __future__ import annotations

import os

# Python imports sitecustomize during interpreter startup, before train_ctc.py
# imports torch. Pin the CPU math/scheduler environment here so hosted runner
# CPU capability and BLAS defaults cannot silently change the training path.
# GitHub-hosted x64 runners support AVX2; using it as the common baseline avoids
# AVX2/AVX512 drift without forcing the numerically different generic fallback.
_DETERMINISTIC_CPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "AVX2",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "PYTHONHASHSEED": "0",
}

for _name, _value in _DETERMINISTIC_CPU_ENV.items():
    os.environ[_name] = _value
