from __future__ import annotations

import os

# Python imports sitecustomize during interpreter startup, before train_ctc.py
# imports torch. Pin the CPU math/scheduler environment here so hosted runner
# CPU capability and BLAS defaults cannot silently change the training path.
_DETERMINISTIC_CPU_ENV = {
    "OMP_NUM_THREADS": "2",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "2",
    "MKL_CBWR": "COMPATIBLE",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "ATEN_CPU_CAPABILITY": "default",
}

for _name, _value in _DETERMINISTIC_CPU_ENV.items():
    os.environ[_name] = _value
