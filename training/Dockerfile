ARG KWS_TRAINING_BASE
FROM ${KWS_TRAINING_BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_NUM_THREADS=1 \
    MKL_CBWR=COMPATIBLE \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ATEN_CPU_CAPABILITY=default \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KWS_TRAINING_CONTAINER=1

WORKDIR /workspace
COPY training/requirements.lock /tmp/kws-requirements.lock

# The complete Python dependency tree is supplied by one immutable OCI base
# image (`name@sha256:<digest>`), validated by training/build_container.py.
# Do not resolve or upgrade packages while constructing a qualification image.
RUN python - <<'PY'
import pathlib
import platform
import torch
expected = {}
for raw in pathlib.Path('/tmp/kws-requirements.lock').read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    name, version = line.split('==', 1)
    expected[name.strip()] = version.strip()
assert platform.python_version() == expected['python'], (platform.python_version(), expected['python'])
assert str(torch.__version__).split('+', 1)[0] == expected['torch'], (torch.__version__, expected['torch'])
PY

ENTRYPOINT ["python", "training/train_ctc.py"]
