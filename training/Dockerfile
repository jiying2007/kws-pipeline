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
COPY configs/runtime/research-runtime.json /tmp/kws-research-runtime.json
COPY tools/research_runtime.py /tmp/kws-research-runtime.py

# The complete Python dependency tree is supplied by one immutable OCI base
# image (`name@sha256:<digest>`), validated by training/build_container.py.
# Do not resolve or upgrade packages while constructing a qualification image.
RUN python /tmp/kws-research-runtime.py --manifest /tmp/kws-research-runtime.json \
      training-lock --check /tmp/kws-requirements.lock \
    && python /tmp/kws-research-runtime.py --manifest /tmp/kws-research-runtime.json \
      check --role core --packages

ENTRYPOINT ["python", "training/train_ctc.py"]
