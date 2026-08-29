FROM python:3.12.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KWS_TRAINING_CONTAINER=1

WORKDIR /workspace
COPY training/requirements.lock /tmp/kws-requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/kws-requirements.lock

# Build the image by digest in qualification infrastructure and expose that digest
# as KWS_TRAINING_IMAGE_DIGEST when invoking train_ctc.py. The checkpoint records
# both this identity and the lock/Dockerfile hashes.
ENTRYPOINT ["python", "training/train_ctc.py"]
