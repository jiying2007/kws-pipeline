# Official CPU Torch metadata probe

A bounded dependency-planning check on a public repository's standard GitHub-hosted Ubuntu 24.04 runner. It does not install software or download wheel bodies, model weights, audio, or training data, and does not run a model or training. The local environment previously returned HTTP403 for the official linked wheel HEAD and PEP658 metadata; those observations are retained. A result on the separate Actions runner does not restore or qualify the local environment.

The only live request sequence is:

1. GET https://download.pytorch.org/whl/cpu/torch/ (1 MiB body cap), require the exact CPU wheel link and metadata hashes
2. HEAD the exact official linked torch 2.12.1+cpu CPython3.12/manylinux2.28 x86_64 wheel, inspect Content-Length; never read its body
3. GET that link's PEP658 .metadata object (200,000-byte cap), require its SHA256 and package/version identity

Redirects, alternate URLs, authentication changes, retries, installation and wheel GET are prohibited. Failure stops the sequence with FAILED_NO_RETRY and exit2. The workflow is limited to two minutes and the first attempt of one explicit research-branch push. Creating the draft PR does not trigger this probe again. A further source-changing push would be another invocation and is outside the one-run publication protocol.

The metadata-probe job has contents:read, checkout does not persist credentials, and there are no secrets, caches, artifact uploads, repository writes or scheduled triggers. Logs contain only public package metadata and an allowlisted runner/run identity. The standard public runner's compute use is free; this design avoids artifact/cache storage consumption. Future training and checkpoint storage need their own approved protocol.

Exact expected wheel SHA256: ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df

Exact expected metadata SHA256: 2dd7308d91027a8d24fe97f0982dca9a304814248869fb289c1d54e8e6c388ff

Offline tests use only invented metadata and mocked responses. No test contacts package servers. A successful probe establishes only metadata availability and package size/dependencies on that Actions runner. It does not establish installation success, numerical parity with a historical CUDA build, training feasibility, acoustic quality or target-board performance.

Official references:
- https://download.pytorch.org/whl/cpu/torch/
- https://pytorch.org/get-started/previous-versions/
- https://github.com/pytorch/pytorch/blob/v2.12.1/LICENSE
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://docs.github.com/en/billing/concepts/product-billing/github-actions
