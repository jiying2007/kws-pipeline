# Dormant fail-fast setup preparation

This source-only change follows the two retained setup failures. It does not
activate a third experiment, change a completed run, or refresh either old
execution freeze/release. Both old freezes remain exact historical records and
reject the changed setup source. Neither old activation may be reused.

The only runtime change is ordering. All locked source files still download
first. Every locked wheel must then download and pass the existing size/SHA256
checks; offline installation, `pip check`, and installed-byte inventory must
succeed before the same model URL is requested. The complete download receipt is
written only after that later model acquisition succeeds. An earlier failure
retains the existing progress/failure JSON and never produces a complete receipt.
No model import or inference is added to setup.

The actual downloader, exact URLs, hosts, source/runtime/model locks, model SHA,
per-asset checks, shared download counter, no-retry policy and time/resource
budgets are unchanged. This does not claim to fix HTTP 403. It avoids acquiring
483617219 model bytes before discovering an unavailable or invalid dependency.
It does not infer the first run's unknown cause from the second run's diagnosis.

Tests execute only mocked stdlib orchestration with tiny fixture files. Injected
wheel, installation, dependency-check and inventory failures must leave the model
downloader untouched; a model failure must leave no complete receipt. Positive
admission tests use a temporary, explicitly re-frozen synthetic source fixture.
A separate test keeps the real second freeze unchanged and requires its explicit
source-drift rejection. At fail-fast commit `8552b88`, only setup and the test
file differed from that old closure; later README banners explicitly retire the
old activation instructions without refreshing their freezes;
the seven other blind files and every retained historical artifact remain intact.

No official-pip transport replacement is included. A future acquisition design
must separately demonstrate the same fixed identity, no resolver fallback,
no retry, permitted-destination and bounded-response-body guarantees. The user
subsequently authorized one third phased attempt; its transport review
is in `attempt-3/PIP_TRANSPORT_REVIEW.md`. That authorization remains unused
because equivalent official-pip acquisition bounds have not been established.
No third-run gate or new execution freeze is created by this preparation.
