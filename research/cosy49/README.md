# Cosy49 single-attempt execution revision

The preparation record below describes the earlier disarmed snapshot. This revision enables only the declared one-push branch under the exact execution release. The protocol, model, data, mathematics, resources and call budgets are unchanged. Execution requires both reviewed source/readback bindings and the actual EXECUTION-RELEASE.json; no retry, resume or later evaluation is authorized by this package.

# Disarmed Cosy49 fixed300 training source

This package prepares one new research candidate from the original A20 state. It reuses the previously successful fixed300 CPU dependency, process guard, exact input transport, failure-evidence and saved-only finalizer infrastructure. The previous candidate is terminated; its weights and optimizer are never resumed.

No training, model control, audio frontend, dependency download/install, workflow launch or public write was performed in this preparation. The proposed workflow is explicitly disabled and both release approvals are false.

## Training delta

- Exact49 order: original D20 weak20, original Qwen human12, then Cosy actual17, aliases A–P and R. The inaudible Q and all development/sealed recordings are absent
- Equal one-third source-cohort means of each source's full CTC NLL divided by actual target length; D is 小窝 and I is 小屋
- PRE-CMVN49×95×400 zero-padded input,2505 valid rows, original800-element CMVN exactly once inside each required forward
- First pre-update forward is the original49 control. Its existing logits and CTC are saved before the same loss graph is used for backward/update1. No added control forward or repeated CTC
-300 full-batch fresh AdamW updates, seed610104, original390520 trainable parameters, unchanged optimizer and CPU FP32 settings, then one fixed terminal forward
- Every update checks all28 optimizer states after the update, including finite CPU FP32 step/exp_avg/exp_avg_sq tensors and exact step/moment shapes. This unmeasured scan overhead is within the unchanged training guard
- Success requires301 forwards,300 backwards,300 updates and300 completed optimizer-state scans; interrupted started/completed journals retain uncertainty
- First and terminal111720-byte logits, per-row CTC/greedy diagnostics,300 scalar update records, terminal weights/optimizer/RNG and source bindings are retained. Training-fit change is diagnostic only

## Remaining admission blockers

The saved Cosy17 feature hashes, independent saved-output audit, clean-public projection and exact remote byte/CI audit are closed. The single public ZIP is bound to kws-data commit a3f38afed7c2fc9bc214bce8c22928c1a2cb675b, archive SHA256 d43806895671140a22119021d409a7e1dbfb7facfde3392907573dc5e91d3ea3. Its19 allowlisted inert members (17 feature blobs, manifest, README) have exact whole-archive and per-member bytes/hashes; no bundle code is executed. The explicit public-manifest adapter maps nested features to local loader names without altering bytes. The private acquisition manifest is not a training transport.

The runtime and proposed workflow remain disarmed. Final source review and durable readback, runner-image revalidation, an exact new training/public-output approval, and a reviewed change to the disabled workflow remain separate steps. Data publication and its successful generic asset CI do not establish training success or quality.

The original32 public prepared input manifest remains byte-identical and bound to immutable kws-data commit d0d54cf635189bdc522c9d0f6135082e81250fd8. This runtime does not read old32 control logits for initial49 diagnostics.

Numerical protection, the future98×2 paired audio comparison, endpoint selection, ARM/export and publication stages are not wired into the training job. No quality, generalization, device or runtime-regression claim follows from source preparation.

## Pure checks

Run the stdlib-only fixtures with:

    python3 -B -m unittest discover -s research/cosy49/tests -v

Tests use invented tensors/HTTP/ZIP data and AST/source inspection only. Default runtime entrypoints reject without a release. This is not a measured model/training test.
