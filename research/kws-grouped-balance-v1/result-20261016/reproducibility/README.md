# Public reproducibility evidence

This directory preserves the original claim, launch wrapper, reservation, process receipts/logs, frozen independent verifier and tolerances, trusted freeze, exact executed 23-file source snapshot, and official commit/tree/CI evidence. The prelaunch ZIP is original, not regenerated. Preserved source is historical and enabled only as evidence; the active campaign is consumed and disabled. Never execute the launch wrapper or model runner.

## Integrity and portable verification

Run python ../../verify_archived_result.py from this reproducibility directory (or use its absolute path). It reconstructs the 66 raw-result parts in memory and checks the original hash manifest, receipt sequence and saved-token statistics. It does not call a model or independently recompute CTC. manifest.json and the enclosing archive-manifest.json bind these reproducibility files.

For the frozen full float64 CTC verifier, restore the sibling layout named in grouped-balance-result-verification/README.md: source snapshot, claim, execution evidence and wrapper from this directory; extract grouped-balance-result/ and grouped-balance-result-acceptance/ from the reconstructed raw ZIP into that same base. The archive must first pass the portable integrity verifier; never execute ZIP contents or launch wrappers.

IMPORTANT: The unchanged original verifier binds the absolute launch-command digest. Full CLI verification therefore requires the historical base /workspace/scratch/6c2ef8b5a46e/kws-next-stage-20261015. A relocated base fails fixed_launch_command by design. Do not modify receipts, trusted freeze or verifier to conceal that mismatch. The model environment, wheelhouse and private input audio are not needed for saved-evidence verification. Choose a fresh report directory because existing reports cannot be overwritten. This is a saved-probability check, not a rerun or hardware/process attestation.

## Test provenance

The preserved test-harness-source-relocation.diff and JSON describe a test-only path adjustment from horizon-integrated to grouped-balance-integrated; verifier, tolerances and scientific source were unchanged. The prelaunch ZIP preserves original prelaunch test bytes; the adjacent current test harness and logs retain later path relocation evidence. Failed initial reconstruction-test-normal.log is preserved rather than presented as a pass. The outer fixture is a reconstructed synthetic fixture; it is not experimental evidence. No internal review reports are included.
