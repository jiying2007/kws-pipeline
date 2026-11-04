# Storage evidence and next-step conditions

The read-only [run 41824495359](https://github.com/jiying2007/kws-pipeline/actions/runs/41824495359) succeeded on 2026-11-04. Its fresh ubuntu24 / 20261026.354.1 runner reported 91,610,882,048 available bytes (about 85.32 GiB) for root, workspace and Docker storage, with equal device IDs. Device equality is not complete mount-identity evidence. The observation follows checkout and source tests and satisfies the unchanged 25 GiB threshold. It is not the failed run's machine and does not reconstruct its capacity or reserve the next runner.

## Existing guards and measured sources

- Primary locked downloads total 3,340,966,447 bytes: packages 523,723,580; Qwen assets 1,880,560,703; Sense assets 936,682,164. The effective adapter requires initial free space of 16 GiB and checks workspace 12 GiB, installed environment 3 GiB and build temporary files 2 GiB. See research/source-screen32-v1/setup_adapter.py lines 142–146. The underlying helper's old 12 GiB error text does not override its effective 16 GiB check.
- Whisper wheels total 287,213,471 bytes, source 1,874,441 and the prior actual checkpoint 483,617,219: 772,705,131 bytes combined. Its setup requires initial free space of 8 GiB and checks site-packages at 3 GiB. See research/source-screen32-whisper-v1/setup_runtime.py lines 233–234 and 275–278.
- Download sizes are not installed or peak occupancy. Whisper's download-plus-site-packages sum excludes bootstrap, temporary files, home, JSON and transient unpacking. Primary checks are checkpoints, not a continuous filesystem quota.
- Image compressed-layer/config declarations total 45,449,534 bytes, not unpacked disk usage. Previous artifacts did not retain actual runtime peak, installed size or the failed runner's free bytes.

## Safe next steps

1. Preserve consumed experiment branches and failures. The old prepare stopped before Docker; all three model preparations were not_run.
2. Any new, separately reviewed prepare must record numeric capacity in the actual execution job before Docker, compare that recorded snapshot against the unchanged 25 GiB gate, and stop if insufficient. This probe does not waive that gate.
3. Preserve primary's 16 GiB and Whisper's 8 GiB setup guards. Record allowlisted aggregate checkpoints around image acquisition, setup, preparation and policy freeze. Do not export raw ASR/model payload as diagnostics.
4. Both runtimes currently coexist. Sequential preparation could reduce overlap, but deleting even reproducible task-owned runtime files needs its own reviewed scope and authorization. No cleanup is implemented or authorized by this plan, and cleanup cannot repair a failure before any task runtime exists.
5. Do not lower thresholds to pass, loop retries, or switch services around restrictions. If the actual execution runner is insufficient, stop and identify an allowed adequately provisioned environment.
6. Full ASR remains blocked pending controlled raw-evidence persistence and independently reviewed train provenance and trust roots. Machine consensus is not human truth; this plan grants no training admission.

The completed measurement performed zero model calls, downloads, package installations, Docker commands, cleanup actions and experiment retries.
