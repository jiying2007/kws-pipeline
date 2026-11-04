# One-shot read-only runner storage evidence

This separate campaign measures capacity on one fresh `ubuntu-24.04` GitHub-hosted
runner. It does not rerun native preparation, model setup, audio recognition or
any consumed experiment. The original preparation run `41822443746`, source
prefix `2784c3a`, stopped before Docker on its unchanged 25 GiB free-space gate.
That old run did not capture numeric capacity; this new run is a different runner
and cannot reconstruct that old run's exact free space.

The workflow runs only for creation of the exact new public branch
`research/runner-storage-20261104-v1`, workflow run number 1 and attempt 1. The
Python admission independently verifies those fields, the zero `before` SHA,
event/checkout/workflow SHA equality, repository identity and GitHub-hosted Linux
X64 environment. A single job has a five-minute timeout. Checkout and artifact
actions retain the exact pinned commits from the reviewed preparation workflow.
The parent publishes and launches only after review; no workflow was run while
authoring or testing these files.

Only three paths are measured: root, the checked-out workspace and
`/var/lib/docker`. Per target there is one `stat`, one explicit `statvfs` and one
`shutil.disk_usage` call, which performs its own filesystem-capacity snapshot.
There is no directory traversal, `du`, mount-table parsing, Docker invocation,
container/image listing, environment dump, package installation, model download,
model call, cleanup, deletion, threshold change or retry. Optional installed
toolchain directory sizing is deliberately omitted because it would require a
potentially broad walk. The collector creates only its exclusive output directory
and one aggregate JSON file. Tests create only disposable invented fixtures.

The report contains numeric byte/inode counts and device IDs/major/minor numbers
under three fixed labels. Pairwise device equality indicates shared filesystem
devices, not distinct mount identities, and does not reveal mount paths. Capacity
snapshots are sequential rather than atomic and are not required to agree.
`disk_usage.available_bytes` is the same available-to-user measure used by the
unchanged preparation reserve check. The report retains both capacity snapshots,
uses their conservative minimum for `meets_unchanged_prepare_reserve`, and makes
that comparison informational only. It never authorizes model execution or
changes the 25 GiB gate. A missing or inaccessible Docker path is reported as a fixed status
and numeric errno with no exception text. A partial measurement still yields the
aggregate artifact but fails the collection step.

Exact current source identity consists of the head SHA, workflow SHA/path, run
ID/number/attempt and SHA-256 of `source-lock.json`. Optional runner-image metadata
uses only `ImageOS` exactly equal to `ubuntu24` and `ImageVersion` matching eight
digits followed by two dot-separated groups of one to three digits. Missing or
invalid values become null; no arbitrary environment values are exported.
The lock binds the four fixed
public source files including this workflow and tests. The parent must compare
that lock digest and committed source to its reviewed publication record. No raw
filesystem paths, native payload, source audio, model weights, private file bodies,
container/image names, credentials or arbitrary logs are included in the report.
Only the exact `status.json` file is uploaded, with seven-day artifact retention.
GitHub's ordinary action/source-test logs are not repackaged as evidence.

Offline tests use standard-library mocks and invented numeric counters. They
exercise no actual disk-capacity measurement, Docker, model, remote operation or
paid API. Run the test file with `python3 -I -B`, then with `-O` and `-OO` as in the
workflow. Syntax is checked against Python 3.8 by those tests. Test success is not
evidence of actual runner capacity or a successful remote measurement.
