# B24R1 software delivery entry

`once.py` supplies the concrete post-qualification packaging and installation
path. The committed `selection.json` remains null. The corresponding
`kws-data/automation/english_v6_b24r1_delivery_inputs.json` is also inert; its
pipeline commit, manifest digest and file inventory are null. This change does
not select a candidate or start an installation.

The actual [B24R1 SOURCE run 45534020313](https://github.com/jiying2007/kws-data/actions/runs/45534020313) stopped at the ninth Parakeet
observation: `b24r1-primary-84-mero-fast` returned an empty transcript, after
eight valid returns. The fixed source contract does not permit a partial-panel
pass, a retry or automatic R2. No B24R1 PRIMARY/KWS quality result exists, so
this candidate cannot open the installation path. The selection and entry
binding remain null. The implementation below is retained as inactive
delivery infrastructure; it does not rehabilitate the failed source run or
grant another candidate the B24R1 identity.

The one-time workflow is hosted in **jiying2007/kws-data**, which already owns
the original Actions artifacts. Its existing `github.token` uses only
`contents: read` and `actions: read`. It checks out the exact separately
reviewed public **kws-pipeline** selection commit. No cross-repository token or
new credential is required.

## Exact reviewed inputs

After the original primary and replication quality gates and the independent
engineering readback actually pass, retain the readback's exact emitted files
in reviewed pipeline Git:

- `readbacks/<completion-sha256>/completion.json`
- `readbacks/<verifier-sha256>.py`

The delivery manifest pins both complete byte sequences. The entry checks that
the completion report's candidate, pack, SDK commit, complete files, four
engineering artifact pins, quality locks and scope equal that manifest. It
does not execute the saved verifier. A report whose JSON says PASS cannot
replace the existing reviewed selection and manifest digest.

The four original engineering ZIPs stay in their original public run. The
entry fetches their exact IDs, first-successful-run identities, Git heads,
sizes and SHA-256 digests, then preserves their complete bytes for packaging.
It downloads neither a replacement candidate nor rebuilt weights. Existing
`delivery.py` rechecks every archive CRC, result member, model and installed
SDK member before making the delivery tar.

### Baseline container provenance

The original B0 archive is artifact **14514228406**, run **44495594627**, head
`d0b567fe37147aae6ab5d7bdec3721b879a0b22d`, exactly 501,276 bytes and SHA-256
`91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702`.
That archive contains the original PT, NPZ and ONNX exports; it does not contain
an already-saved `.pack` file.

For explicit B0 rollback, the entry serializes the original NPZ's contiguous
little-endian float32 payload bytes in the unchanged `pack_mero43.py` geometry
order. It copies those raw tensor bytes without a numeric conversion and adds
the existing 16-byte pack header. The NPZ must have SHA-256
`02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696`.
The final B0 container must be exactly 216,960 bytes with SHA-256
`ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b`.

The receipt explicitly records `baseline_container_serialized: true`,
`existing_downloaded_pack_claimed: false`, `weights_modified: false` and
`model_forwards: 0`. The candidate pack follows a separate path: it is copied
unchanged from the actual original engineering result ZIP.

## Reviewed activation

Root review first binds the selected pipeline manifest as described in
`README.md`. Preserve the exact small readback report and source at the paths
above; never place placeholder PASS evidence there. A subsequent reviewed
kws-data commit binds:

- `pipeline_repository`: `jiying2007/kws-pipeline`;
- `pipeline_commit`: the exact 40-hex delivery-selection commit;
- `manifest_sha256`: that selected manifest's complete SHA-256;
- `pipeline_files`: all nine exact files returned by
  `once.expected_pipeline_files(manifest, manifest_raw)`, each with `bytes`,
  `sha256`, `git_blob_sha1` and original Git `mode`;
- `enabled: true` only after those actual original proofs are complete.

The nine files are the existing delivery utility, this entry, selection,
loader C source, loader CMake file, unchanged B0 packer, selected manifest,
original completion report and original readback source. The executing entry
checks every file against that exact pipeline Git tree; extra or missing files
in the binding are rejected. The manifest's `pipeline_commit` continues to
name the original SDK source `88a043c6c00bb3a1a62f45098a023c7601042ded`.
The entry binding's `pipeline_commit` names the later delivery code and
selection commit. These are distinct identities.

Activation is one workflow-only child of the reviewed kws-data preparation,
on `research/en-v6-b24r1-delivery-once-20261125`. Replace only the two
`__B24R1_REVIEWED_DELIVERY_PARENT__` tokens in
`.github/workflows/english-v6-b24r1-delivery-once.yml`. The frozen template stays
unchanged. The entry requires a public push, exact branch, attempt 1, exact
parent and exactly one original push run for this workflow and branch. It has
no dispatch, schedule, workflow-run continuation or automatic retry route.

## Actual post-selection work and retained evidence

The actual job packages the authenticated original candidate exports and
three SDKs plus B0, installs them into an immutable manifest-named directory,
and verifies every installed byte and prescribed mode. It then builds only
`loader_check.c` against the delivered **GCC SDK**. The SDK and model are not
rebuilt. The clean consumer runs twice: once with the selected candidate and
once with B0. Each run calls the pack loader once, with zero stream
initialization, PCM reads or model forwards.

The public run artifact is named
`english-v6-b24r1-software-delivery-<run-id>` and retains:

- `heym43-B24R1-software-delivery.tar.gz`, containing the complete selected
  software and its exact delivery manifest;
- package, immutable install, installed verification, candidate load and B0
  rollback-load receipts;
- exact original API provenance and the selected completion report/verifier;
- consumer source/build identities, compiler output and the final result.

Successful delivery requires
`DELIVERED_ORIGINAL_SOFTWARE_WITH_CANDIDATE_AND_B0_LOAD_VERIFIED` in the final
receipt and a successful original workflow run. A failure retains `STOP.json`
with its stage, even if an earlier packaging step had produced a tar. An
intermediate tar or successful compile alone is not a completed delivery.

No B24R1 installation has been executed. An eligible, independently reviewed
delivery job would verify installation on its clean GitHub runner. The tar
would be the portable installation deliverable; no running user process is
switched. B0 rollback loading has a separate clean model-load check.
Physical SSC305 execution, field FRR, continuous FAR/h and physical latency
remain outside this software evidence. The existing research-only scope and
false product/CAL permissions are preserved, and no Release publisher is used.

## Contracts

Pipeline CI runs the existing delivery contracts plus `test_once.py` in normal,
optimized and stripped-docstring modes. The data entry has its own stdlib-only
contract job. Tests use invented temporary data and mocked process boundaries;
they do not download artifacts, compile or load a model, execute a verifier or
read audio. The real B0 byte-only container algorithm was separately checked
against the retained original archive and fixed output hash without importing
NumPy, Torch or ONNX Runtime or writing a pack file.
