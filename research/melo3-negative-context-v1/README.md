# M3–M5: one fixed 300 ms appended-context condition

M3 (human words 你好小屋) falsely activated K1 after exactly 4,800 digital-zero samples were appended to its original PCM16 stream. M4 (小屋小屋) and M5 (你好你好) had no events under the same condition. This is 1/3 nonwake clips with an event, a descriptive observation rather than a population false-activation rate.

| Clip | Human words | Original events | Modified events | Saved greedy text |
|---|---|---:|---:|---|
| M3 | 你好小屋 | 0 | 1 | 你好小窝窝屋 |
| M4 | 小屋小屋 | 0 | 0 | 屋屋窝屋 |
| M5 | 你好你好 | 0 | 0 | 你好小你好 |

Every observed event is retained in the raw trace, exact observations and summary.
- M3: K1, score 0.8743068007994056, decoder frames 51–84, input available 14400 samples, original EOF 12446 samples, difference 1954 samples (0.122125 s). This is callback availability, not acoustic word-end latency.

The original five-clip result remains 0/2 expected-keyword hits and 0/3 nonwake clips with events. Its controls were reused, not rerun. In the separate earlier M1/M2 appended-context run, M1 had a K1 event and M2 remained a miss, or 1/2 expected-keyword hits. This negative run now matches that fixed 300 ms context condition; the two historical runs remain separate. M1, M2 and M6 were excluded from this run.

All 78 original model rows (468 FP32 scalars) whose frontend support is inside original EOF remain bit-identical. Compare by absolute center using `(center + 2) * 160 + 400 <= original EOF`: M3 centers 0–72 (25 rows), M4 0–87 (30), M5 0–66 (23). Original M3/M5 third callbacks were finishes with 6/4 rows and original M4 fourth finish had one row. Each corresponding modified callback is a full feed with 10 rows. Callback grouping and availability changes are expected and separate from logit drift.

The saved run contains 108 model rows, 108 searched decoder rows, 328 fbank rows, 13 callbacks, 13 feed calls and 3 finishes across three cold streams. All 53,411 samples and all model logits are retained. Activation may stop a callback's search early; 0 computed model rows were not searched in this run.

These single-voice, exposed synthetic clips do not establish population FAR/FRR, independent-source qualification, general improvement, acoustic tail truncation or a production tail policy. Human words were heard on native 44.1 kHz audio; the existing 16 kHz derivative was not independently heard. Human labels, earlier machine FAIL, source quarantine and licensing limitations are unchanged. No duration, threshold or decoder sweep was run.

## Reproduce the saved comparison

Use the unchanged core in `research/melo5-a20-diagnostic-v1` and `research/2026-10-06-melo5-a20-diagnostic`. Dependencies are pinned by byte size and SHA256 in `metadata/bindings.json`, with source commit `462406f60145e37bd0e00141e51c5d768568dfa9` and data commit `014ce1c70fa083c4bf6e485400efd4ac79195311`. With source and data siblings in their respective checkouts:

```sh
python3 -B research/melo3-negative-context-v1/verify_context.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo3-negative-context
python3 -B research/melo3-negative-context-v1/tests/test_context.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo3-negative-context -v
```

Use `--core-source` and `--core-data` when the core folders are elsewhere. The verifier first validates the unchanged core, checks this closed four-file data appendix and reproduces all 43,571 bytes of observations. Nine focused tests reuse the prior annex pattern: exact reproduction/accounting, preserved versus added context, missing/duplicate centers, changed callback grouping and EOF-relative availability, wrong tail length, changed source dependency and an event-summary omission/invention guard. Only Python's standard library is used. No audio, model, frontend, decoder, compiler or network is invoked.

## Evidence projection and source reuse

- `original_A20.raw.jsonl`: private acquired 20,567 bytes, SHA256 `5805bf033b7f39a5c77a20986ceadaa45ca743195ffb3f2374c43dec4eb49513`; public 20,559 bytes, SHA256 `743e97c86540a1917c938170fc21b511fc6a4b75e61f5877a13bce8cd226b98d`. Remove only run_start.pid, preserving every other acquired byte including all following 37 records and numeric spellings.
- `observations.json`: private acquired 43,571 bytes, SHA256 `94c8843cb0f1540a61ab8145992d39b8e49dff05a98354fbd99b97cf4184f082`; public 43,571 bytes, SHA256 `94c8843cb0f1540a61ab8145992d39b8e49dff05a98354fbd99b97cf4184f082`. None. Exact historical observations retain acquired context/control raw hashes. Verification scores public projections, then restores only those two declared provenance hashes for exact report comparison.
- `resources.json`: private acquired 2,919 bytes, SHA256 `55ce7652c8a1e34421e6f5d3f6b413d7ab7f6d50ca0d906b9dba05c1bcead508`; public 2,143 bytes, SHA256 `eaa4d96a98fb8521900dd01cfa1d798c96ef5331517ea6c7b41a47001a22310e`. Remove only process-sample absolute monotonic_ns and release-file inventory; clarify resource_scope inherited-memory caveat. All measurements, relative timings, bindings and limitations retained.

The exact observations deliberately name acquired raw hashes. A temporary saved-control receipt changes only its expected raw hash so the frozen comparator can read the core public projection. Verification then maps only the context and control provenance hashes back to their acquired identities for exact historical report comparison. No PID is recovered and no scientific values change.

Small checked source substitutions reconstruct the exact frozen collector, scorer and plan from the already-public core. The input header, prefix comparator, manifest and geometry are copied byte-for-byte. Reconstruction checks both base and result hashes in a temporary directory and never modifies the core. The plan is its historical pre-execution snapshot, not a statement that this completed run remains unexecuted. The compiler command and collector executable identity are recorded without including or running the executable. Runtime, model, decoder, prune and threshold identities are unchanged.

The input recipe preserves each bound original WAV’s PCM bytes, appends exactly 9,600 zero bytes, and updates only its RIFF/data sizes. Original, modified, prefix and zero-tail hashes are retained. No original or modified WAV, model payload, duplicate runtime, full scorer or acquisition launcher is included. This is a saved-output reproduction appendix, not a turnkey acquisition package.

## Resource observations and limits

Whole-child CPU was 0.055213000000000005 s and launch-to-reap wall time was 0.064600761 s. Lifetime peak RSS was 12,404 KiB and may include inherited Python memory before exec. Maximum sampled RSS was 2,876 KiB; sampling is not an isolated or exact native peak. Supervisor usage is excluded. Native timers retain the core's post-GO, post-dlopen to pre-final-record scope, including initialization and preceding trace output; feed/finish timings include callback output.

These are one local run’s observations, without continuous thread/child-absence or embedded-board qualification. Model/runtime/library identity is unchanged; timing and RSS differences from the previous host are not a performance comparison. Research evidence only; there is no new commercial or generated-output license grant.
