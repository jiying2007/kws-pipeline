# Hey Mero 43K — standalone C11 SDK handoff

This directory is the single-English-keyword research runtime, not the historical two-keyword head. Only **Hey Mero** is active. It is an independently built library, with **no change to the existing shipping `KwsPipeline::core`**.

## Offline compiler / application integration

```sh
cmake -S research/english_v5_mero -B /tmp/mero43-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHEY_MERO43_BUILD_TESTS=ON \
  -DHEY_MERO43_BUILD_HOST_TOOL=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/heym43-sdk
cmake --build /tmp/mero43-build --parallel 2
ctest --test-dir /tmp/mero43-build --output-on-failure
cmake --install /tmp/mero43-build
```

Link in a consuming CMake application:

```cmake
find_package(HeyMero43 0.1 CONFIG REQUIRED)
target_link_libraries(my_16khz_pcm_application PRIVATE HeyMero43::stream)
```

The installed public headers are `heym43/mero43_stream.h` and `heym43/mero_ctc.h`. No Python runtime or model training is required on SSC305. Cross-compile with the project-specific Cortex-A32 ARM hard-float sysroot/toolchain; do not claim actual-board execution from a cross-compile. The core requires a caller-owned model arena and stream state; query `en_v5_m43_model_bytes()` and `en_v5_m43_stream_bytes()` rather than hardcoding structure sizes. Inference uses no heap allocations, file I/O or threads in the core. The caller supplies **16 kHz mono little-endian signed PCM16** with monotonic sample indices, consumes events by callback, and calls reset or discontinuity explicitly on audio gaps.

## Live post-AFE input on POSIX systems

Set `-DHEY_MERO43_BUILD_LIVE_TOOL=ON` in the SDK build to install `bin/heym43-live`. Feed one continuous session of 16 kHz mono signed little-endian PCM16 from the board's post-AFE output. The adapter reads raw bytes from stdin and writes one JSON line per complete wake event to stdout. The capture application owns the microphone, channel selection, sample rate and AFE.

```sh
: "${B0_PACK:?path to the authenticated B0 pack}"
: "${MAX_GAP_SAMPLES:?maximum inter-token gap in samples}"
: "${COOLDOWN_SAMPLES:?refractory period in samples}"
: "${POST_AFE_FIFO:?raw PCM FIFO from the capture application}"
heym43-live "$B0_PACK" "$MAX_GAP_SAMPLES" "$COOLDOWN_SAMPLES" < "$POST_AFE_FIFO"
```

Verify the B0 pack against SHA-256 `ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b` before starting the adapter. The two decoder settings are mandatory. Product settings require separate calibration. A short read can split a sample. The adapter retains its first byte until the second arrives, rejects a trailing half-sample at EOF and reports sample, frame and event totals to stderr at a clean EOF.

The output fields are `event`, `keyword_id`, `frame_index`, `available_after_sample` and `callback_monotonic_ns`. Frame indices begin at zero. The sample clock counts consumed input samples, and the monotonic timestamp comes from the callback's own clock. Physical latency also requires a time-aligned, independently observed word ending. Keep stdout drained because a blocked consumer stalls audio ingestion. A raw pipe cannot signal a dropped interval. End the process on capture loss and start a new process/session when capture resumes, or use the C API's explicit discontinuity operation in the capture application.

## Model provenance (not in the SDK)

- Frozen real **43,876-param / 256-update FSMN-CTC**: kws-data run [44495594627](https://github.com/jiying2007/kws-data/actions/runs/44495594627), artifact `14514228406`, original ZIP SHA256 `91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702`.
- Repack on the build host ONLY using `pack_mero43.py` with that original immutable ZIP, then separately authenticate the output pack SHA before provisioning or loading it. The fixed C11 package is **216,960 bytes**. Never regenerate weights from arbitrary `*.pt`, use self-declared unlabeled model weights, or silently accept a new model package. The C loader validates package format/finite tensors but is **not** a cryptographic SHA verifier.
- Verified software numerical chain: [PyTorch↔x86 C11 run 44541193521](https://github.com/jiying2007/kws-data/actions/runs/44541193521) maximum absolute delta `0.0001220703125` on the 10 frozen prospective original PCM, and [x86↔Cortex-A32 QEMU run 44542782505](https://github.com/jiying2007/kws-data/actions/runs/44542782505) with max absolute delta `0.0`, full frame-token and event-frame agreement.

## Device completion gates (not met by this package)

An SSC305 integrator still needs the **actual final board microphone / audio front-end**, continuity/reset handling, sound-echo treatment and per-stream bounded memory accounting. Retain physical-board CPU and RAM measurements, P95 end-to-end inference/callback delay and long-run stability, independent real-human 3–5 m FRR, natural continuous FAR/h and untouched CAL/LOCKED test proof. The above ten synthetic development originals do **not** estimate real FRR or FAR/h, and QEMU is not an SSC305. This target is a valid *integration research SDK*, **not a released/quality-qualified wake-word model**.

No Work/Codex, private repository, paid external ASR/TTS or recurring monitoring job is needed for this standalone build.

## Candidate quality closure — B20 (2026-11-24)

The B17 paired model checkpoints were evaluated once on the new B20 synthetic panel together with the fixed B17 natural speech panel in [kws-data run 45337450827](https://github.com/jiying2007/kws-data/actions/runs/45337450827). The run completed all **1,008 model forwards**: 84 admitted originals × three fixed gains × four frozen model roles. The original logits and attempt journals were independently read back without new inference.

The numerical result is **NO_PROMOTION**. Relative to the paired control, the challenger lost one positive condition and added three natural-speech false-trigger conditions. It recovered none of the four B9-recoverable original/gain cases across three source groups. Natural ordinary-speech events at 0.75 gain were 7 versus B0's 6. The original opening reader rejects this result; B20 replication and candidate engineering validation remain closed.

The original producer ZIP omitted 32 `.github` files. A separately identified supplement binds each missing file to the real producer Git tree, Git blob and SHA256; the original artifact is unchanged and is not described as complete. All 17 original input archives and all 120 original source rows were authenticated, and the 84 admitted rows exactly reproduce the original admission. See [the B20 evidence directory](https://github.com/jiying2007/kws-data/tree/main/research/english_v6_b20_results_20261124) and `V6_QUALITY_EVIDENCE.json`.

The installed integration baseline remains **B0**. No B17/B20 candidate is released or substituted into this SDK. This result is development evidence; it provides no measured continuous FAR/h, field 3–5 m FRR, or physical SSC305 result.
## Candidate quality closure — B21 (2026-11-24)

The blank-aware positive-retention paired TRAIN experiment completed 192 updates per arm, retaining the 43,876-parameter FSMN-CTC. The existing 96 synthetic originals were reused byte-for-byte. The amended [source run 45378709327](https://github.com/jiying2007/kws-data/actions/runs/45378709327) added 24 fresh natural-speech originals and returned all 600 native observations. Source admission retained 22 synthetic positives, 48 synthetic ordinary utterances and 20 natural utterances from 12 readers.

The [first PRIMARY run 45387856667](https://github.com/jiying2007/kws-data/actions/runs/45387856667) completed all **1,080 model forwards**: 90 admitted originals × three fixed gains × four frozen models. Original result artifact `14917736688` has ZIP SHA256 `1be5f7aa27d4b2331c076a98e87e8a50ab51483f71cd08cfdd2b172fea0cefa7`. Its numerical verdict is **NO_PROMOTION**.

At gains 1.0 / 0.75 / 0.5, the B21 challenger detected 19 / 19 / 10 of the 22 positives, versus B0's 20 / 19 / 7. Natural-speech false-event totals were 7 / 6 / 6, versus B0's 9 / 9 / 7 and B9's 20 / 19 / 19; seven readers improved against B9 at original gain. These aggregate gains do not satisfy the frozen paired gate: one B0 positive condition was lost, three natural original/gain conditions added false events against B9 or the paired control (four reference rows), and no required positive source family was recovered. The matched control itself achieved 19 / 19 / 10 positive hits and 7 / 5 / 5 natural false events, so the new blank-retention term did not show the required positive recovery over its control.

Strict isolated readback found that the original PRIMARY producer archive omitted historical source-verification dependencies. The PRIMARY root lacks 41 files from the separately saved original 314-file source producer: 40 nonworkflow files and the source execution workflow. Their original bytes are authenticated against that source producer's manifest and Git tree; every existing intersection is unchanged. A separately identified retrospective supplement supplies only the 40 unchanged Python files, each bound to both the source and PRIMARY Git trees; the source execution workflow is not copied. Using this separate diagnostic root, the independent readback reauthenticated all 120 source originals and their 600 native observations, then verified all 1,080 saved arrays and their 171,732 frames and reproduced **NO_PROMOTION**. The combined readback SHA256 is `483be35deb1634bb2d60ffb5c6b539d7b7708cf96e2177b6eec75d5fd275d415`; the supplement manifest SHA256 is `c1b7036ff4e96ce83b33db2b9932bb156ce299bfe13907834e2458b7bae72a50`. The original PRIMARY artifact remains unchanged and incomplete, and the strict production reader is unchanged. This supplemental audit grants no replication, engineering or promotion authorization. Exact artifact and readback identities are recorded in `V6_QUALITY_EVIDENCE.json`.

**B21 replication and candidate engineering remain closed; B0 remains the installed integration baseline.** No B21 pack is released or substituted into this SDK. Closure reads only saved original evidence and makes no new model, TTS, ASR or training calls. This development panel does not measure continuous FAR/h, field 3–5 m FRR, or physical SSC305 performance.

## Candidate quality closure — B22 (2026-11-25 Asia/Shanghai)

The experiment identifier remains **20261124**. The [completed B22 TRAIN run 45400279395](https://github.com/jiying2007/kws-data/actions/runs/45400279395) trained one union-retention challenger for **192 updates and 1,847 total forwards**: 1,844 optimization forwards, two checkpoint-reload checks and one ONNX export. It used the unchanged 88 TRAIN originals, 122 positive-retention conditions (the original 120 plus two previously unprotected conditions) and 264 saved ordinary-speech teacher arrays. The 43,876-parameter architecture remained fixed. The original B21 challenger was retained as an immutable historical control, with zero new control updates or forwards; this comparison does not claim a control trained in the same job.

The [source continuation run 45428013301](https://github.com/jiying2007/kws-data/actions/runs/45428013301) preserved all 96 synthetic and 24 natural originals from [source run 45402179335](https://github.com/jiying2007/kws-data/actions/runs/45402179335), and reused its 240 complete phone-observer returns. It made the first 120 calls to each of the three text observers, completing **600 valid native observations** in total without regenerating audio or repeating a consumed observer call. The unchanged source gate admitted **14 synthetic positives, 46 synthetic ordinary utterances and 17 natural originals from 10 readers**, including 10 synthetic paired source groups. The 43 UNKNOWN originals remained in the complete 120-original exposure and overlap denominator and were excluded from model scoring.

The [first B22 PRIMARY run 45431814493](https://github.com/jiying2007/kws-data/actions/runs/45431814493), head `c6ff655ede7631700e8d0c3593273ada2e210983`, completed all **924 forwards**: 77 admitted originals × three fixed gains × four frozen models. The workflow conclusion was `success`; the scientific verdict was **NO_PROMOTION**. The counts below use gains **1.0 / 0.75 / 0.5**, respectively; gains are derived conditions, not independent source samples.

| Frozen model | Positive hits, out of 14 | Natural false events, on 17 originals | Synthetic ordinary false events, on 46 originals |
| --- | --- | --- | --- |
| B0 | 13 / 10 / 6 | 8 / 7 / 6 | 1 / 0 / 0 |
| B9 | 12 / 11 / 8 | 19 / 21 / 20 | 7 / 2 / 0 |
| Historical B21 control | 13 / 12 / 7 | 6 / 10 / 5 | 1 / 0 / 0 |
| B22 challenger | 13 / 12 / 6 | 6 / 10 / 5 | 1 / 0 / 0 |

The original B14 gate recovered **zero required positive source families**. It lost one original/gain positive condition, `b22-primary-272-mero-slow` at gain 0.5, against both B0 and the historical control: one lost condition, two reference rows. Natural events at gain 0.75 were **10 versus B0's 7**. Eight natural readers improved against B9, and no added ordinary-event or extra-positive-event reference rows were recorded; those results do not satisfy the remaining frozen gates.

The original PRIMARY artifacts are preserved unchanged:

- Result artifact `14939368185`, 5,089,742 bytes, ZIP SHA256 `b435e484fd8411477be8c3e537172661309a91602490d77d0cca1599c92a49a5`.
- Producer artifact `14940325352`, 33,094,710 bytes, ZIP SHA256 `47a17903a99958def67191c36f2dfa19b8b8038796d85a3932d21e9011ada226`.
- Consumed artifact `14938549644`, 384 bytes, ZIP SHA256 `90140702973c61e14dd5ca0dc077ad1136916584b680def79a95ac1ffc1a5536`.

An isolated fail-only readback authenticated all 12 original input archives, all 120 source originals and 600 observations, the complete original Git-bound producers, every consumption journal, and all **924 saved/raw arrays and 145,308 frames**. The PRIMARY producer contains 1,266 files and the source producer 1,262 files; no retrospective source supplement was needed. Two independent reads produced byte-identical reports and recomputed the original B14 verdict as **NO_PROMOTION**. The original PASS-only reader remains unchanged. The separate fail-only reader authenticates this failure and grants no replication, candidate engineering or promotion permission.

The original `RESULT.json` SHA256 is `5196b99be90855a5f8bf83c676df497d0d56d89d19885b7837ff0535452c3d95`. The exact saved quality-lock file SHA256 is `458a8af963dc959dca658b46800efdf1a23e1703cd397aa1b962340bdb6ed56a`; its sorted, compact JSON identity without a trailing newline is `e347323b5f10ed2b17501c511d53a3065adf45ee1446d006356d4097f11c8287`. The complete independent readback SHA256 is `dc94c1495eca5319aac20f90d233e62377d4625fdd9a02bcb7fa2c8aebb69746`. The full small artifact pins, quality lock, code hashes and exact aggregate decision are recorded in [V6_QUALITY_EVIDENCE.json](V6_QUALITY_EVIDENCE.json).

**B22 replication and candidate engineering remain closed; B0 remains the installed integration baseline.** No B22 model pack was created for this SDK or released. This closure uses saved evidence and adds zero training updates, model forwards, TTS or ASR calls. Physical SSC305 execution, field 3–5 m FRR and natural continuous FAR/h remain **NOT_RUN**, with no measured field values or product-promotion claim.

## Candidate quality closure — B23 (2026-11-25 Asia/Shanghai)

The [B23 TRAIN run 45435538577](https://github.com/jiying2007/kws-data/actions/runs/45435538577) completed one 192-update, 1,847-forward phase. Its sole scientific change was to use the worst known-silent prefix frame (`amax`) in the blank loss. The original 88 TRAIN sources, 122 retention conditions, 264 saved teacher arrays and all other B22 settings remained fixed. The historical B22 checkpoint was copied as an immutable control. B23 challenger PT SHA256 is `784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f`.

The [new independent source run 45438758355](https://github.com/jiying2007/kws-data/actions/runs/45438758355) preserved all 120 original audio sources and all 600 native observations. Its unchanged source gate admitted 78 originals: 13 synthetic positives, 47 synthetic ordinary speech and 18 natural speech originals from 11 people, including 9 paired source groups. All 42 unknown originals remain in the exposure and overlap denominator.

The [first B23 PRIMARY run 45443640293](https://github.com/jiying2007/kws-data/actions/runs/45443640293), head `992e4d80a0f784d7fa8eaaa9e2eaf49770b38ef2`, completed all **936 forwards**: 78 originals × three fixed gains × four frozen models. The workflow concluded `success`; the scientific verdict is **NO_PROMOTION**. Counts below use gains **1.0 / 0.75 / 0.5**; these are derived conditions of the same source originals.

| Frozen model | Positive originals with an event (of 13) | Natural ordinary events (18 originals) | Synthetic ordinary events (47 originals) |
|---|---:|---:|---:|
| B0 | 12 / 12 / 9 | 7 / 7 / 3 | 2 / 1 / 0 |
| B9 | 13 / 11 / 9 | 12 / 18 / 13 | 10 / 1 / 1 |
| Historical B22 control | 12 / 12 / 10 | 3 / 4 / 2 | 3 / 0 / 0 |
| B23 challenger | **0 / 0 / 0** | 0 / 0 / 0 | 0 / 0 / 0 |

The challenger lost 34 positive original/gain conditions across 12 originals relative to B0 and/or the historical control, represented by 67 reference-specific failure rows. It restored no required positive family. Zero ordinary-speech events accompany zero positive detections, so this checkpoint does not satisfy the promotion gate. These development counts provide no measured continuous FAR/h or field FRR.

An independent isolated fail-only reader reauthenticated all 12 original input ZIPs, all 120 source originals and 600 native observations, both complete Git-bound producer trees, every consumption journal, and all **936 saved/raw arrays and 153,216 frames**. The primary and source producer roots contain 1,331 and 1,327 files respectively; no retrospective supplement is used. The original B14 rule recomputes **NO_PROMOTION**. Readback SHA256 is `918b5ef21eae43a4fedcc04fa01fba1e46f7a9ebd4a6ad017a5c9fe042e08cac`. Exact artifact, reader and lock identities are recorded in `V6_QUALITY_EVIDENCE.json`.

**B23 replication and candidate engineering remain closed; B0 remains the installed integration baseline.** No B23 model pack was created for this SDK or released. This closure adds zero training, model, TTS or ASR calls. Physical SSC305 execution, real-human 3–5 m FRR and natural continuous FAR/h remain **NOT_RUN**.

## Historical candidate source closure — B24 (2026-11-25 Asia/Shanghai)

**B24's first SOURCE execution is permanently closed at `FAILED_EMPTY_NATIVE_TRANSCRIPT`. No B24 PRIMARY model scoring was performed.** The [original source run 45456888767](https://github.com/jiying2007/kws-data/actions/runs/45456888767), head `273e7bd0ae4b16972236f77aacae0eb5d1a12520`, concluded `failure`. The generated audio, every returned native payload and the original consumption records are retained unchanged.

The preceding [B24 TRAIN run 45449190562](https://github.com/jiying2007/kws-data/actions/runs/45449190562) completed one new **192-update, 1,847-forward phase** from the actual B23 checkpoint using fresh AdamW. It retained the original 88 TRAIN sources, 122 positive-retention conditions, 264 saved ordinary-speech teacher arrays, common draw, `amax` prefix loss, architecture and decoder. No B23 optimizer state was restored and no earlier B23 update was reexecuted. The immutable historical B23 checkpoint was copied as control. The B24 challenger PT SHA256 is `f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4`; its original training assessment remains `3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad`.

The preregistered TRAIN-only diagnostic read all **1,844 saved optimization arrays and 340,900 frames**. It found complete suffix events in 202 of 384 saved positive observations, zero prefix-shortcut event observations and zero base frame-0 HEY observations. These are pre-update TRAIN observations, not a final-checkpoint development assessment. The two frozen rejection conditions were not triggered; this did not establish quality or promotion. The TRAIN readback SHA256 is `ce63cbbd9a3d76e278d55dc57164b236714c49b39cecc1183a1452bf0e5fc301`.

The source job completed **96 synthetic and 24 whole natural originals**. Its five observers had a planned ceiling of 600 native calls. The original returned evidence establishes the following actual boundary:

| Observer | Native attempts | Returned raw payloads retained | Valid payloads and captures | Never entered |
|---|---:|---:|---:|---:|
| Qwen | 120 | 120 | 120 | 0 |
| Whisper | 120 | 120 | 120 | 0 |
| Parakeet | 49 | 49 | 48 | 71 |
| XLSR | 120 | 120 | 120 | 0 |
| LV60 | 120 | 120 | 120 | 0 |
| **Total** | **529** | **529** | **528** | **71** |

Parakeet's 49th original, `b24-primary-284-mero-fast`, returned exit code 0 without a timeout. Its preserved stdout is exactly one LF byte; the unchanged raw parser rejected the empty normalized transcript as `EMPTY_NATIVE_TRANSCRIPT`. The process, stdout, stderr and attempt/return journals were all saved. The observer stopped the entire panel immediately, leaving 71 inputs never entered. There are **zero entered calls with a missing native return or missing raw payload**. The empty response is retained as a failed response and is not converted into an ordinary-speech label or a valid abstention.

The [original readback job](https://github.com/jiying2007/kws-data/actions/runs/45456888767/job/138676788255) rejected `ALL_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES`. Its final upload found no output file. The final GitHub inventory therefore contains **14 actual artifacts and no SCREEN artifact**. No admission count, `SOURCE_GATE_PASS`, `INSUFFICIENT_SOURCE`, PRIMARY score or numerical `NO_PROMOTION` verdict is assigned to this failed source run. The raw readback log SHA256 is `e8c2553c277138c0ac2674433e00308b3f02b7550360ed1f3dc5f58b85bd7bb0`.

An independent saved-byte readback through [the isolated source-stop reader](https://github.com/jiying2007/kws-data/blob/e3117e897d478a76192f2dac1e0ac3149d12be6f/automation/english_v6_b24r1_b24_source_failure.py) verified all 14 original artifact digests and CRCs, all **1,399 original producer Git blobs**, every whole WAV/PCM and synthetic native chunk for all 120 originals, all five complete observer file inventories, and every provider's 120 journal slots. From the original captured code, it reparsed **528 valid native payloads, 18,258 complete phone frames and the same rejected empty return**, with no checkout fallback or scientific runtime. Its final canonical report is 11,055 bytes, SHA256 **`4063a57f1d20ffe9f534248067e1adcbe9c25ab65cb5bd3682b5db44d3ba39d8`**. The exact 14 artifact pins, original lock and reader identity are recorded in [V6_QUALITY_EVIDENCE.json](V6_QUALITY_EVIDENCE.json); raw API payloads and transcripts are not copied into this SDK handoff.

**B0 remains the installed integration baseline.** The old B24 SOURCE cannot retry or continue; its 120 PCM identities and all 72 original primary/reserve people remain excluded from later independent panels. The original 36-person sealed reserve stays unopened. B24 replication, candidate engineering and SDK model-pack release remain closed. A source failure does not satisfy B25's required complete PRIMARY `NO_PROMOTION` opening, so no B25 training follows from this record. This closure adds zero training updates, model, TTS or ASR calls. Physical SSC305 execution, real-human 3–5 m FRR and natural continuous FAR/h remain **NOT_RUN**.


## Latest completed fixed-weight assessment — B24Q1 (2026-11-25 Asia/Shanghai)

**The separately approved B24Q1 PRIMARY is complete and independently verified as NO_PROMOTION.** The [original run 45571993561](https://github.com/jiying2007/kws-data/actions/runs/45571993561), head d04924467cafc6ce4438ac18c91880975d4d4205, completed all **1,008 forwards** on 84 qualified originals: 25 synthetic positives, 47 synthetic ordinary originals and 12 natural originals from nine readers. All 120 source identities and 600 valid native observations remain accounted for; 36 five-valid originals were unqualified, with no quarantine or missing native return.

The table gives gain **1.0 / 0.75 / 0.5**. Positive counts are originals with events; ordinary counts are total events. Gains are derived conditions of the same originals.

| Cohort | B0 | B9 control role | Historical B23 control | B24 challenger |
| --- | --- | --- | --- | --- |
| Synthetic positive hits / 25 | 20 / 17 / 10 | 22 / 21 / 13 | 0 / 0 / 0 | 18 / 18 / 8 |
| Synthetic ordinary events / 47 originals | 2 / 1 / 0 | 4 / 2 / 2 | 0 / 0 / 0 | 1 / 1 / 0 |
| Natural ordinary events / 12 originals | 5 / 5 / 2 | 15 / 14 / 12 | 0 / 0 / 1 | 3 / 5 / 2 |

B24 recovered 16 B9-recoverable positive groups and improved six natural reader groups. It nevertheless lost six B0-positive original/gain conditions on three originals, and increased ordinary events in 11 conditions on eight originals (13 reference comparisons). No additional positive duplicate event or natural total above B0 occurred. The two failed per-original preservation conditions make the unchanged B14 result NO_PROMOTION.

The isolated fail-only reader reopened all 12 original input ZIPs and the three original output ZIPs, authenticated the complete source/native/TRAIN ancestry and both original Git-bound producer roots, and independently verified **all 1,008 saved/raw arrays and 128,340 frames**. It reproduced the original decision with zero new model, source or observer calls and no retrospective producer supplement. Canonical readback SHA-256 is **9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f**. [The immutable complete evidence record](https://github.com/jiying2007/kws-data/tree/7c93681e8c1adc8483175712a378fe5a43717e01/research/english_v6_b24q1_primary_no_promotion_20261125) includes the original artifact lock, complete readback, aggregate closure and exact file identities.

**B0 remains the installed integration baseline.** Q1 is consumed; its sealed replica and candidate engineering stay unopened, and no Q1 pack is selected or installed. All 72 Q1 primary/reserve people remain excluded from future independent panels. The original B24 and B24R1 SOURCE stops and old B25/B26 opening conditions remain unchanged. Physical SSC305, real-human 3–5 m FRR, natural continuous FAR/hour and product calibration remain unmeasured.
